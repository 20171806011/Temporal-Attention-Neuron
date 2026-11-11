import ast,time,numpy as np
import model,core,population,reference,baseline_native_tan
from design import *
tic=time.perf_counter();checks=0;maxerr=0.;fd=[]
def check(v,msg):
    global checks
    checks+=1
    assert bool(v),msg
def close(a,b,msg):
    global maxerr
    e=float(np.max(abs(np.asarray(a)-np.asarray(b))));maxerr=max(maxerr,e);check(e<1e-8,msg)
def main():
    old=ast.parse((R/'baseline_native_tan.py').read_text());new=ast.parse((R/'core.py').read_text())
    for name in ('cell','cell_backward','distribute'):
        check(ast.dump(next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name))==ast.dump(next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)),'original unit function unchanged '+name)
    rng=np.random.default_rng(79973001);x=rng.uniform(-.3,1.4,(4,24,6));w=population.initialize(79972001,6);y=np.array([0,1,1,0]);w[156:162]=.8;w[162:174]=rng.normal(1.1,.6,12);w[174:]=.4
    for arm,G,L in itertools.product(ARMS,GS,LS):
        z=model.evaluate(x,w,arm,G,L);ref=reference.replay(x,w,arm,G,L)
        for k in z:close(z[k],ref[k],k)
        scalar=reference.scalar_oracle(x[:2],w,arm,G,L)
        for k in ('counts','states','histories','spikes'):close(z[k][:2],scalar[k],'source scalar '+k)
        for reg in REGS:
            loss,g,c=model.loss_grad(x,y,w,arm,G,L,reg);_,anchor=model.forward(x,w,arm,G,L);close(loss,model.objective(x,y,w,arm,G,L,reg,anchor),'anchored center')
            check(np.linalg.norm(g-2*reg*w/176)>1e-6,'nontrivial gradient fixture')
            for j in range(6):
                direction=rng.normal(size=176);direction/=np.linalg.norm(direction);eps=1e-6
                numeric=(model.objective(x,y,w+eps*direction,arm,G,L,reg,anchor)-model.objective(x,y,w-eps*direction,arm,G,L,reg,anchor))/(2*eps);analytic=float(g@direction);error=abs(numeric-analytic);check(error<2e-5+2e-4*max(abs(numeric),abs(analytic)),'gain and coupling gradient');fd.append(error)
    for arm in ARMS:
        c,a=model.forward(x,w,arm,1.,.5);cc,aa=baseline_native_tan.forward(x,w,arm,6,16);close(c,cc,'baseline parameterization');close(np.stack([v['states'] for v in a]),np.stack([v['states'] for v in aa]),'baseline states')
    dump(R/'PRECHECK.json',dict(status='PASS',checks=checks,max_forward_error=maxerr,max_fd_error=max(fd),wall_seconds=time.perf_counter()-tic,scientific_seeds_used=False,scope='Unchanged source-cell AST; separate vector replay and actual scalar class; nontrivial anchored surrogate finite differences for all gains/couplings/regularizers.'));print(read(R/'PRECHECK.json'))
if __name__=='__main__':
    try:main()
    except Exception as e:dump(R/f'PRECHECK_FAILURE_{len(list(R.glob("PRECHECK_FAILURE_*.json")))+1}.json',dict(reason=str(e),checks=checks));raise
