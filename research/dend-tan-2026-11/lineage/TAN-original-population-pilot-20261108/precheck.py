import json,time
from pathlib import Path
import numpy as np
import original_tan,native_tan,population,oracle,base_data
R=Path(__file__).resolve().parent;START=time.perf_counter();CHECKS=0;MAXERR=0.;FD=[]
def check(v,msg):
    global CHECKS
    CHECKS+=1
    assert v,msg
    if time.perf_counter()-START>90:raise TimeoutError('90 second fixture cap')
def close(a,b,msg,atol=1e-9):
    global MAXERR
    err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))));MAXERR=max(MAXERR,err);check(err<atol,msg+f': {err}')
def assemble(b,D):
    q=16+D;x=np.zeros((len(b['y']),q+8,6));x[:,8:16,:4]=b['write'];x[:,q:,4]=1;return x,q
def main():
    sequences=[np.zeros(12),np.full(16,.7),np.array([0,0,4,4,4,4,4,4,0,0,0,3,0,-1,2,0.5]),np.array([0,-1000,1000,0,-999,999,0,0,0,0,0,0])]
    for att in (True,False):
        for sequence in sequences:
            ref=original_tan.make_variant('TAN-full' if att else 'TAN-no-attention');mem=np.zeros((1,1));H=np.zeros((1,1,5))
            for k,current in enumerate(sequence):
                zs=ref.forward(float(current));mem,H,z,a=native_tan.cell(np.full((1,1),current),mem,H,att)
                check(z.item()==zs,'original unit hard spike');close(mem.item(),ref.h,'original unit membrane');close(H[0,0],ref.history,'rolling window includes current')
                expected=np.r_[np.zeros(max(0,4-k)),sequence[max(0,k-4):k+1]];close(H[0,0],expected,'window padding and order')
                if att:
                    S=max(0,current-ref.history.mean());scores=2*S*ref.history;close(a['alpha'][0,0],ref.softmax(scores),'normalization and epsilon')
                check(np.isfinite(mem).all() and np.isfinite(H).all(),'finite boundary output')
        for initial,expected in ((1.,0),(np.nextafter(1.,2.),1)):
            ref=original_tan.make_variant('TAN-full' if att else 'TAN-no-attention');ref.h=initial;zs=ref.forward(0.)
            mem,H,z,a=native_tan.cell(np.zeros((1,1)),np.full((1,1),initial),np.zeros((1,1,5)),att);check(z.item()==expected==zs,'strict threshold equality and nextafter');close(mem.item(),ref.h,'hard reset')
    mem,H,z,a=native_tan.cell(np.zeros((1,1)),np.zeros((1,1)),np.zeros((1,1,5)),True)
    close(a['denom'],np.array([[[5+1e-9]]]),'zero-score denominator');check(a['surprise'].item()==0 and a['context'].item()==0 and z.item()==0,'zero surprise/context boundary')
    mem,H,z,a=native_tan.cell(np.array([[4.]]),np.zeros((1,1)),np.zeros((1,1,5)),True);check(z.item()==1 and H[0,0,-1]==4 and mem.item()==0,'spike reset retains history')
    base=base_data.make_base(16,79961001)
    for h,n,D in ((2,4,24),(6,2,0)):
        w=population.initialize(79962001,h);x,q=assemble({k:v[:n] if np.ndim(v) else v for k,v in base.items()},D)
        for arm in ('TAN_FULL','TAN_NOATT'):
            counts,cache=native_tan.forward(x,w,arm,h,q);ref=oracle.replay(x,w,arm,h,q)
            check(np.array_equal(counts,ref['counts']),'population scalar-source oracle counts');close(np.stack([a['states'] for a in cache],1),ref['states'],'population scalar-source states')
            spikes=np.stack([np.concatenate((a['bs'],a['hs'],a['os']),1) for a in cache],1);check(np.array_equal(spikes,ref['spikes']),'population scalar-source hard spikes')
            histories=np.stack([np.concatenate([a[k]['history'] for k in ('b','h','o')],axis=1) for a in cache],1);close(histories,ref['histories'],'all per-unit histories')
    # An explicit anchored surrogate objective validates derivatives, not the hard-spike derivative.
    rng=np.random.default_rng(79963001);h=2;n=2;T=24;q=16;x=rng.normal(.4,.9,(n,T,6));y=np.array([0,1]);w=population.initialize(79964001,h);wi,wr,db,vb,wo,ob=population.unpack(w,h)
    # Fixture-only driven currents exercise both variants, including all layers.
    # Scientific initialization is unchanged and is not evaluated here.
    wi[:]=rng.normal(.8,.3,wi.shape);wr[:]=rng.normal(.1,.1,wr.shape);vb[:]=.8;wo[:]=[[2.,-1.],[1.2,1.5]];ob[:]=.4
    for arm in ('TAN_FULL','TAN_NOATT'):
        loss,g,c=native_tan.loss_grad(x,y,w,arm,h,q);_,anchor=native_tan.forward(x,w,arm,h,q);close(loss,native_tan.objective(x,y,w,arm,h,q,anchor),'anchored objective agrees at center')
        task=g-.0002*w/len(w);check(np.linalg.norm(task)>1e-5,'nontrivial task gradient fixture');_,gz,_=native_tan.loss_grad(x,y,w,arm,h,q,zero_surrogate=True);close(gz,.0002*w/len(w),'zero surrogate leaves only regularizer')
        directions=[rng.normal(size=len(w)) for _ in range(12)]+[np.eye(len(w))[j] for j in (0,12*h,12*h+2*h*h,len(w)-3,len(w)-1)]
        for vector in directions:
            vector/=np.linalg.norm(vector);eps=1e-6;numeric=(native_tan.objective(x,y,w+eps*vector,arm,h,q,anchor)-native_tan.objective(x,y,w-eps*vector,arm,h,q,anchor))/(2*eps);analytic=float(g@vector);err=abs(numeric-analytic);tol=2e-6+2e-5*max(abs(numeric),abs(analytic));check(err<tol,'anchored directional derivative '+arm);FD.append(dict(arm=arm,analytic=analytic,numeric=float(numeric),absolute_error=err,tolerance=tol))
    # The local current-history VJP has genuine past-slot contributions.
    mem,H,z,a=native_tan.cell(np.array([[2.]]),np.array([[.1]]),np.array([[[.2,.3,.7,.5,.9]]]),True)
    gm,dH=native_tan.cell_backward(a,np.zeros((1,1)),np.ones((1,1)),True);check(np.linalg.norm(dH[...,:4])>1e-8,'non-current history gradients retained')
    for arm in ('DEND','POINT'):
        l,g,c=native_tan.loss_grad(x,y,w,arm,h,q);l0,g0,c0=population.loss_grad(x,y,w,arm,'CE',h,ret=(.95,.85),query=q);check(l==l0 and np.array_equal(g,g0) and np.array_equal(c,c0),'unchanged reference wrapper')
    record=dict(status='PASS',checks=CHECKS,max_unit_numeric_error=MAXERR,max_fd_absolute_error=max(d['absolute_error'] for d in FD),wall_seconds=time.perf_counter()-START,scientific_seeds_used=False,fixture_seeds=[79961001,79962001,79963001,79964001],gradient_contract='Derivative of the explicitly anchored local surrogate, including window paths and epsilon/subtract-max term. No exact hard-spike derivative claimed.',fd=FD)
    (R/'PRECHECK.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='fd'},indent=2))
if __name__=='__main__':
    try:main()
    except Exception as e:
        path=R/f'PRECHECK_FAILURE_{len(list(R.glob("PRECHECK_FAILURE_*.json")))+1}.json';path.write_text(json.dumps(dict(exception=type(e).__name__,reason=str(e),checks=CHECKS,wall_seconds=time.perf_counter()-START),indent=2)+'\n');raise
