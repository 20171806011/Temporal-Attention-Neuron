import time,numpy as np
import model,population,reference
from common import *
tic=time.perf_counter();checks=0;maxerr=0.;errors=[]
def check(v,msg):
    global checks
    checks+=1
    assert bool(v),msg
def close(a,b,msg):
    global maxerr
    e=float(np.max(abs(np.asarray(a).ravel()-np.asarray(b).ravel())));maxerr=max(maxerr,e);check(e<1e-8,msg)
def main():
    sequences=[np.zeros(24),np.ones(24),np.full(24,4.),np.r_[np.zeros(8),4.,np.zeros(15)],4*(np.arange(24)%2),np.random.default_rng(79983001).normal(0,3,24)]
    for full in (True,False):
        for seq in sequences:
            refs=reference.source_attention(seq,full);H=np.zeros((1,1,5))
            for t,I in enumerate(seq):
                A,H,a=model.attention(np.array([[I]]),H,full);ref=refs[t]
                for own,src in [('raw','S_raw'),('S','S_t'),('Q','Q'),('K','K'),('V','V'),('C','C_t'),('gate','gating'),('A','A_t')]:close(a[own],ref[src],'actual source attention '+own)
                if full:close(a['score'],ref['attention_scores'],'source scores');close(a['alpha'],ref['attention_weights'],'source epsilon normalization')
                close(H,np.r_[np.zeros(max(0,4-t)),seq[max(0,t-4):t+1]],'window includes current')
    rng=np.random.default_rng(79984001);x=rng.uniform(-.2,1.2,(4,24,6));w=population.initialize(79985001,6);y=np.array([0,1,1,0])
    for arm in ARMS:
        z=model.evaluate(x,w,arm,16);ref=reference.replay(x,w,arm,16)
        for name in z:close(z[name],ref[name],'independent '+name)
        scalar=reference.scalar_oracle(x[:2],w,arm,16)
        for name in ('counts','states','spikes'):close(z[name][:2],scalar[name],'scalar source '+name)
        loss,g,c=model.loss_grad(x,y,w,arm,16);_,anchor=model.forward(x,w,arm,16);close(loss,model.objective(x,y,w,arm,16,anchor),'anchored objective');check(np.linalg.norm(g-.0002*w/176)>1e-5,'nontrivial gradient');close(model.loss_grad(x,y,w,arm,16,True)[1],.0002*w/176,'zero spike surrogate')
        for j in range(12):
            d=rng.normal(size=176);d/=np.linalg.norm(d);eps=1e-6;numeric=(model.objective(x,y,w+eps*d,arm,16,anchor)-model.objective(x,y,w-eps*d,arm,16,anchor))/(2*eps);analytic=float(g@d);e=abs(numeric-analytic);errors.append(e);check(e<2e-5+2e-4*max(abs(numeric),abs(analytic)),'full history proxy gradient')
    for q in (16,40):
        xx=rng.uniform(0,1,(4,q+8,6));c,a=model.forward(xx,w,'DEND',q);cc,aa=population.forward(xx,w,'DEND',6,ret=(.95,.85),query=q);close(c,cc,'baseline counts');close(np.stack([v['states'] for v in a]),np.stack([v['states'] for v in aa]),'baseline allstates');l,g,_=model.loss_grad(xx,y,w,'DEND',q);ll,gg,_=population.loss_grad(xx,y,w,'DEND','CE',6,ret=(.95,.85),query=q);close(l,ll,'baseline loss');close(g,gg,'baseline gradient')
    dump(R/'PRECHECK.json',dict(status='PASS',checks=checks,max_forward_error=maxerr,max_fd_error=max(errors),wall_seconds=time.perf_counter()-tic,scientific_seeds_used=False,scope='Actual original scalar forward locals, independent hybrid replay, DEND identity forward/gradient, anchored surrogate finite differences; no scientific performance selection.'));print(read(R/'PRECHECK.json'))
if __name__=='__main__':
    try:main()
    except Exception as e:dump(R/f'PRECHECK_FAILURE_{len(list(R.glob("PRECHECK_FAILURE_*.json")))+1}.json',dict(reason=str(e),checks=checks));raise
