"""Fixtures only: no study seeds, trained weights, or scientific test results."""
from pathlib import Path
import json,time
import numpy as np
import core_model as model
import engine,data
from reference import native
R=Path(__file__).resolve().parent
def main():
    tic=time.perf_counter();checks=0;maximum=0.;fdmax=0.
    def close(a,b,tol=1e-10):
        nonlocal checks,maximum
        err=float(np.max(np.abs(a-b)));maximum=max(maximum,err);checks+=1;assert err<tol,(err,tol)
    b=data.make_base(32,7492000);x,q=data.assemble(b,24,'clean');maps=engine.donors(b['y'],7492001)
    for arm,ret,loss in [('DEND',[.95,.85],'RATE'),('DEND',[0,.85],'RATE'),('POINT',[.95,.85],'CE')]:
        w=model.initialize(7492002,24);c,k=model.forward(x,w,arm,24,ret=ret,query=q);state=engine.boundary(k,28)
        ref=native(x,w,arm,24,ret,q);close(c,ref['counts']);close(engine.trace(c,k)['states'],ref['states'][:8])
        for part in engine.PARTS:
            st=engine.mix(state,maps['opposite'],part);v=engine.resume(x,w,arm,ret,st);r=native(x,w,arm,24,ret,q,start=28,initial_state=st)
            close(v['counts'],r['counts']);close(v['states'],r['states'][:8,28:]);close(v['spikes'],r['spikes'][:8,28:])
            if part=='all':close(v['counts'],c[maps['opposite']])
            if ret[0]==0 and part=='d':close(v['counts'],c)
        close(engine.resume(x,w,arm,ret,state)['counts'],c)
        # Finite differences of the explicitly anchored surrogate, not the hard threshold.
        groups=[];grad=np.zeros_like(w)
        for delay in (0,24):
            xx,qq=data.assemble(b,delay,'clean');_,anchor=model.forward(xx,w,arm,24,ret=ret,query=qq)
            _,g,_=model.loss_grad(xx,b['y'],w,arm,loss,24,ret=ret,query=qq);grad+=g/2;groups.append((xx,qq,anchor))
        for idx in [0,100,287,288,900,1439,1440,1487,1488,1511,1512,1561]:
            delta=np.zeros_like(w);delta[idx]=1e-6
            objectives=[]
            for sign in (-1,1):objectives.append(sum(model.objective(xx,b['y'],w+sign*delta,arm,loss,24,anchor=a,ret=ret,query=qq)/2 for xx,qq,a in groups))
            fd=(objectives[1]-objectives[0])/2e-6;err=abs(fd-grad[idx]);fdmax=max(fdmax,err);checks+=1;assert err<2e-7,(arm,ret,idx,fd,grad[idx])
    rng=np.random.default_rng(7492003);labels=np.repeat([0,1],16);a=rng.normal(0,.1,(32,8));a[:,0]+=labels*3-1.5;b=a+.01*rng.normal(size=a.shape)
    checks+=1;assert np.array_equal(engine.ridge(a,labels,b)['prediction'],labels)
    result=dict(status='PASS',fixture_checks=checks,max_native_error=maximum,max_surrogate_fd_error=fdmax,seconds=time.perf_counter()-tic,scientific_data_generated=False)
    (R/'PRECHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
