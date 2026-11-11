"""Only fixture seeds; targeted checks for changed tasks, clock and constraints."""
import importlib.util,time
import numpy as np
import model,data,audit
from common import R,dump,utc,describe,interventions
def main():
    start=time.perf_counter();checks=0;fdmax=0.;numfd=0
    b=data.make_base(32,749101);ind=audit.base(32,749101)
    for k in b:assert np.array_equal(b[k],ind[k]);checks+=1
    for delay,condition in data.CONDITIONS:
        x,q=data.assemble(b,delay,condition);ix,iq=audit.assemble(b,delay,condition);assert q==iq and np.array_equal(x,ix);checks+=1
        assert np.array_equal(x[:,8:16,:4],b['write']);assert np.all(x[:,:8]==0);assert np.all(x[:,q:,:4]==0);assert np.all(x[:,q:,4]==1)
        if condition=='clean':assert np.all(x[:,16:q]==0)
        else:assert np.array_equal(x[:,18:22,5],b['distractor']);assert np.all(x[:,16:q,:5]==0)
    sp=importlib.util.spec_from_file_location('previous',R/'vendor/previous_model.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
    for h in (6,24):
      w=model.initialize(749102,h)
      assert np.array_equal(w,audit.initial(749102,h))
      for arm in ('DEND','POINT'):
       for ret in ((.95,.85),(.99,.95),(0.,.95)):
        for delay in (0,24):
          x,q=data.assemble(b,delay,'clean');x=x[:2];y=b['y'][:2]
          c,cache=model.forward(x,w,arm,h,ret=ret,query=q)
          if h==6 and delay==0:
              oc,_=old.forward(x,w,arm,h,ret=ret);assert np.array_equal(c,oc)
          nv=audit.native(x,w,arm,h,ret,q);assert np.array_equal(c,nv['counts']);checks+=1
          assert np.all((nv['spikes']==0)|(nv['spikes']==1))
          for kind in ('RATE','CE'):
              _,g,_=model.loss_grad(x,y,w,arm,kind,h,ret=ret,query=q)
              coordinates=np.unique(np.linspace(0,len(w)-1,24,dtype=int))
              for k in coordinates:
                  p=w.copy();m=w.copy();p[k]+=1e-6;m[k]-=1e-6
                  fd=(model.objective(x,y,p,arm,kind,h,cache,ret,q)-model.objective(x,y,m,arm,kind,h,cache,ret,q))/2e-6
                  fdmax=max(fdmax,abs(fd-g[k]));numfd+=1
        x,q=data.assemble(b,24,'clean')
        for iv in interventions():
            c,cache=model.forward(x,w,arm,h,ret=ret,query=q,intervention=iv);v=audit.native(x,w,arm,h,ret,q,iv)
            assert np.array_equal(c,v['counts']);state=np.stack([a['states'] for a in cache],1);assert np.max(abs(state-v['states']))<1e-10
            desc=describe(c,cache,b['y'],b['pattern'],h,arm,q,iv,ret);audit.describe(desc,v,b,h,arm,q);checks+=1
            if iv['kind']=='sham':bc,_=model.forward(x,w,arm,h,ret=ret,query=q);assert np.array_equal(c,bc)
    assert fdmax<1e-7
    # Output biases demonstrate exactly eight counted native spikes at either query time.
    for delay in (0,48):
        w=np.zeros(model.parameter_count(6));model.unpack(w,6)[-1][:]=np.nextafter(.5,1.)
        x=np.zeros((2,24+delay,6));c,cache=model.forward(x,w,'POINT',6,ret=(0,0),query=16+delay);assert np.all(c==8)
        model.unpack(w,6)[-1][:]=.5;c,_=model.forward(x,w,'POINT',6,ret=(0,0),query=16+delay);assert np.all(c==0)
    dump(R/'PRECHECK.json',dict(status='PASS',utc=utc(),direct_checks=checks,independent_metric_checks=audit.CHECKS,finite_differences=numfd,max_gradient_error=fdmax,max_independent_error=audit.MAXERR,seconds=time.perf_counter()-start))
    print('PRECHECK PASS',checks,numfd,fdmax,flush=True)
if __name__=='__main__':main()
