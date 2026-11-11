"""Isolated fixtures across all widths; no scientific selection outcomes."""
import time
import numpy as np
import model,engine,base_data,independent
from reference import native
from common import *
def main():
    checks=0;maxerr=0.;fdmax=0.;tic=time.perf_counter()
    def close(a,b,tol=1e-9):
        nonlocal checks,maxerr
        e=float(np.max(np.abs(a-b)));maxerr=max(maxerr,e);checks+=1;assert e<tol,(e,tol)
    b=base_data.make_base(80,7993300);mapping=donors(b['y'],7993301)
    for h in SIZES:
        for arm in ARMS:
            w=model.initialize(7993400+h,h)
            for D in (24,96):
                x,q=assemble(b,D,'clean');tick=16+D//2;saved,resources,st=engine.evaluate(x,w,arm,h,q,tick);ref=independent.replay(x,w,arm,h,q,boundary_tick=tick)
                close(saved['counts'],ref['counts']);close(saved['states'],ref['states']);close(saved['spikes'],ref['spikes'])
                for j in range(4):close(st[j],ref['boundary'][j])
                small=native(x[:8],w,arm,h,RET,q);close(small['counts'],saved['counts'][:8]);close(small['states'],saved['states']);close(small['spikes'],saved['spikes'])
                for part in PARTS:
                    mixed=[a[mapping['opposite']].copy() if j in PARTS[part] else a.copy() for j,a in enumerate(st)];sv,co=engine.suffix(x,w,arm,h,q,tick,mixed);rf=independent.replay(x,w,arm,h,q,start=tick,initial=mixed)
                    close(sv['counts'],rf['counts']);close(sv['states'],rf['states']);close(sv['spikes'],rf['spikes']);close(sv['resources'],rf['resources'][:,:5])
                    if part=='all':close(sv['counts'],saved['counts'][mapping['opposite']])
                same,_=engine.suffix(x,w,arm,h,q,tick,st);close(same['counts'],saved['counts'])
            # Both loss candidates, two-group averaged anchored surrogate finite differences.
            tiny={k:v[:16] for k,v in b.items() if np.ndim(v)>0}
            for loss in LOSSES:
                g=np.zeros_like(w);groups=[]
                for D in (0,24):
                    x,q=assemble(tiny,D,'clean');_,anchor=model.forward(x,w,arm,h,ret=RET,query=q);_,gg,_=model.loss_grad(x,tiny['y'],w,arm,loss,h,ret=RET,query=q);g+=gg/2;groups.append((x,q,anchor))
                for idx in [0,12*h-1,12*h,12*h+2*h*h-1,len(w)-2,len(w)-1]:
                    d=np.zeros_like(w);d[idx]=1e-6;f=[]
                    for sign in (-1,1):f.append(sum(model.objective(x,tiny['y'],w+sign*d,arm,loss,h,anchor=a,ret=RET,query=q)/2 for x,q,a in groups))
                    err=abs((f[1]-f[0])/2e-6-g[idx]);checks+=1;fdmax=max(fdmax,err);assert err<2e-7,(h,arm,loss,idx,err)
            print('fixture',h,arm,'complete',flush=True)
    dump(R/'PRECHECK.json',dict(status='PASS',checks=checks,max_native_error=maxerr,max_surrogate_fd_error=fdmax,wall_seconds=time.perf_counter()-tic,scientific_data_generated=False))
if __name__=='__main__':main()
