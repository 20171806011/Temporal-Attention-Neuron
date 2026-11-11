"""Independent scalar-per-unit native replay adapted from the preserved prior auditor."""
import numpy as np
def native(x,w,arm,h,ret,q,intervention=None,start=0,initial_state=None):
    i=0;W=w[:12*h].reshape(6,2*h);i+=12*h;B=w[i:i+2*h*h].reshape(h,2*h);i+=2*h*h
    db=w[i:i+2*h];i+=2*h;vb=w[i:i+h];i+=h;O=w[i:i+2*h].reshape(h,2);i+=2*h;ob=w[i:i+2]
    n,T,_=x.shape;d=np.zeros((n,2*h));v=np.zeros((n,h));o=np.zeros((n,2));p=np.zeros((n,h))
    counts=np.zeros((n,2));states=np.zeros((n,T,3*h+2));events=np.zeros_like(states);rec=np.zeros(n);out=np.zeros(n)
    iv=intervention or {'kind':'none'}
    if initial_state is not None:
        d,v,o,p=[a.copy() for a in initial_state]
    for t in range(start,T):
        if t==iv.get('tick') and iv['kind']=='zero':
            if iv['part'] in ('d','all'):d[:]=0
            if iv['part'] in ('v','all'):v[:]=0
            if iv['part']=='all':o[:]=0;p[:]=0
        cut=iv['kind']=='cut_recurrence' and 16<=t<q
        local=np.zeros_like(d);hs=np.zeros_like(v);os_=np.zeros_like(o)
        for j in range(2*h):
            incoming=np.sum(x[:,t]*W[:,j],axis=1);feedback=np.zeros(n) if cut else np.sum(p*B[:,j],axis=1)
            dc=ret[0]*d[:,j]+.25*(incoming+feedback+db[j])
            if arm=='DEND':d[:,j]=dc;local[:,j]=np.maximum(dc,0)
            else:
                firing=(dc>.5).astype(float);events[:,t,j]=firing;local[:,j]=firing;d[:,j]=dc*(1-firing)
            if not cut:rec+=np.sum(p*(B[:,j]!=0),axis=1)
        for j in range(h):
            vc=ret[1]*v[:,j]+.5*(local[:,2*j]+local[:,2*j+1])+vb[j]
            hs[:,j]=(vc>.5).astype(float);v[:,j]=vc*(1-hs[:,j]);events[:,t,2*h+j]=hs[:,j]
        for j in range(2):
            oc=ret[1]*o[:,j]+np.sum(p*O[:,j],axis=1)+ob[j]
            os_[:,j]=(oc>.5).astype(float);o[:,j]=oc*(1-os_[:,j]);events[:,t,3*h+j]=os_[:,j];out+=np.sum(p*(O[:,j]!=0),axis=1)
        if t>=q:counts+=os_
        states[:,t]=np.concatenate((d,v,o),axis=1);p=hs
    return dict(counts=counts,spikes=events,states=states,rec=rec,out=out)
