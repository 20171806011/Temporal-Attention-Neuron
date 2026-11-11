"""Independent native evaluator; no experimental forward or training imports."""
import numpy as np
from population_independent import make_base,initial
def split(w,h):
    parts=[];pos=0
    for shape in ((6,2*h),(h,2*h),(2*h,),(h,),(h,2),(2,)):
        size=int(np.prod(shape));parts.append(w[pos:pos+size].reshape(shape));pos+=size
    return parts
def step(current,mem,hist,full):
    hist=np.concatenate((hist[:,:,1:],current[:,:,None]),axis=2)
    surprise=np.maximum(current-np.sum(hist,axis=2)/5,0);Q=surprise[:,:,None]*2.;K=hist*1.;V=hist*1.
    if full:
        scores=Q*K;scores=scores-np.max(scores,axis=2,keepdims=True);exponent=np.exp(scores*1.);alpha=exponent/(np.sum(exponent,axis=2,keepdims=True)+1e-9);context=np.sum(alpha*V,axis=2)
    else:context=np.sum(V,axis=2)/5
    candidate=.5*mem+np.tanh(surprise)*context;spike=np.greater(candidate,.5).astype(float);mem=candidate*np.logical_not(spike)
    return mem,hist,spike,candidate,surprise,context
def assemble(base,D):
    q=16+D;x=np.zeros((len(base['y']),q+8,6));x[:,8:16,:4]=base['write'];x[:,q:,4]=1;return x,q
def replay(x,w,arm,h,q):
    wi,wr,db,vb,wo,ob=split(w,h);n,T,_=x.shape;d=np.zeros((n,2*h));v=np.zeros((n,h));o=np.zeros((n,2));p=np.zeros((n,h));hist=[np.zeros((n,U,5)) for U in (2*h,h,2)];count=np.zeros((n,2));states=[];spikes=[];histories=[];candidates=[];surprises=[];contexts=[];r=np.zeros((n,8));tan=arm.startswith('TAN_')
    for t in range(T):
        I=np.einsum('ni,ij->nj',x[:,t],wi,optimize=False)+np.einsum('ni,ij->nj',p,wr,optimize=False)+db
        if tan:
            d,hist[0],b,dc,sb,cb=step(I,d,hist[0],arm=='TAN_FULL')
            current=.5*(b[:,::2]+b[:,1::2])+vb
            v,hist[1],hidden,vc,sv,cv=step(current,v,hist[1],arm=='TAN_FULL')
            current=np.einsum('ni,ij->nj',p,wo,optimize=False)+ob
            o,hist[2],out,oc,so,co=step(current,o,hist[2],arm=='TAN_FULL')
            histories.append(np.concatenate(hist,axis=1)[:8]);surprises.append(np.concatenate((sb,sv,so),axis=1)[:8]);contexts.append(np.concatenate((cb,cv,co),axis=1)[:8])
        else:
            dc=.95*d+.25*I
            if arm=='DEND':b=np.zeros_like(dc);local=np.maximum(dc,0);d=dc
            else:b=np.greater(dc,.5).astype(float);local=b;d=dc*np.logical_not(b)
            vc=.85*v+.5*(local[:,::2]+local[:,1::2])+vb;hidden=np.greater(vc,.5).astype(float);v=vc*np.logical_not(hidden)
            oc=.85*o+np.einsum('ni,ij->nj',p,wo,optimize=False)+ob;out=np.greater(oc,.5).astype(float);o=oc*np.logical_not(out)
        if t>=q:count+=out
        r[:,0]+=b.sum(1);r[:,1]+=hidden.sum(1);r[:,2]+=out.sum(1);r[:,3]+=np.einsum('ni,i->n',p,np.count_nonzero(wr,axis=1));r[:,4]+=np.einsum('ni,i->n',p,np.count_nonzero(wo,axis=1))
        if 16<=t<q:r[:,5]+=b.sum(1);r[:,6]+=hidden.sum(1);r[:,7]+=out.sum(1)
        states.append(np.concatenate((d,v,o),axis=1)[:8]);spikes.append(np.concatenate((b,hidden,out),axis=1)[:8]);candidates.append(np.concatenate((dc,vc,oc),axis=1)[:8]);p=hidden
    result=dict(counts=count,states=np.stack(states,1),spikes=np.stack(spikes,1),candidates=np.stack(candidates,1),resources=r)
    if tan:result.update(histories=np.stack(histories,1),surprises=np.stack(surprises,1),contexts=np.stack(contexts,1))
    return result
