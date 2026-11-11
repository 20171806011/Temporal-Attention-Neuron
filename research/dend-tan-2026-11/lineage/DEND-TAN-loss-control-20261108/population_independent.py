"""Independent equation transcription using einsum reductions, no experimental imports."""
import numpy as np
def replay(x,w,arm,h,q,start=0,initial=None,boundary_tick=None):
    shapes=[(6,2*h),(h,2*h),(2*h,),(h,),(h,2),(2,)];parts=[];cursor=0
    for shape in shapes:
        size=int(np.prod(shape));parts.append(np.reshape(w[cursor:cursor+size],shape));cursor+=size
    W,B,bd,bv,O,bo=parts;n,T,_=x.shape
    state=[np.zeros((n,k)) for k in (2*h,h,2,h)] if initial is None else [a.copy() for a in initial]
    d,v,o,p=state;c=np.zeros((n,2));events=[];traces=[];saved_boundary=None;r=np.zeros((n,8))
    for t in range(start,T):
        if t==boundary_tick:saved_boundary=[a.copy() for a in (d,v,o,p)]
        z=.95*d+.25*(np.einsum('ni,ij->nj',x[:,t],W,optimize=False)+np.einsum('ni,ij->nj',p,B,optimize=False)+bd)
        if arm=='DEND':local=np.where(z>0,z,0);d=z;branch=np.zeros_like(z)
        else:branch=np.greater(z,.5).astype(float);local=branch;d=z*np.logical_not(branch)
        z=.85*v+.5*(local[:,::2]+local[:,1::2])+bv;hidden=np.greater(z,.5).astype(float);v=z*np.logical_not(hidden)
        z=.85*o+np.einsum('ni,ij->nj',p,O,optimize=False)+bo;output=np.greater(z,.5).astype(float);o=z*np.logical_not(output)
        if t>=q:c+=output
        r[:,0]+=branch.sum(1);r[:,1]+=hidden.sum(1);r[:,2]+=output.sum(1)
        r[:,3]+=np.einsum('ni,i->n',p,np.count_nonzero(B,axis=1));r[:,4]+=np.einsum('ni,i->n',p,np.count_nonzero(O,axis=1))
        if 16<=t<q:r[:,5]+=branch.sum(1);r[:,6]+=hidden.sum(1);r[:,7]+=output.sum(1)
        traces.append(np.concatenate((d,v,o),axis=1)[:8]);events.append(np.concatenate((branch,hidden,output),axis=1)[:8]);p=hidden
    return dict(counts=c,states=np.stack(traces,1),spikes=np.stack(events,1),resources=r,boundary=saved_boundary)

def make_base(n,seed):
    r=np.random.default_rng(seed);p=np.repeat(np.arange(16),n//16);r.shuffle(p);bits=np.array([[(int(v)//(2**j))%2 for j in range(4)] for v in p]);y=np.array([(v[0] and v[1]) or (v[2] and v[3]) for v in bits],int)
    hi=r.uniform(.8,1.2,(n,4));lo=r.uniform(0,.1,(n,4));a=np.where(bits==1,hi,lo)
    return dict(pattern=p,y=y,write=np.clip(a[:,None,:]+r.normal(0,.03,(n,8,4)),0,None),distractor=r.uniform(0,.5,(n,4)),seed=np.array(seed))
def assemble(b,D,kind):
    q=16+D;x=np.zeros((len(b['y']),q+8,6))
    for t in range(8):x[:,8+t,:4]=b['write'][:,t]
    x[:,q:,4]=1
    if kind=='interference':
        for t in range(4):x[:,18+t,5]=b['distractor'][:,t]
    return x,q
def initial(seed,h):
    r=np.random.default_rng(seed);s=(6/h)**.5
    return np.r_[r.normal(0,.45,12*h),r.normal(0,.25,2*h*h)*s,np.repeat(.15,2*h),np.repeat(.05,h),r.normal(0,.5,2*h)*s,np.repeat(.1,2)]
