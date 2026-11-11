"""Shared fixed writing stimuli; exact-zero external hold and controlled distractor."""
import numpy as np
CONDITIONS=[(0,'clean'),(8,'clean'),(24,'clean'),(48,'clean'),(8,'interference'),(24,'interference'),(48,'interference')]
TRAIN_CONDITIONS=[(0,'clean'),(8,'clean'),(24,'clean'),(24,'interference')]
def make_base(n,seed):
    assert n%16==0
    r=np.random.default_rng(seed);pattern=np.repeat(np.arange(16),n//16);r.shuffle(pattern)
    bits=(pattern[:,None]>>np.arange(4))&1;y=((bits[:,0]&bits[:,1])|(bits[:,2]&bits[:,3])).astype(int)
    high=r.uniform(.8,1.2,(n,4));low=r.uniform(0,.1,(n,4));amp=np.where(bits,high,low)
    write=np.maximum(amp[:,None,:]+r.normal(0,.03,(n,8,4)),0)
    distractor=r.uniform(0,.5,(n,4))
    return dict(pattern=pattern,y=y,write=write,distractor=distractor,seed=np.array(seed))
def assemble(base,delay,condition):
    assert delay in (0,8,24,48) and condition in ('clean','interference')
    assert delay>=8 or condition=='clean'
    n=len(base['y']);q=16+delay;x=np.zeros((n,q+8,6))
    x[:,8:16,:4]=base['write'];x[:,q:,4]=1
    if condition=='interference':x[:,18:22,5]=base['distractor']
    return x,q
