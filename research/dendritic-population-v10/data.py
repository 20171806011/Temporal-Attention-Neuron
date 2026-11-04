"""AND-of-pairs/OR binding: balanced Boolean patterns, continuous noisy currents."""
import numpy as np

def make(n,seed):
    assert n%16==0
    rng=np.random.default_rng(seed)
    pattern=np.repeat(np.arange(16),n//16);rng.shuffle(pattern)
    bits=(pattern[:,None]>>np.arange(4))&1
    y=((bits[:,0]&bits[:,1])|(bits[:,2]&bits[:,3])).astype(int)
    high=rng.uniform(.8,1.2,(n,4));low=rng.uniform(0,.1,(n,4))
    amp=np.where(bits,high,low);x=np.zeros((n,24,6))
    x[:,8:16,:4]=amp[:,None,:]
    x[:,16:,4]=1
    x[:,7:10,5]=rng.uniform(0,.5,(n,3))
    x=np.maximum(x+rng.normal(0,.03,x.shape),0)
    return x,y,pattern
