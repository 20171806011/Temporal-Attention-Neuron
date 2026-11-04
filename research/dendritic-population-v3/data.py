"""Easy sustained two-channel identity, NOT delayed XOR or memory."""
import numpy as np
def make(n,seed):
    r=np.random.default_rng(seed);y=np.repeat(np.arange(2),n//2);r.shuffle(y)
    x=np.zeros((n,24,6));amp=r.uniform(.8,1.2,n);on=r.integers(2,5,n)
    for i in range(n):x[i,on[i]:,y[i]]=amp[i]
    x[:,16:,4]=1;x[:,7:10,5]=r.uniform(0,.5,(n,3));x=np.maximum(x+r.normal(0,.03,x.shape),0)
    return x,y
