import numpy as np
from data_v2 import make as xor
from data_v3 import make as identity

def make(level,n,seed):
    if level==3:return xor(n,seed)
    if level==0:
        x,y=identity(n,seed);return x,y,y.copy()
    r=np.random.default_rng(seed);y=np.repeat(np.arange(2),n//2);r.shuffle(y)
    amp=r.uniform(.8,1.2,n);x=np.zeros((n,24,6));start=12 if level==1 else 8
    for i in range(n):x[i,start:start+4,y[i]]=amp[i]
    x[:,16:,4]=1;x[:,7:10,5]=r.uniform(0,.5,(n,3));x=np.maximum(x+r.normal(0,.03,x.shape),0)
    return x,y,y.copy()

def dev_seed(level,s):return 2026112000+100*(1 if level==2 else level)+10*s
