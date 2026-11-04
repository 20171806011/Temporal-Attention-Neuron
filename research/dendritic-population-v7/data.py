import numpy as np

def make(n,seed):
    assert n%4==0
    rng=np.random.default_rng(seed); pair=np.repeat(np.arange(4),n//4);rng.shuffle(pair)
    a=pair//2;b=pair%2;y=a^b
    x=np.zeros((n,24,6));amp=rng.uniform(.8,1.2,(n,2));j=rng.integers(0,3,(n,2))
    for i in range(n):
        x[i,2+j[i,0]:6+j[i,0],a[i]]=amp[i,0]
        x[i,10+j[i,1]:14+j[i,1],2+b[i]]=amp[i,1]
    x[:,16:,4]=1
    x[:,7:10,5]=rng.uniform(0,.5,(n,3))
    x=np.maximum(x+rng.normal(0,.03,x.shape),0)
    return x,y,pair
