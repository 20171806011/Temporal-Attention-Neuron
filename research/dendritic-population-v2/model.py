"""Native-spiking population; shared parameterization, NumPy float64."""
import numpy as np
H=6
B=12
P=176
SHAPES=[('input',(6,12)),('recurrent',(6,12)),('branch_bias',(12,)),('soma_bias',(6,)),('output',(6,2)),('output_bias',(2,))]
def unpack(theta):
    out={}; i=0
    for k,s in SHAPES:
        n=int(np.prod(s)); out[k]=theta[i:i+n].reshape(s); i+=n
    assert i==P
    return out

def initialize(seed):
    rng=np.random.default_rng(seed)
    return np.concatenate([rng.normal(0,.45,72),rng.normal(0,.25,72),np.full(12,.15),np.full(6,.05),rng.normal(0,.5,12),np.full(2,.1)])

def forward(x,theta,arm,cut_cross=False,trace=False):
    """Labels never enter dynamics; binary native threshold/reset in both arms."""
    assert arm in ('DEND','POINT')
    w=unpack(theta); rec=w['recurrent'].copy()
    if cut_cross:
        rec*=np.arange(H)[:,None]==np.arange(B)[None,:]//2
    n,t,_=x.shape
    d=np.zeros((n,B)); v=np.zeros((n,H)); o=np.zeros((n,2)); prev=np.zeros((n,H))
    count=np.zeros((n,2)); events=np.zeros((n,3)); states=[]; spikes=[]; candidates=[]
    for k in range(t):
        dc=.85*d+.25*(x[:,k]@w['input']+prev@rec+w['branch_bias'])
        if arm=='DEND':
            local=np.maximum(dc,0); d=dc; bs=np.zeros_like(d)
        else:
            bs=(dc>.5).astype(float); local=bs; d=dc*(1-bs)
        vc=.65*v+.5*local.reshape(n,H,2).sum(2)+w['soma_bias']
        hs=(vc>.5).astype(float); v=vc*(1-hs)
        oc=.65*o+prev@w['output']+w['output_bias']
        os=(oc>.5).astype(float); o=oc*(1-os); prev=hs
        if k>=16:count+=os
        events+=np.stack((bs.sum(1),hs.sum(1),os.sum(1)),axis=1)
        if trace:
            states.append(np.concatenate((d,v,o),1)); spikes.append(np.concatenate((bs,hs,os),1)); candidates.append(np.concatenate((dc,vc,oc),1))
    pred=np.where(count[:,0]==count[:,1],-1,count.argmax(1))
    result={'counts':count,'prediction':pred,'events':events}
    if trace:result.update(states=np.stack(states,1),spikes=np.stack(spikes,1),candidates=np.stack(candidates,1))
    return result

def objective(x,y,theta,arm):
    c=forward(x,theta,arm)['counts']*.5
    c=c-c.max(1,keepdims=True)
    return float(np.mean(np.log(np.exp(c).sum(1))-c[np.arange(len(y)),y])+.0001*np.mean(theta*theta))
