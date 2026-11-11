"""Independent native tap capture; no experimental capture/model imports."""
import numpy as np
from reference import split,attention

TAPS=(9,10,14,15,19,23)
def capture(x,w,arm,taps=TAPS):
    assert x.shape[1:]==(24,6) and arm in ('DEND','FULL','NOATT','GAIN2')
    taps=tuple(taps);assert len(set(taps))==len(taps) and tuple(sorted(taps))==taps
    wi,wr,db,vb,wo,ob=split(w);n=len(x)
    d=np.zeros((n,12));v=np.zeros((n,6));o=np.zeros((n,2));previous=np.zeros((n,6));H=np.zeros((n,12,5))
    counts=np.zeros((n,2));resources=np.zeros((n,4))
    saved={key:[] for key in ('d','v','o','p','tap_counts')}
    if arm in ('FULL','NOATT'):saved['H']=[]
    for t in range(24):
        current=np.einsum('ni,ij->nj',x[:,t],wi,optimize=False)+np.einsum('ni,ij->nj',previous,wr,optimize=False)+db
        if arm in ('FULL','NOATT'):A,H,_=attention(current,H,arm=='FULL')
        else:A=current.copy() if arm=='GAIN2' else np.zeros_like(current)
        d=.95*d+.25*(current+A);local=np.maximum(d,0)
        candidate=.85*v+.5*(local[:,::2]+local[:,1::2])+vb
        hidden=(candidate>.5).astype(float);v=candidate*(1-hidden)
        candidate=.85*o+np.einsum('ni,ij->nj',previous,wo,optimize=False)+ob
        out=(candidate>.5).astype(float);o=candidate*(1-out)
        if t>=16:counts+=out
        resources[:,0]+=hidden.sum(1);resources[:,1]+=out.sum(1)
        resources[:,2]+=np.einsum('ni,i->n',previous,np.count_nonzero(wr,axis=1))
        resources[:,3]+=np.einsum('ni,i->n',previous,np.count_nonzero(wo,axis=1))
        previous=hidden
        if t in taps:
            for key,value in (('d',d),('v',v),('o',o),('p',previous),('tap_counts',counts)):saved[key].append(value.copy())
            if 'H' in saved:saved['H'].append(H.copy())
    return dict(counts=counts,resources=resources,**{key:np.stack(value,axis=1) for key,value in saved.items()})

def features(captured,group,tap=15,taps=TAPS):
    group={'branch':'d','soma':'v','previous_spikes':'p','output':'o','history':'H','augmented':'aug'}.get(group,group)
    if group=='counts':return captured['counts']
    index=tuple(taps).index(tap)
    if group in ('d','v','o','p'):return captured[group][:,index]
    if group=='H':return captured['H'][:,index].reshape(len(captured['counts']),60)
    joint=np.concatenate([captured[k][:,index] for k in ('d','v','p','o')],axis=1)
    if group=='joint':return joint
    if group=='aug':return np.concatenate((joint,captured['H'][:,index].reshape(len(joint),60)),axis=1)
    raise ValueError(group)
