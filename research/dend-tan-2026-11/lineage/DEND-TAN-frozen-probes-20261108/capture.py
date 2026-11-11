"""Tap the unchanged native forward; only six post-step snapshots are retained."""
import numpy as np
import model,population
TAPS=(9,10,14,15,19,23)
def capture(x,w,arm):
    wi,wr,db,vb,wo,ob=population.unpack(w,6);pieces=[]
    for start in range(0,len(x),32):
        xx=x[start:start+32];counts,cache=model.forward(xx,w,arm,16);a=[cache[t] for t in TAPS];states=np.stack([z['states'] for z in a],1)
        z=dict(d=states[:,:,:12],v=states[:,:,12:18],o=states[:,:,18:],p=np.stack([z['hs'] for z in a],1),counts=counts)
        cum=np.cumsum(np.stack([k['os'] if t>=16 else np.zeros_like(k['os']) for t,k in enumerate(cache)],1),axis=1);z['tap_counts']=cum[:,TAPS]
        if arm in ('FULL','NOATT'):z['H']=np.stack([k['attention']['H'] for k in a],1)
        resources=np.zeros((len(xx),4))
        for k in cache:
            resources[:,0]+=k['hs'].sum(1);resources[:,1]+=k['os'].sum(1);resources[:,2]+=k['prev']@(wr!=0).sum(1);resources[:,3]+=k['prev']@(wo!=0).sum(1)
        z['resources']=resources;pieces.append(z)
    return {key:np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
def features(z,group,t):
    if group=='counts':assert t==23;return z['counts'].astype(float)
    j=TAPS.index(t)
    if group in ('branch','soma','previous_spikes','output'):return z[dict(branch='d',soma='v',previous_spikes='p',output='o')[group]][:,j]
    base=np.concatenate((z['d'][:,j],z['v'][:,j],z['p'][:,j],z['o'][:,j]),1)
    if group=='joint':return base
    h=z['H'][:,j].reshape(len(base),60)
    if group=='history':return h
    if group=='augmented':return np.concatenate((base,h),1)
    raise ValueError(group)
def configurations(arm):
    groups=('branch','soma','previous_spikes','output','joint')+(('history','augmented') if arm in ('FULL','NOATT') else ())
    return [(g,t) for g in groups for t in TAPS]+[('counts',23)]
