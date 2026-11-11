"""Native state transplantation and fixed held-out diagnostic probes."""
import numpy as np
import core_model as model

H=24
PARTS={'d':(0,), 'vp':(1,3), 'dvp':(0,1,3), 'o':(2,), 'all':(0,1,2,3)}
COLS={'d':np.arange(48),'vp':np.r_[48:72,74:98],
      'dvp':np.r_[0:72,74:98],'o':np.arange(72,74),'all':np.arange(98)}

def predict(c):
    return np.where(c[:,0]==c[:,1],-1,c.argmax(1))

def basic(c,y):
    p=predict(c)
    return dict(balanced_accuracy=float(np.mean([np.mean(p[y==j]==j) for j in (0,1)])),
                accuracy=float(np.mean(p==y)),coverage=float(np.mean(p>=0)),
                silence=float(np.mean(c.sum(1)==0)),
                prediction_counts=[int(np.sum(p==j)) for j in (-1,0,1)],
                recalls=[float(np.mean(p[y==j]==j)) for j in (0,1)])

def donors(y,seed):
    r=np.random.default_rng(seed);opp=np.zeros(len(y),int);same=opp.copy()
    for j in (0,1):
        ids=np.flatnonzero(y==j);pool=r.permutation(np.flatnonzero(y!=j))
        opp[ids]=np.resize(pool,len(ids));order=r.permutation(ids);same[order]=np.roll(order,1)
    assert np.all(y[opp]!=y) and np.all(y[same]==y) and np.all(same!=np.arange(len(y)))
    return dict(opposite=opp,same=same)

def boundary(cache,t):
    a=cache[t-1];z=a['states']
    return [z[:,:48].copy(),z[:,48:72].copy(),z[:,72:74].copy(),a['hs'].copy()]

def features(cache,ids):
    return np.stack([np.concatenate(boundary(cache,t),1)[ids] for t in range(28,41)])

def mix(state,ids,part):
    return [a[ids].copy() if i in PARTS[part] else a.copy() for i,a in enumerate(state)]

def resume(x,w,arm,ret,state,start=28,q=40):
    """Identical hard equations, no labels; state supplied before update start."""
    wi,wr,db,vb,wo,ob=model.unpack(w,H);d,v,o,p=[a.copy() for a in state]
    n,T,_=x.shape;counts=np.zeros((n,2));states=[];spikes=[];rec=np.zeros(n);out=np.zeros(n)
    for t in range(start,T):
        dc=ret[0]*d+.25*(x[:,t]@wi+p@wr+db)
        if arm=='DEND':bs=np.zeros_like(dc);local=np.maximum(dc,0);d=dc
        else:bs=(dc>.5).astype(float);local=bs;d=dc*(1-bs)
        vc=ret[1]*v+.5*local.reshape(n,H,2).sum(2)+vb;hs=(vc>.5).astype(float);v=vc*(1-hs)
        oc=ret[1]*o+p@wo+ob;os=(oc>.5).astype(float);o=oc*(1-os)
        if t>=q:counts+=os
        rec+=p@(wr!=0).sum(1);out+=p@(wo!=0).sum(1)
        states.append(np.concatenate((d,v,o),1)[:8]);spikes.append(np.concatenate((bs,hs,os),1)[:8]);p=hs
    return dict(counts=counts,states=np.stack(states,1),spikes=np.stack(spikes,1),rec=rec,out=out)

def trace(c,cache):
    return dict(counts=c,states=np.stack([a['states'][:8] for a in cache],1),
                spikes=np.stack([np.concatenate((a['bs'],a['hs'],a['os']),1)[:8] for a in cache],1))

def cost(cache,w,arm,q):
    wi,wr,db,vb,wo,ob=model.unpack(w,H);T=len(cache)
    sums=lambda key,seq:float(np.mean(np.sum(np.stack([a[key] for a in seq]),axis=(0,2)))) if seq else 0.
    rec=sum(np.mean(a['prev']@(wr!=0).sum(1)) for a in cache)
    out=sum(np.mean(a['prev']@(wo!=0).sum(1)) for a in cache)
    branch=sums('bs',cache)
    return dict(branch_events=branch,hidden_events=sums('hs',cache),output_events=sums('os',cache),
       hold_hidden_events=sums('hs',cache[16:q]),hold_branch_events=sums('bs',cache[16:q]),
       recurrent_deliveries=float(rec),output_deliveries=float(out),branch_deliveries=branch,
       internal_deliveries=float(rec+out+branch),graded_local_transfers=48*T if arm=='DEND' else 0,
       state_updates=74*T,forward_macs=1488*T,parameters=1562,continuous_stored=74,
       persistent_continuous=74,
       previous_spike_registers=24,output_counters=2)

def transfer(op,same,base,y,mapping):
    po,ps,pb=map(predict,(op,same,base));don=mapping['opposite'];rows=[]
    for j in (0,1):
        ids=y==j;good=ids&(pb==y)&(pb[don]==y[don]);n=int(np.sum(good))
        rows.append(dict(direction=f'{j}->{1-j}',n=int(np.sum(ids)),
            opposite_target_fraction=float(np.mean(po[ids]==1-j)),
            same_target_fraction=float(np.mean(ps[ids]==1-j)),
            contrast=float(np.mean(po[ids]==1-j)-np.mean(ps[ids]==1-j)),
            donor_prediction_agreement=float(np.mean(po[ids]==pb[don][ids])),
            both_baseline_correct_n=n,conditional_flip=float(np.mean(po[good]==1-j)) if n else None))
    return dict(directions=rows,macro_contrast=float(np.mean([a['contrast'] for a in rows])))

def ridge(train,labels,test):
    mu=train.mean(0);sd=np.maximum(train.std(0),1e-8);a=(train-mu)/sd;b=(test-mu)/sd
    w=a.T@np.linalg.solve(a@a.T+np.eye(len(a)),2*labels-1)
    scores=b@w
    return dict(mean=mu,std=sd,weight=w,scores=scores,prediction=(scores>=0).astype(int))

def probes(fit,test,fit_pattern,test_pattern,seed):
    fy=(fit_pattern==12).astype(int);ty=(test_pattern==12).astype(int);rows=[];arrays={'fit_features':fit,'test_features':test,'fit_pattern':fit_pattern,'test_pattern':test_pattern}
    for ti,t in enumerate(range(28,41)):
        shuffled=np.random.default_rng(3111910000+100*seed+t).permutation(fy)
        arrays[f'shuffled_{t}']=shuffled
        raw=test[ti,:,:48];a=raw[ty==0];b=raw[ty==1];ca=a.mean(0);cb=b.mean(0);den=np.linalg.norm(ca)*np.linalg.norm(cb)
        geom=dict(d_centroid_cosine=float(ca@cb/den) if den else None,d_centroid_distance=float(np.linalg.norm(ca-cb)),
                  d_within_rms=float(np.sqrt(np.mean(np.r_[np.sum((a-ca)**2,1),np.sum((b-cb)**2,1)]))))
        for part,cols in COLS.items():
            row=dict(time=t,hold_steps=t-16,part=part,fit_n=len(fy),test_n=len(ty),**geom)
            for name,lab in [('real',fy),('shuffled',shuffled)]:
                result=ridge(fit[ti][:,cols],lab,test[ti][:,cols])
                for key,val in result.items():arrays[f'{t}_{part}_{name}_{key}']=val
                row[name+'_accuracy']=float(np.mean(result['prediction']==ty))
            rows.append(row)
    return rows,arrays
