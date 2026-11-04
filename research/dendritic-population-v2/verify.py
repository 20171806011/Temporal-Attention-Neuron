"""Separate dynamics implementation; imports neither model, data, train nor evaluate."""
from pathlib import Path
import hashlib,json,time
import numpy as np
R=Path(__file__).resolve().parent
checks=0

def check(ok,why):
    global checks
    checks+=1
    if not ok:raise AssertionError(why)

def regenerate(n,seed):
    g=np.random.default_rng(seed);p=np.repeat(np.arange(4),n//4);g.shuffle(p)
    amps=g.uniform(.8,1.2,(n,2));shift=g.integers(0,3,(n,2));x=np.zeros((n,24,6))
    for i in range(n):
        for t in range(24):
            if 2+shift[i,0]<=t<6+shift[i,0]:x[i,t,p[i]//2]=amps[i,0]
            if 10+shift[i,1]<=t<14+shift[i,1]:x[i,t,2+p[i]%2]=amps[i,1]
    x[:,16:,4]=1
    x[:,7:10,5]=g.uniform(0,.5,(n,3));x=np.maximum(x+g.normal(0,.03,x.shape),0)
    return x,(p//2)^(p%2),p

def replay(x,t,arm,cut):
    n=len(x);wi=t[:72].reshape(6,12);wr=t[72:144].reshape(6,12).copy();db=t[144:156];vb=t[156:162];wo=t[162:174].reshape(6,2);ob=t[174:]
    if cut:
        for a in range(6):
            for b in range(12):
                if a!=b//2:wr[a,b]=0
    state=np.zeros((n,20));old=np.zeros((n,6));ss=[];cs=[];zs=[]
    for step in range(24):
        cand=np.zeros((n,20));spike=np.zeros((n,20));nextstate=np.zeros_like(state);loc=np.zeros((n,12))
        for b in range(12):
            drive=np.zeros(n);recur=np.zeros(n)
            for a in range(6):drive+=x[:,step,a]*wi[a,b];recur+=old[:,a]*wr[a,b]
            cand[:,b]=.85*state[:,b]+.25*(drive+recur+db[b])
            if arm=='POINT':
                spike[:,b]=cand[:,b]>.5;loc[:,b]=spike[:,b];nextstate[:,b]=cand[:,b]*(1-spike[:,b])
            else:loc[:,b]=np.maximum(cand[:,b],0);nextstate[:,b]=cand[:,b]
        for h in range(6):
            k=12+h;cand[:,k]=.65*state[:,k]+.5*(loc[:,2*h]+loc[:,2*h+1])+vb[h]
            spike[:,k]=cand[:,k]>.5;nextstate[:,k]=cand[:,k]*(1-spike[:,k])
        for o in range(2):
            drive=np.zeros(n)
            for h in range(6):drive+=old[:,h]*wo[h,o]
            k=18+o;cand[:,k]=.65*state[:,k]+drive+ob[o];spike[:,k]=cand[:,k]>.5;nextstate[:,k]=cand[:,k]*(1-spike[:,k])
        old=spike[:,12:18];state=nextstate;ss.append(state);cs.append(cand);zs.append(spike)
    spikes=np.stack(zs,1);counts=spikes[:,16:,18:].sum(1)
    pred=np.where(counts[:,0]==counts[:,1],-1,counts.argmax(1))
    events=np.stack([spikes[:,:,:12].sum((1,2)),spikes[:,:,12:18].sum((1,2)),spikes[:,:,18:].sum((1,2))],1)
    return dict(states=np.stack(ss,1),candidates=np.stack(cs,1),spikes=spikes,counts=counts,prediction=pred,events=events)

def ci(data):
    rng=np.random.default_rng(2026110499);boot=[]
    for _ in range(2000):
        means=[]
        for s in rng.integers(0,3,3):
            d,p=data[s];indices=[]
            for k in range(4):indices.extend(rng.choice(np.flatnonzero(p==k),128))
            means.append(d[np.array(indices)].mean())
        boot.append(np.mean(means))
    return float(np.mean([d.mean() for d,p in data])),np.quantile(boot,[.025,.975])

def main():
    begin=time.perf_counter();out=R/'results';digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    freeze=json.loads((R/'FREEZE.json').read_text())
    for f,h in freeze['sha256'].items():check(digest(R/f)==h,'frozen '+f)
    lock=json.loads((out/'SELECTION.json').read_text());start=json.loads((out/'TEST_STARTED.json').read_text());summary=json.loads((out/'summary.json').read_text())
    check(digest(out/'SELECTION.json')==start['selection_sha256']==summary['selection_sha256'],'selection lock')
    check(digest(R/'FREEZE.json')==lock['freeze_sha256'],'freeze lock')
    for f,h in summary['hashes'].items():check(digest(out/f)==h,'result hash '+f)
    logs=json.loads((out/'training_log.json').read_text());check(len(logs)==12,'full search budget');maxerr=0;correct={};pairs={};episodes=0
    for rec in logs:
        s=rec['seed'];arm=rec['arm'];lr=rec['lr'];check(digest(out/rec['file'])==rec['sha256'],'candidate hash')
        theta=np.load(out/f'initial_{s}.npy',allow_pickle=False).copy();rng=np.random.default_rng(2026110460+s)
        check(len(rec['history'])==100,'updates per candidate')
        for k,(plus,minus,norm) in enumerate(rec['history']):
            direction=rng.choice([-1.,1.],176);c=.15/(k+1)**.101;rate=lr/(k+1)**.602
            new=np.clip(theta-rate*np.clip((plus-minus)/(2*c)*direction,-5,5),-2,2)
            check(abs(np.linalg.norm(new-theta)-norm)<1e-12,'logged update norm');theta=new
        check(np.array_equal(theta,np.load(out/rec['file'],allow_pickle=False)),'SPSA parameter update replay')
        dev=np.load(out/f'development_{s}.npz',allow_pickle=False)
        for prefix,n,seed in [('',128,2026110400+10*s),('v',128,2026110401+10*s)]:
            xx,yy,pp=regenerate(n,seed)
            check(np.array_equal(xx,dev[prefix+'x']) and np.array_equal(yy,dev[prefix+'y']) and np.array_equal(pp,dev[prefix+'pair']),'development generation')
        val=replay(dev['vx'],theta,arm,False);acc=float(np.mean(val['prediction']==dev['vy']))
        check(acc==rec['validation_accuracy'],'candidate validation accuracy')
        z=val['counts']*.5;z-=z.max(1,keepdims=True)
        loss=float(np.mean(np.log(np.exp(z).sum(1))-z[np.arange(128),dev['vy']])+.0001*np.mean(theta*theta))
        check(abs(loss-rec['validation_objective'])<1e-12,'validation loss')
    for r in lock['selected']:
        candidates=[z for z in logs if (z['seed'],z['arm'])==(r['seed'],r['arm'])]
        chosen=min(candidates,key=lambda z:(-z['validation_accuracy'],z['validation_objective'],z['lr']))
        check(r['file']==chosen['file'],'validation-only selection')
    for s in range(3):
        data=np.load(out/f'test_{s}.npz',allow_pickle=False);x,y,p=regenerate(512,2026110402+10*s);pairs[s]=p
        check(np.array_equal(x,data['x']) and np.array_equal(y,data['y']) and np.array_equal(p,data['pair']),'fresh test generation')
        for arm in ('DEND','POINT'):
            choice=next(r for r in lock['selected'] if (r['seed'],r['arm'])==(s,arm));theta=np.load(out/choice['file'],allow_pickle=False)
            for mode in ('intact','cut_cross','initial'):
                t=np.load(out/f'initial_{s}.npy',allow_pickle=False) if mode=='initial' else theta
                v=replay(x,t,arm,mode=='cut_cross');saved=np.load(out/f'test_{s}_{arm}_{mode}.npz',allow_pickle=False);episodes+=len(x)
                for name in saved.files:
                    if name in ('states','candidates'):
                        residual=float(np.max(np.abs(v[name]-saved[name])));maxerr=max(maxerr,residual);check(residual<1e-10,'trajectory '+name)
                    else:check(np.array_equal(v[name],saved[name]),'exact '+name)
                row=next(r for r in summary['rows'] if (r['seed'],r['arm'],r['mode'])==(s,arm,mode));ok=v['prediction']==y;correct[s,arm,mode]=ok
                check(row['accuracy']==float(ok.mean()) and row['correct']==int(ok.sum()),'accuracy')
                check(row['coverage']==float(np.mean(v['prediction']>=0)) and row['abstentions']==int(np.sum(v['prediction']<0)),'coverage')
                check(np.array_equal(row['events_mean'],v['events'].mean(0)),'spike cost')
    groups=[(summary['primary'],[(correct[s,'DEND','intact'].astype(float)-correct[s,'POINT','intact'],pairs[s]) for s in range(3)])]
    for arm in ('DEND','POINT'):
        for name,mode in [('interaction','cut_cross'),('learning','initial')]:
            groups.append((summary[name][arm],[(correct[s,arm,'intact'].astype(float)-correct[s,arm,mode],pairs[s]) for s in range(3)]))
    for saved,ds in groups:
        avg,limits=ci(ds);check(avg==saved['difference'] and np.array_equal(limits,saved['percentile95']),'hierarchical paired CI')
    p=summary['primary'];check(summary['useful_advantage']==bool(p['difference']>.05 and p['percentile95'][0]>0),'primary criterion')
    result=dict(status='PASS',checks=checks,test_model_episodes=episodes,test_scalar_state_updates=episodes*24*20,max_state_candidate_residual=maxerr,seconds=time.perf_counter()-begin,limits='Separate equation implementation in same workflow; full dynamics replay, logged SPSA updates and validation selection checked; training objective values at every perturbation are not independently recomputed.')
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
