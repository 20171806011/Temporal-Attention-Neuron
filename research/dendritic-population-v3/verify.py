"""Independent native dynamics/data/selection/Adam audit; gradient check is separate."""
from pathlib import Path
import json,hashlib,time
import numpy as np
R=Path(__file__).resolve().parent
checks=0
def check(ok,why):
    global checks
    checks+=1
    if not ok:raise AssertionError(why)

def regenerate(n,seed):
    r=np.random.default_rng(seed);y=np.repeat(np.arange(2),n//2);r.shuffle(y)
    a=r.uniform(.8,1.2,n);on=r.integers(2,5,n);x=np.zeros((n,24,6))
    for i in range(n):
        for k in range(on[i],24):x[i,k,y[i]]=a[i]
    x[:,16:,4]=1;x[:,7:10,5]=r.uniform(0,.5,(n,3));x=np.maximum(x+r.normal(0,.03,x.shape),0)
    return x,y

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

def ci(ds):
    rng=np.random.default_rng(2026110599);b=[]
    for _ in range(2000):
        vals=[]
        for s in rng.integers(0,3,3):
            d,y=ds[s];ids=[]
            for k in (0,1):ids.extend(rng.choice(np.flatnonzero(y==k),256))
            vals.append(d[np.array(ids)].mean())
        b.append(np.mean(vals))
    return float(np.mean([d.mean() for d,y in ds])),np.quantile(b,[.025,.975])

def main():
    start=time.perf_counter();out=R/'results';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for f,h in json.loads((R/'FREEZE.json').read_text())['sha256'].items():check(sha(R/f)==h,'frozen '+f)
    log=json.loads((out/'training_log.json').read_text());lock=json.loads((out/'SELECTION.json').read_text());check(len(log)==24,'candidate budget')
    check(lock['freeze_sha256']==sha(R/'FREEZE.json'),'freeze lock');development={}
    for s in range(3):
        data=np.load(out/f'development_{s}.npz',allow_pickle=False)
        for prefix,seed in [('',2026110500+10*s),('v',2026110501+10*s)]:
            x,y=regenerate(128,seed);check(np.array_equal(x,data[prefix+'x']) and np.array_equal(y,data[prefix+'y']),'development data')
        development[s]={n:data[n] for n in data.files}
        r=np.random.default_rng(2026110550+s);theta=np.concatenate([r.normal(0,.45,72),r.normal(0,.25,72),np.full(12,.15),np.full(6,.05),r.normal(0,.5,12),np.full(2,.1)])
        check(np.array_equal(theta,np.load(out/f'initial_{s}.npy',allow_pickle=False)),'initialization')
    for z in log:
        check(sha(out/z['file'])==z['sha256'],'candidate hash');trace=np.load(out/z['trace'],allow_pickle=False);h=trace['history'];g=trace['clipped_gradients'];check(h.shape==(200,4) and g.shape==(200,176),'update budget')
        theta=np.load(out/f"initial_{z['seed']}.npy",allow_pickle=False).copy();m=np.zeros(176);v=np.zeros(176)
        for k in range(200):
            check(np.linalg.norm(g[k])<=1+1e-12,'gradient norm bound')
            m=.9*m+.1*g[k];v=.999*v+.001*g[k]*g[k]
            new=np.clip(theta-z['lr']*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)
            check(abs(np.linalg.norm(new-theta)-h[k,2])<1e-12,'update norm');theta=new
        check(np.array_equal(theta,np.load(out/z['file'],allow_pickle=False)),'complete Adam replay')
        d=development[z['seed']];val=replay(d['vx'],theta,z['arm'],False);y=d['vy']
        check(z['validation_accuracy']==float(np.mean(val['prediction']==y)),'validation accuracy')
        check(z['validation_coverage']==float(np.mean(val['prediction']>=0)) and z['validation_abstentions']==int(np.sum(val['prediction']<0)),'coverage')
        target=np.full((128,2),.125);target[np.arange(128),y]=.75
        check(z['validation_rate_error']==float(np.mean((val['counts']/8-target)**2)),'validation rate error')
        check(np.array_equal(z['validation_events_mean'],val['spikes'].sum(1).mean(0)),'validation event costs')
    configs=[]
    for kind in ('CE','RATE'):
        for lr in (.01,.03):
            rows=[z for z in log if z['loss']==kind and z['lr']==lr]
            configs.append(dict(loss=kind,lr=lr,min_accuracy=min(z['validation_accuracy'] for z in rows),mean_accuracy=float(np.mean([z['validation_accuracy'] for z in rows])),mean_rate_error=float(np.mean([z['validation_rate_error'] for z in rows]))))
    check(configs==lock['configurations'],'configuration summaries')
    best=min(configs,key=lambda z:(-z['min_accuracy'],-z['mean_accuracy'],z['mean_rate_error'],z['lr'],0 if z['loss']=='RATE' else 1))
    check(best==lock['shared_configuration'],'shared selection ordering')
    selected=[z for z in log if (z['loss'],z['lr'])==(best['loss'],best['lr'])];check(selected==lock['selected'],'selected candidates')
    gate=all(z['validation_accuracy']>=.9 and z['validation_coverage']>=.9 for z in selected);check(gate==lock['gate_pass'],'development gate')
    if not gate:
        check(not (out/'TEST_STARTED.json').exists(),'no test when gate failed');episodes=0;maxerr=0
    else:
        summary=json.loads((out/'summary.json').read_text());marker=json.loads((out/'TEST_STARTED.json').read_text())
        check(marker['selection_sha256']==sha(out/'SELECTION.json')==summary['selection_sha256'],'pretest selection lock')
        for f,h in summary['hashes'].items():check(sha(out/f)==h,'test hash '+f)
        correct={};labels={};episodes=0;maxerr=0
        for s in range(3):
            x,y=regenerate(512,2026110502+10*s);labels[s]=y;d=np.load(out/f'test_{s}.npz',allow_pickle=False)
            check(np.array_equal(x,d['x']) and np.array_equal(y,d['y']),'fresh test data')
            for a in ('DEND','POINT'):
                z=next(z for z in selected if (z['seed'],z['arm'])==(s,a))
                for mode in ('trained','initial'):
                    t=np.load(out/(z['file'] if mode=='trained' else f'initial_{s}.npy'),allow_pickle=False);v=replay(x,t,a,False);episodes+=len(x)
                    saved=np.load(out/f'test_{s}_{a}_{mode}.npz',allow_pickle=False)
                    for name in saved.files:
                        if name in ('states','candidates'):
                            err=float(np.max(abs(v[name]-saved[name])));maxerr=max(maxerr,err);check(err<1e-10,'trajectory residual')
                        else:check(np.array_equal(v[name],saved[name]),'exact spikes/counts/predictions')
                    row=next(q for q in summary['rows'] if (q['seed'],q['arm'],q['mode'])==(s,a,mode));ok=v['prediction']==y;correct[s,a,mode]=ok
                    check(row['accuracy']==float(ok.mean()) and row['coverage']==float(np.mean(v['prediction']>=0)),'test accuracy/coverage')
                    check(row['abstentions']==int(np.sum(v['prediction']<0)),'abstentions')
                    check(np.array_equal(row['events_mean'],v['spikes'].sum(1).mean(0)),'test costs')
                    check(row['silent_fraction']==float(np.mean(np.all(v['counts']==0,1))) and row['both_saturated_fraction']==float(np.mean(np.all(v['counts']==8,1))),'collapse rates')
        groups=[(summary['paired_difference'],[(correct[s,'DEND','trained'].astype(float)-correct[s,'POINT','trained'],labels[s]) for s in range(3)])]
        for a in ('DEND','POINT'):groups.append((summary['learning'][a],[(correct[s,a,'trained'].astype(float)-correct[s,a,'initial'],labels[s]) for s in range(3)]))
        for record,ds in groups:
            mean,lim=ci(ds);check(record['difference']==mean and np.array_equal(record['percentile95'],lim),'bootstrap')
        check(summary['sanity_pass']==all(z['accuracy']>=.9 and z['coverage']>=.9 for z in summary['rows'] if z['mode']=='trained'),'sanity criterion')
    result=dict(status='PASS',checks=checks,test_model_episodes=episodes,test_scalar_state_updates=episodes*24*20,max_state_candidate_residual=maxerr,seconds=time.perf_counter()-start,scope='Separate native/data/selection implementation; complete Adam replay from stored clipped gradients. Gradient chain rule verified separately; not independent human replication.')
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
