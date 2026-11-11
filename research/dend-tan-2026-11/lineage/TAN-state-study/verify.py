"""Independent native/data/Adam/statistical replay; no study/engine/model imports."""
from pathlib import Path
import json,time,hashlib,sys
import numpy as np
R=Path(__file__).resolve().parent
CHECKS=0;MAXERR=0.;START=time.perf_counter()
def check(ok,reason):
    global CHECKS
    CHECKS+=1
    if not bool(ok):raise AssertionError(reason)
    if time.perf_counter()-START>300:raise TimeoutError('300 second independent verification cap')
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b,name):
    global MAXERR
    e=float(np.max(np.abs(a-b))) if np.size(a) else 0.;MAXERR=max(MAXERR,e);check(e<1e-10,name)
def initialize(seed,h):
    r=np.random.default_rng(seed)
    return np.r_[r.normal(0,.45,12*h),r.normal(0,.25,2*h*h),np.repeat(.15,2*h),np.repeat(.05,h),r.normal(0,.5,2*h),np.repeat(.1,2)]
def native(x,w,arm,h=6,ret=(.85,.65),spec=None,donor=None):
    n,T,_=x.shape;z=0
    W=np.array(w[z:z+12*h]).reshape(6,2*h);z+=12*h
    B=np.array(w[z:z+2*h*h]).reshape(h,2*h);z+=2*h*h
    db=w[z:z+2*h];z+=2*h;vb=w[z:z+h];z+=h
    O=np.array(w[z:z+2*h]).reshape(h,2);z+=2*h;ob=w[z:z+2]
    mem=np.zeros((n,T+1,4*h+2));state=np.zeros((n,T,3*h+2));event=np.zeros_like(state);candidate=np.zeros_like(state)
    count=np.zeros((n,2));rec=np.zeros(n);out=np.zeros(n);delta=np.zeros(n)
    spec=spec or {'kind':'none'};d=np.zeros((n,2*h));v=np.zeros((n,h));o=np.zeros((n,2));p=np.zeros((n,h))
    parts={'d':range(0,2*h),'v':range(2*h,3*h),'o':range(3*h,3*h+2),'p':range(3*h+2,4*h+2),'dv':range(0,3*h),'all':range(0,4*h+2)}
    for t in range(T):
        a=np.concatenate((d,v,o,p),axis=1)
        if spec.get('tick')==t and spec['kind'] in ('zero','transplant'):
            ids=list(parts[spec['component']]);old=a[:,ids].copy()
            a[:,ids]=0 if spec['kind']=='zero' else donor[:,t,ids]
            delta+=np.abs(a[:,ids]-old).sum(axis=1)
        d=a[:,:2*h].copy();v=a[:,2*h:3*h].copy();o=a[:,3*h:3*h+2].copy();p=a[:,3*h+2:].copy()
        cutting=spec['kind']=='path' and t in spec['window']
        branch=np.zeros_like(d);hs=np.zeros_like(v);os=np.zeros_like(o)
        for j in range(2*h):
            rw=B[:,j].copy()
            if cutting and spec['path']=='self':rw[j//2]=0
            drive=np.sum(x[:,t,:]*W[:,j],axis=1)+np.sum(p*rw,axis=1)+db[j]
            cand=ret[0]*d[:,j]+.25*drive;candidate[:,t,j]=cand
            if arm=='DEND':branch[:,j]=np.maximum(cand,0);d[:,j]=cand
            else:
                spike=(cand>.5).astype(float);branch[:,j]=spike;event[:,t,j]=spike;d[:,j]=cand*(1-spike)
            rec+=np.sum(p*(rw!=0),axis=1)
        for j in range(h):
            incoming=branch[:,2*j]+branch[:,2*j+1]
            if cutting and spec['path']=='local':incoming=np.zeros(n)
            cand=ret[1]*v[:,j]+.5*incoming+vb[j];candidate[:,t,2*h+j]=cand
            hs[:,j]=(cand>.5).astype(float);v[:,j]=cand*(1-hs[:,j]);event[:,t,2*h+j]=hs[:,j]
        for j in range(2):
            ow=O[:,j].copy()
            if cutting and spec['path']=='output':ow[:]=0
            cand=ret[1]*o[:,j]+np.sum(p*ow,axis=1)+ob[j];candidate[:,t,3*h+j]=cand
            os[:,j]=(cand>.5).astype(float);o[:,j]=cand*(1-os[:,j]);event[:,t,3*h+j]=os[:,j]
            out+=np.sum(p*(ow!=0),axis=1)
        if t>=16:count+=os
        p=hs;state[:,t]=np.concatenate((d,v,o),axis=1);mem[:,t+1]=np.concatenate((d,v,o,p),axis=1)
    return dict(counts=count,states=state,spikes=event,candidates=candidate,memory=mem,recurrent=rec,outgoing=out,delta=delta)
def metric(c,y):
    pred=np.full(len(y),-1);pred[c[:,0]>c[:,1]]=0;pred[c[:,1]>c[:,0]]=1
    recalls=[float(np.sum((y==j)&(pred==j))/np.sum(y==j)) for j in range(2)]
    return dict(accuracy=float(np.sum(pred==y)/len(y)),balanced_accuracy=sum(recalls)/2,coverage=float(np.sum(pred!=-1)/len(y)),class_recall=recalls,
                flipped_label_fraction=float(np.mean(pred==1-y)),silent_fraction=float(np.mean(np.sum(c,axis=1)==0)),nonsilent_tie_fraction=float(np.mean((pred<0)&(np.sum(c,axis=1)>0))),
                confusion=[[int(np.sum((y==j)&(pred==k))) for k in (0,1,-1)] for j in (0,1)])
def compare_row(row,nv,y,pattern,h,arm):
    m=metric(nv['counts'],y)
    for name,val in m.items():close(np.asarray(val),np.asarray(row[name]),'metric '+name)
    check(row['passed']==(m['balanced_accuracy']>=.9 and m['coverage']>=.9),'pass gate')
    z=nv['spikes'];T=z.shape[1];events=[z[:,:,:2*h].sum((1,2)).mean(),z[:,:,2*h:3*h].sum((1,2)).mean(),z[:,:,3*h:].sum((1,2)).mean()]
    for name,val in zip(('branch_events','hidden_events','output_events'),events):close(np.array(val),np.array(row[name]),name)
    for name,val in [('total_events',sum(events)),('recurrent_event_deliveries',nv['recurrent'].mean()),('hidden_output_deliveries',nv['outgoing'].mean()),('intervention_l1_mean',nv['delta'].mean()),('consumed_internal_event_deliveries',nv['recurrent'].mean()+nv['outgoing'].mean()+events[0])]:close(np.array(val),np.array(row[name]),name)
    check(row['continuous_state_updates']==(3*h+2)*T,'state update accounting')
    check(row['dense_forward_macs']==(2*h*h+14*h)*T,'dense MAC accounting')
    check(row['parameters']==2*h*h+17*h+2,'parameter accounting')
    check(row['scheduled_graded_local_transfers']==(2*h*T if arm=='DEND' else 0),'graded transfers')
    p=np.where(nv['counts'][:,0]==nv['counts'][:,1],-1,nv['counts'].argmax(1))
    for k in np.unique(pattern):close(np.array(row['pattern_recalls'][str(k)]),np.array(np.mean(p[pattern==k]==y[pattern==k])),'pattern recall')
def verify_native(saved,v,subset=None):
    check(np.array_equal(saved['counts'],v['counts']),'exact native counts')
    if 'spikes' in saved:
        q=slice(None) if subset is None else slice(0,subset)
        check(np.array_equal(saved['spikes'],v['spikes'][q]),'exact native spikes')
        for key in ('states','candidates','memory'):
            if key in saved:close(saved[key],v[key][q],key+' trajectory')
def xor_data(n,seed):
    g=np.random.default_rng(seed);pat=np.repeat(np.arange(4),n//4);g.shuffle(pat);bit1=pat//2;bit2=pat%2
    amp=g.uniform(.8,1.2,(n,2));jitter=g.integers(0,3,(n,2));x=np.zeros((n,24,6));twin=np.zeros_like(x)
    for i in range(n):
        for k in range(24):
            if 2+jitter[i,0]<=k<6+jitter[i,0]:x[i,k,bit1[i]]=amp[i,0];twin[i,k,1-bit1[i]]=amp[i,0]
            if 10+jitter[i,1]<=k<14+jitter[i,1]:x[i,k,2+bit2[i]]=amp[i,1];twin[i,k,2+bit2[i]]=amp[i,1]
    x[:,16:,4]=1;twin[:,16:,4]=1;noise=g.uniform(0,.5,(n,3));x[:,7:10,5]=noise;twin[:,7:10,5]=noise
    eps=g.normal(0,.03,x.shape);x=np.clip(x+eps,0,None);twin=np.clip(twin+eps,0,None)
    same=np.arange(n)
    for k in range(4):
        ids=np.where(pat==k)[0]
        for j,i in enumerate(ids):same[i]=ids[j-1]
    return dict(x=x,twin=twin,y=bit1^bit2,pair=pat,same=same)
def binding(n,seed):
    rng=np.random.default_rng(seed);pat=np.repeat(np.arange(16),n//16);rng.shuffle(pat)
    b=np.array([[(int(p)//(2**k))%2 for k in range(4)] for p in pat]);y=np.array([(p[0] and p[1]) or (p[2] and p[3]) for p in b],int)
    high=rng.uniform(.8,1.2,(n,4));low=rng.uniform(0,.1,(n,4));amp=np.where(b==1,high,low)
    x=np.zeros((n,24,6))
    for k in range(8,16):x[:,k,:4]=amp
    x[:,16:,4]=1;x[:,7:10,5]=rng.uniform(0,.5,(n,3));x=np.clip(x+rng.normal(0,.03,x.shape),0,None)
    return dict(x=x,y=y,pattern=pat)
def ci(vals):
    a=np.asarray(vals);idx=np.random.default_rng(2611079001).integers(0,5,(20000,5));means=np.sum(a[idx],axis=1)/5
    return float(a.sum()/5),np.percentile(means,[2.5,97.5])
def audit_mechanism():
    out=R/'mechanism';rows=read(out/'ROWS.json');specs={r['name']:r for r in read(out/'SPECS.json')}
    check(len(rows)==5*3*(len(specs)+1),'complete mechanism grid')
    for s in range(5):
        saved=dict(np.load(out/f'data_s{s}.npz'));d=xor_data(512,2611070100+10*s)
        for key in d:check(np.array_equal(d[key],saved[key]),'XOR RNG '+key)
        check(np.array_equal(d['x'][:,8:],d['twin'][:,8:]),'identical post-write future')
        for arm,cohort in [('DEND','retrained'),('DEND','intact'),('POINT','retrained')]:
            w=np.load(R/'weights'/f's{s}_{arm}_{cohort}.npy');base=native(d['x'],w,arm);twin=native(d['twin'],w,arm)
            if cohort=='retrained':
                b=w[72:144].reshape(6,12);mask=np.arange(6)[:,None]!=np.arange(12)[None,:]//2;check(np.all(b[mask]==0),'existing no-cross weights retained')
            for row in [r for r in rows if r['seed']==s and r['arm']==arm and r['cohort']==cohort]:
                name=row['condition'];labels=d['y']
                if name=='twin':nv=twin;labels=1-labels
                elif name=='baseline':nv=base
                else:
                    spec=specs[name];donor=None
                    if spec['kind']=='transplant':donor=twin['memory'] if spec['donor']=='flip' else base['memory'][d['same']]
                    nv=native(d['x'],w,arm,spec=spec,donor=donor)
                    if 'sham' in name:check(np.array_equal(nv['counts'],base['counts']),'time-matched sham')
                    if name=='hold_flip_all':check(np.array_equal(nv['counts'],twin['counts']),'full donor recovery')
                verify_native(np.load(out/f'{row["id"]}.npz'),nv,32);compare_row(row,nv,labels,d['pair'],6,arm)
            print('VERIFY mechanism',s,arm,cohort,flush=True)
    a=[];bd=[];sv=[]
    for s in range(5):
        def q(name):return next(r['flipped_label_fraction'] for r in rows if r['seed']==s and r['arm']=='DEND' and r['cohort']=='retrained' and r['condition']==name)
        b=q('hold_flip_d')-q('hold_same_d');v=q('hold_flip_v')-q('hold_same_v');bd.append(b);sv.append(v);a.append(b-v)
    summary=read(out/'SUMMARY.json')
    for key,vals in [('primary',a),('branch_transfer',bd),('soma_transfer',sv)]:
        mean,limits=ci(vals);close(np.array(mean),np.array(summary[key]['mean']),key+' mean');close(limits,np.array(summary[key]['ci95']),key+' interval')
    check(summary['criterion_met']==(np.mean(a)>=.2 and ci(a)[1][0]>0),'frozen primary criterion')
    return len(rows)
def audit_training(stage):
    out=R/stage;rows=read(out/'ROWS.json');tuning=stage=='tuning';nseeds=2 if tuning else 5
    check(len(rows)==(32 if tuning else 20),'complete fit grid')
    updates=0
    for s in range(nseeds):
        data=dict(np.load(out/f'data_s{s}.npz'));regen=binding(128,(2611071000 if tuning else 2611074000)+10*s)
        for key in regen:check(np.array_equal(data[key],regen[key]),'training RNG '+key)
        if tuning:
            regenval=binding(256,2611072000+10*s)
            for key in regenval:check(np.array_equal(data['v'+key],regenval[key]),'validation RNG '+key)
        for row in [r for r in rows if r['seed']==s]:
            h=row['h'];arm=row['arm'];cfg=row['candidate'];tr=dict(np.load(out/f'{row["id"]}_train.npz'))
            seed=(2611073000 if tuning else 2611075000)+100*h+s;w=initialize(seed,h);check(np.array_equal(w,tr['initial']),'paired initialization')
            m=np.zeros_like(w);v=np.zeros_like(w);g=tr['gradients'];hist=tr['history'];cs=tr['counts']
            check(g.shape==(200,len(w)) and cs.shape==(200,128,2),'full update traces')
            for k in range(200):
                check(np.isfinite(g[k]).all(),'finite gradients');check(abs(np.linalg.norm(g[k])-min(1,hist[k,1]))<1e-10,'gradient clip')
                if cfg['loss']=='RATE':
                    target=np.full((128,2),.125);target[np.arange(128),data['y']]=.75;loss=np.mean((cs[k]/8-target)**2)
                else:
                    logits=cs[k]/2;logits-=logits.max(1,keepdims=True);prob=np.exp(logits);prob/=prob.sum(1,keepdims=True);loss=-np.log(prob[np.arange(128),data['y']]).mean()
                close(np.array(loss+.0001*np.mean(w*w)),np.array(hist[k,0]),'preupdate objective')
                if k in (0,99,199):
                    nv=native(data['x'],w,arm,h,cfg['ret']);check(np.array_equal(nv['counts'],cs[k]),'sampled native training trajectory')
                m=.9*m+.1*g[k];v=.999*v+.001*g[k]*g[k]
                w=np.clip(w-cfg['lr']*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2);updates+=1
            check(np.array_equal(w,tr['theta']),'all Adam updates exact')
            if tuning:
                nv=native(data['vx'],w,arm,h,cfg['ret']);verify_native(np.load(out/f'{row["id"]}_val.npz'),nv);compare_row(row,nv,data['vy'],data['vpattern'],h,arm)
            else:
                nv=native(data['x'],w,arm,h,cfg['ret']);compare_row(row,nv,data['y'],data['pattern'],h,arm)
        print('VERIFY',stage,s,flush=True)
    return updates
def audit_selection():
    rows=read(R/'tuning/ROWS.json');lock=read(R/'tuning/SELECTION_LOCK.json')
    for arm in ('DEND','POINT'):
        for h in (2,6):
            scores=[]
            for c in range(4):
                r=[x for x in rows if x['arm']==arm and x['h']==h and x['candidate']['id']==c]
                check(len(r)==2,'two tuning seeds per candidate');scores.append((sum(x['balanced_accuracy'] for x in r)/2,sum(x['coverage'] for x in r)/2,-c))
            winner=-max(scores)[2];check(lock['selected'][f'{arm}_h{h}']['id']==winner,'independent validation selection')
    final=read(R/'final_train/ROWS.json')
    for r in final:check(r['candidate']==lock['selected'][f'{r["arm"]}_h{r["h"]}'],'selected recipe used')
    for stage,filename in [('tuning','SELECTION_LOCK.json'),('final_train','TRAIN_LOCK.json')]:
        l=read(R/stage/filename)
        check(l['test_generated'] is False,'pre-test lock declaration')
        for name,d in l['files'].items():check(digest(R/stage/name)==d,'locked artifact '+name)
    check(read(R/'final_train/TRAIN_LOCK.json')['selection_sha256']==digest(R/'tuning/SELECTION_LOCK.json'),'selection lock provenance')
def audit_test():
    rows=read(R/'test/ROWS.json');check(len(rows)==20,'all final models tested');summary=read(R/'test/SUMMARY.json')
    check(summary['train_lock_sha256']==digest(R/'final_train/TRAIN_LOCK.json'),'test checkpoint provenance')
    for s in range(5):
        d=dict(np.load(R/'test'/f'data_s{s}.npz'));regen=binding(512,2611076000+10*s)
        for key in regen:check(np.array_equal(d[key],regen[key]),'held-out RNG '+key)
        for r in [r for r in rows if r['seed']==s]:
            w=np.load(R/'final_train'/f'{r["id"]}_train.npz')['theta'];v=native(d['x'],w,r['arm'],r['h'],r['candidate']['ret'])
            verify_native(np.load(R/'test'/f'{r["id"]}.npz'),v);compare_row(r,v,d['y'],d['pattern'],r['h'],r['arm'])
    for h in (2,6):
        dif=[]
        for s in range(5):
            def ba(a):return next(r['balanced_accuracy'] for r in rows if r['h']==h and r['seed']==s and r['arm']==a)
            dif.append(ba('DEND')-ba('POINT'))
        mean,limits=ci(dif);close(np.array(mean),np.array(summary['contrasts'][str(h)]['mean']),'architecture contrast');close(limits,np.array(summary['contrasts'][str(h)]['ci95']),'architecture interval')
    return len(rows)
def disjoint():
    seen=set();number=0
    for stage in ('mechanism','tuning','final_train','test'):
        for p in sorted((R/stage).glob('data_*.npz')):
            d=np.load(p);keys=['x','vx'] if stage=='tuning' else ['x']
            for key in keys:
                for x in d[key]:
                    h=hashlib.sha256(x.tobytes()).digest();check(h not in seen,'no exact episode duplicates across splits');seen.add(h);number+=1
    return number
def main():
    freeze=read(R/'FREEZE.json')
    for name,d in freeze['files'].items():check(digest(R/name)==d,'frozen source '+name)
    mechanisms=audit_mechanism();updates=audit_training('tuning')+audit_training('final_train');audit_selection();tests=audit_test();episodes=disjoint()
    result=dict(status='PASS',checks=CHECKS,max_continuous_residual=MAXERR,mechanism_evaluations=mechanisms,reconstructed_adam_updates=updates,final_test_models=tests,
                distinct_base_episodes=episodes,seconds=time.perf_counter()-START,scope='Independently coded internal replay; no scientific model/engine imports. Not external replication. Full mechanism/test count replay; first 32 retained mechanism state/spike trajectories; full held-out native trajectories; all Adam updates; sampled training forwards; independently regenerated data, selection, metrics, costs and intervals.')
    (R/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':
    try:main()
    except Exception as e:
        (R/'VERIFICATION_FAILURE.json').write_text(json.dumps(dict(type=type(e).__name__,reason=str(e),checks=CHECKS,seconds=time.perf_counter()-START),indent=2)+'\n');raise
