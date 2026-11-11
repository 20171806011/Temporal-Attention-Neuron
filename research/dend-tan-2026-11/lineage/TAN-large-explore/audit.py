"""Independent task/native/optimizer replay. Imports no experimental modules."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
R=Path(__file__).resolve().parent
CHECKS=0;MAXERR=0.;START=time.perf_counter();PRIOR=0.
TRAIN=[(0,'clean'),(8,'clean'),(24,'clean'),(24,'interference')]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(v,msg):
    global CHECKS
    CHECKS+=1
    if not bool(v):raise AssertionError(msg)
    if time.perf_counter()-START+PRIOR>180:raise TimeoutError('Cumulative 180-second independent audit cap')
def close(a,b,msg):
    global MAXERR
    error=float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) if np.size(a) else 0.;MAXERR=max(MAXERR,error);check(error<1e-9,msg)
def seed(stage,s,h=6):
    b=3000010000+{'explore_tune':0,'explore':100000,'confirm_tune':1000000,'confirm':1100000}[stage]
    return dict(train=[b+100*s+10*g for g in range(4)],evaluation=b+10000+100*s,initial=b+20000+1000*h+s)
def base(n,s):
    r=np.random.default_rng(s);p=np.repeat(np.arange(16),n//16);r.shuffle(p)
    b=np.array([[(int(v)//(2**j))%2 for j in range(4)] for v in p]);y=np.array([(v[0] and v[1]) or (v[2] and v[3]) for v in b],int)
    high=r.uniform(.8,1.2,(n,4));low=r.uniform(0,.1,(n,4));a=np.where(b==1,high,low)
    write=np.clip(a[:,None,:]+r.normal(0,.03,(n,8,4)),0,None);distractor=r.uniform(0,.5,(n,4))
    return dict(pattern=p,y=y,write=write,distractor=distractor,seed=np.array(s))
def assemble(b,delay,cond):
    q=16+delay;x=np.zeros((len(b['y']),q+8,6))
    for k in range(8):x[:,8+k,:4]=b['write'][:,k]
    for k in range(q,q+8):x[:,k,4]=1
    if cond=='interference':
        for k in range(4):x[:,18+k,5]=b['distractor'][:,k]
    return x,q
def initial(s,h):
    r=np.random.default_rng(s);factor=(6/h)**.5
    return np.r_[r.normal(0,.45,12*h),r.normal(0,.25,2*h*h)*factor,np.repeat(.15,2*h),np.repeat(.05,h),r.normal(0,.5,2*h)*factor,np.repeat(.1,2)]
def native(x,w,arm,h,ret,q,intervention=None):
    i=0;W=w[:12*h].reshape(6,2*h);i+=12*h;B=w[i:i+2*h*h].reshape(h,2*h);i+=2*h*h
    db=w[i:i+2*h];i+=2*h;vb=w[i:i+h];i+=h;O=w[i:i+2*h].reshape(h,2);i+=2*h;ob=w[i:i+2]
    n,T,_=x.shape;d=np.zeros((n,2*h));v=np.zeros((n,h));o=np.zeros((n,2));p=np.zeros((n,h))
    counts=np.zeros((n,2));states=np.zeros((n,T,3*h+2));events=np.zeros_like(states);rec=np.zeros(n);out=np.zeros(n)
    iv=intervention or {'kind':'none'}
    for t in range(T):
        if t==iv.get('tick') and iv['kind']=='zero':
            if iv['part'] in ('d','all'):d[:]=0
            if iv['part'] in ('v','all'):v[:]=0
            if iv['part']=='all':o[:]=0;p[:]=0
        cut=iv['kind']=='cut_recurrence' and 16<=t<q
        local=np.zeros_like(d);hs=np.zeros_like(v);os_=np.zeros_like(o)
        for j in range(2*h):
            incoming=np.sum(x[:,t]*W[:,j],axis=1);feedback=np.zeros(n) if cut else np.sum(p*B[:,j],axis=1)
            dc=ret[0]*d[:,j]+.25*(incoming+feedback+db[j])
            if arm=='DEND':d[:,j]=dc;local[:,j]=np.maximum(dc,0)
            else:
                firing=(dc>.5).astype(float);events[:,t,j]=firing;local[:,j]=firing;d[:,j]=dc*(1-firing)
            if not cut:rec+=np.sum(p*(B[:,j]!=0),axis=1)
        for j in range(h):
            vc=ret[1]*v[:,j]+.5*(local[:,2*j]+local[:,2*j+1])+vb[j]
            hs[:,j]=(vc>.5).astype(float);v[:,j]=vc*(1-hs[:,j]);events[:,t,2*h+j]=hs[:,j]
        for j in range(2):
            oc=ret[1]*o[:,j]+np.sum(p*O[:,j],axis=1)+ob[j]
            os_[:,j]=(oc>.5).astype(float);o[:,j]=oc*(1-os_[:,j]);events[:,t,3*h+j]=os_[:,j];out+=np.sum(p*(O[:,j]!=0),axis=1)
        if t>=q:counts+=os_
        states[:,t]=np.concatenate((d,v,o),axis=1);p=hs
    return dict(counts=counts,spikes=events,states=states,rec=rec,out=out)
def basic(c,y):
    pred=np.full(len(y),-1);pred[c[:,0]>c[:,1]]=0;pred[c[:,1]>c[:,0]]=1
    recalls=[float(np.mean(pred[y==j]==j)) for j in (0,1)]
    return dict(balanced_accuracy=sum(recalls)/2,coverage=float(np.mean(pred>=0)),accuracy=float(np.mean(pred==y))),pred
def describe(row,v,b,h,arm,q):
    m,p=basic(v['counts'],b['y']);z=v['spikes'];T=z.shape[1]
    for k,val in m.items():close(val,row[k],k)
    check(row['passed']==(m['balanced_accuracy']>=.9 and m['coverage']>=.9),'pass gate')
    values=dict(branch_events=z[:,:,:2*h].sum((1,2)).mean(),hidden_events=z[:,:,2*h:3*h].sum((1,2)).mean(),output_events=z[:,:,3*h:].sum((1,2)).mean(),hold_events=z[:,16:q].sum((1,2)).mean(),hold_hidden_events=z[:,16:q,2*h:3*h].sum((1,2)).mean(),recurrent_deliveries=v['rec'].mean(),output_deliveries=v['out'].mean(),branch_deliveries=z[:,:,:2*h].sum((1,2)).mean(),internal_deliveries=(v['rec']+v['out']+z[:,:,:2*h].sum((1,2))).mean())
    for k,val in values.items():close(val,row[k],k)
    for k,val in [('state_updates',(3*h+2)*T),('forward_macs',(2*h*h+14*h)*T),('graded_local_transfers',2*h*T if arm=='DEND' else 0)]:check(row[k]==val,k)
    close([float(np.mean(p[b['pattern']==j]==b['y'][b['pattern']==j])) for j in range(16)],row['pattern_recall'],'pattern recall')
def check_base(path,n,s):
    saved=dict(np.load(path));reg=base(n,s)
    for k in reg:check(np.array_equal(saved[k],reg[k]),'data regeneration '+k)
    return saved
def objective(c,y,kind,w):
    if kind=='RATE':
        target=np.full_like(c,.125);target[np.arange(len(y)),y]=.75;loss=np.mean((c/8-target)**2)
    else:
        z=c*.5;z-=z.max(1,keepdims=True);p=np.exp(z);p/=p.sum(1,keepdims=True);loss=-np.log(p[np.arange(len(y)),y]).mean()
    return loss+.0001*np.mean(w*w)
def training(stage):
    out=R/stage;tuning=stage.endswith('_tune');rows=read(out/('ROWS.json' if tuning else 'TRAIN_ROWS.json'));updates=0
    for s in sorted({r['seed'] for r in rows}):
        groups=[check_base(out/f'train_s{s}_g{g}.npz',32,seed(stage,s)['train'][g]) for g in range(4)]
        for row in [r for r in rows if r['seed']==s]:
            trace=dict(np.load(out/f'{row["id"]}_train.npz'));h=row['h'];cfg=row['config'];w=initial(seed(stage,s,h)['initial'],h)
            check(np.array_equal(w,trace['initial']),'initialization');grads=trace['gradients'];cs=trace['counts'];hist=trace['history'];check(grads.shape==(200,len(w)) and cs.shape==(200,128,2),'full fixed updates retained')
            m=np.zeros_like(w);v=np.zeros_like(w)
            for k in range(200):
                l=sum(objective(cs[k,g*32:(g+1)*32],groups[g]['y'],cfg['loss'],w)/4 for g in range(4));close(l,hist[k,0],'all stored count objectives')
                close(np.linalg.norm(grads[k]),min(1,hist[k,1]),'clipping norm')
                # Only fixed initial update of all fits and last update of final fits: targeted replay.
                if k==0 or (not tuning and k==199):
                    for g,(delay,condition) in enumerate(TRAIN):
                        x,q=assemble(groups[g],delay,condition);a=native(x,w,row['arm'],h,cfg['ret'],q)
                        check(np.array_equal(a['counts'],cs[k,g*32:(g+1)*32]),'sampled training native counts')
                m=.9*m+.1*grads[k];v=.999*v+.001*grads[k]*grads[k]
                w=np.clip(w-cfg['lr']*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2);updates+=1
            check(np.array_equal(w,trace['theta']),'all Adam updates exact')
        print('AUDIT',stage,'training',s,flush=True)
    return updates
def validate_tuning(stage):
    out=R/stage;rows=read(out/'ROWS.json');lock=read(out/'SELECTION_LOCK.json')
    for name,h in lock['files'].items():check(sha(out/name)==h,'selection artifact lock')
    for s in (0,1):
        b=check_base(out/f'eval_s{s}.npz',128,seed(stage,s)['evaluation'])
        for r in [r for r in rows if r['seed']==s]:
            w=np.load(out/f'{r["id"]}_train.npz')['theta'];saved=np.load(out/f'{r["id"]}_val.npz')['counts'];scores=[];cov=[]
            for g,(delay,condition) in enumerate(TRAIN):
                x,q=assemble(b,delay,condition);v=native(x,w,r['arm'],r['h'],r['config']['ret'],q);check(np.array_equal(v['counts'],saved[g]),'all tuning native counts');describe(r['condition_metrics'][g],v,b,r['h'],r['arm'],q)
                met,_=basic(v['counts'],b['y']);scores.append(met['balanced_accuracy']);cov.append(met['coverage'])
            close(np.mean(scores),r['score'],'candidate validation mean');close(np.mean(cov),r['coverage'],'candidate coverage mean')
    for key,cfg in lock['selected'].items():
        parts=key.split('_');arm=parts[0];h=int(parts[1][1:]);mode=parts[2];ranking=[]
        for c in range(4):
            rr=[r for r in rows if (r['arm'],r['h'],r['mode'],r['config']['id'])==(arm,h,mode,c)];check(len(rr)==2,'two tuning blocks')
            ranking.append((sum(a['score'] for a in rr)/2,sum(a['coverage'] for a in rr)/2,-c))
        check(-max(ranking)[2]==cfg['id'],'independent selection rule')
def profile(stage):
    out=R/stage;rows=read(out/'ROWS.json');tr=read(out/'TRAIN_ROWS.json');lock=read(out/'TRAIN_LOCK.json');tuning='explore_tune' if stage=='explore' else 'confirm_tune'
    selection=read(R/tuning/'SELECTION_LOCK.json');check(lock['selection_sha256']==sha(R/tuning/'SELECTION_LOCK.json'),'pre-test selection provenance');check(not lock['test_generated'],'pre-test lock declaration')
    for n,h in lock['files'].items():check(sha(out/n)==h,'pre-test weight artifact')
    for s in sorted({r['seed'] for r in rows}):
        b=check_base(out/f'eval_s{s}.npz',256,seed(stage,s)['evaluation'])
        for modelrow in [r for r in tr if r['seed']==s]:
            h=modelrow['h'];arm=modelrow['arm'];cfg=modelrow['config'];check(cfg==selection['selected'][f'{arm}_h{h}_{modelrow["mode"]}'],'selected config used')
            w=np.load(out/f'{modelrow["id"]}_train.npz')['theta'];rr=[r for r in rows if r['model_id']==modelrow['id']];check(len(rr)==12,'all seven conditions and five ablations retained')
            baseline_counts=None
            for r in rr:
                x,q=assemble(b,r['delay'],r['condition']);v=native(x,w,arm,h,cfg['ret'],q,r['intervention']);saved=np.load(out/f'{r["id"]}.npz')
                check(np.array_equal(v['counts'],saved['counts']),'all profile counts');check(np.array_equal(v['spikes'][:8],saved['spikes']),'saved native events');close(v['states'][:8],saved['states'],'saved states');describe(r,v,b,h,arm,q)
                if r['delay']==24 and r['condition']=='clean' and r['intervention']['kind']=='none':baseline_counts=v['counts'].copy()
                if r['intervention']['kind']=='sham':check(np.array_equal(baseline_counts,v['counts']),'time matched sham')
            print('AUDIT profile',stage,modelrow['id'],flush=True)
    return len(rows)
def main(mode):
    global PRIOR
    if mode=='confirm':PRIOR=read(R/'AUDIT_explore.json')['seconds']
    for n,h in read(R/'EXPLORE_FREEZE.json')['files'].items():check(sha(R/n)==h,'source freeze')
    stages=('explore_tune','explore') if mode=='explore' else ('confirm_tune','confirm')
    updates=sum(training(s) for s in stages);validate_tuning(stages[0]);n=profile(stages[1])
    result=dict(status='PASS',mode=mode,checks=CHECKS,optimizer_updates=updates,profile_rows=n,max_error=MAXERR,seconds=time.perf_counter()-START,prior_audit_seconds=PRIOR,
        scope='Independent task/native/accounting/Adam implementation. All candidate validation/profile counts, stored 8-episode traces, all optimizer updates, fixed initial training forwards and last final-fit forwards. Not all training gradients independently derived; no old study replay.')
    (R/f'AUDIT_{mode}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
    mode=sys.argv[1]
    try:main(mode)
    except Exception as e:
        (R/f'AUDIT_{mode}_FAILURE.json').write_text(json.dumps(dict(reason=str(e),checks=CHECKS,seconds=time.perf_counter()-START),indent=2)+'\n');raise
