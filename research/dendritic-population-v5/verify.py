"""Independent V5 equations, data, optimizer, selection and seed-unit audit.

Imports no frozen scientific modules. Reconstructs saved-gradient Adam updates;
separate gradient audits validate the surrogate chain rule. This is internal
workflow verification, not external replication.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np
R=Path(__file__).resolve().parent
checks=0

def check(ok,why):
    global checks
    checks+=1
    if not ok:raise AssertionError(why)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_json(p):return json.loads(p.read_text())

def initialize(seed):
    rng=np.random.default_rng(seed)
    return np.concatenate([rng.normal(0,.45,72),rng.normal(0,.25,72),np.full(12,.15),np.full(6,.05),rng.normal(0,.5,12),np.full(2,.1)])

def regenerate_xor(n,seed):
    g=np.random.default_rng(seed);pair=np.repeat(np.arange(4),n//4);g.shuffle(pair)
    amps=g.uniform(.8,1.2,(n,2));shift=g.integers(0,3,(n,2));x=np.zeros((n,24,6))
    for i in range(n):
        for k in range(24):
            if 2+shift[i,0]<=k<6+shift[i,0]:x[i,k,pair[i]//2]=amps[i,0]
            if 10+shift[i,1]<=k<14+shift[i,1]:x[i,k,2+pair[i]%2]=amps[i,1]
    x[:,16:,4]=1;x[:,7:10,5]=g.uniform(0,.5,(n,3));x=np.maximum(x+g.normal(0,.03,x.shape),0)
    return x,(pair//2)^(pair%2),pair

def replay(x,t,arm,reset_at=None):
    n=len(x);wi=t[:72].reshape(6,12);wr=t[72:144].reshape(6,12).copy();db=t[144:156];vb=t[156:162];wo=t[162:174].reshape(6,2);ob=t[174:]
    state=np.zeros((n,20));old=np.zeros((n,6));ss=[];cs=[];zs=[]
    for step in range(24):
        if step==reset_at:state=np.zeros((n,20));old=np.zeros((n,6))
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


def adam(initial,gradients,lr):
    theta=initial.copy();m=np.zeros(176);v=np.zeros(176);norms=[]
    check(gradients.shape==(200,176),'200x176 stored clipped gradients')
    for k,g in enumerate(gradients):
        check(np.isfinite(g).all() and np.linalg.norm(g)<=1+1e-12,'finite unit-norm clipped gradient')
        m=.9*m+.1*g;v=.999*v+.001*g*g
        new=np.clip(theta-lr*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)
        norms.append(float(np.linalg.norm(new-theta)));theta=new
    return theta,np.array(norms)

def loss_from_counts(c,y,theta,kind):
    if kind=='RATE':
        target=np.full_like(c,.125);target[np.arange(len(y)),y]=.75
        value=np.mean((c/8-target)**2)
    else:
        logits=c*.5;logits-=logits.max(1,keepdims=True);prob=np.exp(logits);prob/=prob.sum(1,keepdims=True)
        value=-np.log(prob[np.arange(len(y)),y]).mean()
    return float(value+.0001*np.mean(theta*theta))

def independent_metrics(v,y):
    pred=v['prediction'];c=v['counts'];n=len(y);cm=np.zeros((2,3),dtype=int)
    for truth,p in zip(y,pred):cm[int(truth),int(p) if p>=0 else 2]+=1
    recalls=[float(cm[k,k]/np.sum(y==k)) for k in (0,1)]
    target=np.full_like(c,.125);target[np.arange(n),y]=.75
    return dict(accuracy=float(np.mean(pred==y)),coverage=float(np.mean(pred>=0)),abstentions=int(np.sum(pred<0)),abstention_rate=float(np.mean(pred<0)),silent_fraction=float(np.all(c==0,axis=1).mean()),both_saturated_fraction=float(np.all(c==8,axis=1).mean()),rate_error=float(np.mean((c/8-target)**2)),class_recall=recalls,confusion=cm.tolist(),events_mean=v['spikes'].sum(1).mean(0).tolist())

def seed_interval(delta,seed):
    delta=np.asarray(delta,dtype=float);check(delta.shape==(10,),'ten seed-pair principal units')
    rng=np.random.default_rng(seed);indices=rng.integers(0,10,size=(20000,10))
    means=delta[indices].mean(axis=1)
    return dict(difference=float(delta.mean()),percentile95=np.quantile(means,[.025,.975]).tolist())
def inspect_data(out,name):
    file=np.load(out/name,allow_pickle=False);d={k:file[k] for k in file.files}
    for prefix,key in [('', 'train_seed'),('v','val_seed')]:
        if key not in d:continue
        x,y,p=regenerate_xor(128,int(d[key]))
        check(np.array_equal(x,d[prefix+'x']) and np.array_equal(y,d[prefix+'y']) and np.array_equal(p,d['vpair' if prefix else 'pair']),'independent dataset '+name+' '+prefix)
    return d

def check_metric(saved,actual,why):
    for k,v in actual.items():
        if isinstance(v,list):check(np.array_equal(saved[k],v),why+' '+k)
        else:check(saved[k]==v,why+' '+k)
    check(saved['failed']==bool(actual['accuracy']<.9 or actual['coverage']<.9),why+' failed gate')

def audit_training(out,rows,datasets,selected=None):
    seen={};updates=0;objectives=0
    check(len({r['id'] for r in rows})==len(rows),'unique training row IDs')
    for z in rows:
        stage,s,arm=z['stage'],z['seed'],z['arm'];check(arm in ('DEND','POINT'),'known arm');check(z['steps']==200,'fixed optimizer steps')
        check(sha(out/z['file'])==z['sha256'],'checkpoint hash '+z['id'])
        theta=np.load(out/z['file'],allow_pickle=False);check(theta.shape==(176,) and np.isfinite(theta).all(),'finite 176 parameters')
        d=datasets[z['data_file']]
        if stage=='A':
            check(s in range(10) and (z['loss'],z['lr'],z['role'])==('RATE',.01,'fixed'),'A fixed configuration')
            expected_init=2026116200+s;expected_train=2026116000+10*s
            check('vx' not in d and z['validation'] is None,'A has no validation selection')
        elif stage=='search':
            check(s in range(5) and z['role']=='candidate' and z['loss'] in ('CE','RATE') and z['lr'] in (.003,.01,.03),'complete search space member')
            expected_init=2026117100+s;expected_train=2026117000+10*s
            check(int(d['val_seed'])==expected_train+1,'search validation namespace')
        else:
            check(stage=='B' and s in range(10) and z['role'] in ('fixed','selected'),'B final seed and role')
            expected_init=2026118200+s;expected_train=2026118000+10*s
            cfg={'loss':'RATE','lr':.01} if z['role']=='fixed' else selected[arm]
            check((z['loss'],z['lr'])==(cfg['loss'],cfg['lr']),'B locked role recipe')
            check('vx' not in d and z['validation'] is None,'B no final validation filtering')
        check(z['initial_seed']==expected_init and int(d['train_seed'])==expected_train,'prespecified seed namespaces')
        initial=initialize(expected_init);check(np.array_equal(initial,np.load(out/z['initial_file'],allow_pickle=False)),'exact initialization')
        if z['reused']:
            check(stage=='B' and z['role']=='selected' and (z['loss'],z['lr'])==('RATE',.01),'reuse only exact fixed recipe')
            parent=next(r for r in rows if r['id']==z['reused_from'])
            check(parent['role']=='fixed' and (parent['seed'],parent['arm'])==(s,arm),'reuse paired parent')
            for k in ('file','trace','initial_file','data_file','initial_seed','sha256'):check(z[k]==parent[k],'reuse exact '+k)
            check(z['seconds']==0,'no duplicate training time')
        elif stage=='B' and z['role']=='selected':check((z['loss'],z['lr'])!=('RATE',.01),'reuse required when identical recipe')
        if z['trace'] not in seen:
            tr=np.load(out/z['trace'],allow_pickle=False);g=tr['clipped_gradients'];hist=tr['history'];check(hist.shape==(200,3) and np.isfinite(hist).all(),'complete finite optimizer log')
            final,norms=adam(initial,g,z['lr']);check(np.allclose(norms,hist[:,2],rtol=0,atol=1e-12),'all update norms')
            check(np.allclose(np.linalg.norm(g,axis=1),np.minimum(hist[:,1],1),rtol=0,atol=1e-12),'raw norms versus clipped gradient norms')
            v=replay(d['x'],initial,arm);expected=loss_from_counts(v['counts'],d['y'],initial,z['loss']);check(abs(expected-hist[0,0])<1e-12,'independent first training objective')
            seen[z['trace']]=final;updates+=200;objectives+=1
        check(np.array_equal(theta,seen[z['trace']]),'complete Adam reconstruction '+z['id'])
        if stage=='search':
            v=replay(d['vx'],theta,arm);check_metric(z['validation'],independent_metrics(v,d['vy']),'validation '+z['id'])
    return {'trajectories':len(seen),'updates':updates,'independent_initial_objectives':objectives}

def selection(rows):
    candidates={};selected={}
    for arm in ('DEND','POINT'):
        configs=[]
        for loss in ('CE','RATE'):
            for lr in (.003,.01,.03):
                g=[r for r in rows if r['stage']=='search' and (r['arm'],r['loss'],r['lr'])==(arm,loss,lr)]
                check(len(g)==5 and {r['seed'] for r in g}==set(range(5)),'five development seeds per candidate')
                configs.append(dict(loss=loss,lr=lr,min_accuracy=min(r['validation']['accuracy'] for r in g),mean_accuracy=float(np.mean([r['validation']['accuracy'] for r in g])),mean_rate_error=float(np.mean([r['validation']['rate_error'] for r in g]))))
        candidates[arm]=configs;selected[arm]=min(configs,key=lambda z:(-z['min_accuracy'],-z['mean_accuracy'],z['mean_rate_error'],z['lr'],0 if z['loss']=='RATE' else 1))
    return {'selected':selected,'candidates':candidates}

def assert_disjoint(sets):
    owner={};n=0
    for name,x in sets.items():
        for i,episode in enumerate(x):
            digest=hashlib.sha256(np.ascontiguousarray(episode).tobytes()).hexdigest()
            check(digest not in owner,'no exact repeated episode across independent splits '+name)
            owner[digest]=(name,i);n+=1
    return n

def audit_tests(out,pre,final,all_records,datasets):
    marker=read_json(out/'TEST_STARTED.json');summary=read_json(out/'TEST_SUMMARY.json')
    check(marker['prepare_lock_sha256']==sha(out/'PREPARE_LOCK.json'),'test start preparation identity')
    check(marker['final_lock_sha256']==sha(out/'FINAL_LOCK.json')==summary['final_lock_sha256'],'final weights locked before tests')
    for f,h in summary['hashes'].items():check(sha(out/f)==h,'test file hash '+f)
    rows=summary['rows'];expected={(phase,s,arm,role,mode) for phase in ('A','B') for s in range(10) for arm in ('DEND','POINT') for role in (('fixed',) if phase=='A' else ('fixed','selected')) for mode in ('intact','no_history')}
    keys=[(r['phase'],r['seed'],r['arm'],r['role'],r['mode']) for r in rows]
    check(len(rows)==120 and len(set(keys))==120 and set(keys)==expected,'all planned test rows and conditions')
    scored={};metric_cache={};accuracy={};maxres=0;unique_episodes=0;prefix_equal=0;testsets={}
    for phase in ('A','B'):
        for s in range(10):
            expected_seed=(2026116100 if phase=='A' else 2026118100)+10*s;name=f'test_{phase}_s{s}.npz';d=np.load(out/name,allow_pickle=False)
            x,y,p=regenerate_xor(512,expected_seed)
            check(int(d['seed'])==expected_seed and np.array_equal(x,d['x']) and np.array_equal(y,d['y']) and np.array_equal(p,d['pair']),'independent untouched test generation')
            check(np.array_equal(np.bincount(p,minlength=4),np.full(4,128)),'balanced four-pair test set')
            testsets[name]=x
            for row in [r for r in rows if r['phase']==phase and r['seed']==s]:
                arm,role,mode=row['arm'],row['role'],row['mode']
                source=next(r for r in all_records if (r['stage'],r['seed'],r['arm'],r['role'])==(phase,s,arm,role))
                check(row['source_file']==source['file'],'test weight provenance')
                key=(phase,s,arm,source['file'],mode)
                if key in scored:
                    check(row['reused'] and row['file']==scored[key],'exact duplicate evaluation reuse')
                    check_metric(row,metric_cache[key],'alias metrics')
                else:
                    check(not row['reused'],'first evaluation not an alias')
                    xx=x.copy()
                    if mode=='no_history':xx[:,:8,:]=0
                    theta=np.load(out/source['file'],allow_pickle=False);v=replay(xx,theta,arm);saved=np.load(out/row['file'],allow_pickle=False)
                    check(set(saved.files)=={'counts','prediction','spikes','states','candidates'},'complete saved test trajectory')
                    for field in saved.files:
                        if field in ('states','candidates'):
                            residual=float(np.max(abs(v[field]-saved[field])));maxres=max(maxres,residual);check(residual<1e-10,'independent native trajectory roundoff')
                        else:check(np.array_equal(v[field],saved[field]),'exact native spikes/counts/predictions')
                    if mode=='no_history':
                        check(np.array_equal(v['states'][:,:8],np.repeat(v['states'][:1,:8],512,axis=0)),'prefix-erased identical state through pre-step8')
                        check(np.array_equal(v['spikes'][:,:8],np.repeat(v['spikes'][:1,:8],512,axis=0)),'prefix-erased identical spike history')
                        prefix_equal+=1
                    actual=independent_metrics(v,y);check_metric(row,actual,'test metrics')
                    scored[key]=row['file'];metric_cache[key]=actual;unique_episodes+=len(y)
                accuracy[(phase,s,arm,role,mode)]=row['accuracy']
    check(summary['unique_model_evaluations']==len(scored),'unique evaluation count')
    check(summary['unique_test_histories']==20*512,'unique underlying histories')
    check(len(set(summary['hashes']))==20+len(scored),'exact test data/trajectory hash inventory')
    def vals(phase,role,arm,mode='intact'):
        return np.array([accuracy[(phase,s,arm,role,mode)] for s in range(10)])
    intervals={}
    def record(delta,seed):
        base=seed_interval(delta,seed)
        return {'seed_pair_differences':np.asarray(delta).tolist(),'mean':base['difference'],'percentile95':base['percentile95'],'unit':'independently trained seed pair','n_seed_pairs':10}
    for phase,role,seed in [('A','fixed',2026119001),('B','selected',2026119002),('B','fixed',2026119003)]:
        intervals[f'{phase}_{role}_DEND_minus_POINT']=record(vals(phase,role,'DEND')-vals(phase,role,'POINT'),seed)
    for arm in ('DEND','POINT'):intervals[f'B_{arm}_selected_minus_fixed']=record(vals('B','selected',arm)-vals('B','fixed',arm),2026119003)
    attenuation=(vals('B','fixed','DEND')-vals('B','fixed','POINT'))-(vals('B','selected','DEND')-vals('B','selected','POINT'))
    intervals['B_paired_gap_attenuation']=record(attenuation,2026119003)
    for phase,role in [('A','fixed'),('B','fixed'),('B','selected')]:
        for arm in ('DEND','POINT'):intervals[f'{phase}_{role}_{arm}_intact_minus_no_history']=record(vals(phase,role,arm)-vals(phase,role,arm,'no_history'),2026119004)
    check(intervals==summary['intervals'],'all principal seed-unit paired intervals and paired configuration attenuation')
    ia,ib=intervals['A_fixed_DEND_minus_POINT'],intervals['B_selected_DEND_minus_POINT']
    check(summary['primary']=='B_selected_DEND_minus_POINT' and summary['practical_threshold']==.05,'preregistered primary and practical threshold')
    check(summary['stable_direction']==bool(ia['percentile95'][0]>0 and ib['percentile95'][0]>0),'stable-direction rule')
    check(summary['stable_practically_meaningful']==bool(ia['percentile95'][0]>.05 and ib['percentile95'][0]>.05),'practical stability rule')
    splitsets={}
    for name,d in datasets.items():
        splitsets[name+' training']=d['x']
        if 'vx' in d:splitsets[name+' validation']=d['vx']
    splitsets.update(testsets);distinct=assert_disjoint(splitsets)
    return dict(logical_test_rows=len(rows),unique_model_evaluations=len(scored),test_model_episodes=unique_episodes,logical_test_model_episodes=len(rows)*512,test_scalar_state_updates=unique_episodes*24*20,unique_test_histories=20*512,distinct_cross_split_histories=distinct,no_history_prefix_checks=prefix_equal,max_state_candidate_residual=maxres)

def main():
    started=time.perf_counter();out=R/'results';freeze=read_json(R/'FREEZE.json')
    for n,h in freeze['sha256'].items():check(sha(R/n)==h,'frozen '+n)
    for n,h in freeze.get('dependencies',{}).items():check(sha(R.parent/n)==h,'frozen dependency '+n)
    pre=read_json(out/'PREPARE_LOCK.json');fin=read_json(out/'FINAL_LOCK.json')
    check(pre['freeze_sha256']==sha(R/'FREEZE.json'),'preparation freeze identity')
    check(fin['prepare_lock_sha256']==sha(out/'PREPARE_LOCK.json'),'final selection lock identity')
    check(pre['test_generated'] is False and fin['test_generated'] is False,'both locks recorded before test')
    for lock in (pre,fin):
        for name,h in lock['files'].items():check(sha(out/name)==h,'locked file '+name)
    check(pre['rows']==read_json(out/'PREPARE_LOG.json'),'preparation log locked')
    check(fin['rows']==read_json(out/'FINAL_LOG.json'),'final log locked')
    rows=pre['rows']+fin['rows'];expected_a={(s,a) for s in range(10) for a in ('DEND','POINT')};a=[r for r in pre['rows'] if r['stage']=='A']
    check(len(a)==20 and {(r['seed'],r['arm']) for r in a}==expected_a,'ten paired fixed-recipe A training seeds')
    search=[r for r in pre['rows'] if r['stage']=='search'];expected_search={(s,a,l,lr) for s in range(5) for a in ('DEND','POINT') for l in ('CE','RATE') for lr in (.003,.01,.03)}
    check(len(pre['rows'])==80 and len(search)==60 and {(r['seed'],r['arm'],r['loss'],r['lr']) for r in search}==expected_search,'equal complete B development search budgets')
    expected_b={(s,a,role) for s in range(10) for a in ('DEND','POINT') for role in ('fixed','selected')}
    check(len(fin['rows'])==40 and {(r['seed'],r['arm'],r['role']) for r in fin['rows']}==expected_b,'ten paired B seeds and both configuration roles')
    names={r['data_file'] for r in rows};datasets={name:inspect_data(out,name) for name in sorted(names)}
    check(len(names)==25,'ten A plus five search plus ten B independent datasets')
    prep_audit=audit_training(out,pre['rows'],datasets)
    choices=selection(pre['rows']);check(choices==pre['selection']==read_json(out/'SELECTION.json'),'independent architecture-specific validation selections')
    final_audit=audit_training(out,fin['rows'],datasets,choices['selected'])
    tests=audit_tests(out,pre,fin,rows,datasets)
    result=dict(status='PASS',checks=checks,preparation=prep_audit,final_training=final_audit,**tests,seconds=time.perf_counter()-started,scope='Separate RNG, native equations, metric, validation selection and seed-unit bootstrap implementation; full stored-gradient Adam reconstruction and independent initial training objectives. Separate gradient audits are required; this is internal workflow verification, not external replication.')
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
