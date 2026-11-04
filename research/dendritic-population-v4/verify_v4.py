"""Independent V4 native-equation, RNG, optimizer and selection audit.

No experiment code is imported. Discrete results must match exactly; float
trajectories tolerate only reduction-order roundoff. Saved optimizer gradients
are replayed rather than claimed to be independently differentiated here.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
R=Path(__file__).resolve().parent
checks=0

def check(ok,why):
    global checks
    checks+=1
    if not ok:raise AssertionError(why)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read_json(path):return json.loads(path.read_text())

def initialize(seed):
    rng=np.random.default_rng(seed)
    return np.concatenate([rng.normal(0,.45,72),rng.normal(0,.25,72),np.full(12,.15),np.full(6,.05),rng.normal(0,.5,12),np.full(2,.1)])

def cross_mask():
    mask=np.ones(176)
    for h in range(6):
        for b in range(12):
            if h!=b//2:mask[72+h*12+b]=0
    check(int(mask.sum())==116,'116 effective masked parameters')
    return mask

def regenerate_xor(n,seed):
    g=np.random.default_rng(seed);pair=np.repeat(np.arange(4),n//4);g.shuffle(pair)
    amps=g.uniform(.8,1.2,(n,2));shift=g.integers(0,3,(n,2));x=np.zeros((n,24,6))
    for i in range(n):
        for k in range(24):
            if 2+shift[i,0]<=k<6+shift[i,0]:x[i,k,pair[i]//2]=amps[i,0]
            if 10+shift[i,1]<=k<14+shift[i,1]:x[i,k,2+pair[i]%2]=amps[i,1]
    x[:,16:,4]=1;x[:,7:10,5]=g.uniform(0,.5,(n,3));x=np.maximum(x+g.normal(0,.03,x.shape),0)
    return x,(pair//2)^(pair%2),pair

def regenerate_identity(n,seed):
    g=np.random.default_rng(seed);y=np.repeat(np.arange(2),n//2);g.shuffle(y)
    amp=g.uniform(.8,1.2,n);onset=g.integers(2,5,n);x=np.zeros((n,24,6))
    for i in range(n):
        for k in range(onset[i],24):x[i,k,y[i]]=amp[i]
    x[:,16:,4]=1;x[:,7:10,5]=g.uniform(0,.5,(n,3));x=np.maximum(x+g.normal(0,.03,x.shape),0)
    return x,y,y.copy()

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


def scalar_metrics(v,y):
    counts=v['counts'];pred=v['prediction'];target=np.full_like(counts,.125);target[np.arange(len(y)),y]=.75
    return dict(accuracy=float(np.mean(pred==y)),coverage=float(np.mean(pred>=0)),abstentions=int(np.sum(pred<0)),abstention_rate=float(np.mean(pred<0)),silent_fraction=float(np.mean(np.all(counts==0,axis=1))),both_saturated_fraction=float(np.mean(np.all(counts==8,axis=1))),rate_error=float(np.mean((counts/8-target)**2)),events_mean=v['spikes'].sum(1).mean(0).tolist())

def paired_interval(data,seed):
    rng=np.random.default_rng(seed);draw=[]
    for _ in range(2000):
        means=[]
        for s in rng.integers(0,5,5):
            delta,strata=data[s];idx=[]
            for label in np.unique(strata):
                group=np.flatnonzero(strata==label);idx.extend(rng.choice(group,len(group)))
            means.append(delta[np.array(idx)].mean())
        draw.append(np.mean(means))
    return dict(difference=float(np.mean([d.mean() for d,p in data])),percentile95=np.quantile(draw,[.025,.975]).tolist())

def reconstruct_adam(initial,gradients,lr,mask=None):
    theta=initial.copy();mask=np.ones(176) if mask is None else mask
    theta*=mask;m=np.zeros(176);v=np.zeros(176);snapshots={0:theta.copy()};norms=[]
    check(gradients.ndim==2 and gradients.shape[1]==176,'Adam gradient shape')
    for k,g in enumerate(gradients):
        check(np.all(np.isfinite(g)) and np.linalg.norm(g)<=1+1e-12,'finite bounded Adam gradient')
        check(np.array_equal(g,g*mask),'masked gradients zero')
        m=(.9*m+.1*g)*mask;v=(.999*v+.001*g*g)*mask
        new=np.clip(theta-lr*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)*mask
        norms.append(float(np.linalg.norm(new-theta)));theta=new
        if k+1 in (100,200):snapshots[k+1]=theta.copy()
    snapshots[len(gradients)]=theta.copy()
    return snapshots,np.array(norms)

def reconstruct_spsa(initial,gradients,lr,directions,probes,direction_seed):
    theta=initial.copy();rng=np.random.default_rng(direction_seed);norms=[]
    check(gradients.shape==(100,176) and directions.shape==(100,176) and probes.shape==(100,2),'SPSA full budget and traces')
    for k in range(100):
        direction=rng.choice([-1.,1.],176);check(np.array_equal(direction,directions[k]),'SPSA paired perturbation identity')
        c=.15/(k+1)**.101;rate=lr/(k+1)**.602
        expected=np.clip((probes[k,0]-probes[k,1])/(2*c)*direction,-5,5)
        check(np.array_equal(expected,gradients[k]),'SPSA probe-derived clipped update')
        new=np.clip(theta-rate*gradients[k],-2,2);norms.append(float(np.linalg.norm(new-theta)));theta=new
    return theta,np.array(norms)

def selection_summary(rows):
    configurations=[]
    for loss in ('CE','RATE'):
        for lr in (.01,.03):
            r=[z for z in rows if z['loss']==loss and z['lr']==lr]
            check(len(r)==10,'ten arm-seed candidates per configuration')
            configurations.append(dict(loss=loss,lr=lr,min_accuracy=min(z['accuracy'] for z in r),mean_accuracy=float(np.mean([z['accuracy'] for z in r])),mean_rate_error=float(np.mean([z['rate_error'] for z in r]))))
    selected=min(configurations,key=lambda z:(-z['min_accuracy'],-z['mean_accuracy'],z['mean_rate_error'],z['lr'],0 if z['loss']=='RATE' else 1))
    return configurations,selected

def gate_for_rows(rows):
    gates={a:all(z['accuracy']>=.9 and z['coverage']>=.9 for z in rows if z['arm']==a) for a in ('DEND','POINT')}
    for arm in gates:check(sum(z['arm']==arm for z in rows)==5,'all five seeds in gate')
    ceiling=all(z['accuracy']>=.99 for z in rows)
    return gates,ceiling

def regenerate(level,n,seed):
    if level==3:return regenerate_xor(n,seed)
    if level==0:return regenerate_identity(n,seed)
    rng=np.random.default_rng(seed);y=np.repeat(np.arange(2),n//2);rng.shuffle(y)
    amp=rng.uniform(.8,1.2,n);x=np.zeros((n,24,6));start=12 if level==1 else 8
    for i in range(n):
        for k in range(start,start+4):x[i,k,y[i]]=amp[i]
    x[:,16:,4]=1;x[:,7:10,5]=rng.uniform(0,.5,(n,3));x=np.maximum(x+rng.normal(0,.03,x.shape),0)
    return x,y,y.copy()

def check_metric_record(saved,actual,why):
    for name,value in actual.items():
        if name=='events_mean':check(np.array_equal(saved[name],value),why+' events')
        else:check(saved[name]==value,why+' '+name)

def loss_from_counts(counts,y,theta,kind):
    if kind=='RATE':
        target=np.full_like(counts,.125);target[np.arange(len(y)),y]=.75
        value=np.mean((counts/8-target)**2)
    else:
        z=counts*.5;z-=z.max(1,keepdims=True);prob=np.exp(z);prob/=prob.sum(1,keepdims=True)
        value=-np.log(prob[np.arange(len(y)),y]).mean()
    return float(value+.0001*np.mean(theta*theta))

def inspect_data(out,name):
    f=np.load(out/name,allow_pickle=False);data={k:f[k] for k in f.files};level=int(data['level']);seed=int(data['train_seed']);n=len(data['y'])
    check(n==128,'training histories per seed')
    x,y,p=regenerate(level,n,seed)
    check(np.array_equal(x,data['x']) and np.array_equal(y,data['y']) and np.array_equal(p,data['pair']),'training data '+name)
    if 'vx' in data:
        x,y,p=regenerate(level,128,int(data['val_seed']))
        check(np.array_equal(x,data['vx']) and np.array_equal(y,data['vy']) and np.array_equal(p,data['vpair']),'validation data '+name)
    return data

def audit_log(out,rows,datasets):
    seen={};sampled=0;updates=0;mask=cross_mask()
    check(len({z['id'] for z in rows})==len(rows),'unique logged endpoint IDs')
    for z in rows:
        check(sha(out/z['file'])==z['sha256'],'checkpoint hash '+z['id'])
        theta=np.load(out/z['file'],allow_pickle=False);check(theta.shape==(176,) and np.isfinite(theta).all(),'finite weights')
        d=datasets[z['data_file']];check(int(d['level'])==z['level'],'task-level metadata')
        s=z['seed'];level=z['level'];stage=z['stage'];check(s in range(5) and z['arm'] in ('DEND','POINT'),'paired arm/seed')
        expected_init=(2026111100+s if stage=='A' else 2026113000+100*level+s if stage=='R' else 2026112500+s)
        check(z['initial_seed']==expected_init,'prespecified initialization seed')
        if stage=='A':check(int(d['train_seed'])==2026111000+10*s and int(d['val_seed'])==2026111001+10*s,'A dataset schedule')
        elif stage=='R':check(int(d['train_seed'])==2026114000+100*level+10*s and 'vx' not in d,'replication training-only dataset')
        else:
            base=2026112000+100*(1 if level==2 else level)+10*s
            check(int(d['train_seed'])==base and int(d['val_seed'])==base+1,'B/C dataset schedule')
        if z['method']=='CUT':
            check(z['condition']=='cut' and z['steps']==0 and z['trace'] is None,'inference-only cut')
            source=np.load(out/z['source_file'],allow_pickle=False);check(np.array_equal(theta,source*mask),'inference cut exact mask')
        else:
            initial=initialize(z['initial_seed']);active=mask if z['condition']=='retrained' else np.ones(176)
            check(np.array_equal(np.load(out/z['initial_file'],allow_pickle=False),initial*active),'initial weights and mask')
            if z['trace'] not in seen:
                tr=np.load(out/z['trace'],allow_pickle=False);history=tr['history'];g=tr['clipped_gradients'];steps=z['trajectory_steps'];check(history.shape==(steps,3),'optimizer history shape')
                check(steps==(100 if z['method']=='SPSA' else 200),'fixed trajectory budget')
                if z['method']=='ADAM':
                    points,norms=reconstruct_adam(initial,g,z['lr'],active)
                    check(tr['directions'].size==0,'Adam has no SPSA directions')
                else:
                    check(z['loss']=='CE' and stage=='A' and z['condition']=='intact','SPSA scope')
                    end,norms=reconstruct_spsa(initial,g,z['lr'],tr['directions'],history[:,:2],2026111200+s);points={100:end,0:initial}
                check(np.allclose(norms,history[:,2],rtol=0,atol=1e-12),'logged optimizer update norms')
                # Independent forward objective at the initial perturbation or initial Adam state.
                if z['method']=='ADAM':
                    v=replay(d['x'],initial*active,z['arm'],False);expected=loss_from_counts(v['counts'],d['y'],initial*active,z['loss'])
                    check(abs(expected-history[0,0])<1e-12,'initial logged Adam objective')
                    sampled+=1
                else:
                    for j,sign in enumerate((1,-1)):
                        perturbed=initial+sign*.15*tr['directions'][0];v=replay(d['x'],perturbed,z['arm'],False)
                        check(abs(loss_from_counts(v['counts'],d['y'],perturbed,'CE')-history[0,j])<1e-12,'initial SPSA probe objective')
                        sampled+=1
                updates+=steps;seen[z['trace']]=points
            check(np.array_equal(theta,seen[z['trace']][z['steps']]),'complete optimizer reconstruction '+z['id'])
            if z['condition']=='retrained':check(np.array_equal(theta,theta*mask),'masked final weights')
        if z['validation'] is not None:
            v=replay(d['vx'],theta,z['arm'],False);check_metric_record(z['validation'],scalar_metrics(v,d['vy']),'validation '+z['id'])
    return dict(trajectories=len(seen),updates=updates,sampled_objective_values=sampled)

def audit_development(out,log,lock):
    a=[z for z in log if z['stage']=='A'];check(len(a)==100,'A endpoint and SPSA record count')
    expected_a={(s,a,m,l,lr,b) for s in range(5) for a in ('DEND','POINT') for m in ('ADAM','SPSA') for l in (('CE','RATE') if m=='ADAM' else ('CE',)) for lr in (.01,.03) for b in ((100,200) if m=='ADAM' else (100,))}
    check({(z['seed'],z['arm'],z['method'],z['loss'],z['lr'],z['steps']) for z in a}==expected_a,'A complete paired factorial and fixed-rate SPSA controls')
    selections=[]
    for budget in (100,200):
        rows=[dict(z['validation'],loss=z['loss'],lr=z['lr'],arm=z['arm']) for z in a if z['method']=='ADAM' and z['steps']==budget]
        configs,selected=selection_summary(rows);selections.append(dict(budget=budget,candidates=configs,selected=selected))
    check(selections==lock['selections']==read_json(out/'A_SELECTION.json'),'A exact shared selections')
    cfg=selections[1]['selected'];check(cfg==lock['configuration'],'B/C uses A200 configuration')
    b=[z for z in log if z['stage']=='B'];levels=sorted({z['level'] for z in b});check(levels==list(range(len(levels))) and levels,'sequential ladder from L0')
    table=[]
    for level in levels:
        rows=[z for z in b if z['level']==level];check(len(rows)==10 and {(z['seed'],z['arm']) for z in rows}=={(s,a) for s in range(5) for a in ('DEND','POINT')},'B five pairs')
        for z in rows:check((z['loss'],z['lr'],z['steps'],z['method'],z['condition'])==(cfg['loss'],cfg['lr'],200,'ADAM','intact'),'fixed B configuration')
        gates,ceiling=gate_for_rows([dict(z['validation'],arm=z['arm']) for z in rows])
        table.append(dict(level=level,gates=gates,ceiling=ceiling,test_eligible=any(gates.values()) and not ceiling))
        if not any(gates.values()):check(level==levels[-1],'stop at first joint failure')
    if levels[-1]<3:check(not any(table[-1]['gates'].values()),'early ladder stop requires joint failure')
    check(table==lock['levels']==read_json(out/'B_LEVELS.json'),'B gates ceiling and eligibility')
    memory=[z['level'] for z in table if z['level'] in (1,2) and any(z['gates'].values())];clevel=max(memory) if memory else None
    check(clevel==lock['c_level'],'highest passed memory level for C')
    c=[z for z in log if z['stage']=='C'];check(len(c)==(20 if clevel is not None else 0),'C condition and paired controls')
    if clevel is not None:
        check({(z['seed'],z['arm'],z['condition'],z['steps'],z['method']) for z in c}=={(s,a,condition,steps,method) for s in range(5) for a in ('DEND','POINT') for condition,steps,method in [('retrained',200,'ADAM'),('cut',0,'CUT')]},'C complete matched controls')
    for z in c:check((z['level'],z['loss'],z['lr'])==(clevel,cfg['loss'],cfg['lr']),'fixed C task and configuration')
    check(lock['failed_comparator_exception'] is True,'approved failed-comparator exception')
    for stage in ('A','B')+ (('C',) if clevel is not None else ()):
        parity=read_json(out/f'parity_{stage}.json');check(parity['status']=='PASS' and parity['intact_and_cross_mask'],'stage native parity')
        check(parity['v2_sha256']==sha(R.parent/'dendritic-population-v2/model.py'),'preserved V2 native source')
    return table

def audit_replication(out,dev,rep,rows):
    eligible=[z['level'] for z in dev['levels'] if z['test_eligible']]
    check(eligible==rep['eligible_levels'],'replication only eligible non-ceiling levels')
    check(rep['development_lock_sha256']==sha(out/'DEVELOPMENT_LOCK.json'),'replication tied to development lock')
    check(read_json(out/'REPLICATION_STARTED.json')['development_lock_sha256']==sha(out/'DEVELOPMENT_LOCK.json'),'replication start lock')
    expected={(l,s,a,c) for l in eligible for s in range(5) for a in ('DEND','POINT') for c in (('intact','retrained') if l==dev['c_level'] else ('intact',))}
    observed={(z['level'],z['seed'],z['arm'],z['condition']) for z in rows}
    check(len(rows)==len(expected) and observed==expected,'all replication seeds and controls retained')
    cfg=dev['configuration']
    for z in rows:
        check((z['stage'],z['method'],z['loss'],z['lr'],z['steps'])==('R','ADAM',cfg['loss'],cfg['lr'],200),'frozen replication configuration')
        check(z['validation'] is None,'no validation reselection of replication seeds')
    if rows:check(rows==read_json(out/'replication_log.json'),'replication log and lock')
    else:check(not (out/'replication_log.json').exists(),'no phantom replication training')
    for name,h in rep['files'].items():check(sha(out/name)==h,'replication file hash '+name)
    parity=read_json(out/'parity_REPLICATION.json');check(parity['status']=='PASS' and parity['intact_and_cross_mask'],'replication native parity')
    return eligible

def audit_tests(out,dev,rep,logs):
    summary=read_json(out/'TEST_SUMMARY.json');marker=read_json(out/'TEST_STARTED.json');eligible=rep['eligible_levels']
    check(marker['development_lock_sha256']==sha(out/'DEVELOPMENT_LOCK.json'),'test uses locked development')
    check(marker['replication_lock_sha256']==sha(out/'REPLICATION_LOCK.json')==summary['replication_lock_sha256'],'test uses locked replication')
    check(marker['eligible_levels']==eligible==summary['test_levels'],'test eligibility lock')
    for name,h in summary['hashes'].items():check(sha(out/name)==h,'test bytes '+name)
    expected={(l,s,a,c) for l in eligible for s in range(5) for a in ('DEND','POINT') for c in (('intact','retrained','cut') if l==dev['c_level'] else ('intact',))}
    rows=summary['rows'];observed={(z['level'],z['seed'],z['arm'],z['condition']) for z in rows}
    check(observed==expected and len(rows)==len(expected),'every frozen test model/seed exactly once')
    correct={};pairs={};episodes=0;maxerr=0
    expected_files={f'test_L{l}_s{s}.npz' for l in eligible for s in range(5)}|{f'test_L{l}_s{s}_{a}_{c}.npz' for l,s,a,c in expected}
    check(expected_files==set(summary['hashes']),'complete expected test file set')
    check({p.name for p in out.glob('test_*.npz')}==expected_files,'no tests for ineligible or ceiling levels')
    for level in eligible:
        for s in range(5):
            seed=2026114002+100*level+10*s;x,y,p=regenerate(level,512,seed);pairs[level,s]=p
            saved=np.load(out/f'test_L{level}_s{s}.npz',allow_pickle=False)
            check(int(saved['seed'])==seed and np.array_equal(x,saved['x']) and np.array_equal(y,saved['y']) and np.array_equal(p,saved['pair']),'untouched test-input RNG')
            for row in [z for z in rows if z['level']==level and z['seed']==s]:
                arm=row['arm'];condition=row['condition'];base='intact' if condition=='cut' else condition
                source=next(z for z in rep['rows'] if (z['level'],z['seed'],z['arm'],z['condition'])==(level,s,arm,base))
                check(source['file']==row['source_file'],'test checkpoint provenance')
                theta=np.load(out/source['file'],allow_pickle=False);v=replay(x,theta,arm,condition=='cut');episodes+=len(x)
                f=np.load(out/row['file'],allow_pickle=False)
                check(set(f.files)=={'counts','prediction','spikes','states','candidates'},'complete saved test trajectory fields')
                for name in f.files:
                    if name in ('states','candidates'):
                        residual=float(np.max(abs(v[name]-f[name])));maxerr=max(maxerr,residual);check(residual<1e-10,'native trajectory roundoff')
                    else:check(np.array_equal(v[name],f[name]),'exact native spikes/counts/predictions')
                check_metric_record(row,scalar_metrics(v,y),'test record')
                correct[level,s,arm,condition]=v['prediction']==y
    differences={};interventions={};claims={}
    for level in eligible:
        ds=[(correct[level,s,'DEND','intact'].astype(float)-correct[level,s,'POINT','intact'],pairs[level,s]) for s in range(5)]
        result=paired_interval(ds,2026115000+level);differences[str(level)]=result
        bg=next(z['gates'] for z in dev['levels'] if z['level']==level)
        intact=[z for z in rows if z['level']==level and z['condition']=='intact'];tg,_=gate_for_rows(intact)
        delta=[]
        for s in range(5):
            left=next(z['validation']['accuracy'] for z in logs if (z['stage'],z['level'],z['seed'],z['arm'])==('B',level,s,'DEND'))
            right=next(z['validation']['accuracy'] for z in logs if (z['stage'],z['level'],z['seed'],z['arm'])==('B',level,s,'POINT'));delta.append(left-right)
        positive=all(d>0 for d in delta) and all(d.mean()>0 for d,p in ds) and bg['DEND'] and not bg['POINT'] and tg['DEND'] and not tg['POINT'] and result['percentile95'][0]>0
        negative=all(d<0 for d in delta) and all(d.mean()<0 for d,p in ds) and bg['POINT'] and not bg['DEND'] and tg['POINT'] and not tg['DEND'] and result['percentile95'][1]<0
        claims[str(level)]=dict(development_gates=bg,test_gates=tg,difference_criteria_met=bool(positive or negative))
        if level==dev['c_level']:
            for arm in ('DEND','POINT'):
                for control in ('retrained','cut'):
                    ds=[(correct[level,s,arm,'intact'].astype(float)-correct[level,s,arm,control],pairs[level,s]) for s in range(5)]
                    interventions[f'L{level}_{arm}_intact_minus_{control}']=paired_interval(ds,2026115000+level)
    check(differences==summary['differences'],'all paired structural bootstrap intervals')
    check(interventions==summary['interventions'],'all paired intervention bootstrap intervals')
    check(claims==summary['claim_checks'],'all three structural-claim criteria and seed direction')
    return dict(test_model_episodes=episodes,test_scalar_state_updates=episodes*24*20,max_state_candidate_residual=maxerr)

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--development-only',action='store_true');args=parser.parse_args()
    start=time.perf_counter();out=R/'results';freeze=read_json(R/'FREEZE.json')
    for name,h in freeze['sha256'].items():check(sha(R/name)==h,'frozen source '+name)
    for name,h in freeze['dependencies'].items():check(sha(R.parent/name)==h,'preserved dependency '+name)
    dev=read_json(out/'DEVELOPMENT_LOCK.json');check(dev['freeze_sha256']==sha(R/'FREEZE.json'),'development freeze identity')
    for name,h in dev['files'].items():check(sha(out/name)==h,'development locked file '+name)
    check(dev['test_generated'] is False,'development lock precedes tests')
    log=read_json(out/'development_log.json');dataset_names={z['data_file'] for z in log};datasets={name:inspect_data(out,name) for name in sorted(dataset_names)}
    development=audit_log(out,log,datasets);table=audit_development(out,log,dev)
    replication=dict(trajectories=0,updates=0,sampled_objective_values=0);testing=dict(test_model_episodes=0,test_scalar_state_updates=0,max_state_candidate_residual=0)
    if not args.development_only:
        rep=read_json(out/'REPLICATION_LOCK.json');rlog=rep['rows'];audit_replication(out,dev,rep,rlog)
        rnames={z['data_file'] for z in rlog};rdatasets={name:inspect_data(out,name) for name in sorted(rnames)}
        replication=audit_log(out,rlog,rdatasets);testing=audit_tests(out,dev,rep,log)
    result=dict(status='PASS',mode='development_only' if args.development_only else 'complete',checks=checks,development=development,replication=replication,**testing,seconds=time.perf_counter()-start,scope='Separate native equations/RNG/metrics/gates/bootstrap; complete stored-gradient Adam and probe-derived SPSA update reconstruction; independent first-step training objective checks. Other gradient values checked by separate gradient audit, not claimed independently recomputed here. Same workflow, not external replication.')
    (out/('verification_development.json' if args.development_only else 'verification.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
