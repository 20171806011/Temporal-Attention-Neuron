from pathlib import Path
import json,hashlib
import numpy as np
import model
R=Path(__file__).resolve().parent
OUT=R/'results'
ARMS=('POINT',)
RECIPES={'RATE_0.01':('RATE',.01),'RATE_0.03':('RATE',.03),'CE_0.01':('CE',.01),'CE_0.03':('CE',.03)}
REGISTRY=json.loads((R/'SEEDS.json').read_text())
CHECKPOINTS=REGISTRY['checkpoint_updates']
CONTRASTS=json.loads((R/'CONTRASTS.json').read_text())

def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze_check():
    for n,h in json.loads((R/'FREEZE.json').read_text())['sha256'].items():assert sha(R/n)==h,n

def count_metrics(c,y):
    pred=np.where(c[:,0]==c[:,1],-1,c.argmax(1));n=len(y);cm=np.zeros((2,3),dtype=int)
    for t,p in zip(y,pred):cm[int(t),int(p) if p>=0 else 2]+=1
    silent=np.all(c==0,1);ties=pred<0;marg=c[np.arange(n),y]-c[np.arange(n),1-y]
    acc=float(np.mean(pred==y));cov=float(np.mean(pred>=0));sf=float(silent.mean())
    return dict(accuracy=acc,coverage=cov,abstentions=int(ties.sum()),abstention_rate=float(ties.mean()),
        silent_fraction=sf,nonsilent_tie_fraction=float((ties&~silent).mean()),both_saturated_fraction=float(np.all(c==8,1).mean()),
        class_recall=[float(cm[k,k]/np.sum(y==k)) for k in (0,1)],confusion=cm.tolist(),
        mean_signed_count_margin=float(marg.mean()),median_signed_count_margin=float(np.median(marg)),
        signed_count_margin_histogram=np.bincount(marg.astype(int)+8,minlength=17).tolist(),
        failed=bool(acc<.9 or cov<.9),fully_silent=bool(sf==1),fully_abstaining=bool(cov==0))

def describe(v,y):
    q=count_metrics(v['counts'],y);z=v['spikes'];c=v['candidates']
    q.update(branch_event_rate=float(z[:,:,:12].mean()),hidden_event_rate=float(z[:,:,12:18].mean()),
        output_event_rate=float(z[:,:,18:].mean()),readout_hidden_event_rate=float(z[:,16:,12:18].mean()),
        readout_output_event_rate=float(z[:,16:,18:].mean()),
        output_candidate_margin_quantiles=np.quantile(c[:,16:,18:]-.5,[0,.05,.5,.95,1]).tolist())
    return q

def save_native(path,v):
    np.savez_compressed(path,**{k:v[k] for k in ('counts','prediction','spikes','states','candidates')})

def bootstrap(values,primary=False):
    a=np.asarray(values,dtype=float);n=REGISTRY['n_pairs'];assert a.shape==(n,)
    rng=np.random.default_rng(REGISTRY['analysis_seed']);ix=rng.integers(0,n,(20000,n));means=a[ix].mean(1)
    result=dict(mean=float(a.mean()),percentile95=np.quantile(means,[.025,.975]).tolist(),paired_values=a.tolist(),
        n_pairs=n,unit='independently trained seed/data block paired across four POINT cells',degenerate=bool(np.ptp(means)==0),primary_accuracy=primary)
    if primary:
        sd=float(a.std(ddof=1));se=sd/np.sqrt(n);q=CONTRASTS['primary_t_critical']
        result.update(student_t95=[float(a.mean()-q*se),float(a.mean()+q*se)],sample_sd=sd,standard_error=float(se),degrees_of_freedom=n-1,t_critical=q)
    return result

def wilson(k,n=104):
    z=1.959963984540054;p=k/n;d=1+z*z/n;center=(p+z*z/(2*n))/d;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(max(0,center-half)),float(min(1,center+half))]

def analyze(testrows,trainrows):
    def vals(arm,recipe,key,rows=testrows):
        return np.array([next(r[key] for r in rows if (r['arm'],r['recipe'],r['seed'])==(arm,recipe,s)) for s in range(REGISTRY['n_pairs'])],dtype=float)
    intervals={}
    for metric in ('accuracy','failed'):
        blocks=np.column_stack([vals(*column.split('|'),metric) for column in CONTRASTS['columns']])
        for contrast in CONTRASTS['contrasts']:
            delta=blocks@np.asarray(contrast['coefficients'])
            intervals[metric+'_'+contrast['name']]=bootstrap(delta,primary=metric=='accuracy' and contrast['primary_accuracy'])
    cells=[]
    for arm in ARMS:
        for recipe in RECIPES:
            rr=[r for r in testrows if (r['arm'],r['recipe'])==(arm,recipe)];tr=[r for r in trainrows if (r['arm'],r['recipe'])==(arm,recipe)]
            rates={k:dict(count=sum(int(r[k]) for r in rr),fraction=float(np.mean([r[k] for r in rr])),wilson95=wilson(sum(int(r[k]) for r in rr))) for k in ('failed','fully_silent','fully_abstaining')}
            transitions={f'train_{a}_test_{b}':sum((not bool(t['failed']))==a and (not bool(r['failed']))==b for t,r in zip(sorted(tr,key=lambda r:r['seed']),sorted(rr,key=lambda r:r['seed']))) for a in (False,True) for b in (False,True)}
            cells.append(dict(arm=arm,recipe=recipe,n=104,mean_accuracy=float(np.mean([r['accuracy'] for r in rr])),
                mean_coverage=float(np.mean([r['coverage'] for r in rr])),mean_silent_fraction=float(np.mean([r['silent_fraction'] for r in rr])),
                mean_abstention_rate=float(np.mean([r['abstention_rate'] for r in rr])),mean_nonsilent_tie_fraction=float(np.mean([r['nonsilent_tie_fraction'] for r in rr])),
                mean_train_accuracy=float(np.mean([r['accuracy'] for r in tr])),run_events=rates,pass_transitions=transitions))
            intervals[f'{arm}_{recipe}_test_minus_train_accuracy']=bootstrap(vals(arm,recipe,'accuracy')-vals(arm,recipe,'accuracy',trainrows))
    primary_name='accuracy_POINT_loss_lr_interaction';v=intervals[primary_name];lo,hi=v['student_t95']
    interpretation=dict(primary_interval_method='two-sided paired-mean Student-t95',
        negative_direction_supported=bool(hi<0),opposite_direction_supported=bool(lo>0),
        negative_magnitude_beyond_5pp_supported=bool(hi<-.05),
        bootstrap_negative_direction=bool(v['percentile95'][1]<0),
        bootstrap_opposite_direction_supported=bool(v['percentile95'][0]>0),
        bootstrap_direction_disagrees=bool((hi<0,lo>0)!=(v['percentile95'][1]<0,v['percentile95'][0]>0)),
        limits='Primary t interval governs interpretation; bootstrap is prespecified sensitivity only. No equivalence from zero crossing or degenerate intervals. New-data internal replication, not external independent replication.')
    return dict(primary_family=[primary_name],intervals=intervals,cells=cells,primary_interpretation=interpretation,
        multiplicity='One prespecified primary interaction with two-sided Student-t95 interval. Bootstrap95 is sensitivity only; all other intervals are unadjusted secondary exploratory descriptions.',
        interpretation='104 independent paired seed/data blocks across four fixed POINT cells; no selection, no DEND comparison, no pooling with V7, no optional stopping.')
