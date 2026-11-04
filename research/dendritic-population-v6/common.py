from pathlib import Path
import json,hashlib
import numpy as np
import model
R=Path(__file__).resolve().parent
OUT=R/'results'
ARMS=('DEND','POINT')
RECIPES={'RATE_0.01':('RATE',.01),'CE_0.03':('CE',.03)}
REGISTRY=json.loads((R/'SEEDS.json').read_text())
CHECKPOINTS=REGISTRY['checkpoint_updates']

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

def bootstrap(values):
    a=np.asarray(values,dtype=float);assert a.shape==(20,)
    rng=np.random.default_rng(REGISTRY['analysis_seed']);ix=rng.integers(0,20,(20000,20));means=a[ix].mean(1)
    return dict(mean=float(a.mean()),percentile95=np.quantile(means,[.025,.975]).tolist(),paired_values=a.tolist(),
        n_pairs=20,unit='independently trained seed/data pair',degenerate=bool(np.ptp(means)==0))

def wilson(k,n=20):
    z=1.959963984540054;p=k/n;d=1+z*z/n;center=(p+z*z/(2*n))/d;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(max(0,center-half)),float(min(1,center+half))]

def analyze(testrows,trainrows):
    def vals(arm,recipe,key,rows=testrows):
        return np.array([next(r[key] for r in rows if (r['arm'],r['recipe'],r['seed'])==(arm,recipe,s)) for s in range(20)],dtype=float)
    intervals={}
    for key in ('accuracy','failed'):
        dc,dr=vals('DEND','CE_0.03',key),vals('DEND','RATE_0.01',key)
        pc,pr=vals('POINT','CE_0.03',key),vals('POINT','RATE_0.01',key)
        for name,a in [('DEND_CE_minus_RATE',dc-dr),('POINT_CE_minus_RATE',pc-pr),
                       ('RATE_DEND_minus_POINT',dr-pr),('CE_DEND_minus_POINT',dc-pc),('recipe_by_architecture_interaction',(dc-dr)-(pc-pr))]:
            intervals[key+'_'+name]=bootstrap(a)
    cells=[]
    for arm in ARMS:
        for recipe in RECIPES:
            rr=[r for r in testrows if (r['arm'],r['recipe'])==(arm,recipe)];tr=[r for r in trainrows if (r['arm'],r['recipe'])==(arm,recipe)]
            rates={k:dict(count=sum(int(r[k]) for r in rr),fraction=float(np.mean([r[k] for r in rr])),wilson95=wilson(sum(int(r[k]) for r in rr))) for k in ('failed','fully_silent','fully_abstaining')}
            transitions={f'train_{a}_test_{b}':sum((not bool(t['failed']))==a and (not bool(r['failed']))==b for t,r in zip(sorted(tr,key=lambda r:r['seed']),sorted(rr,key=lambda r:r['seed']))) for a in (False,True) for b in (False,True)}
            cells.append(dict(arm=arm,recipe=recipe,n=20,mean_accuracy=float(np.mean([r['accuracy'] for r in rr])),
                mean_coverage=float(np.mean([r['coverage'] for r in rr])),mean_silent_fraction=float(np.mean([r['silent_fraction'] for r in rr])),
                mean_abstention_rate=float(np.mean([r['abstention_rate'] for r in rr])),mean_nonsilent_tie_fraction=float(np.mean([r['nonsilent_tie_fraction'] for r in rr])),
                mean_train_accuracy=float(np.mean([r['accuracy'] for r in tr])),run_events=rates,pass_transitions=transitions))
            intervals[f'{arm}_{recipe}_test_minus_train_accuracy']=bootstrap(vals(arm,recipe,'accuracy')-vals(arm,recipe,'accuracy',trainrows))
    return dict(primary='accuracy_DEND_CE_minus_RATE',intervals=intervals,cells=cells,
        interpretation='Exploratory fixed-recipe study,20pairedtrainingseeds; no hyperparameter selection. Positive failure difference is worse; zero-crossing is not equivalence; degenerate bootstrap does not prove population equality.')
