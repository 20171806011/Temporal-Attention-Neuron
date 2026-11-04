from pathlib import Path
import json,hashlib
import numpy as np
import model
R=Path(__file__).resolve().parent
OUT=R/'results'
ARMS=('DEND','POINT')
CONDITIONS=('intact','retrained','cut')
REGISTRY=json.loads((R/'SEEDS.json').read_text())
CHECKPOINTS=REGISTRY['checkpoint_updates']

def mask():
    active=np.ones(176);wr=active[72:144].reshape(6,12)
    for source in range(6):
        for branch in range(12):
            wr[source,branch]=float(source==branch//2)
    assert active.sum()==116
    return active


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
    q['balanced_accuracy']=float(np.mean(q['class_recall']))
    q.update(activity_latency(v,y))
    q.update(branch_event_rate=float(z[:,:,:12].mean()),hidden_event_rate=float(z[:,:,12:18].mean()),
        output_event_rate=float(z[:,:,18:].mean()),readout_hidden_event_rate=float(z[:,16:,12:18].mean()),
        readout_output_event_rate=float(z[:,16:,18:].mean()),
        output_candidate_margin_quantiles=np.quantile(c[:,16:,18:]-.5,[0,.05,.5,.95,1]).tolist())
    return q

def save_native(path,v):
    np.savez_compressed(path,**{k:v[k] for k in ('counts','prediction','spikes','states','candidates')})

def activity_latency(v,y):
    spikes=v['spikes'];states=v['states'];n=len(y)
    prefix=np.cumsum(spikes[:,16:,18:],axis=1);pred=np.where(prefix[:,:,0]==prefix[:,:,1],-1,prefix.argmax(2));final=pred[:,-1]
    settled=np.logical_and.accumulate((pred==final[:,None])[:,::-1],axis=1)[:,::-1]&(final[:,None]>=0)
    correct=settled&(final[:,None]==y[:,None])
    variation=np.abs(np.diff(np.concatenate([np.zeros_like(states[:,:1]),states],axis=1),axis=1)).sum((1,2))
    return dict(spike_events_mean={name:float(spikes[:,:,lo:hi].sum((1,2)).mean()) for name,lo,hi in [('branch',0,12),('hidden',12,18),('output',18,20)]},
        total_spike_events_mean=float(spikes.sum((1,2)).mean()),state_l1_change_mean=float(variation.mean()),
        correct_settled_fraction_by_step=correct.mean(0).tolist(),unique_settled_fraction_by_step=settled.mean(0).tolist(),
        latency_scope='Retrospective prefix-count settling within readout steps1..8, no label-conditioned dynamics or online stopping; wrong/abstaining fits stay in all-episode denominator. Activity is not energy.')

def bootstrap(values):
    a=np.asarray(values,dtype=float);assert a.shape==(5,)
    ix=np.random.default_rng(REGISTRY['analysis_seed']).integers(0,5,(20000,5));means=a[ix].mean(1)
    return dict(mean=float(a.mean()),percentile95=np.quantile(means,[.025,.975]).tolist(),paired_values=a.tolist(),n_pairs=5,degenerate=bool(np.ptp(a)==0),unit='paired training/data seed block')

def wilson(k,n=5):
    z=1.959963984540054;p=k/n;d=1+z*z/n;center=(p+z*z/(2*n))/d;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(max(0,center-half)),float(min(1,center+half))]

def analyze(testrows,trainrows):
    def vals(arm,condition,key):
        return np.array([next(r[key] for r in testrows if (r['arm'],r['condition'],r['seed'])==(arm,condition,s)) for s in range(5)],float)
    cells=[];intervals={};interpretation={}
    for arm in ARMS:
        for condition in CONDITIONS:
            rows=[r for r in testrows if (r['arm'],r['condition'])==(arm,condition)]
            cells.append(dict(arm=arm,condition=condition,n=5,mean_accuracy=float(vals(arm,condition,'accuracy').mean()),
                mean_balanced_accuracy=float(vals(arm,condition,'balanced_accuracy').mean()),mean_coverage=float(vals(arm,condition,'coverage').mean()),
                failures=int(vals(arm,condition,'failed').sum()),failure_wilson95=wilson(int(vals(arm,condition,'failed').sum())),
                fully_silent=int(vals(arm,condition,'fully_silent').sum()),fully_abstaining=int(vals(arm,condition,'fully_abstaining').sum()),
                mean_total_spikes=float(vals(arm,condition,'total_spike_events_mean').mean()),mean_state_l1_change=float(vals(arm,condition,'state_l1_change_mean').mean()),
                correct_settled_fraction_by_step=np.mean([r['correct_settled_fraction_by_step'] for r in rows],axis=0).tolist()))
        for metric in ('accuracy','failed'):
            for condition in ('retrained','cut'):
                intervals[f'{arm}_intact_minus_{condition}_{metric}']=bootstrap(vals(arm,'intact',metric)-vals(arm,condition,metric))
        intact=vals(arm,'intact','failed')==0;retrained=vals(arm,'retrained','failed')==0
        interpretation[arm]=dict(intact_passes=int(intact.sum()),retrained_passes=int(retrained.sum()),all_intact_pass=bool(intact.all()),all_retrained_pass=bool(retrained.all()),
            observed_no_cross_success=bool(retrained.any()),limits='No-cross retains multiple cells, self recurrence and state. Success refutes necessity for passing this finite task/gate; failure only shows this recipe/budget did not learn reliably. No physical impossibility or single-cell conclusion.')
    return dict(primary_family=[a+'_intact_minus_retrained_accuracy' for a in ARMS],cells=cells,intervals=intervals,interpretation=interpretation,
        uncertainty='Five paired seed blocks; all eight intervals pointwise exploratory without multiple-comparison protection. Degenerate or zero-crossing intervals do not establish equivalence. No hyperparameter search.')
