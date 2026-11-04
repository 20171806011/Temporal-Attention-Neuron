from pathlib import Path
import json,hashlib,subprocess
import numpy as np
import model
R=Path(__file__).resolve().parent
OUT=R/'results'
ARMS=('DEND','POINT')
SIZES=(6,4,2,1)
REGISTRY=json.loads((R/'SEEDS.json').read_text())
CHECKPOINTS=REGISTRY['checkpoint_updates']

def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze_check():
    for n,h in json.loads((R/'FREEZE.json').read_text())['sha256'].items():assert sha(R/n)==h,n

def require_committed(paths):
    repo=R.parent.parent
    head=subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,check=True,text=True).stdout.strip()
    for path in paths:
        p=Path(path);relative=p.relative_to(repo).as_posix()
        blob=subprocess.run(['git','show',f'{head}:{relative}'],cwd=repo,capture_output=True,check=True).stdout
        assert hashlib.sha256(blob).hexdigest()==sha(p),f'Uncommitted or changed gate file: {relative}'
    return head

def count_metrics(c,y):
    pred=np.where(c[:,0]==c[:,1],-1,c.argmax(1));n=len(y);cm=np.zeros((2,3),dtype=int)
    for t,p in zip(y,pred):cm[int(t),int(p) if p>=0 else 2]+=1
    silent=np.all(c==0,1);ties=pred<0;marg=c[np.arange(n),y]-c[np.arange(n),1-y]
    acc=float(np.mean(pred==y));cov=float(np.mean(pred>=0));sf=float(silent.mean())
    recall=[float(cm[k,k]/np.sum(y==k)) for k in (0,1)];ba=float(np.mean(recall))
    return dict(accuracy=acc,balanced_accuracy=ba,coverage=cov,abstentions=int(ties.sum()),abstention_rate=float(ties.mean()),
        silent_fraction=sf,nonsilent_tie_fraction=float((ties&~silent).mean()),both_saturated_fraction=float(np.all(c==8,1).mean()),
        class_recall=recall,confusion=cm.tolist(),mean_signed_count_margin=float(marg.mean()),median_signed_count_margin=float(np.median(marg)),
        signed_count_margin_histogram=np.bincount(marg.astype(int)+8,minlength=17).tolist(),
        failed=bool(ba<.9 or cov<.9),fully_silent=bool(sf==1),fully_abstaining=bool(cov==0))

def activity_latency(v,y,h):
    spikes=v['spikes'];states=v['states']
    prefix=np.cumsum(spikes[:,16:,3*h:],axis=1);pred=np.where(prefix[:,:,0]==prefix[:,:,1],-1,prefix.argmax(2));final=pred[:,-1]
    settled=np.logical_and.accumulate((pred==final[:,None])[:,::-1],axis=1)[:,::-1]&(final[:,None]>=0)
    correct=settled&(final[:,None]==y[:,None])
    latency=np.where(correct.any(1),correct.argmax(1)+1,9)
    variation=np.abs(np.diff(np.concatenate([np.zeros_like(states[:,:1]),states],axis=1),axis=1)).sum((1,2))
    events={name:float(spikes[:,:,lo:hi].sum((1,2)).mean()) for name,lo,hi in [('branch',0,2*h),('hidden',2*h,3*h),('output',3*h,3*h+2)]}
    return dict(spike_events_mean=events,total_spike_events_mean=float(spikes.sum((1,2)).mean()),state_l1_change_mean=float(variation.mean()),
        dense_event_delivery_proxy_mean=events['branch']+(2*h+2)*events['hidden']+events['output'],
        correct_settled_fraction_by_step=correct.mean(0).tolist(),unique_settled_fraction_by_step=settled.mean(0).tolist(),
        capped_correct_settled_latency_mean=float(latency.mean()),latency_failure_cap=9,
        latency_scope='Retrospective prefix-count settling within readout steps1..8; wrong/abstaining episodes receive9. No online stopping, no label-conditioned dynamics. Activity is not energy.')

def describe(v,y,pattern,h):
    q=count_metrics(v['counts'],y);z=v['spikes'];c=v['candidates'];pred=v['prediction']
    q.update(activity_latency(v,y,h))
    q.update(pattern_recall=[float(np.mean(pred[pattern==p]==y[pattern==p])) for p in range(16)],
        branch_event_rate=float(z[:,:,:2*h].mean()),hidden_event_rate=float(z[:,:,2*h:3*h].mean()),output_event_rate=float(z[:,:,3*h:].mean()),
        readout_hidden_event_rate=float(z[:,16:,2*h:3*h].mean()),readout_output_event_rate=float(z[:,16:,3*h:].mean()),
        output_candidate_margin_quantiles=np.quantile(c[:,16:,3*h:]-.5,[0,.05,.5,.95,1]).tolist())
    return q

def save_native(path,v):
    np.savez_compressed(path,**{k:v[k] for k in ('counts','prediction','spikes','states','candidates')})

def bootstrap(values):
    a=np.asarray(values,dtype=float);assert a.shape==(5,)
    ix=np.random.default_rng(REGISTRY['analysis_seed']).integers(0,5,(20000,5));means=a[ix].mean(1)
    return dict(mean=float(a.mean()),percentile95=np.quantile(means,[.025,.975]).tolist(),paired_values=a.tolist(),n_pairs=5,degenerate=bool(np.ptp(a)==0),unit='paired training/data seed block')

def wilson(k,n=5):
    z=1.959963984540054;p=k/n;d=1+z*z/n;center=(p+z*z/(2*n))/d;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(max(0,center-half)),float(min(1,center+half))]

def analyze(testrows,trainrows):
    def vals(arm,h,key):return np.array([next(r[key] for r in testrows if (r['arm'],r['h'],r['seed'])==(arm,h,s)) for s in range(5)],float)
    cells=[];intervals={}
    for h in SIZES:
        for arm in ARMS:
            rows=[r for r in testrows if (r['arm'],r['h'])==(arm,h)]
            cells.append(dict(arm=arm,h=h,n=5,mean_accuracy=float(vals(arm,h,'accuracy').mean()),mean_balanced_accuracy=float(vals(arm,h,'balanced_accuracy').mean()),
                mean_coverage=float(vals(arm,h,'coverage').mean()),failures=int(vals(arm,h,'failed').sum()),failure_wilson95=wilson(int(vals(arm,h,'failed').sum())),
                fully_silent=int(vals(arm,h,'fully_silent').sum()),fully_abstaining=int(vals(arm,h,'fully_abstaining').sum()),
                mean_total_spikes=float(vals(arm,h,'total_spike_events_mean').mean()),mean_state_l1_change=float(vals(arm,h,'state_l1_change_mean').mean()),
                mean_capped_correct_settled_latency=float(vals(arm,h,'capped_correct_settled_latency_mean').mean()),
                correct_settled_fraction_by_step=np.mean([r['correct_settled_fraction_by_step'] for r in rows],axis=0).tolist()))
        intervals[f'h{h}_DEND_minus_POINT_balanced_accuracy']=bootstrap(vals('DEND',h,'balanced_accuracy')-vals('POINT',h,'balanced_accuracy'))
    return dict(primary_family=list(intervals),cells=cells,intervals=intervals,
        uncertainty='Five paired seed blocks; four pointwise exploratory percentile intervals without familywise protection. Zero-crossing/degenerate intervals do not establish equivalence. No best-size selection or hyperparameter search.',
        interpretation='Learnt binding/size comparison with strong spiking branch baseline. No pooling intervention; this does not isolate causal necessity of local nonlinear partition, prove representation impossibility, or establish physical energy efficiency.')

def resources(h):
    return dict(h=h,trainable_parameters=model.parameter_count(h),float_states=3*h+2,previous_hidden_spike_bits=h,readout_counters=2,
        dend_represented_cells=h+2,point_represented_cells=3*h+2,dend_graded_compartments=2*h,point_branch_cells=2*h,
        dense_matrix_macs_per_episode_step=2*h*h+14*h,local_branch_nonlinearities_per_step=2*h,
        dend_hard_thresholds_per_step=h+2,point_hard_thresholds_per_step=3*h+2,
        dend_reset_operations_per_step=h+2,point_reset_operations_per_step=3*h+2)
