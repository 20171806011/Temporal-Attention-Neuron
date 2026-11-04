"""Independent V9 data, native dynamics, optimizer and seed-unit audit.

No imports from the scientific model, trainer or evaluator. Native replay uses
separately written scalar equations. Saved gradients reconstruct Adam updates;
this does not independently rederive every surrogate gradient. Internal workflow
verification is not outside replication or peer review.
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


def active_mask(condition):
    m=np.ones(176)
    if condition in ('retrained','cut'):
        for source in range(6):
            for target in range(6):
                if source != target:
                    m[72+source*12+target*2:72+source*12+target*2+2]=0
        check(int(m.sum())==116 and int((m==0).sum())==60,'60 cross-cell slots removed,12 self slots retained')
    else:check(condition=='intact','registered condition')
    return m


def adam(initial,gradients,lr,active):
    """Reconstruct all parameters/moments; no fitting or scientific imports."""
    check(gradients.shape==(200,176),'200 by176 saved gradients')
    fixed=active==0
    check(np.all(initial[fixed]==0),'fixed initial coordinates exactly zero')
    theta=initial.copy();m=np.zeros(176);v=np.zeros(176);weights=[theta.copy()];update_norms=[]
    for k,g in enumerate(gradients):
        check(np.isfinite(g).all() and np.linalg.norm(g)<=1+1e-12,'finite globally clipped projected gradient')
        check(np.all(g[fixed]==0),'all fixed gradient coordinates exactly zero')
        m=(.9*m+.1*g)*active;v=(.999*v+.001*g*g)*active
        check(np.all(m[fixed]==0) and np.all(v[fixed]==0),'fixed Adam moments remain zero')
        new=np.clip(theta-lr*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8),-2,2)*active
        check(np.all(new[fixed]==0),'fixed parameters remain zero at every update')
        update_norms.append(float(np.linalg.norm(new-theta)));theta=new;weights.append(theta.copy())
    return np.stack(weights),np.array(update_norms)


def loss_from_counts(counts, y, theta, kind):
    if kind == 'RATE':
        target = np.full_like(counts, .125)
        target[np.arange(len(y)), y] = .75
        value = np.mean((counts / 8 - target) ** 2)
    else:
        check(kind == 'CE', 'registered loss')
        logits = counts * .5
        logits -= logits.max(1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(1, keepdims=True)
        value = -np.log(probability[np.arange(len(y)), y]).mean()
    return float(value + .0001 * np.mean(theta * theta))


def count_metrics(counts, y):
    pred = np.where(counts[:, 0] == counts[:, 1], -1, counts.argmax(1))
    n = len(y)
    matrix = np.zeros((2, 3), dtype=int)
    for truth, guess in zip(y, pred):
        matrix[int(truth), int(guess) if guess >= 0 else 2] += 1
    accuracy = float(np.mean(pred == y))
    coverage = float(np.mean(pred >= 0))
    silent = np.all(counts == 0, axis=1)
    ties = pred < 0
    margins = counts[np.arange(n), y] - counts[np.arange(n), 1-y]
    histogram = [int(np.sum(margins == k)) for k in range(-8, 9)]
    return dict(accuracy=accuracy, coverage=coverage,
                abstentions=int(np.sum(pred < 0)),
                abstention_rate=float(np.mean(pred < 0)),
                silent_fraction=float(silent.mean()),
                nonsilent_tie_fraction=float(np.mean(ties & ~silent)),
                both_saturated_fraction=float(np.all(counts == 8, axis=1).mean()),
                class_recall=[float(matrix[k, k] / np.sum(y == k)) for k in (0, 1)],
                confusion=matrix.tolist(),
                mean_signed_count_margin=float(margins.mean()),
                median_signed_count_margin=float(np.median(margins)),
                signed_count_margin_histogram=histogram,
                failed=bool(accuracy < .9 or coverage < .9),
                fully_silent=bool(silent.all()),
                fully_abstaining=bool(ties.all()))


def independent_description(v,y):
    result=count_metrics(v['counts'],y);spikes=v['spikes'];states=v['states'];n=len(y)
    result['balanced_accuracy']=float(np.mean(result['class_recall']))
    groups={'branch_event_rate':spikes[:,:,:12],'hidden_event_rate':spikes[:,:,12:18],
            'output_event_rate':spikes[:,:,18:],'readout_hidden_event_rate':spikes[:,16:,12:18],
            'readout_output_event_rate':spikes[:,16:,18:]}
    result.update({key:float(value.mean()) for key,value in groups.items()})
    result['output_candidate_margin_quantiles']=np.quantile(v['candidates'][:,16:,18:]-.5,[0,.05,.5,.95,1]).tolist()
    result['spike_events_mean']={name:float(spikes[:,:,lo:hi].sum((1,2)).mean())
                                for name,lo,hi in [('branch',0,12),('hidden',12,18),('output',18,20)]}
    result['total_spike_events_mean']=float(spikes.sum((1,2)).mean())
    # Explicit recurrence/loop calculation, separate from common.py's diff/accumulate.
    previous=np.zeros_like(states[:,0]);variation=np.zeros(n)
    for step in range(24):
        variation+=np.abs(states[:,step]-previous).sum(1);previous=states[:,step]
    result['state_l1_change_mean']=float(variation.mean())
    cumulative=np.zeros((n,2));winners=[]
    for step in range(16,24):
        cumulative+=spikes[:,step,18:]
        winners.append(np.where(cumulative[:,0]==cumulative[:,1],-1,cumulative.argmax(1)))
    winners=np.stack(winners,axis=1);final=winners[:,-1];correct=[];unique=[]
    for k in range(8):
        settled=(final>=0)&np.all(winners[:,k:]==final[:,None],axis=1)
        unique.append(float(settled.mean()));correct.append(float((settled&(final==y)).mean()))
    result['correct_settled_fraction_by_step']=correct
    result['unique_settled_fraction_by_step']=unique
    result['latency_scope']='Retrospective prefix-count settling within readout steps1..8, no label-conditioned dynamics or online stopping; wrong/abstaining fits stay in all-episode denominator. Activity is not energy.'
    check(abs(correct[-1]-result['accuracy'])<1e-15,'final settled-correct fraction equals accuracy')
    check(abs(unique[-1]-result['coverage'])<1e-15,'final settled-unique fraction equals coverage')
    check(all(correct[k]<=correct[k+1] for k in range(7)),'retrospective correct-settling curve monotone')
    check(all(unique[k]<=unique[k+1] for k in range(7)),'retrospective unique-settling curve monotone')
    return result


def verify_native(saved, actual, why):
    max_residual = 0.0
    for field in ('spikes', 'counts', 'prediction'):
        check(np.array_equal(saved[field], actual[field]), why + ' exact ' + field)
    check(np.all((actual['spikes'] == 0) | (actual['spikes'] == 1)),
          why + ' hard binary spikes')
    for field in ('states', 'candidates'):
        residual = float(np.max(abs(actual[field] - saved[field])))
        max_residual = max(max_residual, residual)
        check(residual < 1e-10, why + ' scalar replay ' + field)
    return max_residual


def paired_interval(delta, rng_seed, quantiles=(.025, .975)):
    delta = np.asarray(delta, dtype=float)
    check(delta.shape == (5,), 'five independently trained paired seed blocks')
    generator = np.random.default_rng(rng_seed)
    indices = generator.integers(0, 5, size=(20000, 5))
    means = delta[indices].mean(1)
    return float(delta.mean()), np.quantile(means, quantiles).tolist()


def wilson(k, n=5):
    check(isinstance(k, (int, np.integer)) and 0 <= k <= n,
          'integer event count')
    z = 1.959963984540054
    proportion = k / n
    den = 1 + z*z/n
    center = (proportion + z*z/(2*n)) / den
    half = z / den * np.sqrt(proportion*(1-proportion)/n + z*z/(4*n*n))
    return [max(0.0, float(center-half)), min(1.0, float(center+half))]


def compare(actual, expected, why, atol=1e-12):
    """Recursive comparison, exact categories and tightly bounded float roundoff."""
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and actual.keys() == expected.keys(), why + ' keys')
        for key, value in expected.items():
            compare(actual[key], value, why + '.' + key, atol)
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(actual) == len(expected), why + ' list')
        for i, value in enumerate(expected):
            compare(actual[i], value, why + '[' + str(i) + ']', atol)
    elif isinstance(expected, bool):
        check(type(actual) is bool and actual == expected, why + ' boolean')
    elif isinstance(expected, (float, np.floating)):
        check(np.isfinite(actual) and abs(actual-expected) <= atol, why + ' numeric')
    else:
        check(actual == expected, why + ' exact')


def check_description(row, native, labels, why):
    description = independent_description(native, labels)
    compare({key: row[key] for key in description}, description, why, atol=1e-10)


def inspect_dataset(out, name, n, expected_seed):
    with np.load(out/name, allow_pickle=False) as file:
        check(set(file.files) == {'x', 'y', 'pair', 'seed'}, 'dataset fields ' + name)
        d = {key: file[key] for key in file.files}
    check(int(d['seed']) == expected_seed, 'prespecified dataset seed ' + name)
    x, y, pair = regenerate_xor(n, expected_seed)
    check(d['x'].shape == (n, 24, 6), 'input shape ' + name)
    check(np.array_equal(d['x'], x) and np.array_equal(d['y'], y)
          and np.array_equal(d['pair'], pair), 'separate delayed XOR generator ' + name)
    check(np.array_equal(np.bincount(pair, minlength=4), np.full(4, n//4)),
          'exact pair balance ' + name)
    check(np.array_equal(y, (pair//2) ^ (pair % 2)), 'XOR target ' + name)
    return d


def audit_training(out,lock,registry):
    rows=lock['rows'];wanted={(s,a,c) for s in range(5) for a in ('DEND','POINT') for c in ('intact','retrained')}
    keys=[(r['seed'],r['arm'],r['condition']) for r in rows]
    check(len(rows)==len(set(keys))==20 and set(keys)==wanted,'20 complete registered training runs')
    check(lock['history_columns']==['loss','raw_gradient_norm','update_norm','preupdate_accuracy','preupdate_coverage','preupdate_silent_fraction','preupdate_mean_signed_count_margin','preupdate_nonsilent_tie_fraction'],'history schema')
    datasets={f'train_s{s}.npz':inspect_dataset(out,f'train_s{s}.npz',128,2090000000+10*s) for s in range(5)}
    checkpoint_updates=registry['checkpoint_updates'];updates=checkpoints=exact_train_episodes=initial_objectives=0;max_residual=0.0;masked_updates=0
    for row in rows:
        s,arm,condition=row['seed'],row['arm'],row['condition']
        check(row['id']==f's{s}_{arm}_{condition}','condition identity')
        check((row['loss'],row['lr'],row['steps'])==('RATE',.01,200),'fixed recipe and budget')
        check(row['initial_seed']==2090020000+s,'paired initialization seed')
        check(row['data_file']==f'train_s{s}.npz','matched train inputs')
        active=active_mask(condition);fixed=active==0;d=datasets[row['data_file']]
        initial=initialize(2090020000+s)*active
        check(np.array_equal(initial,np.load(out/row['initial_file'],allow_pickle=False)),'exact paired base initialization plus prescribed mask')
        check(sha(out/row['file'])==row['sha256'],'final weight hash')
        theta=np.load(out/row['file'],allow_pickle=False)
        check(theta.shape==(176,) and np.isfinite(theta).all(),'finite final weights')
        check(np.all(theta[fixed]==0),'masked final weights exactly zero')
        with np.load(out/row['trace'],allow_pickle=False) as tr:
            check(set(tr.files)=={'history','clipped_gradients','train_counts'},'complete trace schema')
            history,gradients,counts=tr['history'],tr['clipped_gradients'],tr['train_counts']
        check(history.shape==(200,8) and np.isfinite(history).all(),'finite full history')
        check(counts.shape==(200,128,2) and np.isfinite(counts).all() and np.all((counts>=0)&(counts<=8)) and np.array_equal(counts,counts.astype(int)),'native bounded integer counts')
        weights,update_norms=adam(initial,gradients,row['lr'],active)
        check(np.array_equal(weights[-1],theta),'all Adam steps reconstruct final weights exactly')
        check(np.allclose(update_norms,history[:,2],atol=1e-12,rtol=0),'all update norms')
        check(np.allclose(np.linalg.norm(gradients,axis=1),np.minimum(history[:,1],1),atol=1e-12,rtol=0),'projected raw gradient norm and clipping')
        with np.load(out/row['checkpoints'],allow_pickle=False) as cp:
            check(set(cp.files)=={'updates','theta'},'checkpoint schema')
            check(np.array_equal(cp['updates'],checkpoint_updates),'fixed checkpoint schedule')
            check(np.array_equal(cp['theta'],weights[checkpoint_updates]),'all saved checkpoint vectors reconstruct')
            check(np.all(cp['theta'][:,fixed]==0),'all saved masked checkpoint slots zero')
        for k in range(200):
            cm=count_metrics(counts[k],d['y'])
            expected=[loss_from_counts(counts[k],d['y'],weights[k],'RATE'),cm['accuracy'],cm['coverage'],cm['silent_fraction'],cm['mean_signed_count_margin'],cm['nonsilent_tie_fraction']]
            check(np.allclose(history[k,[0,3,4,5,6,7]],expected,atol=1e-12,rtol=0),'every count-derived loss/metric history')
        diagnostics=read_json(out/row['diagnostics']);check(len(diagnostics)==9,'nine diagnostic rows')
        for j,k in enumerate(checkpoint_updates):
            check(diagnostics[j]['updates']==k,'diagnostic checkpoint alignment')
            native=replay(d['x'],weights[k],arm);check_description(diagnostics[j],native,d['y'],'checkpoint native metrics/activity/latency')
            if k<200:
                check(np.array_equal(native['counts'],counts[k]),'checkpoint counts match stored update trace')
                check(abs(loss_from_counts(native['counts'],d['y'],weights[k],'RATE')-history[k,0])<1e-12,'checkpoint objective independently replayed')
                if k==0:initial_objectives+=1
            else:
                with np.load(out/row['train_native'],allow_pickle=False) as saved:
                    check(set(saved.files)=={'counts','prediction','spikes','states','candidates'},'full final training native trace')
                    max_residual=max(max_residual,verify_native(saved,native,'final training'))
                check_description(row,native,d['y'],'final training row');exact_train_episodes+=len(d['y'])
            checkpoints+=1
        updates+=200
        if condition=='retrained':masked_updates+=200
    return datasets,dict(trajectories=20,reconstructed_updates=updates,masked_updates_with_zero_parameter_gradient_and_moments=masked_updates,
        native_training_checkpoints=checkpoints,native_training_checkpoint_episodes=checkpoints*128,
        full_saved_training_trajectory_episodes=exact_train_episodes,independently_replayed_initial_objectives=initial_objectives,
        count_derived_history_rows=updates,max_training_state_candidate_residual=max_residual)


def audit_cut_weights(out,lock):
    rows=lock['cut_rows'];wanted={(s,a) for s in range(5) for a in ('DEND','POINT')}
    check(len(rows)==10 and {(r['seed'],r['arm']) for r in rows}==wanted,'all10 predetermined inference cuts')
    training={r['id']:r for r in lock['rows']};active=active_mask('cut')
    for row in rows:
        s,a=row['seed'],row['arm'];source=training[f's{s}_{a}_intact']
        check(row['id']==f's{s}_{a}_cut' and row['condition']=='cut','cut identity')
        check(row['source_id']==source['id'] and row['source_file']==source['file'],'cut source is paired trained intact model')
        check((row['loss'],row['lr'],row['steps'])==('RATE',.01,0),'cut adds no training')
        check(row['data_file']==source['data_file'],'cut retains source data provenance')
        actual=np.load(out/row['file'],allow_pickle=False);expected=np.load(out/source['file'],allow_pickle=False)*active
        check(np.array_equal(actual,expected),'cut weights exactly mask times trained intact weights')
        check(np.all(actual[active==0]==0),'all60 cut coordinates exactly zero')
        check(sha(out/row['file'])==row['sha256'],'cut weight hash')
        check(row['file'] in lock['files'],'cut weight file included in pretest lock')
    return dict(derived_models=10,additional_training_updates=0,mask_zero_slots=60,self_recurrent_slots_retained=12)


def interval_record(values,seed):
    a=np.asarray(values,dtype=float);mean,ci=paired_interval(a,seed)
    return dict(mean=mean,percentile95=ci,paired_values=a.tolist(),n_pairs=5,degenerate=bool(np.ptp(a)==0),unit='paired training/data seed block')


def independent_statistics(testrows,analysis_seed):
    def values(arm,condition,key):
        by_seed={r['seed']:r[key] for r in testrows if r['arm']==arm and r['condition']==condition}
        check(set(by_seed)==set(range(5)),'all five paired blocks per condition')
        return np.asarray([by_seed[s] for s in range(5)],dtype=float)
    cells=[];intervals={};interpretation={}
    for arm in ('DEND','POINT'):
        for condition in ('intact','retrained','cut'):
            rows=sorted([r for r in testrows if (r['arm'],r['condition'])==(arm,condition)],key=lambda r:r['seed'])
            k=int(values(arm,condition,'failed').sum())
            cells.append(dict(arm=arm,condition=condition,n=5,
                mean_accuracy=float(values(arm,condition,'accuracy').mean()),mean_balanced_accuracy=float(values(arm,condition,'balanced_accuracy').mean()),
                mean_coverage=float(values(arm,condition,'coverage').mean()),failures=k,failure_wilson95=wilson(k),
                fully_silent=int(values(arm,condition,'fully_silent').sum()),fully_abstaining=int(values(arm,condition,'fully_abstaining').sum()),
                mean_total_spikes=float(values(arm,condition,'total_spike_events_mean').mean()),mean_state_l1_change=float(values(arm,condition,'state_l1_change_mean').mean()),
                correct_settled_fraction_by_step=np.asarray([r['correct_settled_fraction_by_step'] for r in rows]).mean(0).tolist()))
        for metric in ('accuracy','failed'):
            intact=values(arm,'intact',metric)
            for condition in ('retrained','cut'):
                intervals[f'{arm}_intact_minus_{condition}_{metric}']=interval_record(intact-values(arm,condition,metric),analysis_seed)
        intact=values(arm,'intact','failed')==0;retrained=values(arm,'retrained','failed')==0
        interpretation[arm]=dict(intact_passes=int(intact.sum()),retrained_passes=int(retrained.sum()),all_intact_pass=bool(intact.all()),all_retrained_pass=bool(retrained.all()),
            observed_no_cross_success=bool(retrained.any()),limits='No-cross retains multiple cells, self recurrence and state. Success refutes necessity for passing this finite task/gate; failure only shows this recipe/budget did not learn reliably. No physical impossibility or single-cell conclusion.')
    return dict(primary_family=[a+'_intact_minus_retrained_accuracy' for a in ('DEND','POINT')],cells=cells,intervals=intervals,interpretation=interpretation)


def audit_tests(out,lock,registry):
    summary=read_json(out/'TEST_SUMMARY.json')
    check(summary['train_lock_sha256']==sha(out/'TRAIN_LOCK.json'),'test summary checkpoint lock')
    check(read_json(out/'TEST_STARTED.json')['train_lock_sha256']==sha(out/'TRAIN_LOCK.json'),'test start lock')
    for name,digest in summary['hashes'].items():check(sha(out/name)==digest,'test artifact hash '+name)
    rows=summary['rows'];sources=lock['rows']+lock['cut_rows'];source_index={r['id']:r for r in sources}
    check(rows==read_json(out/'TEST_LOG.json'),'full test log retained')
    check(len(rows)==len({r['id'] for r in rows})==30,'30 unique final evaluations')
    wanted={(s,a,c) for s in range(5) for a in ('DEND','POINT') for c in ('intact','retrained','cut')}
    check({(r['seed'],r['arm'],r['condition']) for r in rows}==wanted,'all six conditions at all5seeds')
    check({r['id'] for r in rows}==set(source_index),'no missing or selected test condition')
    datasets={f'test_s{s}.npz':inspect_dataset(out,f'test_s{s}.npz',512,2090010000+10*s) for s in range(5)}
    max_residual=0.0
    for row in rows:
        source=source_index[row['id']]
        for key in ('seed','arm','condition'):check(row[key]==source[key],'test/source '+key)
        check(row['source_file']==source['file'],'exact pretest source weight')
        check(row['data_file']==f"test_s{row['seed']}.npz",'paired test data')
        d=datasets[row['data_file']];theta=np.load(out/source['file'],allow_pickle=False);native=replay(d['x'],theta,row['arm'])
        with np.load(out/row['file'],allow_pickle=False) as saved:
            check(set(saved.files)=={'counts','prediction','spikes','states','candidates'},'full test native trace')
            max_residual=max(max_residual,verify_native(saved,native,'held-out test'))
        check_description(row,native,d['y'],'test native metrics/activity/latency')
    check(set(summary['hashes'])==set(datasets)|{r['file'] for r in rows},'exact test dataset/native inventory')
    check(summary['unique_test_histories']==2560 and summary['model_test_episodes']==15360,'unique histories and repeated model evaluations separated')
    expected=independent_statistics(rows,registry['analysis_seed']);actual=summary['statistics']
    compare({key:actual[key] for key in expected},expected,'all independent seed-unit statistics',atol=1e-12)
    check(len(actual['cells'])==6 and len(actual['intervals'])==8,'six cells/eight pointwise exploratory intervals')
    return datasets,dict(test_models=30,test_model_episodes=15360,unique_test_histories=2560,max_test_state_candidate_residual=max_residual,
        paired_seed_intervals=8,wilson_failure_intervals=6,latency_curve='retrospective settled prefix counts with all-episode denominators; no stopping rule')


def assert_disjoint(datasets):
    seen=set()
    for name,d in datasets.items():
        for x in d['x']:
            digest=hashlib.sha256(np.ascontiguousarray(x).tobytes()).digest()
            check(digest not in seen,'no exact cross/within-split duplicate '+name);seen.add(digest)
    return len(seen)


def main():
    start=time.perf_counter();out=R/'results';freeze=read_json(R/'FREEZE.json')
    for name,digest in freeze['sha256'].items():check(sha(R/name)==digest,'frozen source '+name)
    for name,digest in freeze.get('dependencies',{}).items():check(sha(R.parent/name)==digest,'frozen dependency '+name)
    for name in ('model.py','data.py'):check(sha(R/name)==sha(R.parent/'dendritic-population-v8'/name),'native model/task byte-identical toV8 '+name)
    registry=read_json(R/'SEEDS.json');expected=[dict(seed=s,train_seed=2090000000+10*s,test_seed=2090010000+10*s,initial_seed=2090020000+s) for s in range(5)]
    check(registry['n_pairs']==5 and registry['pairs']==expected,'five prespecified fresh paired seeds')
    check(registry['analysis_seed']==2090030001,'fixed analysis RNG')
    check(registry['checkpoint_updates']==[0,1,5,10,20,50,100,150,200],'fixed9checkpoint schedule')
    check(registry['sampled_updates']==[0,99,199],'fixed sampled derivatives')
    seed_values=[v for row in expected for k,v in row.items() if k!='seed']+[registry['analysis_seed']]
    check(len(set(seed_values))==16,'disjoint train/test/initial/analysis seed values')
    lock=read_json(out/'TRAIN_LOCK.json');check(lock['freeze_sha256']==sha(R/'FREEZE.json'),'train source lock')
    check(lock['test_generated'] is False,'train lock predates test')
    marker=read_json(out/'TRAIN_STARTED.json')
    check(marker['freeze_sha256']==sha(R/'FREEZE.json') and marker['seed_registry_sha256']==sha(R/'SEEDS.json'),'start immutable source/seed identities')
    for name,digest in lock['files'].items():check(sha(out/name)==digest,'locked training/cut artifact '+name)
    check(lock['rows']==read_json(out/'TRAIN_LOG.json'),'complete training rows locked')
    training_data,training=audit_training(out,lock,registry);cuts=audit_cut_weights(out,lock)
    test_data,tests=audit_tests(out,lock,registry);distinct=assert_disjoint({**training_data,**test_data})
    check(distinct==3200,'five unique128train/512test split pairs')
    result=dict(status='PASS',checks=checks,training=training,inference_cuts=cuts,tests=tests,distinct_training_test_histories=distinct,seconds=time.perf_counter()-start,
        scope='Separate RNG, initialization, scalar native equations, all count metrics, activity/state-change/retrospective settling metrics, masked Adam reconstruction, cut provenance, paired bootstrap and Wilson implementation. All fixed coordinates/gradients/moments remain zero. Full final train/test native traces match; saved checkpoint metrics and count histories match. No all-update native replay or independent rederivation of all proxy gradients. Internal workflow checks are not external replication; finite optimization failure does not establish physical necessity.')
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
