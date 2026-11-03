"""One fixed history-split external probe family on unchanged native M1."""
from pathlib import Path
import sys,json,hashlib,time,itertools,platform,resource
import numpy as np
OUT=Path(__file__).resolve().parent;REPO=OUT.parent.parent/'Temporal-Attention-Neuron';P1B=REPO/'p1_experiments/p1b'
sys.path.insert(0,str(P1B))
import models_p1b_v2 as m
import runner_p1b as runner
from task_generator import TaskGenerator
PLAN='8ffcda523dfc042f9d64473c361ecc39c6f69b1f9d2b2af7994bbd35dea6c1ff'
MODEL='6b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f'
LEARNED=['QUERY_ONLY','NATIVE_OUTPUT_Q','CUE_U_Q','CUE_U_A_Q','SHUFFLED_STATE_Q']
ARMS=['FROZEN_NATIVE',*LEARNED,'ZERO','CORRECT_HISTORY_ORACLE']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def savejson(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def exponents(d):
    return np.array([x for total in range(5) for x in itertools.product(range(total+1),repeat=d) if sum(x)==total],dtype=int)

def design(raw,train):
    mean=raw[train].mean(0);scale=raw[train].std(0);scale[scale==0]=1.
    x=(raw-mean)/scale;exp=exponents(raw.shape[1]);poly=np.prod(x[:,None,:]**exp[None,:,:],axis=2)
    pm=poly[train].mean(0);ps=poly[train].std(0);ps[ps==0]=1.;pm[0]=0.;ps[0]=1.
    Z=(poly-pm)/ps;assert np.all(Z[:,0]==1.)
    return Z,dict(raw_mean=mean.tolist(),raw_scale=scale.tolist(),exponents=exp.tolist(),expanded_mean=pm.tolist(),expanded_scale=ps.tolist())

def metrics(pair_rows,arm):
    if not pair_rows:return dict(n_histories=0,n_pairs=0)
    pred=np.array([r['predictions'][arm] for r in pair_rows]);raw=np.array([r['raw_predictions'][arm] for r in pair_rows]);y=np.array([r['target'] for r in pair_rows]);err=abs(pred-y);re=abs(raw-y)
    return dict(n_histories=len(set(r['input_id'] for r in pair_rows)),n_pairs=len(pair_rows),mean_ecomp=float(err.max(1).mean()),mae_per_role=err.mean(0).tolist(),rmse_per_role=np.sqrt((err*err).mean(0)).tolist(),
        max_error=float(err.max()),strict_0_5_joint=float(np.all(err<.5,axis=1).mean()),raw_mean_ecomp=float(re.max(1).mean()),raw_max_error=float(re.max()),clip_count_per_role=np.sum(raw!=pred,axis=0).tolist(),clip_pair_count=int(np.any(raw!=pred,axis=1).sum()))

def main():
    started=time.perf_counter();assert sha(OUT/'PREREGISTRATION.md')==PLAN and sha(P1B/'models_p1b_v2.py')==MODEL
    cal=json.loads((P1B/'summary_p1b.json').read_text())['calibration'];gen=TaskGenerator();splitrng=np.random.RandomState(2026110222)
    inputs=[];trialmap={};splits={}
    for N,seed in ((2,2026110220),(3,2026110221)):
        rng=np.random.RandomState(seed);trials=[gen.generate_single_trial(rng,i,seed+i,N=N) for i in range(1792)]
        order=splitrng.permutation(1792);labels={int(i):label for label,ids in [('train',order[:1024]),('validation',order[1024:1280]),('test',order[1280:])] for i in ids}
        for t in trials:
            hid=f'N{N}_{t.trial_id}';trialmap[hid]=t;splits[hid]=labels[t.trial_id]
            inputs.append(dict(id=hid,N=N,trial_id=t.trial_id,seed_metadata=t.seed,split=labels[t.trial_id],keys=t.keys.tolist(),values=t.values.tolist(),mask=t.mask.tolist(),positions=t.event_positions,
                queries=list(t.query_cues),targets=list(t.target_values),target_indices=list(t.target_idx),rejection_count=t.rejection_count,all_positive=bool(np.all(t.keys[t.event_positions]>1.4))))
    ip=OUT/'inputs.jsonl';ip.write_text(''.join(json.dumps(r)+'\n' for r in inputs));ih=sha(ip);(OUT/'INPUTS_SHA256').write_text(ih+'  inputs.jsonl\n');print('saved inputs before model execution',ih,flush=True)
    shufflerng=np.random.RandomState(2026110223);donors={}
    for N,split in itertools.product((2,3),('train','validation','test')):
        ids=[r['id'] for r in inputs if r['N']==N and r['split']==split];order=shufflerng.permutation(len(ids))
        for j,idx in enumerate(order):donors[ids[int(idx)]]=ids[int(order[(j+1)%len(order)])]
    assert all(k!=v and splits[k]==splits[v] for k,v in donors.items());savejson(OUT/'split_and_donor_ledger.json',dict(splits=splits,donors=donors,seeds=dict(N2=2026110220,N3=2026110221,split=2026110222,shuffle=2026110223)))
    records=[];states={};selected={N:[r['id'] for r in inputs if r['N']==N and r['split']=='test'][:8] for N in (2,3)};structural=[];runner_residual=0.;runner_count=0
    for rec in inputs:
        t=trialmap[rec['id']];state,_,_=m.simulate_encoding(t.keys,t.values,t.mask,'M1',A_0=cal['A_0']);original=state.to_array().copy()
        for delay in (0.,5.,10.):
            cue=m.simulate_delay(state,delay,'M1',A_0=cal['A_0']);snapshot=cue.to_array().copy();assert cue.r==0.
            states[(rec['id'],delay)]=snapshot.tolist();pair=[]
            for role,q in enumerate(t.query_cues):
                res=m.simulate_cue_and_readout(cue,float(q),'M1',**cal);pair.append(res['v_hat_analog'])
                assert np.array_equal(snapshot,cue.to_array()) and np.array_equal(original,state.to_array())
                best=min(t.event_positions,key=lambda p:(float(q)-float(t.keys[p]))**2);oracle=float(t.values[best]);assert oracle==float(t.target_values[role])
                record=dict(row_id=len(records),input_id=rec['id'],N=rec['N'],trial_id=t.trial_id,split=rec['split'],all_positive=rec['all_positive'],delay_ms=delay,role=role,q=float(q),target=float(t.target_values[role]),
                    u_cue=float(cue.u),A_cue=float(cue.A),r_cue=float(cue.r),u_readout=res['u_readout'],native_prediction=res['v_hat_analog'],r_deadline_minus=res['r_deadline_minus'],spike_count=res['spike_count'],oracle_prediction=oracle)
                records.append(record)
                if rec['id'] in selected[rec['N']]:
                    variants=[m.simulate_cue_and_readout(m.CarrierState(cue.u,A,cue.r),float(q),'M1',**cal) for A in (0.,5.)]
                    assert variants[0]==variants[1]==res
                    structural.append(dict(input_id=rec['id'],delay_ms=delay,role=role,q=float(q),u=cue.u,r=cue.r,original_A=cue.A,interventions=[0.,5.],all_native_results_exactly_equal=True,u_readout=res['u_readout'],raw_spikes=res['raw_spikes']))
            if t.trial_id<4:
                reference=runner.evaluate_episode(t,delay,'M1',**cal);delta=max(abs(reference['v_hat_A']-pair[0]),abs(reference['v_hat_B']-pair[1]));runner_residual=max(runner_residual,delta);assert delta==0.;runner_count+=1
    assert len(records)==21504 and len(structural)==96
    (OUT/'native_observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));savejson(OUT/'A_invisibility_checks.json',dict(checkpoints=len(structural),intervention_windows=2*len(structural),cases=structural,scope='Within-bounds A intervention; only structural query invisibility, not useful-information proof or reachable-encoding claim.'))
    # Feature assembly uses only declared observables. N/split/delay serve grouping/indexing, not features.
    features={
        'QUERY_ONLY':np.array([[r['q']] for r in records]),
        'NATIVE_OUTPUT_Q':np.array([[r['u_readout'],r['q']] for r in records]),
        'CUE_U_Q':np.array([[r['u_cue'],r['q']] for r in records]),
        'CUE_U_A_Q':np.array([[r['u_cue'],r['A_cue'],r['q']] for r in records]),
        'SHUFFLED_STATE_Q':np.array([[*states[(donors[r['input_id']],r['delay_ms'])][:2],r['q']] for r in records])}
    np.savez_compressed(OUT/'allowed_features.npz',**features)
    train=np.array([r['split']=='train' for r in records]);targets=np.array([r['target'] for r in records]);train_ids=np.where(train)[0]
    assert train.sum()==12288
    model_artifacts={};designs={}
    for view in LEARNED:
        Z,transform=design(features[view],train);designs[view]=Z;Xt=Z[train];yt=targets[train];penalty=np.eye(Z.shape[1])*.001;penalty[0,0]=0.
        gram=Xt.T@Xt/len(Xt)+penalty;rhs=Xt.T@yt/len(Xt);beta=np.linalg.solve(gram,rhs)
        assert np.max(abs(gram@beta-rhs))<1e-10
        model_artifacts[view]=dict(**transform,coefficient=beta.tolist(),degree=4,ridge_lambda=.001,objective='mean squared training error plus lambda times nonintercept coefficient square sum',n_terms=Z.shape[1],training_rows=len(Xt),
            training_row_ids_sha256=hashlib.sha256(train_ids.astype('<i8').tobytes()).hexdigest(),regularized_gram_condition=float(np.linalg.cond(gram)),normal_equation_max_residual=float(np.max(abs(gram@beta-rhs))))
    fitfile=OUT/'frozen_models.json';savejson(fitfile,dict(input_hash=ih,features_sha256=sha(OUT/'allowed_features.npz'),models=model_artifacts,output_clip=[-3.,3.],validation_or_test_used_to_fit=False));fit_hash=sha(fitfile)
    print('fitted models frozen before validation/test predictions',fit_hash,flush=True)
    raw={view:designs[view]@np.array(model_artifacts[view]['coefficient']) for view in LEARNED}
    raw.update(FROZEN_NATIVE=np.array([r['native_prediction'] for r in records]),ZERO=np.zeros(len(records)),CORRECT_HISTORY_ORACLE=np.array([r['oracle_prediction'] for r in records]))
    pred={k:np.clip(v,-3.,3.) if k in LEARNED else v.copy() for k,v in raw.items()}
    paired=[]
    for i in range(0,len(records),2):
        a,b=records[i:i+2];assert a['input_id']==b['input_id'] and a['delay_ms']==b['delay_ms'] and a['role']==0 and b['role']==1
        paired.append(dict(input_id=a['input_id'],N=a['N'],split=a['split'],all_positive=a['all_positive'],delay_ms=a['delay_ms'],queries=[a['q'],b['q']],target=[a['target'],b['target']],
            shuffled_state_donor=donors[a['input_id']],predictions={arm:pred[arm][i:i+2].tolist() for arm in ARMS},raw_predictions={arm:raw[arm][i:i+2].tolist() for arm in ARMS}))
    (OUT/'predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in paired));assert len(paired)==10752
    cells=[]
    for split,stratum,N,delay in itertools.product(('train','validation','test'),('full','all_positive'),('pooled',2,3),('pooled',0.,5.,10.)):
        rr=[r for r in paired if r['split']==split and (stratum=='full' or r['all_positive']) and (N=='pooled' or r['N']==N) and (delay=='pooled' or r['delay_ms']==delay)]
        for arm in ARMS:cells.append(dict(split=split,stratum=stratum,N=N,delay_ms=delay,arm=arm,**metrics(rr,arm)))
    comparisons=[('CUE_U_A_Q','FROZEN_NATIVE'),('NATIVE_OUTPUT_Q','FROZEN_NATIVE'),('CUE_U_Q','NATIVE_OUTPUT_Q'),('CUE_U_A_Q','CUE_U_Q'),('SHUFFLED_STATE_Q','CUE_U_A_Q'),('QUERY_ONLY','CUE_U_A_Q')]
    brng=np.random.RandomState(2026110224);contrasts=[];history_losses=[]
    for rec in inputs:
        if rec['split']!='test':continue
        rr=[r for r in paired if r['input_id']==rec['id']]
        history_losses.append(dict(input_id=rec['id'],N=rec['N'],all_positive=rec['all_positive'],losses={arm:float(np.mean([max(abs(np.array(r['predictions'][arm])-r['target'])) for r in rr])) for arm in ARMS}))
    savejson(OUT/'test_history_losses.json',history_losses)
    for stratum in ('full','all_positive'):
        groups={N:[r for r in history_losses if r['N']==N and (stratum=='full' or r['all_positive'])] for N in (2,3)}
        indices={N:brng.randint(0,len(rows),size=(2000,len(rows))) for N,rows in groups.items()};total=sum(len(x) for x in groups.values())
        for left,right in comparisons:
            values={N:np.array([r['losses'][left]-r['losses'][right] for r in rr]) for N,rr in groups.items()}
            boots=sum(values[N][indices[N]].sum(1) for N in (2,3))/total;mean=sum(x.sum() for x in values.values())/total
            contrasts.append(dict(stratum=stratum,left=left,right=right,n_histories=total,counts_by_N={str(N):len(groups[N]) for N in groups},mean_ecomp_difference=float(mean),percentile95_interval=np.quantile(boots,[.025,.975]).tolist(),replicates=2000,scope='paired history bootstrap, delays averaged within history; descriptive, no multiple-comparison adjustment'))
    result=dict(preregistration_sha256=PLAN,script_sha256=sha(Path(__file__)),inputs_sha256=ih,frozen_models_sha256=fit_hash,model_source_sha256=MODEL,calibration=cal,python=platform.python_version(),numpy=np.__version__,
        histories=len(inputs),query_rows=len(records),paired_rows=len(paired),arm_count=len(ARMS),split_histories={s:sum(r['split']==s for r in inputs) for s in ('train','validation','test')},
        split_query_rows={s:sum(r['split']==s for r in records) for s in ('train','validation','test')},cells=cells,paired_test_contrasts=contrasts,
        validation=dict(r_cue_zero_for_all=True,checkpoints_unmutated=True,runner_checks=runner_count,max_runner_residual=runner_residual,A_invisibility_checkpoints=len(structural),no_shuffle_self_donors=True,model_features_exclude_keys_values_positions_N_delay_ids=True,validation_test_not_used_for_fitting=True),
        wall_seconds=time.perf_counter()-started,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='External supervised diagnostic with fixed polynomial degree4 and no selection; different stage features/capacities, direct q, and bounded output are extra access/compute. Decoder failure does not imply absence; original labels unchanged.')
    assert sha(P1B/'models_p1b_v2.py')==MODEL;savejson(OUT/'results.json',result)
    print(json.dumps({'test_pooled_delay_cells':[c for c in cells if c['split']=='test' and c['N']=='pooled' and c['delay_ms']=='pooled'],'contrasts':contrasts,'wall_seconds':result['wall_seconds']},indent=2))

if __name__=='__main__':main()
