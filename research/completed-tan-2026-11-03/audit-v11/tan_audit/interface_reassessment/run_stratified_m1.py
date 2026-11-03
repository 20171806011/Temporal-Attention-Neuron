"""Diagnostic replay of unchanged M1, with prespecified input-information strata."""
from pathlib import Path
import sys,json,hashlib,time,platform,resource,itertools
import numpy as np
OUT=Path(__file__).resolve().parent;REPO=OUT.parent.parent/'Temporal-Attention-Neuron';P1B=REPO/'p1_experiments/p1b'
sys.path.insert(0,str(P1B))
import models_p1b_v2 as m
import runner_p1b as runner
PLAN='8e0f71c73850097c6373ff3c1d312e5a2b74e4448f669cd0f0b185ca24213c68'
AMENDMENT='39334648f9260fd86796c05d8e6ba10e44a589ae7a43e545fe02e6e5178e3d80'
MODEL='6b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def waveform_oracle(content,novelty,q):
    """Offline full-event-waveform observer; never called by the native carrier."""
    positive=[i for i,x in enumerate(novelty) if x>0]
    matches=[i for i in positive if novelty[i]+1.4==q]
    if len(matches)==1:
        i=matches[0];return content[i]-(novelty[i]+1.4),'POSITIVE_KEY_INVERSE'
    assert not matches
    zero=[i for i,x in enumerate(novelty) if x==0]
    assert len(zero)==1,'Oracle only authorized on zero_count<=1 histories'
    return content[zero[0]]-q,'UNIQUE_ZERO_EVENT_BY_VALID_QUERY'

def classify(trial):
    active=trial.keys[trial.event_positions];pattern=''.join('1' if x>1.4 else '0' for x in active);z=pattern.count('0')
    partition='ALL_POSITIVE' if z==0 else 'EXACTLY_ONE_ZERO' if z==1 else 'AT_LEAST_TWO_ZERO'
    return pattern,z,partition

def metric(rows,total):
    if not rows:return dict(n=0,fraction_of_full_N=0.,joint_accuracy=None,role_mae=None,role_rmse=None,mean_ecomp=None)
    err=abs(np.array([x['prediction'] for x in rows])-np.array([x['target'] for x in rows]));succ=np.all(err<.5,axis=1)
    return dict(n=len(rows),fraction_of_full_N=len(rows)/total,joint_count=int(succ.sum()),joint_accuracy=float(succ.mean()),role_mae=err.mean(0).tolist(),role_rmse=np.sqrt(np.mean(err*err,axis=0)).tolist(),mean_ecomp=float(err.max(1).mean()),max_error=float(err.max()))

def main():
    start=time.perf_counter();assert digest(OUT/'ANALYSIS_PLAN.md')==PLAN and digest(OUT/'PLAN_AMENDMENT_1.md')==AMENDMENT and digest(P1B/'models_p1b_v2.py')==MODEL
    frozen=json.loads((P1B/'summary_p1b.json').read_text());cal=frozen['calibration'];trials_by_N=dict(zip((2,3),runner.generate_confirmatory_datasets()))
    inputs=[]
    for N,trials in trials_by_N.items():
        for t in trials:
            pattern,z,partition=classify(t)
            inputs.append(dict(id=f'N{N}_{t.trial_id}',N=N,trial_id=t.trial_id,seed_metadata=t.seed,keys=t.keys.tolist(),values=t.values.tolist(),mask=t.mask.tolist(),positions=t.event_positions,
                queries=list(t.query_cues),target_indices=list(t.target_idx),targets=list(t.target_values),gt_add=t.gt_add,gt_sub=t.gt_sub,rejection_count=t.rejection_count,
                novelty_pattern=pattern,zero_count=z,partition=partition))
    ip=OUT/'stratified_inputs.jsonl';ip.write_text(''.join(json.dumps(x)+'\n' for x in inputs));ih=digest(ip);(OUT/'STRATIFIED_INPUTS_SHA256').write_text(ih+'  stratified_inputs.jsonl\n');print('saved inputs before metrics',ih,flush=True)
    rows=[];checkpoint_checks=0;runner_checks=0;max_runner_residual=0.;oracle_max_error=0.;oracle_histories=0
    for N,trials in trials_by_N.items():
        for t in trials:
            pattern,z,partition=classify(t)
            # The unchanged registered model sees only its prescribed wrapper inputs.
            state,_,_=m.simulate_encoding(t.keys,t.values,t.mask,'M1',A_0=cal['A_0'])
            state_copy=state.to_array().copy();opred=None;omethod=None
            if z<=1:
                current=[float(t.keys[p]+t.values[p]) for p in t.event_positions]
                novelty=[max(0.,abs(float(t.keys[p]))-1.4) for p in t.event_positions]
                answer=[waveform_oracle(current,novelty,float(q)) for q in t.query_cues]
                opred=[a[0] for a in answer];omethod=[a[1] for a in answer];oracle_histories+=1
                oracle_max_error=max(oracle_max_error,float(max(abs(np.array(opred)-t.target_values))));assert max(abs(np.array(opred)-t.target_values))<1e-12
            for delay in (0.,5.,10.):
                cue=m.simulate_delay(state,delay,'M1',A_0=cal['A_0']);cue_copy=cue.to_array().copy()
                res=[m.simulate_cue_and_readout(cue,float(q),'M1',g_dec=cal['g_dec'],b_dec=cal['b_dec'],A_0=cal['A_0']) for q in t.query_cues]
                assert np.array_equal(state_copy,state.to_array()) and np.array_equal(cue_copy,cue.to_array());checkpoint_checks+=1
                prediction=[x['v_hat_analog'] for x in res];target=list(t.target_values);err=abs(np.array(prediction)-target);e_comp=float(err.max())
                if t.trial_id<8:
                    ref=runner.evaluate_episode(t,delay,'M1',cal['g_dec'],cal['b_dec'],cal['A_0']);rr=max(abs(ref['v_hat_A']-prediction[0]),abs(ref['v_hat_B']-prediction[1]),abs(ref['e_comp']-e_comp));max_runner_residual=max(max_runner_residual,rr);assert rr<1e-12;runner_checks+=1
                rows.append(dict(input_id=f'N{N}_{t.trial_id}',N=N,trial_id=t.trial_id,delay_ms=delay,novelty_pattern=pattern,zero_count=z,partition=partition,sufficient_waveform_query_identifiable=z<=1,
                    queries=list(t.query_cues),target=target,prediction=prediction,absolute_errors=err.tolist(),joint_success=bool(np.all(err<.5)),e_comp=e_comp,
                    encoding_state=state_copy.tolist(),cue_state=cue_copy.tolist(),u_readout=[x['u_readout'] for x in res],r_deadline_minus=[x['r_deadline_minus'] for x in res],
                    offline_waveform_oracle_prediction=opred,offline_waveform_oracle_method=omethod,oracle_is_model_access=False))
        print('completed N',N,'histories',len(trials),flush=True)
    assert len(inputs)==4000 and len(rows)==12000
    (OUT/'stratified_predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    cells=[];whole_comparison=[]
    for N,delay in itertools.product((2,3),(0.,5.,10.)):
        cellrows=[r for r in rows if r['N']==N and r['delay_ms']==delay]
        total=len(cellrows);whole=metric(cellrows,total)
        saved=frozen['cell_results'][f'N{N}_delay{int(delay)}ms']['model_summaries']['M1']
        assert whole['joint_accuracy']==saved['joint_delivery_acc'];assert abs(whole['mean_ecomp']-saved['mean_e_comp'])<1e-12
        whole_comparison.append(dict(N=N,delay_ms=delay,joint_accuracy=whole['joint_accuracy'],saved_joint_accuracy=saved['joint_delivery_acc'],mean_ecomp=whole['mean_ecomp'],saved_mean_ecomp=saved['mean_e_comp'],mean_ecomp_difference=whole['mean_ecomp']-saved['mean_e_comp']))
        cells.append(dict(N=N,delay_ms=delay,group_kind='WHOLE',group='ALL',**whole))
        for bits in itertools.product('01',repeat=N):
            pattern=''.join(bits);rr=[r for r in cellrows if r['novelty_pattern']==pattern];cells.append(dict(N=N,delay_ms=delay,group_kind='PATTERN',group=pattern,**metric(rr,total)))
        for label in ('ALL_POSITIVE','EXACTLY_ONE_ZERO','AT_LEAST_TWO_ZERO'):
            rr=[r for r in cellrows if r['partition']==label];cells.append(dict(N=N,delay_ms=delay,group_kind='PARTITION',group=label,**metric(rr,total)))
        rr=[r for r in cellrows if r['zero_count']<=1];cells.append(dict(N=N,delay_ms=delay,group_kind='SUFFICIENT_INFORMATION',group='AT_MOST_ONE_ZERO',**metric(rr,total)))
    result=dict(analysis_plan_sha256=PLAN,amendment_sha256=AMENDMENT,script_sha256=digest(Path(__file__)),inputs_sha256=ih,model_sha256=MODEL,runner_sha256=digest(P1B/'runner_p1b.py'),generator_sha256=digest(REPO/'p1_experiments/task_generator.py'),frozen_summary_sha256=digest(P1B/'summary_p1b.json'),
        calibration=cal,python=platform.python_version(),numpy=np.__version__,histories=4000,paired_episode_rows=12000,role_outputs=24000,cells=cells,whole_cell_frozen_comparison=whole_comparison,
        validation=dict(checkpoints_unchanged=checkpoint_checks,runner_episode_checks=runner_checks,max_runner_episode_residual=max_runner_residual,oracle_unique_histories=oracle_histories,oracle_max_abs_error=oracle_max_error),
        wall_seconds=time.perf_counter()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Descriptive frozen-contract M1 reanalysis; original aggregate known in advance; no fitting, new model access, subgroup pass labels or causal attribution. Historical trial bytes unavailable; replay conditional on frozen generator and installed NumPy.')
    (OUT/'stratified_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
