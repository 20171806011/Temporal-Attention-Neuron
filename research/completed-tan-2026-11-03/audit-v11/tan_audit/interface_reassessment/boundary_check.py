"""Deterministic audit of the exact registered P1B input/transition boundary."""
from pathlib import Path
from fractions import Fraction as Q
import sys, json, hashlib, inspect, time
import numpy as np

OUT=Path(__file__).resolve().parent
REPO=OUT.parent.parent/'Temporal-Attention-Neuron'
P1B=REPO/'p1_experiments/p1b'
sys.path.insert(0,str(REPO/'p1_experiments'))
sys.path.insert(0,str(P1B))
import models_p1b_v2 as m
import runner_p1b as runner
from task_generator import TrialData

PLAN='8e0f71c73850097c6373ff3c1d312e5a2b74e4448f669cd0f0b185ca24213c68'
AMENDMENT='39334648f9260fd86796c05d8e6ba10e44a589ae7a43e545fe02e6e5178e3d80'
MODEL='6b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f'
PHYSICAL=('M1','M2a','M2b','M2c','M2d')
PAIRS=(('low_A','low_B'),('primed_A','primed_B'),('positive_A','positive_B'))

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def exact_fixtures():
    specs=[
        ('low_A',[0,4],['.8','1'],['1','-1'],['.8','1']),
        ('low_B',[0,4],['1','.8'],['.8','-.8'],['.8','1']),
        ('primed_A',[0,2,4],['2','.8','1'],['0','1.5','-.5'],['.8','1']),
        ('primed_B',[0,2,4],['2','1','.8'],['0','1.3','-.3'],['.8','1']),
        ('positive_A',[0,4],['1.6','2'],['1','-1'],['1.6','2']),
        ('positive_B',[0,4],['2','1.6'],['.6','-.6'],['1.6','2'])]
    result=[]
    for label,positions,ks,vs,qs in specs:
        keys=list(map(Q,ks));vals=list(map(Q,vs));queries=list(map(Q,qs))
        assert 2<=len(keys)<=3 and len(keys)==len(set(keys))
        assert all(Q(3,10)<=k<Q(5,2) for k in keys)
        assert all(-3<=v<3 for v in vals)
        assert min(abs(x-y) for i,x in enumerate(keys) for y in keys[i+1:])>=Q(1,20)
        targets=[vals[keys.index(q)] for q in queries]
        segments=[]
        for slot in range(5):
            active=slot in positions
            i=positions.index(slot) if active else None
            content=keys[i]+vals[i] if active else Q(0)
            novelty=max(Q(0),keys[i]-Q(7,5)) if active else Q(0)
            segments.append(dict(slot=slot,start_ms=20*slot,duration_ms=10,steps=100,phase='presentation',mask=active,content=str(content),novelty=str(novelty),query_current='0'))
            segments.append(dict(slot=slot,start_ms=20*slot+10,duration_ms=10,steps=100,phase='ISI',mask=active,content='0',novelty='0',query_current='0'))
        result.append(dict(id=label,N=len(keys),positions=positions,keys=[str(x) for x in keys],values=[str(x) for x in vals],queries=[str(x) for x in queries],targets=[str(x) for x in targets],encoding_segments=segments,
            delay_segments={str(d):dict(duration_ms=d,steps=10*d,content='0',novelty='0',query_current='0') for d in (0,5,10)},
            independent_query_segments=[dict(role=i,duration_ms=10,steps=100,content='0',novelty='0',query_current=str(q)) for i,q in enumerate(queries)]))
    return result

def trial_from_exact(rec,idx):
    keys=np.zeros(5);values=np.zeros(5);mask=np.zeros(5,dtype=bool)
    keys[rec['positions']]=[float(Q(x)) for x in rec['keys']]
    values[rec['positions']]=[float(Q(x)) for x in rec['values']]
    mask[rec['positions']]=True;queries=tuple(float(Q(x)) for x in rec['queries'])
    target_idx=tuple(next(p for p in rec['positions'] if keys[p]==q) for q in queries)
    target_values=tuple(float(values[p]) for p in target_idx)
    sorted_keys=sorted(keys[mask]);mid=sorted_keys[1] if len(sorted_keys)==3 else None
    flags=tuple(bool(mid is not None and q==mid) for q in queries)
    assert np.all(keys[mask]>=.3) and np.all(keys[mask]<2.5)
    assert np.all(values[mask]>=-3) and np.all(values[mask]<3)
    assert min(np.diff(sorted_keys))>=.05
    return TrialData(trial_id=idx,seed=0,window_size=5,N=rec['N'],event_positions=rec['positions'],keys=keys,values=values,mask=mask,
        target_idx=target_idx,target_keys=queries,target_values=target_values,query_cues=queries,gt_add=sum(target_values),gt_sub=target_values[0]-target_values[1],
        t1_is_middle=flags[0],t2_is_middle=flags[1],is_middle_target=any(flags),rejection_count=0)

def main():
    started=time.perf_counter()
    assert digest(OUT/'ANALYSIS_PLAN.md')==PLAN and digest(OUT/'PLAN_AMENDMENT_1.md')==AMENDMENT
    source_paths=[P1B/'models_p1b_v2.py',P1B/'runner_p1b.py',REPO/'p1_experiments/task_generator.py',P1B/'P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md',P1B/'summary_p1b.json']
    before={str(p.relative_to(REPO)):digest(p) for p in source_paths};assert digest(source_paths[0])==MODEL
    calibration=json.loads((P1B/'summary_p1b.json').read_text())['calibration']
    exact=exact_fixtures();lookup={r['id']:r for r in exact}
    pair_proofs=[]
    for a,b in PAIRS:
        x,y=lookup[a],lookup[b]
        same_full=x['encoding_segments']==y['encoding_segments']
        same_content=all(p['content']==q['content'] for p,q in zip(x['encoding_segments'],y['encoding_segments']))
        same_queries=x['queries']==y['queries'];gap=max(abs(Q(p)-Q(q)) for p,q in zip(x['targets'],y['targets']))
        expected=a!='positive_A';assert same_full==expected and same_content and same_queries
        pair_proofs.append(dict(pair=[a,b],same_content_only=same_content,same_complete_encoding_ports_masks_timing=same_full,same_query_paths=same_queries,
            target_linf_gap=str(gap),minimax_and_equal_mixture_mean_ecomp_lower_bound=str(gap/2) if same_full else None,
            classification='FULL_REGISTERED_DRIVE_COLLISION' if same_full else 'PROJECTED_CONTENT_EQUALITY_ONLY'))
    b=Q(149,150);primed_exact=[Q(3,5)*(1-b**100)*b**power for power in (300,700)]
    assert all(x>0 for x in primed_exact)
    proof=dict(fixtures=exact,pair_proofs=pair_proofs,primed_M1_exact_gate_at_low_onsets=[str(x) for x in primed_exact],
        positive_branch_inverse=dict(key='novelty+7/5',value='content-key',condition='registered positive keys and novelty>0'),
        binary64_qualification=dict(positive_novelty_key_subtraction_exact_by_sterbenz=True,max_value_ambiguity_for_same_rounded_content=2**-50,
            scope='ULP-scale value ambiguity only; not a 0.5-tolerance obstruction'),
        query_aware_sufficient_condition='At most one zero-novelty active event; decode positive keys, otherwise valid query belongs to unique low event.',
        scope='Exact rational construction, not original-distribution prevalence; raw arrays remain available upstream and M0 uses them.')
    (OUT/'exact_witnesses.json').write_text(json.dumps(proof,indent=2)+'\n')
    trials={r['id']:trial_from_exact(r,i) for i,r in enumerate(exact)}
    original=m.euler_step;signature=inspect.signature(original)
    input_cols=list(signature.parameters);output_cols=['u_next','A_next','r_next','spike','u_candidate']
    cols=input_cols+output_cols;ix={name:i for i,name in enumerate(cols)}
    calls=[];count=0
    def wrapper(*args,**kwargs):
        nonlocal count
        bound=signature.bind(*args,**kwargs);bound.apply_defaults();out=original(*args,**kwargs)
        calls.append([np.nan if bound.arguments[k] is None else float(bound.arguments[k]) for k in input_cols]+[np.nan if x is None else float(x) for x in out]);count+=1
        return out
    def capture(fn):
        calls.clear();result=fn();arr=np.array(calls,dtype=float).reshape((-1,len(cols)));return result,arr
    def eq(a,b):return np.array_equal(a,b,equal_nan=True)
    arrays={};metadata=[];outputs=[];physical={};interventions=[]
    try:
        m.euler_step=wrapper
        for label,trial in trials.items():
            ex=lookup[label]
            expected_ports=np.array([[float(Q(seg[k])) for k in ('content','novelty','query_current')] for seg in ex['encoding_segments'] for _ in range(100)])
            schedule=np.array([[seg['slot'],int(seg['mask']),int(seg['phase']=='presentation'),seg['start_ms']+j*.1] for seg in ex['encoding_segments'] for j in range(100)])
            arrays[label+'_schedule']=schedule
            for carrier in PHYSICAL:
                (state,trace,mean_A),enc=capture(lambda:m.simulate_encoding(trial.keys,trial.values,trial.mask,carrier,A_0=calibration['A_0'],record_trace=True))
                assert len(enc)==1000
                assert np.max(abs(enc[:,[ix['I_content'],ix['novelty'],ix['I_query']]]-expected_ports))<1e-15
                initial=[0.,calibration['A_0'] if carrier in ('M2b','M2d') else 0.,0.]
                assert enc[0,:3].tolist()==initial
                assert eq(np.column_stack([trace[k] for k in ('u','A','r')]),enc[:,:3])
                assert state.to_array().tolist()==enc[-1,[ix['u_next'],ix['A_next'],ix['r_next']]].tolist()
                assert abs(mean_A-enc[:,ix['A']].mean())<1e-15
                expected_clamp=calibration['A_0'] if carrier in ('M2b','M2d') else np.nan
                assert eq(enc[:,ix['clamp_A_const']],np.full(1000,expected_clamp))
                for slot in trial.event_positions:
                    end=slot*200+99;pre=enc[end,[ix['u_next'],ix['A_next'],ix['r_next']]];post=enc[end+1,:3]
                    intended=pre.copy()
                    if carrier in ('M2a','M2d'):intended[[0,2]]=0.
                    elif carrier=='M2c':intended[:]=0.
                    assert eq(intended,post)
                    interventions.append(dict(history=label,carrier=carrier,slot=slot,time_ms=20*slot+10,before=pre.tolist(),after=post.tolist(),rule='reset_u_r' if carrier in ('M2a','M2d') else 'reset_u_A_r' if carrier=='M2c' else 'none'))
                arrays[f'{label}_{carrier}_encoding']=enc
                physical[(label,carrier,'encoding')]=enc
                metadata.append(dict(history=label,carrier=carrier,initial_state=initial,final_encoding_state=state.to_array().tolist(),mean_A=mean_A))
                for delay in (0.,5.,10.):
                    state_before=state.to_array().copy()
                    checkpoint,darr=capture(lambda:m.simulate_delay(state,delay,carrier,A_0=calibration['A_0']))
                    assert np.array_equal(state_before,state.to_array()) and len(darr)==int(10*delay)
                    assert not len(darr) or np.all(darr[:,[ix['I_content'],ix['novelty'],ix['I_query']]]==0)
                    arrays[f'{label}_{carrier}_d{int(delay)}_delay']=darr
                    query_results=[];qarrs=[]
                    for role,q in enumerate(trial.query_cues):
                        checkpoint_before=checkpoint.to_array().copy()
                        result,qarr=capture(lambda:m.simulate_cue_and_readout(checkpoint,q,carrier,g_dec=calibration['g_dec'],b_dec=calibration['b_dec'],A_0=calibration['A_0']))
                        assert np.array_equal(checkpoint_before,checkpoint.to_array()) and len(qarr)==100
                        assert np.all(qarr[:,[ix['I_content'],ix['novelty']]]==0) and np.all(qarr[:,ix['I_query']]==q)
                        assert qarr[0,:3].tolist()==checkpoint_before.tolist()
                        assert result['raw_spikes']==qarr[:,ix['spike']].astype(int).tolist()
                        assert result['eligible_spikes']==result['raw_spikes'][:99]
                        arrays[f'{label}_{carrier}_d{int(delay)}_q{role}']=qarr
                        query_results.append(result);qarrs.append(qarr)
                    physical[(label,carrier,delay)]=(enc,darr,*qarrs)
                    metadata.append(dict(history=label,carrier=carrier,delay=delay,checkpoint=checkpoint.to_array().tolist(),query_results=query_results))
        physical_call_count=count
        runner_call_hashes=[]
        for label,trial in trials.items():
            for model in runner.ALL_MODELS:
                for delay in (0.,5.,10.):
                    result,record=capture(lambda:runner.evaluate_episode(trial,delay,model,**calibration))
                    if model=='M0':
                        assert len(record)==0 and result['v_hat_A']==trial.target_values[0] and result['v_hat_B']==trial.target_values[1]
                    else:
                        carrier='M1' if model.startswith('M4_') else model
                        expected=np.concatenate(physical[(label,carrier,delay)],axis=0)
                        assert eq(record,expected)
                    runner_call_hashes.append(dict(history=label,model=model,delay=delay,euler_calls=len(record),sha256=hashlib.sha256(record.tobytes()).hexdigest()))
                    outputs.append(dict(history=label,N=trial.N,model=model,delay=delay,queries=list(trial.query_cues),targets=list(trial.target_values),**result))
        by={(r['history'],r['model'],r['delay']):r for r in outputs}
        pair_actual=[]
        for a,b in PAIRS:
            same=a!='positive_A';details=[]
            assert eq(arrays[a+'_schedule'],arrays[b+'_schedule'])
            for carrier in PHYSICAL:
                ea=physical[(a,carrier,'encoding')];eb=physical[(b,carrier,'encoding')]
                content_eq=eq(ea[:,ix['I_content']],eb[:,ix['I_content']])
                novelty_eq=eq(ea[:,ix['novelty']],eb[:,ix['novelty']])
                assert content_eq and novelty_eq==same
                for delay in (0.,5.,10.):
                    paths_a=physical[(a,carrier,delay)];paths_b=physical[(b,carrier,delay)]
                    if same:assert all(eq(x,y) for x,y in zip(paths_a,paths_b))
                    details.append(dict(carrier=carrier,delay=delay,content_equal=content_eq,novelty_equal=novelty_eq,full_calls_including_states_and_parameters_equal=all(eq(x,y) for x,y in zip(paths_a,paths_b))))
            for model in runner.ALL_MODELS:
                for delay in (0.,5.,10.):
                    x,y=by[(a,model,delay)],by[(b,model,delay)]
                    pred_equal=(x['v_hat_A'],x['v_hat_B'])==(y['v_hat_A'],y['v_hat_B'])
                    if same:assert pred_equal==(model!='M0')
            pair_actual.append(dict(pair=[a,b],expected_full_input_collision=same,details=details))
        primed=physical[('primed_A','M1','encoding')];primed_checks=[]
        for slot,exact_A in zip((2,4),primed_exact):
            r=primed[200*slot:200*slot+100];curr=r[:,ix['I_content']];gate=r[:,ix['A']]
            assert np.all(gate>0) and np.all(curr>0)
            assert abs(gate[0]-float(exact_A))<1e-14
            primed_checks.append(dict(slot=slot,gate_at_onset=float(gate[0]),minimum_gate=float(gate.min()),content_current=float(curr[0]),minimum_synaptic_product=float(np.min(gate*curr)),all_gate_and_content_strictly_positive=True))
        np.savez_compressed(OUT/'actual_calls.npz',**arrays)
        (OUT/'boundary_outputs.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in outputs))
        (OUT/'boundary_trace_metadata.json').write_text(json.dumps(dict(columns=cols,metadata=metadata,postevent_interventions=interventions,runner_call_hashes=runner_call_hashes,
            source_function_boundary='euler_step observes listed arguments; simulate_encoding upstream still receives raw arrays; masks/timing implement scheduling and postevent controls.',nan_encoding='None clamp/candidate stored as NaN in NPZ; never scientific equality by approximate comparison.'),indent=2)+'\n')
        after={str(p.relative_to(REPO)):digest(p) for p in source_paths};assert before==after
        result=dict(status='PASS',analysis_plan_sha256=PLAN,amendment_sha256=AMENDMENT,script_sha256=digest(Path(__file__)),source_sha256=before,calibration=calibration,
            histories=6,pairs=3,physical_carriers=list(PHYSICAL),runner_models=list(runner.ALL_MODELS),delays=[0,5,10],runner_episodes=len(outputs),role_outputs=2*len(outputs),
            physical_recorded_euler_calls=physical_call_count,runner_recheck_euler_calls=count-physical_call_count,total_actual_euler_calls=count,npz_arrays=len(arrays),
            exact_pair_proofs=pair_proofs,actual_pair_checks=pair_actual,primed_positive_gate_checks=primed_checks,
            full_interface_positive_novelty_collision='RULED_OUT_IN_IDEAL_ARITHMETIC; only ULP-scale value-sum rounding ambiguity in binary64',
            upstream_raw_arrays_equal_in_collision_pairs=False,model_M0_distinguishes_collision_pairs=True,source_files_unchanged=True,
            limitations=['Constructed collisions are not prevalence estimates.','Input information loss is at registered drive transduction, not entire Python API or original window-based TAN.','Positive branch is injective; zero/one low event is query-identifiable from full waveform.','State/readout insufficiency is separate from input-path ambiguity.'],wall_seconds=time.perf_counter()-started)
        (OUT/'BOUNDARY_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:m.euler_step=original

if __name__=='__main__':main()
