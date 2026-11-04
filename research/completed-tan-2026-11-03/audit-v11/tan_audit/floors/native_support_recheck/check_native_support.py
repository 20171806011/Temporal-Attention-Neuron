#!/usr/bin/env python3
"""Exact rational timeline and native floating implementation cross-checks.

This audit does not modify the model or replace registered decision rules.
"""
import sys
sys.dont_write_bytecode = True
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import math
import platform
import numpy as np

ROOT=Path(__file__).resolve().parents[3] / 'Temporal-Attention-Neuron'
OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'p1_experiments'),str(ROOT/'p1_experiments/p1b')]
import models_p1b_v2 as model
from task_generator import TrialData
from runner_p1b import evaluate_episode

LAM=F(99,100)
H=F(1,10)
QMAX=F(5,2)


def exact_step(u,r,q):
    if r>0:
        return F(0), max(F(0),r-H), 0, None
    candidate=LAM*u+q/100
    if candidate>=1:
        return F(0), F(2), 1, candidate
    return candidate,F(0),0,candidate


def exact_cue(u,q,n=110,r=F(0)):
    rows=[]
    for k in range(1,n+1):
        before=r
        u,r,s,candidate=exact_step(u,r,q)
        rows.append({'step':k,'spike':s,'u':u,'r':r,
                     'r_before':before,'candidate':candidate})
    return rows


def float_cue(u,q,n=110,r=0.):
    rows=[]
    A=0.
    for k in range(1,n+1):
        before=r
        u,A,r,s,candidate=model.euler_step(u,A,r,0.,0.,q)
        rows.append({'step':k,'spike':s,'u':u,'r':r,
                     'r_before':before,'candidate':candidate})
    return rows


def serialize(rows):
    return [{k:(str(v) if isinstance(v,F) else v) for k,v in row.items()}
            for row in rows]


def first_spike(rows):
    return next((row['step'] for row in rows if row['spike']),None)


def trial(values):
    keys=np.array([.5,0.,0.,0.,1.])
    vv=np.array([values[0],0.,0.,0.,values[1]])
    mask=np.array([True,False,False,False,True])
    return TrialData(trial_id=0,seed=0,window_size=5,N=2,
        event_positions=[0,4],keys=keys,values=vv,mask=mask,
        target_idx=(0,4),target_keys=(.5,1.),target_values=tuple(values),
        query_cues=(.5,1.),gt_add=sum(values),gt_sub=values[0]-values[1],
        t1_is_middle=False,t2_is_middle=False,is_middle_target=False,rejection_count=0)


def main():
    assert model.H==.1 and model.TAU_M==10. and model.TAU_REF==2.
    assert model.N_CUE_STEPS==100 and model.THETA_TH==1. and model.U_RESET==0.
    envelope=LAM**100
    # Exact inequalities establish the earliest possible spike times.
    u35=QMAX+(envelope-QMAX)*LAM**35
    u36=QMAX+(envelope-QMAX)*LAM**36
    reset50=QMAX*(1-LAM**50)
    reset51=QMAX*(1-LAM**51)
    assert u35<1<u36 and reset50<1<reset51
    exact=exact_cue(envelope,QMAX)
    floating=float_cue(float(envelope),float(QMAX))
    assert [r['step'] for r in exact if r['spike']]==[36,107]
    assert [r['step'] for r in floating if r['spike']]==[36,107]
    for rows in (exact,floating):
        assert rows[35]['spike']==1
        assert all(row['candidate'] is None for row in rows[36:56])
        assert rows[55]['r']==0 and rows[56]['candidate'] is not None
        assert rows[105]['candidate']<1 and rows[106]['spike']==1

    # A conservative 2^-40 absolute error per step dwarfs the IEEE rounding
    # budget for source-task arithmetic (bounded values/intermediates < 10).
    # Even this cumulative outward perturbation cannot permit a step-35 spike.
    delta=F(1,2**40)
    silent_upper=F(1)
    for _ in range(100):
        silent_upper=LAM*silent_upper+delta
    cue_upper=silent_upper
    for _ in range(35):
        cue_upper=LAM*cue_upper+QMAX/100+delta
    reset_upper=F(0)
    for _ in range(50):
        reset_upper=LAM*reset_upper+QMAX/100+delta
    assert cue_upper<1 and reset_upper<1

    # Negative voltages cannot defeat the invariant. Positive r clamps u=0.
    # These samples check source execution; the report supplies general proof.
    silence_checks=[]
    samples=[F(-100),F(-3),F(-1,100),F(0),F(1,2),F(999999,1000000)]
    for u0 in samples:
        for r0 in [F(0),F(1,1000),F(1,10),F(19,10),F(2)]:
            er=exact_cue(u0,F(0),n=100,r=r0)
            fr=float_cue(float(u0),0.,n=100,r=float(r0))
            assert not any(x['spike'] for x in er+fr)
            assert er[-1]['r']==0 and fr[-1]['r']==0
            assert er[-1]['u']<envelope
            assert fr[-1]['u']<=float(silent_upper)
            silence_checks.append({'u_initial':str(u0),'r_initial':str(r0),
                                   'exact_final_u':str(er[-1]['u']),
                                   'float_final_u':fr[-1]['u']})

    # Interior-domain witness reaches first spike 36 in the actual encoding.
    # First key .5 has zero novelty. Last key2.499 supplies novelty1.099.
    # Compute a subthreshold linear response; set final event voltage .999.
    k_last=F(2499,1000)
    u=A=F(0)
    for _ in range(100):
        u=LAM*u+A/100
        A=F(149,150)*A+(k_last-F(7,5))/150
    v_last=F(999,1000)/u-k_last
    assert F(-3)<v_last<F(3)
    keys=np.array([.5,0.,0.,0.,float(k_last)])
    values=np.array([0.,0.,0.,0.,float(v_last)])
    mask=np.array([True,False,False,False,True])
    state,trace,_=model.simulate_encoding(keys,values,mask,'M1',record_trace=True)
    native=model.simulate_cue_and_readout(state,float(k_last),'M1')
    native_extended=float_cue(state.u,float(k_last))
    assert np.max(trace['u'])<1 and state.r==0
    assert first_spike(native_extended)==36
    assert [r['step'] for r in native_extended if r['spike']]==[36,107]
    assert native['spike_count']==1 and native['v_hat_count']==-2.

    # Test all 100 intended first-spike steps through the production function.
    # Early-step fixtures deliberately use out-of-task q: decoder conformance
    # only. They do not establish native task reachability.
    first_step_fixtures=[]
    for desired in range(1,101):
        q=F(101) if desired==1 else (1/(1-LAM**desired)+1/(1-LAM**(desired-1)))/2
        er=exact_cue(F(0),q,n=100)
        assert first_spike(er)==desired
        result=model.simulate_cue_and_readout(model.CarrierState(),float(q),'M1')
        raw_first=next(k+1 for k,s in enumerate(result['raw_spikes']) if s)
        assert raw_first==desired
        assert len(result['raw_spikes'])==100 and len(result['eligible_spikes'])==99
        if desired==100:
            assert result['spike_count']==0
            assert result['abstained_count'] and result['abstained_latency']
            assert result['v_hat_count']==result['v_hat_latency']==0.
            expected_bin=None
        else:
            expected_bin=desired//5
            actual_bin=int((desired*model.H)/.5)
            assert expected_bin==actual_bin
            assert result['t_first']==desired*model.H
            assert result['v_hat_latency']==-2.85+expected_bin*.30
            assert not result['abstained_latency']
        first_step_fixtures.append({'desired_first':desired,'q':float(q),
            'q_in_clean_range':.3<=float(q)<=2.5,'raw_first':raw_first,
            'eligible_count':result['spike_count'],'bin':expected_bin,
            'latency_output':result['v_hat_latency'],
            'abstained':result['abstained_latency']})
    assert first_step_fixtures[98]['bin']==19
    assert first_step_fixtures[99]['abstained']
    registered_boundary=model.simulate_cue_and_readout(model.CarrierState(),1.58,'M1')
    assert [i+1 for i,s in enumerate(registered_boundary['raw_spikes']) if s]==[100]
    assert registered_boundary['spike_count']==0 and registered_boundary['abstained_latency']

    # Actual runner retains zero in E_comp while abstention forces failure,
    # even when zero equals both targets exactly.
    silence_scores=[]
    for vals in [(0.,0.),(1.2,-.4)]:
        t=trial(vals)
        for channel in ('M4_Count','M4_Latency'):
            row=evaluate_episode(t,0.,channel,1.,0.,0.)
            assert row['abstained_A'] and row['abstained_B']
            assert not row['succ_A'] and not row['succ_B'] and not row['joint_succ']
            assert row['v_hat_A']==row['v_hat_B']==0.
            assert math.isclose(row['e_comp'],max(map(abs,vals)),abs_tol=1e-15)
            silence_scores.append({'targets':vals,'channel':channel,'output':row})

    files=['p1_experiments/p1b/models_p1b_v2.py',
           'p1_experiments/p1b/runner_p1b.py','p1_experiments/task_generator.py',
           'p1_experiments/p1b/P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md']
    output={'python':platform.python_version(),'numpy':np.__version__,
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
        'exact_bounds':{'silent_u_upper':str(envelope),'step35':str(u35),
            'step36':str(u36),'reset_step50':str(reset50),'reset_step51':str(reset51)},
        'bound_decimals':{'silent_u_upper':float(envelope),'step35':float(u35),
            'step36':float(u36),'reset_step50':float(reset50),'reset_step51':float(reset51)},
        'conservative_rounding_check':{'per_step_absolute_error_budget':str(delta),
            'step35_upper':float(cue_upper),'reset_step50_upper':float(reset_upper)},
        'exact_timeline':serialize(exact),'float_timeline':floating,
        'silence_checks':silence_checks,'first_step_fixtures':first_step_fixtures,
        'source_domain_witness':{'keys':keys.tolist(),'values':values.tolist(),
            'u_at_cue':state.u,'r_at_cue':state.r,'A_at_cue':state.A,
            'max_recorded_encoding_u':float(np.max(trace['u'])),
            'native_eligible_spikes':[i+1 for i,s in enumerate(native['eligible_spikes']) if s],
            'extended_cue_spikes':[r['step'] for r in native_extended if r['spike']]},
        'silence_scores':silence_scores,
        'registered_q1_58_boundary':{'raw_spike_steps':[100],
            'eligible_count':registered_boundary['spike_count'],
            'abstained_count':registered_boundary['abstained_count'],
            'abstained_latency':registered_boundary['abstained_latency']},
        'claims':{'native_count_output_superset':[-2.,0.],
            'native_latency_bin_superset':list(range(7,20)),
            'silence_in_composition':True,'silence_successful_delivery':False}}
    (OUT/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'exact_and_float_spikes':[36,107],
        'silence_state_cases':len(silence_checks),'first_step_fixtures':len(first_step_fixtures),
        'all_latency_bin_indices_match_integer_division':True,
        'native_reachable_first36_witness':output['source_domain_witness'],
        'source_silence_score_cases':len(silence_scores)},indent=2))


if __name__=='__main__':
    main()
