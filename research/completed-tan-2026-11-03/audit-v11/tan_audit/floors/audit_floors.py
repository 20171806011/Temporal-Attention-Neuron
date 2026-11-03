#!/usr/bin/env python3
"""Independent exact relaxed-oracle audit; outputs stay outside the repository.

Run: python3 tan_audit/floors/audit_floors.py
Only stdlib + installed NumPy/SciPy and the frozen repository's functions.
No archived outputs are loaded or treated as verified evidence.
"""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import platform
import sys
sys.dont_write_bytecode = True
import numpy as np

ROOT = Path(__file__).resolve().parents[2] / 'Temporal-Attention-Neuron'
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'p1_experiments'), str(ROOT/'p1_experiments/p1b')]
import models_p1b_v2 as model
import test_p1b_g1_fixtures as fixtures


def oracle_exact(symbols):
    """CDF of distance to alphabet via clipped 1D Voronoi cells.

    Each half-cell contributes min(t, length) / 6. With independent targets,
    max-distance CDF is F_D(t)^2. Piecewise quadratic integration is exact.
    """
    alphabet = sorted(set(F(x) for x in symbols))
    assert F(-3) <= alphabet[0] < alphabet[-1] <= F(3)
    boundaries = [F(-3)] + [(a+b)/2 for a,b in zip(alphabet,alphabet[1:])] + [F(3)]
    half_lengths = []
    for i,a in enumerate(alphabet):
        half_lengths += [a-boundaries[i], boundaries[i+1]-a]
    knots = sorted(set([F(0)] + half_lengths))
    emax = F(0)
    eabs = F(0)
    pieces = []
    for lo,hi in zip(knots,knots[1:]):
        mid = (lo+hi)/2
        slope = F(sum(length>mid for length in half_lengths),6)
        intercept = sum((length for length in half_lengths if length<mid),F(0))/6
        emax += (1-intercept**2)*(hi-lo) - slope*intercept*(hi**2-lo**2) - slope**2*(hi**3-lo**3)/3
        eabs += (1-intercept)*(hi-lo) - slope*(hi**2-lo**2)/2
        pieces.append({'lower':str(lo),'upper':str(hi),'cdf_slope':str(slope),'cdf_intercept':str(intercept)})
    return {'alphabet':[str(a) for a in alphabet], 'E_comp_fraction':str(emax),
            'E_comp':float(emax), 'single_operand_MAE_fraction':str(eabs),
            'cdf_pieces':pieces}


def numerical_check(alphabet, seed=2026110251, n=262144):
    rng=np.random.RandomState(seed)
    target=rng.uniform(-3,3,size=(n,2))
    levels=np.array([float(F(a)) for a in alphabet])
    error=np.min(np.abs(target[:,:,None]-levels[None,None,:]),axis=2)
    losses=error.max(axis=1)
    return {'n':n,'seed':seed,'mean':float(losses.mean()),
            'standard_error':float(losses.std(ddof=1)/np.sqrt(n))}


def main():
    all_latency=[F(-57,20)+F(3*j,10) for j in range(20)]
    alphabets={
        'count_full_codebook': list(range(-2,3)),
        'latency_20_only': all_latency,
        'latency_20_plus_zero': all_latency+[F(0)],
        'task_count_superset': [F(-2),F(0)],
        'task_latency_superset': all_latency[7:]+[F(0)],
    }
    expected={'count_full_codebook':F(25,54), 'latency_20_only':F(1,10),
              'latency_20_plus_zero':F(3139,32000), 'task_count_superset':F(13,9)}
    results={}
    for name,alphabet in alphabets.items():
        result=oracle_exact(alphabet)
        if name in expected:
            assert F(result['E_comp_fraction'])==expected[name], (name,result)
        result['monte_carlo']=numerical_check(alphabet)
        assert abs(result['E_comp']-result['monte_carlo']['mean']) < 6*result['monte_carlo']['standard_error']
        results[name]=result

    # Exact identity on a rational grid, including cancellation/sign changes.
    exact_cases=0
    for numerator_a in range(-48,49):
        for numerator_b in range(-48,49):
            ea,eb=F(numerator_a,16),F(numerator_b,16)
            assert (abs(ea+eb)+abs(ea-eb))/2 == max(abs(ea),abs(eb))
            exact_cases+=1

    # Native production controls: state upper envelope, gate-blind cue, old A.
    ub=(1-model.H/model.TAU_M)**100
    envelope=model.simulate_cue_and_readout(model.CarrierState(ub,5.,0.),2.5,'M1')
    first=next(i+1 for i,s in enumerate(envelope['raw_spikes']) if s)
    assert first==36 and sum(envelope['eligible_spikes'])==1
    gate_zero=model.simulate_cue_and_readout(model.CarrierState(ub,0.,0.),2.5,'M1')
    for field in ['u_readout','v_hat_analog','v_hat_count','v_hat_latency','raw_spikes']:
        assert envelope[field]==gate_zero[field]
    old=model.euler_step(0.,0.,0.,I_content=10.,novelty=5.,I_query=0.)
    assert old[0]==0 and old[1]>0 # Uses old A for current.

    # These registered tests are bounded; do not run suite orchestration.
    selected=['test_g1_1_state_closure_and_refractory_readout',
              'test_g1_5_state_collision_non_injectivity',
              'test_g1_6a_codec_benchmarks',
              'test_g1_6b_actual_spike_conformance',
              'test_g1_7_euler_accuracy_and_convergence']
    for name in selected:
        getattr(fixtures,name)()

    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'p1_experiments/task_generator.py',
                      ROOT/'p1_experiments/p1b/models_p1b_v2.py',
                      ROOT/'p1_experiments/p1b/runner_p1b.py',
                      ROOT/'p1_experiments/p1b/P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md']}
    document={'python':platform.python_version(),'numpy':np.__version__,
              'commit':'85213f0d417d54e4f48b301ed1c0e4200ef3c421',
              'archive_available':False,'source_sha256':hashes,
              'exact_metric_identity_cases':exact_cases,
              'selected_registered_fixtures_passed':selected,
              'cue_envelope':{'u_upper_bound':ub,'q_upper_bound':2.5,
                              'first_spike_step':first,'count':envelope['spike_count'],
                              'latency_min_bin':7,'query_A_blind_check_passed':True,
                              'old_A_current_check_passed':True},
              'oracle_results':results}
    (OUT/'results.json').write_text(json.dumps(document,indent=2)+'\n')
    print(json.dumps({'exact_metric_identity_cases':exact_cases,
                      'oracle_floors':{k:v['E_comp_fraction'] for k,v in results.items()},
                      'registered_tests_passed':len(selected)},indent=2))


if __name__=='__main__':
    main()
