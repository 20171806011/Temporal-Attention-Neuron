"""Bounded independent audit of the original frozen P1B carrier, not a new fit.

Run: PYTHONDONTWRITEBYTECODE=1 python3 repository_boundary_audit.py
No repository files or historical outputs are modified.
"""
from pathlib import Path
import contextlib
import csv
import hashlib
import io
import json
import math
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1] / 'Temporal-Attention-Neuron'
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO/'p1_experiments/p1b'), str(REPO/'p1_experiments')]
import models_p1b_v2 as model
import test_p1b_g1_fixtures as fixture
from task_generator import TaskGenerator

def main():
    lam = 1-model.H/model.TAU_M
    u_upper = lam**100
    first_min = math.ceil(math.log((2.5-1)/(2.5-u_upper))/math.log(lam))
    fresh_min = math.ceil(math.log((2.5-1)/2.5)/math.log(lam))
    boundary = model.simulate_cue_and_readout(
        model.CarrierState(u=u_upper, A=0, r=0), 2.5, 'M1')
    boundary_spikes = [j+1 for j,s in enumerate(boundary['raw_spikes']) if s]
    assert first_min == 36 and fresh_min == 51
    assert first_min + 20 + fresh_min == 107
    assert boundary_spikes == [36]
    # Independently continue 120 steps to verify the algebraic second-spike bound.
    u,A,r = u_upper,0.,0.
    continued = []
    for j in range(120):
        u,A,r,s,_ = model.euler_step(u,A,r,0.,0.,2.5)
        if s: continued.append(j+1)
    assert continued == [36,107]
    gen = TaskGenerator(window_size=5)
    rows = []
    counts = {0:0,1:0}
    for N in [2,3]:
        seed = 2026110200+N
        rng = np.random.RandomState(seed)
        for tid in range(128):
            trial = gen.generate_single_trial(rng, tid, seed, N)
            state,_,_ = model.simulate_encoding(trial.keys,trial.values,trial.mask,'M1')
            assert state.u <= u_upper + 1e-14 and state.r == 0
            for delay in [0.,5.,10.]:
                cue = model.simulate_delay(state,delay,'M1')
                for role,q in enumerate(trial.query_cues):
                    result = model.simulate_cue_and_readout(cue,q,'M1')
                    k = result['spike_count']
                    assert k in (0,1), (N,tid,delay,role,k)
                    assert result['v_hat_count'] in (-2.,0.)
                    if result['t_first'] is not None:
                        assert result['t_first'] >= 3.6-1e-12
                    counts[k] += 1
                    rows.append(dict(N=N,seed=seed,trial_id=tid,delay=delay,
                                     role=role,q=q,u_cue=cue.u,A_cue=cue.A,r_cue=cue.r,
                                     count=k,count_value=result['v_hat_count'],
                                     first_latency=result['t_first']))
    with (OUT/'native_boundary_trials.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Invoke only bounded fixtures without repository-writing smoke workflows.
    names = [
        'test_g1_1_state_closure_and_refractory_readout',
        'test_g1_5_state_collision_non_injectivity',
        'test_g1_6b_actual_spike_conformance',
        'test_g1_7_euler_accuracy_and_convergence',
    ]
    logs=io.StringIO()
    with contextlib.redirect_stdout(logs):
        for name in names:
            getattr(fixture,name)()
    (OUT/'native_selected_fixtures.log').write_text(logs.getvalue())
    files=['p1_experiments/task_generator.py','p1_experiments/p1b/models_p1b_v2.py',
           'p1_experiments/p1b/runner_p1b.py','p1_experiments/p1b/test_p1b_g1_fixtures.py',
           'p1_experiments/p1b/P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md']
    report=dict(
        numpy=np.__version__,fresh_histories=256,role_queries=len(rows),
        seed_per_N={2:2026110202,3:2026110203},
        counts=counts,conservative_u_upper=u_upper,
        min_first_step=first_min,min_fresh_step=fresh_min,
        min_second_step=107,boundary_spikes=boundary_spikes,
        continued120_spikes=continued,
        task_reachable_count_alphabet=[-2,0],
        relaxed_count_population_Ecomp_floor='13/9',
        selected_fixtures_passed=names,
        source_sha256={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in files},
        limitations=['Fresh conditional reproduction, not historical byte replay.',
                     'Population floor assumes independent Uniform[-3,3] operands and allows arbitrary oracle symbol choice.',
                     'Physical bound applies to original clean task q<=2.5 and tau_m10, not added content probes or modified time constants.',
                     'Generic counts2..5 fixture uses query drives outside original task domain.'])
    (OUT/'native_boundary_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
