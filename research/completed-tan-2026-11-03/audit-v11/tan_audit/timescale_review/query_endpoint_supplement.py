"""Supplement previously omitted query endpoint ranges, without rerunning decoders.

Reuses the exact saved targets, declared disturbance draw order and fixed arms.
Writes only new audit artifacts; never changes supplied source or its outputs.
"""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent
SRC=ROOT.parent/'prior_sources/timescale-heldout'
result=json.loads((SRC/'results.json').read_text())
targets=np.loadtxt(SRC/'heldout_targets.csv',delimiter=',',skiprows=1)
assert hashlib.sha256((SRC/'heldout_targets.csv').read_bytes()).hexdigest()==result['targets_sha256']
N=len(targets);rng=np.random.RandomState(20261106)
Zu=rng.standard_normal((N,100));ZA=rng.standard_normal((N,100));Zs=rng.standard_normal((N,3))
stats=lambda x:dict(mean=float(np.mean(x)),min=float(np.min(x)),max=float(np.max(x)))
out={}
for name,arm in result['arms'].items():
    tm,tA=arm['tau_m'],arm['tau_A'];u=np.zeros(N);A=np.zeros(N)
    # Actual drift recurrence, avoiding polynomial-rounding changes to endpoints.
    for n in range(1000):
        j=0 if n<100 else 1 if 800<=n<900 else None
        k=0. if j is None else [.8,1.8][j]
        v=np.zeros(N) if j is None else targets[:,j]
        D=0. if j is None else max(0.,k-1.4)+.1*(v+3)
        nu=u+(-u+A*(k+v))/tm*.1
        nA=A+(-A+D)/tA*.1
        u,A=nu,nA
    for variable,values in [('u',u),('A',A)]:
        expected=arm['encoding_resources']['endpoint_'+variable]
        assert all(abs(value-expected[key])<1e-14 for key,value in stats(values).items())
    initial_u=u.copy();initial_A=A.copy();cells={}
    for scenario,su,sA in [('noiseless',0.,0.),('sensor_only',0.,0.),
                          ('membrane_process_only',1e-6,0.),('sensor_and_membrane',1e-6,0.),
                          ('gate_process_only',0.,1e-6)]:
        u=initial_u.copy();A=initial_A.copy();invalid=np.zeros(N,dtype=bool)
        for step in range(100):
            nu=u+(-u+.1*A+.8)/tm*.1;nA=A+(-A)/tA*.1
            u=nu+su*Zu[:,step];A=nA+sA*ZA[:,step]
            invalid|=(nu>=1)|(u>=1)|(A<0)|(A>5)
        assert not invalid.any()
        cells[scenario]=dict(endpoint_u_after_state_innovation=stats(u),
                             endpoint_A_after_state_innovation=stats(A),
                             physical_branch_departures=int(invalid.sum()))
    out[name]=dict(tau_m=tm,tau_A=tA,cells=cells,
                  f1_over_f2=arm['polynomial_A'][1]/arm['polynomial_A'][2],
                  d100=arm['O100'][1][1])
report=dict(date='2026-11-02',purpose='Fill omitted query endpoint ranges only; no decoder, fit, new test arm or performance re-evaluation.',
            targets_sha256=result['targets_sha256'],noise_seed=20261106,
            n_histories=N,arms=out,
            endpoint_semantics='Physical state immediately after the 100th query drift and state innovation; sensor noise is observation-only and excluded.',
            original_outputs_modified=False,
            reset_carryover='Not simulated; associated energy and throughput costs excluded.')
(ROOT/'query_endpoint_ranges.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({name:{'f1_over_f2':arm['f1_over_f2'],'d100':arm['d100'],
                       'noiseless_query_endpoints':arm['cells']['noiseless']} for name,arm in out.items()},indent=2))
