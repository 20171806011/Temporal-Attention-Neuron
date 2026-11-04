"""Independent ledger-only range/precision/leak audit; primary files are read-only."""
from pathlib import Path
from fractions import Fraction as F
import json, math

ROOT=Path(__file__).resolve().parent
rows=[json.loads(x) for x in (ROOT/'predictions.jsonl').read_text().splitlines()]
inputs={r['id']:r for r in map(json.loads,(ROOT/'inputs.jsonl').read_text().splitlines())}

def errmax(pred,target):return max(abs(a-b) for a,b in zip(pred,target))

table=[]
for group in ('iid','adversarial'):
    for arm in ('interpolation','M0_cache'):
        scenarios=sorted(set(r['scenario'] for r in rows if r['group']==group and r['arm']==arm and r['variant']=='original'))
        for scenario in scenarios:
            rr=[r for r in rows if r['group']==group and r['arm']==arm and r['scenario']==scenario and r['variant']=='original']
            raw=[errmax(r['raw_predictions'],r['targets']) for r in rr if r['raw_predictions'] is not None]
            scored=[errmax(r['predictions'],r['targets']) for r in rr]
            table.append(dict(group=group,arm=arm,scenario=scenario,n=len(rr),invalid=sum(not r['valid'] for r in rr),raw_finite_n=len(raw),
                raw_mean_ecomp=sum(raw)/len(raw),raw_max_error=max(raw),scored_mean_ecomp=sum(scored)/len(scored),scored_max_error=max(scored)))

checks={name:0 for name in ('coefficient_fullscale_noise','query_key_noise','coefficient_plus_query','write_and_query_key_noise','common_post_leak10','differential_post_leak10')}
worst_slack={name:math.inf for name in checks};max_leak_identity_residual=0.
for r in rows:
    if r['arm']!='interpolation' or r['variant']!='original' or r['scenario'] not in checks:continue
    scenario=r['scenario'];rec=inputs[r['id']];raw=r['raw_predictions'];assert raw is not None
    if scenario=='coefficient_fullscale_noise':bound=.597
    elif scenario=='query_key_noise':bound=.012
    elif scenario=='coefficient_plus_query':bound=.609012
    elif scenario=='write_and_query_key_noise':bound=6*(.0001+.0001)/(.05-2*.0001)
    else:
        la=math.exp(-10/150);lb=la if scenario=='common_post_leak10' else math.exp(-10/100)
        k=rec['keys'];v=rec['values'];a=(v[1]-v[0])/(k[1]-k[0])
        for q,pred,target in zip(rec['queries'],raw,rec['targets']):
            theoretical=(la-lb)*a*q+(lb-1)*target
            max_leak_identity_residual=max(max_leak_identity_residual,abs((pred-target)-theoretical))
        bound=300*abs(la-lb)+3*abs(1-lb)
    observed=errmax(raw,r['targets']);assert observed<=bound+1e-10
    assert abs(r['raw_error_bound']-bound)<1e-12
    checks[scenario]+=1;worst_slack[scenario]=min(worst_slack[scenario],bound-observed)
assert max_leak_identity_residual<1e-10

precision=[]
for r in rows:
    if r['group']!='adversarial' or r['arm']!='interpolation' or r['scenario']!='float32':continue
    a,b=map(F.from_float,r['state']);raw_error=[];exact_stored_line_error=[];readout_rounding=[]
    for q,pred,target in zip(r['queries'],r['raw_predictions'],r['targets']):
        exact=a*F.from_float(q)+b
        raw_error.append(abs(pred-target));exact_stored_line_error.append(float(abs(exact-F.from_float(target))))
        readout_rounding.append(float(F.from_float(pred)-exact))
    precision.append(dict(id=r['id'],state=r['state'],status=r['status'],a_overshoot=max(0,abs(r['state'][0])-120),
        b_overshoot=max(0,abs(r['state'][1])-297),raw_error=raw_error,exact_stored_line_error=exact_stored_line_error,
        readout_rounding=readout_rounding,scored_error=[abs(p-t) for p,t in zip(r['predictions'],r['targets'])]))

result=dict(metrics=table,raw_bound_check_counts=checks,minimum_bound_slack=worst_slack,max_leak_identity_residual=max_leak_identity_residual,
    adversarial_float32=precision,primary_files_modified=False,
    interpretation='Range fallback error and unclipped arithmetic error are distinct. Finite-precision boundary rejection is a predeclared device policy, not a theorem of mathematical retrieval failure.')
(ROOT/'RANGE_PRECISION_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'checked_bound_rows':sum(checks.values()),'max_leak_identity_residual':max_leak_identity_residual,
    'adversarial_float32_rejected':sum(x['status']!='VALID' for x in precision),'adversarial_float32_rawmax':max(max(x['raw_error']) for x in precision)},indent=2))
