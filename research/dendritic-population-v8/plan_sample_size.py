"""Pre-data planning only, using the frozen V7 pilot; no model calls."""
from pathlib import Path
import json,hashlib,math
import numpy as np
from scipy.stats import t
import scipy
R=Path(__file__).resolve().parent
source=R.parent/'dendritic-population-v7/results/TEST_SUMMARY.json'
pilot=json.loads(source.read_text())['statistics']['intervals']['accuracy_POINT_loss_lr_interaction']['paired_values']
sd=float(np.std(pilot,ddof=1));target=.10
minimum=next(n for n in range(3,2000) if t.ppf(.975,n-1)*sd/np.sqrt(n)<=target)
assert minimum==104
q=float(t.ppf(.975,minimum-1));half=q*sd/np.sqrt(minimum)
cost=json.loads((R.parent/'dendritic-population-v7/RESOURCE_COST.json').read_text())
point_seconds=sum(c['recorded_training_seconds'] for c in cost['cells'] if c['arm']=='POINT')
result=dict(pilot_commit='0c74816da453003490f313c8ee7939e16840e4f2',pilot_source=str(source.relative_to(R.parent)),pilot_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),pilot_n=20,pilot_paired_interaction=pilot,pilot_sample_sd=sd,
    target_primary_halfwidth=target,fixed_n=minimum,fixed_runs=4*minimum,critical_value=q,planned_halfwidth=float(half),inflated_sd_factor=1.5,inflated_sd_halfwidth=float(1.5*half),
    minimum_n_if_inflated_sd_for_same_target=next(n for n in range(3,2000) if t.ppf(.975,n-1)*1.5*sd/np.sqrt(n)<=target),
    v7_POINT_run_count=80,v7_POINT_training_seconds=point_seconds,projected_training_seconds=point_seconds/80*minimum*4,
    versions=dict(numpy=np.__version__,scipy=scipy.__version__),
    interpretation='Plug-in Student-t precision planning.104 is the smallest integer giving planned half-width at most10pp under V7 sample SD. Actual precision/power is not guaranteed;1.5x SD is sensitivity, not a variance confidence bound. Count fixed before any V8 data; no optional stopping or outcome-based expansion.')
(R/'PLANNING.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='pilot_paired_interaction'},indent=2))
