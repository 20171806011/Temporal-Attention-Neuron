"""Read-only supplemental checks of stored P1A input metadata and paired noise."""
from pathlib import Path
import csv,gzip,json,hashlib
HERE=Path(__file__).resolve().parent
p=HERE.parent.parent/'Temporal-Attention-Neuron/p1_experiments/results/run_p1a_rev2/trial_inputs.csv.gz'
n=mid=0;maxnoise=0.;low={}
with gzip.open(p,'rt',newline='') as f:
 for r in csv.DictReader(f):
  k=json.loads(r['keys']);v=json.loads(r['values']);m=json.loads(r['mask']);pos=json.loads(r['event_positions']);N=int(r['N']);targets=[int(r['target_idx1']),int(r['target_idx2'])];assert pos==[i for i,x in enumerate(m) if x] and len(pos)==N
  for role,t in enumerate(targets,1):
   assert t in pos and float(r[f'target_key{role}'])==k[t] and float(r[f'target_val{role}'])==v[t]
   expected=N==3 and k[t]==sorted(k[i] for i in pos)[1];assert expected==(r[f't{role}_is_middle']=='True')
   if r['condition']=='C2_N3_zero_noise':mid+=expected
  assert float(r['gt_add'])==v[targets[0]]+v[targets[1]] and float(r['gt_sub'])==v[targets[0]]-v[targets[1]]
  identity=(r['seed'],r['trial_id']);base=(k,v,m,pos,targets)
  if r['condition']=='C3_N3_low_noise':low[identity]=(base,[float(r['q1']),float(r['q2'])])
  if r['condition']=='C4_N3_high_noise':
   orig,queries=low[identity];assert base==orig
   for role,t in enumerate(targets,1):maxnoise=max(maxnoise,abs(float(r[f'q{role}'])-max(.01,k[t]+5*(queries[role-1]-k[t]))))
  n+=1
assert n==60000 and mid==6705 and maxnoise<1e-14
out={'status':'PASS','input_field_consistency_rows':n,'independently_recomputed_C2_middle_queries':mid,'C3_C4_matched_histories':len(low),'max_shared_noise_scaling_residual':maxnoise,'scope':'Read-only supplemental validation of target fields, positions, middle labels and paired query-noise scaling; no generated trials.','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(HERE/'P1A_INPUT_FIELD_CHECKS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
