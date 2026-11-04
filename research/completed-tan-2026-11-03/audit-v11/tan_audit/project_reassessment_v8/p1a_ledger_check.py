"""Read-only independent arithmetic audit of existing P1A Rev2 ledgers; no new trials."""
from pathlib import Path
import csv,gzip,json,hashlib,math,time,resource
import numpy as np
HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent/'Temporal-Attention-Neuron'
P=REPO/'p1_experiments/results/run_p1a_rev2'
start=time.perf_counter()
models=['QueryBlindControl','PositiveScalarKernel','SignedScalarKernel','HistoricalSprint4_2_KernelStaticControl','FrozenSRotationHarmonicReference','ScalarMetricAttention','VectorQKAddressAttention']
rows=[]
with gzip.open(P/'trial_inputs.csv.gz','rt',newline='') as f:
 for r in csv.DictReader(f):
  rows.append(dict(id=(r['condition'],int(r['seed']),int(r['trial_id'])),keys=json.loads(r['keys']),values=json.loads(r['values']),mask=json.loads(r['mask']),q=[float(r['q1']),float(r['q2'])],target=[int(r['target_idx1']),int(r['target_idx2'])],middle=[r['t1_is_middle']=='True',r['t2_is_middle']=='True']))
assert len(rows)==60000
ids={r['id']:i for i,r in enumerate(rows)};assert len(ids)==len(rows)
keys=np.array([r['keys'] for r in rows]);values=np.array([r['values'] for r in rows]);mask=np.array([r['mask'] for r in rows]);q=np.array([r['q'] for r in rows]);target=np.array([r['target'] for r in rows]);middle=np.array([r['middle'] for r in rows]);N=mask.sum(1)
assert np.all(np.isin(N,[2,3]));assert np.all(mask[np.arange(len(rows))[:,None],target]);assert np.all(target[:,0]!=target[:,1])
truth=values[np.arange(len(rows))[:,None],target]
expected={}
for model in models:
 if model=='QueryBlindControl':
  w=np.column_stack([np.argmax(mask,axis=1),mask.shape[1]-1-np.argmax(mask[:,::-1],axis=1)])
  probs=np.repeat((mask/N[:,None])[:,None,:],2,axis=1)
 else:
  if model=='PositiveScalarKernel':scores=np.repeat((2*keys)[:,None,:],2,axis=1)
  elif model=='SignedScalarKernel':scores=np.where(q>=1.4,1.,-1.)[:,:,None]*keys[:,None,:]
  elif model=='ScalarMetricAttention':scores=-(q[:,:,None]-keys[:,None,:])**2
  elif model=='VectorQKAddressAttention':scores=np.cos(.4*q)[:,:,None]*np.cos(.4*keys)[:,None,:]+np.sin(.4*q)[:,:,None]*np.sin(.4*keys)[:,None,:]
  elif model=='FrozenSRotationHarmonicReference':scores=np.cos(.4*np.array([1.8,2.6]))[None,:,None]*np.cos(.4*keys)[:,None,:]+np.sin(.4*np.array([1.8,2.6]))[None,:,None]*np.sin(.4*keys)[:,None,:]
  else:
   norm=np.sqrt(keys**2+keys**4);phi1=np.divide(keys,norm,out=np.zeros_like(keys),where=norm!=0);phi2=np.divide(keys**2,norm,out=np.zeros_like(keys),where=norm!=0)
   S=np.array([1.8,2.6]);scores=(2*S*np.cos(.4*S))[None,:,None]*phi1[:,None,:]+(2*S*np.sin(.4*S))[None,:,None]*phi2[:,None,:]
  scores=np.where(mask[:,None,:],scores,-np.inf);w=scores.argmax(2)
  ex=np.exp(10*(scores-scores.max(2,keepdims=True)));probs=ex/ex.sum(2,keepdims=True)
 hard=values[np.arange(len(rows))[:,None],w];soft=(probs*values[:,None,:]).sum(2)
 expected[model]=dict(w=w,hard=hard,soft=soft,correct=w==target)
assert np.array_equal(expected['ScalarMetricAttention']['w'],expected['VectorQKAddressAttention']['w'])
coverage=np.zeros((len(rows),len(models)),dtype=np.uint8);maxdiff={k:0. for k in ['hard_values','soft_values','hard_add','hard_sub','soft_add','soft_sub']};counts={};middle_hits={};noutputs=0
with gzip.open(P/'model_outputs.csv.gz','rt',newline='') as f:
 for r in csv.DictReader(f):
  id=(r['condition'],int(r['seed']),int(r['trial_id']));i=ids[id];m=r['model'];j=models.index(m);coverage[i,j]+=1;e=expected[m]
  assert [int(r['w1']),int(r['w2'])]==e['w'][i].tolist()
  assert [r['c1_correct']=='True',r['c2_correct']=='True']==e['correct'][i].tolist()
  assert (r['both_correct']=='True')==bool(e['correct'][i].all())
  savedhard=np.array([float(r['v1_hat']),float(r['v2_hat'])]);savedsoft=np.array([float(r['v1_soft']),float(r['v2_soft'])]);h=e['hard'][i];s=e['soft'][i];t=truth[i]
  maxdiff['hard_values']=max(maxdiff['hard_values'],float(abs(savedhard-h).max()));maxdiff['soft_values']=max(maxdiff['soft_values'],float(abs(savedsoft-s).max()))
  for key,got,field in [('hard_add',abs(h.sum()-t.sum()),'err_add_hard'),('hard_sub',abs((h[0]-h[1])-(t[0]-t[1])),'err_sub_hard'),('soft_add',abs(s.sum()-t.sum()),'err_add_soft'),('soft_sub',abs((s[0]-s[1])-(t[0]-t[1])),'err_sub_soft')]:maxdiff[key]=max(maxdiff[key],abs(float(r[field])-got))
  cond=id[0];k=(cond,m);counts.setdefault(k,[0,0]);counts[k][0]+=1;counts[k][1]+=int(e['correct'][i].all());middle_hits.setdefault(k,[0,0]);middle_hits[k][0]+=int(middle[i].sum());middle_hits[k][1]+=int((middle[i]&e['correct'][i]).sum());noutputs+=1
assert noutputs==420000 and np.all(coverage==1);assert max(maxdiff.values())<1e-12
# Exact membership in nearest-neighbor Voronoi cell provides the query-noise explanation.
left=np.full((len(rows),2),-np.inf);right=np.full((len(rows),2),np.inf)
for role in range(2):
 tk=keys[np.arange(len(rows)),target[:,role]];below=np.where(mask&(keys<tk[:,None]),keys,-np.inf).max(1);above=np.where(mask&(keys>tk[:,None]),keys,np.inf).min(1)
 left[:,role]=(tk+below)/2;right[:,role]=(tk+above)/2
inside=(q>left)&(q<right)
assert np.array_equal(inside,expected['ScalarMetricAttention']['correct'])
summary=[dict(condition=c,model=m,n=n,both_hits=h,both_accuracy=h/n,middle_queries=middle_hits[c,m][0],middle_hits=middle_hits[c,m][1]) for (c,m),(n,h) in counts.items()]
source_paths=['p1_experiments/models.py','p1_experiments/task_generator.py','p1_experiments/run_p1a_experiment.py','p1_experiments/oracle_and_evaluator.py','p1_experiments/THEORETICAL_PROPOSITION_1D_METRIC.md','p1_experiments/REPLAY_PROTOCOL_AMENDMENT_1.md','p1_experiments/P1A_FINAL_BASELINE_MANIFEST.md']
source_paths += ['p1_experiments/results/run_p1a_rev2/'+f for f in ['trial_inputs.csv.gz','model_outputs.csv.gz','P1A_SCIENTIFIC_REPORT.md','g1_gate_report.json','post_write_recomputation_audit.json','condition_summary.csv']]
result=dict(status='PASS',scope='Independent formula replay of existing saved inputs/outputs, not fresh sampling or a new experiment',histories=len(rows),output_rows=noutputs,coverage_exactly_one_per_model=True,scalar_vector_winner_identity_queries=len(rows)*2,nearest_neighbor_noise_cell_checks=len(rows)*2,max_abs_query_key_angle=float(np.max(np.where(mask[:,None,:],abs(.4*(q[:,:,None]-keys[:,None,:])),0))),all_observed_distances_inside_cosine_monotonic_domain=True,max_absolute_differences=maxdiff,condition_counts=summary,source_sha256={s:hashlib.sha256((REPO/s).read_bytes()).hexdigest() for s in source_paths},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),python_numpy=np.__version__,wall_seconds=time.perf_counter()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
assert result['max_abs_query_key_angle']<math.pi
(HERE/'P1A_LEDGER_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['condition_counts','source_sha256']},indent=2))
for c in ['C2_N3_zero_noise','C3_N3_low_noise','C4_N3_high_noise']:
 print(c,[x for x in summary if x['condition']==c and x['model'] in ['SignedScalarKernel','ScalarMetricAttention','VectorQKAddressAttention']])
