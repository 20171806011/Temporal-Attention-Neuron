"""Bounded independent check of saved binding-control ledgers; no primary rewrite."""
from pathlib import Path
from fractions import Fraction as F
import collections,hashlib,json,sys,copy
import numpy as np
P=Path(__file__).resolve().parent
ROOT=P.parent.parent/'Temporal-Attention-Neuron'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'p1_experiments'))
from task_generator import TaskGenerator

inputs=[json.loads(x) for x in (P/'inputs.jsonl').read_text().splitlines()]
rows=[json.loads(x) for x in (P/'predictions.jsonl').read_text().splitlines()]
resources=[json.loads(x) for x in (P/'encoding_resources.jsonl').read_text().splitlines()]
summary=json.loads((P/'results.json').read_text())
assert len(inputs)==len(resources)==768 and len(rows)==2304
byinput={(x['trial_id'],x['variant']):x for x in inputs}
byresource={(x['trial_id'],x['variant']):x for x in resources}
byrow={(x['trial_id'],x['variant'],x['arm']):x for x in rows}
assert len(byinput)==768 and len(byrow)==2304
for filename,field in [('PREREGISTRATION.md','protocol_sha256'),('run_control.py','script_sha256'),('inputs.jsonl','inputs_sha256')]:
 assert hashlib.sha256((P/filename).read_bytes()).hexdigest()==summary[field]
helper=P.parent/'variable_key_test.py'
helper_hash=hashlib.sha256(helper.read_bytes()).hexdigest()
assert helper_hash==json.loads((P.parent/'variable_key_results/summary.json').read_text())['script_sha256']
modelhash=hashlib.sha256((ROOT/'p1_experiments/p1b/models_p1b_v2.py').read_bytes()).hexdigest()
assert modelhash==summary['model_sha256']=='6b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f'

# Regenerate the exact finite input schedule with a continuous RNG; no metrics
# or observed outcomes are used in generation.
gen=TaskGenerator();rng=np.random.RandomState(2026110211)
for i in range(256):
 t=gen.generate_single_trial(rng,i,2026110211,N=2)
 for variant,tt in [('original',t),('query_swap',gen.create_query_swap_variant(t)),('value_swap',gen.create_value_swap_variant(t))]:
  inp=byinput[(i,variant)]
  assert inp['keys']==tt.keys.tolist() and inp['values']==tt.values.tolist()
  assert inp['mask']==tt.mask.tolist() and inp['positions']==tt.event_positions
  assert inp['queries']==list(tt.query_cues) and inp['targets']==list(tt.target_values)
  assert inp['target_indices']==list(tt.target_idx)

# Independent exact 100-step block coefficients, not the primary per-step basis.
am=F(99,100);ag=F(1499,1500)
am100=am**100;ag100=ag**100
cross=(am100-ag100)/(am-ag);geom=(1-am100)/(1-am)
def exact_forward(pos,d,s):
 u=A=F(0)
 for slot in range(5):
  j=pos.index(slot) if slot in pos else None
  for dd,ss in [(F(0) if j is None else d[j],F(0) if j is None else s[j]),(F(0),F(0))]:
   u=am100*u+F(1,100)*ss*(A*cross+dd*(geom-cross))
   A=ag100*A+(1-ag100)*dd
 return u,A
def coefficients(pos):
 z=[F(0),F(0)];one=[F(1),F(0)];two=[F(0),F(1)]
 return tuple(float(x) for x in [exact_forward(pos,one,z)[1],exact_forward(pos,two,z)[1],exact_forward(pos,one,one)[0],exact_forward(pos,one,two)[0],exact_forward(pos,two,two)[0]])
posset=sorted(set(tuple(x['positions']) for x in inputs));coef={pos:coefficients(pos) for pos in posset}
def block_float(k,v,pos):
 u=A=0.;max_u=max_A=0.
 for slot in range(5):
  j=pos.index(slot) if slot in pos else None
  for d,s in [(0. if j is None else max(0.,k[j]-1.4)+.1*(v[j]+3),0. if j is None else k[j]+v[j]),(0.,0.)]:
   newA=float(ag100)*A+(1-float(ag100))*d
   # Convex membrane recurrence bounds all intermediate points, even s<0.
   max_u=max(max_u,u,s*max(A,newA) if s>=0 else s*min(A,newA))
   max_A=max(max_A,A,newA)
   u=float(am100)*u+.01*s*(A*float(cross)+d*float(geom-cross));A=newA
 return np.array([u,A,0.]),max_u,max_A

max_endpoint_residual=0.;global_u_bound=0.
for item in inputs:
 p=tuple(item['positions']);k=np.array(item['keys'])[list(p)];v=np.array(item['values'])[list(p)]
 z,_,_=block_float(k,v,p)
 max_endpoint_residual=max(max_endpoint_residual,float(np.max(abs(z-byresource[(item['trial_id'],item['variant'])]['endpoint']))))
for p in posset:
 # The coarser block convex bound may exceed1 although actual trajectory stays
 # below it. Verify this upper-envelope trajectory exactly at every step.
 u=A=F(0);ub=Ab=F(0)
 for n in range(1000):
  slot,within=divmod(n,200);active=slot in p and within<100
  s=F(11,2) if active else F(0);d=F(17,10) if active else F(0)
  u=am*u+F(1,100)*A*s;A=ag*A+F(1,1500)*d
  ub=max(ub,u);Ab=max(Ab,A)
 assert ub<1 and Ab<5
 global_u_bound=max(global_u_bound,float(ub))
assert max_endpoint_residual<1e-13

max_inverse_difference=0.;independent_status=collections.Counter();root_checks=0
for r in rows:
 inp=byinput[(r['trial_id'],r['variant'])]
 assert r['targets']==inp['targets']
 if r['arm']=='M0':
  kk=np.array(r['buffered_keys']);vv=np.array(r['buffered_values'])
  assert r['predictions']==[float(vv[np.argmin((kk-q)**2)]) for q in r['queries']]==r['targets']
  continue
 f=r['features'];assert set(f)=={'state','keys','positions','queries'}
 p=tuple(f['positions']);truekeys=np.array(inp['keys'])[list(p)]
 expectedkeys=truekeys if r['arm']=='correct_metadata' else truekeys[::-1]
 assert f['keys']==expectedkeys.tolist() and f['queries']==inp['queries']
 assert f['state']==byresource[(r['trial_id'],r['variant'])]['endpoint']
 a1,a2,b11,b12,b22=coef[p];k=np.array(f['keys']);u,A,_=f['state']
 c=np.maximum(0.,k-1.4)+.3;ratio=a1/a2
 d2constant=(A-a1*c[0])/a2
 s2constant=k[1]+(d2constant-c[1])/.1
 FF=b12*c[0]+b22*d2constant;GG=.1*(b12-b22*ratio)
 polynomial=[b11*.1-GG*ratio,
             b11*(c[0]+.1*k[0])+GG*s2constant-FF*ratio,
             b11*c[0]*k[0]+FF*s2constant-u]
 roots=np.roots(polynomial);valid=[]
 for x in roots:
  if abs(np.imag(x))>1e-10:continue
  x=float(np.real(x));v=np.array([x,(d2constant-.1*ratio*x-c[1])/.1])
  if np.any(v < -3.-1e-10) or np.any(v>3.+1e-10):continue
  z,ub,Ab=block_float(k,v,p)
  # All legal-domain candidates have already passed the exact global encoding
  # envelope proof above; do not reject using the deliberately loose block bound.
  if Ab>=5 or np.max(abs(z-f['state']))>1e-10:continue
  if not any(np.max(abs(v-vold))<=1e-7 for vold in valid):valid.append(v)
 assert len(valid)==(1 if r['valid'] else 0),(r['trial_id'],r['variant'],r['arm'],valid,r['status'])
 if valid:
  pred=np.array([valid[0][int(np.argmin(abs(k-q)))] for q in f['queries']])
  max_inverse_difference=max(max_inverse_difference,float(np.max(abs(pred-r['predictions']))))
 else:assert r['predictions']==[0.,0.]
 independent_status[(r['arm'],bool(valid))]+=1;root_checks+=1
assert max_inverse_difference<1e-8

recomputed={};original_losses={}
for arm in ['correct_metadata','wrong_metadata','M0']:
 recomputed[arm]={}
 for variant in ['original','query_swap','value_swap']:
  rr=[byrow[(i,variant,arm)] for i in range(256)]
  err=abs(np.array([r['predictions'] for r in rr])-np.array([r['targets'] for r in rr]));valid=np.array([r['valid'] for r in rr])
  score={'mean_e_comp':float(np.max(err,axis=1).mean()),'coverage':float(valid.mean()),
         'valid_joint_delivery_0_5':float(np.mean(valid&np.all(err<.5,axis=1))),
         'local_valid_joint_0_05':float(np.mean(valid&np.all(err<.05,axis=1)))}
  for key,value in score.items():assert abs(value-summary['metrics'][arm][variant][key])<1e-14
  # Independently score paired sum and difference as a check of the max identity.
  pred=np.array([r['predictions'] for r in rr]);truth=np.array([r['targets'] for r in rr])
  pairloss=(abs(pred.sum(axis=1)-truth.sum(axis=1))+abs(np.diff(pred,axis=1)[:,0]-np.diff(truth,axis=1)[:,0]))/2
  assert np.max(abs(pairloss-err.max(axis=1)))<1e-14
  recomputed[arm][variant]=score
  if variant=='original':original_losses[arm]=err.max(axis=1)
 for i in range(256):
  a=byrow[(i,'original',arm)];b=byrow[(i,'query_swap',arm)]
  assert a['predictions'][::-1]==b['predictions'] and a['status']==b['status']
diff=original_losses['wrong_metadata']-original_losses['correct_metadata']
rng=np.random.RandomState(2026110212);bootstrap=np.array([np.mean(diff[rng.choice(256,size=256,replace=True)]) for _ in range(2000)])
ci=np.quantile(bootstrap,[.025,.975])
assert np.max(abs(ci-summary['paired_wrong_minus_correct_original']['percentile95_ci']))<1e-14

# Interface test only: passing one query instead of both cannot expose another
# role's cue to root selection. Does not import primary main or regenerate data.
sys.path.insert(0,str(P))
import run_control as primary
single_query_checks=0
for arm in ['correct_metadata','wrong_metadata']:
 for i in range(8):
  r=byrow[(i,'original',arm)]
  for j in (0,1):
   features=copy.deepcopy(r['features']);features['queries']=[features['queries'][j]]
   got=primary.inverse(features)
   assert got['status']==r['status'] and got['predictions'][0]==r['predictions'][j]
   single_query_checks+=1

out={'primary_outputs_rewritten':False,'input_rows':768,'prediction_rows':2304,'role_predictions':4608,
     'resource_rows':768,'base_seed_reproduction_passed':True,'protocol_script_input_model_hashes_match':True,
     'imported_coefficients_helper_sha256':helper_hash,'independent_block_endpoint_max_residual':max_endpoint_residual,
     'independent_whole_domain_conservative_voltage_upper':global_u_bound,
     'independent_inverse_rows_checked':root_checks,'independent_inverse_max_prediction_difference':max_inverse_difference,
     'single_query_interface_checks':single_query_checks,'recomputed_metrics':recomputed,
     'paired_bootstrap_95_ci':ci.tolist(),
     'qualification':['Wrong key/slot metadata changes both assumed encoding map and query-to-slot lookup.','Full-state and external metadata oracle, no two-probe/native addressing claim.','Recorded state extrema omit initial state0; post-step A minima can exceed0.','Imported coefficient helper identity is separately verified here.']}
(P/'INDEPENDENT_CHECKS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
