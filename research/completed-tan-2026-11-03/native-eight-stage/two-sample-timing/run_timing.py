#!/usr/bin/env python3
from pathlib import Path
import hashlib, importlib.util, json, platform, sys
import numpy as np
import scipy
BASE=Path(__file__).resolve().parent.parent;OUT=Path(__file__).resolve().parent
HASH='6b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f'

def main():
 assert hashlib.sha256((BASE/'models_p1b_v2.py').read_bytes()).hexdigest()==HASH
 priorpath=BASE/'content-gate-rank/results.json';prior=json.loads(priorpath.read_text());J=np.array(prior['arms']['content']['state_sensitivity']['J']);Jinv=np.linalg.inv(J)
 spec=importlib.util.spec_from_file_location('timing_models',BASE/'models_p1b_v2.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 a=.99;b=149/150;q=.8;k=.001
 def d(n):return k*(b**n-a**n)/(b-a) if n else 0.
 def O(n):return np.array([[1.,0.],[a**n,d(n)]])
 def encode(values):
  u=A=r=0.
  for n in range(1000):
   event=0 if n<100 else 1 if 800<=n<900 else None
   active=event is not None;key=[.8,1.8][event] if active else 0.;v=values[event] if active else 0.
   u,A,r,spike,_=m.euler_step(u,A,r,key+v if active else 0.,max(0.,key-1.4)+(.1*(v+3) if active else 0.),0.)
   assert spike==0 and r==0
  return np.array([u,A])
 def query(x):
  u,A=x;r=0.;states=[x.copy()];candidates=[]
  assert A<2 and A>=0 and u<1
  for n in range(100):
   u,A,r,spike,candidate=m.euler_step(float(u),float(A),r,.1,0.,.8)
   assert spike==0 and r==0
   states.append(np.array([u,A]));candidates.append(candidate)
  return np.array(states),1-max(candidates)
 base=encode([0.,0.]);assert np.array_equal(base,np.array(prior['arms']['content']['base_state']))
 delta=np.array([.001,-.001]);pert=encode(delta);trace,margin=query(pert)
 result={'source_sha256':HASH,'prior_results_sha256':hashlib.sha256(priorpath.read_bytes()).hexdigest(),'protocol_sha256':hashlib.sha256((OUT/'PROTOCOL.md').read_bytes()).hexdigest(),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'seed':20261102,'paired_paths':4096,'base_state':base.tolist(),'perturbed_state':pert.tolist(),'delta_values':delta.tolist(),'grid_checks':[],'observations':{},'noise':[]}
 for v1 in [-.5,0.,.5]:
  for v2 in [-.5,0.,.5]:
   x=encode([v1,v2]);t,mar=query(x);checks={}
   for n in [1,100]:
    measured=np.array([x[0],t[n,0]]);expected=O(n)@x+np.array([0.,q*(1-a**n)])
    residual=float(np.max(np.abs(measured-expected)));assert residual<2e-15;checks[str(n)]=residual
   result['grid_checks'].append({'values':[v1,v2],'initial_state':x.tolist(),'query_threshold_margin':mar,'analytic_residual_by_m':checks,'full_100step_integrated_content_probe':1.0,'gated_synaptic_input_proxy':float(.1*.1*t[:-1,1].sum())})
 np.savetxt(OUT/'perturbed_full100step_query_trace.csv',trace,delimiter=',',header='u,A',comments='')
 for n in [1,100]:
  obs=O(n);composed=obs@J
  result['observations'][str(n)]={'O':obs.tolist(),'singular_values_O':np.linalg.svd(obs,compute_uv=False).tolist(),'composed_J':composed.tolist(),'singular_values_composed_J':np.linalg.svd(composed,compute_uv=False).tolist(),'condition_composed_J':float(np.linalg.cond(composed)),'dimensionless_composed_J_vscale3_uscale1_singular_values':np.linalg.svd(3*composed,compute_uv=False).tolist(),'A_sensor_sd_amplification':float(np.sqrt(1+a**(2*n))/d(n)),'A_membrane_process_sd_amplification':float(np.sqrt(sum(a**(2*j) for j in range(n)))/d(n)),'A_gate_process_sd_amplification':float(np.sqrt(sum(d(j)**2 for j in range(n)))/d(n)),'decision_latency_ms':n*.1}
 rng=np.random.RandomState(20261102);N=4096;Zu=rng.standard_normal((N,100));ZA=rng.standard_normal((N,100));Zs=rng.standard_normal((N,3))
 for scenario,ss,su,sA in [('sensor_only',1e-6,0.,0.),('membrane_process_only',0.,1e-6,0.),('sensor_and_membrane',1e-6,1e-6,0.),('gate_process_only',0.,0.,1e-6)]:
  u=np.full(N,pert[0]);A=np.full(N,pert[1]);sample={};branch=0;maxu=-np.inf;minA=np.inf;maxA=-np.inf
  for step in range(100):
   newu=u+(-u+A*.1+.8)/10*.1
   newA=A+(-A)/15*.1
   u=newu+su*Zu[:,step];A=newA+sA*ZA[:,step]
   branch+=int(np.count_nonzero((newu>=1)|(u>=1)|(A<0)|(A>5)))
   maxu=max(maxu,float(u.max()));minA=min(minA,float(A.min()));maxA=max(maxA,float(A.max()))
   if step in [0,99]:sample[step+1]=u.copy()
  assert branch==0,'Branch departure; stop analytic covariance comparison'
  for n in [1,100]:
   sensor_col=1 if n==1 else 2
   y=np.column_stack([pert[0]+ss*Zs[:,0],sample[n]+ss*Zs[:,sensor_col]])
   xhat=(y-np.array([0.,q*(1-a**n)]))@np.linalg.inv(O(n)).T
   state_error=xhat-pert
   empirical=np.cov(state_error,rowvar=False,ddof=1)
   var2=(ss**2*(1+a**(2*n))+su**2*sum(a**(2*j) for j in range(n))+sA**2*sum(d(j)**2 for j in range(n)))/d(n)**2
   expected=np.array([[ss**2,-a**n*ss**2/d(n)],[-a**n*ss**2/d(n),var2]])
   z=[]
   for i in range(2):z.append(float((empirical[i,i]-expected[i,i])/(expected[i,i]*np.sqrt(2/(N-1)))) if expected[i,i]>0 else None)
   estimate=(xhat-base)@Jinv.T;errors=estimate-delta
   result['noise'].append({'scenario':scenario,'m':n,'sigma_sensor':ss,'sigma_membrane_per_step':su,'sigma_gate_per_step':sA,'physical_branch_departures_over_full100steps':branch,'max_u_post':maxu,'min_A_post':minA,'max_A_post':maxA,'empirical_state_error_covariance':empirical.tolist(),'analytic_state_error_covariance':expected.tolist(),'variance_estimate_z_scores':z,'state_error_mean':state_error.mean(axis=0).tolist(),'analytic_value_noise_covariance':(Jinv@expected@Jinv.T).tolist(),'empirical_value_noise_covariance':np.cov(errors,rowvar=False,ddof=1).tolist(),'value_rmse':np.sqrt(np.mean(errors**2,axis=0)).tolist(),'value_mean_error':errors.mean(axis=0).tolist(),'noise_free_local_bias':(Jinv@(pert-base)-delta).tolist()})
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 for x in result['noise']:print(x['scenario'],x['m'],'RMSE',x['value_rmse'],'var_z',x['variance_estimate_z_scores'])
if __name__=='__main__':main()
