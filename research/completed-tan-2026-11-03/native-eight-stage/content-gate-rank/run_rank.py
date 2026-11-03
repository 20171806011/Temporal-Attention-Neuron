#!/usr/bin/env python3
from pathlib import Path
import hashlib, importlib.util, itertools, json, platform, sys
import numpy as np
import scipy
BASE=Path(__file__).resolve().parent.parent;OUT=Path(__file__).resolve().parent
HASH='6b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f'

def main():
 assert hashlib.sha256((BASE/'models_p1b_v2.py').read_bytes()).hexdigest()==HASH
 assert hashlib.sha256((BASE/'summary_p1b.json').read_bytes()).hexdigest()=='5a19a5b20225fdad51641990b61e189f9b00569abdaf08207c5e2d4941136ff9'
 spec=importlib.util.spec_from_file_location('rank_models',BASE/'models_p1b_v2.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 cal=json.loads((BASE/'summary_p1b.json').read_text())['calibration'];eta=.1
 def encode(arm,values):
  u=A=r=0.;du=np.zeros(2);dA=np.zeros(2);states=[];spikes=[];currents=[];candidates=[]
  for n in range(1000):
   event=0 if n<100 else (1 if 800<=n<900 else None)
   active=event is not None;k=([.8,1.8][event] if active else 0.);v=(values[event] if active else 0.)
   indicator=np.zeros(2)
   if active:indicator[event]=1.
   novelty=max(0.,abs(k)-1.4)
   drive=novelty+(eta*(v+3) if arm=='content' else 3*eta if arm=='offset' else 0.)*active
   I=k+v if active else 0.
   nextdu=.99*du+.01*(I*dA+A*indicator)
   nextdA=(149/150)*dA+(.1/15)*eta*indicator*(arm=='content')
   currents.append(A*I)
   u,A,r,spike,candidate=m.euler_step(u,A,r,I,drive,0.)
   du,dA=nextdu,nextdA;states.append([u,A,r]);spikes.append(spike);candidates.append(candidate)
  assert sum(spikes)==0 and r==0 and max(x[1] for x in states)<5
  # At A=0 original arm has a key-only boundary; dA/dv remains exactly0.
  return np.array([u,A]),np.vstack([du,dA]),np.array(states),{'encoding_spikes':sum(spikes),'max_u_candidate':max(candidates),'threshold_margin':1-max(candidates),'max_A':max(x[1] for x in states),'upper_A_clip_margin':5-max(x[1] for x in states),'min_A_post':min(x[1] for x in states),'mean_A_post':float(np.mean(np.array(states)[:,1])),'integrated_abs_synaptic_input_proxy':float(.1*np.abs(currents).sum())}
 def cue(x):return m.simulate_cue_and_readout(m.CarrierState(u=float(x[0]),A=float(x[1]),r=0.),.8,'M1',**cal)
 O=np.array([[1.,0.],[.99,.001]])
 def probe(x):
  nxt=m.euler_step(float(x[0]),float(x[1]),0.,.1,0.,.8)
  assert nxt[3]==0
  y=np.array([x[0],nxt[0]])
  assert np.max(np.abs(y-(O@x+np.array([0.,.008]))))<1e-15
  return y
 def spectral(J):
  sv=np.linalg.svd(J,compute_uv=False)
  return {'J':J.tolist(),'singular_values':sv.tolist(),'determinant':float(np.linalg.det(J)) if J.shape==(2,2) else None,'condition_number':float(sv[0]/sv[-1]) if len(sv)>1 and sv[-1]>0 else None}
 result={'source_sha256':HASH,'protocol_sha256':hashlib.sha256((OUT/'PROTOCOL.md').read_bytes()).hexdigest(),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'eta':eta,'query_content_probe':.1,'calibration':cal,'arms':{},'noise':[]}
 bases={}
 for arm in ['original','offset','content']:
  x,J,states,info=encode(arm,[0.,0.]);bases[arm]=(x,J,states)
  grid=[]
  for v in itertools.product([-.5,0.,.5],repeat=2):
   _,_,_,g=encode(arm,v);grid.append({'values':v,**g})
  diffs=[]
  for h in [1e-4,1e-5]:
   fd=np.empty((2,2));fdout=np.empty(2)
   for i in range(2):
    plus=np.zeros(2);minus=np.zeros(2);plus[i]=h;minus[i]=-h
    xp,_,_,_=encode(arm,plus);xm,_,_,_=encode(arm,minus)
    fd[:,i]=(xp-xm)/(2*h);fdout[i]=(cue(xp)['v_hat_analog']-cue(xm)['v_hat_analog'])/(2*h)
   expected=cal['g_dec']*.99**100*J[0]
   assert np.max(np.abs(fd-J))<1e-9 and np.max(np.abs(fdout-expected))<1e-9
   diffs.append({'h':h,'finite_difference_J':fd.tolist(),'max_abs_J_residual':float(np.max(np.abs(fd-J))),'finite_difference_actual_affine_readout_J':fdout.tolist(),'analytic_actual_affine_readout_J':expected.tolist(),'max_abs_readout_residual':float(np.max(np.abs(fdout-expected)))})
  armres={'base_state':x.tolist(),'base_branch_and_input':info,'state_sensitivity':spectral(J),'dimensionless_state_sensitivity_v_scale3_u_scale1_A_scale5':spectral(np.diag([1.,.2])@J*3),'dimensionless_probe_sensitivity_v_scale3_u_scale1':spectral(O@J*3),'modified_probe_sensitivity':spectral(O@J),'original_raw_readout_J':(.99**100*J[0:1]).tolist(),'grid_branch_checks':grid,'finite_differences':diffs,'actual_base_readout':cue(x),'modified_probe_base':probe(x).tolist()}
  result['arms'][arm]=armres
  np.savetxt(OUT/f'{arm}_base_trace.csv',states,delimiter=',',header='u_post,A_post,r_post',comments='')
 assert bases['offset'][2].tobytes()==bases['content'][2].tobytes()
 result['offset_content_base_trajectories_bitwise_identical']=True
 # Original recurrence fidelity at all9 grid points.
 for v in itertools.product([-.5,0.,.5],repeat=2):
  state,_,_=m.simulate_encoding(np.array([.8,0,0,0,1.8]),np.array([v[0],0,0,0,v[1]]),np.array([1,0,0,0,1],dtype=bool),'M1')
  ours,_,_,_=encode('original',v);assert state.to_array()[:2].tobytes()==ours.tobytes()
 result['original_matches_frozen_encoding_all_9_points']=True
 u=bases['content'][0][0];blind=[cue([u,A]) for A in [0.,.1,1.,5.]]
 assert all(x==blind[0] for x in blind)
 result['matched_u_vary_A_actual_cue_all_return_fields_equal']=True
 rng=np.random.RandomState(20261102);draws={'state':rng.standard_normal((2000,2)),'original_raw_readout':rng.standard_normal((2000,1)),'modified_probe':rng.standard_normal((2000,2))};delta=np.array([.001,-.001])
 for arm,(x,J,_) in bases.items():
  xp,_,_,_=encode(arm,delta)
  measurements={'state':(J,xp-x),'original_raw_readout':(.99**100*J[0:1],np.array([cue(xp)['u_readout']-cue(x)['u_readout']])),'modified_probe':(O@J,probe(xp)-probe(x))}
  for mode,(K,change) in measurements.items():
   inverse=np.linalg.pinv(K,rcond=1e-14)
   for sigma in [0.,1e-8,1e-6,1e-4]:
    estimate=(change[None,:]+sigma*draws[mode])@inverse.T
    error=estimate-delta
    result['noise'].append({'arm':arm,'observation':mode,'sigma_raw_observation_units':sigma,'n_draws':2000,'delta_values':delta.tolist(),'rmse_per_value':np.sqrt(np.mean(error**2,axis=0)).tolist(),'mean_error_per_value':error.mean(axis=0).tolist(),'linear_noise_covariance':(sigma**2*inverse@inverse.T).tolist(),'local_linearization_or_nullspace_bias':(inverse@change-delta).tolist(),'singular_values_over_sigma':(np.linalg.svd(K,compute_uv=False)/sigma).tolist() if sigma else None})
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 for arm,info in result['arms'].items():print(arm,info['state_sensitivity'], 'probe',info['modified_probe_sensitivity']['singular_values'])
if __name__=='__main__':main()
