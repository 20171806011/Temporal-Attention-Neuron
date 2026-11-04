#!/usr/bin/env python3
from fractions import Fraction as Q
from pathlib import Path
import csv,hashlib,importlib.util,itertools,json,math,platform,sys
import numpy as np
import scipy
BASE=Path(__file__).resolve().parent.parent;OUT=Path(__file__).resolve().parent
HASH='6b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f'
ARMS=[('baseline',10,15),('gate_only',10,150),('uniform',100,150)]

def coefficients(tm,tA):
 h=Q(1,10);am=1-h/tm;ag=1-h/tA;A=[Q(0)]*3;U=[Q(0)]*6
 for n in range(1000):
  evt=0 if n<100 else 1 if 800<=n<900 else None;k=[Q(4,5),Q(9,5)][evt] if evt is not None else Q(0)
  prod=[k*A[0],k*A[1],k*A[2],Q(0),Q(0),Q(0)]
  if evt==0:prod[1]+=A[0];prod[3]+=A[1];prod[4]+=A[2]
  if evt==1:prod[2]+=A[0];prod[4]+=A[1];prod[5]+=A[2]
  U=[am*u+h/tm*p for u,p in zip(U,prod)];A=[ag*x for x in A]
  if evt is not None:A[0]+=h/tA*(max(Q(0),k-Q(7,5))+Q(3,10));A[evt+1]+=h/tA*Q(1,10)
 dets=[]
 for x,y in itertools.product([Q(-1,2),Q(1,2)],repeat=2):
  det=A[2]*(U[1]+2*U[3]*x+U[4]*y)-A[1]*(U[2]+U[4]*x+2*U[5]*y);assert det<0;dets.append(float(det))
 return np.array(list(map(float,U))),np.array(list(map(float,A))),dets

def roots(a,b,c):
 if a==0:return [-c/b] if b else []
 disc=b*b-4*a*c
 if disc<0:return []
 z=-.5*(b+math.copysign(math.sqrt(disc),b))
 if z==0:return [-b/(2*a)]
 r=[z/a,c/z];return r[:1] if r[0]==r[1] else r

def summary(x):return {'mean':float(np.mean(x)),'min':float(np.min(x)),'max':float(np.max(x))}

def main():
 assert hashlib.sha256((BASE/'models_p1b_v2.py').read_bytes()).hexdigest()==HASH
 spec=importlib.util.spec_from_file_location('timescale_models',BASE/'models_p1b_v2.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 targets=np.random.RandomState(20261105).uniform(-.5,.5,size=(4096,2));N=len(targets)
 np.savetxt(OUT/'heldout_targets.csv',targets,delimiter=',',header='v1,v2',comments='')
 targethash=hashlib.sha256((OUT/'heldout_targets.csv').read_bytes()).hexdigest();(OUT/'TARGETS_SHA256').write_text(targethash+'  heldout_targets.csv\n');print('frozen target hash before metrics:',targethash)
 nrng=np.random.RandomState(20261106);Zu=nrng.standard_normal((N,100));ZA=nrng.standard_normal((N,100));Zs=nrng.standard_normal((N,3))
 def metrics(pred,valid):
  err=np.abs(pred-targets);r={'n':N,'rmse_per_role':np.sqrt(np.mean(err**2,axis=0)).tolist(),'mae_per_role':err.mean(axis=0).tolist(),'mean_e_comp':float(err.max(axis=1).mean()),'coverage':float(valid.mean()),'local_valid_joint_success_0_05':float(np.mean(valid&np.all(err<.05,axis=1)))}
  if valid.any():r['conditional_valid_only']={'n':int(valid.sum()),'rmse_per_role':np.sqrt(np.mean(err[valid]**2,axis=0)).tolist(),'mae_per_role':err[valid].mean(axis=0).tolist(),'mean_e_comp':float(err[valid].max(axis=1).mean())}
  return r
 result={'source_sha256':HASH,'protocol_sha256':hashlib.sha256((OUT/'PROTOCOL.md').read_bytes()).hexdigest(),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'target_seed':20261105,'noise_seed':20261106,'bootstrap_seed':20261107,'n_histories':N,'targets_sha256':targethash,'zero_baseline':metrics(np.zeros_like(targets),np.ones(N,dtype=bool)),'arms':{}}
 sensor_losses={};zero_loss=abs(targets).max(axis=1)
 for name,tm,tA in ARMS:
  U,A,dets=coefficients(tm,tA);am=1-.1/tm;ag=1-.1/tA;rho=A[1]/A[2];B2=U[3]-rho*U[4]+rho*rho*U[5]
  def forward(v):
   x,y=v.T;return np.column_stack([U[0]+U[1]*x+U[2]*y+U[3]*x*x+U[4]*x*y+U[5]*y*y,A[0]+A[1]*x+A[2]*y])
  def encoding(v):
   count=len(v);u=np.zeros(count);g=np.zeros(count);areaA=np.zeros(count);areaI=np.zeros(count);areaU2=np.zeros(count);maxu=0.;maxg=0.;maxI=0.
   for n in range(1000):
    evt=0 if n<100 else 1 if 800<=n<900 else None;key=[.8,1.8][evt] if evt is not None else 0.;val=v[:,evt] if evt is not None else np.zeros(count)
    drive=max(0.,key-1.4)+(.1*(val+3) if evt is not None else 0.);I=g*(key+val)
    areaA+=.1*g;areaI+=.1*abs(I);areaU2+=.1*u*u
    newu=u+(-u+I)/tm*.1;newg=g+(-g+drive)/tA*.1
    maxu=max(maxu,float(newu.max()));maxg=max(maxg,float(newg.max()));maxI=max(maxI,float(I.max()));u,g=newu,newg
   assert maxu<1 and maxg<5
   return np.column_stack([u,g]),{'max_candidate':maxu,'max_gate':maxg,'max_synaptic_current':maxI,'integral_A':summary(areaA),'integral_abs_Isyn':summary(areaI),'integral_u_squared':summary(areaU2),'endpoint_u':summary(u),'endpoint_A':summary(g)}
  states,resources=encoding(targets);maxpoly=float(np.max(abs(states-forward(targets))));assert maxpoly<1e-13
  scalar_res=0.;scalar_query_res=0.
  dQuery=(.1/tm)*.1*(ag**100-am**100)/(ag-am)
  for v in np.vstack([targets[:8],np.array(list(itertools.product([-.5,.5],repeat=2)))]):
   u=g=r=0.
   for n in range(1000):
    evt=0 if n<100 else 1 if 800<=n<900 else None;key=[.8,1.8][evt] if evt is not None else 0.;val=v[evt] if evt is not None else 0.
    u,g,r,spike,cand=m.euler_step(u,g,r,key+val,max(0.,key-1.4)+(.1*(val+3) if evt is not None else 0.),0.,tau_m=tm,tau_A=tA);assert spike==0 and r==0
   res=float(max(abs(np.array([u,g])-encoding(v[None,:])[0][0])));scalar_res=max(scalar_res,res);assert res<1e-14
   query_expected=am**100*u+dQuery*g+.8*(1-am**100)
   for _ in range(100):
    u,g,r,spike,cand=m.euler_step(u,g,r,.1,0.,.8,tau_m=tm,tau_A=tA);assert spike==0 and r==0
   qres=abs(u-query_expected);scalar_query_res=max(scalar_query_res,qres);assert qres<1e-13
  def decode(state):
   alpha=(state[1]-A[0])/A[2];b1=U[1]-rho*U[2]+alpha*U[4]-2*alpha*rho*U[5];b0=U[0]+U[2]*alpha+U[5]*alpha*alpha-state[0]
   rr=roots(B2,b1,b0)
   if not rr:return [0.,0.],'NO_REAL_ROOT',False,[],None
   valid=[];snaps=[];resids=[];candpairs=[];dom=0
   for x in rr:
    v=np.array([x,alpha-rho*x]);candpairs.append(v.tolist())
    if (v<-.5-1e-10).any() or (v>.5+1e-10).any():continue
    snap=bool((v<-.5).any() or (v>.5).any());v=np.minimum(.5,np.maximum(-.5,v)) if snap else v;dom+=1
    resid=float(max(abs(forward(v[None,:])[0]-state)))
    if resid<=1e-10:valid.append(v);snaps.append(snap);resids.append(resid)
   if len(valid)==1:return valid[0].tolist(),'VALID',snaps[0],candpairs,resids[0]
   return [0.,0.],('AMBIGUOUS' if len(valid)>1 else 'RESIDUAL_REJECTION' if dom else 'OUT_OF_DOMAIN'),False,candpairs,None
  for v in itertools.product([-.5,0.,.5],repeat=2):
   pred,status,_,_,_=decode(forward(np.array([v]))[0]);assert status=='VALID' and max(abs(np.array(pred)-v))<1e-9
  J=np.array([[U[1],U[2]],[A[1],A[2]]]);d=(.1/tm)*.1*(ag**100-am**100)/(ag-am);O=np.array([[1.,0.],[am**100,d]])
  arm={'tau_m':tm,'tau_A':tA,'polynomial_U':U.tolist(),'polynomial_A':A.tolist(),'exact_corner_determinants':dets,'J':J.tolist(),'singular_values_J':np.linalg.svd(J,compute_uv=False).tolist(),'O100':O.tolist(),'singular_values_OJ':np.linalg.svd(O@J,compute_uv=False).tolist(),'condition_OJ':float(np.linalg.cond(O@J)),'dimensionless_J_vscale3_u1_A5':(np.diag([1.,.2])@J*3).tolist(),'max_actual_polynomial_residual':maxpoly,'max_scalar_vector_residual':scalar_res,'max_frozen_scalar_query_analytic_residual':scalar_query_res,'encoding_resources':resources,'upper_corner_branch':encoding(np.array([[.5,.5]]))[1],'cells':[]}
  assert states[:,1].max()<2 and states[:,0].max()<1
  for scenario,ss,su,sA in [('noiseless',0.,0.,0.),('sensor_only',1e-6,0.,0.),('membrane_process_only',0.,1e-6,0.),('sensor_and_membrane',1e-6,1e-6,0.),('gate_process_only',0.,0.,1e-6)]:
   u=states[:,0].copy();g=states[:,1].copy();invalid=np.zeros(N,dtype=bool);maxcand=0.;maxgate=0.;maxI=0.;areaA=np.zeros(N);areaI=np.zeros(N);areaU2=np.zeros(N)
   for step in range(100):
    I=g*.1;areaA+=.1*g;areaI+=.1*abs(I);areaU2+=.1*u*u
    newu=u+(-u+I+.8)/tm*.1;newg=g+(-g)/tA*.1
    maxcand=max(maxcand,float(newu.max()));maxgate=max(maxgate,float(newg.max()));maxI=max(maxI,float(I.max()))
    u=newu+su*Zu[:,step];g=newg+sA*ZA[:,step];invalid|=(newu>=1)|(u>=1)|(g<0)|(g>5)
   if scenario=='noiseless':
    predicted=states@O[1]+.8*(1-am**100);res=float(max(abs(u-predicted)));assert res<1e-13;arm['max_query_analytic_residual']=res
   y=np.column_stack([states[:,0]+ss*Zs[:,0],u+ss*Zs[:,2]])
   xhat=(y-np.array([0.,.8*(1-am**100)]))@np.linalg.inv(O).T
   preds=[];statuses=[];snaps=[];rows=[]
   for i,state in enumerate(xhat):
    pred,status,snap,candidates,resid=decode(state) if not invalid[i] else ([0.,0.],'PHYSICAL_BRANCH_DEPARTURE',False,[],None)
    preds.append(pred);statuses.append(status);snaps.append(snap);rows.append([i,*targets[i],*state,*pred,status,snap,resid,json.dumps(candidates)])
   pred=np.array(preds);valid=np.array([s=='VALID' for s in statuses]);met=metrics(pred,valid)
   if scenario=='noiseless':assert valid.all() and np.max(abs(pred-targets))<1e-8
   if scenario=='sensor_only':sensor_losses[name]=abs(pred-targets).max(axis=1)
   cell={'scenario':scenario,'status_counts':{s:statuses.count(s) for s in sorted(set(statuses))},'numerical_boundary_snaps':int(sum(snaps)),'physical_branch_departures':int(invalid.sum()),'metrics':met,'query_resources':{'max_candidate':maxcand,'max_gate':maxgate,'max_synaptic_current':maxI,'integral_A':summary(areaA),'integral_abs_Isyn':summary(areaI),'integral_u_squared':summary(areaU2)}}
   arm['cells'].append(cell)
   with (OUT/f'{name}_{scenario}_predictions.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['id','v1','v2','u_hat','A_hat','pred1','pred2','status','numerical_boundary_snap','forward_residual','candidate_pairs_json']);w.writerows(rows)
   print(name,scenario,'coverage',met['coverage'],'RMSE',met['rmse_per_role'],'Ecomp',met['mean_e_comp'],'success',met['local_valid_joint_success_0_05'])
  result['arms'][name]=arm
 differences={'baseline_minus_zero':sensor_losses['baseline']-zero_loss,'gate_only_minus_zero':sensor_losses['gate_only']-zero_loss,'uniform_minus_zero':sensor_losses['uniform']-zero_loss,'gate_only_minus_baseline':sensor_losses['gate_only']-sensor_losses['baseline'],'uniform_minus_baseline':sensor_losses['uniform']-sensor_losses['baseline']}
 brng=np.random.RandomState(20261107);boot={k:[] for k in differences}
 for _ in range(20):
  indices=brng.randint(0,N,size=(100,N))
  for key,diff in differences.items():boot[key].extend(diff[indices].mean(axis=1).tolist())
 result['sensor_only_primary_paired_bootstrap']={key:{'mean_ecomp_difference':float(diff.mean()),'percentile95_ci':np.quantile(boot[key],[.025,.975]).tolist(),'replicates':2000} for key,diff in differences.items()}
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
