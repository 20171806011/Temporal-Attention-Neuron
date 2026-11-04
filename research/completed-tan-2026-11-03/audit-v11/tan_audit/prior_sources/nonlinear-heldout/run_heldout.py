#!/usr/bin/env python3
from fractions import Fraction as Q
from pathlib import Path
import csv,hashlib,importlib.util,itertools,json,math,platform,sys
import numpy as np
import scipy
BASE=Path(__file__).resolve().parent.parent;OUT=Path(__file__).resolve().parent
HASH='6b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f'

def coefficients():
 A=[Q(0)]*3;U=[Q(0)]*6
 for n in range(1000):
  evt=0 if n<100 else 1 if 800<=n<900 else None;k=[Q(4,5),Q(9,5)][evt] if evt is not None else Q(0)
  prod=[k*A[0],k*A[1],k*A[2],Q(0),Q(0),Q(0)]
  if evt==0:prod[1]+=A[0];prod[3]+=A[1];prod[4]+=A[2]
  if evt==1:prod[2]+=A[0];prod[4]+=A[1];prod[5]+=A[2]
  U=[Q(99,100)*u+p/100 for u,p in zip(U,prod)]
  A=[Q(149,150)*x for x in A]
  if evt is not None:A[0]+=(max(Q(0),k-Q(7,5))+Q(3,10))/150;A[evt+1]+=Q(1,1500)
 determinants=[]
 for x,y in itertools.product([Q(-1,2),Q(1,2)],repeat=2):
  det=A[2]*(U[1]+2*U[3]*x+U[4]*y)-A[1]*(U[2]+U[4]*x+2*U[5]*y);assert det<0;determinants.append(float(det))
 assert A[2]>0
 return np.array(list(map(float,U))),np.array(list(map(float,A))),determinants

def real_roots(a,b,c):
 if a==0:return [-c/b] if b!=0 else []
 discriminant=b*b-4*a*c
 if discriminant<0:return []
 root=math.sqrt(discriminant);z=-.5*(b+math.copysign(root,b))
 if z==0:return [-b/(2*a)]
 roots=[z/a,c/z]
 return [roots[0]] if roots[0]==roots[1] else roots

def main():
 assert hashlib.sha256((BASE/'models_p1b_v2.py').read_bytes()).hexdigest()==HASH
 spec=importlib.util.spec_from_file_location('heldout_models',BASE/'models_p1b_v2.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 U,A,corners=coefficients();rho=A[1]/A[2];B2=U[3]-rho*U[4]+rho*rho*U[5]
 def forward(v):
  x,y=np.asarray(v).T
  return np.column_stack([U[0]+U[1]*x+U[2]*y+U[3]*x*x+U[4]*x*y+U[5]*y*y,A[0]+A[1]*x+A[2]*y])
 def decode(state):
  alpha=(state[1]-A[0])/A[2];b1=U[1]-rho*U[2]+alpha*U[4]-2*alpha*rho*U[5];b0=U[0]+U[2]*alpha+U[5]*alpha*alpha-state[0]
  roots=real_roots(B2,b1,b0)
  if not roots:return [0.,0.],'NO_REAL_ROOT',False,0,None,[]
  valid=[];residuals=[];snaps=[];domain_count=0;candidate_values=[]
  for x in roots:
   v=np.array([x,alpha-rho*x]);candidate_values.append(v.tolist());snapped=False
   if np.any(v<-.5-1e-10) or np.any(v>.5+1e-10):continue
   if np.any(v<-.5) or np.any(v>.5):v=np.minimum(.5,np.maximum(-.5,v));snapped=True
   domain_count+=1
   residual=float(np.max(np.abs(forward(v[None,:])[0]-state)))
   if residual<=1e-10:valid.append(v);residuals.append(residual);snaps.append(snapped)
  if len(valid)==1:return valid[0].tolist(),'VALID',snaps[0],len(roots),residuals[0],candidate_values
  return [0.,0.],('AMBIGUOUS' if len(valid)>1 else 'RESIDUAL_REJECTION' if domain_count else 'OUT_OF_DOMAIN'),False,len(roots),None,candidate_values
 def encoding(v):
  N=len(v);u=np.zeros(N);g=np.zeros(N);maxu=0.;maxA=0.
  for n in range(1000):
   evt=0 if n<100 else 1 if 800<=n<900 else None;key=[.8,1.8][evt] if evt is not None else 0.;val=v[:,evt] if evt is not None else np.zeros(N)
   drive=max(0.,key-1.4)+(.1*(val+3) if evt is not None else 0.)
   newu=u+(-u+g*(key+val))/10*.1;newg=g+(-g+drive)/15*.1
   maxu=max(maxu,float(newu.max()));maxA=max(maxA,float(newg.max()));u,g=newu,newg
  assert maxu<1 and maxA<5
  return np.column_stack([u,g]),{'max_encoding_candidate':maxu,'max_encoding_gate':maxA}
 rng=np.random.RandomState(20261103);targets=rng.uniform(-.5,.5,size=(4096,2));N=len(targets)
 states,branch=encoding(targets);poly=forward(targets);maxpoly=float(np.max(np.abs(states-poly)));assert maxpoly<1e-13
 checkvalues=np.vstack([targets[:8],np.array(list(itertools.product([-.5,.5],repeat=2)))]);frozen_residual=0.
 for v in checkvalues:
  u=g=r=0.
  for n in range(1000):
   evt=0 if n<100 else 1 if 800<=n<900 else None;key=[.8,1.8][evt] if evt is not None else 0.;val=v[evt] if evt is not None else 0.
   u,g,r,spike,candidate=m.euler_step(u,g,r,key+val,max(0.,key-1.4)+(.1*(val+3) if evt is not None else 0.),0.)
   assert spike==0 and r==0
  residual=float(np.max(np.abs(np.array([u,g])-encoding(v[None,:])[0][0])));frozen_residual=max(frozen_residual,residual);assert residual<1e-14
 sanity=[]
 for v in itertools.product([-.5,0.,.5],repeat=2):
  decoded,status,snap,count,residual,candidates=decode(forward(np.array([v]))[0]);assert status=='VALID' and np.max(np.abs(np.array(decoded)-v))<1e-9
  sanity.append({'values':v,'decoded':decoded,'status':status,'boundary_snap':snap,'residual':residual})
 assert real_roots(0.,2.,-4.)==[2.] and real_roots(1.,0.,1.)==[]
 J=np.array([[U[1],U[2]],[A[1],A[2]]]);Jinv=np.linalg.inv(J);base=np.array([U[0],A[0]])
 def metrics(pred,valid):
  err=np.abs(pred-targets);out={'n':N,'rmse_per_role':np.sqrt(np.mean(err**2,axis=0)).tolist(),'mae_per_role':err.mean(axis=0).tolist(),'mean_e_comp':float(err.max(axis=1).mean()),'max_abs_error':float(err.max()),'coverage':float(np.mean(valid)),'validity_aware_local_joint_success_0_05':float(np.mean(valid & np.all(err<.05,axis=1)))}
  if np.any(valid):out['conditional_valid_only']={'n':int(valid.sum()),'rmse_per_role':np.sqrt(np.mean(err[valid]**2,axis=0)).tolist(),'mae_per_role':err[valid].mean(axis=0).tolist(),'mean_e_comp':float(err[valid].max(axis=1).mean())}
  return out
 result={'source_sha256':HASH,'protocol_sha256':hashlib.sha256((OUT/'PROTOCOL.md').read_bytes()).hexdigest(),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'target_seed':20261103,'noise_seed':20261104,'n_histories':N,'polynomial_U':U.tolist(),'polynomial_A':A.tolist(),'exact_fraction_corner_determinants':corners,'max_actual_vs_polynomial_residual':maxpoly,'max_frozen_scalar_check_residual':frozen_residual,'heldout_encoding_branch':branch,'upper_corner_branch':encoding(np.array([[.5,.5]]))[1],'boundary_solver_checks':sanity,'zero_baseline':metrics(np.zeros_like(targets),np.ones(N,dtype=bool)),'cells':[]}
 assert states[:,1].max()<2 and states[:,0].max()<1
 np.savetxt(OUT/'heldout_targets.csv',targets,delimiter=',',header='v1,v2',comments='');result['targets_sha256']=hashlib.sha256((OUT/'heldout_targets.csv').read_bytes()).hexdigest()
 nrng=np.random.RandomState(20261104);Zu=nrng.standard_normal((N,100));ZA=nrng.standard_normal((N,100));Zs=nrng.standard_normal((N,3))
 a=.99;b=149/150
 for scenario,ss,su,sA in [('noiseless',0.,0.,0.),('sensor_only',1e-6,0.,0.),('membrane_process_only',0.,1e-6,0.),('sensor_and_membrane',1e-6,1e-6,0.),('gate_process_only',0.,0.,1e-6)]:
  u=states[:,0].copy();g=states[:,1].copy();invalid=np.zeros(N,dtype=bool);sample={};maxcandidate=0.
  for step in range(100):
   newu=u+(-u+g*.1+.8)/10*.1;newg=g+(-g)/15*.1;maxcandidate=max(maxcandidate,float(newu.max()))
   u=newu+su*Zu[:,step];g=newg+sA*ZA[:,step];invalid|=(newu>=1)|(u>=1)|(g<0)|(g>5)
   if step in [0,99]:sample[step+1]=u.copy()
  for n in [1,100]:
   d=.001*(b**n-a**n)/(b-a);obs=np.array([[1.,0.],[a**n,d]])
   y=np.column_stack([states[:,0]+ss*Zs[:,0],sample[n]+ss*Zs[:,1 if n==1 else 2]])
   estimated_state=(y-np.array([0.,.8*(1-a**n)]))@np.linalg.inv(obs).T
   predictions=[];statuses=[];snaps=[];rows=[]
   for i,state in enumerate(estimated_state):
    pred,status,snap,nroots,residual,candidates=decode(state) if not invalid[i] else ([0.,0.],'PHYSICAL_BRANCH_DEPARTURE',False,0,None,[])
    predictions.append(pred);statuses.append(status);snaps.append(snap);rows.append([i,*targets[i],*state,*pred,status,snap,nroots,residual,json.dumps(candidates)])
   pred=np.array(predictions);valid=np.array([x=='VALID' for x in statuses]);counts={x:statuses.count(x) for x in sorted(set(statuses))}
   linear=(estimated_state-base)@Jinv.T
   cell={'scenario':scenario,'m':n,'sigma_sensor':ss,'sigma_membrane':su,'sigma_gate':sA,'physical_branch_departures':int(invalid.sum()),'max_query_preinnovation_candidate':maxcandidate,'status_counts':counts,'numerical_boundary_snaps':int(sum(snaps)),'bounded_exact_inverse':metrics(pred,valid),'unbounded_linear_diagnostic':metrics(linear,np.ones(N,dtype=bool))}
   if scenario=='noiseless':assert valid.all() and np.max(np.abs(pred-targets))<1e-8
   result['cells'].append(cell)
   with (OUT/f'{scenario}_m{n}_predictions.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['id','v1','v2','u_hat','A_hat','pred1','pred2','status','numerical_boundary_snap','real_root_count','forward_residual','candidate_value_pairs_json']);w.writerows(rows)
   print(scenario,n,'coverage',valid.mean(),'RMSE',cell['bounded_exact_inverse']['rmse_per_role'],'Ecomp',cell['bounded_exact_inverse']['mean_e_comp'],'local_success',cell['bounded_exact_inverse']['validity_aware_local_joint_success_0_05'])
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
