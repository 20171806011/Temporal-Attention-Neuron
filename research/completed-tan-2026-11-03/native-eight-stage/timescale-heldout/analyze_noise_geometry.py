#!/usr/bin/env python3
"""Post-execution analytic noise geometry, no new simulations or fitted parameters."""
from pathlib import Path
import json
import numpy as np
P=Path(__file__).resolve().parent
r=json.loads((P/'results.json').read_text());out={'note':'Post-execution reporting only. Full composite OJ, input normalized to domain half-width .5. Sensor covariance in raw voltage units. Singular covariance explicitly treated as noiseless constraint, never regularized. Weak directions in normalized value coordinates have arbitrary sign.','arms':{}}
for name,arm in r['arms'].items():
 a=1-.1/arm['tau_m'];b=1-.1/arm['tau_A'];k=.1/arm['tau_m']*.1
 def d(n):return k*(b**n-a**n)/(b-a) if n else 0.
 J=np.array(arm['J']);O=np.array(arm['O100']);G=O@J*.5;rows=[]
 for label,ss,su,sA in [('sensor_only',1e-6,0.,0.),('membrane_process_only',0.,1e-6,0.),('sensor_and_membrane',1e-6,1e-6,0.),('gate_process_only',0.,0.,1e-6)]:
  variance=np.array([ss**2,ss**2+su**2*sum(a**(2*j) for j in range(100))+sA**2*sum(d(j)**2 for j in range(100))])
  row={'scenario':label,'measurement_noise_covariance':np.diag(variance).tolist()}
  if (variance>0).all():
   W=G/np.sqrt(variance)[:,None];_,sv,V=np.linalg.svd(W)
   row.update({'full_whitened_composite_singular_values':sv.tolist(),'weakest_value_direction':V[-1].tolist(),'condition':float(sv[0]/sv[-1])})
  else:
   direction=np.array([-G[0,1],G[0,0]]);direction/=np.linalg.norm(direction)
   row.update({'noise_free_first_sample_constraint':G[0].tolist(),'remaining_value_direction':direction.tolist(),'remaining_direction_noise_normalized_gain':float(abs(G[1]@direction)/np.sqrt(variance[1]))})
  rows.append(row)
 out['arms'][name]=rows
(P/'noise_geometry.json').write_text(json.dumps(out,indent=2)+'\n')
