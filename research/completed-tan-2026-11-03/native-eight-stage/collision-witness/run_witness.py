#!/usr/bin/env python3
from fractions import Fraction as Q
from pathlib import Path
import csv, hashlib, importlib.util, json, platform, sys
import numpy as np
import scipy
BASE=Path(__file__).resolve().parent.parent;OUT=Path(__file__).resolve().parent
MODEL_HASH='6b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f'

def exact_coefficients(model,A0,keys):
    A=A0 if model=='M2b' else Q(0);b=c1=c2=Q(0);gates=[]
    for n in range(1000):
        i=0 if n<100 else (1 if 800<=n<900 else None)
        key=Q(0) if i is None else keys[i]
        gates.append(A)
        b=Q(99,100)*b+Q(1,100)*A*key
        c1=Q(99,100)*c1+(Q(1,100)*A if i==0 else 0)
        c2=Q(99,100)*c2+(Q(1,100)*A if i==1 else 0)
        A=A0 if model=='M2b' else Q(149,150)*A+max(Q(0),key-Q(7,5))/150
    R=c1/c2
    assert b-c1+c2*R==b+c1-c2*R==b
    if model=='M2b':assert R==Q(99,100)**800
    current_max=max(Q(0),*(gates[n]*(keys[i]+v) for i,steps in [(0,range(100)),(1,range(800,900))] for n in steps for v in ([-1,1] if i==0 else [-R,R])))
    assert current_max<1 and 0<R<3
    return {'R':R,'base_u':b,'c1':c1,'c2':c2,'max_A':max(gates),'max_syn_current':current_max}

def main():
 assert hashlib.sha256((BASE/'models_p1b_v2.py').read_bytes()).hexdigest()==MODEL_HASH
 assert hashlib.sha256((BASE/'summary_p1b.json').read_bytes()).hexdigest()=='5a19a5b20225fdad51641990b61e189f9b00569abdaf08207c5e2d4941136ff9'
 spec=importlib.util.spec_from_file_location('collision_models',BASE/'models_p1b_v2.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 cal=json.loads((BASE/'summary_p1b.json').read_text())['calibration'];trace=[];original=m.euler_step
 def record(*args,**kwargs):
  res=original(*args,**kwargs);trace.append(list(res));return res
 m.euler_step=record
 output={'commit':'85213f0d417d54e4f48b301ed1c0e4200ef3c421','model_sha256':MODEL_HASH,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'calibration':cal,'cases':[]}
 for model,keys in [('M2b',[Q(4,5),Q(9,5)]),('M1',[Q(9,5),Q(11,5)])]:
  analytic=exact_coefficients(model,Q(str(cal['A_0'])),keys);R=analytic['R'];runs=[]
  # Hex exact fraction serialization avoids decimal string digit limits.
  (OUT/f'{model}_exact_ratio.json').write_text(json.dumps({'numerator_hex':hex(R.numerator),'denominator_hex':hex(R.denominator)},indent=2)+'\n')
  for sign in [-1,1]:
   trace.clear();k=np.array([float(keys[0]),0,0,0,float(keys[1])]);v=np.array([float(sign),0,0,0,-sign*float(R)]);mask=np.array([1,0,0,0,1],dtype=bool)
   state,_,_=m.simulate_encoding(k,v,mask,model,A_0=cal['A_0']);encoding_trace=list(trace)
   assert sum(row[3] for row in encoding_trace)==0 and state.r==0
   readout=m.simulate_cue_and_readout(state,float(keys[0]),model,**cal)
   with (OUT/f'{model}_{sign:+d}_trace.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['step','phase','u_post','A_post','r_post','spike','u_candidate']);w.writerows([i+1,'encoding' if i<1000 else 'cue',*row] for i,row in enumerate(trace))
   runs.append({'sign':sign,'values':v.tolist(),'encoding_final_state':state.to_array().tolist(),'encoding_spikes':sum(row[3] for row in encoding_trace),'encoding_max_u_candidate':max(row[4] for row in encoding_trace),'cue_total_spikes':sum(readout['raw_spikes']),'readout':readout,'first_target_error':abs(readout['v_hat_analog']-sign)})
  minus,plus=runs;u1=minus['encoding_final_state'][0];u2=plus['encoding_final_state'][0]
  delta=np.array(plus['encoding_final_state'])-np.array(minus['encoding_final_state'])
  case={'model':model,'keys':list(map(float,keys)),'analytic':{k:float(v) for k,v in analytic.items()},'exact_rational_cancellation_verified':True,'encoding_threshold_excluded_by_exact_current_bound':True,'runs':runs,'floating_delta_state':delta.tolist(),'floating_state_bytes_equal':np.array(minus['encoding_final_state']).tobytes()==np.array(plus['encoding_final_state']).tobytes(),'u_residual_in_ulps':float(abs(u2-u1)/abs(np.spacing(u1))),'floating_delta_prediction':plus['readout']['v_hat_analog']-minus['readout']['v_hat_analog']}
  output['cases'].append(case);print(model,'R',float(R),'delta',delta,'ULPs',case['u_residual_in_ulps'],'cue spikes',[x['cue_total_spikes'] for x in runs])
 (OUT/'results.json').write_text(json.dumps(output,indent=2)+'\n')
if __name__=='__main__':main()
