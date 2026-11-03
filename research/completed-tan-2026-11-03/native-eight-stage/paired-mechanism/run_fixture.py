#!/usr/bin/env python3
from pathlib import Path
import csv, hashlib, importlib.util, json, platform, sys
import numpy as np
import scipy
BASE=Path(__file__).resolve().parent.parent;OUT=Path(__file__).resolve().parent
HASHES={'models_p1b_v2.py':'6b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f','runner_p1b.py':'47609f1e70ab8de2966ee8305757f44658ef799918d5f92108ac4efe64304c26','summary_p1b.json':'5a19a5b20225fdad51641990b61e189f9b00569abdaf08207c5e2d4941136ff9'}

def main():
 for name,sha in HASHES.items():assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==sha
 spec=importlib.util.spec_from_file_location('reviewed_models',BASE/'models_p1b_v2.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 calibration=json.loads((BASE/'summary_p1b.json').read_text())['calibration']
 original=m.euler_step;trace=[]
 def instrumented(*args,**kwargs):
  result=original(*args,**kwargs);trace.append(list(result));return result
 m.euler_step=instrumented
 runs=[];pairs=[]
 for model in ['M1','M2b']:
  for delay in [0.,5.,10.]:
   members=[]
   for value in [-.5,.5]:
    trace.clear();keys=np.array([.8,0,0,0,1.8]);values=np.array([value,0,0,0,0]);mask=np.array([True,False,False,False,True])
    enc,_,meanA=m.simulate_encoding(keys,values,mask,model,A_0=calibration['A_0'])
    delayed=m.simulate_delay(enc,delay,model,A_0=calibration['A_0'])
    readout=m.simulate_cue_and_readout(delayed,.8,model,**calibration)
    steps=1000+int(round(delay/.1))+100;assert len(trace)==steps
    states=np.array([row[:3] for row in trace]);spikes=[row[3] for row in trace];assert sum(spikes)==0
    name=f'{model}_delay{int(delay)}_value{value:+.1f}'
    with (OUT/f'{name}.csv').open('w',newline='') as f:
     writer=csv.writer(f);writer.writerow(['step','time_ms','phase','u_post','A_post','r_post','spike','u_candidate'])
     for i,row in enumerate(trace):writer.writerow([i+1,(i+1)*.1,'encoding' if i<1000 else ('delay' if i<steps-100 else 'cue'),*row])
    boundaries={'first_event_offset_10ms':states[99].tolist(),'second_event_onset_80ms':states[799].tolist(),'second_event_offset_90ms':states[899].tolist(),'encoding_end_100ms':enc.to_array().tolist(),'cue_onset':delayed.to_array().tolist(),'cue_deadline':states[-1].tolist()}
    row={'name':name,'model':model,'delay_ms':delay,'first_value':value,'boundaries':boundaries,'spikes_all_phases':sum(spikes),'max_u_candidate':max(row[4] for row in trace if row[4] is not None),'readout':readout,'absolute_target_error':abs(readout['v_hat_analog']-value),'delivery_strict_error_lt_0_5':abs(readout['v_hat_analog']-value)<.5}
    runs.append(row);members.append((row,states.copy()))
   minus,plus=members
   boundaries={};exponents={'first_event_offset_10ms':0,'second_event_onset_80ms':700,'second_event_offset_90ms':800,'encoding_end_100ms':900,'cue_onset':900+int(delay/.1),'cue_deadline':1000+int(delay/.1)}
   for key,exponent in exponents.items():
    actual=np.array(plus[0]['boundaries'][key])-np.array(minus[0]['boundaries'][key])
    expected=0. if model=='M1' else calibration['A_0']*(1-.99**100)*.99**exponent
    residual=float(actual[0]-expected)
    assert abs(residual)<1e-15 and actual[1]==actual[2]==0
    boundaries[key]={'actual_delta_state':actual.tolist(),'analytic_delta_u':expected,'residual_u':residual}
   bitwise=minus[1].tobytes()==plus[1].tobytes()
   assert bitwise==(model=='M1')
   delta_prediction=plus[0]['readout']['v_hat_analog']-minus[0]['readout']['v_hat_analog']
   expected_prediction=calibration['g_dec']*boundaries['cue_deadline']['analytic_delta_u']
   assert abs(delta_prediction-expected_prediction)<1e-15
   pairs.append({'model':model,'delay_ms':delay,'full_trajectory_bitwise_identical':bitwise,'boundaries':boundaries,'delta_prediction':delta_prediction,'analytic_delta_prediction':expected_prediction,'prediction_residual':delta_prediction-expected_prediction,'both_members_delivered':minus[0]['delivery_strict_error_lt_0_5'] and plus[0]['delivery_strict_error_lt_0_5']})
 result={'commit':'85213f0d417d54e4f48b301ed1c0e4200ef3c421','source_sha256':HASHES,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'calibration':calibration,'preexecution_protocol_sha256':hashlib.sha256((OUT/'PROTOCOL.md').read_bytes()).hexdigest(),'runs':runs,'pairs':pairs}
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 for pair in pairs:print(pair['model'],pair['delay_ms'],'bitwise_equal',pair['full_trajectory_bitwise_identical'],'delta_vhat',pair['delta_prediction'],'both_delivered',pair['both_members_delivered'])
if __name__=='__main__':main()
