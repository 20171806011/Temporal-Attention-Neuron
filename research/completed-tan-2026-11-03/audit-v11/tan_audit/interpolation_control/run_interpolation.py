"""Explicit idealized online two-register write; no cache at query time."""
from pathlib import Path
from fractions import Fraction as F
import sys,json,hashlib,itertools,math,time,resource
import numpy as np
OUT=Path(__file__).resolve().parent;AUDIT=OUT.parent;REPO=AUDIT.parent/'Temporal-Attention-Neuron'
sys.path[:0]=[str(REPO/'p1_experiments'),str(REPO/'p1_experiments/p1b'),str(AUDIT/'binding_control')]
from task_generator import TaskGenerator
from models_p1b_v2 import euler_step
from run_control import encoding as modified_encoding, inverse as privileged_inverse
SCENARIOS=['clean_float64','coefficient_fullscale_noise','query_key_noise','coefficient_plus_query','write_and_query_key_noise','float32','hold_delay10','common_post_leak10','differential_post_leak10','interwrite_leak150']
M0_SCENARIOS=SCENARIOS[:7]
LAM_STEP=math.exp(-.1/150)

class OnlineWriter:
    __slots__=('x','y','phase','dtype')
    def __init__(self,dtype=float):self.x=dtype(0);self.y=dtype(0);self.phase=0;self.dtype=dtype
    def write(self,k,v):
        T=self.dtype;k=T(k);v=T(v)
        if self.phase==0:self.x,self.y,self.phase=k,v,1;return
        assert self.phase==1
        oldk,oldv=self.x,self.y # combinational old-register wires, not retained state
        denominator=T(k-oldk)
        if abs(float(denominator))<=1e-12:self.x=self.y=T(0);self.phase=3;return
        a=T(T(v-oldv)/denominator);b=T(oldv-T(a*oldk))
        self.x,self.y,self.phase=a,b,2
    def tick(self,leak):
        if leak and self.phase==1:self.x*=LAM_STEP;self.y*=LAM_STEP

def query_two_registers(state,q,dtype=float):
    # Explicitly only 2 real values and current q; no keys/timing/target inputs.
    T=dtype;return float(T(T(T(state[0])*T(q))+T(state[1])))

def write_online(rec,scenario,z):
    dtype=np.float32 if scenario=='float32' else float;w=OnlineWriter(dtype);transitions=[]
    keys=np.array(rec['keys'],dtype=float)
    if scenario=='write_and_query_key_noise':keys=keys+1e-4*np.array(z['writekeys'])
    for step in range(1000):
        slot,within=divmod(step,200)
        if within==0 and slot in rec['positions']:
            j=rec['positions'].index(slot);before=[float(w.x),float(w.y)]
            w.write(keys[j],rec['values'][j]);transitions.append(dict(step=step,phase=w.phase,before=before,after=[float(w.x),float(w.y)],write_key=float(keys[j]),write_value=rec['values'][j]))
        w.tick(scenario=='interwrite_leak150')
    return [float(w.x),float(w.y)],w.phase,transitions

def coef_reference(rec):
    k=rec['keys'];v=rec['values'];a=(v[1]-v[0])/(k[1]-k[0]);return np.array([a,v[0]-a*k[0]])

def coefficient_range(state):
    finite=all(math.isfinite(x) for x in state)
    valid=finite and abs(state[0])<=120*(1+1e-10) and abs(state[1])<=297*(1+1e-10)
    return valid,finite and (abs(state[0])>120 or abs(state[1])>297)

def interpolation(rec,scenario,z):
    state,phase,trans=write_online(rec,scenario,z);base=state.copy();q=np.array(rec['queries'],dtype=float)
    if scenario in ['coefficient_fullscale_noise','coefficient_plus_query']:state=(np.array(state)+np.array([120,297])*.001*np.array(z['state'][:2])).tolist()
    if scenario in ['query_key_noise','coefficient_plus_query','write_and_query_key_noise']:q+=1e-4*np.array(z['query'])
    if scenario=='common_post_leak10':state=(np.array(state)*math.exp(-10/150)).tolist()
    if scenario=='differential_post_leak10':state=(np.array(state)*np.array([math.exp(-10/150),math.exp(-10/100)])).tolist()
    if scenario=='float32':q=np.array(q,dtype=np.float32).astype(float)
    range_ok,roundoff=coefficient_range(state);valid=phase==2 and range_ok
    status='WRITE_SINGULAR' if phase!=2 else 'VALID' if range_ok else 'RANGE_EXCEEDED'
    dtype=np.float32 if scenario=='float32' else float
    raw=[query_two_registers(state,x,dtype) for x in q] if phase==2 else None
    pred=raw if valid else [0.,0.]
    row=dict(state=state,unperturbed_written_state=base,phase=phase,write_transitions=trans,queries=q.tolist(),
        raw_predictions=raw,predictions=pred,valid=valid,status=status,numerical_range_band=roundoff,
        query_input_fields=['two_register_state','q'],persistent_payload_reals=2,phase_bits=2,
        post_delay_ms=10. if scenario in ['hold_delay10','common_post_leak10','differential_post_leak10'] else 0.)
    target=np.array(rec['targets']);ref=coef_reference(rec);trueq=np.array(rec['queries']);ea=eb=eq=0.
    if scenario in ['coefficient_fullscale_noise','coefficient_plus_query']:ea=.12;eb=.297
    if scenario in ['query_key_noise','coefficient_plus_query']:eq=1e-4
    if scenario in ['coefficient_fullscale_noise','query_key_noise','coefficient_plus_query']:
        bound=2.5*ea+eb+(120+ea)*eq
        actual=float(np.max(abs(np.array(raw)-target)));assert actual<=bound+1e-10
        row['raw_error_bound']=bound;row['raw_bound_pass']=True
    if scenario=='write_and_query_key_noise' and phase==2:
        bound=6*(2e-4)/(.05-2e-4);actual=float(np.max(abs(np.array(raw)-target)));assert actual<=bound+1e-10
        row['raw_error_bound']=bound;row['raw_bound_pass']=True
    if scenario in ['common_post_leak10','differential_post_leak10']:
        la=math.exp(-10/150);lb=la if scenario=='common_post_leak10' else math.exp(-10/100)
        expected=(la-lb)*ref[0]*trueq+(lb-1)*target
        err=np.array(raw)-target;assert np.max(abs(err-expected))<=1e-10
        bound=300*abs(la-lb)+3*abs(1-lb);assert max(abs(err))<=bound+1e-10
        row['raw_error_bound']=bound;row['raw_bound_pass']=True
    return row

def cache_control(rec,scenario,z):
    k=np.array(rec['keys']);v=np.array(rec['values']);q=np.array(rec['queries'])
    if scenario in ['coefficient_fullscale_noise','coefficient_plus_query']:
        k+=1.1*.001*np.array(z['state'][:2]);v+=3*.001*np.array(z['state'][2:])
    if scenario=='write_and_query_key_noise':k+=1e-4*np.array(z['writekeys'])
    if scenario in ['query_key_noise','coefficient_plus_query','write_and_query_key_noise']:q+=1e-4*np.array(z['query'])
    if scenario=='float32':k=k.astype(np.float32).astype(float);v=v.astype(np.float32).astype(float);q=q.astype(np.float32).astype(float)
    valid=bool(np.all(k>=.3-2.5e-10) and np.all(k<=2.5+2.5e-10) and np.all(v>=-3-3e-10) and np.all(v<=3+3e-10))
    raw=[float(v[np.argmin((k-x)**2)]) for x in q]
    return dict(state=dict(keys=k.tolist(),values=v.tolist()),queries=q.tolist(),raw_predictions=raw,predictions=raw if valid else [0.,0.],
        valid=valid,status='VALID' if valid else 'RANGE_EXCEEDED',persistent_payload_reals=4,phase_bits=2,query_input_fields=['cached_keys','cached_values','q'])

def modified_controls(rec):
    state,resources=modified_encoding(rec['keys'],rec['values'],tuple(rec['positions']))
    preds=[];probes=[]
    for q in rec['queries']:
        u,A,r=state;spikes=[];last=None;oldr=None
        for step in range(100):
            oldr=r;u,A,r,sp,last=euler_step(u,A,r,0.,0.,q,tau_A=150.);spikes.append(sp)
        read=0. if oldr>0 else last
        preds.append(.1326308169898006*read-.02681648355488859);probes.append(dict(read=read,eligible_count=sum(spikes[:99])))
    native=dict(predictions=preds,raw_predictions=preds,valid=True,status='VALID',state=state.tolist(),queries=rec['queries'],
        probes=probes,encoding_resources=resources,query_input_fields=['state_u_A_r','q'],persistent_payload_reals=3,physical_query_steps_per_role=100)
    f=dict(state=state.tolist(),keys=rec['keys'],positions=rec['positions'],queries=rec['queries']);p=privileged_inverse(f)
    p['persistent_payload_reals']=3;p['extra_key_reals']=2;p['extra_slot_indices']=2;p['raw_predictions']=p['predictions']
    return native,p

def from_trial(t,group='iid',variant='original'):
    pos=t.event_positions
    return dict(id=f'{group}_{t.trial_id}_{variant}',group=group,variant=variant,trial_id=t.trial_id,positions=list(pos),
        keys=t.keys[pos].tolist(),values=t.values[pos].tolist(),queries=list(t.query_cues),targets=list(t.target_values),
        seed_metadata=t.seed,rejections=t.rejection_count)

def adversarial():
    upper=np.nextafter(2.5,-np.inf);pairs=[(.3,np.nextafter(.35,np.inf)),(np.nextafter(upper-.05,-np.inf),upper)]
    rows=[]
    for pair,reverse,sign,pos in itertools.product(pairs,[False,True],[False,True],[(0,4),(3,4)]):
        k=list(pair[::-1] if reverse else pair);v=[-3.,float(np.nextafter(3.,-np.inf))];v=v[::-1] if sign else v
        rows.append(dict(id=f'adv_{len(rows)}',group='adversarial',variant='original',positions=list(pos),keys=k,values=v,queries=k.copy(),targets=v.copy(),kind='minimum_spacing'))
    rows.append(dict(id='adv_constant',group='adversarial',variant='original',positions=[0,4],keys=[1.,float(np.nextafter(1.05,np.inf))],values=[1.25,1.25],queries=[1.,float(np.nextafter(1.05,np.inf))],targets=[1.25,1.25],kind='constant_values'))
    k2=2.
    for _ in range(200):k2*=LAM_STEP
    rows.append(dict(id='adv_prewrite_singular',group='adversarial',variant='original',positions=[0,1],keys=[2.,k2],values=[1.,-1.],queries=[2.,k2],targets=[1.,-1.],kind='prewrite_leak_singularity'))
    for r in rows:
        assert .3<=min(r['keys']) and max(r['keys'])<2.5 and abs(r['keys'][0]-r['keys'][1])>=.05
        assert all(-3<=v<3 for v in r['values'])
    return rows

def rational_check(rec):
    k=list(map(F.from_float,rec['keys']));v=list(map(F.from_float,rec['values']));a=(v[1]-v[0])/(k[1]-k[0]);b=v[0]-a*k[0]
    assert abs(a)<=120 and abs(b)<=297
    for kk,vv in zip(k,v):assert a*kk+b==vv
    records=[]
    for dtype,name in [(float,'float64'),(np.float32,'float32')]:
        w=OnlineWriter(dtype)
        for kk,vv in zip(rec['keys'],rec['values']):w.write(kk,vv)
        aa=F.from_float(float(w.x));bb=F.from_float(float(w.y));ea=abs(aa-a);eb=abs(bb-b)
        er=[];rounding=[];bounds=[]
        for q,target in zip(rec['queries'],rec['targets']):
            qf=float(dtype(q));qq=F.from_float(qf);actual=F.from_float(query_two_registers([w.x,w.y],qf,dtype))
            ideal_stored=aa*qq+bb;ro=abs(actual-ideal_stored);eq=abs(qq-F.from_float(q))
            bound=abs(F.from_float(q))*ea+eb+(abs(a)+ea)*eq+ro
            error=abs(actual-F.from_float(target));assert error<=bound
            er.append(float(error));rounding.append(float(ro));bounds.append(float(bound))
        records.append(dict(dtype=name,delta_a=float(ea),delta_b=float(eb),absolute_errors=er,readout_rounding=rounding,certified_bounds=bounds))
    return dict(id=rec['id'],exact_retrieval=True,exact_a=str(a),exact_b=str(b),float_checks=records)

def metrics(rows):
    e=abs(np.array([r['predictions'] for r in rows])-np.array([r['targets'] for r in rows]));valid=np.array([r['valid'] for r in rows])
    rawvalid=[r for r in rows if r.get('raw_predictions') is not None]
    return dict(n=len(rows),coverage=float(valid.mean()),mean_e_comp=float(e.max(axis=1).mean()),max_abs_error=float(e.max()),mae_per_role=e.mean(axis=0).tolist(),
        rmse_per_role=np.sqrt((e*e).mean(axis=0)).tolist(),valid_joint_0_5=float(np.mean(valid&np.all(e<.5,axis=1))),local_valid_joint_0_05=float(np.mean(valid&np.all(e<.05,axis=1))),
        status_counts={s:sum(r['status']==s for r in rows) for s in sorted(set(r['status'] for r in rows))},
        raw_finite_mean_e_comp=float(np.mean([max(abs(np.array(r['raw_predictions'])-r['targets'])) for r in rawvalid])) if rawvalid else None)

def main():
    start=time.perf_counter();gen=TaskGenerator();rng=np.random.RandomState(2026110213);base=[];variants=[]
    for i in range(512):
        t=gen.generate_single_trial(rng,i,2026110213,N=2);base.append(from_trial(t))
        variants +=[from_trial(gen.create_query_swap_variant(t),variant='query_swap'),from_trial(gen.create_value_swap_variant(t),variant='value_swap')]
    adv=adversarial();items=base+adv;noise_rng=np.random.RandomState(2026110214)
    zs=noise_rng.uniform(-1,1,(len(items),4));zw=noise_rng.uniform(-1,1,(len(items),2));zq=noise_rng.uniform(-1,1,(len(items),2))
    noises={r['id']:dict(state=zs[i].tolist(),writekeys=zw[i].tolist(),query=zq[i].tolist()) for i,r in enumerate(items)}
    ip=OUT/'inputs.jsonl';ip.write_text(''.join(json.dumps(r)+'\n' for r in items+variants));nh=OUT/'noise.json';nh.write_text(json.dumps(noises)+'\n')
    ih=hashlib.sha256(ip.read_bytes()).hexdigest();(OUT/'INPUTS_SHA256').write_text(ih+'  inputs.jsonl\n');print('frozen inputs',ih,flush=True)
    exact=[rational_check(r) for r in base[:16]+adv];(OUT/'exact_and_precision_checks.json').write_text(json.dumps(exact,indent=2)+'\n')
    rows=[]
    def add(rec,arm,case,out):rows.append(dict(id=rec['id'],group=rec['group'],variant=rec['variant'],arm=arm,scenario=case,targets=rec['targets'],**out))
    for rec in items:
        z=noises[rec['id']]
        for case in SCENARIOS:add(rec,'interpolation',case,interpolation(rec,case,z))
        for case in M0_SCENARIOS:add(rec,'M0_cache',case,cache_control(rec,case,z))
        native,priv=modified_controls(rec);add(rec,'modified_native_analog','clean_float64',native);add(rec,'modified_privileged_inverse','clean_float64',priv)
    zero=dict(state=[0.]*4,writekeys=[0.]*2,query=[0.]*2)
    for rec in variants:
        add(rec,'interpolation','clean_float64',interpolation(rec,'clean_float64',zero));add(rec,'M0_cache','clean_float64',cache_control(rec,'clean_float64',zero))
        native,priv=modified_controls(rec);add(rec,'modified_native_analog','clean_float64',native);add(rec,'modified_privileged_inverse','clean_float64',priv)
    bykey={(r['id'],r['arm'],r['scenario']):r for r in rows}
    for i in range(512):
        for arm in ['interpolation','M0_cache','modified_native_analog','modified_privileged_inverse']:
            a=bykey[(f'iid_{i}_original',arm,'clean_float64')];q=bykey[(f'iid_{i}_query_swap',arm,'clean_float64')]
            assert a['status']==q['status'] and max(abs(np.array(a['predictions'])[::-1]-q['predictions']))<1e-10
    for r in rows:
        if r['arm']=='M0_cache' and r['scenario']=='clean_float64':assert r['predictions']==r['targets'] and r['valid']
        if r['arm']=='interpolation' and r['scenario']=='clean_float64':assert r['valid'] and max(abs(np.array(r['predictions'])-r['targets']))<1e-10
    singular=bykey[('adv_prewrite_singular','interpolation','interwrite_leak150')];assert singular['status']=='WRITE_SINGULAR'
    table=[]
    for key in sorted(set((r['group'],r['variant'],r['arm'],r['scenario']) for r in rows)):
        rr=[r for r in rows if (r['group'],r['variant'],r['arm'],r['scenario'])==key]
        table.append(dict(zip(['group','variant','arm','scenario'],key),**metrics(rr)))
    (OUT/'predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    result=dict(protocol_sha256=hashlib.sha256((OUT/'PREREGISTRATION.md').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs_sha256=ih,
        noise_sha256=hashlib.sha256(nh.read_bytes()).hexdigest(),target_seed=2026110213,noise_seed=2026110214,iid_histories=512,adversarial_histories=18,clean_dependent_variants=1024,
        prediction_rows=len(rows),role_predictions=2*len(rows),metrics=table,exact_precision_histories=len(exact),raw_bounds_all_pass=True,
        validation=dict(M0_clean_exact=True,clean_interpolation_error_below_1e_10=True,query_swaps=True,prewrite_leak_singularity_status=singular['status']),
        resources=dict(interpolation_persistent_reals=2,controller_phase_states=4,controller_bits=2,phase_states_description=['empty','first','ready','error'],
            separate_raw_key_value_ports=True,event_onset_strobe=True,trial_reset=True,external_clock=True,write_arithmetic_second_event={'subtractions':3,'divisions':1,'multiplications':1},
            query_arithmetic={'multiplications':1,'additions':1},atomic_two_register_overwrite=True,combinational_temporaries=True,ideal_analog_output=True,
            raw_register_fullscales=[120,297],m0_persistent_reals=4,m0_key_center=1.4,m0_key_fullscale=1.1,m0_value_fullscale=3,
            physical_neuron_equivalence=False,energy_estimate=None,host_wall_seconds=time.perf_counter()-start,host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        caveats=['Standard two-point interpolation, new control relative to inspected repository only.','Raw ports/write arithmetic/phase switch/ideal readout differ from TAN.','Normalized coordinate noise is an engineering assumption, not equal biological noise or energy.','No N3 or globalneuron claim.'])
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'rows':len(rows),'metrics':table,'resources':result['resources']},indent=2))

if __name__=='__main__':main()
