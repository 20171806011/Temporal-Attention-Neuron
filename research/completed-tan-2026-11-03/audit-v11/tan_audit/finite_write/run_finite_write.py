"""Fixed finite-time residual writer; all closed forms are evaluator-only oracles."""
from pathlib import Path
import sys,json,math,hashlib,time,resource,importlib.util,itertools
import numpy as np
OUT=Path(__file__).resolve().parent;REPO=OUT.parent.parent/'Temporal-Attention-Neuron'
sys.path.insert(0,str(REPO/'p1_experiments'))
from task_generator import TaskGenerator
PROTOCOL='01a94f178132b6b6210a8609dfcd4393c22f8c73f847cfddbc23fd410513a7a1'
FIXTURES='a002df84a5dd80430289895ad3a4261a55118cf6ecbd40456956377028f1387d'
MODEL='6b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f'
G=1/(1-.99**100);GF=math.log(24)/(10*.05**2);NF=math.ceil(10*GF*2.2**2)
ARMS={'A100':(100,.1,8000/1937),'FAST10':(NF,10/NF,GF)}

class Writer:
    """Three payload arrays and three counted scratch arrays; one lane per history."""
    __slots__=('K','V','S','phase','scratch')
    def __init__(self,n):
        self.K=np.zeros(n);self.V=np.zeros(n);self.S=np.zeros(n);self.phase=0
        self.scratch=[np.zeros(n) for _ in range(3)]
    def clear_scratch(self):
        for x in self.scratch:x.fill(0.)
    def capture_step(self,k,v,gain):
        assert self.phase==0
        for target,incoming in ((self.K,k),(self.V,v)):
            np.multiply(incoming,gain,out=self.scratch[0])
            np.subtract(self.scratch[0],target,out=self.scratch[0])
            np.multiply(self.scratch[0],.01,out=self.scratch[0])
            np.add(target,self.scratch[0],out=target)
        self.clear_scratch()
    def write_step(self,k,v,alpha):
        assert self.phase==1
        np.subtract(k,self.K,out=self.scratch[0])
        np.subtract(v,self.V,out=self.scratch[1])
        np.multiply(self.scratch[0],self.S,out=self.scratch[2])
        np.subtract(self.scratch[1],self.scratch[2],out=self.scratch[1])
        np.multiply(self.scratch[0],self.scratch[1],out=self.scratch[2])
        np.multiply(alpha,self.scratch[2],out=self.scratch[2])
        np.add(self.S,self.scratch[2],out=self.S)
        self.clear_scratch()
    def query(self,q):
        assert self.phase==2
        np.subtract(q,self.K,out=self.scratch[0])
        np.multiply(self.S,self.scratch[0],out=self.scratch[1])
        np.add(self.V,self.scratch[1],out=self.scratch[2])
        output=self.scratch[2].copy();self.clear_scratch();return output
    def state(self):return np.column_stack((self.K,self.V,self.S))

def serialize(t,variant):
    p=t.event_positions
    return dict(id=f'iid_{t.trial_id}_{variant}',base_id=f'iid_{t.trial_id}',group='iid',variant=variant,positions=p,
        keys=t.keys[p].tolist(),values=t.values[p].tolist(),queries=list(t.query_cues),targets=list(t.target_values),seed_metadata=t.seed)

def variants(rec):
    a=dict(rec,base_id=rec['id'],variant='original')
    q=dict(a,id=rec['id']+'_query_swap',variant='query_swap',queries=rec['queries'][::-1],targets=rec['targets'][::-1])
    v=dict(a,id=rec['id']+'_value_swap',variant='value_swap',values=rec['values'][::-1])
    v['targets']=[v['values'][rec['keys'].index(x)] for x in rec['queries']]
    return [a,q,v]

def execute(records,arm,gain=G,save_trajectories=True):
    n,h,gamma=ARMS[arm];N=len(records);w=Writer(N)
    keys=np.array([r['keys'] for r in records]);vals=np.array([r['values'] for r in records]);q=np.array([r['queries'] for r in records]);targets=np.array([r['targets'] for r in records]);pos=np.array([r['positions'] for r in records])
    selected=np.array([i for i,r in enumerate(records) if r['group']=='adversarial' and r['variant']=='original'],dtype=int)
    extrema_min=np.zeros((N,3));extrema_max=np.zeros((N,3));variation=np.zeros((N,3));integral_sq=np.zeros((N,3));rate_sq=np.zeros((N,3));invalid=np.zeros(N,dtype=bool);roundoff=np.zeros(N,dtype=bool)
    snapshots={};full_capture=[w.state()[selected]];full_write=[]
    def observe(old,dt):
        new=w.state();delta=new-old
        extrema_min[:]=np.minimum(extrema_min,new);extrema_max[:]=np.maximum(extrema_max,new)
        variation[:]+=abs(delta);integral_sq[:]+=old*old*dt;rate_sq[:]+=delta*delta/dt
        lo=np.array([0.,-3.,-128.]);hi=np.array([2.5,3.,128.])
        invalid[:]|=np.any((new<lo-1e-12)|(new>hi+1e-12)|~np.isfinite(new),axis=1)
        roundoff[:]|=np.any((new<lo)|(new>hi),axis=1)
    snapshots['capture_0']=w.state().copy()
    for step in range(1,101):
        old=w.state();w.capture_step(keys[:,0],vals[:,0],gain);observe(old,.1)
        full_capture.append(w.state()[selected])
        if step in (1,50,100):snapshots[f'capture_{step}']=w.state().copy()
    w.phase=1;capture=w.state().copy();capture_errors=capture[:,:2]-np.column_stack((keys[:,0],vals[:,0]))
    if gain==G:assert np.max(abs(capture_errors))<1e-12
    between=20*(pos[:,1]-pos[:,0])-10
    integral_sq[:]+=capture*capture*between[:,None]
    snapshots['write_0']=w.state().copy();full_write.append(w.state()[selected])
    for step in range(1,n+1):
        old=w.state();w.write_step(keys[:,1],vals[:,1],h*gamma);observe(old,h)
        full_write.append(w.state()[selected])
        if step in (1,n//2,n):snapshots[f'write_{step}']=w.state().copy()
    w.phase=2;final=w.state().copy();post=100-(20*pos[:,1]+10);integral_sq[:]+=final*final*post[:,None]
    before_query=final.copy();pred=np.column_stack([w.query(q[:,j]) for j in range(2)]);assert np.array_equal(before_query,w.state())
    assert all(np.all(t==0) for t in w.scratch)
    # Independent evaluator reference only; never passed to the Writer.
    d=keys[:,1]-capture[:,0];e=vals[:,1]-capture[:,1];factor=1-h*gamma*d*d
    oracle=np.zeros(N);continuous=np.zeros(N);nz=d!=0
    oracle[nz]=e[nz]/d[nz]*(1-factor[nz]**n)
    continuous[nz]=e[nz]/d[nz]*(-np.expm1(-gamma*d[nz]*d[nz]*10))
    residual=final[:,2]-oracle;assert np.max(abs(residual))<1e-10
    continuous_pred=capture[:,1,None]+continuous[:,None]*(q-capture[:,0,None])
    if gain==G:
        assert np.all(abs(d)>=.05-1e-12)
        if arm=='FAST10':assert np.all(factor>=0) and np.max(abs(pred-targets))<.25+1e-10
        else:assert np.max(abs(pred-targets))<=6*(1935/1937)**100+1e-10
    raw=pred.copy();pred[invalid]=0
    rows=[]
    for i,r in enumerate(records):
        resources=dict(payload_reals=3,payload_bits=192,scratch_reals=3,scratch_bits=192,stage_bits=2,incoming_raw_ports=2,incoming_port_bits_each=64,query_and_output_bits=64,
            capture_updates=100,write_updates=n,capture_h_ms=.1,write_h_ms=h,write_gamma=gamma,write_gamma_units='1/(key_unit^2*ms)',effective_write_pole=gamma*d[i]**2,
            capture_gain=gain,write_duration_ms=10.,capture_duration_ms=10.,episode_duration_ms=100.,between_events_hold_ms=int(between[i]),postwrite_hold_ms=int(post[i]),
            capture_multiplications=400,capture_subtractions=200,capture_additions=200,write_multiplications=3*n,write_subtractions=3*n,write_additions=n,
            query_multiplications_per_query=1,query_subtractions_per_query=1,query_additions_per_query=1,runtime_data_dependent_divisions=0,
            payload_allocated_real_ms=300.,scratch_allocated_real_ms=300.,scratch_active_real_ms=60.,stage_bit_ms=200.,physical_arithmetic_latency_simulated=False,physical_energy_estimate=None,
            coordinate_min=extrema_min[i].tolist(),coordinate_max=extrema_max[i].tolist(),coordinate_total_variation=variation[i].tolist(),coordinate_square_integrals=integral_sq[i].tolist(),coordinate_rate_square_integrals=rate_sq[i].tolist())
        rows.append(dict(input_id=r['id'],base_id=r['base_id'],group=r['group'],variant=r['variant'],arm=arm,targets=r['targets'],queries=r['queries'],predictions=pred[i].tolist(),raw_predictions=raw[i].tolist(),
            valid=not bool(invalid[i]),status='VALID' if not invalid[i] else 'STATE_RANGE_FAILURE',numerical_boundary_roundoff=bool(roundoff[i]),capture_state=capture[i].tolist(),capture_errors=capture_errors[i].tolist(),final_state=final[i].tolist(),
            final_stage_bits=2,stage_transitions=[0,1,2],scratch_zero_after_updates=True,query_input_fields=['K','V','S','q','shared_constants'],snapshots={k:v[i].tolist() for k,v in snapshots.items()},
            resources=resources,effective_gap=float(d[i]),euler_factor=float(factor[i]),geometric_endpoint_slope=float(oracle[i]),geometric_slope_residual=float(residual[i]),continuous_reference_predictions=continuous_pred[i].tolist(),observed_ecomp=float(max(abs(pred[i]-targets[i])))))
    if save_trajectories:
        np.savez_compressed(OUT/f'{arm}_fixture_trajectories.npz',fixture_ids=np.array([records[i]['id'] for i in selected]),positions=pos[selected],capture=np.array(full_capture),write=np.array(full_write),capture_dt=.1,write_dt=h,between_hold_ms=between[selected],post_hold_ms=post[selected])
    return rows

def native_control(fixtures):
    mp=REPO/'p1_experiments/p1b/models_p1b_v2.py';assert hashlib.sha256(mp.read_bytes()).hexdigest()==MODEL
    spec=importlib.util.spec_from_file_location('finite_native',mp);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    cases=[];traces={}
    for label in ('port_A','port_B'):
        r=next(x for x in fixtures if x['id']==label);k=np.zeros(5);v=np.zeros(5);mask=np.zeros(5,dtype=bool)
        k[r['positions']]=r['keys'];v[r['positions']]=r['values'];mask[r['positions']]=True
        state,trace,_=m.simulate_encoding(k,v,mask,'M1',record_trace=True)
        query=[m.simulate_cue_and_readout(state,x,'M1') for x in r['queries']]
        cases.append(dict(id=label,keys=r['keys'],values=r['values'],queries=r['queries'],targets=r['targets'],content=(k+v).tolist(),novelty=np.maximum(0,k-1.4).tolist(),final_state=state.to_array().tolist(),query_results=query))
        for key,value in trace.items():traces[label+'_'+key]=value
    assert cases[0]['content']==cases[1]['content'] and cases[0]['novelty']==cases[1]['novelty'] and cases[0]['query_results']==cases[1]['query_results']
    for key in ('u','A','r'):assert np.array_equal(traces['port_A_'+key],traces['port_B_'+key])
    np.savez_compressed(OUT/'native_collision_traces.npz',**traces)
    result=dict(cases=cases,all_port_histories_and_state_traces_equal=True,same_queries=True,target_linf_gap=1.8,minimax_pair_error_lower_bound=.9,scope='constructed interface collision, not IID prevalence; default analog coefficients are not a calibrated performance competitor')
    (OUT/'native_interface_control.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

def metrics(rows):
    err=abs(np.array([r['predictions'] for r in rows])-np.array([r['targets'] for r in rows]));valid=np.array([r['valid'] for r in rows]);cref=abs(np.array([r['continuous_reference_predictions'] for r in rows])-np.array([r['targets'] for r in rows]))
    return dict(n=len(rows),coverage=float(valid.mean()),mean_ecomp=float(err.max(1).mean()),mae_per_role=err.mean(0).tolist(),rmse_per_role=np.sqrt((err*err).mean(0)).tolist(),max_error=float(err.max()),strict_0_5_joint=float(np.mean(valid&np.all(err<.5,axis=1))),
        continuous_reference_mean_ecomp=float(cref.max(1).mean()),continuous_reference_max_error=float(cref.max()),max_euler_vs_continuous_prediction_difference=float(np.max(abs(np.array([r['predictions'] for r in rows])-np.array([r['continuous_reference_predictions'] for r in rows])))),
        numerical_boundary_roundoff_rows=sum(r['numerical_boundary_roundoff'] for r in rows),status_counts={s:sum(r['status']==s for r in rows) for s in sorted(set(r['status'] for r in rows))})

def main():
    started=time.perf_counter();assert hashlib.sha256((OUT/'PREREGISTRATION.md').read_bytes()).hexdigest()==PROTOCOL;assert hashlib.sha256((OUT/'FIXTURES.json').read_bytes()).hexdigest()==FIXTURES
    gen=TaskGenerator();rng=np.random.RandomState(2026110217);inputs=[]
    for i in range(512):
        t=gen.generate_single_trial(rng,i,2026110217,N=2)
        inputs.extend([serialize(t,'original'),serialize(gen.create_query_swap_variant(t),'query_swap'),serialize(gen.create_value_swap_variant(t),'value_swap')])
    fixtures=json.loads((OUT/'FIXTURES.json').read_text())
    for r in fixtures:inputs.extend(variants(r))
    ip=OUT/'inputs.jsonl';ip.write_text(''.join(json.dumps(r)+'\n' for r in inputs));ih=hashlib.sha256(ip.read_bytes()).hexdigest();(OUT/'INPUTS_SHA256').write_text(ih+'  inputs.jsonl\n');print('frozen inputs before execution',ih,flush=True)
    rows=[]
    for arm in ARMS:
        rr=execute(inputs,arm);rows.extend(rr);print(arm,'completed',len(rr),'rows',flush=True)
    (OUT/'predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    assert len(inputs)==1611 and len(rows)==3222 and all(r['valid'] for r in rows)
    by={(r['base_id'],r['variant'],r['arm']):r for r in rows}
    for base in set(r['base_id'] for r in rows):
        for arm in ARMS:
            a=by[(base,'original',arm)];q=by[(base,'query_swap',arm)]
            assert a['final_state']==q['final_state'] and a['predictions'][::-1]==q['predictions']
    nc=native_control(fixtures)
    uc=next(r for r in inputs if r['base_id']=='uncalibrated_capture_collision' and r['variant']=='original')
    diagnostic=execute([uc],'FAST10',gain=1.,save_trajectories=False)[0]
    (OUT/'uncalibrated_capture_control.json').write_text(json.dumps(diagnostic,indent=2)+'\n')
    cells=[]
    for group,variant,arm in itertools.product(('iid','adversarial'),('original','query_swap','value_swap'),ARMS):
        rr=[r for r in rows if r['group']==group and r['variant']==variant and r['arm']==arm];cells.append(dict(group=group,variant=variant,arm=arm,**metrics(rr)))
    resources={}
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm and r['group']=='iid' and r['variant']=='original']
        resources[arm]={}
        for key in ('effective_write_pole','coordinate_min','coordinate_max','coordinate_total_variation','coordinate_square_integrals','coordinate_rate_square_integrals'):
            x=np.array([r['resources'][key] for r in rr]);resources[arm][key]=dict(mean=np.mean(x,axis=0).tolist(),min=np.min(x,axis=0).tolist(),max=np.max(x,axis=0).tolist())
    result=dict(protocol_sha256=PROTOCOL,fixtures_sha256=FIXTURES,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),inputs_sha256=ih,model_sha256=MODEL,generator_sha256=hashlib.sha256((REPO/'p1_experiments/task_generator.py').read_bytes()).hexdigest(),
        fresh_histories=512,adversarial_histories=25,inputs=len(inputs),prediction_rows=len(rows),role_outputs=2*len(rows),capture_gain=G,arms={k:dict(write_updates=v[0],h=v[1],gamma=v[2]) for k,v in ARMS.items()},metrics=cells,resources=resources,
        validation=dict(all_rows_valid=True,query_swap_exact=True,max_capture_abs_error=max(max(abs(x) for x in r['capture_errors']) for r in rows),max_geometric_slope_residual=max(abs(r['geometric_slope_residual']) for r in rows),native_interface_collision=nc['all_port_histories_and_state_traces_equal']),
        uncalibrated_fixture_ecomp=diagnostic['observed_ecomp'],wall_seconds=time.perf_counter()-started,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitation='Enriched raw ports, three payload plus three scratch reals, calibrated capture, gated perfect hold, high gain/update rate and algebraic readout. No physical bandwidth/energy/noise validation; original labels unchanged.')
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
