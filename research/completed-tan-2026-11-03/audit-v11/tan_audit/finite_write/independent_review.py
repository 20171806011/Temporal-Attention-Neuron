"""Independent generator, geometric trajectory and resource audit; no primary imports."""
from pathlib import Path
from fractions import Fraction as F
import json, hashlib, ast, math, time, itertools
import numpy as np
P=Path(__file__).resolve().parent

def near(a,b,atol=1e-10,rtol=1e-11):
    assert np.allclose(a,b,atol=atol,rtol=rtol),(a,b)

def main():
    started=time.perf_counter()
    names=['PREREGISTRATION.md','FIXTURES.json','run_finite_write.py','inputs.jsonl','predictions.jsonl','results.json','native_interface_control.json','uncalibrated_capture_control.json','A100_fixture_trajectories.npz','FAST10_fixture_trajectories.npz','native_collision_traces.npz']
    before={x:hashlib.sha256((P/x).read_bytes()).hexdigest() for x in names}
    result=json.loads((P/'results.json').read_text());inp=[json.loads(s) for s in (P/'inputs.jsonl').read_text().splitlines()];rows=[json.loads(s) for s in (P/'predictions.jsonl').read_text().splitlines()]
    assert len(inp)==1611 and len(rows)==3222 and len({r['id'] for r in inp})==1611
    for field,name in [('protocol_sha256','PREREGISTRATION.md'),('fixtures_sha256','FIXTURES.json'),('script_sha256','run_finite_write.py'),('inputs_sha256','inputs.jsonl')]:assert result[field]==before[name]
    # Replicate the inspected generator's random call order independently.
    rng=np.random.RandomState(2026110217);expected=[];rejected=0
    for trial in range(512):
        while True:
            keys=np.sort(rng.uniform(.3,2.5,size=2))
            if keys[1]-keys[0]>=.05:break
            rejected+=1
        keys=rng.permutation(keys).tolist();values=rng.uniform(-3,3,size=2).tolist();pos=sorted(rng.choice(5,size=2,replace=False).tolist());targetpos=rng.choice(pos,size=2,replace=False).tolist();qi=[pos.index(j) for j in targetpos]
        for variant in ('original','query_swap','value_swap'):
            vv=values[::-1] if variant=='value_swap' else values;qq=qi[::-1] if variant=='query_swap' else qi
            expected.append(dict(id=f'iid_{trial}_{variant}',base_id=f'iid_{trial}',group='iid',variant=variant,positions=pos,keys=keys,values=vv,queries=[keys[j] for j in qq],targets=[vv[j] for j in qq],seed_metadata=2026110217))
    assert inp[:1536]==expected
    fixtures=json.loads((P/'FIXTURES.json').read_text());assert len(fixtures)==25
    for f in fixtures:
        a=dict(f,base_id=f['id'],variant='original')
        q=dict(a,id=f['id']+'_query_swap',variant='query_swap',queries=f['queries'][::-1],targets=f['targets'][::-1])
        v=dict(a,id=f['id']+'_value_swap',variant='value_swap',values=f['values'][::-1]);v['targets']=[v['values'][f['keys'].index(t)] for t in f['queries']]
        expected.extend([a,q,v])
    assert inp==expected
    for r in inp:
        assert len(set(r['positions']))==2 and sorted(r['positions'])==r['positions'] and all(0<=p<5 for p in r['positions'])
        assert all(.3<=k<2.5 for k in r['keys']) and abs(r['keys'][1]-r['keys'][0])>=.05
        assert all(-3<=v<3 for v in r['values']) and sorted(r['queries'])==sorted(r['keys'])
        assert r['targets']==[r['values'][r['keys'].index(q)] for q in r['queries']]
    N=len(inp);keys=np.array([r['keys'] for r in inp]);values=np.array([r['values'] for r in inp]);queries=np.array([r['queries'] for r in inp]);targets=np.array([r['targets'] for r in inp]);positions=np.array([r['positions'] for r in inp])
    K=np.zeros(N);V=np.zeros(N);G=1/(1-.99**100);capture=[np.column_stack((K,V,np.zeros(N)))];capture_integral=np.zeros((N,3));capture_rate=np.zeros((N,3));boundary=np.zeros(N,dtype=bool)
    for j in range(100):
        previous=np.column_stack((K,V,np.zeros(N)))
        K=K+.01*(G*keys[:,0]-K);V=V+.01*(G*values[:,0]-V)
        current=np.column_stack((K,V,np.zeros(N)));capture.append(current)
        capture_integral+=.1*previous**2;capture_rate+=(current-previous)**2/.1
        boundary|=(K>2.5)|(K<0)|(V>3)|(V<-3)
    capture=np.array(capture);near(K,keys[:,0],atol=1e-12);near(V,values[:,0],atol=1e-12)
    gf=math.log(24)/(10*.05**2);nf=math.ceil(10*gf*2.2**2);assert nf==6153
    arms={'A100':(100,.1,8000/1937),'FAST10':(nf,10/nf,gf)}
    indexed={(r['input_id'],r['arm']):r for r in rows};assert len(indexed)==3222
    max_endpoint=0.;max_trajectory=0.;max_resources=0.;resource_fields_checked=0
    snapshots_checked=0;fixture_rows=0
    for arm,(n,h,gamma) in arms.items():
        d=keys[:,1]-K;e=values[:,1]-V;r=1-h*gamma*d*d;sstar=e/d
        Sn=sstar*(1-r**n);state=np.column_stack((K,V,Sn));pred=V[:,None]+Sn[:,None]*(queries-K[:,None]);continuous=V[:,None]+(sstar*(-np.expm1(-10*gamma*d*d)))[:,None]*(queries-K[:,None])
        sump=(1-r**n)/(1-r);sum2=(1-r**(2*n))/(1-r*r)
        ssint=h*sstar*sstar*(n-2*sump+sum2)
        srint=sstar*sstar*(1-r)**2*sum2/h
        svar=np.where(r>=0,abs(Sn),abs(sstar)*(1-r)*(1-abs(r)**n)/(1-abs(r)))
        peak=np.where(r>=0,Sn,sstar*(1-r))
        mins=np.column_stack((np.zeros(N),np.minimum(0,V),np.minimum(0,peak)))
        maxs=np.column_stack((K,np.maximum(0,V),np.maximum(0,peak)))
        integral=capture_integral.copy();integral[:,0]+=K*K*(90-20*positions[:,0]);integral[:,1]+=V*V*(90-20*positions[:,0]);integral[:,2]=ssint+Sn*Sn*(90-20*positions[:,1])
        rate=capture_rate.copy();rate[:,2]=srint
        variation=np.column_stack((abs(K),abs(V),svar))
        for i,rec in enumerate(inp):
            x=indexed[(rec['id'],arm)];assert x['valid'] and x['status']=='VALID' and x['final_stage_bits']==2 and x['stage_transitions']==[0,1,2]
            assert x['targets']==rec['targets'] and x['queries']==rec['queries'] and x['query_input_fields']==['K','V','S','q','shared_constants']
            near(x['capture_state'],capture[-1,i]);near(x['capture_errors'],[K[i]-keys[i,0],V[i]-values[i,0]])
            near(x['final_state'],state[i]);near(x['predictions'],pred[i]);near(x['raw_predictions'],pred[i]);near(x['continuous_reference_predictions'],continuous[i]);near(x['geometric_endpoint_slope'],Sn[i])
            max_endpoint=max(max_endpoint,float(np.max(abs(np.array(x['final_state'])-state[i]))))
            near(x['geometric_slope_residual'],x['final_state'][2]-Sn[i]);near(x['effective_gap'],d[i]);near(x['euler_factor'],r[i]);near(x['observed_ecomp'],max(abs(np.array(x['predictions'])-targets[i])))
            assert x['numerical_boundary_roundoff']==bool(boundary[i]) and x['scratch_zero_after_updates']
            for j in (0,1,50,100):near(x['snapshots'][f'capture_{j}'],capture[j,i]);snapshots_checked+=1
            for j in (0,1,n//2,n):near(x['snapshots'][f'write_{j}'],[K[i],V[i],sstar[i]*(1-r[i]**j)]);snapshots_checked+=1
            rr=x['resources'];expected_resources=dict(payload_reals=3,payload_bits=192,scratch_reals=3,scratch_bits=192,stage_bits=2,incoming_raw_ports=2,incoming_port_bits_each=64,query_and_output_bits=64,capture_updates=100,write_updates=n,capture_h_ms=.1,write_h_ms=h,write_gamma=gamma,write_gamma_units='1/(key_unit^2*ms)',effective_write_pole=gamma*d[i]**2,capture_gain=G,write_duration_ms=10.,capture_duration_ms=10.,episode_duration_ms=100.,between_events_hold_ms=int(20*(positions[i,1]-positions[i,0])-10),postwrite_hold_ms=int(90-20*positions[i,1]),capture_multiplications=400,capture_subtractions=200,capture_additions=200,write_multiplications=3*n,write_subtractions=3*n,write_additions=n,query_multiplications_per_query=1,query_subtractions_per_query=1,query_additions_per_query=1,runtime_data_dependent_divisions=0,payload_allocated_real_ms=300.,scratch_allocated_real_ms=300.,scratch_active_real_ms=60.,stage_bit_ms=200.,physical_arithmetic_latency_simulated=False,physical_energy_estimate=None)
            for k,v in expected_resources.items():
                if isinstance(v,(float,np.floating)):near(rr[k],v)
                else:assert rr[k]==v,(k,rr[k],v)
                resource_fields_checked+=1
            for k,v in [('coordinate_min',mins[i]),('coordinate_max',maxs[i]),('coordinate_total_variation',variation[i]),('coordinate_square_integrals',integral[i]),('coordinate_rate_square_integrals',rate[i])]:
                near(rr[k],v,atol=1e-8,rtol=1e-10);max_resources=max(max_resources,float(np.max(abs(np.array(rr[k])-v))));resource_fields_checked+=1
        z=np.load(P/f'{arm}_fixture_trajectories.npz');ids=z['fixture_ids'].tolist();assert ids==[x['id'] for x in inp if x['group']=='adversarial' and x['variant']=='original'];sel=[next(i for i,x in enumerate(inp) if x['id']==id_) for id_ in ids]
        near(z['capture'],capture[:,sel]);near(z['positions'],positions[sel]);near(z['capture_dt'],.1);near(z['write_dt'],h);near(z['between_hold_ms'],20*(positions[sel,1]-positions[sel,0])-10);near(z['post_hold_ms'],90-20*positions[sel,1])
        curve=sstar[sel][None,:]*(1-r[sel][None,:]**np.arange(n+1)[:,None]);near(z['write'][:,:,0],np.broadcast_to(K[sel],curve.shape));near(z['write'][:,:,1],np.broadcast_to(V[sel],curve.shape));near(z['write'][:,:,2],curve)
        max_trajectory=max(max_trajectory,float(np.max(abs(z['write'][:,:,2]-curve))));fixture_rows+=len(ids)
        if arm=='FAST10':assert np.all(r>=0) and np.max(abs(pred-targets))<=.25+1e-10
    # Every metric cell and saved aggregate is recomputed from independent rows.
    for cell in result['metrics']:
        rr=[x for x in rows if all(x[k]==cell[k] for k in ('arm','group','variant'))];err=abs(np.array([x['predictions'] for x in rr])-np.array([x['targets'] for x in rr]));ce=abs(np.array([x['continuous_reference_predictions'] for x in rr])-np.array([x['targets'] for x in rr]))
        assert cell['n']==len(rr) and cell['coverage']==1 and cell['status_counts']=={'VALID':len(rr)}
        for k,v in dict(mean_ecomp=err.max(1).mean(),mae_per_role=err.mean(0),rmse_per_role=np.sqrt((err*err).mean(0)),max_error=err.max(),strict_0_5_joint=np.mean(np.all(err<.5,axis=1)),continuous_reference_mean_ecomp=ce.max(1).mean(),continuous_reference_max_error=ce.max(),max_euler_vs_continuous_prediction_difference=np.max(abs(np.array([x['predictions'] for x in rr])-np.array([x['continuous_reference_predictions'] for x in rr])))).items():near(cell[k],v)
        assert cell['numerical_boundary_roundoff_rows']==sum(x['numerical_boundary_roundoff'] for x in rr)
    for arm,fields in result['resources'].items():
        rr=[x for x in rows if x['arm']==arm and x['group']=='iid' and x['variant']=='original']
        for k,agg in fields.items():
            a=np.array([x['resources'][k] for x in rr])
            for stat,fn in [('mean',np.mean),('min',np.min),('max',np.max)]:near(agg[stat],fn(a,axis=0))
    by={(x['base_id'],x['variant'],x['arm']):x for x in rows}
    for base in set(x['base_id'] for x in rows):
        for arm in arms:
            a=by[(base,'original',arm)];b=by[(base,'query_swap',arm)];assert a['predictions'][::-1]==b['predictions'] and a['final_state']==b['final_state']
    # Rational minimax certificate and native interface collision.
    rho=F(1935,1937);assert 6*rho**2405>F(1,2) and 6*rho**2406<F(1,2);near(float(6*rho**100),5.411106902157678)
    nc=json.loads((P/'native_interface_control.json').read_text());a,b=nc['cases'];assert a['content']==b['content'] and a['novelty']==b['novelty'] and a['queries']==b['queries'] and a['query_results']==b['query_results'];assert a['final_state']==b['final_state']==[0.,0.,0.]
    assert nc['minimax_pair_error_lower_bound']==.9;near(max(abs(np.array(a['targets'])-b['targets']))/2,.9)
    z=np.load(P/'native_collision_traces.npz')
    for k in ('u','A','r'):assert np.array_equal(z['port_A_'+k],z['port_B_'+k]) and np.all(z['port_A_'+k]==0)
    for case in (a,b):
        for q,qr in zip(case['queries'],case['query_results']):
            near(qr['v_hat_analog'],q*(1-.99**100));assert qr['spike_count']==0 and qr['raw_spikes']==[0]*100 and qr['eligible_spikes']==[0]*99
    uc=json.loads((P/'uncalibrated_capture_control.json').read_text());f=next(x for x in fixtures if x['id']=='uncalibrated_capture_collision');kk=vv=ss=0.
    for _ in range(100):kk+=.01*(f['keys'][0]-kk);vv+=.01*(f['values'][0]-vv)
    for _ in range(nf):ss+=(10/nf)*gf*(f['keys'][1]-kk)*(f['values'][1]-vv-(f['keys'][1]-kk)*ss)
    near(uc['final_state'],[kk,vv,ss]);near(uc['predictions'],[vv+ss*(q-kk) for q in f['queries']]);assert uc['observed_ecomp']==2.
    tree=ast.parse((P/'run_finite_write.py').read_text());writer=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Writer');methods={x.name:x for x in writer.body if isinstance(x,ast.FunctionDef)}
    assert [a.arg for a in methods['query'].args.args]==['self','q']
    for name in ('capture_step','write_step','query'):
        assert not any(isinstance(x,ast.BinOp) and isinstance(x.op,ast.Div) for x in ast.walk(methods[name]))
    assert before=={x:hashlib.sha256((P/x).read_bytes()).hexdigest() for x in names}
    output=dict(status='PASS',independent_generator=True,generator_rejections=rejected,input_count=1611,ledger_rows=3222,role_outputs=6444,metrics_cells=len(result['metrics']),snapshots_checked=snapshots_checked,complete_fixture_trajectories_checked=fixture_rows,resource_fields_checked=resource_fields_checked,max_geometric_endpoint_slope_difference=max_endpoint,max_full_fixture_slope_difference=max_trajectory,max_absolute_resource_formula_difference=max_resources,all_sources_preserved=True,exact_required_minimax_updates=2406,native_same_port_finite_error_bound=.9,uncalibrated_fixture_error=2.,controller_counter_alternative_bits={'A100':7,'FAST10':13},seconds=time.perf_counter()-started,scope='Three payload plus three counted scratch registers; stage bits exclude externally supplied event timing/clock. Scratch active real-ms is provisioned capacity, not measured occupied time. No physical energy or all-neuron claim.')
    (P/'INDEPENDENT_REVIEW_CHECKS.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))

if __name__=='__main__':main()
