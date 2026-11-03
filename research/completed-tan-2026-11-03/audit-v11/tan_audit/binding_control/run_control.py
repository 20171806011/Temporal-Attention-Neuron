"""Frozen positive control: external key/slot binding, not neuron-only retrieval."""
from pathlib import Path
import sys,json,csv,hashlib,math,itertools,time,resource,platform
import numpy as np
OUT=Path(__file__).resolve().parent
AUDIT=OUT.parent
REPO=AUDIT.parent/'Temporal-Attention-Neuron'
sys.path[:0]=[str(REPO/'p1_experiments/p1b'),str(REPO/'p1_experiments'),str(AUDIT)]
from models_p1b_v2 import euler_step
from task_generator import TaskGenerator
from variable_key_test import coeffs

def encoding(keys,values,positions):
    u=A=r=0.;areaA=areaI=areaU2=0.;spikes=clips=0
    extremes={k:[float('inf'),-float('inf')] for k in ['candidate','u','A','current']}
    for n in range(1000):
        slot,within=divmod(n,200);j=positions.index(slot) if slot in positions and within<100 else None
        s=0. if j is None else float(keys[j]+values[j])
        d=0. if j is None else max(0.,float(keys[j])-1.4)+.1*(float(values[j])+3.)
        I=A*s;areaA+=.1*A;areaI+=.1*abs(I);areaU2+=.1*u*u
        rawA=A+(-A+d)/150.*.1;clips+=int(rawA<0 or rawA>5)
        u,A,r,sp,cand=euler_step(u,A,r,s,d,0.,tau_A=150.)
        spikes+=sp
        for key,val in [('candidate',cand),('u',u),('A',A),('current',I)]:
            if val is not None:
                extremes[key][0]=min(extremes[key][0],val);extremes[key][1]=max(extremes[key][1],val)
    return np.array([u,A,r]),dict(integral_A=areaA,integral_abs_Isyn=areaI,integral_u_squared=areaU2,
         ranges=extremes,endpoint=[u,A,r],spikes=spikes,clips=clips,physical_steps=1000,duration_ms=100.)

def vector_encoding(keys,values,positions):
    # Independent unreset batch recurrence used only in validation.
    u=np.zeros(len(keys));A=np.zeros(len(keys))
    for n in range(1000):
        slot,within=divmod(n,200);s=np.zeros(len(keys));d=np.zeros(len(keys))
        if within<100:
            for i,pos in enumerate(positions):
                if slot in pos:
                    j=pos.index(slot);s[i]=keys[i,j]+values[i,j];d[i]=max(0.,keys[i,j]-1.4)+.1*(values[i,j]+3)
        u=u+(-u+A*s)*.01;A=A+(-A+d)/150.*.1
    return np.column_stack([u,A,np.zeros(len(keys))])

def poly_params(k,pos):
    a1,a2,b11,b12,b22=coeffs(tuple(pos))
    c=np.maximum(0,np.array(k)-1.4)+.3
    from numpy.polynomial import Polynomial as P
    d1=P([c[0],.1]);s1=P([k[0],1.])
    return (a1,a2,b11,b12,b22),c

def state_formula(v,k,c):
    a1,a2,b11,b12,b22=c;d=np.maximum(0,np.array(k)-1.4)+.3+.1*np.array(v);s=np.array(k)+v
    return np.array([b11*d[0]*s[0]+(b12*d[0]+b22*d[1])*s[1],a1*d[0]+a2*d[1],0.])

def stable_roots(a,b,c):
    if a==0:return [-c/b] if b!=0 else []
    disc=b*b-4*a*c
    if disc<0:return []
    q=-.5*(b+math.copysign(math.sqrt(disc),b))
    if q==0:return [-b/(2*a)]
    return [q/a,c/q]

def inverse(features):
    assert set(features)=={'state','keys','positions','queries'}
    z=np.array(features['state']);k=np.array(features['keys']);pos=tuple(features['positions']);qs=features['queries']
    c,const=poly_params(k,pos);a1,a2,b11,b12,b22=c
    from numpy.polynomial import Polynomial as P
    d1=P([const[0],.1]);d2=P([z[1]/a2])-a1/a2*d1
    s1=P([k[0],1.]);s2=P([k[1]-const[1]/.1])+d2/.1
    pol=b11*d1*s1+(b12*d1+b22*d2)*s2-z[0]
    co=np.pad(pol.coef,(0,max(0,3-len(pol.coef))))
    roots=stable_roots(float(co[2]),float(co[1]),float(co[0]));candidates=[];valid=[];merges=snaps=replays=0
    for x in roots:
        v=np.array([x,float((d2(x)-const[1])/.1)])
        row={'raw_values':v.tolist(),'status':'PENDING','boundary_snap':False};candidates.append(row)
        if not np.all(np.isfinite(v)) or np.any(v < -3.-1e-10) or np.any(v >3.+1e-10):row['status']='OUT_OF_DOMAIN';continue
        snap=bool(np.any(v< -3.) or np.any(v>3.));row['boundary_snap']=snap;snaps+=snap
        if snap:v=np.maximum(-3.,np.minimum(3.,v))
        row['values']=v.tolist();res=float(np.max(abs(state_formula(v,k,c)-z)));row['polynomial_residual']=res
        if res>1e-10:row['status']='POLYNOMIAL_RESIDUAL';continue
        replay,resrc=encoding(k,v,pos);replays+=1;row['replayed_state']=replay.tolist()
        row['replay_residual']=float(np.max(abs(replay-z)));row['candidate_margin']=1-resrc['ranges']['candidate'][1]
        if resrc['spikes'] or resrc['clips'] or replay[2]!=0 or row['candidate_margin']<1e-10:row['status']='BRANCH_REJECTION';continue
        if row['replay_residual']>1e-10:row['status']='REPLAY_RESIDUAL';continue
        if any(float(np.max(abs(v-old)))<=1e-7 for old in valid):merges+=1;row['status']='MERGED_DUPLICATE';continue
        row['status']='VALID_CANDIDATE';valid.append(v)
    status='VALID' if len(valid)==1 else 'AMBIGUOUS' if len(valid)>1 else 'NO_REAL_ROOT' if not roots else 'NO_VALID_ROOT'
    pred=[0.,0.];query_slots=[]
    if status=='VALID':
        for q in qs:
            matches=np.flatnonzero(np.abs(k-q)<=1e-12).tolist()
            if len(matches)!=1:status='QUERY_MATCH_FAILURE';break
            query_slots.append(matches[0])
        if status=='VALID':pred=[float(valid[0][j]) for j in query_slots]
    return dict(predictions=pred,status=status,valid=status=='VALID',candidates=candidates,
        boundary_snaps=int(snaps),merged_roots=merges,candidate_replays=replays,query_slots=query_slots,
        input_fields=sorted(features),features=features)

def serialize(t,variant):
    return dict(trial_id=t.trial_id,variant=variant,seed_metadata=t.seed,keys=t.keys.tolist(),values=t.values.tolist(),
                mask=t.mask.tolist(),positions=t.event_positions,queries=list(t.query_cues),target_indices=list(t.target_idx),
                targets=list(t.target_values),gt_add=t.gt_add,gt_sub=t.gt_sub)

def metrics(rows):
    true=np.array([r['targets'] for r in rows]);pred=np.array([r['predictions'] for r in rows]);err=abs(pred-true);valid=np.array([r['valid'] for r in rows])
    m=dict(n=len(rows),coverage=float(valid.mean()),mean_e_comp=float(err.max(axis=1).mean()),mae_per_role=err.mean(axis=0).tolist(),
        rmse_per_role=np.sqrt((err**2).mean(axis=0)).tolist(),valid_joint_delivery_0_5=float(np.mean(valid&np.all(err<.5,axis=1))),
        local_valid_joint_0_05=float(np.mean(valid&np.all(err<.05,axis=1))),status_counts={s:sum(r['status']==s for r in rows) for s in sorted(set(r['status'] for r in rows))})
    if valid.any():m['conditional_valid_only']=dict(n=int(valid.sum()),mean_e_comp=float(err[valid].max(axis=1).mean()),mae_per_role=err[valid].mean(axis=0).tolist())
    return m

def main():
    started=time.perf_counter();clock=time.process_time()
    modelpath=REPO/'p1_experiments/p1b/models_p1b_v2.py'
    assert hashlib.sha256(modelpath.read_bytes()).hexdigest()=='6b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f'
    envelopes=[]
    for pos in itertools.combinations(range(5),2):
        z,res=encoding([2.5,2.5],[3.,3.],pos)
        assert res['spikes']==res['clips']==0 and res['ranges']['candidate'][1]<1 and res['ranges']['A'][1]<5
        envelopes.append(dict(positions=list(pos),resources=res))
    gen=TaskGenerator();rng=np.random.RandomState(2026110211);trials=[];inputs=[]
    for i in range(256):
        t=gen.generate_single_trial(rng,i,2026110211,N=2)
        for variant,tt in [('original',t),('query_swap',gen.create_query_swap_variant(t)),('value_swap',gen.create_value_swap_variant(t))]:
            assert all(tt.target_values[j]==tt.values[tt.target_idx[j]] and tt.query_cues[j]==tt.keys[tt.target_idx[j]] for j in (0,1))
            trials.append((variant,tt));inputs.append(serialize(tt,variant))
    ip=OUT/'inputs.jsonl';ip.write_text(''.join(json.dumps(t)+'\n' for t in inputs));ih=hashlib.sha256(ip.read_bytes()).hexdigest()
    (OUT/'INPUTS_SHA256').write_text(ih+'  inputs.jsonl\n');print('frozen inputs before encoding/metrics',ih,flush=True)
    validate=[t for v,t in trials if v=='original'][:16]
    ks=np.array([t.keys[t.event_positions] for t in validate]+[[2.5,2.5]]*10)
    vs=np.array([t.values[t.event_positions] for t in validate]+[[3.,3.]]*10)
    positions=[tuple(t.event_positions) for t in validate]+list(itertools.combinations(range(5),2))
    vector=vector_encoding(ks,vs,positions);v_res=0.
    for i,pos in enumerate(positions):
        scalar,_=encoding(ks[i],vs[i],pos);v_res=max(v_res,float(np.max(abs(vector[i]-scalar))))
    assert v_res<1e-13
    rows=[];resources=[];inverse_seconds={'correct_metadata':0.,'wrong_metadata':0.,'M0':0.};encsecs=0.
    for variant,t in trials:
        pos=tuple(t.event_positions);k=t.keys[list(pos)];v=t.values[list(pos)];start=time.perf_counter();z,res=encoding(k,v,pos);encsecs+=time.perf_counter()-start
        assert res['spikes']==res['clips']==0
        resources.append(dict(trial_id=t.trial_id,variant=variant,**res))
        for arm,keys in [('correct_metadata',k),('wrong_metadata',k[::-1])]:
            f=dict(state=z.tolist(),keys=keys.tolist(),positions=list(pos),queries=list(t.query_cues))
            st=time.perf_counter();d=inverse(f);inverse_seconds[arm]+=time.perf_counter()-st
            rows.append(dict(trial_id=t.trial_id,variant=variant,arm=arm,targets=list(t.target_values),**d))
        st=time.perf_counter();pred=[float(v[np.argmin((k-q)**2)]) for q in t.query_cues];inverse_seconds['M0']+=time.perf_counter()-st
        assert pred==list(t.target_values)
        rows.append(dict(trial_id=t.trial_id,variant=variant,arm='M0',targets=list(t.target_values),predictions=pred,status='VALID',valid=True,
            buffered_keys=k.tolist(),buffered_values=v.tolist(),positions=list(pos),queries=list(t.query_cues),comparisons=4))
    table={}
    for arm in ['correct_metadata','wrong_metadata','M0']:
        table[arm]={var:metrics([r for r in rows if r['arm']==arm and r['variant']==var]) for var in ['original','query_swap','value_swap']}
    for arm in table:
        for i in range(256):
            a=next(r for r in rows if r['arm']==arm and r['variant']=='original' and r['trial_id']==i)
            b=next(r for r in rows if r['arm']==arm and r['variant']=='query_swap' and r['trial_id']==i)
            assert a['status']==b['status'] and np.max(abs(np.array(a['predictions'])[::-1]-b['predictions']))<=1e-10
    losses={arm:np.array([max(abs(np.array(r['predictions'])-r['targets'])) for r in rows if r['arm']==arm and r['variant']=='original']) for arm in table}
    diff=losses['wrong_metadata']-losses['correct_metadata'];brng=np.random.RandomState(2026110212)
    boot=np.array([diff[brng.randint(0,256,256)].mean() for _ in range(2000)])
    zeros=[]
    for var in ['original','query_swap','value_swap']:
        rs=[dict(targets=list(t.target_values),predictions=[0,0],valid=True,status='NULL') for vv,t in trials if vv==var]
        zeros.append(dict(variant=var,metrics=metrics(rs)))
    (OUT/'predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    (OUT/'encoding_resources.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in resources))
    result=dict(protocol_sha256=hashlib.sha256((OUT/'PREREGISTRATION.md').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs_sha256=ih,base_histories=256,variants_per_history=3,arms=3,prediction_rows=len(rows),role_predictions=2*len(rows),
        model_sha256=hashlib.sha256(modelpath.read_bytes()).hexdigest(),numpy=np.__version__,python=platform.python_version(),
        metrics=table,zero_null=zeros,upper_envelopes=envelopes,scalar_vector_max_residual=v_res,
        paired_wrong_minus_correct_original=dict(mean=float(diff.mean()),percentile95_ci=np.quantile(boot,[.025,.975]).tolist(),replicates=2000,seed=2026110212),
        validation=dict(M0_exact=True,query_swaps_exact=True,input_count=768,original_generator_unchanged=True),
        resources=dict(encoding_simulation_seconds=encsecs,external_decoder_seconds=inverse_seconds,total_wall_seconds=time.perf_counter()-started,
            process_cpu_seconds=time.process_time()-clock,process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            candidate_replays={arm:sum(r.get('candidate_replays',0) for r in rows if r['arm']==arm) for arm in table},
            numerical_boundary_snaps=sum(r.get('boundary_snaps',0) for r in rows),merged_roots=sum(r.get('merged_roots',0) for r in rows),
            physical_neural_duration_per_history_ms=100.,physical_query_probe_steps=0,
            information_contract={'correct_metadata':'carrier3coordinates; external2keys+2slotindices; full(u,A,r)readout; currentq; exactconstants; external nonlinear solver',
                'wrong_metadata':'same resources, keyslotassociation exchanged','M0':'external2keys+2values+2slotindices; currentq; four keydistance calculations per paired query; no neuron'}),
        caveats=['Fresh IID basehistories only; swaps are dependent diagnostic variants.','No native/two-probe/spike-only success inferred.','No encoding/query/sensor noise or N3 tested.'])
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'metrics':table,'paired_contrast':result['paired_wrong_minus_correct_original'],'resources':result['resources']},indent=2))

if __name__=='__main__':main()
