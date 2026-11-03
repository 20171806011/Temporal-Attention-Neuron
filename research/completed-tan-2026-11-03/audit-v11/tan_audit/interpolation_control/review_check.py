"""Independent ledger, interface, exact arithmetic and mechanism verification."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter,defaultdict
import ast,hashlib,json,math,sys
import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]/'Temporal-Attention-Neuron'
sys.dont_write_bytecode=True
sys.path.insert(0,str(REPO/'p1_experiments'))
from task_generator import TaskGenerator

def checksum(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def serialized(t,variant):
    return dict(id=f'iid_{t.trial_id}_{variant}',group='iid',variant=variant,trial_id=t.trial_id,positions=list(t.event_positions),
                keys=t.keys[t.event_positions].tolist(),values=t.values[t.event_positions].tolist(),queries=list(t.query_cues),targets=list(t.target_values),
                seed_metadata=t.seed,rejections=t.rejection_count)

inputs=[json.loads(s) for s in (ROOT/'inputs.jsonl').read_text().splitlines()]
data={r['id']:r for r in inputs};rows=[json.loads(s) for s in (ROOT/'predictions.jsonl').read_text().splitlines()]
noise=json.loads((ROOT/'noise.json').read_text());result=json.loads((ROOT/'results.json').read_text())
assert len(inputs)==len(data)==1554
assert len(rows)==14166 and result['role_predictions']==28332
for name,key in [('PREREGISTRATION.md','protocol_sha256'),('run_interpolation.py','script_sha256'),('inputs.jsonl','inputs_sha256'),('noise.json','noise_sha256')]:
    assert checksum(ROOT/name)==result[key]

# Recreate original generator samples without importing experimental code.
gen=TaskGenerator();rng=np.random.RandomState(2026110213)
for i in range(512):
    t=gen.generate_single_trial(rng,i,2026110213,N=2)
    for variant,trial in [('original',t),('query_swap',gen.create_query_swap_variant(t)),('value_swap',gen.create_value_swap_variant(t))]:
        wanted=serialized(trial,variant);assert data[wanted['id']]==wanted

for d in inputs:
    assert len(d['keys'])==len(d['values'])==len(d['queries'])==2
    assert sorted(d['positions'])==d['positions'] and len(set(d['positions']))==2
    assert 0<=min(d['positions'])<max(d['positions'])<=4
    assert all(.3<=x<2.5 for x in d['keys']) and abs(d['keys'][1]-d['keys'][0])>=.05
    assert all(-3<=x<3 for x in d['values'])
    for q,t in zip(d['queries'],d['targets']):assert t==d['values'][d['keys'].index(q)]
original=[d for d in inputs if d['variant']=='original'];assert len(original)==530
nr=np.random.RandomState(2026110214);zs=nr.uniform(-1,1,(530,4));zw=nr.uniform(-1,1,(530,2));zq=nr.uniform(-1,1,(530,2))
for i,d in enumerate(original):
    assert noise[d['id']]==dict(state=zs[i].tolist(),writekeys=zw[i].tolist(),query=zq[i].tolist())

scenarios=['clean_float64','coefficient_fullscale_noise','query_key_noise','coefficient_plus_query','write_and_query_key_noise','float32','hold_delay10','common_post_leak10','differential_post_leak10','interwrite_leak150']
expected=set()
for d in inputs:
    if d['variant']=='original':
        expected.update((d['id'],'interpolation',s) for s in scenarios)
        expected.update((d['id'],'M0_cache',s) for s in scenarios[:7])
    else:expected.update((d['id'],a,'clean_float64') for a in ['interpolation','M0_cache'])
    expected.update((d['id'],a,'clean_float64') for a in ['modified_native_analog','modified_privileged_inverse'])
actual={(r['id'],r['arm'],r['scenario']) for r in rows}
assert len(actual)==len(rows) and actual==expected
grouped=defaultdict(list);lookup={}
for r in rows:
    d=data[r['id']];assert r['targets']==d['targets'] and r['group']==d['group'] and r['variant']==d['variant']
    grouped[(r['group'],r['variant'],r['arm'],r['scenario'])].append(r);lookup[(r['id'],r['arm'],r['scenario'])]=r
    assert r['valid']==(r['status']=='VALID')
    if not r['valid']:assert r['predictions']==[0.,0.]

max_metric_delta=0.
for cell in result['metrics']:
    key=tuple(cell[k] for k in ['group','variant','arm','scenario']);rr=grouped[key]
    e=np.abs(np.array([r['predictions'] for r in rr])-np.array([r['targets'] for r in rr]));valid=np.array([r['valid'] for r in rr])
    calculated=dict(n=len(rr),coverage=float(valid.mean()),mean_e_comp=float(e.max(axis=1).mean()),max_abs_error=float(e.max()),
                    mae_per_role=e.mean(axis=0).tolist(),rmse_per_role=np.sqrt((e*e).mean(axis=0)).tolist(),
                    valid_joint_0_5=float(np.mean(valid&np.all(e<.5,axis=1))),local_valid_joint_0_05=float(np.mean(valid&np.all(e<.05,axis=1))))
    for k,v in calculated.items():
        delta=float(np.max(np.abs(np.asarray(v)-cell[k])));max_metric_delta=max(max_metric_delta,delta);assert delta<1e-13
    assert dict(Counter(r['status'] for r in rr))==cell['status_counts']
    raw=[r for r in rr if r.get('raw_predictions') is not None]
    m=float(np.mean([max(abs(np.array(r['raw_predictions'])-r['targets'])) for r in raw])) if raw else None
    assert m==cell['raw_finite_mean_e_comp']

# Source access audit: only two data registers and static dtype/controller.
source=(ROOT/'run_interpolation.py').read_text();tree=ast.parse(source)
qfun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='query_two_registers')
assert [a.arg for a in qfun.args.args]==['state','q','dtype']
names={n.id for n in ast.walk(qfun) if isinstance(n,ast.Name)}
assert names <= {'state','q','dtype','T','float'}
writer=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OnlineWriter')
slots=next(n for n in writer.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='__slots__' for t in n.targets))
assert ast.literal_eval(slots.value)==('x','y','phase','dtype')
for r in rows:
    if r['arm']=='interpolation':
        assert r['query_input_fields']==['two_register_state','q'] and len(r['state'])==2
        assert [x['step'] for x in r['write_transitions']]==[200*x for x in data[r['id']]['positions']]
        assert [x['phase'] for x in r['write_transitions']] in ([1,2],[1,3])

raw_bounds=0;max_bound_excess=-float('inf');precision=[];max_leak_residual=0.;max_state_relative_delta=0.
la=math.exp(-10/150);lb=math.exp(-10/100);step=math.exp(-.1/150)
for r in rows:
    if r['arm']!='interpolation':continue
    d=data[r['id']];s=r['scenario'];target=np.array(d['targets']);q=np.array(d['queries'])
    kf=list(map(F.from_float,d['keys']));vf=list(map(F.from_float,d['values']))
    a=(vf[1]-vf[0])/(kf[1]-kf[0]);b=vf[0]-a*kf[0]
    assert abs(a)<=120 and abs(b)<=297
    for kk,vv in zip(kf,vf):assert a*kk+b==vv
    z=noise[d['id']] if d['variant']=='original' else dict(state=[0.]*4,writekeys=[0.]*2,query=[0.]*2)
    if s!='float32' and r['phase']==2:
        kk=list(kf);vv=list(vf)
        if s=='write_and_query_key_noise':kk=[F.from_float(d['keys'][i]+1e-4*z['writekeys'][i]) for i in (0,1)]
        if s=='interwrite_leak150':
            gap=200*(d['positions'][1]-d['positions'][0]);factor=F.from_float(step)**gap
            kk[0]*=factor;vv[0]*=factor
        ea_exact=(vv[1]-vv[0])/(kk[1]-kk[0]);eb_exact=(kk[1]*vv[0]-kk[0]*vv[1])/(kk[1]-kk[0])
        expected_state=np.array([float(ea_exact),float(eb_exact)])
        if s in ['coefficient_fullscale_noise','coefficient_plus_query']:expected_state+=np.array([.12,.297])*z['state'][:2]
        if s=='common_post_leak10':expected_state*=la
        if s=='differential_post_leak10':expected_state*=np.array([la,lb])
        rel=float(np.max(abs(np.array(r['state'])-expected_state)/(1+abs(expected_state))))
        max_state_relative_delta=max(max_state_relative_delta,rel);assert rel<1e-10
    if r['phase']==2:
        raw=np.array(r['raw_predictions']);state=np.array(r['state']);qq=np.array(r['queries'])
        if s!='float32':assert np.max(abs(raw-(state[0]*qq+state[1])))<1e-12
        inrange=abs(state[0])<=120*(1+1e-10) and abs(state[1])<=297*(1+1e-10)
        assert r['valid']==inrange
    if s in ['coefficient_fullscale_noise','query_key_noise','coefficient_plus_query','write_and_query_key_noise','common_post_leak10','differential_post_leak10']:
        if s=='coefficient_fullscale_noise':bound=.597
        elif s=='query_key_noise':bound=.012
        elif s=='coefficient_plus_query':bound=.609012
        elif s=='write_and_query_key_noise':bound=6*.0002/(.05-.0002)
        elif s=='common_post_leak10':bound=3*(1-la)
        else:bound=300*abs(la-lb)+3*(1-lb)
        assert abs(r['raw_error_bound']-bound)<1e-13
        excess=float(max(abs(np.array(r['raw_predictions'])-target))-bound)
        assert excess<=1e-10;max_bound_excess=max(max_bound_excess,excess);raw_bounds+=1
    if s in ['common_post_leak10','differential_post_leak10']:
        ll=la if s=='common_post_leak10' else lb
        predicted_error=(la-ll)*float(a)*q+(ll-1)*target
        residual=float(np.max(abs((np.array(r['raw_predictions'])-target)-predicted_error)))
        max_leak_residual=max(max_leak_residual,residual);assert residual<1e-10
    if s=='interwrite_leak150':
        gap=200*(d['positions'][1]-d['positions'][0]);old=np.array(d['keys'][:1]+d['values'][:1],float)
        # Independent closed-form decay; classify denominator separately from
        # floating per-step singular witness, whose small residual is expected.
        expected_old=old*step**gap;before=np.array(r['write_transitions'][1]['before'])
        assert np.max(abs(before-expected_old))<1e-13
        denom=d['keys'][1]-before[0]
        assert (r['phase']==3)==(abs(denom)<=1e-12)
    if s in ['clean_float64','float32']:
        aa,bb=map(F.from_float,r['state']);ea,eb=abs(aa-a),abs(bb-b)
        errmax=boundmax=roundmax=0.
        for qq,qt,vv,raw in zip(r['queries'],d['queries'],d['targets'],r['raw_predictions']):
            qreal=F.from_float(qt);qactual=F.from_float(qq);rawexact=F.from_float(raw)
            rounding=abs(rawexact-(aa*qactual+bb))
            bound=abs(qreal)*ea+eb+(abs(a)+ea)*abs(qactual-qreal)+rounding
            error=abs(rawexact-F.from_float(vv));assert error<=bound
            errmax=max(errmax,float(error));boundmax=max(boundmax,float(bound));roundmax=max(roundmax,float(rounding))
        precision.append(dict(id=r['id'],scenario=s,group=d['group'],status=r['status'],raw_max_error=errmax,
                              exact_error_bound=boundmax,readout_rounding=roundmax,delta_a=float(ea),delta_b=float(eb)))

# Recompute all noisy caches directly; no experimental query function imported.
for r in rows:
    if r['arm']!='M0_cache':continue
    d=data[r['id']];s=r['scenario'];k=np.array(d['keys']);v=np.array(d['values']);q=np.array(d['queries'])
    z=noise[d['id']] if d['variant']=='original' else dict(state=[0.]*4,writekeys=[0.]*2,query=[0.]*2)
    if s in ['coefficient_fullscale_noise','coefficient_plus_query']:k+=.0011*np.array(z['state'][:2]);v+=.003*np.array(z['state'][2:])
    if s=='write_and_query_key_noise':k+=.0001*np.array(z['writekeys'])
    if s in ['query_key_noise','coefficient_plus_query','write_and_query_key_noise']:q+=.0001*np.array(z['query'])
    if s=='float32':k=k.astype(np.float32).astype(float);v=v.astype(np.float32).astype(float);q=q.astype(np.float32).astype(float)
    pred=[float(v[0] if abs(k[0]-qq)<=abs(k[1]-qq) else v[1]) for qq in q]
    assert pred==r['raw_predictions']
    assert k.tolist()==r['state']['keys'] and v.tolist()==r['state']['values']

# Test modified native analog on 34 existing histories via independent block
# encoding and direct cue recurrence; no additional histories or model tuning.
def block(u,A,content,drive):
    am=.99;ag=1-1/1500;cross=(am**100-ag**100)/(am-ag);s=(1-am**100)/(1-am)
    return am**100*u+.01*content*(A*cross+drive*(s-cross)),ag**100*A+(1-ag**100)*drive
max_modified_state_residual=max_modified_readout_residual=0.
for d in original[:16]+[x for x in original if x['group']=='adversarial']:
    u=A=0.
    for slot in range(5):
        if slot in d['positions']:
            j=d['positions'].index(slot);k=d['keys'][j];v=d['values'][j];content=k+v;drive=max(0,k-1.4)+.1*(v+3)
        else:content=drive=0.
        u,A=block(u,A,content,drive);u,A=block(u,A,0.,0.)
    rr=lookup[(d['id'],'modified_native_analog','clean_float64')]
    res=max(abs(u-rr['state'][0]),abs(A-rr['state'][1]));max_modified_state_residual=max(max_modified_state_residual,res);assert res<1e-12
    for q,pred in zip(d['queries'],rr['predictions']):
        uu=u;r=0.
        for _ in range(100):
            oldr=r
            if r>0:uu=0.;r=max(0,r-.1);candidate=None
            else:
                candidate=.99*uu+.01*q
                if candidate>=1:uu=0.;r=2.
                else:uu=candidate;r=0.
        read=0. if oldr>0 else candidate
        out=.1326308169898006*read-.02681648355488859
        res=abs(out-pred);max_modified_readout_residual=max(max_modified_readout_residual,res);assert res<1e-12

precision_summary={}
for group,scenario in [('iid','clean_float64'),('iid','float32'),('adversarial','clean_float64'),('adversarial','float32')]:
    pp=[p for p in precision if p['group']==group and p['scenario']==scenario]
    precision_summary[group+'_'+scenario]=dict(n=len(pp),max_raw_error=max(p['raw_max_error'] for p in pp),max_exact_bound=max(p['exact_error_bound'] for p in pp),
                                             max_readout_rounding=max(p['readout_rounding'] for p in pp),statuses=dict(Counter(p['status'] for p in pp)))
report=dict(prediction_rows_checked=len(rows),metrics_cells_checked=len(grouped),input_records_checked=len(inputs),original_iid_histories=512,
            adversarial_histories=18,noise_arrays_reproduced=True,max_metric_delta=max_metric_delta,raw_bounds_checked=raw_bounds,max_bound_excess=max_bound_excess,
            exact_query_and_precision_records_checked=len(precision),precision_summary=precision_summary,
            max_independent_state_relative_delta=max_state_relative_delta,
            max_leak_identity_residual=max_leak_residual,modified_native_checks=34,max_modified_state_residual=max_modified_state_residual,
            max_modified_readout_residual=max_modified_readout_residual,query_interface_only_two_state_numbers_and_q=True,
            primary_outputs_modified=False,source_hashes={p:checksum(ROOT/p) for p in ['PREREGISTRATION.md','run_interpolation.py','inputs.jsonl','noise.json','predictions.jsonl','results.json']})
(ROOT/'REVIEW_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
