"""Exact boundary certificates, without IID sampling or native carrier simulation."""
from pathlib import Path
from fractions import Fraction as Q
import hashlib,json
OUT=Path(__file__).resolve().parent;REPO=OUT.parent.parent/'Temporal-Attention-Neuron'
def F(x):return Q(str(x))
def hist(name,positions,keys,values,queries):
    k=list(map(F,keys));v=list(map(F,values));q=list(map(F,queries))
    return dict(id=name,positions=positions,keys=k,values=v,queries=q,targets=[v[k.index(x)] for x in q])
H=[hist('L0',[0,4],[.8,1],[1,-1],[.8,1]),
   hist('L1',[0,4],[1,.8],[.8,-.8],[.8,1]),
   hist('P0',[0,2,4],[2,.8,1],[0,1.5,-.5],[.8,1]),
   hist('P1',[0,2,4],[2,1,.8],[0,1.3,-.3],[.8,1]),
   hist('H0',[0,4],[1.6,2],[1,-1],[1.6,2]),
   hist('H1',[0,4],[2,1.6],[.6,-.6],[1.6,2])]
def event_inputs(h):return [(k+v,max(Q(0),k-Q(7,5))) for k,v in zip(h['keys'],h['values'])]
def path(h,delay,q):
    inputs=event_inputs(h);by_slot=dict(zip(h['positions'],inputs));rows=[]
    for n in range(1000):
        slot=n//200;active_slot=slot in by_slot;event_on=active_slot and n%200<100
        I,nu=by_slot[slot] if event_on else (Q(0),Q(0))
        # Deliberately include the presenter's mask/timing/reset strobe as well as currents.
        rows.append((n,I,nu,Q(0),active_slot,event_on,active_slot and n%200==99))
    for n in range(int(delay*10)):rows.append((1000+n,Q(0),Q(0),Q(0),False,False,False))
    start=len(rows)
    for n in range(100):rows.append((start+n,Q(0),Q(0),q,False,False,False))
    return rows
def canonical(x):
    if isinstance(x,Q):return str(x)
    if isinstance(x,dict):return {k:canonical(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [canonical(v) for v in x]
    return x
def digest(x):return hashlib.sha256(json.dumps(canonical(x),sort_keys=True).encode()).hexdigest()
for h in H:
    assert len(h['keys']) in (2,3) and len(set(h['positions']))==len(h['keys'])
    assert all(Q(3,10)<=k<Q(5,2) for k in h['keys'])
    assert all(-3<=v<3 for v in h['values'])
    assert all(abs(a-b)>=Q(1,20) for i,a in enumerate(h['keys']) for b in h['keys'][i+1:])
    assert len(set(h['queries']))==2 and all(q in h['keys'] for q in h['queries'])
pairs=[]
for a,b,expected in ((H[0],H[1],True),(H[2],H[3],True),(H[4],H[5],False)):
    checks=[]
    assert a['queries']==b['queries'] and a['positions']==b['positions']
    for delay in (0,5,10):
        for role,q in enumerate(a['queries']):
            pa=path(a,delay,q);pb=path(b,delay,q);same=pa==pb
            assert same==expected
            differing=[i for i,(x,y) in enumerate(zip(pa,pb)) if x!=y]
            assert all(x[1]==y[1] and x[3:]==y[3:] for x,y in zip(pa,pb))
            checks.append(dict(delay_ms=delay,role=role,steps=len(pa),full_path_equal=same,content_query_mask_clock_equal=True,
                novelty_difference_steps=len(differing),path_a_sha256=digest(pa),path_b_sha256=digest(pb)))
    gap=max(abs(x-y) for x,y in zip(a['targets'],b['targets']))
    pairs.append(dict(histories=[a['id'],b['id']],raw_event_arrays_equal=(a['keys']==b['keys'] and a['values']==b['values']),
        checks=checks,target_linf_gap=str(gap),pairwise_minimax_bound=str(gap/2) if expected else None,
        bound_applicable=expected))
for h in H[4:]:
    for (k,v),(I,nu) in zip(zip(h['keys'],h['values']),event_inputs(h)):
        assert nu>0 and (nu+Q(7,5),I-nu-Q(7,5))==(k,v)
b=Q(149,150);a40=Q(3,5)*(1-b**100)*b**300;a80=Q(3,5)*(1-b**100)*b**700
assert a40>0 and a80>0 and a80<a40
sources=['p1_experiments/p1b/P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md','p1_experiments/p1b/models_p1b_v2.py','p1_experiments/p1b/runner_p1b.py','p1_experiments/task_generator.py']
result=dict(status='PASS',kind='Exact transduction/control-path certificate, not a native-model run or IID experiment',histories=canonical(H),
    pairs=pairs,all_positive_exact_inverse_verified=True,primed_gate_at_low_onsets=dict(A40=float(a40),A80=float(a80),strictly_positive_exact=True,formula='(3/5)*(1-(149/150)^100)*(149/150)^(300 or 700)'),
    positive_branch_binary64_content_value_ambiguity_upper_bound=2**-50,
    source_sha256={f:hashlib.sha256((REPO/f).read_bytes()).hexdigest() for f in sources},
    scope='L/P pairs collide at registered carrier transduction, not raw presenter API or M0. H pair shares content only and is NOT a full-input collision. No prevalence bound.')
(OUT/'THEORY_CERTIFICATES.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('histories','pairs','source_sha256')},indent=2))
