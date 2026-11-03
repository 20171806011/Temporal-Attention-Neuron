"""Independent rational block-recurrence check; does not import primary test code."""
from pathlib import Path
from fractions import Fraction as Q
import hashlib,json

ROOT=Path(__file__).resolve().parent
am=Q(99,100); aa=Q(1499,1500); bm=Q(1,100)
am100=am**100;aa100=aa**100
cross=(am100-aa100)/(am-aa)
sum_m=(1-am100)/(1-am)

def block(u,A,s,d):
    new_A=aa100*A+(1-aa100)*d
    new_u=am100*u+bm*s*(A*cross+d*(sum_m-cross))
    lo_A=min(A,new_A); hi_A=max(A,new_A)
    # Gate is monotone within each constant-input block. Each membrane update
    # is a convex combination of its previous value and A*s.
    upper=max(u,s*hi_A if s>=0 else s*lo_A)
    return new_u,new_A,upper,lo_A,hi_A

def run(pos,d,s):
    u=A=Q(0);max_u=Q(0);max_A=Q(0)
    for slot in range(5):
        j=pos.index(slot) if slot in pos else None
        dd=Q(0) if j is None else d[j]
        ss=Q(0) if j is None else s[j]
        for ds,ss_ in [(dd,ss),(Q(0),Q(0))]:
            u,A,bound,low,high=block(u,A,ss_,ds)
            assert low>=0 and high<5
            max_u=max(max_u,bound);max_A=max(max_A,high)
    return u,A,max_u,max_A

def coeffs(pos):
    zeros=[Q(0),Q(0)];left=[Q(1),Q(0)];right=[Q(0),Q(1)]
    a1=run(pos,left,zeros)[1];a2=run(pos,right,zeros)[1]
    b11=run(pos,left,left)[0];b12=run(pos,left,right)[0];b22=run(pos,right,right)[0]
    return a1,a2,b11,b12,b22

out=[]
for domain in ('narrow','full'):
    row=json.loads((ROOT/'variable_key_results'/f'{domain}_first_witness.json').read_text())
    pos=row['positions'];k=list(map(lambda x:Q(str(x)),row['anchor_keys']));v=list(map(lambda x:Q(str(x)),row['anchor_values']))
    a1,a2,b11,b12,b22=coeffs(pos)
    drive=lambda kk,vv:kk-Q(7,5)+Q(1,10)*(vv+3)
    d=[drive(k[i],v[i]) for i in (0,1)];s=[k[i]+v[i] for i in (0,1)]
    u,A,ub,Ab=run(pos,d,s)
    new_v1=v[0]+Q(str(row['shift']));new_d1=drive(k[0],new_v1);new_s1=k[0]+new_v1
    new_d2=(A-a1*new_d1)/a2
    new_s2=(u-b11*new_d1*new_s1)/(b12*new_d1+b22*new_d2)
    new_k2=(new_d2-Q(1,10)*new_s2+Q(11,10))/Q(9,10)
    new_v2=new_s2-new_k2
    nu,nA,nub,nAb=run(pos,[new_d1,new_d2],[new_s1,new_s2])
    lo,hi=(Q(-1,2),Q(1,2)) if domain=='narrow' else (Q(-3),Q(3))
    assert lo<=new_v1<=hi and lo<=new_v2<=hi
    assert Q(7,5)<new_k2<=Q(5,2) and abs(k[0]-new_k2)>=Q(1,20)
    assert max(ub,nub)<1 and u==nu and A==nA
    out.append(dict(domain=domain,attempt_id=row['attempt_id'],
                    exact_rational_state_equality=True,exact_separation=str(abs(new_v1-v[0])),
                    conservative_membrane_upper=float(max(ub,nub)),
                    conservative_gate_upper=float(max(Ab,nAb)),
                    exact_pair_rounded={'k2':float(new_k2),'v1':float(new_v1),'v2':float(new_v2)},
                    u=float(u),A=float(A)))

rows=[json.loads(x) for x in (ROOT/'variable_key_results/attempts.jsonl').read_text().splitlines()]
assert len(rows)==2560 and [r['attempt_id'] for r in rows]==list(range(2560))
accepted=[r for r in rows if r['accepted']]
for r in accepted:
    assert r['query']==r['anchor_keys'][0]==r['paired_keys'][0]
    assert abs((r['paired_values'][0]-r['anchor_values'][0])-r['shift'])<1e-14
    assert r['state_residual']<=1e-12
    assert max(r['native_observation_residual'],r['probe_observation_residual'])<=1e-11
    for side in ('anchor','paired'):
        lo,hi=(-.5,.5) if r['domain']=='narrow' else (-3.,3.)
        assert all(lo<=x<=hi for x in r[side+'_values'])
        kk=r[side+'_keys'];assert all(1.4<x<=2.5 for x in kk) and abs(kk[0]-kk[1])>=.05
        br=r[side+'_branch'];assert br['spikes']==br['clips']==0 and br['margin']>=1e-8
        assert r[side+'_state'][2]==0
summary=json.loads((ROOT/'variable_key_results/summary.json').read_text())
assert len(accepted)==summary['accepted']==1882
assert summary['protocol_sha256']==hashlib.sha256((ROOT/'PREREGISTRATION.md').read_bytes()).hexdigest()
assert summary['script_sha256']==hashlib.sha256((ROOT/'variable_key_test.py').read_bytes()).hexdigest()
result=dict(witnesses=out,ledger_attempts_checked=len(rows),accepted_rows_checked=len(accepted),
            accepted_by_domain={d:sum(r['domain']==d for r in accepted) for d in ('narrow','full')},
            primary_output_rewritten=False,
            qualification='Exact rational witnesses reconstruct transformed histories from decimal anchors; rounded primary JSON pairs remain approximate.')
(ROOT/'independent_variable_key_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
