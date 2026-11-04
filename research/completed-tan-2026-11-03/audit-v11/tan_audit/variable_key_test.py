"""Preregistered finite-budget counterexamples; never modifies repository outputs."""
from pathlib import Path
import sys, json, hashlib, platform, csv, math, itertools
from decimal import Decimal, localcontext
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent / 'Temporal-Attention-Neuron'
sys.path[:0] = [str(REPO/'p1_experiments/p1b'), str(REPO/'p1_experiments')]
from models_p1b_v2 import euler_step, simulate_encoding, CarrierState, simulate_cue_and_readout
from task_generator import TaskGenerator

def coeffs(pos, dtype=float):
    one = dtype(1); am = one-dtype(1)/dtype(100); aa = one-dtype(1)/dtype(1500)
    bm, ba = one-am, one-aa
    a1=a2=b11=b12=b22=dtype(0)
    for n in range(1000):
        slot, within = divmod(n,200)
        x1 = int(slot==pos[0] and within<100)
        x2 = int(slot==pos[1] and within<100)
        b11 = am*b11 + bm*a1*x1
        b12 = am*b12 + bm*a1*x2
        b22 = am*b22 + bm*a2*x2
        a1 = aa*a1 + ba*x1
        a2 = aa*a2 + ba*x2
    return a1,a2,b11,b12,b22

def formula(k,v,c):
    a1,a2,b11,b12,b22=c
    d=k-1.4+.1*(v+3); s=k+v
    return np.array([b11*d[0]*s[0]+(b12*d[0]+b22*d[1])*s[1], a1*d[0]+a2*d[1], 0.])

def encode(k,v,pos,eta=.1,tau_A=150.):
    u=A=r=0.; spikes=0; max_cand=-float('inf'); max_A=0.; clips=0
    for n in range(1000):
        slot,within=divmod(n,200)
        j = pos.index(slot) if slot in pos and within<100 else None
        content=0. if j is None else float(k[j]+v[j])
        drive=0. if j is None else max(0.,abs(float(k[j]))-1.4)+eta*(float(v[j])+3.)
        raw_next=A+(-A+drive)/tau_A*.1
        clips+=int(raw_next<0 or raw_next>5)
        u,A,r,spike,cand=euler_step(u,A,r,content,drive,0.,tau_A=tau_A)
        spikes+=spike
        max_A=max(max_A,A)
        if cand is not None: max_cand=max(max_cand,cand)
    return np.array([u,A,r]),{'spikes':spikes,'clips':clips,'margin':1.-max_cand,'max_A':max_A}

def probe(z,q):
    u,A,r=map(float,z); obs=[u]; sp=[]; cands=[]
    for i in range(100):
        u,A,r,s,c=euler_step(u,A,r,.1,0.,q,tau_A=150.)
        obs.append(u); sp.append(s); cands.append(0. if c is None else c)
    return np.array(obs+sp+cands), [obs[0],obs[-1]]

def knownkeys_inverse(k,z,c,lo,hi):
    # Polynomial coefficients use state and supplied keys, never the true value.
    a1,a2,b11,b12,b22=c; c1=k[0]-1.1; c2=k[1]-1.1
    d1=np.polynomial.Polynomial([c1,.1])
    d2=np.polynomial.Polynomial([z[1]/a2,0])-a1/a2*d1
    s1=np.polynomial.Polynomial([k[0],1.])
    s2=np.polynomial.Polynomial([k[1]-c2/.1,0])+d2/.1
    poly=b11*d1*s1+(b12*d1+b22*d2)*s2-z[0]
    sols=[]
    for root in poly.roots():
        if abs(np.imag(root))>1e-8: continue
        v1=float(np.real(root)); v2=float((d2(v1)-c2)/.1)
        if lo<=v1<=hi and lo<=v2<=hi and not any(abs(v1-x[0])<1e-7 for x in sols):
            sols.append([v1,v2])
    return {'solutions':sols,'abstain':len(sols)!=1,'prediction':sols[0] if len(sols)==1 else [0.,0.]}

def singular_values(k,v,c):
    a1,a2,b11,b12,b22=c; d=k-1.4+.1*(v+3); s=k+v
    jac=np.array([[b11*(.1*s[0]+d[0])+b12*.1*s[1],b12*d[0]+b22*(.1*s[1]+d[1])],[.1*a1,.1*a2]])
    return np.linalg.svd(jac,compute_uv=False).tolist()

def decimal_witness(row):
    with localcontext() as ctx:
        ctx.prec=120
        D=Decimal; c=coeffs(tuple(row['positions']),D)
        a1,a2,b11,b12,b22=c
        k=[D(str(x)) for x in row['anchor_keys']]; v=[D(str(x)) for x in row['anchor_values']]
        drive=lambda kk,vv: kk-D('1.4')+D('.1')*(vv+D(3))
        d=[drive(k[i],v[i]) for i in range(2)]; s=[k[i]+v[i] for i in range(2)]
        A=a1*d[0]+a2*d[1]; u=b11*d[0]*s[0]+(b12*d[0]+b22*d[1])*s[1]
        nv=v[0]+D(str(row['shift'])); nd1=drive(k[0],nv); ns1=k[0]+nv
        nd2=(A-a1*nd1)/a2
        ns2=(u-b11*nd1*ns1)/(b12*nd1+b22*nd2)
        nk2=(nd2-D('.1')*ns2+D('1.1'))/D('.9'); nv2=ns2-nk2
        def run(kk,vv):
            uu=aa=D(0); m=D('-Infinity')
            for n in range(1000):
                slot,within=divmod(n,200)
                j=row['positions'].index(slot) if slot in row['positions'] and within<100 else None
                ss=D(0) if j is None else kk[j]+vv[j]
                dd=D(0) if j is None else drive(kk[j],vv[j])
                cand=uu+(-uu+aa*ss)/D(100)
                newA=aa+(-aa+dd)/D(1500)
                assert cand<D(1) and D(0)<=newA<=D(5)
                uu,aa=cand,newA; m=max(m,cand)
            return uu,aa,D(1)-m
        z=run(k,v); nz=run([k[0],nk2],[nv,nv2]); residual=max(abs(z[i]-nz[i]) for i in (0,1))
        assert residual<D('1e-100'),residual
        return {'domain':row['domain'],'attempt_id':row['attempt_id'],'precision':120,'state_residual':str(residual),
            'k2_prime':str(nk2),'v1_prime':str(nv),'v2_prime':str(nv2),'minimum_margin':str(min(z[2],nz[2]))}

def controls():
    rng=np.random.RandomState(2026110210); gen=TaskGenerator(); residuals=[]
    for N in (2,3):
        for i in range(32):
            t=gen.generate_single_trial(rng,i,2026110210,N=N)
            pos=tuple(t.event_positions); k=t.keys[list(pos)];v=t.values[list(pos)]
            # encode accepts arbitrary N for validation; coeffs uses N=2 only.
            z,info=encode(k,v,pos,eta=0.,tau_A=15.)
            native,_,_=simulate_encoding(t.keys,t.values,t.mask,'M1')
            residuals.append(float(np.max(np.abs(z-native.to_array()))))
    assert max(residuals)==0.
    import test_p1b_g1_fixtures as fixtures
    fixture_names=['test_g1_0_governance_and_provenance','test_g1_1_state_closure_and_refractory_readout']
    for name in fixture_names: getattr(fixtures,name)()
    return {'original_wrapper_trials':64,'max_state_error':max(residuals),'native_fixtures_passed':fixture_names}

def main():
    out=ROOT/'variable_key_results';out.mkdir(exist_ok=True)
    ctrl=controls();rng=np.random.RandomState(2026110209);rows=[]; witnesses={}
    for domain,lo,hi,delta in [('narrow',-.5,.5,.2),('full',-3.,3.,1.2)]:
        for pos in itertools.combinations(range(5),2):
            c=coeffs(pos);a1,a2,b11,b12,b22=c
            for anchor_id in range(64):
                k=rng.uniform(1.45,2.45,2)
                while abs(k[0]-k[1])<.05:k=rng.uniform(1.45,2.45,2)
                v=rng.uniform(lo,hi,2);z,branch=encode(k,v,pos)
                for shift in (delta,-delta):
                    row={'attempt_id':len(rows),'domain':domain,'positions':list(pos),'anchor_id':anchor_id,'shift':shift,
                         'anchor_keys':k.tolist(),'anchor_values':v.tolist(),'anchor_state':z.tolist(),'anchor_branch':branch,'accepted':False}
                    rows.append(row)
                    v1=v[0]+shift;d1=k[0]-1.4+.1*(v1+3);s1=k[0]+v1
                    # Use ideal algebraic anchor state, separately compared to actual Euler output.
                    z_formula=formula(k,v,c);d2=(z_formula[1]-a1*d1)/a2;den=b12*d1+b22*d2
                    row['anchor_formula_residual']=float(np.max(np.abs(z_formula-z)))
                    if not math.isfinite(den) or abs(den)<=1e-14:row['reason']='denominator';continue
                    s2=(z_formula[0]-b11*d1*s1)/den;k2=(d2-.1*s2+1.1)/.9;v2=s2-k2
                    nk=np.array([k[0],k2]);nv=np.array([v1,v2]);row['paired_keys']=nk.tolist();row['paired_values']=nv.tolist()
                    if not all(lo<=x<=hi for x in nv):row['reason']='value_domain';continue
                    if not all(.3<=x<=2.5 and x>1.4 for x in nk) or abs(nk[0]-nk[1])<.05:row['reason']='key_domain_or_spacing';continue
                    nz,nb=encode(nk,nv,pos);row['paired_state']=nz.tolist();row['paired_branch']=nb
                    if any(b['spikes'] or b['clips'] or b['margin']<1e-8 for b in [branch,nb]) or z[2]!=0 or nz[2]!=0:
                        row['reason']='branch';continue
                    row['paired_formula_residual']=float(np.max(np.abs(formula(nk,nv,c)-nz)))
                    residual=float(np.max(np.abs(z-nz)));row['state_residual']=residual
                    if residual>1e-12:row['reason']='state_residual';continue
                    orig=simulate_cue_and_readout(CarrierState(*z),float(k[0]),'M1')
                    changed=simulate_cue_and_readout(CarrierState(*nz),float(k[0]),'M1')
                    native_res=max(abs(orig[f]-changed[f]) for f in ['u_readout','v_hat_count','v_hat_latency'])
                    native_res=max(native_res,float(np.max(np.abs(np.array(orig['raw_spikes'])-np.array(changed['raw_spikes'])))))
                    p,p2=probe(z,float(k[0]));np_,np2=probe(nz,float(k[0]));pres=float(np.max(np.abs(p-np_)))
                    row.update(native_observation_residual=native_res,probe_observation_residual=pres,
                        native_anchor_prediction={x:orig[x] for x in ['u_readout','v_hat_count','v_hat_latency']},
                        native_paired_prediction={x:changed[x] for x in ['u_readout','v_hat_count','v_hat_latency']},
                        probe_anchor_samples=p2,probe_paired_samples=np2)
                    if max(native_res,pres)>1e-11:row['reason']='observations';continue
                    assert max(row['anchor_formula_residual'],row['paired_formula_residual'])<1e-12
                    row.update(accepted=True,reason='accepted',query=float(k[0]),
                        anchor_singular_values=singular_values(k,v,c),paired_singular_values=singular_values(nk,nv,c),
                        anchor_knownkeys_inverse=knownkeys_inverse(k,z,c,lo,hi),
                        paired_knownkeys_inverse=knownkeys_inverse(nk,nz,c,lo,hi))
                    if domain not in witnesses:witnesses[domain]=decimal_witness(row)
    cells=[]
    for domain in ['narrow','full']:
        for pos in itertools.combinations(range(5),2):
            rr=[r for r in rows if r['domain']==domain and r['positions']==list(pos)];ok=[r for r in rr if r['accepted']]
            reasons={x:sum(r['reason']==x for r in rr) for x in sorted(set(r['reason'] for r in rr))}
            errors=[];success=0;abstain=0
            for r in ok:
                for side in ['anchor','paired']:
                    inv=r[side+'_knownkeys_inverse'];err=float(np.max(np.abs(np.array(inv['prediction'])-r[side+'_values'])))
                    errors.append(err);abstain+=inv['abstain'];success+=not inv['abstain'] and err<.05
            cells.append({'domain':domain,'positions':list(pos),'attempts':len(rr),'accepted':len(ok),'reasons':reasons,
                'max_state_residual':max([r['state_residual'] for r in ok],default=None),
                'minimum_branch_margin':min([min(r['anchor_branch']['margin'],r['paired_branch']['margin']) for r in ok],default=None),
                'minimum_fixedkey_singular_value':min([min(r['anchor_singular_values'][-1],r['paired_singular_values'][-1]) for r in ok],default=None),
                'privileged_inverse_abstentions':int(abstain),'privileged_inverse_total':2*len(ok),'privileged_joint_local_success':int(success),
                'privileged_mean_max_error':float(np.mean(errors)) if errors else None})
    summary={'protocol_sha256':hashlib.sha256((ROOT/'PREREGISTRATION.md').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'repository_commit':'85213f0d417d54e4f48b301ed1c0e4200ef3c421',
        'python':platform.python_version(),'numpy':np.__version__,'controls':ctrl,'attempts':len(rows),'accepted':sum(r['accepted'] for r in rows),
        'cells':cells,'decimal_witnesses':list(witnesses.values()),'limits':'Adversarial constructed pairs, not IID risk. Exact-state equality is ideal algebra only. Known-key inverse has extra input information.'}
    (out/'attempts.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    for domain in witnesses:
        first=next(r for r in rows if r['accepted'] and r['domain']==domain)
        (out/(domain+'_first_witness.json')).write_text(json.dumps(first,indent=2)+'\n')
    print(json.dumps({'attempts':summary['attempts'],'accepted':summary['accepted'],'cells':cells,'decimal_witnesses':list(witnesses.values())},indent=2))

if __name__=='__main__':main()
