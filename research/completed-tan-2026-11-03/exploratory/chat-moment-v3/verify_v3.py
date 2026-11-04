#!/usr/bin/env python3
"""Independent result-checking route; does NOT import learner/update code.
Uses evaluator canonical labels solely to verify saved decisions, moment bounds,
root containment, accepted outputs and unchanged refused snapshots.
This is same-agent verification, not a third-party experimental replication.
"""
import argparse, hashlib, json, struct
from fractions import Fraction as F
from pathlib import Path


def frac(x):
    if isinstance(x,dict) and 'exact' in x: return F(x['exact'])
    return F(x)


def rounded(x):
    n=x.numerator//x.denominator; r=x-n
    return n+int(r>F(1,2) or (r==F(1,2) and n%2))


def digest_state(s,method):
    if method=='certified_moments':
        xs=s['m_codes']+s['b_codes']+[s['n'],s['em_units'],s['eb_units'],s['sigma_units']]
        bs=struct.pack('<'+'q'*len(xs),*xs)
    elif method=='certified_explicit_table':
        xs=s['key_codes']+s['value_codes']+[s['n'],s['sigma_units']]
        bs=struct.pack('<'+'q'*len(xs),*xs)
    else: bs=json.dumps(s,sort_keys=True).encode()
    return hashlib.sha256(bs).hexdigest()


def main():
    a=argparse.ArgumentParser(); a.add_argument('--results',default='results_v3.json');a.add_argument('--spec',default='frozen_spec.json');a.add_argument('--out',default='verification_v3.json'); args=a.parse_args()
    raw=Path(args.results).read_bytes(); o=json.loads(raw); specraw=Path(args.spec).read_bytes();spec=json.loads(specraw)
    assert o['spec_sha256']==hashlib.sha256(specraw).hexdigest()
    checks=0; transactions=0; accepts=0; rejects=0; readchecks=0; candidates=0
    for run in o['runs']:
        meth=run['method'];cfg=run['condition'];L=spec['Nmax'];h=F(1,2**cfg['fractional_bits']);eps=F(spec['required_value_error']);Delta=F(cfg['minimum_canonical_separation'])
        anchors={};latest={};prev=None;query_count=0
        s=F(cfg['s']); keymap={'A':-s,'B':s,'C':F(0),'D':F(cfg['future_key_D']),'E':F(cfg['future_key_E'])}
        for record in run['records']:
            transactions+=1; ev=record['event']; lab=ev['label']; value=frac(ev['value']);obs=frac(ev['observation']);zz=frac(ev['sensor_bound']);state=record['state'];dec=record['decision']
            assert frac(ev['canonical_key'])==keymap[lab]
            assert abs(obs-keymap[lab])<=zz; checks+=2
            if dec['accepted']:
                accepts+=1; old=lab in latest
                assert dec['kind']==('old' if old else 'new'); checks+=1
                if not old:
                    anchors[lab]=obs if meth.startswith('exact_') else F(rounded(obs/h))*h
                elif meth=='certified_moments':
                    assert state['m_codes']==prev['m_codes'];checks+=1
                latest[lab]=value
                assert frac(record['accepted_latest_error'])<=eps;checks+=1
            else:
                rejects+=1
                assert state==prev; checks+=1
                assert record['state_digest_before']==record['state_digest_after'];checks+=1
                assert dec['bitwise_unchanged'] is True
                assert record['requested_write_completed'] is False
                if dec['reason']=='new_key_capacity_full':
                    assert len(latest)==L and lab not in latest;checks+=1
                if record.get('uncommitted_candidate_raw_value_error_evaluator_only') is not None: candidates+=1
            assert state['n']==len(latest);checks+=1
            assert digest_state(state,meth)==record['state_digest_after'];checks+=1
            assert record['requested_write_completed']==dec['accepted'];checks+=1
            labs=list(latest)
            if meth=='certified_moments':
                mm=[F(x)*h for x in state['m_codes']];bb=[F(x)*h for x in state['b_codes']]
                em=F(state['em_units'])*h;eb=F(state['eb_units'])*h
                for j,stored in enumerate(mm):assert abs(stored-sum(anchors[k]**j for k in labs))<=em;checks+=1
                for j,stored in enumerate(bb):assert abs(stored-sum(latest[k]*anchors[k]**j for k in labs))<=eb;checks+=1
                assert mm[0]==len(labs);checks+=1
                sigma=F(state['sigma_units'])*h
            elif meth.startswith('exact_'):
                for j,stored in enumerate(state['m']):assert F(stored)==sum((anchors[k]**j for k in labs),F(0));checks+=1
                for j,stored in enumerate(state['b']):assert F(stored)==sum((latest[k]*anchors[k]**j for k in labs),F(0));checks+=1
                sigma=F(state['sigma'])
            else:
                for i,k in enumerate(labs):
                    assert F(state['key_codes'][i])*h==anchors[k];checks+=1
                    assert abs(F(state['value_codes'][i])*h-latest[k])<=h/2;checks+=1
                sigma=F(state['sigma_units'])*h
            assert all(abs(anchors[k]-keymap[k])<=sigma for k in labs);checks+=len(labs)
            if meth!='certified_explicit_table':
                intervals=record['post_certificate']['roots']; ordered=sorted(anchors.values())
                assert len(intervals)==len(ordered);checks+=1
                for iv,anchor in zip(intervals,ordered):assert frac(iv[0])<=anchor<=frac(iv[1]);checks+=1
                if dec['accepted']:
                    cert=record['post_certificate']
                    assert frac(cert['alpha_lower'])>4*frac(cert['deltaC']);checks+=1
                    assert max([frac(t) for t in cert['errors']],default=F(0))<=eps;checks+=1
            for key,ans in record['queries'].items():
                query_count+=1
                assert frac(ans['target_committed'])==latest[key];checks+=1
                if ans['answered']:
                    readchecks+=1;err=abs(frac(ans['value'])-latest[key])
                    assert err==frac(ans['absolute_error']);checks+=1
                    assert err<=frac(ans['error_bound'])<=eps;checks+=1
                elif dec['accepted']:raise AssertionError('accepted state has unanswered ordinary retained-key query')
            for key,ans in record['requested_queries'].items():
                if ans['answered']:
                    assert frac(ans['requested_gap'])==abs(frac(ans['value'])-frac(ans['requested_value']));checks+=1
            prev=state
        assert query_count==run['summary']['read_attempts'];checks+=1
    out={'scope':'same-agent separate verifier; not independent third-party rerun','result_sha256':hashlib.sha256(raw).hexdigest(),
         'spec_sha256':hashlib.sha256(specraw).hexdigest(),'runs':len(o['runs']),'transactions':transactions,
         'accepted':accepts,'refused':rejects,'read_checks':readchecks,'uncommitted_candidate_diagnostics':candidates,
         'assertions_counted':checks,'all_assertions_passed':True,
         'excluded_claims':['No physical hardware error calibration','No population success rate','No original TAN or v11 rerun','No unbounded-write guarantee']}
    Path(args.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
