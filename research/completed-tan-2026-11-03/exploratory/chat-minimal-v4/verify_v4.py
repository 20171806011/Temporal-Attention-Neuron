#!/usr/bin/env python3
"""Independent-path v4 record checker; does not import or execute either learner.
Checks exact input schedule, state transitions/budgets, interval certificates,
canonical scoring, query results and bytewise refusal invariance.
Same Chat agent; not external third-party reproduction.
"""
from fractions import Fraction as F
from pathlib import Path
import argparse, hashlib, json, struct

COUNT=0

def check(ok,msg):
    global COUNT
    COUNT+=1
    if not ok: raise AssertionError(msg)

def rat(x):
    if isinstance(x,dict) and 'exact' in x:return F(x['exact'])
    return F(x)

def interval(x): return rat(x[0]),rat(x[1])
def point(x):return x,x
def plus(a,b):return a[0]+b[0],a[1]+b[1]
def neg(a):return -a[1],-a[0]
def times(a,b):
    p=[a[i]*b[j] for i in (0,1) for j in (0,1)]
    return min(p),max(p)
def divide(a,b):
    check(not b[0]<=0<=b[1],'division interval crosses zero')
    return times(a,(min(1/b[0],1/b[1]),max(1/b[0],1/b[1])))
def polymul(a,b):
    out=[point(F(0)) for _ in range(len(a)+len(b)-1)]
    for i,x in enumerate(a):
        for j,y in enumerate(b):out[i+j]=plus(out[i+j],times(x,y))
    return out

def evalpoly(a,x):
    out=point(F(0))
    for c in reversed(a):out=plus(times(out,point(x)),c)
    return out

def ncoeff(m,n,mu):
    e=[point(F(1))]
    for r in range(1,n+1):
        z=point(F(0))
        for j in range(1,r+1):
            term=times(e[r-j],(m[j]-mu,m[j]+mu))
            z=plus(z,term if j%2 else neg(term))
        e.append(divide(z,point(F(r))))
    return [e[n-j] if (n-j)%2==0 else neg(e[n-j]) for j in range(n+1)]

def lrows(nodes):
    rows=[]
    for i,x in enumerate(nodes):
        coeff=[point(F(1))]
        for j,y in enumerate(nodes):
            if i==j:continue
            coeff=polymul(coeff,[neg(y),point(F(1))])
            den=plus(x,neg(y))
            coeff=[divide(c,den) for c in coeff]
        rows.append(coeff)
    return rows

def roundint(x):
    n=x.numerator//x.denominator;d=x-n
    return n+int(d>F(1,2) or (d==F(1,2) and n%2!=0))
def ceilint(x):return -((-x.numerator)//x.denominator)

def decoded(s):
    h=F(1,2**s['fractional_bits']);n=s['n']
    if 'm_codes' in s:
        md=dict(zip(s['m_exponents'],[h*z for z in s['m_codes']]))
        md[0]=F(n)
        return h,n,md,[h*z for z in s['b_codes']],h*s['mu_units'],h*s['beta_units'],h*s['sigma_units']
    return h,n,None,None,None,None,h*s['sigma_units']

def snapshot(s):
    if 'm_codes' in s:fs=s['m_codes']+s['b_codes']+[s['n'],s['mu_units'],s['beta_units'],s['sigma_units']]
    else:fs=s['key_codes']+s['value_codes']+[s['n'],s['sigma_units']]
    return struct.pack('<'+'q'*len(fs),*fs).hex()

def assess_address(roots,sigma,x,zeta,Delta):
    X=(max(F(-1),x-zeta),min(F(1),x+zeta))
    J=[(max(F(-1),a-sigma),min(F(1),b+sigma)) for a,b in roots]
    ids=[i for i,(a,b) in enumerate(J) if not (X[1]<a or b<X[0])]
    if len(ids)==1:
        i=ids[0];a,b=J[i]
        if max(abs(X[0]-b),abs(X[1]-a))<Delta:return 'old',i
    if not ids and all(max(F(0),X[0]-b,a-X[1])>=Delta for a,b in J):return 'new',None
    return 'ambiguous',None

def verify_certificate(s,c,anchors=None,latest=None):
    h,n,m,b,mu,beta,sigma=decoded(s)
    roots=[interval(x) for x in c['roots']]
    check(len(roots)==n,'root count')
    if m is None:
        expected=[point(h*z) for z in s['key_codes'][:n]]
        check(roots==expected,'table exact roots')
        return roots,[h*z for z in s['value_codes'][:n]],[h/2]*n
    coeff=ncoeff(m,n,mu)
    check(coeff==[interval(z) for z in c['coefficient_intervals']],'Newton coefficient certificate')
    check(c['moment_indices_used']==list(range(1,n+1)),'no high moment input')
    check(c['value_indices_used']==list(range(n)),'first N value moments only')
    centers=[rat(x) for x in c['centers']]
    for i,(lo,hi) in enumerate(roots):
        check(-1<=lo<=centers[i]<=hi<=1,'root domain/center')
        if i:check(roots[i-1][1]<lo,'disjoint roots')
        a,bb=evalpoly(coeff,lo),evalpoly(coeff,hi)
        check((a[1]<0 and bb[0]>0) or (a[0]>0 and bb[1]<0),'strict root signs')
        check([a,bb]==[interval(z) for z in c['endpoint_polynomial_intervals'][i]],'endpoint records')
    lr=lrows(roots);pc=lrows([point(t) for t in centers]);pred=[];errs=[]
    for i in range(n):
        yi=sum((pc[i][j][0]*b[j] for j in range(n)),F(0));yi_iv=point(F(0))
        for j in range(n):yi_iv=plus(yi_iv,times(lr[i][j],(b[j]-beta,b[j]+beta)))
        ee=max(abs(yi_iv[0]-yi),abs(yi_iv[1]-yi))
        check(yi==rat(c['pred'][i]),'point value')
        check(yi_iv==interval(c['value_intervals'][i]),'interval value')
        check(ee==rat(c['errors'][i]),'error certificate')
        pred.append(yi);errs.append(ee)
    if anchors is not None:
        labs=sorted(anchors,key=lambda x:anchors[x])
        check(len(labs)==n,'count equals accepted distinct')
        for i,lab in enumerate(labs):
            check(roots[i][0]<=anchors[lab]<=roots[i][1],'actual anchor enclosed')
            check(abs(pred[i]-latest[lab])<=errs[i],'certificate covers committed truth')
        for j,code in zip(s['m_exponents'],s['m_codes']):
            check(abs(code*h-sum((a**j for a in anchors.values()),F(0)))<=mu,'actual key moment bound')
        for j,code in zip(s['b_exponents'],s['b_codes']):
            check(abs(code*h-sum((latest[l]*anchors[l]**j for l in latest),F(0)))<=beta,'actual value moment bound')
    return roots,pred,errs


def expected_events(conf):
    z=F(1,10000);ss=F(conf['s'])
    keys={'A':-ss,'B':ss,'C':F(0),'D':F(conf['future_key_D']),'E':F(conf['future_key_E'])}
    vals=[F(1,4),F(-1,2),F(3,4),F(3,4),F(-3,4),F(1,2),F(-1,4),F(7,8),F(-1,8),F(3,8),F(1,8),F(1,8)]
    labs=['A','B','A','A','A','B','C','C','A','D','E','A']
    noise=[z/2,-z/2,-z/2,-z/2,ss,z/3,z/3,-z/3,z/4,-z/2,z/2,-z/4]
    zz=[z]*12;zz[4]=ss+z
    return keys,[(i+1,l,keys[l]+e,zet,v) for i,(l,e,zet,v) in enumerate(zip(labs,noise,zz,vals))]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',default='frozen_spec_v4.json');ap.add_argument('--results',default='results_v4.json');ap.add_argument('--out',default='verification_v4.json');args=ap.parse_args()
    sb=Path(args.spec).read_bytes();spec=json.loads(sb);r=json.loads(Path(args.results).read_text());L=spec['Nmax'];eps=F(spec['required_value_error'])
    check(r['spec_sha256']==hashlib.sha256(sb).hexdigest(),'frozen spec hash')
    accepted=rejected=0; paired=0
    for run in r['runs']:
        conf=run['condition'];Delta=F(conf['minimum_canonical_separation']);keys,events=expected_events(conf)
        anchors={};latest={};requests={};prev_state=None;prev_cert=None
        for rec,ev in zip(run['records'],events):
            eid,lab,obs,zeta,value=ev; er=rec['event'];s0=rec['state_before'];s1=rec['state_after'];dec=rec['decision']
            check((er['id'],er['label'])==(eid,lab),'event id')
            for k,v in [('observation',obs),('sensor_bound',zeta),('value',value),('canonical_key',keys[lab])]:check(rat(er[k])==v,'frozen event '+k)
            check(abs(obs-keys[lab])<=zeta and abs(obs-keys[lab])>0,'matched nonzero bounded observation')
            check(snapshot(s0)==rec['snapshot_hex_before'],'pre bytes')
            check(snapshot(s1)==rec['snapshot_hex_after'],'post bytes')
            if prev_state is not None:check(s0==prev_state,'state chain')
            else:check(s0['n']==0,'starts empty')
            h,n,m,b,mu,beta,sigma=decoded(s0)
            roots0=[] if prev_cert is None else [interval(x) for x in prev_cert['roots']]
            kind,idx=assess_address(roots0,sigma,obs,zeta,Delta)
            requests[lab]=value
            candidate=dec.get('detail',{}).get('candidate_state_for_external_audit')
            ccert=dec.get('detail',{}).get('candidate_certificate')
            if candidate is not None:
                verify_certificate(candidate,ccert)
                detail=dec['detail']; anchor=rat(detail['candidate_anchor_used'])
                cm=candidate['m_codes'];cb=candidate['b_codes']
                if kind=='new':
                    check(anchor==h*roundint(obs/h),'rounded first anchor')
                    check(candidate['n']==n+1,'new count')
                    check(cm==[roundint((z*h+anchor**j)/h) for z,j in zip(s0['m_codes'],s0['m_exponents'])],'all stored NEW key moments')
                    check(cb==[roundint((z*h+value*anchor**j)/h) for z,j in zip(s0['b_codes'],s0['b_exponents'])],'all stored NEW value moments')
                    check(candidate['mu_units']==ceilint((mu+h/2)/h),'new common mu')
                    check(candidate['beta_units']==ceilint((beta+h/2)/h),'new common beta')
                    check(candidate['sigma_units']==max(s0['sigma_units'],ceilint((zeta+abs(anchor-obs))/h)),'anchor error budget')
                else:
                    check(kind=='old','candidate route')
                    check(cm==s0['m_codes'] and candidate['n']==n,'no repeat key accumulation')
                    pred=rat(prev_cert['pred'][idx]); E=rat(prev_cert['errors'][idx]);I=interval(prev_cert['roots'][idx])
                    check(anchor==rat(prev_cert['centers'][idx]),'old registered-center use')
                    residual=value-pred;rad=max(abs(anchor-I[0]),abs(anchor-I[1]))
                    check(cb==[roundint((z*h+residual*anchor**j)/h) for z,j in zip(s0['b_codes'],s0['b_exponents'])],'OLD residual update')
                    check(candidate['beta_units']==ceilint((beta+E+abs(residual)*L*rad+h/2)/h),'same L bound, not tightened')
                    check(candidate['mu_units']==s0['mu_units'] and candidate['sigma_units']==s0['sigma_units'],'old key budgets unchanged')
            if dec['accepted']:
                accepted+=1;check(kind in ('old','new'),'accepted address certificate')
                check(dec['kind']==kind,'reported route')
                check((lab in latest)==(kind=='old'),'true identity agrees')
                if kind=='new':
                    check(n<L,'capacity');anchors[lab]=rat(dec['anchor_used'])
                latest[lab]=value
                if candidate is not None:check(s1==candidate,'atomic candidate commit')
                if m is None:
                    check(abs(F(s1['value_codes'][s1['key_codes'].index(roundint(anchors[lab]/h))])*h-value)<=h/2,'table latest')
            else:
                rejected+=1;check(s1==s0 and rec['snapshot_hex_before']==rec['snapshot_hex_after'],'all refusal bits unchanged')
                if dec['reason']=='address_unresolved':check(kind=='ambiguous','address refusal source')
                elif dec['reason']=='new_key_capacity_full':check(kind=='new' and n==L,'new recognised before full rejection')
                elif dec['reason']=='candidate_content_budget_exceeded':check(max(rat(x) for x in ccert['errors'])>eps,'content refusal threshold')
                else:raise AssertionError('Unexpected refusal: '+dec['reason'])
            cert=rec['post_certificate'];roots,preds,errs=verify_certificate(s1,cert,anchors,latest)
            _,nn,_,_,_,_,sig1=decoded(s1)
            check(nn==len(latest),'persistent distinct count')
            check({l:rat(v) for l,v in rec['evaluator_only_anchors'].items()}==anchors,'evaluator anchors')
            check({l:rat(v) for l,v in rec['evaluator_only_committed_latest'].items()}==latest,'evaluator latest')
            for queryfield,targets in [('queries',latest),('requested_queries',requests)]:
                check(set(rec[queryfield])==set(targets),'query label set')
                for qlab,target in targets.items():
                    qr=rec[queryfield][qlab];z=F(1,10000);qo=keys[qlab]+(z/2 if ord(qlab)%2 else -z/2)
                    check(rat(qr['observation'])==qo and rat(qr['sensor_bound'])==z,'same-noise query')
                    qkind,qi=assess_address(roots,sig1,qo,z,Delta)
                    expected=qkind=='old' and errs[qi]<=eps
                    check(qr['answered']==expected,'read decision')
                    if expected:
                        check(rat(qr['value'])==preds[qi],'query value')
                        check(rat(qr['error_bound'])==errs[qi],'query bound')
                        check(rat(qr['absolute_error'])==abs(preds[qi]-target),'query reported error')
                        if queryfield=='queries':check(abs(preds[qi]-target)<=errs[qi],'true committed guarantee')
            if not dec['accepted']:check(rec['requested_write_completed'] is False,'refusal not success')
            prev_state=s1;prev_cert=cert
    for conf in spec['conditions']:
        rr=[run for run in r['runs'] if run['condition']['name']==conf['name']]
        red=next(x for x in rr if x['method']=='redundant_common_certificate')
        mini=next(x for x in rr if x['method']=='minimal_common_certificate')
        for a,b in zip(red['records'],mini['records']):
            aa,bb=a['state_after'],b['state_after']
            check(aa['m_codes'][1:L+1]==bb['m_codes'] and aa['b_codes'][:L]==bb['b_codes'],'shared moments')
            check(all(aa[k]==bb[k] for k in ['n','mu_units','beta_units','sigma_units']),'shared control')
            check(a['queries']==b['queries'] and a['post_certificate']==b['post_certificate'],'identical certificate/output')
            check(a['decision']['accepted']==b['decision']['accepted'] and a['decision']['reason']==b['decision']['reason'],'identical decisions');paired+=1
    result={'all_checks_passed':True,'assertions':COUNT,'accepted_transactions':accepted,'rejected_transactions':rejected,
        'paired_moment_transitions':paired,'spec_sha256':hashlib.sha256(sb).hexdigest(),
        'results_sha256':hashlib.sha256(Path(args.results).read_bytes()).hexdigest(),
        'verifier_imports_learner':False,'scope':'Separately implemented checker by same Chat agent; not external independent reproduction.'}
    Path(args.out).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
