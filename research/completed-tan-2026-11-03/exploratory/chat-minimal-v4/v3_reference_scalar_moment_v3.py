#!/usr/bin/env python3
"""Scalar-key moment memory, certified non-destructive transactions.
Conceptual Chat-sandbox experiment, NOT TAN/P1-B/v11 production execution.
Learners accept only (observed scalar, supplied value, sensor error bound).
Canonical identities, keys and latest targets are held by the evaluator alone.
Persistent moment memory has no node/key array; nodes are transient reconstructions.

Run: python scalar_moment_v3.py --out results_v3.json
Requires numpy and sympy. Fixed-point persistent memory; exact-rational certificate
workspace is explicitly an additional resource, not a floating-point proof claim.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, platform, struct, sys
from dataclasses import dataclass
from fractions import Fraction as F
from pathlib import Path
from typing import Any
import numpy as np
import sympy as sp

MAX_RATIONAL_NUM_BITS = 0
MAX_RATIONAL_DEN_BITS = 0
RATIONAL_SAMPLES = 0


def track(x: F) -> F:
    global MAX_RATIONAL_NUM_BITS, MAX_RATIONAL_DEN_BITS, RATIONAL_SAMPLES
    x = F(x)
    MAX_RATIONAL_NUM_BITS = max(MAX_RATIONAL_NUM_BITS, abs(x.numerator).bit_length())
    MAX_RATIONAL_DEN_BITS = max(MAX_RATIONAL_DEN_BITS, x.denominator.bit_length())
    RATIONAL_SAMPLES += 1
    return x


@dataclass(frozen=True)
class I:
    lo: F
    hi: F
    def __post_init__(self):
        object.__setattr__(self, 'lo', track(F(self.lo)))
        object.__setattr__(self, 'hi', track(F(self.hi)))
        if self.lo > self.hi:
            raise ValueError('reversed interval')
    @staticmethod
    def point(x): return I(F(x), F(x))
    def __add__(self, other):
        o = other if isinstance(other, I) else I.point(other)
        return I(self.lo+o.lo, self.hi+o.hi)
    __radd__ = __add__
    def __neg__(self): return I(-self.hi, -self.lo)
    def __sub__(self, other): return self + (-other if isinstance(other, I) else -F(other))
    def __rsub__(self, other): return I.point(other) - self
    def __mul__(self, other):
        o = other if isinstance(other, I) else I.point(other)
        q = [self.lo*o.lo, self.lo*o.hi, self.hi*o.lo, self.hi*o.hi]
        return I(min(q), max(q))
    __rmul__ = __mul__
    def __truediv__(self, other):
        o = other if isinstance(other,I) else I.point(other)
        if o.lo <= 0 <= o.hi: raise ZeroDivisionError('interval contains zero')
        inv = I(min(1/o.lo,1/o.hi), max(1/o.lo,1/o.hi))
        return self*inv
    def absmax(self): return max(abs(self.lo),abs(self.hi))
    def intersects(self, other): return not (self.hi < other.lo or other.hi < self.lo)
    def width(self): return self.hi-self.lo
    def mid(self): return track((self.lo+self.hi)/2)
    def power(self,j):
        if j==0: return I.point(1)
        if j%2==1: return I(self.lo**j,self.hi**j)
        return I(0 if self.lo<=0<=self.hi else min(self.lo**j,self.hi**j),max(self.lo**j,self.hi**j))


def fnum(x):
    if isinstance(x,F): return {'decimal':float(x), 'exact':str(x)}
    if isinstance(x,I): return [fnum(x.lo),fnum(x.hi)]
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)


def round_int(x: F) -> int:
    x=F(x); n=x.numerator//x.denominator; rem=x-n
    if rem>F(1,2) or (rem==F(1,2) and n%2): n+=1
    return int(n)


def ceil_int(x:F)->int: return -((-x.numerator)//x.denominator)


def qround(x:F,h:F)->int:
    n=round_int(x/h)
    if not -(2**63)<=n<2**63: raise OverflowError('persistent int64 overflow')
    return n


def qceil(x:F,h:F)->int:
    n=ceil_int(x/h)
    if not 0<=n<2**63: raise OverflowError('budget int64 overflow')
    return n


def powers(x:F,n:int): return [track(x**j) for j in range(n+1)]


def poly_mul(a,b):
    isiv = any(isinstance(t,I) for t in a+b)
    z=I.point(0) if isiv else F(0)
    out=[z for _ in range(len(a)+len(b)-1)]
    for i,x in enumerate(a):
        for j,y in enumerate(b): out[i+j]=out[i+j]+x*y
    return out


def poly_at(a,x):
    y=I.point(0) if any(isinstance(t,I) for t in a) else F(0)
    for t in a[::-1]: y=y*x+t
    return y


def newton_coeffs(m,n,em=F(0),interval=False):
    one=I.point(1) if interval else F(1)
    zero=I.point(0) if interval else F(0)
    e=[one]
    for r in range(1,n+1):
        s=zero
        for j in range(1,r+1):
            mj=I(m[j]-em,m[j]+em) if interval else m[j]
            s=s+((-1)**(j-1))*e[r-j]*mj
        e.append(s/r)
    return [((-1)**(n-j))*e[n-j] for j in range(n+1)]


def lagrange_rows(nodes):
    interval=any(isinstance(x,I) for x in nodes)
    one=I.point(1) if interval else F(1)
    rows=[]
    for i,x in enumerate(nodes):
        p=[one]
        for j,y in enumerate(nodes):
            if i==j: continue
            p=poly_mul(p,[-y,one]); den=x-y
            p=[t/den for t in p]
        rows.append(p)
    return rows


class Refuse(Exception):
    def __init__(self,reason,detail=None):
        self.reason=reason; self.detail=detail or {}; super().__init__(reason)


def root_cert(m,n,L,em,h,exact=False):
    if n==0: return [],[],{'root_doublings':0,'redundancy_checked':True}
    cm=newton_coeffs(m,n)
    if exact:
        z=sp.Symbol('z')
        pol=sp.Poly(sum(sp.Rational(c.numerator,c.denominator)*z**j for j,c in enumerate(cm)),z)
        rr=sp.polys.polytools.ground_roots(pol)
        if len(rr)!=n or any(v!=1 for v in rr.values()): raise Refuse('exact_root_factorization_failed')
        cc=sorted(F(int(k.p),int(k.q)) for k in rr)
        if any(abs(x)>1 for x in cc): raise Refuse('root_out_of_domain')
        ints=[I.point(x) for x in cc]
        doublings=0
    else:
        rr=np.roots(np.array([float(x) for x in cm[::-1]],dtype=float))
        if len(rr)!=n or np.max(np.abs(np.imag(rr)))>1e-8: raise Refuse('uncertified_root_proposal')
        hh=F(1,2**52)
        cc=sorted(F(qround(F.from_float(float(np.real(x))),hh))*hh for x in rr)
        if len(set(cc))!=n: raise Refuse('duplicate_root_proposal')
        ci=newton_coeffs(m,n,em,True)
        success=False
        for doublings in range(41):
            rad=2*h*(2**doublings)
            ints=[I(c-rad,c+rad) for c in cc]
            if any(x.lo < -1 or x.hi > 1 for x in ints): continue
            if any(ints[i].hi>=ints[i+1].lo for i in range(n-1)): continue
            ok=True
            for x in ints:
                a=poly_at(ci,x.lo); b=poly_at(ci,x.hi)
                if not ((a.hi<0 and b.lo>0) or (a.lo>0 and b.hi<0)):
                    ok=False; break
            if ok: success=True; break
        if not success: raise Refuse('root_intervals_uncertified',{'proposal':cc})
    for j in range(2*L+1):
        tot=sum((x.power(j) for x in ints),I.point(0))
        mj=I.point(m[j]) if j==0 else I(m[j]-em,m[j]+em)
        if not tot.intersects(mj): raise Refuse('higher_moment_inconsistency',{'j':j})
    return cc,ints,{'root_doublings':doublings,'redundancy_checked':True}


def view_from_moments(m,b,n,L,em,eb,h,exact=False):
    cc,roots,aux=root_cert(m,n,L,em,h,exact)
    if n==0: return {'centers':[],'roots':[],'pred':[],'errors':[],'alpha_lower':F(1),'deltaC':F(0),'spectral_ok':True,**aux}
    lr=lagrange_rows(roots); lc=lagrange_rows(cc)
    absrows=[[z.absmax() for z in row] for row in lr]
    b_inf=max(sum(row) for row in absrows)
    b_one=max(sum(absrows[i][j] for i in range(n)) for j in range(n))
    alpha=track(1/(b_inf*b_one))
    dc=(L+1)*em
    pred=[track(sum(row[j]*b[j] for j in range(n))) for row in lc]
    errs=[]; valints=[]
    for i in range(n):
        ti=sum((lr[i][j]*I(b[j]-eb,b[j]+eb) for j in range(n)),I.point(0))
        valints.append(ti)
        errs.append(track(max(abs(ti.lo-pred[i]),abs(ti.hi-pred[i]))))
    return {'centers':cc,'roots':roots,'pred':pred,'errors':errs,
            'value_intervals':valints,'alpha_lower':alpha,'deltaC':dc,
            'spectral_ok':alpha>4*dc,**aux}


def address(roots,sigma,obs,zeta,Delta):
    """No labels: sensor interval plus certified canonical intervals and Delta promise."""
    X=I(max(F(-1),obs-zeta),min(F(1),obs+zeta))
    J=[I(max(F(-1),r.lo-sigma),min(F(1),r.hi+sigma)) for r in roots]
    feasible=[i for i,r in enumerate(J) if X.intersects(r)]
    if len(feasible)>1: return 'ambiguous',None,{'possible_old_count':len(feasible)}
    if len(feasible)==1:
        i=feasible[0]; ji=J[i]
        maxdist=max(abs(X.lo-ji.hi),abs(X.hi-ji.lo))
        if maxdist<Delta:
            return 'old',i,{'possible_old_count':1,'old_new_exclusion_upper':maxdist}
        return 'ambiguous',None,{'possible_old_count':1,'old_new_exclusion_upper':maxdist}
    mindists=[max(F(0),X.lo-r.hi,r.lo-X.hi) for r in J]
    if all(t>=Delta for t in mindists):
        return 'new',None,{'possible_old_count':0,'minimum_new_separation':min(mindists) if mindists else None}
    return 'ambiguous',None,{'possible_old_count':0,'new_separation_lower':min(mindists) if mindists else None}


class FixedMoments:
    method='certified_moments'
    def __init__(self,L,p,Delta,eps):
        self.L=L; self.p=p; self.h=F(1,2**p); self.Delta=Delta; self.eps=eps
        self.m=[0]*(2*L+1); self.b=[0]*(L+1)
        self.n=0; self.em_u=0; self.eb_u=0; self.sigma_u=0
    def snapshot(self):
        return struct.pack('<'+'q'*(len(self.m)+len(self.b)+4),*(self.m+self.b+[self.n,self.em_u,self.eb_u,self.sigma_u]))
    def state_public(self):
        return {'m_codes':self.m.copy(),'b_codes':self.b.copy(),'n':self.n,'em_units':self.em_u,'eb_units':self.eb_u,'sigma_units':self.sigma_u,'fractional_bits':self.p}
    def vecs(self): return [F(x)*self.h for x in self.m],[F(x)*self.h for x in self.b]
    def view(self):
        m,b=self.vecs()
        if m[0]!=self.n: raise Refuse('count_integrity')
        return view_from_moments(m,b,self.n,self.L,self.em_u*self.h,self.eb_u*self.h,self.h)
    def write(self,obs,value,zeta):
        before=self.snapshot()
        try:
            if zeta<0 or abs(value)>1 or not -1<=obs<=1: raise Refuse('input_domain')
            vw=self.view()
            kind,idx,ad=address(vw['roots'],self.sigma_u*self.h,obs,zeta,self.Delta)
            if kind=='ambiguous': raise Refuse('address_unresolved',ad)
            if kind=='new' and self.n==self.L: raise Refuse('new_key_capacity_full',ad)
            if not vw['spectral_ok']: raise Refuse('current_spectrum_uncertified',{'alpha_lower':vw['alpha_lower'],'deltaC':vw['deltaC']})
            cand=copy.deepcopy(self)
            m,b=self.vecs()
            if kind=='new':
                anchor=F(qround(obs,self.h))*self.h
                pm=powers(anchor,2*self.L); pb=powers(anchor,self.L)
                cand.m=[qround(m[j]+pm[j],self.h) for j in range(2*self.L+1)]
                cand.b=[qround(b[j]+value*pb[j],self.h) for j in range(self.L+1)]
                cand.n+=1
                cand.em_u=qceil(self.em_u*self.h+self.h/2,self.h)
                cand.eb_u=qceil(self.eb_u*self.h+self.h/2,self.h)
                cand.sigma_u=max(self.sigma_u,qceil(zeta+abs(anchor-obs),self.h))
                rr=F(0); used_error=F(0)
            else:
                anchor=vw['centers'][idx]; rr=value-vw['pred'][idx]
                rad=max(abs(anchor-vw['roots'][idx].lo),abs(anchor-vw['roots'][idx].hi))
                pb=powers(anchor,self.L)
                cand.b=[qround(b[j]+rr*pb[j],self.h) for j in range(self.L+1)]
                used_error=vw['errors'][idx]
                newbound=self.eb_u*self.h+used_error+abs(rr)*self.L*rad+self.h/2
                cand.eb_u=qceil(newbound,self.h)
            post=cand.view()
            detail={'kind':kind,'address':ad,'candidate_alpha_lower':post['alpha_lower'],
                    'candidate_deltaC':post['deltaC'],'candidate_max_content_error':max(post['errors'],default=F(0)),
                    'root_doublings':post['root_doublings'],'old_value_error_used':used_error,
                    'candidate_state_for_external_audit':cand.state_public(),'candidate_anchor_used':anchor}
            if not post['spectral_ok']: raise Refuse('candidate_spectrum_uncertified',detail)
            if max(post['errors'],default=F(0))>self.eps: raise Refuse('candidate_content_budget_exceeded',detail)
            self.m,self.b,self.n=cand.m,cand.b,cand.n
            self.em_u,self.eb_u,self.sigma_u=cand.em_u,cand.eb_u,cand.sigma_u
            return {'accepted':True,'reason':'committed','kind':kind,'anchor_used':anchor,'detail':detail,'post_digest':hashlib.sha256(self.snapshot()).hexdigest()}
        except Refuse as ex:
            assert before==self.snapshot(),'Refused transaction mutated persistent state'
            return {'accepted':False,'reason':ex.reason,'detail':ex.detail,'bitwise_unchanged':True,'post_digest':hashlib.sha256(before).hexdigest()}
    def query(self,obs,zeta):
        try:
            vw=self.view(); kind,idx,ad=address(vw['roots'],self.sigma_u*self.h,obs,zeta,self.Delta)
            if kind=='new': return {'answered':False,'reason':'certified_absent'}
            if kind!='old': return {'answered':False,'reason':'address_unresolved'}
            if not vw['spectral_ok']: return {'answered':False,'reason':'spectrum_uncertified'}
            if vw['errors'][idx]>self.eps:return {'answered':False,'reason':'content_uncertified'}
            return {'answered':True,'value':vw['pred'][idx],'error_bound':vw['errors'][idx]}
        except Refuse as ex:return {'answered':False,'reason':ex.reason}


class ExactMoments:
    method='exact_rational_moments_same_noisy_observation'
    def __init__(self,L,p,Delta,eps):
        self.L=L; self.p=p; self.h=F(1,2**p); self.Delta=Delta; self.eps=eps
        self.m=[F(0)]*(2*L+1); self.b=[F(0)]*(L+1); self.n=0; self.sigma=F(0)
    def snapshot(self):return json.dumps(self.state_public(),sort_keys=True).encode()
    def state_public(self):return {'m':[str(x) for x in self.m],'b':[str(x) for x in self.b],'n':self.n,'sigma':str(self.sigma)}
    def view(self):return view_from_moments(self.m,self.b,self.n,self.L,F(0),F(0),self.h,True)
    def write(self,obs,value,zeta):
        before=self.snapshot()
        vw=self.view(); kind,idx,ad=address(vw['roots'],self.sigma,obs,zeta,self.Delta)
        reason='address_unresolved' if kind=='ambiguous' else ('new_key_capacity_full' if kind=='new' and self.n==self.L else None)
        if reason:
            assert self.snapshot()==before
            return {'accepted':False,'reason':reason,'detail':ad,'bitwise_unchanged':True,'post_digest':hashlib.sha256(before).hexdigest()}
        if kind=='new':
            anchor=obs
            self.m=[x+anchor**j for j,x in enumerate(self.m)]
            self.b=[x+value*anchor**j for j,x in enumerate(self.b)]
            self.n+=1; self.sigma=max(self.sigma,zeta)
        else:
            anchor=vw['centers'][idx]; r=value-vw['pred'][idx]
            self.b=[x+r*anchor**j for j,x in enumerate(self.b)]
        return {'accepted':True,'reason':'committed','kind':kind,'anchor_used':anchor,'detail':ad,'post_digest':hashlib.sha256(self.snapshot()).hexdigest()}
    def query(self,obs,zeta):
        vw=self.view(); kind,idx,ad=address(vw['roots'],self.sigma,obs,zeta,self.Delta)
        if kind!='old':return {'answered':False,'reason':'certified_absent' if kind=='new' else 'address_unresolved'}
        return {'answered':True,'value':vw['pred'][idx],'error_bound':F(0)}


class FixedTable:
    method='certified_explicit_table'
    def __init__(self,L,p,Delta,eps):
        self.L=L; self.p=p; self.h=F(1,2**p); self.Delta=Delta; self.eps=eps
        self.keys=[0]*L; self.vals=[0]*L; self.n=0; self.sigma_u=0
    def snapshot(self):return struct.pack('<'+'q'*(2*self.L+2),*(self.keys+self.vals+[self.n,self.sigma_u]))
    def state_public(self):return {'key_codes':self.keys.copy(),'value_codes':self.vals.copy(),'n':self.n,'sigma_units':self.sigma_u,'fractional_bits':self.p}
    def roots(self):return [I.point(F(self.keys[i])*self.h) for i in range(self.n)]
    def write(self,obs,value,zeta):
        before=self.snapshot(); kind,idx,ad=address(self.roots(),self.sigma_u*self.h,obs,zeta,self.Delta)
        reason='address_unresolved' if kind=='ambiguous' else ('new_key_capacity_full' if kind=='new' and self.n==self.L else None)
        if reason:
            assert self.snapshot()==before
            return {'accepted':False,'reason':reason,'detail':ad,'bitwise_unchanged':True,'post_digest':hashlib.sha256(before).hexdigest()}
        anchor=F(qround(obs,self.h))*self.h if kind=='new' else F(self.keys[idx])*self.h
        cand=copy.deepcopy(self)
        if kind=='new':
            idx=self.n; cand.keys[idx]=qround(anchor,self.h); cand.n+=1
            cand.sigma_u=max(self.sigma_u,qceil(zeta+abs(anchor-obs),self.h))
        cand.vals[idx]=qround(value,self.h)
        self.keys,self.vals,self.n,self.sigma_u=cand.keys,cand.vals,cand.n,cand.sigma_u
        return {'accepted':True,'reason':'committed','kind':kind,'anchor_used':anchor,'detail':ad,'post_digest':hashlib.sha256(self.snapshot()).hexdigest()}
    def query(self,obs,zeta):
        kind,idx,ad=address(self.roots(),self.sigma_u*self.h,obs,zeta,self.Delta)
        if kind!='old':return {'answered':False,'reason':'certified_absent' if kind=='new' else 'address_unresolved'}
        return {'answered':True,'value':F(self.vals[idx])*self.h,'error_bound':self.h/2}


def make_events(c):
    z=F(1,10000); s=F(c['s'])
    keys={'A':-s,'B':s,'C':F(0),'D':F(c['future_key_D']),'E':F(c['future_key_E'])}
    raw=[('A',F(1,4),z/2,z),('B',F(-1,2),-z/2,z),('A',F(3,4),-z/2,z),
         ('A',F(3,4),-z/2,z),('A',F(-3,4),s,s+z),('B',F(1,2),z/3,z),
         ('C',F(-1,4),z/3,z),('C',F(7,8),-z/3,z),('A',F(-1,8),z/4,z),
         ('D',F(3,8),-z/2,z),('E',F(1,8),z/2,z),('A',F(1,8),-z/4,z)]
    return keys,[{'id':i+1,'label':lab,'canonical_key':keys[lab],'value':v,'observation':keys[lab]+e,'sensor_bound':zz,'actual_sensor_error':abs(e)} for i,(lab,v,e,zz) in enumerate(raw)]


def check_state(learner,anchors,latest):
    # Evaluator-only reference construction, never fed back to learner.
    labs=list(latest)
    if isinstance(learner,FixedMoments):
        m,b=learner.vecs(); em=learner.em_u*learner.h; eb=learner.eb_u*learner.h
        me=max([abs(m[j]-sum(anchors[l]**j for l in labs)) for j in range(2*learner.L+1)],default=F(0))
        be=max([abs(b[j]-sum(latest[l]*anchors[l]**j for l in labs)) for j in range(learner.L+1)],default=F(0))
        assert me<=em,('moment budget violated',me,em)
        assert be<=eb,('content budget violated',be,eb)
        return {'actual_max_moment_error':me,'moment_error_budget':em,'actual_max_b_error':be,'b_error_budget':eb}
    if isinstance(learner,ExactMoments):
        assert learner.m==[sum((anchors[l]**j for l in labs),F(0)) for j in range(2*learner.L+1)]
        assert learner.b==[sum((latest[l]*anchors[l]**j for l in labs),F(0)) for j in range(learner.L+1)]
        return {'exact_latest_moment_invariants':True}
    return {'table_quantization_bound':learner.h/2}


def score_queries(learner,keys,latest):
    out={}; z=F(1,10000)
    for lab,target in latest.items():
        e=z/2 if ord(lab)%2 else -z/2
        ans=learner.query(keys[lab]+e,z)
        row={**ans,'canonical_key_evaluator_only':keys[lab],'target_committed':target,'observed_query':keys[lab]+e}
        if ans['answered']:
            row['absolute_error']=abs(ans['value']-target)
            assert row['absolute_error']<=ans['error_bound'],('certificate violated',learner.method,lab,row)
        out[lab]=row
    return out


def exact_fixture():
    L=3; ks=[F(-1,2),F(0),F(1,2)]; ms=[F(0)]*7; bs=[F(0)]*4; latest={}
    seq=[(ks[0],F(1,4)),(ks[1],F(-1,2)),(ks[0],F(3,4)),(ks[2],F(1,8))]
    checks=[]
    for k,v in seq:
        old=k in latest
        if old:
            fs=sp.Matrix([[sp.Rational(x.numerator,x.denominator)**j for x in latest] for j in range(4)])
            C=fs*fs.T; b=sp.Matrix([sp.Rational(x.numerator,x.denominator) for x in bs]); w=C.pinv()*b
            fk=sp.Matrix([sp.Rational(k.numerator,k.denominator)**j for j in range(4)])
            r=sp.Rational(v.numerator,v.denominator)-(fk.T*w)[0]
            bs=[x+F(int(r.p),int(r.q))*k**j for j,x in enumerate(bs)]
        else:
            ms=[x+k**j for j,x in enumerate(ms)]; bs=[x+v*k**j for j,x in enumerate(bs)]
        latest[k]=v
        C=sp.Matrix([[sp.Rational(ms[i+j].numerator,ms[i+j].denominator) for j in range(4)] for i in range(4)])
        b=sp.Matrix([sp.Rational(x.numerator,x.denominator) for x in bs]); w=C.pinv()*b
        for key,t in latest.items():
            f=sp.Matrix([sp.Rational(key.numerator,key.denominator)**j for j in range(4)])
            assert (f.T*w)[0]==sp.Rational(t.numerator,t.denominator)
        checks.append({'old':old,'distinct':len(latest),'exact_interpolation':True})
    fs=sp.Matrix([[sp.Rational(x.numerator,x.denominator)**j for x in ks] for j in range(4)])
    g=sp.Matrix([sp.Rational(3,4)**j for j in range(4)])
    assert fs.row_join(g).det()!=0
    return {'updates':checks,'new_at_capacity_still_independent':True,'determinant':str(fs.row_join(g).det())}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--spec',default=str(Path(__file__).with_name('frozen_spec.json'))); ap.add_argument('--out',default='results_v3.json'); args=ap.parse_args()
    specbytes=Path(args.spec).read_bytes(); spec=json.loads(specbytes)
    results={'scope':spec['scope'],'spec_sha256':hashlib.sha256(specbytes).hexdigest(),
             'environment':{'python':sys.version.split()[0],'numpy':np.__version__,'sympy':sp.__version__},
             'randomness':'none','exact_fixture':exact_fixture(),'runs':[]}
    methods=[FixedMoments,FixedTable,ExactMoments]
    for conf in spec['conditions']:
        keys,events=make_events(conf); Delta=F(conf['minimum_canonical_separation'])
        assert all(abs(a-b)>=Delta for i,a in enumerate(keys.values()) for j,b in enumerate(keys.values()) if i<j)
        for cls in methods:
            lm=cls(spec['Nmax'],conf['fractional_bits'],Delta,F(spec['required_value_error']))
            latest={}; anchors={}; records=[]; requests={}
            for ev in events:
                before=lm.snapshot(); prequeries=score_queries(lm,keys,latest); requests[ev['label']]=ev['value']
                dec=lm.write(ev['observation'],ev['value'],ev['sensor_bound'])
                lab=ev['label']; old_latest=latest.copy()
                if dec['accepted']:
                    assert (lab in latest)==(dec['kind']=='old'),('wrong identity/routing accepted',conf['name'],cls.method,ev,dec)
                    if lab not in latest: anchors[lab]=dec['anchor_used']
                    latest[lab]=ev['value']
                else: assert before==lm.snapshot(),'refusal mutated state'
                candidate_error=None
                candidate_info=dec.get('detail',{}).get('candidate_state_for_external_audit')
                if not dec['accepted'] and candidate_info is not None and isinstance(lm,FixedMoments):
                    # Counterfactual scorer only: never commits, feeds back, or retunes.
                    tmp=FixedMoments(lm.L,lm.p,lm.Delta,lm.eps)
                    tmp.m=candidate_info['m_codes']; tmp.b=candidate_info['b_codes']; tmp.n=candidate_info['n']
                    tmp.em_u=candidate_info['em_units']; tmp.eb_u=candidate_info['eb_units']; tmp.sigma_u=candidate_info['sigma_units']
                    al=anchors.copy(); tl=latest.copy()
                    if lab not in al: al[lab]=dec['detail']['candidate_anchor_used']
                    tl[lab]=ev['value']
                    tv=tmp.view(); ordered=sorted(al,key=lambda label:al[label])
                    assert all(iv.lo<=al[label]<=iv.hi for iv,label in zip(tv['roots'],ordered))
                    candidate_error=max(abs(tv['pred'][ii]-tl[label]) for ii,label in enumerate(ordered))
                budget=check_state(lm,anchors,latest)
                q=score_queries(lm,keys,latest)
                requested_queries={}
                # Requested targets are NOT committed-state truth. Report their gaps,
                # but never apply the committed-content certificate to them.
                for requested_label, requested_value in requests.items():
                    ze=F(1,10000); qe=ze/2 if ord(requested_label)%2 else -ze/2
                    qa=lm.query(keys[requested_label]+qe,ze)
                    requested_queries[requested_label]={**qa,'requested_value':requested_value,
                        'requested_gap':abs(qa['value']-requested_value) if qa['answered'] else None}
                if isinstance(lm,(FixedMoments,ExactMoments)):
                    cv=lm.view()
                    root_intervals=cv['roots']
                    sorted_anchor_values=sorted(anchors.values())
                    assert len(root_intervals)==len(sorted_anchor_values)
                    assert all(iv.lo<=aa<=iv.hi for iv,aa in zip(root_intervals,sorted_anchor_values))
                    post_certificate={k:cv[k] for k in ['roots','centers','alpha_lower','deltaC','spectral_ok','errors']}
                else:
                    post_certificate={'roots':lm.roots(),'value_error_bound':lm.h/2}
                for key,row in q.items():
                    if dec['accepted']:
                        assert row['answered'],('accepted but uncertified subsequent ordinary read',conf['name'],cls.method,ev,row)
                        assert row['absolute_error']<=F(spec['required_value_error'])
                other_errors=[r['absolute_error'] for key,r in q.items() if key!=lab and r['answered']]
                drift=[abs(q[key]['value']-prequeries[key]['value']) for key in old_latest if key!=lab and q[key]['answered'] and prequeries[key]['answered']]
                if not dec['accepted']:
                    assert q==prequeries,'rejection altered query results'
                request_fulfilled=dec['accepted']
                reject_gap=None
                if not dec['accepted'] and lab in latest and q[lab]['answered']:
                    reject_gap=abs(q[lab]['value']-ev['value'])
                rec={'event':ev,'decision':dec,'state':lm.state_public(),'state_digest_before':hashlib.sha256(before).hexdigest(),
                     'state_digest_after':hashlib.sha256(lm.snapshot()).hexdigest(),'budget_evaluator':budget,
                     'queries':q,'requested_queries':requested_queries,'post_certificate':post_certificate,'max_other_retention_error':max(other_errors,default=F(0)),
                     'max_other_output_change':max(drift,default=F(0)),
                     'accepted_latest_error':q[lab]['absolute_error'] if dec['accepted'] else None,
                     'requested_write_completed':request_fulfilled,'rejected_request_gap_if_readable':reject_gap,
                     'uncommitted_candidate_raw_value_error_evaluator_only':candidate_error,
                     'rejected_new_request_absent':not dec['accepted'] and lab not in latest}
                records.append(rec)
            answered=[r for rec in records for r in rec['queries'].values() if r['answered']]
            attempts=sum(len(rec['queries']) for rec in records)
            accepted=[rec for rec in records if rec['decision']['accepted']]
            pending=[{'label':lab,'requested':v,'committed':latest.get(lab),'missing':lab not in latest} for lab,v in requests.items() if lab not in latest or latest[lab]!=v]
            summary={'attempts':len(records),'accepted':len(accepted),'rejected':len(records)-len(accepted),
                     'write_coverage':len(accepted)/len(records),'read_attempts':attempts,'answered_reads':len(answered),
                     'read_coverage_committed_keys':len(answered)/attempts if attempts else None,
                     'requested_read_attempts':sum(len(r['requested_queries']) for r in records),
                     'requested_answered_reads':sum(q['answered'] for r in records for q in r['requested_queries'].values()),
                     'max_accepted_latest_error':max((r['accepted_latest_error'] for r in accepted),default=F(0)),
                     'max_other_retention_error':max((r['max_other_retention_error'] for r in records),default=F(0)),
                     'rejected_events':[{'event':r['event']['id'],'reason':r['decision']['reason']} for r in records if not r['decision']['accepted']],
                     'pending_requested_targets':pending,'all_refusal_snapshots_identical':True,
                     'all_budget_and_read_checks_passed':True}
            results['runs'].append({'condition':conf,'method':cls.method,'summary':summary,'records':records})
    results['certificate_workspace']={'rational_interval_samples_instrumented':RATIONAL_SAMPLES,
                                     'max_instrumented_numerator_bits':MAX_RATIONAL_NUM_BITS,
                                     'max_instrumented_denominator_bits':MAX_RATIONAL_DEN_BITS,
                                     'caution':'Maximum seen operand size, not total peak heap or a fixed-bit physical implementation.'}
    data=json.dumps(results,default=fnum,sort_keys=True,indent=2,allow_nan=False)+'\n'
    Path(args.out).write_text(data)
    for r in results['runs']:
        s=r['summary']; print(r['condition']['name'],r['method'],s['accepted'],'/',s['attempts'],
                            'maxlatest=',float(s['max_accepted_latest_error']),'reject=',s['rejected_events'])
    print('workspace:',results['certificate_workspace'])
    print('output_sha256',hashlib.sha256(data.encode()).hexdigest())

if __name__=='__main__':main()
