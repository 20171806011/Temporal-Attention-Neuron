#!/usr/bin/env python3
"""v4 minimal vs redundant scalar moments under ONE common interval certificate.
Not TAN/P1-B/v11 production. No Codex. Learners receive observed key, value, zeta only.
Primitive exact interval and rounding routines are retained from the included v3
source snapshot. All representation/certificate changes are stated in frozen_spec_v4.json.
Persistent moments use int64 codes; root/value certificates use variable precision.
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


import time

PROFILE: dict[str, Any] | None = None
PHASE = 'unprofiled'


def profile_add(kind, elapsed_ns):
    if PROFILE is None: return
    key = PHASE + ':' + kind
    d = PROFILE.setdefault(key, {'calls': 0, 'wall_ns': 0})
    d['calls'] += 1
    d['wall_ns'] += int(elapsed_ns)


def root_cert_common(m, n, em, h):
    """Uses ONLY count and m1..mN. No high-order consistency or spectral gate."""
    start = time.perf_counter_ns()
    try:
        if n == 0:
            return [], [], {'root_doublings': 0, 'all_roots_certified': True,
                           'moment_indices_used': [], 'coefficient_intervals': [I.point(1)]}
        cm = newton_coeffs(m, n)
        rr = np.roots(np.array([float(x) for x in cm[::-1]], dtype=float))
        if len(rr) != n or np.max(np.abs(np.imag(rr))) > 1e-8:
            raise Refuse('uncertified_root_proposal')
        hh = F(1, 2**52)
        centers = sorted(F(qround(F.from_float(float(np.real(x))), hh))*hh for x in rr)
        if len(set(centers)) != n: raise Refuse('duplicate_root_proposal')
        ci = newton_coeffs(m, n, em, True)
        for doubling in range(41):
            rad = 2*h*(2**doubling)
            roots = [I(c-rad, c+rad) for c in centers]
            if any(x.lo < -1 or x.hi > 1 for x in roots): continue
            if any(roots[i].hi >= roots[i+1].lo for i in range(n-1)): continue
            endpoints = [(poly_at(ci, x.lo), poly_at(ci, x.hi)) for x in roots]
            if all((a.hi < 0 and b.lo > 0) or (a.lo > 0 and b.hi < 0)
                   for a, b in endpoints):
                return centers, roots, {'root_doublings': doubling,
                    'all_roots_certified': True, 'moment_indices_used': list(range(1, n+1)),
                    'coefficient_intervals': ci,
                    'endpoint_polynomial_intervals': [[a,b] for a,b in endpoints]}
        raise Refuse('root_intervals_uncertified', {'proposal': centers})
    finally:
        profile_add('root_reconstruction_certificate', time.perf_counter_ns()-start)


def view_common(m, b, n, mu, beta, h):
    centers, roots, aux = root_cert_common(m, n, mu, h)
    start = time.perf_counter_ns()
    try:
        if n == 0:
            return {'centers': [], 'roots': [], 'pred': [], 'errors': [],
                    'value_intervals': [], 'value_indices_used': [], **aux}
        rows = lagrange_rows(roots)
        pointrows = lagrange_rows(centers)
        pred = [track(sum(row[j]*b[j] for j in range(n))) for row in pointrows]
        valints = [sum((rows[i][j]*I(b[j]-beta,b[j]+beta) for j in range(n)), I.point(0))
                   for i in range(n)]
        errors = [track(max(abs(z.lo-v),abs(z.hi-v))) for z,v in zip(valints,pred)]
        return {'centers': centers, 'roots': roots, 'pred': pred, 'errors': errors,
                'value_intervals': valints, 'value_indices_used': list(range(n)), **aux}
    finally:
        profile_add('value_reconstruction_certificate', time.perf_counter_ns()-start)


def address(roots,sigma,obs,zeta,Delta):
    """Identical v3 address rule. Delta applies to ALL future canonical keys."""
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


class MomentMemory:
    def __init__(self, L, p, Delta, eps, representation):
        if representation not in ('redundant', 'minimal'): raise ValueError('representation')
        self.L=L; self.p=p; self.h=F(1,2**p); self.Delta=Delta; self.eps=eps
        self.representation=representation
        self.method=representation+'_common_certificate'
        self.m_exponents=tuple(range(2*L+1)) if representation=='redundant' else tuple(range(1,L+1))
        self.b_exponents=tuple(range(L+1)) if representation=='redundant' else tuple(range(L))
        self.m=[0]*len(self.m_exponents); self.b=[0]*len(self.b_exponents)
        self.n=0; self.mu_u=0; self.beta_u=0; self.sigma_u=0
    def snapshot(self):
        fields=self.m+self.b+[self.n,self.mu_u,self.beta_u,self.sigma_u]
        return struct.pack('<'+'q'*len(fields),*fields)
    def state_public(self):
        return {'representation':self.representation, 'm_exponents':list(self.m_exponents),
                'b_exponents':list(self.b_exponents),'m_codes':self.m.copy(),'b_codes':self.b.copy(),
                'n':self.n,'mu_units':self.mu_u,'beta_units':self.beta_u,
                'sigma_units':self.sigma_u,'fractional_bits':self.p}
    def vecs(self):
        # m0 is transient control data for minimal; not stored as a seventh scalar.
        mm=[F(v)*self.h for v in self.m]
        if self.representation=='minimal': mm=[F(self.n)]+mm
        return mm,[F(v)*self.h for v in self.b]
    def view(self):
        if not 0 <= self.n <= self.L: raise Refuse('count_domain')
        m,b=self.vecs()
        return view_common(m,b,self.n,self.mu_u*self.h,self.beta_u*self.h,self.h)
    def write(self,obs,value,zeta):
        before=self.snapshot(); cand=None; detail={}
        try:
            if zeta<0 or abs(value)>1 or not -1<=obs<=1: raise Refuse('input_domain')
            vw=self.view()
            kind,idx,ad=address(vw['roots'],self.sigma_u*self.h,obs,zeta,self.Delta)
            if kind=='ambiguous': raise Refuse('address_unresolved',ad)
            if kind=='new' and self.n==self.L: raise Refuse('new_key_capacity_full',ad)
            cand=copy.deepcopy(self)
            if kind=='new':
                anchor=F(qround(obs,self.h))*self.h
                cand.m=[qround(F(z)*self.h+anchor**j,self.h) for z,j in zip(self.m,self.m_exponents)]
                cand.b=[qround(F(z)*self.h+value*anchor**j,self.h) for z,j in zip(self.b,self.b_exponents)]
                cand.n+=1
                cand.mu_u=qceil(self.mu_u*self.h+self.h/2,self.h)
                cand.beta_u=qceil(self.beta_u*self.h+self.h/2,self.h)
                cand.sigma_u=max(self.sigma_u,qceil(zeta+abs(anchor-obs),self.h))
                rr=F(0); used_error=F(0); rad=F(0)
            else:
                anchor=vw['centers'][idx]; rr=value-vw['pred'][idx]
                rad=max(abs(anchor-vw['roots'][idx].lo),abs(anchor-vw['roots'][idx].hi))
                cand.b=[qround(F(z)*self.h+rr*anchor**j,self.h) for z,j in zip(self.b,self.b_exponents)]
                used_error=vw['errors'][idx]
                # Intentionally same L, not L-1, for isolation from budget tightening.
                cand.beta_u=qceil(self.beta_u*self.h+used_error+abs(rr)*self.L*rad+self.h/2,self.h)
            post=cand.view()
            detail={'kind':kind,'address':ad,'candidate_max_content_error':max(post['errors'],default=F(0)),
                    'root_doublings':post['root_doublings'],'old_value_error_used':used_error,
                    'root_radius_used':rad,'residual_used':rr,'candidate_anchor_used':anchor,
                    'candidate_state_for_external_audit':cand.state_public(),
                    'candidate_certificate':post}
            if max(post['errors'],default=F(0))>self.eps:
                raise Refuse('candidate_content_budget_exceeded',detail)
            self.m,self.b,self.n=cand.m,cand.b,cand.n
            self.mu_u,self.beta_u,self.sigma_u=cand.mu_u,cand.beta_u,cand.sigma_u
            return {'accepted':True,'reason':'committed','kind':kind,'anchor_used':anchor,'detail':detail}
        except Refuse as ex:
            assert before==self.snapshot(),'refusal changed persistent bytes'
            extra=dict(ex.detail)
            if cand is not None: extra.setdefault('candidate_state_for_external_audit',cand.state_public())
            return {'accepted':False,'reason':ex.reason,'detail':extra,'bitwise_unchanged':True}
    def query(self,obs,zeta):
        before=self.snapshot()
        try:
            vw=self.view(); kind,idx,ad=address(vw['roots'],self.sigma_u*self.h,obs,zeta,self.Delta)
            if kind=='new': return {'answered':False,'reason':'certified_absent'}
            if kind!='old': return {'answered':False,'reason':'address_unresolved'}
            if vw['errors'][idx]>self.eps: return {'answered':False,'reason':'content_uncertified'}
            return {'answered':True,'value':vw['pred'][idx],'error_bound':vw['errors'][idx]}
        except Refuse as ex: return {'answered':False,'reason':ex.reason}
        finally: assert before==self.snapshot(),'query mutated persistent bytes'


class FixedTable:
    method='explicit_table_same_observations'
    def __init__(self,L,p,Delta,eps):
        self.L=L; self.p=p; self.h=F(1,2**p); self.Delta=Delta; self.eps=eps
        self.keys=[0]*L; self.vals=[0]*L; self.n=0; self.sigma_u=0
    def snapshot(self):return struct.pack('<'+'q'*(2*self.L+2),*(self.keys+self.vals+[self.n,self.sigma_u]))
    def state_public(self):return {'key_codes':self.keys.copy(),'value_codes':self.vals.copy(),'n':self.n,'sigma_units':self.sigma_u,'fractional_bits':self.p}
    def roots(self):return [I.point(F(self.keys[i])*self.h) for i in range(self.n)]
    def write(self,obs,value,zeta):
        before=self.snapshot()
        if zeta<0 or abs(value)>1 or not -1<=obs<=1:
            return {'accepted':False,'reason':'input_domain','bitwise_unchanged':True,'detail':{}}
        kind,idx,ad=address(self.roots(),self.sigma_u*self.h,obs,zeta,self.Delta)
        reason='address_unresolved' if kind=='ambiguous' else ('new_key_capacity_full' if kind=='new' and self.n==self.L else None)
        if reason:
            assert self.snapshot()==before
            return {'accepted':False,'reason':reason,'detail':ad,'bitwise_unchanged':True}
        anchor=F(qround(obs,self.h))*self.h if kind=='new' else F(self.keys[idx])*self.h
        cand=copy.deepcopy(self)
        if kind=='new':
            idx=self.n; cand.keys[idx]=qround(anchor,self.h); cand.n+=1
            cand.sigma_u=max(self.sigma_u,qceil(zeta+abs(anchor-obs),self.h))
        cand.vals[idx]=qround(value,self.h)
        self.keys,self.vals,self.n,self.sigma_u=cand.keys,cand.vals,cand.n,cand.sigma_u
        return {'accepted':True,'reason':'committed','kind':kind,'anchor_used':anchor,'detail':ad}
    def query(self,obs,zeta):
        before=self.snapshot()
        kind,idx,ad=address(self.roots(),self.sigma_u*self.h,obs,zeta,self.Delta)
        assert self.snapshot()==before
        if kind!='old':return {'answered':False,'reason':'certified_absent' if kind=='new' else 'address_unresolved'}
        return {'answered':True,'value':F(self.vals[idx])*self.h,'error_bound':self.h/2}


def make_events(c):
    # Byte-for-byte rational schedule inherited from v3; no data selection.
    z=F(1,10000); s=F(c['s'])
    keys={'A':-s,'B':s,'C':F(0),'D':F(c['future_key_D']),'E':F(c['future_key_E'])}
    raw=[('A',F(1,4),z/2,z),('B',F(-1,2),-z/2,z),('A',F(3,4),-z/2,z),
         ('A',F(3,4),-z/2,z),('A',F(-3,4),s,s+z),('B',F(1,2),z/3,z),
         ('C',F(-1,4),z/3,z),('C',F(7,8),-z/3,z),('A',F(-1,8),z/4,z),
         ('D',F(3,8),-z/2,z),('E',F(1,8),z/2,z),('A',F(1,8),-z/4,z)]
    return keys,[{'id':i+1,'label':lab,'canonical_key':keys[lab],'value':v,'observation':keys[lab]+e,
                  'sensor_bound':zz,'actual_sensor_error':abs(e)} for i,(lab,v,e,zz) in enumerate(raw)]


def exact_minimal_fixture():
    """Exact elementary algebra, not a fourth finite-precision stream competitor."""
    L=3; m=[F(0)]*L; b=[F(0)]*L; N=0; truth={}; logs=[]
    seq=[(F(-1,2),F(1,4)),(F(0),F(-1,2)),(F(-1,2),F(3,4)),
         (F(1,2),F(1,8)),(F(3,4),F(1,3)),(F(0),F(7,8))]
    def decode():
        if N==0:return [],[]
        coeff=newton_coeffs([F(N)]+m,N)
        z=sp.Symbol('z')
        pol=sp.Poly(sum(sp.Rational(c.numerator,c.denominator)*z**j for j,c in enumerate(coeff)),z)
        rs=sp.polys.polytools.ground_roots(pol)
        assert len(rs)==N and all(v==1 for v in rs.values())
        keys=sorted(F(int(r.p),int(r.q)) for r in rs)
        vals=[sum(row[j]*b[j] for j in range(N)) for row in lagrange_rows(keys)]
        return keys,vals
    assert poly_at(newton_coeffs([F(0)]+m,0),F(0))==1
    for key,val in seq:
        before=(m.copy(),b.copy(),N)
        coeff=newton_coeffs([F(N)]+m,N)
        old=poly_at(coeff,key)==0
        ks,vs=decode()
        if not old and N==L:
            assert before==(m,b,N)
            logs.append({'key':key,'accepted':False,'reason':'new_at_capacity','N':N});continue
        if old:
            r=val-vs[ks.index(key)]
            b=[z+r*key**j for j,z in enumerate(b)]
        else:
            m=[z+key**j for j,z in enumerate(m,1)]
            b=[z+val*key**j for j,z in enumerate(b)];N+=1
        truth[key]=val
        assert m==[sum(k**j for k in truth) for j in range(1,L+1)]
        assert b==[sum(v*k**j for k,v in truth.items()) for j in range(L)]
        ks,vs=decode();assert dict(zip(ks,vs))==truth
        logs.append({'key':key,'accepted':True,'kind':'old' if old else 'new','N':N,'decoded_keys':ks,'decoded_values':vs})
    assert poly_at(newton_coeffs([F(N)]+m,N),F(3,4))!=0
    assert poly_at(newton_coeffs([F(N)]+m,N),F(0))==0
    return {'all_exact_assertions_passed':True,'events':logs,
            'empty_polynomial':'1','zero_key_zero_value_payload_equals_empty_but_count_differs':True,
            'full_capacity_new_query_certified_absent':True,'spare_mode_not_used':True}


def check_state(lm,anchors,latest):
    if isinstance(lm,MomentMemory):
        me=max((abs(F(code)*lm.h-sum((anchors[l]**j for l in latest),F(0)))
                for code,j in zip(lm.m,lm.m_exponents)),default=F(0))
        be=max((abs(F(code)*lm.h-sum((latest[l]*anchors[l]**j for l in latest),F(0)))
                for code,j in zip(lm.b,lm.b_exponents)),default=F(0))
        assert me<=lm.mu_u*lm.h and be<=lm.beta_u*lm.h
        return {'actual_max_moment_error':me,'mu':lm.mu_u*lm.h,
                'actual_max_value_moment_error':be,'beta':lm.beta_u*lm.h}
    for lab,a in anchors.items():
        idx=lm.keys[:lm.n].index(qround(a,lm.h))
        assert abs(F(lm.vals[idx])*lm.h-latest[lab])<=lm.h/2
    return {'value_quantization_bound':lm.h/2}


def query_scores(lm,keys,targets,check):
    global PHASE
    PHASE='ordinary_queries' if check else 'requested_target_queries'
    out={}; z=F(1,10000)
    for lab,target in targets.items():
        obs=keys[lab]+(z/2 if ord(lab)%2 else -z/2)
        ans=lm.query(obs,z)
        row={**ans,'observation':obs,'sensor_bound':z,'target':target}
        row['absolute_error']=abs(ans['value']-target) if ans['answered'] else None
        if check and ans['answered']: assert row['absolute_error']<=ans['error_bound']
        out[lab]=row
    return out


def execute_run(conf,spec,method):
    global PROFILE,PHASE,MAX_RATIONAL_NUM_BITS,MAX_RATIONAL_DEN_BITS,RATIONAL_SAMPLES
    MAX_RATIONAL_NUM_BITS=MAX_RATIONAL_DEN_BITS=RATIONAL_SAMPLES=0
    PROFILE={}; wall0=time.perf_counter_ns()
    L=spec['Nmax']; p=conf['fractional_bits']; Delta=F(conf['minimum_canonical_separation']); eps=F(spec['required_value_error'])
    lm=FixedTable(L,p,Delta,eps) if method=='explicit_table_same_observations' else MomentMemory(L,p,Delta,eps,method.split('_')[0])
    keys,events=make_events(conf)
    assert all(abs(a-b)>=Delta for i,a in enumerate(keys.values()) for j,b in enumerate(keys.values()) if i<j)
    latest={}; anchors={}; requests={}; records=[]; write_ns=0
    for ev in events:
        state_before=lm.state_public(); before=lm.snapshot()
        pre=query_scores(lm,keys,latest,True);old_latest=latest.copy();requests[ev['label']]=ev['value']
        PHASE='write';t0=time.perf_counter_ns()
        dec=lm.write(ev['observation'],ev['value'],ev['sensor_bound']);write_ns+=time.perf_counter_ns()-t0
        lab=ev['label']
        if dec['accepted']:
            assert (lab in latest)==(dec['kind']=='old'),'accepted wrong address'
            if lab not in latest:anchors[lab]=dec['anchor_used']
            latest[lab]=ev['value']
        else: assert before==lm.snapshot(),'rejected state changed'
        PHASE='evaluator_state_audit';budget=check_state(lm,anchors,latest)
        q=query_scores(lm,keys,latest,True);rq=query_scores(lm,keys,requests,False)
        PHASE='evaluator_certificate'
        cert=lm.view() if isinstance(lm,MomentMemory) else {'roots':lm.roots(),'value_error_bound':lm.h/2}
        aa=sorted(anchors.values())
        assert len(cert['roots'])==len(aa)
        assert all(iv.lo<=a<=iv.hi for iv,a in zip(sorted(cert['roots'],key=lambda z:z.lo),aa))
        if dec['accepted']:
            assert all(row['answered'] and row['absolute_error']<=eps for row in q.values())
        else: assert q==pre,'refusal changed readable outputs'
        others=[row['absolute_error'] for k,row in q.items() if k!=lab and row['answered']]
        drift=[abs(q[k]['value']-pre[k]['value']) for k in old_latest if k!=lab and q[k]['answered'] and pre[k]['answered']]
        records.append({'event':ev,'decision':dec,'state_before':state_before,'state_after':lm.state_public(),
            'snapshot_hex_before':before.hex(),'snapshot_hex_after':lm.snapshot().hex(),
            'evaluator_only_anchors':anchors.copy(),'evaluator_only_committed_latest':latest.copy(),
            'budget_evaluator':budget,'post_certificate':cert,'queries':q,'requested_queries':rq,
            'accepted_latest_error':q[lab]['absolute_error'] if dec['accepted'] else None,
            'max_other_retention_error':max(others,default=F(0)),
            'max_other_output_change':max(drift,default=F(0)),
            'requested_write_completed':dec['accepted'],
            'rejected_request_gap_if_readable':abs(q[lab]['value']-ev['value']) if not dec['accepted'] and lab in q and q[lab]['answered'] else None,
            'rejected_new_request_absent':not dec['accepted'] and lab not in latest})
    accepted=[r for r in records if r['decision']['accepted']]
    qs=[q for r in records for q in r['queries'].values()]
    rq=[q for r in records for q in r['requested_queries'].values()]
    pending=[{'label':lab,'requested':v,'committed':latest.get(lab),'missing':lab not in latest}
             for lab,v in requests.items() if lab not in latest or latest[lab]!=v]
    summary={'attempts':len(records),'accepted':len(accepted),'rejected':len(records)-len(accepted),
        'write_coverage':len(accepted)/len(records),'read_attempts_committed':len(qs),'answered_reads_committed':sum(q['answered'] for q in qs),
        'read_attempts_requested':len(rq),'answered_reads_requested':sum(q['answered'] for q in rq),
        'requested_target_reads_within_epsilon':sum(q['answered'] and q['absolute_error']<=eps for q in rq),
        'max_accepted_latest_error':max((r['accepted_latest_error'] for r in accepted),default=F(0)),
        'max_other_retention_error':max((r['max_other_retention_error'] for r in records),default=F(0)),
        'rejected_events':[{'event':r['event']['id'],'reason':r['decision']['reason']} for r in records if not r['decision']['accepted']],
        'pending_requested_targets':pending,'all_refusals_bitwise_unchanged':True,'all_evaluator_checks_passed':True,
        'persistent_words':len(lm.snapshot())//8,
        'workspace_operand_tracking':{'interval_rational_endpoints_observed':RATIONAL_SAMPLES,
          'max_numerator_bits':MAX_RATIONAL_NUM_BITS,'max_denominator_bits':MAX_RATIONAL_DEN_BITS,
          'scope':'Instrumented interval operands only; not peak memory or fixed-bit hardware.'}}
    timing={'condition':conf['name'],'method':method,'all_12_writes_wall_ns':write_ns,
        'whole_run_including_scorer_wall_ns':time.perf_counter_ns()-wall0,'phased_reconstruction':PROFILE}
    PROFILE=None;PHASE='unprofiled'
    return {'condition':conf,'method':method,'summary':summary,'records':records},timing


def verify_common_equivalence(runs,L):
    checks=0
    for cond in sorted(set(r['condition']['name'] for r in runs)):
        aa=next(r for r in runs if r['condition']['name']==cond and r['method']=='redundant_common_certificate')
        bb=next(r for r in runs if r['condition']['name']==cond and r['method']=='minimal_common_certificate')
        for a,b in zip(aa['records'],bb['records']):
            sa,sb=a['state_after'],b['state_after']
            assert sa['m_codes'][1:L+1]==sb['m_codes'];checks+=1
            assert sa['b_codes'][:L]==sb['b_codes'];checks+=1
            for name in ['n','mu_units','beta_units','sigma_units']:
                assert sa[name]==sb[name];checks+=1
            for name in ['accepted','reason']:
                assert a['decision'][name]==b['decision'][name];checks+=1
            assert a['queries']==b['queries'];checks+=1
            assert a['requested_queries']==b['requested_queries'];checks+=1
            assert a['post_certificate']==b['post_certificate'];checks+=1
    return {'comparisons':checks,'all_identical':True,
            'interpretation':'One common certificate and identical accessed prefixes: payload deletion is not a coverage intervention.'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--spec',default=str(Path(__file__).with_name('frozen_spec_v4.json')))
    ap.add_argument('--out',default='results_v4.json')
    ap.add_argument('--timing-out',default='timing_v4.json')
    args=ap.parse_args()
    specbytes=Path(args.spec).read_bytes();spec=json.loads(specbytes)
    results={'scope':spec['scope'],'spec_sha256':hashlib.sha256(specbytes).hexdigest(),
        'environment':{'python':sys.version.split()[0],'numpy':np.__version__,'sympy':sp.__version__},
        'randomness':'none; deterministic v3 requests','exact_minimal_fixture':exact_minimal_fixture(),'runs':[]}
    timings=[]
    for conf in spec['conditions']:
        for method in spec['methods']:
            run,t=execute_run(conf,spec,method);results['runs'].append(run);timings.append(t)
            s=run['summary']
            print(conf['name'],method,f"{s['accepted']}/{s['attempts']}",
                  'latest',float(s['max_accepted_latest_error']),'reject',s['rejected_events'],flush=True)
    results['common_certificate_equivalence']=verify_common_equivalence(results['runs'],spec['Nmax'])
    data=json.dumps(results,default=fnum,indent=2,sort_keys=True,allow_nan=False)+'\n'
    Path(args.out).write_text(data)
    Path(args.timing_out).write_text(json.dumps({'scope':'Host wall-clock timings; not energy, not an algorithmic lower bound; root/value phases include their certificates.',
        'environment':results['environment'],'runs':timings},indent=2,sort_keys=True)+'\n')
    print('Correctness JSON SHA256',hashlib.sha256(data.encode()).hexdigest())

if __name__=='__main__':main()
