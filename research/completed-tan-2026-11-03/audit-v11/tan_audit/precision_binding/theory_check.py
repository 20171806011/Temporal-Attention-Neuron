"""Exact fixed-budget arithmetic; no IID data, fitting, or parameter search."""
from fractions import Fraction as Q
from pathlib import Path
import itertools,json

OUT=Path(__file__).resolve().parent
L=Q(3,10);U=Q(5,2);V=Q(3);GAP=Q(1,20)
LEVELS16=2**16-1;LEVELS8=2**8-1
RA=Q(128);RB=Q(320)

def rational_record(x):return {'fraction':str(x),'decimal':float(x)}

def quantize(x,lo,hi,bits):
    assert lo<=x<=hi
    n=(x-lo)*(2**bits-1)/(hi-lo)
    floor=n.numerator//n.denominator;remainder=n-floor
    # Deterministic nearest-even; the half-step proof is tie-rule independent.
    code=floor+int(remainder>Q(1,2) or (remainder==Q(1,2) and floor%2==1))
    assert 0<=code<2**bits
    decoded=lo+(hi-lo)*code/(2**bits-1)
    assert abs(decoded-x)<=(hi-lo)/(2*(2**bits-1))
    return decoded,code

def bound(name,ek,ev):
    gap=GAP-ek
    slope=(2*V+ev)/gap
    intercept=V+U*slope
    prefix=ev+slope*ek
    terminal=U*RA/LEVELS16+RB/LEVELS16
    total=prefix+terminal
    assert gap>0 and slope<RA and intercept<RB
    return {k:rational_record(v) for k,v in dict(
        epsilon_first_key=ek,epsilon_first_value=ev,minimum_effective_gap=gap,
        intermediate_slope_bound=slope,intermediate_intercept_bound=intercept,
        first_query_prefix_error_bound=prefix,second_query_prefix_error_bound=Q(0),
        terminal_quantization_error_bound=terminal,
        worst_query_and_Ecomp_bound=total,
        slope_guard_margin=RA-slope,intercept_guard_margin=RB-intercept).items()}

def interpolation(k,v,kind):
    if kind=='raw':khat,_=quantize(k[0],-RA,RA,16);vhat,_=quantize(v[0],-RB,RB,16)
    elif kind=='phase_aware':khat,_=quantize(k[0],L,U,16);vhat,_=quantize(v[0],-V,V,16)
    elif kind=='endpoint_only':khat,vhat=k[0],v[0]
    else:raise ValueError(kind)
    slope=(v[1]-vhat)/(k[1]-khat)
    intercept=vhat-slope*khat
    assert intercept==v[1]-slope*k[1]
    assert abs(slope)<RA and abs(intercept)<RB
    aq,ac=quantize(slope,-RA,RA,16);bq,bc=quantize(intercept,-RB,RB,16)
    errors=[abs(aq*q+bq-target) for q,target in zip(k,v)]
    return max(errors),dict(first_key=str(khat),first_value=str(vhat),
                           slope=str(slope),intercept=str(intercept),codes=[ac,bc],
                           query_errors=[str(e) for e in errors])

def m0(k,v):
    kq=[quantize(x,L,U,8)[0] for x in k]
    vq=[quantize(x,-V,V,8)[0] for x in v]
    errors=[]
    for j,q in enumerate(k):
        distances=[abs(x-q) for x in kq]
        assert distances[j]<distances[1-j]
        errors.append(abs(vq[j]-v[j]))
    assert max(errors)<=V/LEVELS8
    return max(errors)

def main():
    ek_raw=RA/LEVELS16;ev_raw=RB/LEVELS16
    ek_phase=(U-L)/(2*LEVELS16);ev_phase=V/LEVELS16
    bounds={'raw':bound('raw',ek_raw,ev_raw),'phase_aware':bound('phase_aware',ek_phase,ev_phase)}
    endpoint=U*RA/LEVELS16+RB/LEVELS16
    m0_key=(U-L)/(2*LEVELS8);m0_value=V/LEVELS8
    assert 2*m0_key<GAP
    assert Q(bounds['raw']['worst_query_and_Ecomp_bound']['fraction'])<Q(1,2)
    assert Q(bounds['phase_aware']['worst_query_and_Ecomp_bound']['fraction'])<Q(1,20)
    assert endpoint<Q(1,20) and m0_value<Q(1,20)
    # Fixed analytic boundary fixtures, not generated experimental population.
    key_pairs=[(L,L+GAP),(U-GAP,U),(L,U),(Q(1),Q(21,20))]
    records=[]
    maxima={name:Q(0) for name in ['raw','phase_aware','endpoint_only','M0']}
    for pair,reverse,vs in itertools.product(key_pairs,[False,True],itertools.product([-V,Q(0),V],repeat=2)):
        k=list(pair[::-1] if reverse else pair);v=list(vs)
        assert abs(k[1]-k[0])>=GAP
        row={'keys':list(map(str,k)),'values':list(map(str,v)),'controls':{}}
        for name in ['raw','phase_aware','endpoint_only']:
            error,detail=interpolation(k,v,name)
            cap=endpoint if name=='endpoint_only' else Q(bounds[name]['worst_query_and_Ecomp_bound']['fraction'])
            assert error<=cap
            maxima[name]=max(maxima[name],error)
            row['controls'][name]={'max_error':str(error),**detail}
        error=m0(k,v);maxima['M0']=max(maxima['M0'],error)
        row['controls']['M0']={'max_error':str(error)};records.append(row)
    # Coefficient writer is rank1/noninjective for a fixed second event.
    anchor=[Q(12,5),Q(9,5)]
    first_pairs=[(Q(2,5),Q(-11,5)),(Q(7,5),Q(-1,5))]
    images=[]
    for k,v in first_pairs:
        a=(anchor[1]-v)/(anchor[0]-k);b=v-a*k
        images.append((a,b));assert a==2 and b==Q(-3)
    assert images[0]==images[1]
    result=dict(payload_bits=32,controller_bits=2,registers={'interpolation':[16,16],'M0':[8,8,8,8]},
        quantizer='2^B inclusive-endpoint nearest levels; verification uses nearest-even ties',
        domain={'key_min':str(L),'key_max_closure':str(U),'min_gap':str(GAP),'abs_value_max_closure':str(V)},
        bounds=bounds,endpoint_only_error_bound=rational_record(endpoint),
        cache_key_error_bound=rational_record(m0_key),cache_value_error_bound=rational_record(m0_value),
        cache_identity_margin=rational_record(GAP-2*m0_key),
        fixed_analytic_fixtures=len(records),fixture_maxima={k:rational_record(v) for k,v in maxima.items()},
        fixtures=records,
        smooth_flow_noninjectivity_witness=dict(second=list(map(str,anchor)),first=[list(map(str,x)) for x in first_pairs],same_output=[str(x) for x in images[0]]),
        limitations=['No IID histories generated or parameter search.','Uniform quantizer and arithmetic precision are declared engineering assumptions.',
                    'Endpoint-only retains unquantized first record and is not a matched peak-payload budget.',
                    'Smooth-flow obstruction concerns exact coefficient overwrite, not arbitrary readouts or hybrid resets.'])
    (OUT/'THEORY_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='fixtures'},indent=2))

if __name__=='__main__':main()
