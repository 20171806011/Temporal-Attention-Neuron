"""Exact algebra and bounded diagnostic checks; no fitted/full experiment."""
from fractions import Fraction as Q
from pathlib import Path
import json, math, re, hashlib

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[1]/'Temporal-Attention-Neuron'
L,U,V,gap=Q(3,10),Q(5,2),Q(3),Q(1,20)
Amax=2*V/gap
Bmax=V*(2*U/gap-1)
center=(L+U)/2
Cmax=V*(2*(U-center)/gap-1)

def line(k1,v1,k2,v2):
    a=(v2-v1)/(k2-k1)
    b=v1-a*k1
    assert a*k1+b==v1 and a*k2+b==v2
    return a,b

cases=[]
for x,y in [(L,L+gap),(U-gap,U),(L,U)]:
    for v1 in [-V,V]:
        for v2 in [-V,V]:
            a,b=line(x,v1,y,v2)
            assert abs(a)<=Amax and abs(b)<=Bmax
            assert abs(b+center*a)<=Cmax
            cases.append(dict(keys=[str(x),str(y)],values=[str(v1),str(v2)],a=str(a),b=str(b),center_value=str(b+center*a)))
a,b=line(U-gap,V,U,-V)
assert abs(a)==Amax and abs(b)==Bmax
assert abs(b+center*a)==Cmax

# Exact saturation of the independent coefficient/query perturbation inequality.
q=U;ea=Q(1,10000);eb=Q(1,10000);eq=Q(1,10000)
# Choose positive slope and signs agreeing at the upper target.
a,b=line(U-gap,-V,U,V)
error=(a+ea)*(q+eq)+(b+eb)-(a*q+b)
bound=q*ea+eb+(abs(a)+ea)*eq
assert error==bound

# Uniform versus unequal coefficient attenuation identities.
la,lb=Q(99,100),Q(98,100)
value=a*q+b
assert la*a*q+lb*b-value == (la-lb)*a*q+(lb-1)*value
uniform_error=la*a*q+la*b-value
assert uniform_error==(la-1)*value

# Floating minimum-gap fixture: contract guard must be checked, not assumed.
naive_gap=2.5-2.45
upper=math.nextafter(2.5,-math.inf)
lower=math.nextafter(upper-.05,-math.inf)
assert .3<=lower<upper<2.5 and upper-lower>=.05

# Exact standard arithmetic envelope conditional on exact represented inputs,
# round-to-nearest normal arithmetic, and the primitive operation sequence.
rounding=[]
for name,p in [('binary32',24),('binary64',53)]:
    unit=Q(1,2**p)
    gamma3=3*unit/(1-3*unit)
    epsilon_a=Amax*gamma3
    epsilon_b=U*epsilon_a+unit*(V+U*Amax*(1+gamma3)*(2+unit))
    evaluation_error=unit*((2+unit)*U*Amax*(1+gamma3)+Bmax+epsilon_b)
    total=U*epsilon_a+epsilon_b+evaluation_error
    rounding.append(dict(format=name,unit_roundoff=float(unit),coefficient_a_error_bound=float(epsilon_a),
        coefficient_b_error_bound=float(epsilon_b),query_evaluation_roundoff_bound=float(evaluation_error),
        exact_input_exact_query_total_error_bound=float(total)))

patterns=[r'interpolat',r'lagrange',r'vandermonde',r'divided.difference',r'polyfit',r'polyval',
          r'v2\s*-\s*v1',r'values\[1\]\s*-\s*values\[0\]',r'插值']
rx=re.compile('|'.join(patterns),re.I)
searched=[];hits=[]
for path in sorted(REPO.rglob('*')):
    if not path.is_file() or path.suffix not in {'.py','.md','.tex','.txt'} or '.git' in path.parts:continue
    searched.append(str(path.relative_to(REPO)))
    for number,text in enumerate(path.read_text(errors='replace').splitlines(),1):
        if rx.search(text):hits.append(dict(path=str(path.relative_to(REPO)),line=number,text=text))

result=dict(domain=dict(key_lower=str(L),key_upper=str(U),value_abs_bound=str(V),minimum_spacing=str(gap)),
    tight_support_bounds=dict(abs_slope=str(Amax),abs_intercept=str(Bmax),center=str(center),abs_center_value=str(Cmax)),
    exact_corner_checks=cases,perturbation_bound_saturating_case=dict(actual_error=str(error),bound=str(bound)),
    quantized_normalized_states_16bits=dict(max_normalized_rounding_error=str(Q(1,2**16-1)),
        no_query_noise_error_bound=str((U*Amax+Bmax)/Q(2**16-1))),
    floating_spacing=dict(naive_gap=naive_gap,naive_pass=naive_gap>=.05,valid_keys=[lower,upper],valid_gap=upper-lower),
    arithmetic_bounds=rounding,
    novelty_search=dict(file_count=len(searched),searched_files=searched,patterns=patterns,hits=hits,
        reproduction_scope='Novelty scan requires the full frozen repository; a minimal source subset changes inventory/hits but not the mathematical checks.',
        conclusion='No specific two-point affine streaming memory found in inspected textual sources; matched plotting fits/interpolation and shell text. This is not a literature novelty claim.'),
    source_sha256={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in ['p1_experiments/task_generator.py','p1_experiments/models.py','p1_experiments/p1b/runner_p1b.py']})
(OUT/'theory_checks.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'bounds':result['tight_support_bounds'],'novelty_files':len(searched),'hits':len(hits),'float_spacing':result['floating_spacing'],'arithmetic_bounds':rounding},indent=2))
