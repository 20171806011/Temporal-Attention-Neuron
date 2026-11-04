#!/usr/bin/env python3
"""Independent stdlib-only fixed-alphabet validation; does not import repo code."""
import argparse
import bisect
from fractions import Fraction as Q
import json
import math
from pathlib import Path
import random

LOW, HIGH = Q(-3), Q(3)
ALPHABETS = {
    'count_actual': tuple(map(Q, [-2, -1, 0, 1, 2])),
    'latency_actual_with_zero': tuple(sorted({Q(-285,100)+k*Q(3,10) for k in range(20)} | {Q(0)})),
    'latency_without_zero_reference': tuple(Q(-285,100)+k*Q(3,10) for k in range(20)),
}

def arms(code):
    """Lengths of both arms of clipped Voronoi cells, independently derived."""
    edges = [LOW] + [(a+b)/2 for a,b in zip(code,code[1:])] + [HIGH]
    return [length for i,c in enumerate(code) for length in (c-edges[i],edges[i+1]-c)]

def exact(code):
    lengths = arms(code)
    assert all(x >= 0 for x in lengths)
    assert sum(lengths) == HIGH-LOW
    cuts = sorted({Q(0), *lengths})
    first = second = scalar = Q(0)
    segments = []
    for lo, hi in zip(cuts,cuts[1:]):
        mid=(lo+hi)/2
        slope=Q(sum(x>mid for x in lengths),6)
        intercept=sum((x for x in lengths if x<=mid),Q(0))/6
        # F(t)=intercept+slope*t. Integrate survival of max of two iid distances.
        poly=[1-intercept**2, -2*intercept*slope, -slope**2]
        first += sum((v*(hi**(i+1)-lo**(i+1))/(i+1) for i,v in enumerate(poly)),Q(0))
        second += sum((2*v*(hi**(i+2)-lo**(i+2))/(i+2) for i,v in enumerate(poly)),Q(0))
        scalar += (1-intercept)*(hi-lo)-slope*(hi**2-lo**2)/2
        segments.append({'lo':str(lo),'hi':str(hi),'F_intercept':str(intercept),'F_slope':str(slope)})
    return {'alphabet':[str(c) for c in code], 'cdf_segments':segments,
            'scalar_mae_exact':str(scalar),'scalar_mae':float(scalar),
            'joint_mean_exact':str(first),'joint_mean':float(first),
            'joint_second_moment_exact':str(second),
            'joint_variance':float(second-first**2),'maximum_loss':float(cuts[-1])}

def nearest(x, code):
    i=bisect.bisect_left(code,x)
    return min(abs(x-code[j]) for j in (i-1,i) if 0<=j<len(code))

def fixtures():
    # Exact rational arithmetic; exhaustive decoder pairs for representative target pairs.
    targets=[Q(-3),Q(-285,100),Q(-23,10),Q(-1,2),Q(-75,1000),Q(0),Q(75,1000),Q(1,2),Q(23,10),Q(285,100),Q(3)]
    checks=0
    for name,code in ALPHABETS.items():
        for a in targets:
            for b in targets:
                bound=max(nearest(a,code),nearest(b,code))
                achieved=[]
                for ca in code:
                    for cb in code:
                        ea,eb=ca-a,cb-b
                        metric=(abs(ea+eb)+abs(ea-eb))/2
                        assert metric==max(abs(ea),abs(eb))
                        assert metric>=bound
                        achieved.append(metric)
                        checks+=1
                assert min(achieved)==bound
    return {'exhaustive_exact_pair_checks':checks,'identity_pass':True,'pointwise_bound_pass':True,'oracle_attainment_pass':True}

def run(n,seed):
    result={'seed':seed,'independent_target_pairs':n,'method':'Python stdlib random.Random; paired common target draws across alphabets; no repo imports',
            'fixture_checks':fixtures(),'alphabets':{k:exact(v) for k,v in ALPHABETS.items()}}
    rng=random.Random(seed)
    totals={k:[0.,0.,0.] for k in ALPHABETS}
    codes={k:list(map(float,v)) for k,v in ALPHABETS.items()}
    for _ in range(n):
        a,b=rng.uniform(-3,3),rng.uniform(-3,3)
        for name,code in codes.items():
            da,db=nearest(a,code),nearest(b,code)
            loss=max(da,db)
            totals[name][0]+=loss
            totals[name][1]+=loss*loss
            totals[name][2]+=(da+db)/2
    for name,(s,ss,scalar) in totals.items():
        entry=result['alphabets'][name]
        mean=s/n
        variance=(ss-s*s/n)/(n-1)
        se=math.sqrt(variance/n)
        radius=entry['maximum_loss']*math.sqrt(math.log(2/.05)/(2*n))
        entry['monte_carlo']={'mean':mean,'scalar_mae':scalar/n,'sample_sd':math.sqrt(variance),'standard_error':se,
             'normal_approx_95_ci':[mean-1.959963984540054*se,mean+1.959963984540054*se],
             'hoeffding_95_interval':[mean-radius,mean+radius],
             'z_difference_from_exact':(mean-entry['joint_mean'])/se}
        # A finite sample at codeword targets has zero loss despite positive population floor.
        entry['finite_sample_counterexample']={'target_pairs':[['0','0']] if 'with_zero' in name or name=='count_actual' else [['3/20','3/20']], 'oracle_mean':0}
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--samples',type=int,default=200000)
    p.add_argument('--seed',type=int,default=20261102)
    p.add_argument('--output',type=Path,default=Path(__file__).with_name('results.json'))
    args=p.parse_args()
    if args.samples<2: p.error('--samples must be at least 2')
    result=run(args.samples,args.seed)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for name,entry in result['alphabets'].items():
        print(name, 'exact=',entry['joint_mean_exact'], 'MC=',entry['monte_carlo']['mean'],'SE=',entry['monte_carlo']['standard_error'])
    print('fixtures:',result['fixture_checks'])
