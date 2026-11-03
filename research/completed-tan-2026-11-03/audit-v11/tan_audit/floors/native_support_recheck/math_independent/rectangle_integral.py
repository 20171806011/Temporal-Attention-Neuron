#!/usr/bin/env python3
"""Second-method exact integral over nearest-alphabet Voronoi rectangles.

No CDF calculation and no dependency on earlier floor code or results.
Run with PYTHONDONTWRITEBYTECODE=1 python3 rectangle_integral.py.
"""
from fractions import Fraction as Q
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True


def half_lengths(symbols):
    levels=sorted(set(symbols))
    edges=[Q(-3)]+[(x+y)/2 for x,y in zip(levels[:-1],levels[1:])]+[Q(3)]
    lengths=[]
    for i,x in enumerate(levels):
        lengths.extend([x-edges[i],edges[i+1]-x])
    assert all(x>=0 for x in lengths)
    assert sum(lengths)==6
    return levels,lengths


def rectangle_max_integral(a,b):
    """Integral over [0,a] x [0,b] of max(x,y), in exact rationals.

    Set m=min(a,b), M=max(a,b). On a square of side m, the integral
    is 2*m**3/3. The remaining rectangle contributes
    m*(M**2-m**2)/2, so total m*M**2/2 + m**3/6.
    """
    m,M=min(a,b),max(a,b)
    return m*M*M/2 + m*m*m/6


def solve(symbols):
    levels,halves=half_lengths(symbols)
    contributions=[rectangle_max_integral(a,b) for a in halves for b in halves]
    total=sum(contributions,Q(0))/36 # target density 1/(6*6)
    return {'alphabet':[str(x) for x in levels],
            'half_lengths':[str(x) for x in halves],
            'rectangle_count':len(contributions),
            'unweighted_integral':str(sum(contributions,Q(0))),
            'population_floor':str(total),'population_floor_float':float(total)}


def main():
    alphabets={
      'count_native_superset':[Q(-2),Q(0)],
      'latency_native_superset':[Q(-57,20)+Q(3*j,10) for j in range(7,20)]+[Q(0)],
    }
    report={name:solve(symbols) for name,symbols in alphabets.items()}
    assert Q(report['count_native_superset']['population_floor'])==Q(13,9)
    assert Q(report['latency_native_superset']['population_floor'])==Q(1243,1600)
    # Explicitly quantify binary reconstruction constants as real numbers.
    native_float=[Q.from_float(-2.85+j*0.30) for j in range(7,20)]+[Q(0)]
    float_result=solve(native_float)
    float_floor=Q(float_result['population_floor'])
    float_delta=float_floor-Q(1243,1600)
    report['binary_float_latency_as_exact_reals']={
      'population_floor':str(float_floor),
      'population_floor_float':float(float_floor),
      'difference_from_ideal_rational':str(float_delta),
      'difference_float':float(float_delta)}
    out=Path(__file__).resolve().parent/'results.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
