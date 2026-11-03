#!/usr/bin/env python3
"""Exact rational certificates for the frozen original Probe1 swap study.

No simulator, fitted decoder, sampled data or floating transcendental evaluation
is used to establish an inequality. Decimal/float fields are display only.
"""
from fractions import Fraction as Q
from pathlib import Path
import hashlib
import json
import platform

HERE = Path(__file__).resolve().parent
PLAN = 'd8f0b97300f7fd1d3e3b0c07cd5ca17c3cfb4b40b20efa25792e6ab3dd78df23'
DEGREE = 40
DELTA = Q(1, 10**9)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fq(x):
    """Full exact rational, plus a noncertifying readable decimal."""
    return {'rational': str(x), 'decimal_display': float(x)}


def interval(lo, hi):
    assert lo <= hi
    return {'lower': fq(lo), 'upper': fq(hi)}


def exp_interval(x):
    """T40 <= exp(x) <= T40 + next_term/(1-x/42), exactly."""
    assert Q(0) <= x < DEGREE + 2
    term = Q(1)
    total = term
    for k in range(1, DEGREE + 1):
        term *= x / k
        total += term
    next_term = term * x / (DEGREE + 1)
    ratio_bound = x / (DEGREE + 2)
    tail = next_term / (1 - ratio_bound)
    return total, total + tail


def tanh_interval(x):
    """Use monotonicity of (exp(2x)-1)/(exp(2x)+1)."""
    low, high = exp_interval(2*x)
    return (low-1)/(low+1), (high-1)/(high+1)


def main():
    assert sha(HERE/'PREREGISTRATION.md') == PLAN
    lo = (Q(123,200), Q(127,200))
    hi = (Q(273,200), Q(277,200))
    threshold = Q(1,2)
    assert Q(55,100) < lo[0] < lo[1] < Q(95,100)
    assert Q(105,100) < hi[0] < hi[1] < Q(145,100)
    assert lo[1] < hi[0]
    width = lo[1]-lo[0]
    assert width == hi[1]-hi[0] == Q(1,50)

    # t8 S = (4d-a-b-c)/5; independent conservative bound is frozen.
    s8_exact = ((4*hi[0]-2*lo[1]-hi[1])/5,
                (4*hi[1]-2*lo[0]-hi[0])/5)
    s8 = (Q(557,1000), Q(593,1000))
    assert s8[0] <= s8_exact[0] <= s8_exact[1] <= s8[1]
    gamma = (2*s8[0], 2*s8[1])
    xpos = gamma[0]*hi[0]
    xneg = gamma[1]*lo[1]
    xp_lo, xp_hi = exp_interval(xpos)
    xn_lo, xn_hi = exp_interval(xneg)
    # The negative terms use the upper exponential bound, never the lower.
    context_margin_lower = Q(73,100)*xp_lo-Q(77,100)*xn_hi-1-DELTA
    assert context_margin_lower > 0
    # Q8 = 1+sum(active exponentials)+delta bounds its own denominator.
    _, max_high_exp = exp_interval(gamma[1]*hi[1])
    denominator_upper = 1+2*xn_hi+2*max_high_exp+DELTA
    context8_lower = 1+context_margin_lower/denominator_upper
    gate_exp_lo, gate_exp_hi = exp_interval(2*s8[0])
    assert gate_exp_lo > 3
    gate8_lower = (gate_exp_lo-1)/(gate_exp_lo+1)
    assert gate8_lower > threshold
    current8_lower = gate8_lower*context8_lower
    assert current8_lower > threshold

    # t9 S = (4q-a-b-c-d)/5. h8=0 is established above.
    s9_exact = ((4*hi[0]-2*lo[1]-2*hi[1])/5,
                (4*hi[1]-2*lo[0]-2*hi[0])/5)
    s9_upper = Q(8,25)
    assert 0 < s9_exact[0] <= s9_exact[1] < s9_upper
    query_u_upper = s9_upper*hi[1]
    assert query_u_upper == Q(277,625) < threshold

    # Original B2: positive weighted current implies reset regardless of h7.
    b2_j8_left_lower = (6*lo[0]+8*hi[0])/15
    b2_j8_right_lower = (7*lo[0]+7*hi[0])/15
    assert b2_j8_left_lower > threshold
    assert b2_j8_right_lower > threshold
    b2_gap = ((hi[0]-lo[1])/15, (hi[1]-lo[0])/15)
    assert b2_gap[0] > 0

    # Anchor: exact amplitudes; transcendental values receive rational enclosures.
    a, b = Q(5,8), Q(11,8)
    s8_anchor = b-(2*a+2*b)/5
    assert s8_anchor == Q(23,40)
    e_lo, e_hi = exp_interval(2*s8_anchor*a)
    f_lo, f_hi = exp_interval(2*s8_anchor*b)
    context_anchor = ((2*a*e_lo+2*b*f_lo)/(1+2*e_hi+2*f_hi+DELTA),
                      (2*a*e_hi+2*b*f_hi)/(1+2*e_lo+2*f_lo+DELTA))
    gate_anchor = tanh_interval(s8_anchor)
    current_anchor = (gate_anchor[0]*context_anchor[0],
                      gate_anchor[1]*context_anchor[1])
    assert current_anchor[0] > threshold
    s9_anchor = b-(2*a+3*b)/5
    assert s9_anchor == Q(3,10)
    ae_lo, ae_hi = exp_interval(2*s9_anchor*a)
    be_lo, be_hi = exp_interval(2*s9_anchor*b)
    context9_anchor = ((2*a*ae_lo+3*b*be_lo)/(2*ae_hi+3*be_hi+DELTA),
                       (2*a*ae_hi+3*b*be_hi)/(2*ae_lo+3*be_lo+DELTA))
    gate9_anchor = tanh_interval(s9_anchor)
    u9_anchor = (context9_anchor[0]*gate9_anchor[0],
                 context9_anchor[1]*gate9_anchor[1])
    assert 0 < u9_anchor[0] <= u9_anchor[1] < threshold
    anchor_b2_left = (a+2*b+3*a+4*b+5*b)/15
    anchor_b2_right = (b+2*a+3*a+4*b+5*b)/15
    assert anchor_b2_left == Q(47,40)
    assert anchor_b2_right == Q(9,8)
    assert anchor_b2_left-anchor_b2_right == Q(1,20)

    # Exact mixture density on either nonoverlap amplitude support is 1.
    mixture_density = Q(1,2)/Q(1,2)
    orientation_mass = (width*mixture_density)**5
    union_mass = 2*orientation_mass
    conditional_union_risk = Q(1,2)
    conditional_population_risk = union_mass*conditional_union_risk
    assert orientation_mass == Q(1,312500000)
    assert conditional_population_risk == orientation_mass

    # Symbolic full-window distinction and its flush, separate from response equality.
    left_windows = [['a','b','c','d','q'], ['b','c','d','q','0'], ['c','d','q','0','0']]
    right_windows = [['b','a','c','d','q'], ['a','c','d','q','0'], ['c','d','q','0','0']]
    assert sorted(left_windows[0]) == sorted(right_windows[0])
    assert left_windows[0] != right_windows[0] and left_windows[1] != right_windows[1]
    assert left_windows[2] == right_windows[2]

    result = {
        'status':'PASS', 'protocol_sha256':PLAN,
        'script_sha256':sha(__file__), 'python':platform.python_version(),
        'proof_arithmetic':'fractions.Fraction only; floating fields are display values',
        'exp_certificate':{
            'degree':DEGREE,
            'formula':'T40(x) <= exp(x) <= T40(x) + x^41/41! / (1-x/42)',
            'domain':'0 <= x < 42; consecutive tail ratios bounded by x/42',
            'positive_term_argument':fq(xpos), 'positive_term_exp':interval(xp_lo,xp_hi),
            'negative_term_argument':fq(xneg), 'negative_term_exp':interval(xn_lo,xn_hi),
            'gate_exp_argument':fq(2*s8[0]), 'gate_exp':interval(gate_exp_lo,gate_exp_hi)
        },
        'admissibility':{
            'low_range':interval(*lo), 'high_range':interval(*hi),
            'all_low_below_high_support':True, 'all_high_above_low_support':True,
            'width_each_coordinate':fq(width), 'anchor_low':fq(a), 'anchor_high':fq(b),
            'timeline':'Zero except t5..9; target low/high, three distractors, same query; native present-distractor support.'
        },
        'native_B4_t8':{
            'exact_surprise_range':interval(*s8_exact),
            'frozen_conservative_surprise_range':interval(*s8),
            'gamma_range':interval(*gamma),
            'context_minus_one_numerator_strict_lower':fq(context_margin_lower),
            'context_denominator_upper':fq(denominator_upper),
            'context_strict_lower':fq(context8_lower),
            'gate_exp_minus_three_strict_lower':fq(gate_exp_lo-3),
            'gate_strict_lower':fq(gate8_lower),
            'current_strict_lower':fq(current8_lower),
            'current_minus_threshold_strict_lower':fq(current8_lower-threshold),
            'reset_conclusion':'u8=lambda*h7+current8 > .5 for nonnegative h7; strict native threshold yields h8=0 on both histories.'
        },
        'native_B4_t9_and_future':{
            'exact_surprise_range':interval(*s9_exact),
            'frozen_surprise_upper':fq(s9_upper),
            'query_u_strict_upper':fq(query_u_upper),
            'threshold_minus_upper':fq(threshold-query_u_upper),
            'query_gate_strictly_positive':True,
            'context_identity':'Same multiset and q; mean, surprise, sum(exp), and sum(value*exp) equal in ideal arithmetic, including the same +1e-9 denominator.',
            'future_gate_identity':'After t9, x=0 and window inputs are nonnegative, so rectified surprise is exactly0.',
            'complete_response_formula':['u9','u9/2','u9/4','u9/8','u9/16',0,0,0,0,0],
            'ordered_left_windows_t9_to_t11':left_windows,
            'ordered_right_windows_t9_to_t11':right_windows,
            'full_state_distinct_at_query':True,
            'window_flush_identical_at_t11':True
        },
        'native_B2_positive_control':{
            'j8_left_lower':fq(b2_j8_left_lower),
            'j8_right_lower':fq(b2_j8_right_lower),
            'query_response_difference_formula':'(b-a)/15',
            'query_response_difference_range':interval(*b2_gap),
            'anchor_u9_left':fq(anchor_b2_left),
            'anchor_u9_right':fq(anchor_b2_right),
            'anchor_gap':fq(anchor_b2_left-anchor_b2_right)
        },
        'anchor':{
            'surprise8':fq(s8_anchor), 'context8':interval(*context_anchor),
            'gate8':interval(*gate_anchor), 'current8':interval(*current_anchor),
            'surprise9':fq(s9_anchor), 'context9':interval(*context9_anchor),
            'u9':interval(*u9_anchor)
        },
        'conditional_probability_and_risk':{
            'conditioning':'Present-distractor trial; original independent fair low/high amplitude mixtures.',
            'nonoverlap_mixture_density':fq(mixture_density),
            'orientation_mass':fq(orientation_mass), 'mirrored_union_mass':fq(union_mass),
            'equal_mixture_classification_error_lower':fq(conditional_union_risk),
            'present_distractor_population_error_lower':fq(conditional_population_risk),
            'argument':'Target/first-distractor swap is a measure-preserving involution between disjoint equal-density boxes; target labels reverse and ideal observable response does not. Each pair must incur at least one unit of total classification error, including randomized classifiers.',
            'limits':'Tiny analytic lower bound, not empirical prevalence, not balanced-three-class error, and not full finite-dataset risk. Does not explain the large reported average failure.'
        },
        'scope':[
            'Task-specific exact observation collision, not a full-state collision or a universal neuron theorem.',
            'Box validity preserves paired permutations; it is not robustness to arbitrary independent sensor/process noise.',
            'Native ideal equality is established algebraically; numerical execution may retain tiny floating residuals.',
            'No theorem is asserted for the reset-suppressed intervention or diagnostic B3 control.',
            'No simulator, estimator, fitting, parameter search or data-dependent choice was used.'
        ]
    }
    (HERE/'theory_certificates.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({
        'status':result['status'], 'protocol_sha256':PLAN,
        'context8_lower':float(context8_lower),
        'gate8_lower':float(gate8_lower),
        'current8_lower':float(current8_lower),
        'current8_threshold_margin_lower':float(current8_lower-threshold),
        'query_u_upper':float(query_u_upper),
        'anchor_current8_interval':[float(x) for x in current_anchor],
        'anchor_u9_interval':[float(x) for x in u9_anchor],
        'B2_gap_range':[float(x) for x in b2_gap],
        'orientation_mass':str(orientation_mass),
        'script_sha256':result['script_sha256']
    },indent=2))


if __name__=='__main__':
    main()
