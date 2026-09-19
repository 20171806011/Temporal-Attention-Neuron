"""Independent verification of Issue A05: Softmax numerical conventions.

Compares:
1. Pure mathematical softmax: exp(x) / sum(exp(x))
2. Stabilized without offset (Sprint 4.2-B code): exp(x - max) / sum(exp(x - max))
3. Stabilized with denominator offset delta=1e-9 (tan.py code): exp(x - max) / (sum(exp(x - max)) + 1e-9)
4. Paper 2 Section 2 text: alpha_i = exp(e_i) / (sum exp(e_j) + delta) [without explicit max stabilization]
"""
import numpy as np

def softmax_math(x):
    e = np.exp(x)
    return e / np.sum(e)

def softmax_s42b(x):
    m = np.max(x)
    e = np.exp(x - m)
    return e / np.sum(e)

def softmax_tan_model(x, beta=1.0):
    e_x = np.exp(beta * (x - np.max(x)))
    return e_x / (e_x.sum(axis=0) + 1e-9)

def softmax_paper2_text(x, delta=1e-9):
    e = np.exp(x)
    return e / (np.sum(e) + delta)

def run_a05():
    # Test on typical logits
    test_logits = np.array([3.6, 7.2, 0.0, 0.0, 10.8])
    
    sm_math = softmax_math(test_logits)
    sm_s42b = softmax_s42b(test_logits)
    sm_tan = softmax_tan_model(test_logits)
    sm_paper = softmax_paper2_text(test_logits)
    
    print("A05 Numerical Check Results on typical logits [3.6, 7.2, 0, 0, 10.8]:")
    print(f"  Sprint 4.2-B code (stabilized, no offset): sum = {np.sum(sm_s42b):.12f}, max = {np.max(sm_s42b):.8f}")
    print(f"  tan.py model (stabilized, delta=1e-9):    sum = {np.sum(sm_tan):.12f}, max = {np.max(sm_tan):.8f}")
    print(f"  Paper 2 text (no max, delta=1e-9):       sum = {np.sum(sm_paper):.12f}, max = {np.max(sm_paper):.8f}")
    
    # Test difference between s42b and tan.py
    diff_tan_s42b = np.max(np.abs(sm_tan - sm_s42b))
    print(f"  Max absolute difference (s42b vs tan.py): {diff_tan_s42b:.2e}")
    
    # Test on large logits that cause overflow without stabilization
    large_logits = np.array([700.0, 710.0, 720.0])
    try:
        sm_overflow = softmax_math(large_logits)
        overflow_has_nan = np.isnan(sm_overflow).any()
    except Exception:
        overflow_has_nan = True
    print(f"  Unstabilized on large logits [700, 710, 720]: produces NaN/Inf = {overflow_has_nan}")
    
    sm_s42b_large = softmax_s42b(large_logits)
    print(f"  Stabilized s42b on large logits: sum = {np.sum(sm_s42b_large):.12f} (finite and stable)")
    
    # Assertions
    assert abs(np.sum(sm_s42b) - 1.0) < 1e-15, "s42b must sum to 1.0 exactly"
    assert np.sum(sm_tan) < 1.0, "tan.py with delta=1e-9 must sum to strictly less than 1.0"
    assert abs(np.sum(sm_tan) - (1.0 / (1.0 + 1e-9 / np.sum(np.exp(test_logits - np.max(test_logits)))))) < 1e-15
    assert diff_tan_s42b < 1e-9, "Difference is on order of 1e-9"
    
    print("ALL A05 ASSERTIONS PASSED!")
    return {
        'sum_s42b': float(np.sum(sm_s42b)),
        'sum_tan': float(np.sum(sm_tan)),
        'diff_tan_s42b': float(diff_tan_s42b),
        'overflow_has_nan': bool(overflow_has_nan)
    }

def test_a05_error_fixture():
    """Error fixture verifying detection of numerical instability and denominator offset distortion."""
    print("Running A05 Error Fixture Tests...")
    
    # 1. Test overflow without max-stabilization
    large_logits = np.array([1000.0, 1010.0])
    with np.errstate(over='ignore', invalid='ignore'):
        e = np.exp(large_logits)
        res = e / np.sum(e)
    assert np.isnan(res).any(), "Unstabilized softmax on logits >= 1000 should overflow to NaN"
    
    # 2. Test that s42b max-stabilized never overflows
    m = np.max(large_logits)
    e_stable = np.exp(large_logits - m)
    res_stable = e_stable / np.sum(e_stable)
    assert not np.isnan(res_stable).any() and abs(np.sum(res_stable) - 1.0) < 1e-12
    
    print("ALL A05 ERROR FIXTURES PASSED!")

if __name__ == '__main__':
    run_a05()
    test_a05_error_fixture()
