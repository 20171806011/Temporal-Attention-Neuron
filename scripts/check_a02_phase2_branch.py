"""Independent verification of Issue A02: Phase 2 decision gate branch gap.

Verifies:
1. Evaluation of conditions 'stable', 'A', 'B', 'C', and 'D' using frozen CSV data.
2. Demonstrates that 'stable' is True, but A, B, C are False, falling into 'D'.
3. Explains the unclassified mixed case (UNCLASSIFIED_MIXED_CASE).
4. Distinguishes event log-trace difference (1.094463) vs diff-in-diff (1.354674).
"""
import os
import csv
import numpy as np

def run_a02(csv_path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    def rw(model, frame):
        for r in rows:
            if r['model'] == model and r['frame'] == frame:
                return {k: float(v) if v not in ('', 'nan', None) else np.nan for k, v in r.items() if k not in ('model', 'frame')}
        raise ValueError(f"Not found: {model}, {frame}")
        
    b4f = rw("B4", "zF")
    b3f = rw("B3", "zF")
    b4c = rw("B4", "zC")
    b1c = rw("B1", "zC")
    c0 = rw("C0", "zU")
    
def evaluate_branch_logic(b4f, b3f, b4c, b1c, c0, min_logt_diff=0.2, stable_expansion=0.5):
    stable = bool(np.isfinite(b4f["dD_delta"]) and
                  b4f["dD_delta"] >= stable_expansion and
                  b4f["dD_event_ci_lo"] > 0)
                  
    c_ok = lambda a, b: np.isfinite(a) and np.isfinite(b)
    
    cond_A1 = c_ok(b4f["dD_event_ci_lo"], b3f["dD_event_ci_hi"]) and b4f["dD_event_ci_lo"] > b3f["dD_event_ci_hi"]
    cond_A2 = c_ok(b4c["dDe"], b1c["dDe"]) and b4c["dDe"] > b1c["dDe"]
    cond_A3 = c_ok(b4c["dDe"], c0["dDe"]) and b4c["dDe"] > c0["dDe"]
    A = stable and cond_A1 and cond_A2 and cond_A3
    
    dlt = b4f["lt_delta"] - b3f["lt_delta"] if np.isfinite(b4f["lt_delta"]) and np.isfinite(b3f["lt_delta"]) else np.nan
    cond_B1 = (not A)
    cond_B2 = not (c_ok(b4f["dD_delta"], b3f["dD_delta"]) and b4f["dD_delta"] > b3f["dD_delta"])
    cond_B3 = c_ok(b4f["dDe"], b3f["dDe"]) and abs(b4f["dDe"] - b3f["dDe"]) < 0.5
    cond_B4 = c_ok(dlt, 0.0) and dlt >= min_logt_diff
    B = cond_B1 and cond_B2 and cond_B3 and cond_B4
    
    cond_C1 = (not A and not B)
    cond_C2 = c_ok(b4c["dDe"], b1c["dDe"]) and abs(b4c["dDe"] - b1c["dDe"]) < 0.5
    cond_C3 = c_ok(b4c["dDe"], c0["dDe"]) and abs(b4c["dDe"] - c0["dDe"]) < 0.5
    C = cond_C1 and cond_C2 and cond_C3
    
    outcome = "A" if A else ("B" if B else ("C" if C else "D"))
    event_logtrace_diff = b4f["ltE"] - b3f["ltE"]
    delta_logtrace_diff = b4f["lt_delta"] - b3f["lt_delta"]
    
    return {
        'b4f_dD_delta': b4f['dD_delta'],
        'b4f_dD_event_ci_lo': b4f['dD_event_ci_lo'],
        'stable': stable,
        'cond_A': {
            'value': A,
            'dD_ci_separation': cond_A1,
            'b4c_gt_b1c': cond_A2,
            'b4c_gt_c0': cond_A3
        },
        'cond_B': {
            'value': B,
            'not_b4f_gt_b3f': cond_B2
        },
        'cond_C': {
            'value': C,
            'b4c_b1c_diff_lt_0p5': cond_C2,
            'b4c_c0_diff_lt_0p5': cond_C3
        },
        'outcome': outcome,
        'event_logtrace_diff': event_logtrace_diff,
        'delta_logtrace_diff': delta_logtrace_diff
    }

def verify_a02_assertions(results):
    assert results['stable'] is True, f"Expected stable=True, got {results['stable']}"
    assert results['cond_A']['value'] is False, f"Expected A=False, got {results['cond_A']['value']}"
    assert results['cond_B']['value'] is False, f"Expected B=False, got {results['cond_B']['value']}"
    assert results['cond_C']['value'] is False, f"Expected C=False, got {results['cond_C']['value']}"
    assert results['outcome'] == 'D', f"Expected outcome=D, got {results['outcome']}"
    assert abs(results['event_logtrace_diff'] - 1.094463096) < 1e-6, f"Event logtrace diff mismatch: {results['event_logtrace_diff']}"
    assert abs(results['delta_logtrace_diff'] - 1.354674209) < 1e-6, f"Delta logtrace diff mismatch: {results['delta_logtrace_diff']}"

def run_a02(csv_path):
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    def rw(model, frame):
        for r in rows:
            if r['model'] == model and r['frame'] == frame:
                return {k: float(v) if v not in ('', 'nan', None) else np.nan for k, v in r.items() if k not in ('model', 'frame')}
        raise ValueError(f"Not found: {model}, {frame}")
        
    b4f = rw("B4", "zF")
    b3f = rw("B3", "zF")
    b4c = rw("B4", "zC")
    b1c = rw("B1", "zC")
    c0 = rw("C0", "zU")
    
    results = evaluate_branch_logic(b4f, b3f, b4c, b1c, c0, min_logt_diff=0.2, stable_expansion=0.5)
    
    print("A02 Numerical Check Results:")
    print(f"  stable: {results['stable']} (dD_delta={results['b4f_dD_delta']:.4f} >= 0.5, CI_lo={results['b4f_dD_event_ci_lo']:.4f} > 0)")
    print(f"  Condition A: {results['cond_A']['value']} (failed at b4c > b1c: {b4c['dDe']:.4f} vs {b1c['dDe']:.4f})")
    print(f"  Condition B: {results['cond_B']['value']} (failed at not(b4f > b3f): {b4f['dD_delta']:.4f} vs {b3f['dD_delta']:.4f})")
    print(f"  Condition C: {results['cond_C']['value']} (failed at |b4c - c0| < 0.5: {abs(b4c['dDe'] - c0['dDe']):.4f} > 0.5)")
    print(f"  Resulting Outcome: {results['outcome']}")
    print("  Branch Gap Diagnosis: stable=True but outcome=D (described as 'No stable event-locked expansion')")
    print(f"  Event log-trace difference (B4 - B3): {results['event_logtrace_diff']:.6f}")
    print(f"  Event-minus-quiet diff-in-diff:       {results['delta_logtrace_diff']:.6f}")
    
    verify_a02_assertions(results)
    print("ALL A02 ASSERTIONS PASSED!")
    return results

def test_a02_error_fixture(csv_path):
    """Negative testing: mutates inputs and verifies that the actual checker functions REJECT them."""
    print("Running A02 Error Fixtures (testing that real checker functions reject perturbed inputs)...")
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    def rw(model, frame):
        for r in rows:
            if r['model'] == model and r['frame'] == frame:
                return {k: float(v) if v not in ('', 'nan', None) else np.nan for k, v in r.items() if k not in ('model', 'frame')}
        raise ValueError(f"Not found: {model}, {frame}")
        
    b4f = rw("B4", "zF")
    b3f = rw("B3", "zF")
    b4c = rw("B4", "zC")
    b1c = rw("B1", "zC")
    c0 = rw("C0", "zU")
    
    # 1. Perturb dD_delta below threshold -> verify_a02_assertions MUST reject
    b4f_corrupt1 = dict(b4f)
    b4f_corrupt1['dD_delta'] = 0.40  # below 0.5
    res1 = evaluate_branch_logic(b4f_corrupt1, b3f, b4c, b1c, c0)
    assert res1['stable'] is False
    try:
        verify_a02_assertions(res1)
        raise RuntimeError("verify_a02_assertions failed to reject corrupted dD_delta!")
    except AssertionError:
        print("  [PASS] Corrupted dD_delta (<0.5) correctly rejected by verify_a02_assertions")
        
    # 2. Perturb CI_lo to negative -> verify_a02_assertions MUST reject
    b4f_corrupt2 = dict(b4f)
    b4f_corrupt2['dD_event_ci_lo'] = -0.05
    res2 = evaluate_branch_logic(b4f_corrupt2, b3f, b4c, b1c, c0)
    assert res2['stable'] is False
    try:
        verify_a02_assertions(res2)
        raise RuntimeError("verify_a02_assertions failed to reject corrupted CI_lo!")
    except AssertionError:
        print("  [PASS] Corrupted negative CI_lo correctly rejected by verify_a02_assertions")
        
    # 3. Perturb b4c to force Condition A = True -> outcome changes to A -> verify_a02_assertions MUST reject
    b4c_corrupt3 = dict(b4c)
    b4c_corrupt3['dDe'] = 2.50  # makes b4c > b1c and b4c > c0
    res3 = evaluate_branch_logic(b4f, b3f, b4c_corrupt3, b1c, c0)
    assert res3['outcome'] == 'A'
    try:
        verify_a02_assertions(res3)
        raise RuntimeError("verify_a02_assertions failed to reject forced outcome A!")
    except AssertionError:
        print("  [PASS] Forced Condition A correctly rejected by verify_a02_assertions")
        
    # 4. Perturb logtrace diff -> verify_a02_assertions MUST reject
    b4f_corrupt4 = dict(b4f)
    b4f_corrupt4['ltE'] = 999.0
    res4 = evaluate_branch_logic(b4f_corrupt4, b3f, b4c, b1c, c0)
    try:
        verify_a02_assertions(res4)
        raise RuntimeError("verify_a02_assertions failed to reject corrupted logtrace!")
    except AssertionError:
        print("  [PASS] Corrupted logtrace diff correctly rejected by verify_a02_assertions")
        
    print("ALL A02 ERROR FIXTURES PASSED (Real checker functions verified to reject corrupted data)!")

if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = os.path.join(base, 'snapshot', 'results', 'tables', 'effective_dimension_summary.csv')
    run_a02(p)
    test_a02_error_fixture(p)
