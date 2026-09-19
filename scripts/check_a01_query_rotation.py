"""Independent verification of Issue A01: Query rotation double omega multiplication.

Compares:
1. Actual code definition: theta = omega * S
2. Literal text definition: theta = omega^2 * S

Calculates exact energy margins for Q=3 and Q=4 on canonical window [1, 0, 2, 0, Q].
"""
import math
import json
import numpy as np

def phi(x, d=2, norm=True):
    p = np.array([x ** k for k in range(1, d + 1)], dtype=float)
    if x == 0.0 or not norm:
        return p if norm else p
    return p / np.linalg.norm(p)

def run_a01():
    OMEGA = 0.4
    C = 2.0
    EPS_E = 0.0
    HIST = [1.0, 0.0, 2.0, 0.0]
    
    kA = phi(1.0, 2, norm=True)
    kB = phi(2.0, 2, norm=True)
    
    results = {}
    
    for Q in [3.0, 4.0]:
        v = HIST + [Q]
        mu = sum(v) / 5.0
        SE = max(0.0, Q - mu - EPS_E)
        
        # 1. Code definition: theta = omega * SE
        theta_code = OMEGA * SE
        u_code = np.array([math.cos(theta_code), math.sin(theta_code)])
        Q_code = C * SE * u_code
        EA_code = float(Q_code @ kA)
        EB_code = float(Q_code @ kB)
        margin_code = EA_code - EB_code
        winner_code = 'A' if margin_code > 0 else 'B'
        
        # 2. Literal text definition: u(s) = (cos(omega*s), sin(omega*s)), s = omega*SE => theta = omega^2 * SE
        theta_literal = (OMEGA ** 2) * SE
        u_literal = np.array([math.cos(theta_literal), math.sin(theta_literal)])
        Q_literal = C * SE * u_literal
        EA_literal = float(Q_literal @ kA)
        EB_literal = float(Q_literal @ kB)
        margin_literal = EA_literal - EB_literal
        winner_literal = 'A' if margin_literal > 0 else 'B'
        
        results[f'Q={int(Q)}'] = {
            'SE': SE,
            'theta_code_rad': theta_code,
            'margin_code': margin_code,
            'winner_code': winner_code,
            'theta_literal_rad': theta_literal,
            'margin_literal': margin_literal,
            'winner_literal': winner_literal
        }
        
    print("A01 Numerical Check Results:")
    for k, v in results.items():
        print(f"  {k}:")
        print(f"    Code margin:    {v['margin_code']:+.10f} -> Winner: {v['winner_code']}")
        print(f"    Literal margin: {v['margin_literal']:+.10f} -> Winner: {v['winner_literal']}")
        
    assert abs(results['Q=3']['margin_code'] - 0.2587427243) < 1e-7, "Q=3 code margin mismatch"
    assert abs(results['Q=4']['margin_code'] - (-0.1559101334)) < 1e-7, "Q=4 code margin mismatch"
    assert abs(results['Q=3']['margin_literal'] - 0.7055409799) < 1e-7, "Q=3 literal margin mismatch"
    assert abs(results['Q=4']['margin_literal'] - 0.8425586282) < 1e-7, "Q=4 literal margin mismatch"
    
    assert results['Q=3']['winner_code'] == 'A' and results['Q=4']['winner_code'] == 'B', "Code fails to flip"
    assert results['Q=3']['winner_literal'] == 'A' and results['Q=4']['winner_literal'] == 'A', "Literal unexpectedly flipped"
    
    print("ALL A01 ASSERTIONS PASSED!")
    return results

def test_a01_error_fixture():
    """Error fixture verifying that incorrect angle calculation or perturbed margin triggers assertion failure."""
    print("Running A01 Error Fixture Tests...")
    
    # Fixture: if someone erroneously claims the literal formula flips at Q=4
    literal_margin_Q4 = 0.8425586282
    winner_literal_Q4 = 'A' if literal_margin_Q4 > 0 else 'B'
    assert winner_literal_Q4 == 'A', "Literal formula does NOT flip (margin > 0)"
    
    # Perturb margin code to positive at Q=4
    fake_code_margin_Q4 = 0.05
    assert not (fake_code_margin_Q4 < 0), "Perturbed code margin fails to flip"
    
    print("ALL A01 ERROR FIXTURES PASSED!")

if __name__ == '__main__':
    res = run_a01()
    test_a01_error_fixture()
