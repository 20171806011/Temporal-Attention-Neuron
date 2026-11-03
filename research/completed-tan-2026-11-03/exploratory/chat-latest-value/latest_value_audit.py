#!/usr/bin/env python3
"""Independent Chat-sandbox audit of a conditional latest-value associative memory.

NOT TAN/P1-B production code; NOT a reproduction of v11 or any ODE simulation.
The memory object receives only the observed feature vector and new value. Keys,
canonical feature arrays and labels held by the test harness are evaluator-only.

Run: python latest_value_audit.py --out results.json --seed 20261102
Dependencies: numpy, sympy. All random draws use numpy.random.RandomState.
"""
from __future__ import annotations
import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any
import numpy as np
import sympy as sp


def pinv64(C: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(C, rcond=1e-13, hermitian=True)


class SummaryMemory:
    """No explicit key table: state consists of C, b and an admission counter."""
    def __init__(self, d: int, threshold: float = 1e-9):
        self.C = np.zeros((d, d), dtype=np.float64)
        self.b = np.zeros(d, dtype=np.float64)
        self.n = 0
        self.threshold = threshold

    def state(self):
        K = pinv64(self.C)
        w = K @ self.b
        P = np.eye(self.C.shape[0]) - self.C @ K
        return K, w, P

    def write(self, f: np.ndarray, value: float) -> dict[str, Any]:
        f = np.asarray(f, dtype=np.float64)
        if f.shape != self.b.shape or not np.all(np.isfinite(f)):
            raise ValueError('Feature dimension mismatch or non-finite feature')
        if not np.isfinite(value):
            raise ValueError('Non-finite value')
        K, w, P = self.state()
        p = P @ f
        residual = float(value - f @ w)
        is_new = np.linalg.norm(p) > self.threshold
        if is_new:
            if self.n == self.C.shape[0]:
                raise RuntimeError('Numerical/contract violation: new direction at capacity')
            expected_w = w + residual * p / float(p @ p)
            expected_P = P - np.outer(p, p) / float(p @ p)
            self.C += np.outer(f, f)
            self.b += value * f
            self.n += 1
        else:
            expected_w = w + residual * (K @ f)
            expected_P = P
            self.b += residual * f
        _, actual_w, actual_P = self.state()
        return {
            'is_new': bool(is_new),
            'p_norm': float(np.linalg.norm(p)),
            'w_equivalence_error': float(np.linalg.norm(actual_w - expected_w, np.inf)),
            'P_equivalence_error': float(np.linalg.norm(actual_P - expected_P, 2)),
        }


def exact_rational_fixture() -> dict[str, Any]:
    fs = [sp.Matrix([1, 0, 0]), sp.Matrix([1, 1, 0]), sp.Matrix([0, 1, 1])]
    events = [(0, 1), (1, -2), (0, 3), (2, 4), (1, 5), (2, -1), (0, 0)]
    C, b = sp.zeros(3), sp.zeros(3, 1)
    seen: list[int] = []
    values: dict[int, sp.Rational] = {}
    rows = []
    for i, value in events:
        f, value = fs[i], sp.Rational(value)
        K = C.pinv()
        w, P = K*b, sp.eye(3)-C*K
        p, r = P*f, value-(f.T*w)[0]
        new = p != sp.zeros(3, 1)
        assert new == (i not in seen)
        if new:
            w_direct = w + r*p/(p.T*p)[0]
            P_direct = P - p*p.T/(p.T*p)[0]
            C, b = C+f*f.T, b+value*f
            seen.append(i)
        else:
            w_direct = w+r*K*f
            P_direct = P
            b = b+r*f
        values[i] = value
        Kn = C.pinv()
        wn, Pn = Kn*b, sp.eye(3)-C*Kn
        F = sp.Matrix.hstack(*[fs[j] for j in seen])
        t = sp.Matrix([values[j] for j in seen])
        assert wn == w_direct
        assert Pn == P_direct
        assert F.T*wn == t
        assert b == F*t
        assert F.T*Kn*F == sp.eye(len(seen))
        rows.append({'key_index_evaluator_only': i, 'new': new,
                     'predictions_exact': [str(x) for x in F.T*wn]})
    assert Pn == sp.zeros(3)
    return {'events': len(events), 'all_exact_assertions_pass': True, 'rows': rows}


def random_exact_contract(seed: int) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    metrics = dict(prediction=0.0, w_equivalence=0.0, projector_equivalence=0.0,
                   b_invariant=0.0)
    events_count = 0
    conds = []
    for _ in range(24):
        while True:
            F = rng.normal(size=(8, 6))
            F /= np.linalg.norm(F, axis=0)
            cond = float(np.linalg.cond(F))
            if cond <= 10.0:
                break
        conds.append(cond)
        mem = SummaryMemory(8)
        latest = np.zeros(6)
        n_seen = 0
        schedule = []
        for i in range(6):
            schedule.append(i)
            schedule.extend(rng.randint(0, i+1, size=2).tolist())
        schedule.extend(rng.randint(0, 6, size=32).tolist())
        for i in schedule:
            is_new = (i == n_seen)
            value = float(rng.uniform(-1, 1))
            info = mem.write(F[:, i], value)
            assert info['is_new'] == is_new
            if is_new:
                n_seen += 1
            latest[i] = value
            _, w, _ = mem.state()
            metrics['prediction'] = max(metrics['prediction'],
                float(np.max(np.abs(F[:, :n_seen].T @ w - latest[:n_seen]))))
            metrics['b_invariant'] = max(metrics['b_invariant'],
                float(np.max(np.abs(mem.b - F[:, :n_seen] @ latest[:n_seen]))))
            metrics['w_equivalence'] = max(metrics['w_equivalence'], info['w_equivalence_error'])
            metrics['projector_equivalence'] = max(metrics['projector_equivalence'], info['P_equivalence_error'])
            events_count += 1
    assert max(metrics.values()) < 1e-9
    return {'streams': 24, 'd': 8, 'max_distinct_keys': 6,
            'events': events_count, 'feature_condition_cap': 10.0,
            'observed_condition_range': [min(conds), max(conds)],
            'max_absolute_errors': metrics, 'all_assertions_pass': True}


def classification_margin(seed: int) -> dict[str, Any]:
    rng = np.random.RandomState(seed + 1)
    d, R, zeta, deltaP, gamma = 5, 2.0, 1e-3, 2e-3, 4e-2
    S = np.eye(d)[:, :2]
    P = np.eye(d) - S @ S.T
    q1 = np.array([np.sqrt(1-deltaP**2), 0, deltaP, 0, 0])
    Qh = np.column_stack([q1, np.eye(d)[:, 1]])
    Ph = np.eye(d) - Qh @ Qh.T
    actual_deltaP = float(np.linalg.norm(Ph-P, 2))
    E = zeta + deltaP*(R+zeta)
    tau = gamma/2
    assert E <= tau < gamma-E
    old_scores, new_scores = [], []
    for _ in range(2000):
        e = rng.normal(size=d); e *= zeta/np.linalg.norm(e)
        f = S[:, rng.randint(2)]
        old_scores.append(float(np.linalg.norm(Ph @ (f+e))))
        s = rng.normal(size=2); s *= 0.6/np.linalg.norm(s)
        n = rng.normal(size=3); n *= gamma/np.linalg.norm(n)
        fnew = np.r_[s, n]
        e = rng.normal(size=d); e *= zeta/np.linalg.norm(e)
        new_scores.append(float(np.linalg.norm(Ph @ (fnew+e))))
    assert max(old_scores) <= E + 1e-14
    assert min(new_scores) >= gamma-E - 1e-14
    assert max(old_scores) <= tau < min(new_scores)
    # A deliberately violated margin with an identical observation for two classes.
    bad_gamma = 1.5e-3
    old = np.array([1., 0.]); new = np.array([1., bad_gamma])
    common = 0.5*(old+new)
    assert np.linalg.norm(common-old) <= zeta
    assert np.linalg.norm(common-new) <= zeta
    return {'R': R, 'zeta': zeta, 'deltaP_requested': deltaP,
            'deltaP_actual': actual_deltaP, 'gamma': gamma, 'E': E, 'tau': tau,
            'cases': 4000, 'classification_errors': 0,
            'old_score_max': max(old_scores), 'new_score_min': min(new_scores),
            'failed_margin_identical_observation': {
                'gamma': bad_gamma, 'observation': common.tolist(),
                'distance_to_either_true_feature': float(np.linalg.norm(common-old))}}


def noisy_dual_bound(seed: int) -> dict[str, Any]:
    rng = np.random.RandomState(seed + 2)
    maximum_ratio = 0.
    max_formula_error = 0.
    for _ in range(1000):
        F = rng.normal(size=(6, 4)); F /= np.linalg.norm(F, axis=0)
        C = F @ F.T; K = pinv64(C)
        t = rng.uniform(-1, 1, size=4)
        w = K @ (F @ t)
        i = rng.randint(4); value = float(rng.uniform(-1, 1))
        zeta = 10**rng.uniform(-6, -3)
        e = rng.normal(size=6); e *= zeta/np.linalg.norm(e)
        x = F[:, i] + e
        r0 = value-t[i]; r = value-x@w
        wp = w+r*(K@x)
        target = t.copy(); target[i] = value
        actual = F.T@wp-target
        a = np.linalg.lstsq(F, e, rcond=None)[0]
        predicted = r*a
        predicted[i] -= e@w
        alpha = float(np.linalg.svd(F, compute_uv=False)[-1]**2)
        bound = zeta*np.linalg.norm(w)+(abs(r0)+zeta*np.linalg.norm(w))*zeta/np.sqrt(alpha)
        max_formula_error = max(max_formula_error, float(np.max(np.abs(actual-predicted))))
        assert np.linalg.norm(actual) <= bound + 1e-10
        maximum_ratio = max(maximum_ratio, float(np.linalg.norm(actual)/bound))
    return {'cases': 1000, 'max_error_to_bound_ratio': maximum_ratio,
            'max_dual_identity_residual': max_formula_error,
            'all_bound_checks_pass': True}


def nullspace_contamination() -> dict[str, Any]:
    zeta = 1e-3
    C = np.diag([1., 0.]); K = pinv64(C); Q = C@K
    b = np.zeros(2); f_observed = np.array([1., zeta])
    r = 1.0
    b_naive = b + r*f_observed
    b_projected = b + r*(Q@f_observed)
    w_naive = K@b_naive; w_projected = K@b_projected
    f_new = np.array([0., 1.]); value_new = 0.
    C_new = C+np.outer(f_new, f_new)
    after_naive = np.linalg.solve(C_new, b_naive+value_new*f_new)
    after_projected = np.linalg.solve(C_new, b_projected+value_new*f_new)
    assert np.array_equal(w_naive, w_projected)
    assert after_naive[1] == zeta
    assert after_projected[1] == 0.0
    return {'zeta': zeta, 'before_new_key_readout_naive': w_naive.tolist(),
            'before_new_key_readout_projected': w_projected.tolist(),
            'after_new_key_outputs_naive': after_naive.tolist(),
            'after_new_key_outputs_projected': after_projected.tolist(),
            'all_assertions_pass': True}


def collinearity_and_precision() -> dict[str, Any]:
    rows = []
    zeta = 1e-6
    for dtype in [np.float64, np.float32]:
        cutoff = float(100*np.finfo(dtype).eps)
        for gap in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8]:
            F = np.array([[1, 1], [0, gap]], dtype=dtype)
            C = F@F.T
            x = np.array([1, zeta], dtype=dtype)
            K = np.linalg.pinv(C, rcond=cutoff, hermitian=True)
            s = np.linalg.svd(C, compute_uv=False)
            rank = int(np.sum(s > cutoff*s[0]))
            y = F.T@(K@x)
            rows.append({'dtype': np.dtype(dtype).name, 'gap': gap,
                'zeta': zeta, 'rcond': cutoff, 'numerical_rank': rank,
                'predictions': [float(z) for z in y],
                'exact_arithmetic_predictions': [1-zeta/gap, zeta/gap],
                'projector_residual_on_observed_key': float(np.linalg.norm((np.eye(2)-C@K)@x))})
    # Repeated update instability when noisy leverage exceeds 2.
    gap = 1e-3
    F = np.array([[1., 1.], [0., gap]])
    C = F@F.T; K = np.linalg.inv(C)
    x = np.array([1., 2*gap]); w = np.zeros(2)
    leverage = float(x@K@x)
    residuals = []
    for _ in range(6):
        r = float(1-x@w)
        residuals.append(r)
        w += r*(K@x)
    assert np.allclose(residuals, [(-4.)**k for k in range(6)], atol=1e-8, rtol=1e-10)
    return {'rows': rows, 'repeated_same_noisy_write': {
        'gap': gap, 'zeta': 2*gap, 'leverage': leverage,
        'residual_sequence': residuals, 'gain_factor_theory': -4.0}}


def guarded_old_update_fixture() -> dict[str, Any]:
    """Projection prevents nullspace pollution; normalization controls one fixed-input gain.
    Neither operation identifies a noisy true key or removes cross-talk.
    """
    gap = 1e-3
    F = np.array([[1., 1.], [0., gap]])
    C = F @ F.T
    K = np.linalg.inv(C)
    Q = C @ K
    x = np.array([1., 2*gap])
    b = np.zeros(2)
    ell = float(x @ K @ x)
    residuals = []
    for _ in range(6):
        w = K @ b
        residual = float(1-x @ w)
        residuals.append(residual)
        b = Q @ b + (residual/ell) * (Q @ x)
    w = K @ b
    assert np.max(np.abs(residuals[1:])) < 1e-10
    assert abs(x @ w-1) < 1e-10
    assert np.allclose(F.T @ w, [-0.2, 0.4], atol=1e-10)
    return {'gap': gap, 'zeta': 2*gap, 'leverage': ell,
            'residual_sequence': residuals,
            'predictions_at_true_keys': (F.T @ w).tolist(),
            'target_predictions': [1.0, 0.0],
            'observed_noisy_feature_prediction': float(x @ w),
            'claim': 'Stable for this repeated noisy observation, not correct true-key overwrite'}


def regularization() -> dict[str, Any]:
    gap = 1e-3
    F = np.array([[1., 1.], [0., gap]])
    C = F@F.T; G = F.T@F
    alpha = float(np.linalg.eigvalsh(G)[0])
    rows = []
    for lam in [0., 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
        w = np.linalg.solve(C+lam*np.eye(2), F[:, 0])
        y = F.T@w
        denom = gap**2*(1+lam)+2*lam+lam**2
        exact = np.array([(gap**2+lam)/denom, lam/denom])
        assert np.allclose(y, exact, rtol=1e-9, atol=1e-10)
        bound = lam/(alpha+lam) if lam else 0.
        assert np.linalg.norm(y-np.array([1., 0.])) <= bound+1e-9
        rows.append({'lambda': lam, 'predictions': y.tolist(),
                     'closed_form_predictions': exact.tolist(),
                     'error_norm': float(np.linalg.norm(y-np.array([1., 0.]))),
                     'spectral_bound': bound})
    return {'gap': gap, 'alpha': alpha, 'rows': rows}


def comparisons() -> dict[str, Any]:
    F = np.array([[1., 1.], [0., 1.]])
    C = F@F.T; f = F[:, 0]
    # Both keys previously have value 0. Now overwrite first key with 1.
    dual = F.T@np.linalg.solve(C, f)
    averaged = F.T@np.linalg.solve(C+np.outer(f, f), f)
    delta = F.T@(f/(f@f))
    assert np.allclose(dual, [1, 0])
    assert np.allclose(averaged, [.5, 0])
    assert np.allclose(delta, [1, 1])
    # Same (w,P), but two different metrics need different protected updates.
    d1 = np.array([1., 0.])
    d2 = np.linalg.solve(C, f)
    # Non-membership cannot be inferred from C: two rotated orthonormal dictionaries.
    Q = np.array([[1., -1.], [1., 1.]])/np.sqrt(2.)
    assert np.allclose(Q@Q.T, np.eye(2))
    return {'features_columns': F.tolist(), 'new_target': [1., 0.],
        'dual_overwrite': dual.tolist(), 'additive_average': averaged.tolist(),
        'normalized_plain_delta': delta.tolist(),
        'same_w_P_different_required_duals': [d1.tolist(), d2.tolist()],
        'membership_ambiguity': {'dictionary_1': np.eye(2).tolist(),
            'dictionary_2': Q.tolist(), 'both_C_identity_b_zero': True,
            'incoming_feature': [1., 0.]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--out', type=Path, default=Path('results.json'))
    args = parser.parse_args()
    results = {
        'scope': 'Independent conceptual Chat sandbox; not TAN production or v11 reproduction',
        'environment': {'python': sys.version.split()[0], 'numpy': np.__version__,
                        'sympy': sp.__version__, 'platform': platform.platform()},
        'seed': args.seed, 'rng': 'numpy.random.RandomState',
        'exact_rational': exact_rational_fixture(),
        'random_exact_contract': random_exact_contract(args.seed),
        'classification_margin': classification_margin(args.seed),
        'noisy_dual_bound': noisy_dual_bound(args.seed),
        'nullspace_contamination': nullspace_contamination(),
        'collinearity_and_precision': collinearity_and_precision(),
        'regularization': regularization(),
        'guarded_old_update': guarded_old_update_fixture(),
        'comparisons': comparisons(),
        'execution_completed': True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    brief = {
        'scope': results['scope'], 'environment': results['environment'], 'seed': args.seed,
        'exact_rational_events': results['exact_rational']['events'],
        'random_contract': results['random_exact_contract'],
        'margin': results['classification_margin'],
        'noise_bound': results['noisy_dual_bound'],
        'nullspace_counterexample': results['nullspace_contamination'],
        'collinearity_float64': [r for r in results['collinearity_and_precision']['rows'] if r['dtype']=='float64'],
        'regularization': results['regularization'],
        'comparison': results['comparisons'],
        'guarded_old_update': results['guarded_old_update'],
    }
    print(json.dumps(brief, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
