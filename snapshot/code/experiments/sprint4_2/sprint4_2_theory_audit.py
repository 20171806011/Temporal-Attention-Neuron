#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sprint4_2_theory_audit.py
=========================
TAN-II Sprint 4.2-A -- automated audit of theorem-level expectations
(pure geometry; no Monte Carlo, no RNG).

This program verifies the closed-form statements of
docs/TANII_SPRINT4_2_THEORY.md by INDEPENDENT exact computations:

  Th1/Th2  d=1 sign structure (positive vs signed scalar)         [exact]
  Th3      P[w=A] = 1/2 for d=2, root-based arc measure, for
           equal-norm / unequal-norm / near-collinear pairs       [exact]
  Th4      F_d = 2*mu*(1-mu) = 1/2 for d=2 (from exact mu)        [exact]
  Th5      FlipRate_CF(theta) = theta/pi for d=2, computed by
           exact root decomposition of {cos(phi)*cos(phi-theta)<0}[exact]
  softmax  alpha_A = sigma(Q^T dK) identity                       [exact]
  JSD      closed-form Bernoulli JSD for the three pre-registered
           fixed-history cases (trap / flip / no-flip)            [exact]
  E6       ordering capacity: exact d=2 hyperplane-arrangement
           enumeration {2,6,12}; LP feasibility sets (d=1 pos/sgn,
           d=2 M=3/M=4, d=3/8 M=3/M=4 full-rank)                  [exact]

Any FAIL -> exit code 1 and "STOP AND DEBUG" (pre-registered stop rule).
Usage: python sprint4_2_theory_audit.py
"""

import math
import sys

import numpy as np
from scipy.optimize import linprog

EPS = 1e-6          # LP strict-margin
EPS_DOM = 1e-3      # positive-domain lower bound for d=1 LP
TOL = 1e-12
TOL_ANGLE = 1e-9

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def softmax2(a, b):
    s = math.exp(a) + math.exp(b)
    return math.exp(a) / s


def bernoulli_jsd(p, q):
    p = min(max(p, 1e-300), 1.0 - 1e-300)
    q = min(max(q, 1e-300), 1.0 - 1e-300)
    m = 0.5 * (p + q)

    def kl(x, y):
        return x * math.log(x / y) + (1.0 - x) * math.log((1.0 - x) / (1.0 - y))
    return 0.5 * (kl(p, m) + kl(q, m))


def exact_flip_measure(theta):
    """Exact measure on [0,2pi) of {phi: cos(phi)*cos(phi-theta) < 0},
    divided by 2*pi, via root decomposition of the two factors."""
    roots = []
    for r in (math.pi / 2.0, 3.0 * math.pi / 2.0):
        roots.append(r % (2.0 * math.pi))
    for r in (theta + math.pi / 2.0, theta + 3.0 * math.pi / 2.0):
        roots.append(r % (2.0 * math.pi))
    pts = sorted(set(round(r, 12) for r in roots))
    total = 0.0
    n = len(pts)
    for k in range(n):
        a = pts[k]
        b = pts[(k + 1) % n] if k + 1 < n else pts[0] + 2.0 * math.pi
        mid = 0.5 * (a + b)
        if math.cos(mid) * math.cos(mid - theta) < 0:
            total += (b - a)
    return total / (2.0 * math.pi)


def exact_halfplane_measure(phi_v):
    """Exact measure/(2pi) of {phi: cos(phi - phi_v) > 0} on [0,2pi)."""
    roots = sorted(set(round(r % (2.0 * math.pi), 12)
                       for r in (phi_v + math.pi / 2.0,
                                 phi_v + 3.0 * math.pi / 2.0)))
    total = 0.0
    for k in range(2):
        a = roots[k]
        b = roots[(k + 1) % 2] if k + 1 < 2 else roots[0] + 2.0 * math.pi
        mid = 0.5 * (a + b)
        if math.cos(mid - phi_v) > 0:
            total += (b - a)
    return total / (2.0 * math.pi)


def arrangement_2d(K):
    """Exact central arrangement of the lines {Q: Q^T(Ki-Kj)=0} on S^1.
    Returns (boundary_points, region_winners, region_measures)."""
    M = len(K)
    lines = []
    for i in range(M):
        for j in range(i + 1, M):
            v = np.asarray(K[i], float) - np.asarray(K[j], float)
            if np.linalg.norm(v) < 1e-12:
                continue
            phi_v = math.atan2(float(v[1]), float(v[0]))
            lines.append((phi_v + math.pi / 2.0) % math.pi)
    lines.sort()
    ded = []
    for a in lines:
        if not ded or abs(a - ded[-1]) > TOL_ANGLE:
            ded.append(a)
        elif a - ded[-1] < -math.pi + TOL_ANGLE:  # wrap-around duplicate
            ded.append(a)
    if not ded:
        return [0.0, 2.0 * math.pi], [tuple(range(M))], [1.0]
    pts = sorted(set(round(a, 12) for a in ded) |
                 set(round((a + math.pi) % (2.0 * math.pi), 12) for a in ded))
    n = len(pts)
    winners, measures = [], []
    for k in range(n):
        a0 = pts[k]
        a1 = pts[k + 1] if k + 1 < n else pts[0] + 2.0 * math.pi
        mid = 0.5 * (a0 + a1)
        Q = np.array([math.cos(mid), math.sin(mid)])
        scores = [float(Q @ np.asarray(Ki, float)) for Ki in K]
        winners.append(tuple(sorted(range(M), key=lambda i: -scores[i])))
        measures.append((a1 - a0) / (2.0 * math.pi))
    return pts, winners, measures


def lp_feasible(d, K, perm, domain=None):
    """LP existence of Q with Q.K_perm[k] - Q.K_perm[k+1] >= EPS for all k,
    |Q_l| <= 1 (d=1 'positive': Q in [EPS_DOM, 1]; 'signed': Q in [-1,1])."""
    M = len(K)
    rows, b = [], []
    for k in range(M - 1):
        v = np.atleast_1d(np.asarray(K[perm[k]], float)
                          - np.asarray(K[perm[k + 1]], float))
        rows.append(-v)
        b.append(-EPS)
    if d == 1:
        bounds = [(EPS_DOM, 1.0)] if domain == "positive" else [(-1.0, 1.0)]
    else:
        bounds = [(-1.0, 1.0)] * d
    A = np.array(rows) if rows else None
    bu = np.array(b) if b else None
    res = linprog(c=np.zeros(d), A_ub=A, b_ub=bu, bounds=bounds, method="highs")
    return res.status == 0


def main():
    import itertools

    # ---------------- Th1 / Th2: d=1 sign structure ----------------
    for q in (0.5, 3.0):
        wA = (q * 1.0 > q * 2.0)   # K_A=1, K_B=2
        check(f"Th1: Q={q}>0 -> winner B (not A)", wA is False)
    for q in (-0.5, -3.0):
        wA = (q * 1.0 > q * 2.0)
        check(f"Th1: Q={q}<0 -> winner A", wA is True)
    check("Th2: Q=+1 -> argmax key (B)", (1.0 * 1.0) < (1.0 * 2.0))
    check("Th2: Q=-1 -> argmin key (A)", (-1.0 * 1.0) > (-1.0 * 2.0))

    # ---------------- Th3: exact halfplane measure, d=2 ----------------
    for pname, (Ka, Kb) in [("P1", (np.array([1.0, 0.0]),
                                    np.array([0.0, 1.0]))),
                            ("P2", (np.array([1.0, 0.0]),
                                    np.array([0.0, 3.0]))),
                            ("P3", (np.array([1.0, 0.0]),
                                    np.array([1.0, 0.05])))]:
        v = Ka - Kb
        phi_v = math.atan2(float(v[1]), float(v[0]))
        mu = exact_halfplane_measure(phi_v)
        check(f"Th3: P[w=A]=1/2 exact ({pname})", abs(mu - 0.5) < TOL,
              f"mu={mu:.15f}")

    # ---------------- Th4: F_d = 2*mu*(1-mu), d=2 ----------------
    v = np.array([1.0, -1.0])
    phi_v = math.atan2(float(v[1]), float(v[0]))
    mu = exact_halfplane_measure(phi_v)
    F = 2.0 * mu * (1.0 - mu)
    check("Th4: F_d = 1/2 exact (d=2)", abs(F - 0.5) < TOL, f"F={F:.15f}")

    # ---------------- Th5: FlipRate_CF(theta) = theta/pi, d=2 ----------------
    thetas = [0.0, math.pi / 12.0, math.pi / 6.0, math.pi / 4.0, math.pi / 3.0,
              math.pi / 2.0, 2.0 * math.pi / 3.0, 3.0 * math.pi / 4.0,
              5.0 * math.pi / 6.0, 11.0 * math.pi / 12.0, math.pi]
    for th in thetas:
        m = exact_flip_measure(th)
        an = th / math.pi
        check(f"Th5: FlipRate(theta={th/ math.pi:.4f}pi) = theta/pi exact",
              abs(m - an) < TOL, f"measured={m:.15f} analytic={an:.15f}")

    # ---------------- softmax == sigmoid identity ----------------
    ok_id = True
    for x in (-4.0, -1.0, -0.5, 0.0, 0.8660254, 1.0, 4.0):
        if abs(softmax2(x, 0.0) - sigmoid(x)) > 1e-15:
            ok_id = False
    check("identity: softmax([x,0])_0 == sigma(x)", ok_id)

    # ---------------- JSD closed forms (three fixed histories) ----------------
    # (i) positive scalar trap: K_A=1.0,K_B=2.0 -> alpha_A(gamma)=sigma(-gamma)
    p1, q1 = sigmoid(-1.0), sigmoid(-4.0)
    jsd1 = bernoulli_jsd(p1, q1)
    flip1 = (1.0 * (1.0 - 2.0) > 0) != (4.0 * (1.0 - 2.0) > 0)
    check("E4(i): positive scalar JSD(gamma=1,4) > 0.05", jsd1 > 0.05,
          f"JSD={jsd1:.6f}")
    check("E4(i): positive scalar flip == 0", flip1 is False)

    # (ii) d=2 flip case: dK=e1, Q_A=e1, Q_B at 120 deg
    p2, q2 = sigmoid(1.0), sigmoid(math.cos(2.0 * math.pi / 3.0))
    jsd2 = bernoulli_jsd(p2, q2)
    flip2 = (1.0 > 0) != (math.cos(2.0 * math.pi / 3.0) > 0)
    check("E4(ii): d=2 theta=2pi/3 JSD > 0.05", jsd2 > 0.05, f"JSD={jsd2:.6f}")
    check("E4(ii): d=2 theta=2pi/3 flip == 1", flip2 is True)

    # (iii) d=2 no-flip case: Q_B at 30 deg
    p3, q3 = sigmoid(1.0), sigmoid(math.cos(math.pi / 6.0))
    jsd3 = bernoulli_jsd(p3, q3)
    flip3 = (1.0 > 0) != (math.cos(math.pi / 6.0) > 0)
    check("E4(iii): d=2 theta=pi/6 JSD > 5e-4", jsd3 > 5e-4, f"JSD={jsd3:.6f}")
    check("E4(iii): d=2 theta=pi/6 flip == 0", flip3 is False)

    # ---------------- E6: ordering capacity ----------------
    K2 = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    K3 = [np.array([1.0, 0.0]), np.array([0.0, 1.0]),
          np.array([1.0, 1.0]) / math.sqrt(2.0)]
    K4p = [np.array([1.0, 0.0]), np.array([0.0, 1.0]),
           np.array([1.0, 1.0]) / math.sqrt(2.0),
           np.array([2.0, 1.0]) / math.sqrt(5.0)]

    for K, M, exp in ((K2, 2, 2), (K3, 3, 6), (K4p, 4, 12)):
        _, winners, measures = arrangement_2d(K)
        distinct = len(set(winners))
        check(f"E6: d=2 M={M} exact distinct orderings == {exp}",
              distinct == exp, f"count={distinct}")
        check(f"E6: d=2 M={M} region measures sum == 1",
              abs(sum(measures) - 1.0) < TOL)

    # LP feasibility sets
    for d, K, M, exp in ((2, K3, 3, 6), (2, K4p, 4, 12)):
        feas = [p for p in itertools.permutations(range(M))
                if lp_feasible(d, K, p)]
        _, winners, _ = arrangement_2d(K)
        wset = set(winners)
        check(f"E6: LP d=2 M={M} feasible set size == {exp}",
              len(feas) == exp, f"n={len(feas)}")
        check(f"E6: LP d=2 M={M} feasible set == arrangement winners",
              set(feas) == wset)

    # d=3, d=8: M=3 (6) and M=4 full-rank (24)
    for d in (3, 8):
        K3d = [np.array([1.0, 0.0, 0.0] + [0.0] * (d - 3)),
               np.array([0.0, 1.0, 0.0] + [0.0] * (d - 3)),
               np.array([1.0, 1.0, 0.0] + [0.0] * (d - 3)) / math.sqrt(2.0)]
        u = np.array([1.0, 1.0, 1.0] + [0.0] * (d - 3)) / math.sqrt(3.0)
        K4d = [np.array([1.0, 0.0, 0.0] + [0.0] * (d - 3)),
               np.array([0.0, 1.0, 0.0] + [0.0] * (d - 3)),
               np.array([0.0, 0.0, 1.0] + [0.0] * (d - 3)), u]
        n3 = sum(lp_feasible(d, K3d, p) for p in itertools.permutations(range(3)))
        n4 = sum(lp_feasible(d, K4d, p) for p in itertools.permutations(range(4)))
        check(f"E6: LP d={d} M=3 all 6 orderings feasible", n3 == 6, f"n={n3}")
        check(f"E6: LP d={d} M=4 (full-rank) all 24 feasible", n4 == 24,
              f"n={n4}")

    # d=1 LP: keys {1,2,3}; positive -> {(2,1,0)}; signed -> {(2,1,0),(0,1,2)}
    K1 = [1.0, 2.0, 3.0]
    pos = [p for p in itertools.permutations(range(3))
           if lp_feasible(1, K1, p, domain="positive")]
    sgn = [p for p in itertools.permutations(range(3))
           if lp_feasible(1, K1, p, domain="signed")]
    check("E6: d=1 positive LP feasible == {(2,1,0)}", set(pos) == {(2, 1, 0)},
          f"feasible={set(pos)}")
    check("E6: d=1 signed LP feasible == {(2,1,0),(0,1,2)}",
          set(sgn) == {(2, 1, 0), (0, 1, 2)}, f"feasible={set(sgn)}")

    # ---------------- report ----------------
    nfail = 0
    for name, ok, detail in CHECKS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" +
              (f"   ({detail})" if detail and not ok else ""))
        if not ok:
            nfail += 1
    print("-" * 78)
    print(f"theory audit: {len(CHECKS)} checks, {nfail} FAIL")
    if nfail:
        print("STOP AND DEBUG (pre-registered stop rule: do not proceed "
              "to Monte Carlo until fixed).")
        return 1
    print("THEORY AUDIT PASS: all theorem-level expectations verified "
          "exactly; Monte Carlo stage may proceed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
