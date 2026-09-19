#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sprint4_2_shortcut_audit.py
===========================
TAN-II Sprint 4.2-A -- geometric shortcut audit (pure geometry).

Pre-registered in docs/TANII_SPRINT4_2_PREREGISTRATION.md, section E5:
  (a) 2-candidate norm invariance: P[w=A] = 1/2 for norm ratios r in
      {1,2,5,10} (theorem Th3: decision boundary passes through origin).
  (b) distractor norm leak: 3 candidates {K_A=e1, K_B=e2, K_D=m*u},
      u=(e1+e2)/sqrt(2), m in {0.5,1,2,5,10}: exact d=2 arrangement measure
      of P[w=D] (and MC verification vs the exact value); P[D] grows with
      m -- the shortcut that equal-norm / randomized controls must remove.
  (c) symmetric equal-norm 3-set {e1, R_120 e1, R_240 e1}: each candidate
      wins exactly 1/3 of query space.

Agreement criterion: |p_hat - p_exact| <= 3*SE + 0.002; deterministic exact
quantities: equality at 1e-12. Any FAIL -> exit 1.

Usage: python sprint4_2_shortcut_audit.py [--outdir DIR]
"""

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEEDS = [20260904, 20260905, 20260906]
N_Q = 10000
CRIT = 0.002
ZCI = 1.96
R_VALS = [1.0, 2.0, 5.0, 10.0]
M_VALS = [0.5, 1.0, 2.0, 5.0, 10.0]
M_GRID = np.round(np.arange(0.4, 10.01, 0.25), 6)

CHECKS = []
N_FAIL = 0


def arrangement_2d(K):
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
        if not ded or abs(a - ded[-1]) > 1e-9:
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


def winner_measure(K, idx):
    _, winners, measures = arrangement_2d(K)
    tot = 0.0
    for w, m in zip(winners, measures):
        if w[0] == idx:
            tot += m
    return tot


def record(name, d, seed, est, an, se, note=""):
    global N_FAIL
    ok = abs(est - an) <= 3.0 * se + CRIT
    if not ok:
        N_FAIL += 1
    CHECKS.append(dict(name=name, d=d, seed=seed, estimate=float(est),
                       analytic=float(an), SE=float(se), status="PASS" if ok
                       else "FAIL", note=note))


def run(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    summary = dict(prereg="docs/TANII_SPRINT4_2_PREREGISTRATION.md",
                   params=dict(seeds=SEEDS, N_Q=N_Q, criterion=CRIT,
                               r_vals=R_VALS, m_vals=M_VALS))

    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    u = (e1 + e2) / math.sqrt(2.0)

    # ---- (a) 2-candidate norm invariance ----
    inv = {}
    for r in R_VALS:
        K = [e1, r * e2]
        exact = winner_measure(K, 0)
        ests = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            g = rng.standard_normal((N_Q, 2))
            g /= np.linalg.norm(g, axis=1, keepdims=True)
            ind = (g @ (K[0] - K[1])) > 0.0
            est = float(np.mean(ind))
            ests.append(est)
            se = math.sqrt(est * (1.0 - est) / N_Q)
            record("5a norm-invariance", 2, seed, est, exact, se,
                   f"r={r}")
        inv[r] = ests
    summary["5a"] = inv

    # ---- (b) distractor norm leak: exact curve + MC verification ----
    curve = {}
    for m in M_GRID:
        K = [e1, e2, m * u]
        curve[float(m)] = winner_measure(K, 2)
    leak = {}
    for m in M_VALS:
        K = [e1, e2, m * u]
        exact = winner_measure(K, 2)
        exactA = winner_measure(K, 0)
        exactB = winner_measure(K, 1)
        ests = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            g = rng.standard_normal((N_Q, 2))
            g /= np.linalg.norm(g, axis=1, keepdims=True)
            scores = g @ np.array(K).T
            w = np.argmax(scores, axis=1)
            est = float(np.mean(w == 2))
            ests.append(est)
            se = math.sqrt(est * (1.0 - est) / N_Q)
            record("5b distractor-leak", 2, seed, est, exact, se, f"m={m}")
        leak[float(m)] = dict(exact=exact, exactA=exactA, exactB=exactB,
                              mc=ests)
    summary["5b"] = dict(curve=curve, points=leak)

    # ---- (c) symmetric equal-norm 3-set ----
    R120 = np.array([math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3)])
    R240 = np.array([math.cos(4 * math.pi / 3), math.sin(4 * math.pi / 3)])
    Ksym = [e1, R120, R240]
    exact_sym = [winner_measure(Ksym, i) for i in range(3)]
    sym = {}
    for i in range(3):
        ests = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            g = rng.standard_normal((N_Q, 2))
            g /= np.linalg.norm(g, axis=1, keepdims=True)
            w = np.argmax(g @ np.array(Ksym).T, axis=1)
            est = float(np.mean(w == i))
            ests.append(est)
            se = math.sqrt(est * (1.0 - est) / N_Q)
            record("5c symmetric-3set", 2, seed, est, exact_sym[i], se,
                   f"candidate {i}")
        sym[i] = ests
    summary["5c"] = dict(exact=exact_sym, mc=sym)

    # ---- figures ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    labels = [str(r) for r in R_VALS]
    means = [float(np.mean(inv[r])) for r in R_VALS]
    ax.bar(labels, means, color="0.75")
    ax.axhline(0.5, color="r", ls="--")
    ax.set_ylim(0.45, 0.55)
    ax.set_title("(a) 2-candidate norm invariance\nP[w=A] vs norm ratio r")
    ax.set_xlabel("r = ||K_B||/||K_A||")

    ax = axes[1]
    ms = [float(m) for m in M_GRID]
    ax.plot(ms, [curve[m] for m in ms], "k-", lw=2, label="exact P[w=D]")
    for m in M_VALS:
        pts = leak[m]["mc"]
        ax.errorbar(m, np.mean(pts), yerr=np.std(pts), fmt="ro", ms=5,
                    capsize=3)
    ax.set_xlabel("distractor norm m")
    ax.set_ylabel("P[w = distractor]")
    ax.set_title("(b) distractor norm leak\n(exact arrangement + MC)")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(["K1", "K2", "K3"], exact_sym, color=["0.75", "0.85", "0.95"])
    ax.axhline(1.0 / 3.0, color="r", ls="--")
    ax.set_ylim(0.0, 0.5)
    ax.set_title("(c) symmetric equal-norm 3-set\nP[w=i] = 1/3 each")
    fig.suptitle("Sprint 4.2-A geometric shortcut audit")
    fig.tight_layout()
    fig.savefig(outdir / "sprint4_2_shortcut.png", dpi=150)
    plt.close(fig)

    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary["script_sha256"] = sha
    summary["checks"] = CHECKS
    summary["n_fail"] = N_FAIL
    summary["numpy"] = np.__version__
    (outdir / "sprint4_2_shortcut_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(outdir / "sprint4_2_shortcut_checks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "d", "seed", "estimate", "exact", "SE", "status",
                    "note"])
        for c in CHECKS:
            w.writerow([c["name"], c["d"], c["seed"], f"{c['estimate']:.8f}",
                        f"{c['analytic']:.8f}", f"{c['SE']:.8f}", c["status"],
                        c["note"]])
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(
        Path(__file__).resolve().parents[3] / "results" / "sprint4_2"))
    args = ap.parse_args()
    summary = run(Path(args.outdir))
    print("=" * 90)
    print("TAN-II Sprint 4.2-A geometric shortcut audit")
    print("=" * 90)
    print(f"5a norm invariance (exact = 1/2): r in {R_VALS}")
    for r in R_VALS:
        print(f"  r={r}: exact 0.5000, MC means "
              f"{[f'{v:.4f}' for v in summary['5a'][r]]}")
    print("5b distractor leak (exact P[w=D]):")
    for m in M_VALS:
        p = summary["5b"]["points"][m]
        print(f"  m={m}: exact {p['exact']:.4f} (A {p['exactA']:.4f}, "
              f"B {p['exactB']:.4f}), MC means "
              f"{[f'{v:.4f}' for v in p['mc']]}")
    print("5c symmetric 3-set (exact = 1/3 each):")
    print(f"  exact: {[f'{v:.4f}' for v in summary['5c']['exact']]}")
    for i in range(3):
        print(f"  K{i+1}: MC means "
              f"{[f'{v:.4f}' for v in summary['5c']['mc'][i]]}")
    print("-" * 90)
    if summary["n_fail"]:
        print(f"STOP AND DEBUG: {summary['n_fail']} FAIL(s).")
    else:
        print("ALL SHORTCUT-AUDIT CHECKS PASS.")
    print(f"outputs written to: {args.outdir}")
    return 1 if summary["n_fail"] else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
