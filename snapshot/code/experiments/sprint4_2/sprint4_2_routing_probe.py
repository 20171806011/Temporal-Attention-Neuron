#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sprint4_2_routing_probe.py
==========================
TAN-II Sprint 4.2-B v1 -- randomized routing probe + shortcut audit.

Pre-registered in docs/TANII_SPRINT4_2B_PREREGISTRATION.md (benchmarks
frozen therein; Amendment 2 for the unnormalized control; Amendment 3 for
corrected fp constants). Ensemble protocol: two events x_A, x_B iid
U(0.3, 2.5), positions uniform over the 12 ordered slot pairs, pads 0,
probe Q in {3.0, 4.0}; FlipRate_CF = P_H[winner(Q_A) != winner(Q_B)] with
winner = argmax of the E-branch energy over the two event slots.

Deterministic quadrature benchmarks (2001x2001 grid, frozen):
  M0: 0 | M1: 1 | M2: 0.444527 | M2unnorm: 0 | M3d3: 0.054937 |
  M3d4: 0.028391 | M3d8: 0.017159 | M4: 0 | M5E: 0.444527 | M5I: 0.405772

MC: 3 seeds x N_H = 10,000; criterion |p_hat - bench| <= 3*SE + 0.002.
Any FAIL -> exit 1.

Usage: python sprint4_2_routing_probe.py [--outdir DIR]
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEEDS = [20260904, 20260905, 20260906]
N_H = 10000
CRIT = 0.002
C = 2.0
EPS_E = 0.0
EPS_I = 0.3
OMEGA = 0.4
QA, QB = 3.0, 4.0
LO, HI = 0.3, 2.5
GRID_N = 2001
OMEGA_SWEEP = [0.2, 0.4, math.pi / 2.0, math.pi]
Q_SCAN = np.round(np.arange(0.5, 5.0001, 0.05), 10)
PAIRS12 = list(itertools.permutations(range(4), 2))

FROZEN = {"M0": 0.0, "M1": 1.0, "M2": 0.444527, "M2unnorm": 0.0,
          "M3d3": 0.054937, "M3d4": 0.028391, "M3d8": 0.017159,
          "M4": 0.0, "M5E": 0.444527, "M5I": 0.405772}

CHECKS = []
N_FAIL = 0


def check(name, ok, detail=""):
    global N_FAIL
    if not ok:
        N_FAIL += 1
    CHECKS.append(dict(name=name, status="PASS" if ok else "FAIL",
                       detail=detail))


def phi(x, d, norm=True):
    p = np.array([x ** k for k in range(1, d + 1)], dtype=float)
    if x == 0.0 or not norm:
        return p
    return p / np.linalg.norm(p)


def uhat(theta, d):
    v = np.zeros(d)
    v[0] = math.cos(theta)
    v[1] = math.sin(theta)
    return v


def event_energy(model, x, Q, eps, d=2, norm=True, sign=1.0):
    """E-branch energy of one event with value x, given the OTHER event y
    enters the window mean via (x + y + Q)/5. The caller passes the pair."""
    # x here is the event value; the mean needs the pair sum, handled by
    # caller via S. This helper takes S directly:
    pass


def energy_of(x, S, model, d=2, norm=True, sign=1.0):
    xa = np.asarray(x, dtype=float)
    Sa = np.asarray(S, dtype=float)
    if model in ("M0", "M4", "M1"):
        return sign * C * Sa * xa
    if norm:
        nv = np.sqrt(np.sum([xa ** (2 * k) for k in range(1, d + 1)], axis=0))
        p1 = xa / np.maximum(nv, 1e-300)
        p2 = xa ** 2 / np.maximum(nv, 1e-300)
    else:
        p1, p2 = xa, xa ** 2
    return np.cos(OMEGA * Sa) * p1 + np.sin(OMEGA * Sa) * p2


def quadrature(model, eps=0.0, d=2, norm=True, sign=1.0, omega=OMEGA):
    grid = np.linspace(LO, HI, GRID_N)
    XX, YY = np.meshgrid(grid, grid)
    s3 = np.maximum(0.0, 0.8 * QA - 0.2 * (XX + YY) - eps)
    s4 = np.maximum(0.0, 0.8 * QB - 0.2 * (XX + YY) - eps)
    if model == "M1":
        # winner(+1) vs winner(-1): always opposite for xA != xB;
        # ties (diagonal of the quadrature grid) excluded from denominator
        m = (XX != YY)
        return float(np.mean((s3 > 0)[m]))
    th3, th4 = omega * s3, omega * s4
    if model in ("M0", "M4"):
        eA3 = sign * C * s3 * XX
        eB3 = sign * C * s3 * YY
        eA4 = sign * C * s4 * XX
        eB4 = sign * C * s4 * YY
    else:
        dA1 = np.sum([XX ** (2 * k) for k in range(1, d + 1)], axis=0)
        dB1 = np.sum([YY ** (2 * k) for k in range(1, d + 1)], axis=0)
        if norm:
            nA = np.sqrt(dA1)
            nB = np.sqrt(dB1)
            a1, a2 = XX / np.maximum(nA, 1e-300), XX ** 2 / np.maximum(nA, 1e-300)
            b1, b2 = YY / np.maximum(nB, 1e-300), YY ** 2 / np.maximum(nB, 1e-300)
        else:
            a1, a2 = XX, XX ** 2
            b1, b2 = YY, YY ** 2
        d1, d2 = a1 - b1, a2 - b2
        p3 = np.cos(th3) * d1 + np.sin(th3) * d2
        p4 = np.cos(th4) * d1 + np.sin(th4) * d2
        return float(np.mean((p3 * p4 < 0) & (s3 > 0) & (s4 > 0)))
    flip = ((eA3 - eB3) * (eA4 - eB4) < 0) & (s3 > 0) & (s4 > 0)
    return float(np.mean(flip))


def run(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    summary = dict(prereg="docs/TANII_SPRINT4_2B_PREREGISTRATION.md",
                   amendments=["docs/TANII_SPRINT4_2B_AMENDMENT_2.md",
                               "docs/TANII_SPRINT4_2B_AMENDMENT_3.md"],
                   params=dict(seeds=SEEDS, N_H=N_H, criterion=CRIT,
                               lo=LO, hi=HI, grid_n=GRID_N, omega=OMEGA,
                               qA=QA, qB=QB))

    # ---- recompute quadratures and audit against frozen values ----
    quads = {}
    for name, (model, eps, d, norm, sign) in {
            "M0": ("M0", 0.0, 2, True, 1.0),
            "M1": ("M1", 0.0, 2, True, 1.0),
            "M2": ("M2", 0.0, 2, True, 1.0),
            "M2unnorm": ("M2", 0.0, 2, False, 1.0),
            "M3d3": ("M3", 0.0, 3, True, 1.0),
            "M3d4": ("M3", 0.0, 4, True, 1.0),
            "M3d8": ("M3", 0.0, 8, True, 1.0),
            "M4": ("M4", 0.0, 2, True, 1.0),
            "M5E": ("M2", 0.0, 2, True, 1.0),
            "M5I": ("M2", 0.3, 2, True, 1.0)}.items():
        q = quadrature(model, eps, d, norm, sign)
        quads[name] = q
        check(f"quadrature audit: {name} recomputed {q:.6f} vs frozen "
              f"{FROZEN[name]}",
              abs(q - FROZEN[name]) < 1e-6)

    # ---- MC ensemble ----
    mc = {}
    for name, (model, eps, d, norm, sign) in {
            "M0": ("M0", 0.0, 2, True, 1.0),
            "M1": ("M1", 0.0, 2, True, 1.0),
            "M2": ("M2", 0.0, 2, True, 1.0),
            "M2unnorm": ("M2", 0.0, 2, False, 1.0),
            "M3d3": ("M3", 0.0, 3, True, 1.0),
            "M3d4": ("M3", 0.0, 4, True, 1.0),
            "M3d8": ("M3", 0.0, 8, True, 1.0),
            "M4": ("M4", 0.0, 2, True, 1.0),
            "M5E": ("M2", 0.0, 2, True, 1.0),
            "M5I": ("M2", 0.3, 2, True, 1.0)}.items():
        ests = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            xA = rng.uniform(LO, HI, N_H)
            xB = rng.uniform(LO, HI, N_H)
            pos = rng.integers(0, 12, N_H)   # protocol position draw
            s3 = np.maximum(0.0, 0.8 * QA - 0.2 * (xA + xB) - eps)
            s4 = np.maximum(0.0, 0.8 * QB - 0.2 * (xA + xB) - eps)
            if model == "M1":
                # sign counterfactual: winner(+1) vs winner(-1)
                flip = (xA != xB) & (s3 > 0)
            else:
                eA3 = energy_of(xA, s3, model, d, norm, sign)
                eB3 = energy_of(xB, s3, model, d, norm, sign)
                eA4 = energy_of(xA, s4, model, d, norm, sign)
                eB4 = energy_of(xB, s4, model, d, norm, sign)
                flip = ((eA3 - eB3) * (eA4 - eB4) < 0) & (s3 > 0) & (s4 > 0)
            est = float(np.mean(flip))
            ests.append(est)
            se = math.sqrt(est * (1.0 - est) / N_H)
            ok = abs(est - FROZEN[name]) <= 3.0 * se + CRIT
            check(f"MC {name} seed {seed}: {est:.5f} vs bench "
                  f"{FROZEN[name]} (3SE+crit)", ok)
        mc[name] = dict(estimates=ests,
                        mean=float(np.mean(ests)),
                        std=float(np.std(ests)))
    summary["quadratures"] = quads
    summary["mc"] = mc

    # ---- flip-vs-Q scan (canonical history [1,0,2,0,Q]) ----
    scan = {}
    def hist_winner(model, Q, d=2, norm=True, eps=0.0, sign=1.0):
        v = np.array([1.0, 0.0, 2.0, 0.0, Q])
        mu = float(v.mean())
        S = max(0.0, Q - mu - eps)
        if model in ("M0", "M4", "M1"):
            lg = sign * C * S * v
        else:
            u = uhat(OMEGA * S, d)
            lg = np.array([float(u @ phi(x, d, norm)) if x > 0 else 0.0
                           for x in v])
        return "A" if lg[0] > lg[2] else "B"
    for name, (model, d, norm, eps, sign) in {
            "M0": ("M0", 2, True, 0.0, 1.0),
            "M2": ("M2", 2, True, 0.0, 1.0),
            "M2unnorm": ("M2", 2, False, 0.0, 1.0),
            "M5E": ("M2", 2, True, 0.0, 1.0),
            "M5I": ("M2", 2, True, 0.3, 1.0)}.items():
        scan[name] = [hist_winner(model, float(q), d, norm, eps, sign)
                      for q in Q_SCAN]
    summary["flip_q_scan"] = {k: v for k, v in scan.items()}
    summary["flip_q_grid"] = [float(q) for q in Q_SCAN]
    check("flip-Q: M2 winner at Q<=3.70 is A and at Q>=3.75 is B",
          all(w == "A" for q, w in zip(Q_SCAN, scan["M2"]) if q <= 3.70) and
          all(w == "B" for q, w in zip(Q_SCAN, scan["M2"]) if q >= 3.75))
    check("flip-Q: M2 endpoints A(3.0)/B(4.0)",
          hist_winner("M2", 3.0) == "A" and hist_winner("M2", 4.0) == "B")
    check("flip-Q: M0 and M2unnorm flat B (no flip anywhere)",
          all(w == "B" for w in scan["M0"]) and
          all(w == "B" for w in scan["M2unnorm"]))
    check("flip-Q: M5-E same crossing as M2",
          scan["M5E"] == scan["M2"])
    check("flip-Q: M5-I winner A at Q<=4.05 and B at Q>=4.10",
          all(w == "A" for q, w in zip(Q_SCAN, scan["M5I"]) if q <= 4.05) and
          all(w == "B" for q, w in zip(Q_SCAN, scan["M5I"]) if q >= 4.10))
    check("flip-Q: M5-I endpoints A(3.0)/A(4.0)",
          hist_winner("M2", 3.0, eps=0.3) == "A" and
          hist_winner("M2", 4.0, eps=0.3) == "A")

    # ---- omega sweep quadratures (robustness landscape) ----
    om = {}
    for w in OMEGA_SWEEP:
        om[str(w)] = dict(M2=quadrature("M2", 0.0, 2, True, 1.0, omega=w),
                          M2unnorm=quadrature("M2", 0.0, 2, False, 1.0,
                                              omega=w))
    summary["omega_sweep_quadratures"] = om
    check("omega sweep: frozen omega=0.4 quadratures reproduced",
          abs(om[str(OMEGA)]["M2"] - 0.444527) < 1e-6 and
          abs(om[str(OMEGA)]["M2unnorm"] - 0.0) < 1e-12)

    # ---- shortcut audit ----
    eq = True
    for d in (2, 3, 4, 8):
        for x in (0.3, 1.0, 2.0, 2.5):
            if abs(np.linalg.norm(phi(x, d)) - 1.0) > 1e-12:
                eq = False
    check("shortcut: normalized keys are exactly unit-norm (equal-norm "
          "automatic)", eq)
    summary["shortcut_notes"] = [
        "winner defined over event slots {pA,pB} only (probe never a "
        "candidate; probe slot reported separately in canonical audit)",
        "amplitudes/positions randomized per protocol (12 ordered pairs)",
        "canonical pad-attention fraction and flip margins in "
        "sprint4_2b_summary.json"]

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, ls in (("M0", "0.6"), ("M2", "C0"), ("M2unnorm", "0.3"),
                     ("M5E", "C0"), ("M5I", "C2")):
        wv = np.array([1.0 if w == "A" else -1.0 for w in scan[name]])
        ax.step(Q_SCAN, wv, where="post", label=name,
                lw=2.5 if name == "M2" else 1.4, alpha=0.9 if name == "M2"
                else 0.7)
    ax.axvline(QA, color="k", ls=":", lw=1)
    ax.axvline(QB, color="k", ls=":", lw=1)
    ax.axvline(3.7071, color="C0", ls="--", lw=1, alpha=0.6)
    ax.text(3.7071, 1.05, "Q* = 3.7071", color="C0", fontsize=8)
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(["B", "A"])
    ax.set_xlabel("query amplitude Q (canonical history [1,0,2,0,Q])")
    ax.set_ylabel("history winner")
    ax.set_title("Counterfactual winner as function of query amplitude")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(-1.25, 1.45)
    fig.tight_layout()
    fig.savefig(outdir / "sprint4_2b_flip_q.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    names = list(FROZEN.keys())
    means = [mc[n]["mean"] for n in names]
    stds = [mc[n]["std"] for n in names]
    benches = [FROZEN[n] for n in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, means, width=0.4, yerr=stds, capsize=3, color="0.7",
           label="MC (3 seeds)")
    ax.plot(x + 0.2, benches, "r^", ms=8, label="frozen quadrature bench")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("FlipRate_CF")
    ax.set_title("Ensemble counterfactual flip rate: MC vs benchmark")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(outdir / "sprint4_2b_ensemble.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ws = [float(k) for k in om]
    ax.plot(ws, [om[k]["M2"] for k in om], "o-", color="C0",
            label="M2 normalized")
    ax.plot(ws, [om[k]["M2unnorm"] for k in om], "s--", color="0.4",
            label="M2 unnormalized (AM2)")
    ax.set_xscale("log")
    ax.set_xlabel("omega")
    ax.set_ylabel("ensemble FlipRate_CF (quadrature)")
    ax.set_title("Omega robustness landscape (deterministic quadrature)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "sprint4_2b_omega.png", dpi=150)
    plt.close(fig)

    # ---- outputs ----
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary["checks"] = CHECKS
    summary["n_fail"] = N_FAIL
    summary["script_sha256"] = sha
    (outdir / "sprint4_2b_probe_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(outdir / "sprint4_2b_probe_checks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "status", "detail"])
        for c in CHECKS:
            w.writerow([c["name"], c["status"], c["detail"]])
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(
        Path(__file__).resolve().parents[3] / "results" / "sprint4_2"))
    args = ap.parse_args()
    summary = run(Path(args.outdir))
    print("=" * 92)
    print("TAN-II Sprint 4.2-B v1 randomized routing probe")
    print("=" * 92)
    print("MC flip rates vs frozen benchmarks:")
    for n, b in FROZEN.items():
        m = summary["mc"][n]
        print(f"  {n:8s}: bench {b:.6f} | MC {m['mean']:.6f} +- {m['std']:.6f}")
    print("-" * 92)
    for c in summary["checks"]:
        if c["status"] != "PASS":
            print(f"  NON-PASS: {c['name']} ({c['detail']})")
    print(f"checks: {len(summary['checks'])} | FAIL: {summary['n_fail']}")
    if summary["n_fail"]:
        print("STOP AND DEBUG.")
    else:
        print("ALL PROBE CHECKS PASS: ensemble FlipRate_CF matches frozen "
              "benchmarks for all models; vector-QK (M2/M5-E) realizes "
              "query-conditioned winner flips; scalar models do not; "
              "unnormalized control is exactly 0 (AM2).")
    print(f"outputs written to: {args.outdir}")
    return 1 if summary["n_fail"] else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
