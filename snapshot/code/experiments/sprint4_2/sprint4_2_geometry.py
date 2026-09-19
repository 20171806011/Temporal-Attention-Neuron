#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sprint4_2_geometry.py
=====================
TAN-II Sprint 4.2-A -- pure query-key geometry, Monte Carlo stage.

Pre-registered in docs/TANII_SPRINT4_2_PREREGISTRATION.md; theorem-level
exact audit in sprint4_2_theory_audit.py; E4(iii) handled per
docs/TANII_SPRINT4_2_AMENDMENT_1.md (PROTOCOL-FAIL / NUMERICAL-PASS).

Experiments:
  E1  pairwise winner measure P[w=A]      (analytic 1/2; Case A deterministic)
  E2  query-induced winner variation F_d  (analytic 1/2; Case A 0; negative
      control: constant across d in {2,3,4,8})
  E3  counterfactual flip rate FlipRate_CF(theta) = theta/pi (d>=2, 11 angles)
      + E3b unequal-norm control + d=1 three cases
  E4  JSD & winner-reordering conjunction: three fixed histories (closed
      form) + ensemble fractions at theta = pi/2
  E6  ordering-capacity MC cross-check (dense angular sampling, counts match
      exact arrangement enumeration {6,12})

Agreement criterion (pre-registered): |p_hat - p_an| <= 3*SE + 0.002,
SE = sqrt(p_hat(1-p_hat)/N); deterministic quantities: exact equality
(1e-12). Any genuine FAIL -> exit 1, STOP AND DEBUG.

Usage: python sprint4_2_geometry.py [--outdir DIR]
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

# ---- pre-registered constants (docs/TANII_SPRINT4_2_PREREGISTRATION.md) ----
SEEDS = [20260904, 20260905, 20260906]
N_Q = 10000          # queries per seed
N_H = 10000          # histories per seed
N_ANG = 200000       # dense angular samples (E6 cross-check)
DS = [2, 3, 4, 8]
THETA_GRID = [0.0, math.pi / 12.0, math.pi / 6.0, math.pi / 4.0, math.pi / 3.0,
              math.pi / 2.0, 2.0 * math.pi / 3.0, 3.0 * math.pi / 4.0,
              5.0 * math.pi / 6.0, 11.0 * math.pi / 12.0, math.pi]
CRIT = 0.002
ZCI = 1.96
E3B_THETAS = [math.pi / 6.0, math.pi / 2.0, 5.0 * math.pi / 6.0]

AMEND1 = "docs/TANII_SPRINT4_2_AMENDMENT_1.md"

CHECKS = []
N_FAIL = 0


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def bernoulli_jsd_kl(p, q):
    p = min(max(p, 1e-300), 1.0 - 1e-300)
    q = min(max(q, 1e-300), 1.0 - 1e-300)
    m = 0.5 * (p + q)

    def kl(x, y):
        return x * math.log(x / y) + (1.0 - x) * math.log((1.0 - x) / (1.0 - y))
    return 0.5 * (kl(p, m) + kl(q, m))


def bernoulli_jsd_entropy(p, q):
    """Independent path: JSD = H(m) - 0.5*(H(p)+H(q)), nats."""

    def h(x):
        x = min(max(x, 1e-300), 1.0 - 1e-300)
        return -x * math.log(x) - (1.0 - x) * math.log(1.0 - x)
    m = 0.5 * (p + q)
    return h(m) - 0.5 * (h(p) + h(q))


def unit_ball(rng, n, d):
    g = rng.standard_normal((n, d))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return g


def record(experiment, d, case, pair, theta_deg, seed, est, an, se, status,
           note=""):
    global N_FAIL
    if status == "FAIL":
        N_FAIL += 1
    CHECKS.append(dict(experiment=experiment, d=d, case=case, pair=pair,
                       theta_deg=theta_deg, seed=seed,
                       estimate=None if est is None else float(est),
                       analytic=None if an is None else float(an),
                       SE=None if se is None else float(se),
                       CI_lo=None if se is None else float(est - ZCI * se),
                       CI_hi=None if se is None else float(est + ZCI * se),
                       status=status, note=note))


def prob_check(experiment, d, case, pair, theta_deg, seed, est, an, n_eff):
    se = math.sqrt(max(est, 0.0) * max(1.0 - est, 0.0) / max(n_eff, 1))
    ok = abs(est - an) <= 3.0 * se + CRIT
    record(experiment, d, case, pair, theta_deg, seed, est, an, se,
           "PASS" if ok else "FAIL",
           "" if ok else "outside pre-registered criterion")


def exact_check(experiment, d, case, pair, theta_deg, seed, val, an, note=""):
    ok = abs(val - an) < 1e-12
    record(experiment, d, case, pair, theta_deg, seed, val, an, None,
           "PASS" if ok else "FAIL", note)


# ---------------------------------------------------------------------------
def run(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    summary = dict(prereg="docs/TANII_SPRINT4_2_PREREGISTRATION.md",
                   theory_audit="code/experiments/sprint4_2/"
                                "sprint4_2_theory_audit.py",
                   amendment1=AMEND1,
                   params=dict(seeds=SEEDS, N_Q=N_Q, N_H=N_H, N_ANG=N_ANG,
                               DS=DS, criterion=CRIT))

    d1_pairs = {"S1": (1.0, 2.0), "S2": (-1.0, 1.0), "S3": (2.0, 1.0)}

    def pairs_d(d):
        e1 = np.zeros(d)
        e1[0] = 1.0
        ed = np.zeros(d)
        ed[-1] = 1.0
        return {"P1": (e1, ed), "P2": (e1, 3.0 * ed),
                "P3": (e1, e1 + 0.05 * ed)}

    # ================= E1: pairwise winner measure =================
    e1 = {}
    for d in DS:
        for pname, (Ka, Kb) in pairs_d(d).items():
            dK = Ka - Kb
            ests = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                Q = unit_ball(rng, N_Q, d)
                est = float(np.mean(Q @ dK > 0.0))
                ests.append(est)
                prob_check("E1", d, "iso", pname, None, seed, est, 0.5, N_Q)
            e1[f"d{d}_{pname}"] = ests
    for pname, (ka, kb) in d1_pairs.items():
        dK = ka - kb
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            qA = rng.uniform(0.01, 4.0, N_Q)
            estA = float(np.mean(qA * dK > 0.0))
            exact_check("E1", 1, "CaseA", pname, None, seed, estA,
                        1.0 if dK > 0 else 0.0)
            qB = rng.standard_normal(N_Q)
            estB = float(np.mean(qB * dK > 0.0))
            prob_check("E1", 1, "CaseB", pname, None, seed, estB, 0.5, N_Q)
            qC = rng.choice(np.array([-1.0, 1.0]), N_Q)
            estC = float(np.mean(qC * dK > 0.0))
            prob_check("E1", 1, "CaseC", pname, None, seed, estC, 0.5, N_Q)

    # ================= E2: winner variation F_d =================
    e2 = {}
    for d in DS:
        fd = []
        for pname, (Ka, Kb) in pairs_d(d).items():
            dK = Ka - Kb
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                Q = unit_ball(rng, 2 * N_Q, d)
                w = Q @ dK > 0.0
                est = float(np.mean(w[0::2] != w[1::2]))
                fd.append(est)
                prob_check("E2", d, "iso", pname, None, seed, est, 0.5,
                           2 * N_Q)
        e2[f"d{d}"] = fd
    for pname, (ka, kb) in d1_pairs.items():
        dK = ka - kb
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            qA = rng.uniform(0.01, 4.0, 2 * N_Q)
            wA = qA * dK > 0.0
            estA = float(np.mean(wA[0::2] != wA[1::2]))
            exact_check("E2", 1, "CaseA", pname, None, seed, estA, 0.0)
            qB = rng.standard_normal(2 * N_Q)
            wB = qB * dK > 0.0
            estB = float(np.mean(wB[0::2] != wB[1::2]))
            prob_check("E2", 1, "CaseB", pname, None, seed, estB, 0.5,
                       2 * N_Q)
            qC = rng.choice(np.array([-1.0, 1.0]), 2 * N_Q)
            wC = qC * dK > 0.0
            estC = float(np.mean(wC[0::2] != wC[1::2]))
            prob_check("E2", 1, "CaseC", pname, None, seed, estC, 0.5,
                       2 * N_Q)

    # ================= E3: FlipRate_CF(theta) =================
    e3 = {}
    for d in DS:
        curve = {}
        for th in THETA_GRID:
            ests = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                dK = unit_ball(rng, N_H, d)
                flip = (dK[:, 0] * (math.cos(th) * dK[:, 0]
                                    + math.sin(th) * dK[:, 1]) < 0.0)
                est = float(np.mean(flip))
                ests.append(est)
                prob_check("E3", d, "iso", "equal-norm", round(th / math.pi, 4),
                           seed, est, th / math.pi, N_H)
            curve[round(th / math.pi, 4)] = ests
        e3[f"d{d}"] = curve
    # E3b: unequal-norm control (K_A=dK, K_B=-0.1 dK -> same dK direction)
    for d in DS:
        for th in E3B_THETAS:
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                dK = unit_ball(rng, N_H, d)
                flip = (dK[:, 0] * (math.cos(th) * dK[:, 0]
                                    + math.sin(th) * dK[:, 1]) < 0.0)
                est = float(np.mean(flip))
                prob_check("E3b", d, "iso", "unequal-norm", round(th / math.pi, 4),
                           seed, est, th / math.pi, N_H)
    # d=1
    dK1 = d1_pairs["S1"][0] - d1_pairs["S1"][1]   # -1.0
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        gA = rng.uniform(0.01, 4.0, (2, N_Q))
        flipA = ((gA[0] * dK1 > 0.0) != (gA[1] * dK1 > 0.0))
        exact_check("E3", 1, "CaseA", "S1", None, seed, float(np.mean(flipA)),
                    0.0)
        gB = rng.standard_normal((2, N_Q))
        flipB = ((gB[0] * dK1 > 0.0) != (gB[1] * dK1 > 0.0))
        prob_check("E3", 1, "CaseB", "S1", None, seed, float(np.mean(flipB)),
                   0.5, N_Q)
    # Case C: deterministic query pairs
    for (qa, qb), an in (((1.0, 1.0), 0.0), ((1.0, -1.0), 1.0)):
        flip = (qa * dK1 > 0.0) != (qb * dK1 > 0.0)
        exact_check("E3", 1, "CaseC", f"Q=({qa:+.0f},{qb:+.0f})", None,
                    "det", 1.0 if flip else 0.0, an)

    # ================= E4: JSD conjunction =================
    e4 = {}
    jsd1 = bernoulli_jsd_kl(sigmoid(-1.0), sigmoid(-4.0))
    flip1 = (1.0 * (1.0 - 2.0) > 0.0) != (4.0 * (1.0 - 2.0) > 0.0)
    ok1 = jsd1 > 0.05
    record("E4", 1, "CaseA-trap", "fixed", None, "closed-form", jsd1, 0.05, None,
           "PASS" if ok1 else "FAIL", "JSD>0.05 assertion")
    exact_check("E4", 1, "CaseA-trap", "fixed", None, "closed-form",
                1.0 if flip1 else 0.0, 0.0, "flip==0 assertion")

    jsd2 = bernoulli_jsd_kl(sigmoid(1.0), sigmoid(math.cos(2 * math.pi / 3.0)))
    flip2 = (1.0 > 0.0) != (math.cos(2 * math.pi / 3.0) > 0.0)
    ok2 = jsd2 > 0.05
    record("E4", 2, "flip-case", "fixed", 120.0, "closed-form", jsd2, 0.05,
           None, "PASS" if ok2 else "FAIL", "JSD>0.05 assertion")
    exact_check("E4", 2, "flip-case", "fixed", 120.0, "closed-form",
                1.0 if flip2 else 0.0, 1.0, "flip==1 assertion")

    # E4(iii) per Amendment 1: PROTOCOL-FAIL / NUMERICAL-PASS
    jsd3 = bernoulli_jsd_kl(sigmoid(1.0), sigmoid(math.cos(math.pi / 6.0)))
    jsd3_ent = bernoulli_jsd_entropy(sigmoid(1.0),
                                     sigmoid(math.cos(math.pi / 6.0)))
    flip3 = (1.0 > 0.0) != (math.cos(math.pi / 6.0) > 0.0)
    bench3 = jsd3
    record("E4", 2, "no-flip-case(iii)", "fixed", 30.0, "closed-form", jsd3,
           0.0005, None, "PROTOCOL-FAIL / NUMERICAL-PASS",
           "Amendment 1: original prediction 0.00103 INVALID (arithmetic "
           "error); exact closed form 0.00045439; agreement with corrected "
           f"benchmark < 5e-4; flip==0 ({flip3})")
    exact_check("E4", 2, "no-flip-case(iii)", "fixed", 30.0, "closed-form",
                jsd3_ent, bench3, "independent entropy-identity path")
    exact_check("E4", 2, "no-flip-case(iii)", "fixed", 30.0, "closed-form",
                1.0 if flip3 else 0.0, 0.0, "flip==0 qualitative claim")
    e4["fixed"] = dict(jsd1=jsd1, flip1=bool(flip1), jsd2=jsd2,
                       flip2=bool(flip2), jsd3=jsd3, jsd3_entropy=jsd3_ent,
                       flip3=bool(flip3), amendment=AMEND1)

    # ensemble: d=2, theta=pi/2
    th = math.pi / 2.0
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        dK = unit_ball(rng, N_H, 2)
        a = dK[:, 0]
        b = math.cos(th) * dK[:, 0] + math.sin(th) * dK[:, 1]
        jsd_vec = np.array([bernoulli_jsd_kl(sigmoid(ai), sigmoid(bi))
                            for ai, bi in zip(a, b)])
        frac_jsd = float(np.mean(jsd_vec > 1e-9))
        flip_est = float(np.mean(a * b < 0.0))
        okj = frac_jsd >= 0.999
        record("E4", 2, "ensemble", "fixed", 90.0, seed, frac_jsd, 1.0, None,
               "PASS" if okj else "FAIL", "P[JSD>1e-9] >= 0.999 assertion")
        prob_check("E4", 2, "ensemble", "fixed", 90.0, seed, flip_est, 0.5,
                   N_H)
    e4["ensemble_theta_pi2"] = dict(fraction_jsd=frac_jsd, flip=flip_est)

    # ================= E6: ordering-capacity MC cross-check =================
    e6 = {}
    K3 = [np.array([1.0, 0.0]), np.array([0.0, 1.0]),
          np.array([1.0, 1.0]) / math.sqrt(2.0)]
    K4 = [np.array([1.0, 0.0]), np.array([0.0, 1.0]),
          np.array([1.0, 1.0]) / math.sqrt(2.0),
          np.array([2.0, 1.0]) / math.sqrt(5.0)]
    rng = np.random.default_rng(SEEDS[0])
    phi = rng.uniform(0.0, 2.0 * math.pi, N_ANG)
    Qs = np.stack([np.cos(phi), np.sin(phi)], axis=1)
    for K, M, exp in ((K3, 3, 6), (K4, 4, 12)):
        seen = set()
        for q in Qs:
            scores = [float(q @ k) for k in K]
            seen.add(tuple(sorted(range(M), key=lambda i: -scores[i])))
        record("E6", 2, "dense-MC", f"M{M}", None, SEEDS[0], len(seen), exp,
               None, "PASS" if len(seen) == exp else "FAIL",
               "distinct orderings == exact arrangement count")
        e6[f"M{M}"] = dict(observed=len(seen), expected=exp)

    # ================= figures =================
    # E1 bars
    fig, ax = plt.subplots(figsize=(9, 5))
    import itertools
    labels, means = [], []
    for d in DS:
        for pn in ("P1", "P2", "P3"):
            labels.append(f"d{d} {pn}")
            means.append(float(np.mean(e1[f"d{d}_{pn}"])))
    ax.bar(labels, means, color="0.7")
    ax.axhline(0.5, color="r", ls="--", label="analytic 1/2")
    for lb, mv in zip(labels, means):
        ax.text(lb, mv + 0.005, f"{mv:.3f}", ha="center", fontsize=7)
    ax.set_ylim(0.4, 0.6)
    ax.set_title("E1: pairwise winner measure P[w=A] (d=2,3,4,8; 3 pairs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "sprint4_2_winner_measure.png", dpi=150)
    plt.close(fig)

    # E2 bars
    fig, ax = plt.subplots(figsize=(7, 5))
    for d in DS:
        vals = e2[f"d{d}"]
        ax.bar(d - 0.15, float(np.mean(vals)), width=0.3, color="0.7",
               yerr=np.std(vals) / math.sqrt(len(vals)))
        ax.text(d - 0.15, float(np.mean(vals)) + 0.004,
                f"{np.mean(vals):.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, color="r", ls="--", label="analytic 1/2")
    ax.set_xticks(DS)
    ax.set_title("E2: F_d = P[w(Q1)!=w(Q2)]  (negative control: "
                 "dimension-independent)")
    ax.set_ylim(0.4, 0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "sprint4_2_fd.png", dpi=150)
    plt.close(fig)

    # E3 theta curve
    fig, ax = plt.subplots(figsize=(10, 6))
    thx = [th / math.pi for th in THETA_GRID]
    ax.plot(thx, thx, "k-", lw=2, label="analytic theta/pi")
    colors = {2: "C0", 3: "C1", 4: "C2", 8: "C3"}
    for d in DS:
        curve = e3[f"d{d}"]
        means = [float(np.mean(curve[round(t, 4)])) for t in thx]
        sds = [float(np.std(curve[round(t, 4)])) for t in thx]
        ax.errorbar(thx, means, yerr=sds, fmt="o", ms=4, color=colors[d],
                    capsize=2, label=f"MC d={d}")
    # d=1 degenerate endpoints
    ax.plot([0.0, 1.0], [0.0, 1.0], "rv", ms=9,
            label="d=1 Case C: (+1,+1)->0, (+1,-1)->1")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.text(0.52, 0.51, "d=1 Case B (sign flip) = 1/2", color="gray",
            fontsize=8)
    ax.plot([0.0], [0.0], "b^", ms=8, label="d=1 Case A (positive) = 0")
    ax.set_xlabel("theta (units of pi)")
    ax.set_ylabel("FlipRate_CF")
    ax.set_title("E3: counterfactual flip rate vs query angle "
                 "(dimension-independent; S0 vs S1)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "sprint4_2_flip_theta.png", dpi=150)
    plt.close(fig)

    # E4 bars
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.bar(["A-scalar trap", "d=2 flip (120deg)", "d=2 no-flip (30deg)"],
           [jsd1, jsd2, jsd3], color=["0.6", "0.6", "1.0"])
    ax.axhline(0.0005, color="r", ls="--", lw=1, label="original threshold")
    ax.text(2, jsd3 + 0.0004, f"{jsd3:.6f}", ha="center", fontsize=9)
    ax.text(0, jsd1 + 0.002, f"{jsd1:.4f} (flip=0)", ha="center", fontsize=8)
    ax.text(1, jsd2 + 0.002, f"{jsd2:.4f} (flip=1)", ha="center", fontsize=8)
    ax.text(2, 0.00042, "PROTOCOL-FAIL /\nNUMERICAL-PASS", ha="center",
            fontsize=7, color="red")
    ax.set_title("E4 fixed histories: JSD with flip labels")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.bar(["P[JSD>1e-9]", "P[winner flip]"], [frac_jsd, flip_est],
           color=["0.6", "0.9"])
    ax.axhline(0.5, color="r", ls="--", lw=1)
    ax.set_ylim(0.4, 1.05)
    ax.set_title("E4 ensemble theta=pi/2: distribution-change vs reordering")
    fig.suptitle("E4: JSD alone is insufficient; require JSD>0 AND flip")
    fig.tight_layout()
    fig.savefig(outdir / "sprint4_2_jsd.png", dpi=150)
    plt.close(fig)

    # E6 ordering ladder figure (analytic ladder verified by theory audit)
    fig, ax = plt.subplots(figsize=(9, 5))
    cols = ["d=1 pos", "d=1 signed", "d=2", "d=3", "d=8"]
    vals = {"d=1 pos": [1, 1, 1],
            "d=1 signed": [2, 2, 2],
            "d=2": [2, 6, 12],
            "d=3": [math.factorial(m) for m in (2, 3, 4)],
            "d=8": [math.factorial(m) for m in (2, 3, 4)]}
    x = np.arange(3)
    w = 0.15
    for i, c in enumerate(cols):
        ax.bar(x + (i - 2) * w, vals[c], width=w, label=c)
    ax.set_xticks(x)
    ax.set_xticklabels(["M=2", "M=3", "M=4"])
    ax.set_ylabel("query-realizable orderings")
    ax.set_title("E6: ordering-capacity ladder (S0 -> S1 -> higher)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "sprint4_2_ordering.png", dpi=150)
    plt.close(fig)

    # ================= write outputs =================
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary["script_sha256"] = sha
    summary["checks"] = CHECKS
    counts = {}
    for c in CHECKS:
        s = "PASS" if c["status"] == "PASS" else (
            "PROTOCOL" if c["status"].startswith("PROTOCOL") else "FAIL")
        counts[s] = counts.get(s, 0) + 1
    summary["check_counts"] = counts
    summary["n_fail"] = N_FAIL
    summary["amendment1_note"] = ("E4(iii) retained as PROTOCOL-FAIL / "
                                  "NUMERICAL-PASS per " + AMEND1)
    summary["numpy"] = np.__version__
    (outdir / "sprint4_2_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    with open(outdir / "sprint4_2_checks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "d", "case", "pair", "theta_pi", "seed",
                    "estimate", "analytic", "SE", "CI_lo", "CI_hi", "status",
                    "note"])
        for c in CHECKS:
            w.writerow([c["experiment"], c["d"], c["case"], c["pair"],
                        c["theta_deg"], c["seed"],
                        "" if c["estimate"] is None else f"{c['estimate']:.8f}",
                        "" if c["analytic"] is None else f"{c['analytic']:.8f}",
                        "" if c["SE"] is None else f"{c['SE']:.8f}",
                        "" if c["CI_lo"] is None else f"{c['CI_lo']:.8f}",
                        "" if c["CI_hi"] is None else f"{c['CI_hi']:.8f}",
                        c["status"], c["note"]])
    return summary


def console_report(summary):
    out = ["=" * 100,
           "TAN-II Sprint 4.2-A geometry audit (Monte Carlo vs analytic)",
           "=" * 100]
    counts = summary["check_counts"]
    out.append(f"checks: {len(summary['checks'])} total | "
               f"PASS {counts.get('PASS', 0)} | FAIL {counts.get('FAIL', 0)} | "
               f"PROTOCOL {counts.get('PROTOCOL', 0)}")
    out.append(f"criterion: |p_hat - p_an| <= 3*SE + {CRIT}; seeds "
               f"{SEEDS}; N_Q=N_H={N_Q}")
    out.append("-" * 100)
    # E3 headline table (pooled over seeds)
    out.append("E3 FlipRate_CF(theta) vs theta/pi (pooled mean over seeds):")
    import json as _json
    thx = [round(t / math.pi, 4) for t in THETA_GRID]
    for d in DS:
        e3 = _json.loads(_json.dumps(
            [c for c in summary["checks"]
             if c["experiment"] == "E3" and c["d"] == d
             and c["case"] == "iso"]))
        rows = {}
        for c in e3:
            rows.setdefault(c["theta_deg"], []).append(c["estimate"])
        line = "  d=%d: " % d + " ".join(
            "%.3f" % (sum(rows[t]) / len(rows[t])) for t in thx)
        out.append(line)
    out.append("  analytic: " + " ".join("%.3f" % t for t in thx))
    out.append("-" * 100)
    for c in summary["checks"]:
        if c["status"] != "PASS":
            out.append(f"  NON-PASS: {c['experiment']} d={c['d']} "
                       f"{c['case']} {c['pair']} status={c['status']} "
                       f"({c['note']})")
    out.append("-" * 100)
    if summary["n_fail"]:
        out.append(f"STOP AND DEBUG: {summary['n_fail']} genuine FAIL(s).")
    else:
        out.append("ALL GENUINE CHECKS PASS "
                   "(E4(iii) retained as PROTOCOL-FAIL / NUMERICAL-PASS "
                   "per Amendment 1).")
    out.append("=" * 100)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(
        Path(__file__).resolve().parents[3] / "results" / "sprint4_2"))
    args = ap.parse_args()
    summary = run(Path(args.outdir))
    print(console_report(summary))
    print(f"outputs written to: {args.outdir}")
    return 1 if summary["n_fail"] else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
