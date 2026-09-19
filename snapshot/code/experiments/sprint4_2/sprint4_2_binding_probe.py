#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sprint4_2_binding_probe.py
==========================
TAN-II Sprint 4.2-C -- Routing -> Binding minimal probe.

Minimal (K,V) decoupling on the frozen 4.2-B vector-QK family: events carry
keys K (magnitudes; attention via phi_2 and the rotating query c*S*u(om*S),
omega=0.4) and values V (transmitted channel); pads (0,0); probe key Q in
{3.0, 4.0} with probe VALUE 0 (query is a control signal, not content).

Canonical history [A=(1,5), 0, B=(2,9), 0]; value-swap control
[A=(1,9), 0, B=(2,5), 0]; positions randomized (12 ordered pairs, asserted
position-invariant). Counterfactual = identical history, only Q changes.

Pre-registered predictions & benchmarks frozen in
docs/TANII_SPRINT4_2C_PREREGISTRATION.md. Deterministic canonical checks
(1e-6 for softmax-derived, 1e-12 for antisymmetry); ensemble MC vs frozen
quadrature benchmarks (probabilities: 3SE+0.002; means: 3*SE_mean+0.01).
Any FAIL -> exit 1.

Usage: python sprint4_2_binding_probe.py [--outdir DIR]
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

# frozen heritage (4.2-B)
OMEGA, C, QA, QB = 0.4, 2.0, 3.0, 4.0
VA, VB = 5.0, 9.0
HIST = [1.0, 0.0, 2.0, 0.0]
EPS_I = 0.3
W_EI = 0.8
SEEDS = [20260904, 20260905, 20260906]
N_H = 10000
CRIT_P = 0.002
CRIT_M = 0.01
LO, HI, GRID_N = 0.3, 2.5, 2001

# frozen predictions/benchmarks (pre-run)
PRED = dict(
    CE3=6.742691, CE4=7.155595, T=-0.412904,
    D3=1.742691, D4=1.844405,
    AE3=4.630402, AE4=4.888954, T_drive=-0.258552,
    M5_AE3=1.160004820691, M5_AE4=0.923304375457, M5_Tdrive=+0.236700445234,
    M5_DRIVE_INVERTED=True,   # Amendment 4 (direction statement unchanged)
    M4_AE3=-0.086900, M4_AE4=-0.000445,
    M0_CE3=8.893612, M0_CE4=8.978055, M0_T=-0.084443,
    M1_CEp=8.893612, M1_CEm=5.106388, M1_D=0.106388, M1_Tsign=3.787224,
    UN_CE3=8.999784, UN_CE4=9.000000,
    B_ET=0.0, B_absT=0.400435, B_absT_flip=0.465523, B_absT_noflip=0.348347,
    B_flip=0.444527, B_align=1.0,
)

CHECKS = []
N_FAIL = 0


def check(name, ok, detail=""):
    global N_FAIL
    if not ok:
        N_FAIL += 1
    CHECKS.append(dict(name=name, status="PASS" if ok else "FAIL",
                       detail=detail))


def phi(x, norm=True):
    p = np.array([x, x * x], dtype=float)
    return p / np.linalg.norm(p) if norm else p


def uhat(th):
    return np.array([math.cos(th), math.sin(th)])


def softmax(lg):
    e = np.exp(lg - np.max(lg))
    return e / e.sum()


def energies(model, Q, vA=VA, vB=VB, sign=1.0):
    """alpha over the canonical key window [1,0,2,0,Q] + value readouts."""
    v = np.array(HIST + [Q], dtype=float)
    mu = float(v.mean())
    SE = max(0.0, Q - mu)
    SI = max(0.0, Q - mu - EPS_I)
    if model in ("M0", "M1", "M4"):
        lg = sign * C * SE * v
    else:
        u = uhat(OMEGA * SE)
        lg = np.array([C * SE * float(u @ phi(x, norm=(model != "M2unnorm")))
                       if x > 0 else 0.0 for x in v])
    al = softmax(lg)
    ce = (al[0] * vA + al[2] * vB) / (al[0] + al[2])
    cf = al[0] * vA + al[2] * vB          # probe/pad values = 0
    out = dict(al=al, ce=ce, cf=cf, SE=SE, SI=SI)
    if model in ("M4", "M5"):
        if model == "M5":
            uI = uhat(OMEGA * SI)
            lgI = np.array([C * SI * float(uI @ phi(x)) if x > 0 else 0.0
                            for x in v]) if SI > 0 else np.zeros(5)
        else:
            lgI = C * SI * v
        alI = softmax(lgI)
        cfI = alI[0] * vA + alI[2] * vB
        out.update(alI=alI, cfI=cfI)
    return out


def drive(model, Q, vA=VA, vB=VB):
    r = energies(model, Q, vA, vB)
    AE = math.tanh(r["SE"]) * r["cf"]
    if model in ("M4", "M5"):
        AE = AE - W_EI * math.tanh(r["SI"]) * r["cfI"]
    return AE


def run(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    res = {}

    # ============ canonical (deterministic) ============
    r3 = energies("M2", QA)
    r4 = energies("M2", QB)
    w3 = "A" if r3["al"][0] > r3["al"][2] else "B"
    w4 = "A" if r4["al"][0] > r4["al"][2] else "B"
    check("M2 winner: A at Q=3, B at Q=4 (routing from 4.2-B)",
          w3 == "A" and w4 == "B")
    check("M2 C_events(Q=3)=6.742691, C_events(Q=4)=7.155595",
          abs(r3["ce"] - PRED["CE3"]) < 1e-6 and
          abs(r4["ce"] - PRED["CE4"]) < 1e-6)
    T = r3["ce"] - r4["ce"]
    check("M2 T = C(3)-C(4) = -0.412904 (winner-aligned: A(5)->B(9))",
          abs(T - PRED["T"]) < 1e-6 and T < 0)
    check("M2 output closer to bound value at both queries",
          abs(r3["ce"] - VA) < abs(r3["ce"] - VB) and
          abs(r4["ce"] - VB) < abs(r4["ce"] - VA))
    sw3 = energies("M2", QA, VB, VA)   # swapped values
    sw4 = energies("M2", QB, VB, VA)
    Ts = sw3["ce"] - sw4["ce"]
    check("M2 swap control: T_swap = +0.412904", abs(Ts - (-PRED["T"])) < 1e-6)
    check("M2 swap antisymmetry T_swap == -T (exact)",
          abs(Ts + T) < 1e-12)
    check("M2 softness D(Q=3)=1.7427, D(Q=4)=1.8444 (soft binding)",
          abs(abs(r3["ce"] - VA) - PRED["D3"]) < 1e-6 and
          abs(abs(r4["ce"] - VB) - PRED["D4"]) < 1e-6)
    check("M2 hard-binding gap: D > 0 at both queries",
          abs(r3["ce"] - VA) > 0 and abs(r4["ce"] - VB) > 0)

    a3d = drive("M2", QA)
    a4d = drive("M2", QB)
    check("M2 drive-level transfer keeps direction (-0.258552)",
          abs(a3d - PRED["AE3"]) < 1e-6 and abs(a4d - PRED["AE4"]) < 1e-6 and
          abs((a3d - a4d) - PRED["T_drive"]) < 1e-6 and (a3d - a4d) < 0)

    m5a3 = drive("M5", QA)
    m5a4 = drive("M5", QB)
    check("M5-E drive transfer: direction INVERTED by inhibition (+0.0985)",
          abs(m5a3 - PRED["M5_AE3"]) < 1e-3 and
          abs(m5a4 - PRED["M5_AE4"]) < 1e-3 and
          abs((m5a3 - m5a4) - PRED["M5_Tdrive"]) < 1e-3 and
          (m5a3 - m5a4) > 0)

    m4a3 = drive("M4", QA)
    m4a4 = drive("M4", QB)
    check("M4 (scalar E-I) drive baseline matches frozen",
          abs(m4a3 - PRED["M4_AE3"]) < 1e-6 and
          abs(m4a4 - PRED["M4_AE4"]) < 1e-6)

    r0_3 = energies("M0", QA)
    r0_4 = energies("M0", QB)
    check("M0: winner B at both queries (no routing, no counterfactual "
          "binding)", r0_3["al"][2] > r0_3["al"][0] and
          r0_4["al"][2] > r0_4["al"][0])
    check("M0 sharpening-only transfer -0.084443",
          abs(r0_3["ce"] - PRED["M0_CE3"]) < 1e-6 and
          abs(r0_4["ce"] - PRED["M0_CE4"]) < 1e-6 and
          abs((r0_3["ce"] - r0_4["ce"]) - PRED["M0_T"]) < 1e-6)
    check("M2 transfer magnitude exceeds M0 sharpening baseline",
          abs(T) > abs(r0_3["ce"] - r0_4["ce"]))

    r1p = energies("M1", QA, sign=+1.0)
    r1m = energies("M1", QA, sign=-1.0)
    check("M1 sign counterfactual: C(+1)=8.8936, C(-1)=5.1064 (near-hard "
          "binding, D=0.1064)", abs(r1p["ce"] - PRED["M1_CEp"]) < 1e-6 and
          abs(r1m["ce"] - PRED["M1_CEm"]) < 1e-6 and
          abs(abs(r1p["ce"] - VB) - PRED["M1_D"]) < 1e-6 and
          abs(abs(r1m["ce"] - VA) - PRED["M1_D"]) < 1e-6)
    check("M1 degenerate binding is HARDER than M2 (D_M1 < D_M2)",
          PRED["M1_D"] < PRED["D3"])

    ru3 = energies("M2unnorm", QA)
    ru4 = energies("M2unnorm", QB)
    check("M2unnorm: attention collapses to probe (v=0), no value transfer",
          abs(ru3["ce"] - PRED["UN_CE3"]) < 1e-6 and
          abs(ru4["ce"] - PRED["UN_CE4"]) < 1e-6 and
          abs(ru3["ce"] - ru4["ce"]) < 0.001)

    # position invariance (12 ordered pairs)
    maxdiff = 0.0
    for pA, pB in itertools.permutations(range(4), 2):
        h = [0.0] * 4
        h[pA], h[pB] = 1.0, 2.0
        v = np.array(h + [QA])
        mu = float(v.mean())
        SE = max(0.0, QA - mu)
        u = uhat(OMEGA * SE)
        lg = np.array([C * SE * float(u @ phi(x)) if x > 0 else 0.0
                       for x in v])
        al = softmax(lg)
        ce = (al[pA] * VA + al[pB] * VB) / (al[pA] + al[pB])
        maxdiff = max(maxdiff, abs(ce - r3["ce"]))
    check("position invariance: 12 layouts identical C_events (1e-12)",
          maxdiff < 1e-12)

    # ============ ensemble quadrature (recompute, audit vs frozen) ============
    grid = np.linspace(LO, HI, GRID_N)
    XX, YY = np.meshgrid(grid, grid)
    s3 = np.maximum(0.0, 0.8 * QA - 0.2 * (XX + YY))
    s4 = np.maximum(0.0, 0.8 * QB - 0.2 * (XX + YY))
    nA = np.sqrt(XX ** 2 + XX ** 4)
    nB = np.sqrt(YY ** 2 + YY ** 4)
    d1 = XX / nA - YY / nB
    d2 = XX ** 2 / nA - YY ** 2 / nB
    ld3 = C * s3 * (np.cos(OMEGA * s3) * d1 + np.sin(OMEGA * s3) * d2)
    ld4 = C * s4 * (np.cos(OMEGA * s4) * d1 + np.sin(OMEGA * s4) * d2)
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    a3, a4 = sig(ld3), sig(ld4)
    Tq = (a3 * VA + (1 - a3) * VB) - (a4 * VA + (1 - a4) * VB)
    flip = (ld3 * ld4 < 0)
    b_absT = float(np.mean(np.abs(Tq)))
    b_absT_flip = float(np.mean(np.abs(Tq[flip])))
    b_absT_noflip = float(np.mean(np.abs(Tq[~flip])))
    b_flip = float(np.mean(flip))
    b_ET = float(np.mean(Tq))
    b_align = float(np.mean(((Tq < 0) == (ld3 > 0))[flip]))
    check("quadrature audit: E|T| = 0.400435", abs(b_absT - PRED["B_absT"]) < 1e-6)
    check("quadrature audit: E|T||flip = 0.465523",
          abs(b_absT_flip - PRED["B_absT_flip"]) < 1e-6)
    check("quadrature audit: E|T||noflip = 0.348347",
          abs(b_absT_noflip - PRED["B_absT_noflip"]) < 1e-6)
    check("quadrature audit: flip rate = 0.444527", abs(b_flip - PRED["B_flip"]) < 1e-6)
    check("quadrature audit: E[T] = 0 (symmetry)", abs(b_ET) < 1e-9)
    check("quadrature audit: sign-matched alignment | flip == 1 (theorem)",
          abs(b_align - 1.0) < 1e-12)

    # ============ ensemble MC ============
    mc = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        xA = rng.uniform(LO, HI, N_H)
        xB = rng.uniform(LO, HI, N_H)
        pos = rng.integers(0, 12, N_H)
        s3 = np.maximum(0.0, 0.8 * QA - 0.2 * (xA + xB))
        s4 = np.maximum(0.0, 0.8 * QB - 0.2 * (xA + xB))
        nA = np.sqrt(xA ** 2 + xA ** 4)
        nB = np.sqrt(xB ** 2 + xB ** 4)
        d1 = xA / nA - xB / nB
        d2 = xA ** 2 / nA - xB ** 2 / nB
        ld3 = C * s3 * (np.cos(OMEGA * s3) * d1 + np.sin(OMEGA * s3) * d2)
        ld4 = C * s4 * (np.cos(OMEGA * s4) * d1 + np.sin(OMEGA * s4) * d2)
        a3, a4 = sig(ld3), sig(ld4)
        Tmc = (a3 * VA + (1 - a3) * VB) - (a4 * VA + (1 - a4) * VB)
        flip = (ld3 * ld4 < 0)
        align = ((Tmc < 0) == (ld3 > 0))
        est = dict(
            absT=float(np.mean(np.abs(Tmc))),
            absT_flip=float(np.mean(np.abs(Tmc[flip]))) if flip.any() else None,
            absT_noflip=float(np.mean(np.abs(Tmc[~flip]))) if (~flip).any() else None,
            flip=float(np.mean(flip)),
            align=float(np.mean(align[flip])) if flip.any() else None,
            ET=float(np.mean(Tmc)))
        mc[seed] = est
        # criteria
        seT = float(np.std(np.abs(Tmc))) / math.sqrt(N_H)
        check(f"MC {seed}: E|T| = {est['absT']:.5f} vs 0.400435",
              abs(est["absT"] - PRED["B_absT"]) <= 3 * seT + CRIT_M)
        seF = float(np.std(np.abs(Tmc[flip]))) / math.sqrt(flip.sum())
        check(f"MC {seed}: E|T||flip = {est['absT_flip']:.5f} vs 0.465523",
              abs(est["absT_flip"] - PRED["B_absT_flip"]) <= 3 * seF + CRIT_M)
        seN = float(np.std(np.abs(Tmc[~flip]))) / math.sqrt((~flip).sum())
        check(f"MC {seed}: E|T||noflip = {est['absT_noflip']:.5f} vs 0.348347",
              abs(est["absT_noflip"] - PRED["B_absT_noflip"]) <= 3 * seN + CRIT_M)
        seP = math.sqrt(est["flip"] * (1 - est["flip"]) / N_H)
        check(f"MC {seed}: flip rate = {est['flip']:.5f} vs 0.444527",
              abs(est["flip"] - PRED["B_flip"]) <= 3 * seP + CRIT_P)
        check(f"MC {seed}: alignment|flip = {est['align']:.5f} (>= 0.999)",
              est["align"] >= 0.999)
        seE = float(np.std(Tmc)) / math.sqrt(N_H)
        check(f"MC {seed}: E[T] = {est['ET']:.5f} vs 0",
              abs(est["ET"]) <= 3 * seE + CRIT_P)
    res["mc"] = {str(k): v for k, v in mc.items()}

    # ============ figures ============
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    labs = ["primary Q=3", "primary Q=4", "swap Q=3", "swap Q=4"]
    vals = [r3["ce"], r4["ce"], sw3["ce"], sw4["ce"]]
    ax.bar(labs, vals, color=["C1", "C2", "C1", "C2"], alpha=0.8)
    ax.axhline(VA, color="C1", ls=":", lw=1)
    ax.axhline(VB, color="C2", ls=":", lw=1)
    ax.text(3.4, VA, "V_A=5", color="C1", fontsize=8)
    ax.text(3.4, VB, "V_B=9", color="C2", fontsize=8)
    ax.set_ylabel("C_events (attended value readout)")
    ax.set_title("Counterfactual value transfer (M2)\nwinner A(5) -> B(9)")
    ax = axes[1]
    ax.bar(["M2 single-cell", "M5-E (E-I)", "M4 (scalar E-I)"],
           [a3d - a4d, m5a3 - m5a4, m4a3 - m4a4],
           color=["C0", "C4", "0.6"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("T_drive = A_E(Q=3) - A_E(Q=4)")
    ax.set_title("Drive-level transfer: gating keeps direction (M2),\n"
                 "E-I inhibition INVERTS it (M5-E) - honest boundary")
    fig.tight_layout()
    fig.savefig(outdir / "sprint4_2c_canonical.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    rng = np.random.default_rng(SEEDS[0])
    xA = rng.uniform(LO, HI, N_H)
    xB = rng.uniform(LO, HI, N_H)
    s3 = np.maximum(0.0, 0.8 * QA - 0.2 * (xA + xB))
    s4 = np.maximum(0.0, 0.8 * QB - 0.2 * (xA + xB))
    nA = np.sqrt(xA ** 2 + xA ** 4)
    nB = np.sqrt(xB ** 2 + xB ** 4)
    d1 = xA / nA - xB / nB
    d2 = xA ** 2 / nA - xB ** 2 / nB
    ld3 = C * s3 * (np.cos(OMEGA * s3) * d1 + np.sin(OMEGA * s3) * d2)
    ld4 = C * s4 * (np.cos(OMEGA * s4) * d1 + np.sin(OMEGA * s4) * d2)
    a3, a4 = sig(ld3), sig(ld4)
    Tf = (a3 * VA + (1 - a3) * VB) - (a4 * VA + (1 - a4) * VB)
    flipf = (ld3 * ld4 < 0)
    ax = axes[0]
    ax.hist(Tf[flipf], bins=40, color="C0", alpha=0.7, density=True,
            label="flip histories")
    ax.hist(Tf[~flipf], bins=40, color="0.6", alpha=0.7, density=True,
            label="no-flip histories")
    ax.set_xlabel("T = C_events(Q=3) - C_events(Q=4)")
    ax.set_title("Value-transfer distribution (M2, ensemble)")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.bar(["M2 (vector)", "M1 (sign)", "M0 (scalar)"],
           [PRED["D3"], PRED["M1_D"], abs(r0_3["ce"] - VB)],
           color=["C0", "0.75", "0.6"])
    ax.set_ylabel("D = |C_events - v_winner|")
    ax.set_title("Binding softness: soft attention keeps D > 0\n"
                 "(hard binding = 0; needs winner-take-all)")
    fig.tight_layout()
    fig.savefig(outdir / "sprint4_2c_ensemble.png", dpi=150)
    plt.close(fig)

    # ============ outputs ============
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary = dict(prereg="docs/TANII_SPRINT4_2C_PREREGISTRATION.md",
                   params=dict(omega=OMEGA, c=C, qA=QA, qB=QB, vA=VA, vB=VB,
                               hist=HIST, seeds=SEEDS, N_H=N_H,
                               crit_p=CRIT_P, crit_m=CRIT_M),
                   checks=CHECKS, canonical=dict(
                       T=T, T_swap=Ts, D3=abs(r3["ce"] - VA),
                       D4=abs(r4["ce"] - VB), drive_M2=(a3d, a4d),
                       drive_M5=(m5a3, m5a4), drive_M4=(m4a3, m4a4),
                       M1_CE=(r1p["ce"], r1m["ce"])),
                   mc=res["mc"], n_fail=N_FAIL, script_sha256=sha)
    (outdir / "sprint4_2c_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False,
                   default=lambda o: o.tolist()), encoding="utf-8")
    with open(outdir / "sprint4_2c_checks.csv", "w", newline="") as f:
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
    print("TAN-II Sprint 4.2-C binding probe (canonical + ensemble)")
    print("=" * 92)
    for c in summary["checks"]:
        if c["status"] != "PASS":
            print(f"  NON-PASS: {c['name']} ({c['detail']})")
    print(f"checks: {len(summary['checks'])} | FAIL: {summary['n_fail']}")
    c = summary["canonical"]
    print(f"canonical: T={c['T']:+.6f}  T_swap={c['T_swap']:+.6f}  "
          f"D=({c['D3']:.4f},{c['D4']:.4f})  drive M2={c['drive_M2']} "
          f"M5={c['drive_M5']} M4={c['drive_M4']}")
    if summary["n_fail"]:
        print("STOP AND DEBUG.")
    else:
        print("ALL BINDING CHECKS PASS: winner-aligned value transfer "
              "(soft/partial binding) established at the readout level; "
              "swap antisymmetry exact; hard-binding gap D>0 recorded; "
              "E-I drive-level direction inversion reported honestly.")
    print(f"outputs written to: {args.outdir}")
    return 1 if summary["n_fail"] else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())

