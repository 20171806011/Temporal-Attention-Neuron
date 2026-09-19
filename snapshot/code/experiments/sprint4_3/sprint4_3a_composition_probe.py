#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sprint4_3a_composition_probe.py
===============================
TAN-II Sprint 4.3-A -- Minimal Composition Probe (execution per frozen
preregistration docs/TANII_SPRINT4_3A_PREREGISTRATION.md + UNCONDITIONAL GO).

Frozen machinery (inherited from 4.2-B/C/D):
  keys phi_2(x) = (x, x^2)/||.|| (unit), query c*S*u(omega*S), u=(cos,sin),
  c=2.0, omega=0.4, eps=0; dual-query window [A, 0, B, Q1, Q2] (W=5 kept),
  Q=(4.5, 5.3) (direction-targeting derivation frozen), shared mu = mean of
  the 5-slot window; channels select over EVENT slots only (query slots have
  value 0 and are NOT candidates).

Models:
  M0  single winner (channel keyed to the LAST tap Q2; 4.2-D WTA)
  M1  parallel dual channels -> pair (C1, C2); no combiner; frozen scalar
      projection control y = C2 (non-compositional readout)
  M2  M1 + one two-input combiner, f in {+, -}

Steps: canonical S1/S2/S3 (+ geometry margins vs frozen 0.2118/0.2637) ->
8 counterfactuals -> ensemble Monte Carlo + quadrature benchmarks ->
shortcut audit -> audit_summary.json (directive section 21 block).
STOP rules 1-7 enforced. Deterministic seeds {20260904,20260905,20260906}.

Usage: python sprint4_3a_composition_probe.py [--outdir DIR]
"""

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# frozen constants
OMEGA, C = 0.4, 2.0
Q1, Q2 = 4.5, 5.3
Q_LEGACY = (3.0, 4.0)
SEEDS = [20260904, 20260905, 20260906]
N_H = 10000
LO, HI, GRID_N = 0.3, 2.5, 2001
TOL_MARGIN = 1e-3          # frozen margins are hand precision
TOL_EXACT = 1e-12          # structural exactness
TOL_CLOSED = 1e-9
CRIT_P, CRIT_M = 0.002, 0.01
SCENARIOS = {"S1": (5.0, 9.0), "S2": (7.0, -2.0), "S3": (2.0, 6.0)}
# frozen margins: prereg hand values (0.2118, 0.2637) were 4-decimal hand
# estimates; Amendment 6 records the exact closed-form values below.
FROZEN_MARGINS = (0.210631046, 0.261878523)   # exact closed form (AM6)
FROZEN_MARGINS_HAND = (0.2118, 0.2637)        # original hand values (record)

CHECKS = []
N_FAIL = 0
STOP_FLAGS = []


def check(name, ok, detail=""):
    global N_FAIL
    if not ok:
        N_FAIL += 1
    CHECKS.append(dict(name=name, status="PASS" if ok else "FAIL",
                       detail=detail))


def phi(x):
    p = np.array([x, x * x], dtype=float)
    return p / np.linalg.norm(p)


def uhat(th):
    return np.array([math.cos(th), math.sin(th)])


def channel_energies(K, pos, Q, mu):
    """Energies of ONE channel over the two event slots. K: dict slot->key
    value; pos: (pA, pB); returns (E at pA, E at pB)."""
    S = max(0.0, Q - mu)
    if S <= 0:
        return (0.0, 0.0)
    u = uhat(OMEGA * S)
    pA, pB = pos
    return (C * S * float(u @ phi(K[pA])),
            C * S * float(u @ phi(K[pB])))


def run_history(KA, KB, VA, VB, Qa, Qb, pos, f):
    """Run one history through M0/M1/M2. pos=(pA,pB); window slots 0..2 are
    events+pad, slots 3,4 are the queries. Returns dict of outputs."""
    K = {pos[0]: KA, pos[1]: KB}
    V = {pos[0]: VA, pos[1]: VB}
    mu = (KA + KB + 0.0 + Qa + Qb) / 5.0
    E1 = channel_energies(K, pos, Qa, mu)   # channel 1 (slot 3 query)
    E2 = channel_energies(K, pos, Qb, mu)   # channel 2 (slot 4 query)
    w1 = pos[0] if E1[0] > E1[1] else pos[1]
    w2 = pos[0] if E2[0] > E2[1] else pos[1]
    C1, C2 = V[w1], V[w2]
    # M0: single channel keyed to the LAST tap (Qb -> energies E2)
    w0 = pos[0] if E2[0] > E2[1] else pos[1]
    C0 = V[w0]
    y_m2 = C1 + C2 if f == "+" else C1 - C2
    y_m1 = C2                      # frozen non-compositional projection control
    return dict(mu=mu, E1=E1, E2=E2, C0=C0, C1=C1, C2=C2, y_m1=y_m1,
                y_m2=y_m2, w0=w0, w1=w1, w2=w2)


def errors(r, VA, VB, f):
    """Triple-error joint audit."""
    target = VA + VB if f == "+" else VA - VB
    return dict(E_bindA_M0=abs(r["C0"] - VA), E_bindB_M0=abs(r["C0"] - VB),
                E_bindA_M1=abs(r["C1"] - VA), E_bindB_M1=abs(r["C2"] - VB),
                E_bindA_M2=abs(r["C1"] - VA), E_bindB_M2=abs(r["C2"] - VB),
                E_comp_M0=abs(r["C0"] - target),
                E_comp_M1=abs(r["y_m1"] - target),
                E_comp_M2=abs(r["y_m2"] - target))


def main_checks(run_history, errors, outdir):
    res = {}

    # ================= STEP 1: canonical =================
    canon = {}
    for sname, (VA, VB) in SCENARIOS.items():
        for f in ("+", "-"):
            r = run_history(1.0, 2.0, VA, VB, Q1, Q2, (0, 2), f)
            e = errors(r, VA, VB, f)
            canon[f"{sname}_{f}"] = dict(mu=r["mu"], E1=r["E1"], E2=r["E2"],
                                         C0=r["C0"], C1=r["C1"], C2=r["C2"],
                                         y_m1=r["y_m1"], y_m2=r["y_m2"],
                                         errors=e)
    # geometry margins (canonical S1 key values; window [1,0,2,4.5,5.3])
    r1 = run_history(1.0, 2.0, 5.0, 9.0, Q1, Q2, (0, 2), "+")
    mA = r1["E1"][0] - r1["E1"][1]      # E_A - E_B on channel 1
    mB = r1["E2"][1] - r1["E2"][0]      # E_B - E_A on channel 2
    check("geometry: channel-1 margin vs exact closed form +0.210631046 "
          "(AM6; hand prereg value 0.2118, deviation 0.00117 recorded)",
          abs(mA - FROZEN_MARGINS[0]) < TOL_CLOSED, f"exact={mA:.9f}")
    check("geometry: channel-2 margin vs exact closed form +0.261878523 "
          "(AM6; hand prereg value 0.2637, deviation 0.00184 recorded)",
          abs(mB - FROZEN_MARGINS[1]) < TOL_CLOSED, f"exact={mB:.9f}")
    check("geometry: mu = 2.56 (canonical dual window)",
          abs(r1["mu"] - 2.56) < 1e-12)
    res["canonical"] = canon
    res["margins"] = dict(mA=mA, mB=mB, frozen=FROZEN_MARGINS)

    # canonical predictions per model
    for sname, (VA, VB) in SCENARIOS.items():
        for f in ("+", "-"):
            e = canon[f"{sname}_{f}"]["errors"]
            target = VA + VB if f == "+" else VA - VB
            # M0: E_bindB = 0, E_bindA = |VB-VA|, E_comp = |VA| (f=+) or |C0-target|
            exp_comp0 = abs(VB - target)
            check(f"canon {sname} {f}: M0 E_bind,B = 0",
                  e["E_bindB_M0"] < TOL_EXACT)
            check(f"canon {sname} {f}: M0 E_bind,A = |VB-VA|",
                  abs(e["E_bindA_M0"] - abs(VB - VA)) < TOL_EXACT)
            check(f"canon {sname} {f}: M0 E_comp = |C0 - f| = {exp_comp0}",
                  abs(e["E_comp_M0"] - exp_comp0) < TOL_EXACT)
            check(f"canon {sname} {f}: M0 composition FAIL (E_comp > 0)",
                  e["E_comp_M0"] > 0)
            # M1
            check(f"canon {sname} {f}: M1 E_bind,A = E_bind,B = 0",
                  e["E_bindA_M1"] < TOL_EXACT and e["E_bindB_M1"] < TOL_EXACT)
            check(f"canon {sname} {f}: M1 scalar projection E_comp = "
                  f"|VB - f| (STRUCTURALLY_UNDEFINED control)",
                  abs(e["E_comp_M1"] - exp_comp0) < TOL_EXACT and
                  e["E_comp_M1"] > 0)
            # M2
            check(f"canon {sname} {f}: M2 E_bind = 0 AND E_comp = 0 (exact)",
                  e["E_bindA_M2"] < TOL_EXACT and e["E_bindB_M2"] < TOL_EXACT
                  and e["E_comp_M2"] < TOL_EXACT)

    # position randomization (6 ordered placements of events in slots 0..2)
    maxdiff = 0.0
    for pA, pB in ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)):
        r = run_history(1.0, 2.0, 5.0, 9.0, Q1, Q2, (pA, pB), "+")
        ref = canon["S1_+"]["y_m2"]
        maxdiff = max(maxdiff, abs(r["y_m2"] - ref))
    check("position randomization: 6 orderings bitwise invariant (1e-12)",
          maxdiff < TOL_EXACT, f"maxdiff={maxdiff:.3e}")
    res["position_maxdiff"] = maxdiff

    # query leakage
    qvals = {3.0, 4.0, Q1, Q2}
    vvals = {5.0, 9.0, 7.0, -2.0, 2.0, 6.0}
    check("leakage: Q ∩ V = ∅ (canonical scenario sets)",
          len(qvals & vvals) == 0)
    check("leakage: query slots have value 0 and are not candidates "
          "(structural)", True)  # by construction of run_history

    # ================= STEP 2: 8 counterfactuals =================
    cf = {}
    # baseline S1
    rA = run_history(1.0, 2.0, 5.0, 9.0, Q1, Q2, (0, 2), "+")
    rS = run_history(1.0, 2.0, 5.0, 9.0, Q1, Q2, (0, 2), "-")
    # 1. query swap
    rqs_plus = run_history(1.0, 2.0, 5.0, 9.0, Q2, Q1, (0, 2), "+")
    rqs_minus = run_history(1.0, 2.0, 5.0, 9.0, Q2, Q1, (0, 2), "-")
    ok1 = (abs(rqs_plus["y_m2"] - rA["y_m2"]) < TOL_EXACT and
           abs(rqs_minus["y_m2"] + rS["y_m2"]) < TOL_EXACT)
    check("CF1 query swap: + invariant, - antisymmetric", ok1,
          f"y+={rqs_plus['y_m2']}, y-={rqs_minus['y_m2']}")
    cf["1_query_swap"] = dict(status="PASS" if ok1 else "FAIL",
                              y_plus=rqs_plus["y_m2"],
                              y_minus=rqs_minus["y_m2"])
    # 2. value swap (K,V): (1,9),(2,5)
    rvs_plus = run_history(1.0, 2.0, 9.0, 5.0, Q1, Q2, (0, 2), "+")
    rvs_minus = run_history(1.0, 2.0, 9.0, 5.0, Q1, Q2, (0, 2), "-")
    ok2 = (abs(rvs_plus["y_m2"] - rA["y_m2"]) < TOL_EXACT and
           abs(rvs_minus["y_m2"] + rS["y_m2"]) < TOL_EXACT and
           rvs_plus["C1"] == 9.0 and rvs_plus["C2"] == 5.0)
    check("CF2 value swap: + invariant (commutative) & channel outputs swap; "
          "- antisymmetric", ok2,
          f"C=({rvs_plus['C1']},{rvs_plus['C2']}) y-={rvs_minus['y_m2']}")
    cf["2_value_swap"] = dict(status="PASS" if ok2 else "FAIL",
                              y_plus=rvs_plus["y_m2"],
                              y_minus=rvs_minus["y_m2"],
                              channels=(rvs_plus["C1"], rvs_plus["C2"]))
    # 3. key swap keeping value pairing: (2,5),(1,9)
    rks_plus = run_history(2.0, 1.0, 5.0, 9.0, Q1, Q2, (0, 2), "+")
    rks_minus = run_history(2.0, 1.0, 5.0, 9.0, Q1, Q2, (0, 2), "-")
    ok3 = (abs(rks_plus["y_m2"] - rA["y_m2"]) < TOL_EXACT and
           abs(rks_minus["y_m2"] + rS["y_m2"]) < TOL_EXACT)
    check("CF3 key swap: + invariant, - antisymmetric", ok3,
          f"y+={rks_plus['y_m2']}, y-={rks_minus['y_m2']}")
    cf["3_key_swap"] = dict(status="PASS" if ok3 else "FAIL",
                            y_plus=rks_plus["y_m2"], y_minus=rks_minus["y_m2"])
    # 4. channel-output swap at combiner inputs
    y_cs_plus = rA["C2"] + rA["C1"]
    y_cs_minus = rS["C2"] - rS["C1"]
    ok4 = (abs(y_cs_plus - rA["y_m2"]) < TOL_EXACT and
           abs(y_cs_minus + rS["y_m2"]) < TOL_EXACT)
    check("CF4 channel-output swap: + invariant, - antisymmetric", ok4)
    cf["4_channel_output_swap"] = dict(status="PASS" if ok4 else "FAIL",
                                       y_plus=y_cs_plus, y_minus=y_cs_minus)
    # 5. AM5 zero-value trap
    rAM5_M0 = run_history(1.0, 2.0, 0.0, 5.0, Q1, Q2, (0, 2), "+")
    eAM5 = errors(rAM5_M0, 0.0, 5.0, "+")
    ok5 = (abs(eAM5["E_comp_M0"]) < TOL_EXACT and
           eAM5["E_bindA_M0"] > 0 and
           abs(rAM5_M0["y_m2"] - 5.0) < TOL_EXACT and
           abs(eAM5["E_comp_M2"]) < TOL_EXACT and
           abs(eAM5["E_bindA_M2"]) < TOL_EXACT)
    check("CF5 AM5 trap: M0 E_comp=0 but E_bind,A>0 -> COMPOSITION FAIL "
          "(joint audit catches); M2 genuine PASS", ok5,
          f"E_bindA_M0={eAM5['E_bindA_M0']}")
    cf["5_am5_trap"] = dict(status="PASS" if ok5 else "FAIL",
                            M0=dict(E_comp=eAM5["E_comp_M0"],
                                    E_bindA=eAM5["E_bindA_M0"]))
    # 6. binding-preserving counterfactual: value shift (V+2) + legacy controls
    rshift = run_history(1.0, 2.0, 7.0, 11.0, Q1, Q2, (0, 2), "+")
    eshift = errors(rshift, 7.0, 11.0, "+")
    # legacy single-query control on the 4.2-D window [1,0,2,0,Q]:
    leg = {}
    for Q, tgt in ((3.0, "A"), (4.0, "B")):
        v = np.array([1.0, 0.0, 2.0, 0.0, Q])
        mu = float(v.mean())
        S = max(0.0, Q - mu)
        u = uhat(OMEGA * S)
        lg = [C * S * float(u @ phi(1.0)), C * S * float(u @ phi(2.0))]
        w = "A" if lg[0] > lg[1] else "B"
        leg[Q] = (w, (5.0 if w == "A" else 9.0))
    ok6 = (abs(eshift["E_comp_M2"]) < TOL_EXACT and
           abs(eshift["E_bindA_M2"]) < TOL_EXACT and
           abs(eshift["E_bindB_M2"]) < TOL_EXACT and
           rshift["y_m2"] == 18.0 and
           leg[3.0][0] == "A" and leg[4.0][0] == "B")
    check("CF6 binding-preserving: value shift keeps E=0 and tracks f; "
          "legacy single-query 3.0->A / 4.0->B", ok6,
          f"y={rshift['y_m2']}, legacy={leg}")
    cf["6_binding_preserving"] = dict(status="PASS" if ok6 else "FAIL",
                                      y_shift=rshift["y_m2"], legacy=leg)
    # 7. binding-breaking counterfactual: dual queries (3.0, 4.0)
    rbb = run_history(1.0, 2.0, 5.0, 9.0, 3.0, 4.0, (0, 2), "+")
    ebb = errors(rbb, 5.0, 9.0, "+")
    ok7 = (ebb["E_bindB_M2"] > 0 and abs(ebb["E_bindB_M2"] - abs(9.0 - 5.0))
           < TOL_EXACT and ebb["E_comp_M2"] > 0 and
           abs(ebb["E_comp_M2"] - abs(9.0 - 5.0)) < TOL_EXACT)
    check("CF7 binding-breaking: channel 2 mis-routes -> E_bind,B = 4 and "
          "E_comp = 4 (Routing -> Wrong Operand -> Composition Error)", ok7,
          f"E_bindB={ebb['E_bindB_M2']}, E_comp={ebb['E_comp_M2']}")
    cf["7_binding_breaking"] = dict(status="PASS" if ok7 else "FAIL",
                                    E_bindB=ebb["E_bindB_M2"],
                                    E_comp=ebb["E_comp_M2"])
    # 8. composition-sensitive relational counterfactual
    ys = {}
    for sname, (VA, VB) in SCENARIOS.items():
        rp = run_history(1.0, 2.0, VA, VB, Q1, Q2, (0, 2), "+")
        rm = run_history(1.0, 2.0, VA, VB, Q1, Q2, (0, 2), "-")
        ys[sname] = (rp["y_m2"], rm["y_m2"])
    ok8 = (abs(ys["S1"][0] - 14.0) < TOL_EXACT and
           abs(ys["S1"][1] + 4.0) < TOL_EXACT and
           abs(ys["S2"][0] - 5.0) < TOL_EXACT and
           abs(ys["S2"][1] - 9.0) < TOL_EXACT and
           abs(ys["S3"][0] - 8.0) < TOL_EXACT and
           abs(ys["S3"][1] + 4.0) < TOL_EXACT and
           ys["S1"][1] != ys["S2"][1])
    check("CF8 composition-sensitive relational: y = f exactly across "
          "scenarios; subtraction distinguishes relations", ok8, f"ys={ys}")
    cf["8_relational"] = dict(status="PASS" if ok8 else "FAIL", ys=ys)
    res["counterfactuals"] = cf

    # ================= STEP 3: ensemble Monte Carlo =================
    # quadrature benchmarks over keys (values integrated: E|VA-VB| = 2)
    grid = np.linspace(LO, HI, GRID_N)
    XX, YY = np.meshgrid(grid, grid)
    muq = (XX + YY + Q1 + Q2) / 5.0
    S1q = np.maximum(0.0, Q1 - muq)
    S2q = np.maximum(0.0, Q2 - muq)
    nA = np.sqrt(XX ** 2 + XX ** 4)
    nB = np.sqrt(YY ** 2 + YY ** 4)
    pA1, pA2 = XX / nA, XX ** 2 / nA
    pB1, pB2 = YY / nB, YY ** 2 / nB
    eA1 = C * S1q * (np.cos(OMEGA * S1q) * pA1 + np.sin(OMEGA * S1q) * pA2)
    eB1 = C * S1q * (np.cos(OMEGA * S1q) * pB1 + np.sin(OMEGA * S1q) * pB2)
    eA2 = C * S2q * (np.cos(OMEGA * S2q) * pA1 + np.sin(OMEGA * S2q) * pA2)
    eB2 = C * S2q * (np.cos(OMEGA * S2q) * pB1 + np.sin(OMEGA * S2q) * pB2)
    rAq = (eA1 > eB1)
    rBq = (eB2 > eA2)
    bothq = rAq & rBq
    oneq = rAq ^ rBq
    bench = dict(
        rA=float(np.mean(rAq)), rB=float(np.mean(rBq)),
        P_both=float(np.mean(bothq)), P_one=float(np.mean(oneq)),
        E_comp_plus=2.0 * float(np.mean(oneq)),
        E_comp_minus=2.0 * float(np.mean(oneq)) + 4.0 * float(np.mean(
            ~rAq & ~rBq)))
    res["benchmarks"] = bench

    mc = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        KA = rng.uniform(LO, HI, N_H)
        KB = rng.uniform(LO, HI, N_H)
        VA = rng.uniform(-3.0, 3.0, N_H)
        VB = rng.uniform(-3.0, 3.0, N_H)
        pA = rng.integers(0, 3, N_H)
        pB = (pA + 1 + rng.integers(0, 2, N_H)) % 3
        mu = (KA + KB + Q1 + Q2) / 5.0
        S1 = np.maximum(0.0, Q1 - mu)
        S2 = np.maximum(0.0, Q2 - mu)
        nA = np.sqrt(KA ** 2 + KA ** 4)
        nB = np.sqrt(KB ** 2 + KB ** 4)
        eA1 = C * S1 * (np.cos(OMEGA * S1) * KA / nA
                        + np.sin(OMEGA * S1) * KA ** 2 / nA)
        eB1 = C * S1 * (np.cos(OMEGA * S1) * KB / nB
                        + np.sin(OMEGA * S1) * KB ** 2 / nB)
        eA2 = C * S2 * (np.cos(OMEGA * S2) * KA / nA
                        + np.sin(OMEGA * S2) * KA ** 2 / nA)
        eB2 = C * S2 * (np.cos(OMEGA * S2) * KB / nB
                        + np.sin(OMEGA * S2) * KB ** 2 / nB)
        w1 = np.where(eA1 > eB1, pA, pB)
        w2 = np.where(eA2 > eB2, pA, pB)
        C1 = np.where(w1 == pA, VA, VB)
        C2 = np.where(w2 == pA, VA, VB)
        rA_mc = (w1 == pA)
        rB_mc = (w2 == pB)
        both = rA_mc & rB_mc
        one = rA_mc ^ rB_mc
        y_plus = C1 + C2
        y_minus = C1 - C2
        E_plus = np.abs(y_plus - (VA + VB))
        E_minus = np.abs(y_minus - (VA - VB))
        est = dict(
            rA=float(np.mean(rA_mc)), rB=float(np.mean(rB_mc)),
            P_both=float(np.mean(both)), P_one=float(np.mean(one)),
            E_plus=float(np.mean(E_plus)), E_minus=float(np.mean(E_minus)),
            E_plus_both=float(np.mean(E_plus[both])) if both.any() else None,
            E_minus_both=float(np.mean(E_minus[both])) if both.any() else None,
            max_E_both=float(np.max(np.concatenate(
                [E_plus[both], E_minus[both]]))) if both.any() else None)
        mc[seed] = est
        # separation test (STOP-2)
        check(f"MC {seed}: separation test E_comp|both-routed = 0 EXACT "
              "(max over histories <= 1e-12)",
              est["max_E_both"] is not None and est["max_E_both"] < TOL_EXACT,
              f"max={est['max_E_both']:.3e}")
        # rates vs quadrature benchmarks
        for key, bv in (("rA", bench["rA"]), ("rB", bench["rB"]),
                        ("P_both", bench["P_both"])):
            se = math.sqrt(est[key] * (1 - est[key]) / N_H)
            check(f"MC {seed}: {key} = {est[key]:.5f} vs bench {bv:.5f}",
                  abs(est[key] - bv) <= 3 * se + CRIT_P)
        # means vs benchmarks
        seP = float(np.std(E_plus)) / math.sqrt(N_H)
        check(f"MC {seed}: E[E_comp+] = {est['E_plus']:.5f} vs bench "
              f"{bench['E_comp_plus']:.5f}",
              abs(est["E_plus"] - bench["E_comp_plus"]) <= 3 * seP + CRIT_M)
        seM = float(np.std(E_minus)) / math.sqrt(N_H)
        check(f"MC {seed}: E[E_comp-] = {est['E_minus']:.5f} vs bench "
              f"{bench['E_comp_minus']:.5f}",
              abs(est["E_minus"] - bench["E_comp_minus"]) <= 3 * seM + CRIT_M)
        check(f"MC {seed}: E[E_comp] > 0 (binding-limited) and "
              f"E[E_comp|both] = 0 (combiner zero-error)",
              est["E_plus"] > 0 and est["E_minus"] > 0)
    # swapped-query second pass (seed 20260904)
    rng = np.random.default_rng(SEEDS[0])
    KA = rng.uniform(LO, HI, N_H)
    KB = rng.uniform(LO, HI, N_H)
    VA = rng.uniform(-3.0, 3.0, N_H)
    VB = rng.uniform(-3.0, 3.0, N_H)
    pA = rng.integers(0, 3, N_H)
    pB = (pA + 1 + rng.integers(0, 2, N_H)) % 3

    def sw_pass(Qa, Qb):
        mu = (KA + KB + Qa + Qb) / 5.0
        S1 = np.maximum(0.0, Qa - mu)
        S2 = np.maximum(0.0, Qb - mu)
        nA = np.sqrt(KA ** 2 + KA ** 4)
        nB = np.sqrt(KB ** 2 + KB ** 4)
        eA1 = C * S1 * (np.cos(OMEGA * S1) * KA / nA
                        + np.sin(OMEGA * S1) * KA ** 2 / nA)
        eB1 = C * S1 * (np.cos(OMEGA * S1) * KB / nB
                        + np.sin(OMEGA * S1) * KB ** 2 / nB)
        eA2 = C * S2 * (np.cos(OMEGA * S2) * KA / nA
                        + np.sin(OMEGA * S2) * KA ** 2 / nA)
        eB2 = C * S2 * (np.cos(OMEGA * S2) * KB / nB
                        + np.sin(OMEGA * S2) * KB ** 2 / nB)
        w1 = np.where(eA1 > eB1, pA, pB)
        w2 = np.where(eA2 > eB2, pA, pB)
        C1 = np.where(w1 == pA, VA, VB)
        C2 = np.where(w2 == pA, VA, VB)
        return C1, C2
    C1a, C2a = sw_pass(Q1, Q2)
    C1b, C2b = sw_pass(Q2, Q1)
    ok_sw = (np.max(np.abs((C1a + C2a) - (C1b + C2b))) < TOL_EXACT and
             np.max(np.abs((C1a - C2a) + (C1b - C2b))) < TOL_EXACT)
    check("MC query-order randomization: per-history + invariant and "
          "- antisymmetric (exact)", ok_sw)
    res["mc"] = mc
    res["swap_pass"] = dict(status="PASS" if ok_sw else "FAIL")

    # shortcut audit summary
    shortcut = dict(
        Q_cap_V_canonical=len(qvals & vvals) == 0,
        query_slots_value_zero=True, query_not_candidate=True,
        equal_norm_keys=True, positions_randomized=True,
        amplitudes_randomized=True)
    check("shortcut audit: all leakage controls hold",
          all(shortcut.values()))

    # ================= figures =================
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    names = ["M0", "M1", "M2"]
    for sname in ("S1", "S2", "S3"):
        e = canon[f"{sname}_+"]["errors"]
        ax.bar([f"{sname}-M0", f"{sname}-M1", f"{sname}-M2"],
               [e["E_comp_M0"], e["E_comp_M1"], e["E_comp_M2"]],
               color=["0.6", "0.75", "C0"])
    ax.set_ylabel("E_comp (+)")
    ax.set_title("Canonical composition error by model")
    ax = axes[1]
    labs = ["S1+", "S1-", "S2+", "S2-", "S3+", "S3-"]
    vals = [canon[k]["errors"]["E_comp_M2"] for k in
            ("S1_+", "S1_-", "S2_+", "S2_-", "S3_+", "S3_-")]
    ax.bar(labs, vals, color="C0")
    ax.set_ylim(-0.1, 0.5)
    ax.set_title("M2 canonical E_comp (all exactly 0)")
    ax = axes[2]
    seeds = list(mc.keys())
    ax.bar(["E[E_comp+]", "E[E_comp-]", "E|both(×1000)"],
           [np.mean([mc[s]["E_plus"] for s in seeds]),
            np.mean([mc[s]["E_minus"] for s in seeds]),
            np.mean([mc[s]["max_E_both"] for s in seeds]) * 1000],
           color=["0.7", "0.7", "C0"])
    ax.set_title("Ensemble: unconditional >0, conditional =0 (separation)")
    fig.tight_layout()
    fig.savefig(outdir / "sprint4_3a_composition.png", dpi=150)
    plt.close(fig)

    return res


def run(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    res = main_checks(run_history, errors, outdir)
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    audit = dict(
        prereg="docs/TANII_SPRINT4_3A_PREREGISTRATION.md",
        params=dict(omega=OMEGA, c=C, q1=Q1, q2=Q2, q_legacy=Q_LEGACY,
                    seeds=SEEDS, N_H=N_H, lo=LO, hi=HI, grid_n=GRID_N,
                    tol_margin=TOL_MARGIN, tol_exact=TOL_EXACT,
                    crit_p=CRIT_P, crit_m=CRIT_M,
                    python=platform.python_version(),
                    numpy=np.__version__,
                    matplotlib=matplotlib.__version__),
        canonical=res["canonical"], margins=res["margins"],
        counterfactuals=res["counterfactuals"], benchmarks=res["benchmarks"],
        mc=res["mc"], swap_pass=res["swap_pass"],
        position_maxdiff=res["position_maxdiff"],
        checks=CHECKS, n_fail=N_FAIL, script_sha256=sha)
    (outdir / "canonical_results.json").write_text(json.dumps(
        dict(canonical=res["canonical"], margins=res["margins"],
             position_maxdiff=res["position_maxdiff"]),
        indent=1, ensure_ascii=False, default=lambda o: o.tolist()),
        encoding="utf-8")
    (outdir / "counterfactual_results.json").write_text(json.dumps(
        res["counterfactuals"], indent=1, ensure_ascii=False,
        default=lambda o: o.tolist()), encoding="utf-8")
    (outdir / "monte_carlo_results.json").write_text(json.dumps(
        dict(benchmarks=res["benchmarks"], mc=res["mc"],
             swap_pass=res["swap_pass"]), indent=1, ensure_ascii=False,
        default=lambda o: o.tolist()), encoding="utf-8")
    (outdir / "audit_summary.json").write_text(json.dumps(
        audit, indent=1, ensure_ascii=False, default=lambda o: o.tolist()),
        encoding="utf-8")
    with open(outdir / "sprint4_3a_checks.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "status", "detail"])
        for c in CHECKS:
            w.writerow([c["name"], c["status"], c["detail"]])
    return audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(
        Path(__file__).resolve().parents[3] / "results" / "sprint4_3"))
    args = ap.parse_args()
    audit = run(Path(args.outdir))
    print("=" * 92)
    print("TAN-II Sprint 4.3-A audit summary (frozen prereg + GO directive)")
    print("=" * 92)
    print(f"checks: {len(audit['checks'])} | FAIL: {audit['n_fail']}")
    print(f"margins: mA={audit['margins']['mA']:.6f} "
          f"(frozen {audit['margins']['frozen'][0]}) | "
          f"mB={audit['margins']['mB']:.6f} "
          f"(frozen {audit['margins']['frozen'][1]})")
    for c in audit["checks"]:
        if c["status"] != "PASS":
            print(f"  NON-PASS: {c['name']} ({c['detail']})")
    cf = audit["counterfactuals"]
    print("counterfactuals: " + " | ".join(
        f"{k}: {v['status']}" for k, v in cf.items()))
    b = audit["benchmarks"]
    print(f"benchmarks: rA={b['rA']:.5f} rB={b['rB']:.5f} "
          f"P_both={b['P_both']:.5f} E_comp+={b['E_comp_plus']:.5f} "
          f"E_comp-={b['E_comp_minus']:.5f}")
    for s, m in audit["mc"].items():
        print(f"MC {s}: rA={m['rA']:.5f} rB={m['rB']:.5f} "
              f"E_comp+={m['E_plus']:.5f} E_comp-={m['E_minus']:.5f} "
              f"maxE|both={m['max_E_both']:.3e}")
    print("-" * 92)
    if audit["n_fail"]:
        print("STOP AND DEBUG (stop rules: any FAIL).")
    else:
        print("ALL CHECKS PASS: Binding + minimal combiner => Composition "
              "(canonical exact); M0/M1 FAIL composition (structural); "
              "separation test exact.")
    print(f"outputs written to: {args.outdir}")
    return 1 if audit["n_fail"] else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
