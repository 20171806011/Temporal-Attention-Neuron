#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sprint4_2_hard_binding.py
=========================
TAN-II Sprint 4.2-D -- Hard Binding minimal probe (winner-take-all).

The ONLY change vs the frozen 4.2-C model is the readout selection function
alpha_i(gamma) = softmax(gamma * logit_i), gamma in {1,2,5,10,100,inf};
gamma=inf is the one-hot (argmax) WTA. gamma=1 reproduces 4.2-C exactly.
Two WTA candidate sets: event-WTA (memory selection, primary) and
full-WTA (all 5 slots, control; probe/pad values are 0).

All predictions frozen in docs/TANII_SPRINT4_2D_PREREGISTRATION.md.
Deterministic canonical checks 1e-6 (antisymmetry 1e-12); ensemble MC vs
frozen quadrature (means 3*SE_mean+0.01; probabilities 3*SE+0.002).
Any FAIL -> exit 1.

Usage: python sprint4_2_hard_binding.py [--outdir DIR]
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

OMEGA, C, QA, QB = 0.4, 2.0, 3.0, 4.0
VA, VB = 5.0, 9.0
EPS_I = 0.3
W_EI = 0.8
HIST = [1.0, 0.0, 2.0, 0.0]
GAMMAS = [1.0, 2.0, 5.0, 10.0, 100.0]
SEEDS = [20260904, 20260905, 20260906]
N_H = 10000
CRIT_P, CRIT_M = 0.002, 0.01
LO, HI, GRID_N = 0.3, 2.5, 2001

# frozen predictions (pre-run, closed form)
FROZEN = dict(
    M2_Cfull={1: (4.890550, 4.943193), 2: (5.242303, 5.267514),
              5: (5.479412, 6.178173), 10: (5.241441, 7.463916),
              100: (5.000000, 8.999999)},
    M2_Dev={1: (1.742691, 1.844405), 2: (1.493762, 1.690682),
            5: (0.860900, 1.257667), 10: (0.279808, 0.695103)},
    M2_Dfull={1: (0.109450, 4.056807), 2: (0.242303, 3.732486),
              5: (0.479412, 2.821827), 10: (0.241441, 1.536084)},
    M2_Tev={1: -0.412904, 2: -0.815556, 5: -1.881433, 10: -3.025090,
            100: -3.999999},
    M2_Tfull={1: -0.052643, 2: -0.025211, 5: -0.698760, 10: -2.222475,
              100: -3.999999},
    M5_Tdrive={1: +0.236700, 2: +0.251584, 5: +0.015700, 10: -1.054093,
               100: -3.803430},
    M5_hard=(1.113437, 4.980861),
    gamma_star=5.098014585,
    E_absT_full={1: 0.586346, 2: 1.042547, 5: 1.815325, 10: 2.406862,
                 100: 2.990795},
    E_absT_full_inf=3.019684,
    flip_rate=0.444527,
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


def logits(model, Q, branch="E", norm=True, sign=1.0):
    v = np.array(HIST + [Q], dtype=float)
    mu = float(v.mean())
    S = max(0.0, Q - mu - (0.0 if branch == "E" else EPS_I))
    if model in ("M0", "M1", "M4"):
        return sign * C * S * v, S
    u = uhat(OMEGA * S)
    return (np.array([C * S * float(u @ phi(x, norm=norm)) if x > 0
                      else 0.0 for x in v]), S)


def alpha_gamma(lg, g):
    if g is None:
        w = int(np.argmax(lg))
        out = np.zeros(5)
        out[w] = 1.0
        return out
    e = np.exp(g * (lg - np.max(lg)))
    return e / e.sum()


def readouts(model, Q, g, vA=VA, vB=VB, norm=True, branch="E", sign=1.0):
    lg, S = logits(model, Q, branch, norm, sign)
    al = alpha_gamma(lg, g)
    cfull = al[0] * vA + al[2] * vB
    cev = (al[0] * vA + al[2] * vB) / (al[0] + al[2]) \
        if al[0] + al[2] > 0 else None
    return lg, S, al, cfull, cev


def run(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    res = {}

    # ============ M2 canonical gamma ladder ============
    ladder = {}
    prev_dev = prev_dfull = None
    for g in GAMMAS:
        r3 = readouts("M2", QA, g)
        r4 = readouts("M2", QB, g)
        w3 = "A" if r3[0][0] > r3[0][2] else "B"
        w4 = "A" if r4[0][0] > r4[0][2] else "B"
        d3 = abs(r3[4] - (VA if w3 == "A" else VB))
        d4 = abs(r4[4] - (VA if w4 == "A" else VB))
        df3 = abs(r3[3] - (VA if w3 == "A" else VB))
        df4 = abs(r4[3] - (VA if w4 == "A" else VB))
        tev = r3[4] - r4[4]
        tfull = r3[3] - r4[3]
        # swap
        s3 = readouts("M2", QA, g, VB, VA)
        s4 = readouts("M2", QB, g, VB, VA)
        tsw = s3[4] - s4[4]
        ladder[g] = dict(cfull=(r3[3], r4[3]), dev=(d3, d4),
                         dfull=(df3, df4), tev=tev, tfull=tfull,
                         winner=(w3, w4), tsw=tsw)
        check(f"M2 g={g}: winner (A,B) invariant", (w3, w4) == ("A", "B"))
        check(f"M2 g={g}: C_full matches frozen",
              abs(r3[3] - FROZEN["M2_Cfull"][g][0]) < 1e-6 and
              abs(r4[3] - FROZEN["M2_Cfull"][g][1]) < 1e-6)
        if g <= 10:
            check(f"M2 g={g}: D_events matches frozen",
                  abs(d3 - FROZEN["M2_Dev"][g][0]) < 1e-6 and
                  abs(d4 - FROZEN["M2_Dev"][g][1]) < 1e-6)
            check(f"M2 g={g}: D_full matches frozen",
                  abs(df3 - FROZEN["M2_Dfull"][g][0]) < 1e-6 and
                  abs(df4 - FROZEN["M2_Dfull"][g][1]) < 1e-6)
        check(f"M2 g={g}: T_events matches frozen",
              abs(tev - FROZEN["M2_Tev"][g]) < 1e-6)
        check(f"M2 g={g}: T_full matches frozen",
              abs(tfull - FROZEN["M2_Tfull"][g]) < 1e-6)
        check(f"M2 g={g}: swap antisymmetry exact (1e-12)",
              abs(tsw + tev) < 1e-12)
        if prev_dev is not None:
            check(f"M2 g={g}: D_events monotone non-increasing (AM5)",
                  d3 <= prev_dev[0] + 1e-12 and d4 <= prev_dev[1] + 1e-12)
            check(f"M2 g={g}: D_full(Q=4) monotone non-increasing (AM5)",
                  df4 <= prev_dfull[1] + 1e-12)
        prev_dev, prev_dfull = (d3, d4), (df3, df4)
    # AM5: D_full(Q=3) has the documented overshoot shape
    check("AM5: D_full(Q=3) overshoot shape (1.0 -> 2.0 -> 5.0 up, 10.0 down)",
          ladder[2.0]["dfull"][0] > ladder[1.0]["dfull"][0] + 1e-12 and
          ladder[5.0]["dfull"][0] > ladder[2.0]["dfull"][0] + 1e-12 and
          ladder[10.0]["dfull"][0] < ladder[5.0]["dfull"][0] - 1e-12)
    # AM5: cancellation-artifact zero of C_full(3, gamma)
    def cfull3(g):
        return readouts("M2", QA, g)[3]
    lo, hi = 1.0, 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cfull3(mid) < VA:
            lo = mid
        else:
            hi = mid
    g_cross3 = 0.5 * (lo + hi)
    check("AM5: C_full(Q=3) crosses V_A at gamma=1.197762904 "
          "(cancellation artifact, not hard binding)",
          abs(g_cross3 - 1.197762904) < 1e-6)
    res["gamma_cross3"] = g_cross3
    # gamma = inf (event-WTA + full-WTA)
    r3i = readouts("M2", QA, None)
    r4i = readouts("M2", QB, None)
    # event-WTA one-hot limit for C_events:
    lg3, S3 = logits("M2", QA)
    lg4, S4 = logits("M2", QB)
    w3i = int(np.argmax(np.array([lg3[0], lg3[2]])))
    w4i = int(np.argmax(np.array([lg4[0], lg4[2]])))
    cev3i = VA if w3i == 0 else VB
    cev4i = VA if w4i == 0 else VB
    check("M2 g=inf: C_full = (5, 9) exact",
          abs(r3i[3] - VA) < 1e-12 and abs(r4i[3] - VB) < 1e-12)
    check("M2 g=inf: C_events = (5, 9) exact (event-WTA)",
          abs(cev3i - VA) < 1e-12 and abs(cev4i - VB) < 1e-12)
    check("M2 g=inf: D = 0 (hard binding only in the discrete limit)",
          abs(r3i[3] - VA) == 0.0 and abs(r4i[3] - VB) == 0.0)
    check("M2 g=inf: T = -4 exact",
          abs((cev3i - cev4i) + 4.0) < 1e-12 and
          abs((r3i[3] - r4i[3]) + 4.0) < 1e-12)
    check("M2 g=100: D strictly positive (finite sharpening is soft)",
          ladder[100]["dev"][0] > 0 and ladder[100]["dfull"][0] > 0)
    res["M2_ladder"] = {str(g): ladder[g] for g in GAMMAS}
    res["M2_hard"] = dict(cfull=(r3i[3], r4i[3]), cev=(cev3i, cev4i))

    # ============ M5-E drive ladder + gamma* ============
    td_vals = {}
    for g in GAMMAS:
        a3, a4 = None, None
        for Q, idx in ((QA, 0), (QB, 1)):
            lgE, SE, alE, cfE, _ = readouts("M5", Q, g)
            lgI, SI = logits("M5", Q, "I")
            alI = alpha_gamma(lgI, g)
            cfI = alI[0] * VA + alI[2] * VB
            AE = math.tanh(SE) * cfE - W_EI * math.tanh(SI) * cfI
            if idx == 0:
                a3 = AE
            else:
                a4 = AE
        td = a3 - a4
        td_vals[g] = td
        check(f"M5-E g={g}: T_drive matches frozen",
              abs(td - FROZEN["M5_Tdrive"][g]) < 1e-6)
    # hard limits
    hard = []
    for Q in (QA, QB):
        lgE, SE, alE, cfE, _ = readouts("M5", Q, None)
        lgI, SI = logits("M5", Q, "I")
        alI = alpha_gamma(lgI, None)
        cfI = alI[0] * VA + alI[2] * VB
        hard.append(math.tanh(SE) * cfE - W_EI * math.tanh(SI) * cfI)
    check("M5-E g=inf: hard drives (1.113437, 4.980861), T_drive = -3.867424",
          abs(hard[0] - FROZEN["M5_hard"][0]) < 1e-6 and
          abs(hard[1] - FROZEN["M5_hard"][1]) < 1e-6 and
          abs((hard[0] - hard[1]) + 3.867424) < 1e-6)
    check("M5-E: T_drive positive at g=1 (4.2-C inversion) and negative at "
          "g=inf (direction restored)", td_vals[1.0] > 0 and
          (hard[0] - hard[1]) < 0)

    def td_of(g):
        vals = []
        for Q in (QA, QB):
            lgE, SE, alE, cfE, _ = readouts("M5", Q, g)
            lgI, SI = logits("M5", Q, "I")
            alI = alpha_gamma(lgI, g)
            cfI = alI[0] * VA + alI[2] * VB
            vals.append(math.tanh(SE) * cfE - W_EI * math.tanh(SI) * cfI)
        return vals[0] - vals[1]
    lo, hi = 1.0, 10.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if td_of(mid) > 0:
            lo = mid
        else:
            hi = mid
    gstar = 0.5 * (lo + hi)
    check("M5-E: gamma* crossing = 5.098014585 (bisection)",
          abs(gstar - FROZEN["gamma_star"]) < 1e-6)
    res["M5_ladder"] = {str(g): td_vals[g] for g in GAMMAS}
    res["M5_hard"] = hard
    res["gamma_star"] = gstar

    # ============ hard limits for controls ============
    # M0: event-WTA (9,9); full-WTA (0,0)
    for Q, exp_cev in ((QA, VB), (QB, VB)):
        lg, S = logits("M0", Q)
        w = int(np.argmax(np.array([lg[0], lg[2]])))
        cev = VA if w == 0 else VB
        al = alpha_gamma(lg, None)
        cf = al[0] * VA + al[2] * VB
        check(f"M0 g=inf Q={Q}: event-WTA = 9, full-WTA = 0 (probe wins)",
              abs(cev - exp_cev) < 1e-12 and abs(cf - 0.0) < 1e-12)
    # M1: event-WTA (9,5); full-WTA (0,0)
    lg_p, _ = logits("M1", QA, "E", True, +1.0)
    lg_m, _ = logits("M1", QA, "E", True, -1.0)
    wp = int(np.argmax(np.array([lg_p[0], lg_p[2]])))
    wm = int(np.argmax(np.array([lg_m[0], lg_m[2]])))
    al_p = alpha_gamma(lg_p, None)
    al_m = alpha_gamma(lg_m, None)
    check("M1 g=inf: event-WTA = (9, 5) exact; full-WTA = (0, 0)",
          abs((VA if wp == 0 else VB) - VB) < 1e-12 and
          abs((VA if wm == 0 else VB) - VA) < 1e-12 and
          abs(al_p[0] * VA + al_p[2] * VB) < 1e-12 and
          abs(al_m[0] * VA + al_m[2] * VB) < 1e-12)
    # M2unnorm: event-WTA (9,9); full-WTA (0,0)
    for Q in (QA, QB):
        lg, S = logits("M2", Q, "E", norm=False)
        w = int(np.argmax(np.array([lg[0], lg[2]])))
        cev = VA if w == 0 else VB
        al = alpha_gamma(lg, None)
        cf = al[0] * VA + al[2] * VB
        check(f"M2unnorm g=inf Q={Q}: event-WTA = 9, full-WTA = 0",
              abs(cev - VB) < 1e-12 and abs(cf - 0.0) < 1e-12)

    # ============ ensemble quadrature (recompute vs frozen) ============
    grid = np.linspace(LO, HI, GRID_N)
    XX, YY = np.meshgrid(grid, grid)
    s3 = np.maximum(0.0, 0.8 * QA - 0.2 * (XX + YY))
    s4 = np.maximum(0.0, 0.8 * QB - 0.2 * (XX + YY))
    nA = np.sqrt(XX ** 2 + XX ** 4)
    nB = np.sqrt(YY ** 2 + YY ** 4)
    pA1, pA2 = XX / nA, XX ** 2 / nA
    pB1, pB2 = YY / nB, YY ** 2 / nB
    lA3 = C * s3 * (np.cos(OMEGA * s3) * pA1 + np.sin(OMEGA * s3) * pA2)
    lB3 = C * s3 * (np.cos(OMEGA * s3) * pB1 + np.sin(OMEGA * s3) * pB2)
    lA4 = C * s4 * (np.cos(OMEGA * s4) * pA1 + np.sin(OMEGA * s4) * pA2)
    lB4 = C * s4 * (np.cos(OMEGA * s4) * pB1 + np.sin(OMEGA * s4) * pB2)
    probe3 = C * s3 * (np.cos(OMEGA * s3) * (QA / np.sqrt(QA ** 2 + QA ** 4))
                       + np.sin(OMEGA * s3) * (QA ** 2 / np.sqrt(QA ** 2 + QA ** 4)))
    probe4 = C * s4 * (np.cos(OMEGA * s4) * (QB / np.sqrt(QB ** 2 + QB ** 4))
                       + np.sin(OMEGA * s4) * (QB ** 2 / np.sqrt(QB ** 2 + QB ** 4)))
    flip = ((lA3 - lB3) * (lA4 - lB4) < 0)
    check("ensemble: flip rate invariant = 0.444527 (quadrature)",
          abs(float(np.mean(flip)) - FROZEN["flip_rate"]) < 1e-6)

    def full_alpha(g, lA, lB, lp):
        rows = np.stack([lA, np.zeros_like(lA), lB, np.zeros_like(lB), lp],
                        axis=-1)
        if g is None:
            w = np.argmax(rows, axis=-1)
            out = np.zeros_like(rows)
            out[np.arange(rows.shape[0])[:, None], np.arange(rows.shape[1])[None, :],
                w] = 1.0
            return out[..., 0], out[..., 2]
        m = rows.max(axis=-1, keepdims=True)
        e = np.exp(g * (rows - m))
        e /= e.sum(axis=-1, keepdims=True)
        return e[..., 0], e[..., 2]

    for g in GAMMAS:
        aA3, aB3 = full_alpha(g, lA3, lB3, probe3)
        aA4, aB4 = full_alpha(g, lA4, lB4, probe4)
        Tf = (aA3 * VA + aB3 * VB) - (aA4 * VA + aB4 * VB)
        check(f"ensemble quadrature g={g}: E|T_full| = {FROZEN['E_absT_full'][g]:.6f}",
              abs(float(np.mean(np.abs(Tf))) - FROZEN["E_absT_full"][g]) < 1e-6)
    # hard limit
    def hard_value(lA, lB, lp):
        rows = np.stack([lA, np.zeros_like(lA), lB, np.zeros_like(lB), lp],
                        axis=-1)
        w = np.argmax(rows, axis=-1)
        return np.where(w == 0, VA, np.where(w == 2, VB, 0.0))
    v3 = hard_value(lA3, lB3, probe3)
    v4 = hard_value(lA4, lB4, probe4)
    check("ensemble quadrature g=inf: E|T_full| = 3.019684",
          abs(float(np.mean(np.abs(v3 - v4))) - FROZEN["E_absT_full_inf"]) < 1e-6)

    # ============ ensemble MC at gamma=100 and inf ============
    mc = {}
    for g, bench in ((100.0, FROZEN["E_absT_full"][100]),
                     (None, FROZEN["E_absT_full_inf"])):
        key = "inf" if g is None else str(int(g))
        ests = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            xA = rng.uniform(LO, HI, N_H)
            xB = rng.uniform(LO, HI, N_H)
            s3 = np.maximum(0.0, 0.8 * QA - 0.2 * (xA + xB))
            s4 = np.maximum(0.0, 0.8 * QB - 0.2 * (xA + xB))
            nA = np.sqrt(xA ** 2 + xA ** 4)
            nB = np.sqrt(xB ** 2 + xB ** 4)
            lA3 = C * s3 * (np.cos(OMEGA * s3) * xA / nA
                            + np.sin(OMEGA * s3) * xA ** 2 / nA)
            lB3 = C * s3 * (np.cos(OMEGA * s3) * xB / nB
                            + np.sin(OMEGA * s3) * xB ** 2 / nB)
            lA4 = C * s4 * (np.cos(OMEGA * s4) * xA / nA
                            + np.sin(OMEGA * s4) * xA ** 2 / nA)
            lB4 = C * s4 * (np.cos(OMEGA * s4) * xB / nB
                            + np.sin(OMEGA * s4) * xB ** 2 / nB)
            probe3 = C * s3 * (np.cos(OMEGA * s3) * QA / np.sqrt(QA ** 2 + QA ** 4)
                               + np.sin(OMEGA * s3) * QA ** 2 / np.sqrt(QA ** 2 + QA ** 4))
            probe4 = C * s4 * (np.cos(OMEGA * s4) * QB / np.sqrt(QB ** 2 + QB ** 4)
                               + np.sin(OMEGA * s4) * QB ** 2 / np.sqrt(QB ** 2 + QB ** 4))
            rows3 = np.stack([lA3, np.zeros(N_H), lB3, np.zeros(N_H), probe3], axis=-1)
            rows4 = np.stack([lA4, np.zeros(N_H), lB4, np.zeros(N_H), probe4], axis=-1)
            if g is None:
                w3 = np.argmax(rows3, axis=-1)
                w4 = np.argmax(rows4, axis=-1)
                vv3 = np.where(w3 == 0, VA, np.where(w3 == 2, VB, 0.0))
                vv4 = np.where(w4 == 0, VA, np.where(w4 == 2, VB, 0.0))
            else:
                m3 = rows3.max(axis=-1, keepdims=True)
                e3 = np.exp(g * (rows3 - m3))
                e3 /= e3.sum(axis=-1, keepdims=True)
                m4 = rows4.max(axis=-1, keepdims=True)
                e4 = np.exp(g * (rows4 - m4))
                e4 /= e4.sum(axis=-1, keepdims=True)
                vv3 = e3[..., 0] * VA + e3[..., 2] * VB
                vv4 = e4[..., 0] * VA + e4[..., 2] * VB
            est = float(np.mean(np.abs(vv3 - vv4)))
            ests.append(est)
            se = float(np.std(np.abs(vv3 - vv4))) / math.sqrt(N_H)
            check(f"MC g={key} seed {seed}: E|T_full| = {est:.5f} vs {bench:.6f}",
                  abs(est - bench) <= 3 * se + CRIT_M)
            if g == 100.0:
                fl = ((lA3 - lB3) * (lA4 - lB4) < 0)
                pf = float(np.mean(fl))
                sef = math.sqrt(pf * (1 - pf) / N_H)
                check(f"MC g=100 seed {seed}: flip rate = {pf:.5f} vs 0.444527",
                      abs(pf - FROZEN["flip_rate"]) <= 3 * sef + CRIT_P)
        mc[key] = ests
    res["mc"] = mc

    # ============ figures ============
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    gx = [1, 2, 5, 10, 100, 400]
    ax = axes[0, 0]
    dev3 = [ladder[g]["dev"][0] for g in GAMMAS] + [0.0]
    dev4 = [ladder[g]["dev"][1] for g in GAMMAS] + [0.0]
    ax.plot(gx, dev3, "o-", color="C1", label="D_events(Q=3)")
    ax.plot(gx, dev4, "s-", color="C2", label="D_events(Q=4)")
    ax.set_xscale("log")
    ax.set_xlabel("gamma (sharpening gain)")
    ax.set_ylabel("D_events")
    ax.set_title("Softness -> 0 only in the discrete limit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax = axes[0, 1]
    td = [td_vals[g] for g in GAMMAS] + [hard[0] - hard[1]]
    ax.plot(gx, td, "o-", color="C4")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(FROZEN["gamma_star"], color="r", ls="--", lw=1)
    ax.text(FROZEN["gamma_star"] * 1.05, 0.4, f"gamma* = {FROZEN['gamma_star']:.2f}",
            color="r", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("gamma")
    ax.set_ylabel("T_drive (M5-E)")
    ax.set_title("E-I value-transfer direction: inverted -> restored")
    ax.grid(alpha=0.25)
    ax = axes[1, 0]
    for i, Q in enumerate((QA, QB)):
        cfs = [ladder[g]["cfull"][i] for g in GAMMAS] + \
              [res["M2_hard"]["cfull"][i]]
        ax.plot(gx, cfs, "o-", label=f"C_full(Q={Q})")
    ax.axhline(VA, color="C1", ls=":", lw=1)
    ax.axhline(VB, color="C2", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("gamma")
    ax.set_ylabel("C_full")
    ax.set_title("Readout converges to the bound values (5 / 9)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax = axes[1, 1]
    q = [FROZEN["E_absT_full"][g] for g in GAMMAS] + [FROZEN["E_absT_full_inf"]]
    ax.plot(gx, q, "o-", color="0.3")
    for key, ests in mc.items():
        xv = 100 if key == "100" else 400
        ax.errorbar(xv, float(np.mean(ests)), yerr=float(np.std(ests)),
                    fmt="r^", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("gamma")
    ax.set_ylabel("E[|T_full|]")
    ax.set_title("Ensemble value-transfer magnitude grows with sharpening")
    ax.grid(alpha=0.25)
    fig.suptitle("Sprint 4.2-D: winner-take-all as the minimal exact-selection "
                 "mechanism")
    fig.tight_layout()
    fig.savefig(outdir / "sprint4_2d_ladder.png", dpi=150)
    plt.close(fig)

    # ============ outputs ============
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary = dict(prereg="docs/TANII_SPRINT4_2D_PREREGISTRATION.md",
                   params=dict(omega=OMEGA, c=C, qA=QA, qB=QB, vA=VA, vB=VB,
                               gammas=GAMMAS, seeds=SEEDS, N_H=N_H),
                   checks=CHECKS, ladder=res, n_fail=N_FAIL,
                   script_sha256=sha)
    (outdir / "sprint4_2d_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False,
                   default=lambda o: o.tolist()), encoding="utf-8")
    with open(outdir / "sprint4_2d_checks.csv", "w", newline="") as f:
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
    print("TAN-II Sprint 4.2-D hard-binding (WTA) probe")
    print("=" * 92)
    for c in summary["checks"]:
        if c["status"] != "PASS":
            print(f"  NON-PASS: {c['name']} ({c['detail']})")
    print(f"checks: {len(summary['checks'])} | FAIL: {summary['n_fail']}")
    print(f"gamma* (M5-E direction restoration) = {summary['ladder']['gamma_star']:.6f}")
    if summary["n_fail"]:
        print("STOP AND DEBUG.")
    else:
        print("ALL HARD-BINDING CHECKS PASS: exact value delivery is achieved "
              "by the one-hot WTA limit (gamma=inf, D=0, T=-4) and only there "
              "(finite sharpening stays soft); WTA preserves routing and "
              "restores the E-I value-transfer direction (gamma* = 5.098).")
    print(f"outputs written to: {args.outdir}")
    return 1 if summary["n_fail"] else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
