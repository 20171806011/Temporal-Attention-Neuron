#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sprint4_1cd_ei_dynamics.py
==========================
TAN-II Sprint 4.1-C/D -- E-I competition and the emergence of selective
computation: causal ablation ladder C1 -> C2 -> C3 -> C4.

Self-contained, deterministic (no RNG), numpy/scipy/matplotlib only.
Scientific settings are PRE-REGISTERED in
docs/TANII_SPRINT4_1CD_THEORY.md (written before this program ran) and are
reproduced verbatim below. This program must not be tuned after results.

Models (shared discrete-time E-I TAN dynamics of the pre-registration):
  C1  static E-I LIF   : drives [w_E x - th_E]_+, [w_I x - th_I]_+
                         (no surprise, no window attention)
  C2  E-I TAN noAttn   : drives tanh(S_E)*mu, tanh(S_I)*mu (surprise gating,
                         forced alpha = 1/W)
  C3  asymmetric       : E branch softmax attention, I branch uniform (mu)
  C4  full classical   : E and I both with own surprise and own attention

Common dynamics (spec section 9, verbatim):
  h_E(t) = lambda_E h_E(t-1) + A_E(t) - w_EI A_I(t)
  h_I(t) = lambda_I h_I(t-1) + w_IE A_E(t) - w_II A_I(t)
  spike: y_a = Theta(h_a - theta_sp); on spike h_a <- 0  (discrete reset;
  the system is NOT a smooth continuous dynamical system).
Dale: all coupling weights >= 0; sign structure lives in the equations only.

Conditional fixed-context analysis (spec section 13): freeze window X* and
probe x* -> A_E, A_I constant -> B_E, B_I constant -> conditional fixed points
h_E* = B_E/(1-lambda_E), h_I* = B_I/(1-lambda_I); conditional nullclines are
the constant lines h_E = h_E* and h_I = h_I*. This is a CONDITIONAL affine
flow for a frozen context, NOT an autonomous 2-D phase portrait.

Primary criteria:
  P1  monotonicity breaking of h_E*(x*) over x* in [0.2, 3.0], grid step
      0.05; finite-difference sign flip + curvature; pre-registered
      tolerances tol_d = 1e-3, tol_c = 2e-3, tol_p = 5e-3; candidate excluded
      (INCONCLUSIVE) if any point within +-2 grid steps is in a spike region.
  P2  counterfactual argmax flip rate between queries Q_A = 3.0 and Q_B = 4.0
      on the same fixed histories (6 layouts); prediction FlipRate = 0
      (scalar-query invariance lemma). C1/C2 have no attention candidate
      space -> pre-registered N/A.

Secondary metrics: SI, FWHM, x_peak, post-peak suppression slope, max
suppression, conditional fixed points, conditional nullclines, attention
allocations alpha_A(x*), alpha_B(x*).

Outputs (default outdir <script_dir>/results/sprint4_1cd/):
  sprint4_1cd_summary.json    machine-readable audit
  sprint4_1cd_curves.csv      full curves
  sprint4_1cd_tuning.png      P1 tuning curves C1-C4
  sprint4_1cd_phaseplanes.png Conditional Fixed-Context Phase Planes
  sprint4_1cd_attention.png   attention allocations
  sprint4_1cd_metrics.png     secondary metrics summary
Usage: python sprint4_1cd_ei_dynamics.py [--outdir DIR]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# PRE-REGISTERED PARAMETERS (see docs/TANII_SPRINT4_1CD_THEORY.md)
# ---------------------------------------------------------------------------
W = 5                       # window length (TAN-I heritage)
LAMBDA_E = 0.5              # E leak (TAN-I heritage)
LAMBDA_I = 0.5              # I leak
W_EI = 0.8                  # I -> E inhibition weight (>= 0, Dale)
W_IE = 1.0                  # E -> I excitation weight (>= 0, Dale)
W_II = 0.2                  # I -> I self-inhibition (>= 0, Dale)

# C1 static input thresholds/weights
W_E = 0.5
W_I = 1.0
TH_E = 0.4                  # x_E^on = TH_E/W_E = 0.8
TH_I = 1.0                  # x_I^on = TH_I/W_I = 1.0
# checks: 0.8 < 1.0 < 3.0  and  W_EI*W_I = 0.8 > W_E = 0.5  (theorem 1)

EPS_E = 0.0                 # E surprise dead zone (TAN-I heritage eps = 0)
EPS_I = 0.3                 # I surprise dead zone > EPS_E
C_E_GAIN = 2.0              # kernel gain = beta*Wq*Wk product (TAN-I heritage)
C_I_GAIN = 2.0

TH_SP = 3.0                 # default spike threshold (sweep below)
TH_SP_SWEEP = [0.5, 1.0, 3.0, 6.0]   # 0.5 = TAN-I heritage

X_A = 1.0                   # history event A magnitude
X_B = 2.0                   # history event B magnitude
HIST_L1 = [X_A, 0.0, X_B, 0.0]       # layout L1: A at slot 0, B at slot 2
LAYOUTS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]  # (pA, pB)
Q_A, Q_B = 3.0, 4.0         # counterfactual query amplitudes (> max content)

X_MIN, X_MAX, DX = 0.2, 3.0, 0.05    # probe grid
N_COND = 400                # conditional recurrence steps
N_EP_MAX = 160              # max episodes (episodic protocol)
CONV_TOL = 1e-12            # episodic convergence tolerance
PHASE_PROBES = [0.6, 1.4, 2.6]       # selected probes for phase planes
ICS = [(0.0, 0.0), (2.5, 0.6), (0.2, 1.2)]  # representative ICs (pre-registered)

# P1 detection tolerances (pre-registered)
TOL_D = 1e-3                # min |first difference| to count as a segment
TOL_C = 2e-3                # curvature: second finite difference < -TOL_C
TOL_P = 5e-3                # prominence over neighbours +-2 grid steps
SPIKE_MARGIN_STEPS = 2      # exclusion neighbourhood for spike contamination

MODELS = ["C1", "C2", "C3", "C4"]

# Hand-derived pre-registered predictions for P1 (theory doc section 5)
P1_PREDICTIONS = {
    "C1": {"verdict": "PASS", "note": "theorem 1: corner maximum at x*=1.0"},
    "C2": {"verdict": "FAIL", "note": "climb-then-plateau; curvature below tol"},
    "C3": {"verdict": "FAIL", "note": "monotone increase until spiking region"},
    "C4": {"verdict": "FAIL", "note": "monotone increase, subthreshold"},
}


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------
def relu(x):
    return np.maximum(0.0, x)


def stable_softmax(logits):
    m = np.max(logits)
    e = np.exp(logits - m)
    return e / np.sum(e)


def drives(model, v, xcur, mu, sE, sI):
    """(A_E, A_I, alphaE, alphaI) for window vector v (len W, v[-1] == xcur)."""
    if model == "C1":
        aE = relu(W_E * xcur - TH_E)
        aI = relu(W_I * xcur - TH_I)
        return float(aE), float(aI), None, None
    if model == "C2":
        return np.tanh(sE) * mu, np.tanh(sI) * mu, None, None
    # C3 / C4: E branch softmax attention
    lE = C_E_GAIN * sE * v
    aE_ = stable_softmax(lE)
    cE = float(np.dot(aE_, v))
    aE = np.tanh(sE) * cE
    if model == "C3":  # I branch uniform
        aI = np.tanh(sI) * mu
        return aE, aI, aE_, None
    # C4: I branch own softmax attention
    lI = C_I_GAIN * sI * v
    aI_ = stable_softmax(lI)
    cI = float(np.dot(aI_, v))
    aI = np.tanh(sI) * cI
    return aE, aI, aE_, aI_


def conditional_step(model, hist, xp):
    """Freeze window X* = hist + [xp]; conditional drives, fixed points,
    surprise values and E-branch attention weights."""
    v = np.asarray(hist + [xp], dtype=float)
    mu = float(np.mean(v))
    sE = relu(xp - mu - EPS_E)
    sI = relu(xp - mu - EPS_I)
    aE, aI, aE_, aI_ = drives(model, v, xp, mu, sE, sI)
    bE = aE - W_EI * aI
    bI = W_IE * aE - W_II * aI
    return dict(mu=mu, sE=float(sE), sI=float(sI), aE=float(aE), aI=float(aI),
                bE=float(bE), bI=float(bI),
                fE=float(bE / (1.0 - LAMBDA_E)),
                fI=float(bI / (1.0 - LAMBDA_I)),
                alphaE=aE_ if aE_ is not None else None)


def episodic_probe(model, xp, theta_sp):
    """Free-running episodic protocol: repeat episode [1, 0, 2, 0, xp];
    zero-padded onset; read h at the probe tap of the converged episode.
    Returns (hE_read, hI_read, evE_last_episode, evI_last_episode, converged)."""
    ep = HIST_L1 + [xp]
    buf = np.zeros(W)
    hE = hI = 0.0
    prev_read = None
    conv = False
    evE = evI = 0
    for e in range(N_EP_MAX):
        evE = evI = 0
        for xv in ep:
            buf[:-1] = buf[1:]
            buf[-1] = xv
            mu = float(np.mean(buf))
            sE = relu(xv - mu - EPS_E)
            sI = relu(xv - mu - EPS_I)
            aE, aI, _, _ = drives(model, buf, xv, mu, sE, sI)
            hE = LAMBDA_E * hE + aE - W_EI * aI
            hI = LAMBDA_I * hI + W_IE * aE - W_II * aI
            if hE >= theta_sp:
                evE += 1
                hE = 0.0
            if hI >= theta_sp:
                evI += 1
                hI = 0.0
        read = (hE, hI)
        if prev_read is not None and abs(read[0] - prev_read[0]) < CONV_TOL \
                and abs(read[1] - prev_read[1]) < CONV_TOL:
            conv = True
            break
        prev_read = read
    return read[0], read[1], evE, evI, bool(conv)


def build_hist(pA, pB):
    h = [0.0] * 4
    h[pA] = X_A
    h[pB] = X_B
    return h


# ---------------------------------------------------------------------------
# P1 detection
# ---------------------------------------------------------------------------
def detect_local_max(xs, fs, spike_flag):
    """Pre-registered finite-difference detection of a strict local maximum.

    Candidate at grid index k requires: rise into k (d[k-1] > TOL_D), fall
    after k (d[k] < -TOL_D), negative curvature (c2[k] < -TOL_C) and
    prominence over +-2 neighbours > TOL_P.  If any point in [k-2, k+2] is in
    a spike region (spike_flag), the candidate is INCONCLUSIVE.
    Returns dict(valid=[...], inconclusive=[...], verdict)."""
    n = len(fs)
    d = np.diff(fs)
    c2 = np.full(n, np.nan)
    c2[1:-1] = fs[2:] - 2.0 * fs[1:-1] + fs[:-2]
    valid, inconcl = [], []
    for k in range(2, n - 2):
        if d[k - 1] > TOL_D and d[k] < -TOL_D and c2[k] < -TOL_C:
            prom = float(fs[k] - min(fs[k - 2], fs[k + 2]))
            if prom <= TOL_P:
                continue
            nb_spiked = bool(spike_flag[k - SPIKE_MARGIN_STEPS:
                                       k + SPIKE_MARGIN_STEPS + 1].any())
            cand = dict(x=float(xs[k]), h=float(fs[k]), prom=prom,
                        c2=float(c2[k]), delta_rise=float(d[k - 1]),
                        delta_fall=float(d[k]))
            if nb_spiked:
                cand["reason"] = "spike-adjacent (+-2 grid steps)"
                inconcl.append(cand)
            else:
                valid.append(cand)
    if valid:
        verdict = "PASS"
    elif inconcl:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FAIL"
    return dict(valid=valid, inconclusive=inconcl, verdict=verdict)


# ---------------------------------------------------------------------------
# P2
# ---------------------------------------------------------------------------
def p2_block(hist):
    """Flip-rate measurement for one layout. C1/C2: pre-registered N/A (no
    attention candidate space)."""
    out = {}
    for m in ("C1", "C2"):
        out[m] = dict(applicable=False,
                      note="no attention candidate space (pre-registered N/A)")
    for m in ("C3", "C4"):
        alphas, fps = {}, {}
        for q in (Q_A, Q_B):
            c = conditional_step(m, hist, q)
            alphas[q] = np.asarray(c["alphaE"])
            fps[q] = c["fE"]
        pA = hist.index(X_A)
        pB = hist.index(X_B)
        wins, ties = {}, {}
        for q in (Q_A, Q_B):
            ta, tb = float(alphas[q][pA]), float(alphas[q][pB])
            if abs(ta - tb) < 1e-15:
                wins[q], ties[q] = None, True
            else:
                wins[q] = "A" if ta > tb else "B"
                ties[q] = False
        flip_a = (wins[Q_A] != wins[Q_B]) and not (ties[Q_A] or ties[Q_B])
        wb = {q: int(np.argmax(alphas[q])) for q in (Q_A, Q_B)}
        flip_b = wb[Q_A] != wb[Q_B]
        out[m] = dict(
            applicable=True,
            alphaA={str(q): float(alphas[q][pA]) for q in (Q_A, Q_B)},
            alphaB={str(q): float(alphas[q][pB]) for q in (Q_A, Q_B)},
            winner_history={str(q): wins[q] for q in (Q_A, Q_B)},
            tie_history={str(q): ties[q] for q in (Q_A, Q_B)},
            winner_fullwindow={str(q): wb[q] for q in (Q_A, Q_B)},
            flip_rate_history=1.0 if flip_a else 0.0,
            flip_rate_fullwindow=1.0 if flip_b else 0.0,
            deltaR=float(fps[Q_B] - fps[Q_A]),
            fE_QA=float(fps[Q_A]), fE_QB=float(fps[Q_B]))
    return out


# ---------------------------------------------------------------------------
# Secondary metrics (on the conditional-fixed-point curve)
# ---------------------------------------------------------------------------
def secondary_metrics(xs, fs, theta_sp, k0):
    """SI over the full scan; FWHM / suppression slope / max suppression
    around the valid primary peak index k0 (None if no valid peak)."""
    fmax, fmin = float(fs.max()), float(fs.min())
    si = (fmax - fmin) / (fmax + fmin + 1e-9)
    out = dict(SI=si, n_grid=len(fs))
    if k0 is None:
        out.update(FWHM=None, x_peak=None, x_peak_refined=None,
                   suppr_slope=None, suppr_slope_r2=None, suppr_slope_n=None,
                   max_suppression=None)
        return out
    peak = float(fs[k0])
    xp = float(xs[k0])
    xp_r = xp
    if 0 < k0 < len(xs) - 1:
        den = fs[k0 - 1] - 2.0 * fs[k0] + fs[k0 + 1]
        if abs(den) > 1e-12:
            xp_r = xp + 0.5 * DX * (fs[k0 - 1] - fs[k0 + 1]) / den
    hmid = 0.5 * (peak + fmin)
    li = ri = None
    for j in range(1, k0 + 1):          # left crossing on the rising edge
        if fs[j - 1] < hmid <= fs[j]:
            den = fs[j] - fs[j - 1]
            li = float(xs[j - 1] + (hmid - fs[j - 1]) * DX / max(den, 1e-15))
            break
    for j in range(k0 + 1, len(fs)):    # right crossing on the falling side
        if fs[j - 1] >= hmid > fs[j]:
            den = fs[j] - fs[j - 1]
            ri = float(xs[j - 1] + (hmid - fs[j - 1]) * DX / min(den, -1e-15))
            break
    fwhm = None if (li is None or ri is None or ri <= li) else float(ri - li)
    # suppression slope: up to 5 consecutive subthreshold steps after the peak
    pts = [j for j in range(k0 + 1, min(k0 + 6, len(fs))) if fs[j] < theta_sp]
    if len(pts) >= 2:
        A = np.polyfit(xs[pts], fs[pts], 1)
        yy = fs[pts]
        resid = yy - np.polyval(A, xs[pts])
        ss_tot = np.sum((yy - np.mean(yy)) ** 2)
        r2 = 1.0 - np.sum(resid ** 2) / max(ss_tot, 1e-30)
        out.update(suppr_slope=float(A[0]), suppr_slope_r2=float(r2),
                   suppr_slope_n=len(pts))
    else:
        out.update(suppr_slope=None, suppr_slope_r2=None,
                   suppr_slope_n=len(pts))
    tail = fs[k0 + 1:]
    mtail = peak if len(tail) == 0 else float(np.min(tail))
    out.update(FWHM=fwhm, x_peak=xp, x_peak_refined=xp_r,
               max_suppression=float(peak - mtail))
    return out


# ---------------------------------------------------------------------------
def run(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    xs = np.round(np.arange(X_MIN, X_MAX + 1e-9 + DX / 2, DX), 10)
    n = len(xs)

    summary = dict(
        prereg="docs/TANII_SPRINT4_1CD_THEORY.md",
        models=MODELS,
        params=dict(W=W, lambda_E=LAMBDA_E, lambda_I=LAMBDA_I, w_EI=W_EI,
                    w_IE=W_IE, w_II=W_II, w_E=W_E, w_I=W_I, th_E=TH_E,
                    th_I=TH_I, eps_E=EPS_E, eps_I=EPS_I, c_E=C_E_GAIN,
                    c_I=C_I_GAIN, th_sp=TH_SP, x_A=X_A, x_B=X_B,
                    hist_L1=HIST_L1, q_A=Q_A, q_B=Q_B,
                    grid=dict(x_min=X_MIN, x_max=X_MAX, dx=DX, n=n),
                    tolerances=dict(tol_d=TOL_D, tol_c=TOL_C, tol_p=TOL_P,
                                    spike_margin_steps=SPIKE_MARGIN_STEPS)))

    curves = {}
    fig_t, axes_t = plt.subplots(2, 2, figsize=(13, 9))
    for mi, m in enumerate(MODELS):
        fpE = np.zeros(n)
        fpI = np.zeros(n)
        epE = np.zeros(n)
        ep_evE = np.zeros(n, dtype=bool)   # E event in the final episode
        for i, xp in enumerate(xs):
            c = conditional_step(m, HIST_L1, float(xp))
            fpE[i], fpI[i] = c["fE"], c["fI"]
            epE[i], _, evE, _, ok = episodic_probe(m, float(xp), TH_SP)
            ep_evE[i] = evE > 0
        sp_zone = fpE >= TH_SP - 1e-12
        onset = float(xs[sp_zone][0]) if sp_zone.any() else None
        det = detect_local_max(xs, fpE, sp_zone)
        det_ep = detect_local_max(xs, epE, ep_evE)

        valid_peaks = det["valid"]
        k0 = None
        if valid_peaks:
            k0 = int(np.argmax([v["h"] for v in valid_peaks]))
            k0 = int(np.argmin(np.abs(xs - valid_peaks[k0]["x"])))
        sec = secondary_metrics(xs, fpE, TH_SP, k0)

        att = {}
        if m in ("C3", "C4"):
            pA, pB = 0, 2   # L1 slot indices of A and B
            rowsA, rowsB, rowsP = [], [], []
            for xp in xs:
                c = conditional_step(m, HIST_L1, float(xp))
                rowsA.append(float(c["alphaE"][pA]))
                rowsB.append(float(c["alphaE"][pB]))
                rowsP.append(float(c["alphaE"][4]))
            att["E_branch"] = dict(alpha_A=rowsA, alpha_B=rowsB,
                                   alpha_probe=rowsP)
            if m == "C4":
                iA, iB, iP = [], [], []
                for xp in xs:
                    c = conditional_step(m, HIST_L1, float(xp))
                    v = np.asarray(HIST_L1 + [xp], dtype=float)
                    sI = relu(xp - c["mu"] - EPS_I)
                    aI = stable_softmax(C_I_GAIN * sI * v)
                    iA.append(float(aI[0]))
                    iB.append(float(aI[2]))
                    iP.append(float(aI[4]))
                att["I_branch"] = dict(alpha_A=iA, alpha_B=iB,
                                       alpha_probe=iP)

        models_entry = dict(
            p1_fp=dict(valid=det["valid"], inconclusive=det["inconclusive"],
                       verdict=det["verdict"], spike_onset_x=onset,
                       spike_fraction=float(sp_zone.mean())),
            p1_episodic=dict(verdict=det_ep["verdict"],
                             valid=det_ep["valid"],
                             inconclusive=det_ep["inconclusive"]),
            metrics=sec, attention=att)
        summary.setdefault("model_results", {})[m] = models_entry
        curves[m] = dict(x=xs.tolist(), fp_E=fpE.tolist(), fp_I=fpI.tolist(),
                         ep_E=epE.tolist(),
                         ep_E_events=ep_evE.astype(int).tolist())

        # ---- tuning panel ----
        ax = axes_t[mi // 2, mi % 2]
        ax.plot(xs, fpE, "k-", lw=2, label="h_E* (conditional fp)")
        ax.plot(xs, epE, "C0--", lw=1.2, label="h_E (episodic read)")
        if onset is not None:
            ax.axvline(onset, color="red", ls=":", lw=1.4,
                       label=f"spike onset (fp>={TH_SP})")
            ax.axvspan(onset, X_MAX, color="red", alpha=0.05)
        ax.axhline(TH_SP, color="red", ls=":", lw=0.8)
        for v in det["valid"]:
            ax.plot(v["x"], v["h"], "r*", ms=15)
        for v in det["inconclusive"]:
            ax.plot(v["x"], v["h"], "gx", ms=9)
        ax.set_title(f"{m}  | P1(fp): {det['verdict']} | P1(episodic): "
                     f"{det_ep['verdict']} | predicted: "
                     f"{P1_PREDICTIONS[m]['verdict']}")
        ax.set_xlabel("probe x*")
        ax.set_ylabel("h_E")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)
    fig_t.suptitle("TAN-II S4.1-C/D  P1: E-unit response vs probe (fixed "
                   "context X* = [1,0,2,0,x*])", fontsize=12)
    fig_t.tight_layout(rect=(0, 0, 1, 0.97))

    # ---- robustness: P1 detection per layout (fp curves) ----
    robust = {}
    for pA, pB in LAYOUTS:
        hist = build_hist(pA, pB)
        per = {}
        for m in MODELS:
            fs = np.array([conditional_step(m, hist, float(xp))["fE"]
                           for xp in xs])
            sp = fs >= TH_SP - 1e-12
            per[m] = detect_local_max(xs, fs, sp)["verdict"]
        robust[f"L{pA}{pB}"] = per
    summary["robustness_layouts"] = robust

    # ---- P2 (per layout) ----
    p2 = {}
    for pA, pB in LAYOUTS:
        p2[f"L{pA}{pB}"] = p2_block(build_hist(pA, pB))
    summary["p2"] = p2

    # ---- spike-threshold sweep ----
    sweep = {}
    for th in TH_SP_SWEEP:
        row = {}
        for m in MODELS:
            fs = np.array([conditional_step(m, HIST_L1, float(xp))["fE"]
                           for xp in xs])
            z = fs >= th - 1e-12
            row[m] = dict(onset_x=float(xs[z][0]) if z.any() else None)
        sweep[str(th)] = row
    summary["spike_threshold_sweep"] = sweep

    # ---- prediction comparison ----
    pred = {}
    for m in MODELS:
        v = summary["model_results"][m]["p1_fp"]["verdict"]
        pv = P1_PREDICTIONS[m]["verdict"]
        pred[m] = dict(predicted=pv, observed=v, hit=(v == pv),
                       note=P1_PREDICTIONS[m]["note"])
    summary["prediction_comparison"] = pred

    # ---- phase-plane figure ----
    fig_p, axes_p = plt.subplots(2, 3, figsize=(17, 10))
    panels = [("C1", 1.4), ("C2", 1.4), ("C3", 1.4),
              ("C4", 0.6), ("C4", 1.4), ("C4", 2.6)]
    for k, (m, xp) in enumerate(panels):
        ax = axes_p[k // 3, k % 3]
        c = conditional_step(m, HIST_L1, xp)
        fE, fI = c["fE"], c["fI"]
        lim = min(max(3.5, abs(fE) * 1.3, abs(fI) * 1.3), 6.0)
        he = np.linspace(-0.5, lim, 15)
        hi = np.linspace(-0.5, lim, 15)
        HE, HI = np.meshgrid(he, hi)
        dE = (LAMBDA_E - 1.0) * (HE - fE)
        dI = (LAMBDA_I - 1.0) * (HI - fI)
        ax.quiver(HE, HI, dE, dI, color="0.65", alpha=0.45, width=0.0025)
        ax.axvline(fE, color="C0", lw=1.6,
                   label=f"cond. nullcline h_E = {fE:.3f}")
        ax.axhline(fI, color="C3", lw=1.6,
                   label=f"cond. nullcline h_I = {fI:.3f}")
        ax.plot([fE], [fI], "ko", ms=6, label="cond. fixed point")
        for x0, y0 in ICS:
            tx, ty = [x0], [y0]
            hx, hy = x0, y0
            for _ in range(60):
                hx = LAMBDA_E * hx + c["bE"]
                hy = LAMBDA_I * hy + c["bI"]
                tx.append(hx)
                ty.append(hy)
            ax.plot(tx, ty, lw=1.4)
        ax.axhline(TH_SP, color="red", ls=":", lw=1)
        ax.axvline(TH_SP, color="red", ls=":", lw=1)
        title = f"{m} @ x* = {xp}: fp = ({fE:.3f}, {fI:.3f})"
        if max(fE, fI) >= TH_SP:
            title += "  [spiking regime]"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("h_E")
        ax.set_ylabel("h_I")
        ax.set_xlim(-0.5, lim)
        ax.set_ylim(-0.5, lim)
        ax.grid(alpha=0.2)
    fig_p.suptitle("Conditional Fixed-Context Phase Plane (X* frozen = "
                   "[1,0,2,0,x*]; NOT a 2-D autonomous phase portrait)",
                   fontsize=12)
    fig_p.tight_layout(rect=(0, 0, 1, 0.96))

    # ---- attention figure ----
    fig_a, axes_a = plt.subplots(2, 2, figsize=(12, 9))
    for m, ax in (("C3", axes_a[0, 0]), ("C4", axes_a[0, 1])):
        a = summary["model_results"][m]["attention"]["E_branch"]
        ax.plot(xs, a["alpha_A"], "C1-", label=r"$\alpha_A$ (x_A=1.0)")
        ax.plot(xs, a["alpha_B"], "C2-", label=r"$\alpha_B$ (x_B=2.0)")
        ax.plot(xs, a["alpha_probe"], color="0.4", lw=1.6,
                label=r"$\alpha_{probe}$")
        ax.set_title(f"{m} E-branch attention allocation")
        ax.set_xlabel("probe x*")
        ax.set_ylabel(r"$\alpha$")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    a = summary["model_results"]["C4"]["attention"]["I_branch"]
    ax = axes_a[1, 0]
    ax.plot(xs, a["alpha_A"], "C1-", label=r"$\alpha_A$")
    ax.plot(xs, a["alpha_B"], "C2-", label=r"$\alpha_B$")
    ax.plot(xs, a["alpha_probe"], color="0.4", lw=1.6,
            label=r"$\alpha_{probe}$")
    ax.set_title("C4 I-branch attention allocation")
    ax.set_xlabel("probe x*")
    ax.set_ylabel(r"$\alpha$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax = axes_a[1, 1]
    for m in MODELS:
        ax.plot(xs, curves[m]["fp_E"], lw=2, label=f"{m} h_E*")
    ax.axhline(TH_SP, color="red", ls=":", lw=1)
    ax.set_title("Overlay: conditional fixed points")
    ax.set_xlabel("probe x*")
    ax.set_ylabel("h_E*")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig_a.suptitle("TAN-II S4.1-C/D attention allocations on fixed context",
                   fontsize=12)
    fig_a.tight_layout(rect=(0, 0, 1, 0.96))

    # ---- metrics figure ----
    fig_m, axes_m = plt.subplots(2, 2, figsize=(12, 8))
    si = [summary["model_results"][m]["metrics"]["SI"] for m in MODELS]
    xpk = [summary["model_results"][m]["metrics"]["x_peak"] for m in MODELS]
    fw = [summary["model_results"][m]["metrics"]["FWHM"] for m in MODELS]
    sl = [summary["model_results"][m]["metrics"]["suppr_slope"]
          for m in MODELS]
    col = ["0.35", "0.55", "0.7", "0.9"]

    def _bar(ax, vals, title, nd=2):
        plotv = [0.0 if v is None else v for v in vals]
        ax.bar(MODELS, plotv, color=col)
        for i, v in enumerate(vals):
            lab = "n/a" if v is None else f"{v:.{nd}f}"
            dy = 0.03 if (v is None or v >= 0) else -0.05
            ax.text(i, (v if v is not None else 0.0) + dy, lab,
                    ha="center", fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=0.2, axis="y")

    _bar(axes_m[0, 0], si, "Selectivity index SI (full scan)")
    _bar(axes_m[0, 1], xpk, "Peak position x_peak (valid max only)")
    _bar(axes_m[1, 0], fw, "FWHM (valid max only)")
    axes_m[1, 1].axhline(0, color="k", lw=0.8)
    _bar(axes_m[1, 1], sl, "Post-peak suppression slope (OLS, <=5 pts)")
    fig_m.suptitle("TAN-II S4.1-C/D secondary metrics", fontsize=12)
    fig_m.tight_layout(rect=(0, 0, 1, 0.96))

    # ---- write files ----
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary["script_sha256"] = sha
    (outdir / "sprint4_1cd_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    import csv as _csv
    with open(outdir / "sprint4_1cd_curves.csv", "w", newline="") as f:
        w = _csv.writer(f)
        hdr = ["x"]
        for m in MODELS:
            hdr += [f"{m}_fpE", f"{m}_fpI", f"{m}_epE", f"{m}_epEvE"]
        w.writerow(hdr)
        for i in range(n):
            row = [float(xs[i])]
            for m in MODELS:
                cu = curves[m]
                row += [cu["fp_E"][i], cu["fp_I"][i], cu["ep_E"][i],
                        cu["ep_E_events"][i]]
            w.writerow(row)

    fig_t.savefig(outdir / "sprint4_1cd_tuning.png", dpi=150)
    fig_p.savefig(outdir / "sprint4_1cd_phaseplanes.png", dpi=150)
    fig_a.savefig(outdir / "sprint4_1cd_attention.png", dpi=150)
    fig_m.savefig(outdir / "sprint4_1cd_metrics.png", dpi=150)
    plt.close("all")
    return summary


# ---------------------------------------------------------------------------
def _fnum(v, nd):
    return "-" if v is None else f"{v:.{nd}f}"


def console_report(summary):
    out = []
    out.append("=" * 98)
    out.append("TAN-II Sprint 4.1-C/D automated audit "
               "(pre-registration: docs/TANII_SPRINT4_1CD_THEORY.md)")
    out.append("=" * 98)
    g = summary["params"]["grid"]
    out.append(f"grid: x* in [{g['x_min']}, {g['x_max']}] step {g['dx']} "
               f"({g['n']} points); default spike threshold {TH_SP}; "
               f"script sha256 {summary['script_sha256'][:16]}...")
    out.append("-" * 98)
    hdr = (f"{'model':4} {'P1(fp)':11} {'P1(epis)':11} {'spike onset':11} "
           f"{'SI':8} {'x_peak':9} {'FWHM':9} {'suppr slope':12} "
           f"{'pred->obs'}")
    out.append(hdr)
    for m in MODELS:
        mo = summary["model_results"][m]
        p1 = mo["p1_fp"]["verdict"]
        p1e = mo["p1_episodic"]["verdict"]
        on = mo["p1_fp"]["spike_onset_x"]
        mt = mo["metrics"]
        pr = summary["prediction_comparison"][m]
        out.append(f"{m:4} {p1:11} {p1e:11} {_fnum(on, 2):11} "
                   f"{_fnum(mt['SI'], 3):8} {_fnum(mt['x_peak'], 2):9} "
                   f"{_fnum(mt['FWHM'], 2):9} "
                   f"{_fnum(mt['suppr_slope'], 3):12} "
                   f"{pr['predicted']}->{pr['observed']}"
                   f"{' HIT' if pr['hit'] else ' MISS'}")
    out.append("-" * 98)
    out.append("P1 detail on the fp curve:")
    for m in MODELS:
        mo = summary["model_results"][m]
        v = mo["p1_fp"]["valid"]
        ic = mo["p1_fp"]["inconclusive"]
        vstr = "; ".join(f"x={x['x']:.2f}, h={x['h']:.4f}, prom={x['prom']:.4f}"
                         for x in v) or "none"
        istr = "; ".join(f"x={x['x']:.2f} ({x['reason']})" for x in ic) \
            or "none"
        out.append(f"  {m}: verdict {mo['p1_fp']['verdict']}; valid: {vstr}")
        out.append(f"      inconclusive: {istr}; spike fraction "
                   f"{mo['p1_fp']['spike_fraction']:.3f}; episodic verdict "
                   f"{mo['p1_episodic']['verdict']}")
    out.append("-" * 98)
    out.append("P2 counterfactual argmax flip (Q_A=3.0 vs Q_B=4.0, 6 layouts):")
    fr_h, fr_w = [], []
    for k, res in summary["p2"].items():
        for m in ("C3", "C4"):
            r = res[m]
            fr_h.append(r["flip_rate_history"])
            fr_w.append(r["flip_rate_fullwindow"])
            out.append(f"  {k} {m}: flip(history)={r['flip_rate_history']:.0f} "
                       f"flip(full-window)={r['flip_rate_fullwindow']:.0f} "
                       f"winner@QA={r['winner_history']['3.0']} "
                       f"winner@QB={r['winner_history']['4.0']} "
                       f"dR={r['deltaR']:+.4f}")
    out.append(f"  aggregate FlipRate(history) = {np.mean(fr_h):.4f}; "
               f"FlipRate(full-window) = {np.mean(fr_w):.4f}")
    out.append("  (C1/C2: pre-registered N/A - no attention candidate space)")
    out.append("-" * 98)
    out.append("Spike-threshold sweep (fp >= threshold): onset x* per model")
    for th, row in summary["spike_threshold_sweep"].items():
        out.append("  th_sp=" + th + ": " + " ".join(
            f"{m}->{_fnum(v['onset_x'], 2)}" for m, v in row.items()))
    out.append("-" * 98)
    out.append("P1 robustness across 6 layouts (fp curves):")
    for k, v in summary["robustness_layouts"].items():
        out.append("  " + k + ": " + " ".join(f"{m}:{vv}" for m, vv in
                                              v.items()))
    out.append("=" * 98)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(
        Path(__file__).resolve().parent / "results" / "sprint4_1cd"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    summary = run(outdir)
    print(console_report(summary))
    print(f"outputs written to: {outdir}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())

