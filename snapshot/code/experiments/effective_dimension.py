"""
TAN Phase 2-Pilot - Event-locked effective response geometry
=============================================================
Frozen-protocol implementation (review-approved 2026-09-04).  See the
module header of the previous review for the full pre-registration;
this header summarises the binding choices implemented below.

Falsifiable question
    Does TAN exhibit event-locked expansion of effective response geometry
    beyond input-only controls?

Terminology (binding): d_PR = Tr(C)^2/Tr(C^2) is the "effective rank of the
(event-locked) response covariance".  It is never called intrinsic/manifold
dimension.  "Dimensionality breathing" is a working name only.

Models / frames (first round, frozen):
    C0 input-only   frame zU = [mu, S, x]
    B1 LIF          frame zC = [h, mu, S, A], A := x_t (null convention)
    B3 TAN-noAttn   frames zC + zF = [h, mu, S, A, alpha_1..5, C]
                    (alpha = 1/5 constant, C = mu by definition)
    B4 Full TAN     frames zC + zF (all computed)
Primary response: Dz_{e,tau} = z'(te+tau) - mean_{s=1..5} z'(te-s).
Secondary: raw z' (state geometry).  Velocity: v = z'(te+tau+1)-z'(te+tau)
(same-event adjacent-phase increment; distinct from Dz).
Amplitude controls (all three always reported): raw pooled,
amplitude-stratified (median split on log10 A_first, pooled-within
covariance; primary for conclusions), per-coordinate linear residualisation
on log10 A_first.
Quiet reference: post-burn-in samples >= 12 steps from any pulse, thinned
to mutual spacing >= 12.  NaN when n < 30 or Tr <= 1e-12 (never faked).
Recovery (isolated events only, next onset > te + 30):
  tau_1/2 on d_D(tau) [dDq + 0.5*(peak-dDq)]; tau_e-fold on logTr(tau)
  (linear fit, needs >= 0.3 nats drop).  95% CI by event bootstrap for the
  primary frame pair (B4 zF, B3 zF).

Estimator note (fix logged 2026-09-04, before the formal run): per-seed
global Z-scoring gives every coordinate an arbitrary per-seed LEVEL.
Pooling raw standardized samples across seeds would inject that level as
artificial between-seed variance.  Quiet blocks and the raw-state
secondary matrix are therefore centred per seed (per-column) before
cross-seed pooling.  Delta-z and velocity quantities are unaffected
(per-seed levels cancel by construction), so all decision variables are
identical with or without the fix; only quiet-state and raw-state
diagnostics change.
Decision gates (fixed thresholds):
  event value = mean over tau in {0,1,2}; stable expansion: DeltaD >= 0.5
  with CI > 0;  A: B4(zF) beyond B3(zF) AND B4(zC) event dD beyond B1(zC)
  and C0(zU) event dD (CI comparison; cross-frame floors use the event
  level because the C0 quiet covariance is identically zero -> NaN);
  B: not A, d_B4 ~= d_B3 (zF), logTr_B4 - logTr_B3 >= 0.2 nats;
  C: not A/B and B4(zC) within B1(zC) and C0(zU);
  D: B4 has no stable event-locked expansion.
Calibration: parity vs code/model/tan.py; LIF quiet dGq in [0.8, 2.0];
all PR in [0.8, dim+1e-9] or NaN; generator sanity (burst fraction > 0.30,
>= 30 episodes, >= 20 distinct quantised window patterns).  --relax-asserts
exists ONLY for plumbing smoke runs.

Outputs: results/figures/fig24_eventlocked.{png,pdf},
fig25_recovery.{png,pdf}, fig26_spectra.{png,pdf} (+ manuscript mirror),
results/tables/effective_dimension_{summary,curves}.csv,
data/experiment_results/effective_dimension_{events,geometry}.npz,
results/logs/effective_dimension.log.

Dependencies: numpy (+ matplotlib).  Deterministic; no result-driven
parameter choice anywhere in this file.
"""
import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Pre-registered constants (frozen)
# ------------------------------------------------------------
W = 5
LAMBDA = 0.5
THETA = 0.5
W_Q, W_K, W_V = 2.0, 1.0, 1.0
BETA = 1.0
EPS = 0.0

T_DEFAULT = 5000
BURN = 500
SEED_LIST = [20260904, 20260905, 20260906, 20260907, 20260908]

GAP_MIN = 6
GAP_P = 1.0 / 28.0
LEN_VALUES = [1, 2, 3, 4]
LEN_PROBS = [0.45, 0.30, 0.17, 0.08]
ISI_CHOICES = [1, 2, 3]
A_LO, A_HI = 0.3, 2.5

TAU_MIN, TAU_MAX = -6, 30
PRE_WIN = 5
EV_TAUS = [0, 1, 2]
QUIET_DIST = 12
MIN_N = 30
TR_FLOOR = 1e-12

MIN_EPISODES = 30
MIN_BURST_FRAC = 0.30
MIN_PATTERNS = 20
STABLE_EXPANSION = 0.5
MIN_LOGT_DIFF = 0.2
MIN_TR_RECOV = 0.3
SPIKE_TOL = 1e-6
PARITY_ATOL = 1e-6
N_BOOT_EV = 2000
N_BOOT_REC = 1000

_BINS = np.array([0.0, 1.0333, 1.7667, np.inf])

FRAME_DIM = {"zU": 3, "zC": 4, "zF": 10}
MODEL_NAME = {"lif": "B1", "noattn": "B3", "tan": "B4", "input": "C0"}


def quant(v):
    return np.digitize(np.asarray(v), _BINS, right=True) - 1


# ============================================================
# 1. Stimulus generator
# ============================================================
def gen_trial(rng, T):
    x = np.zeros(T)
    pulse = np.zeros(T, dtype=bool)
    onsets, first_amp, n_pulses = [], [], []
    t = 0
    while True:
        t += GAP_MIN + int(rng.geometric(GAP_P))
        if t >= T:
            break
        L = int(rng.choice(LEN_VALUES, p=LEN_PROBS))
        amps = rng.uniform(A_LO, A_HI, size=L)
        pos = t
        k = 0
        while k < L and pos < T:
            x[pos] = amps[k]
            pulse[pos] = True
            k += 1
            if k < L:
                pos += int(rng.integers(ISI_CHOICES[0], ISI_CHOICES[-1] + 1))
            else:
                pos += 1
        L = k
        if L >= 1:
            onsets.append(t)
            first_amp.append(float(amps[0]))
            n_pulses.append(L)
    return dict(x=x, pulse=pulse, onsets=np.asarray(onsets, dtype=np.int64),
                first_amp=np.asarray(first_amp),
                n_pulses=np.asarray(n_pulses, dtype=np.int64))


def generator_report(tr, T):
    x, pulse = tr["x"], tr["pulse"]
    ons = tr["onsets"]
    burst = tr["n_pulses"] >= 2
    stream = np.concatenate([np.zeros(W - 1), x])
    win = sliding_window_view(stream, W)
    pat = set()
    for te in ons:
        if BURN + PRE_WIN <= te <= T - 1 - TAU_MAX - 1:
            for tau in range(0, 5):
                pat.add(tuple(quant(win[te + tau])))
    return dict(n_episodes=int(len(ons)), n_pulses=int(pulse.sum()),
                burst_fraction=float(burst.mean()) if len(burst) else
                float("nan"),
                length_hist=[int((tr["n_pulses"] == k).sum())
                             for k in LEN_VALUES],
                n_patterns=len(pat))


def assert_generator(rep, log):
    ok = True
    if rep["n_episodes"] < MIN_EPISODES:
        log.error("  ASSERT FAIL: episodes %d < %d", rep["n_episodes"],
                  MIN_EPISODES)
        ok = False
    if rep["burst_fraction"] <= MIN_BURST_FRAC:
        log.error("  ASSERT FAIL: burst fraction %.3f <= %.2f",
                  rep["burst_fraction"], MIN_BURST_FRAC)
        ok = False
    if rep["n_patterns"] < MIN_PATTERNS:
        log.error("  ASSERT FAIL: pattern diversity %d < %d",
                  rep["n_patterns"], MIN_PATTERNS)
        ok = False
    return ok


# ============================================================
# 2. Simulators (equations identical to code/model/tan.py)
# ============================================================
def simulate_traces(x, kind):
    """Per-step traces of the TAN equations for one trial."""
    T = len(x)
    stream = np.concatenate([np.zeros(W - 1), x])
    win = sliding_window_view(stream, W)          # (T, W)
    h = np.zeros(T)
    u = np.zeros(T)
    mu = np.zeros(T)
    S = np.zeros(T)
    A = np.zeros(T)
    C = np.zeros(T)
    alpha = np.zeros((T, W)) if kind in ("noattn", "tan") else None
    h_prev = 0.0
    for t in range(T):
        w = win[t]
        xt = float(w[-1])
        m = float(w.mean())
        mu[t] = m
        s = max(0.0, xt - m - EPS)          # input-derived surprise for ALL
        S[t] = s                            # kinds (LIF frame needs it too)
        if kind == "lif":
            drv = xt
        else:
            if kind == "noattn":
                alpha[t, :] = 1.0 / W
                C[t] = m
                drv = np.tanh(s) * m
            else:
                e = np.exp(BETA * W_Q * W_K * s * w)
                a = e / (e.sum() + 1e-9)
                alpha[t, :] = a
                c = float((W_V * a * w).sum())
                C[t] = c
                drv = np.tanh(s) * c
        A[t] = drv
        ut = LAMBDA * h_prev + drv
        u[t] = ut
        h[t] = 0.0 if ut > THETA else ut
        h_prev = h[t]
    out = dict(h=h, u=u, mu=mu, S=S, A=A, C=C)
    if alpha is not None:
        out["alpha"] = alpha
    return out


def parity_check(log):
    """Compare with code/model/tan.py classes on seeded trials."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model import tan as _tan
    worst = {}
    for seed in (20260904, 20260905):
        rng = np.random.default_rng(seed)
        x = np.clip(rng.uniform(0.0, 2.5, size=400), 0.0, None)
        ref = {}
        n = _tan.LIFNeuron(lambda_leak=LAMBDA, threshold=THETA)
        hs = np.zeros(len(x))
        for t, xi in enumerate(x):
            n.forward(float(xi))
            hs[t] = n.h
        ref["lif"] = hs
        for key, var in (("tan", "TAN-full"), ("noattn", "TAN-no-attention")):
            n = _tan.make_variant(var)
            hs = np.zeros(len(x))
            for t, xi in enumerate(x):
                n.forward(float(xi))
                hs[t] = n.h
            ref[key] = hs
        for kind in ("lif", "noattn", "tan"):
            tr = simulate_traces(x, kind)
            dev = float(np.max(np.abs(tr["h"] - ref[kind])))
            near = np.abs(tr["u"] - THETA) <= SPIKE_TOL
            sp_ok = bool(np.all((tr["h"] == 0.0) == (ref[kind] == 0.0)
                                if not near.any() else
                                np.all(((tr["h"] == 0.0) ==
                                        (ref[kind] == 0.0))[~near])))
            w = worst.setdefault(kind, [0.0, True])
            w[0] = max(w[0], dev)
            w[1] = w[1] and sp_ok
    ok = True
    for kind, (dev, sp) in worst.items():
        log.info("  parity %-6s max|dh| = %.2e  spike agreement = %s",
                 kind, dev, sp)
        if dev > PARITY_ATOL or not sp:
            ok = False
    return ok


# ============================================================
# 3. Frames, standardisation, geometry
# ============================================================
def build_frames(kind, sim, x):
    """Return {frame_name: Z matrix (T, d)} for a model kind."""
    d = dict(h=sim["h"], mu=sim["mu"], S=sim["S"], A=sim["A"])
    frames = {"zC": np.column_stack([d["h"], d["mu"], d["S"], d["A"]])}
    if kind in ("noattn", "tan"):
        cols = [d["h"], d["mu"], d["S"], d["A"]]
        cols += [sim["alpha"][:, i] for i in range(W)]
        cols += [sim["C"]]
        frames["zF"] = np.column_stack(cols)
    return frames


def standardise(Z, burn):
    mu = Z[burn:].mean(axis=0)
    sd = Z[burn:].std(axis=0)
    const = sd < 1e-12
    Zs = np.zeros_like(Z)
    live = ~const
    if live.any():
        Zs[:, live] = (Z[:, live] - mu[live]) / sd[live]
    return Zs, int(const.sum())


def pr_logtr(X):
    """(d_PR, logTr, lam, n) of row covariance; NaN policy as frozen."""
    n = len(X)
    if n < MIN_N:
        return (float("nan"), float("nan"),
                np.full(X.shape[1], np.nan), n)
    Xc = X - X.mean(axis=0, keepdims=True)
    C = (Xc.T @ Xc) / n
    lam = np.maximum(np.linalg.eigvalsh(C), 0.0)
    tr = float(lam.sum())
    if not np.isfinite(tr) or tr <= TR_FLOOR:
        return (float("nan"), float("nan"), lam, n)
    return (float(tr * tr / float((lam ** 2).sum())), float(np.log(tr)),
            lam, n)


def pooled_within_cov(X, strata):
    s0, s1 = X[strata == 0], X[strata == 1]
    if len(s0) == 0 or len(s1) == 0:
        return np.cov(X.T)
    c0 = s0 - s0.mean(axis=0, keepdims=True)
    c1 = s1 - s1.mean(axis=0, keepdims=True)
    return (c0.T @ c0 + c1.T @ c1) / len(X)


def geometry_of(X, logA):
    """Full frozen geometry suite on a row matrix X (n, d) with log-amps."""
    n = len(X)
    d_pr, lt, lam, _ = pr_logtr(X)
    if np.isfinite(d_pr):
        med = float(np.median(logA))
        strata = (logA >= med).astype(int)
        Cw = pooled_within_cov(X, strata)
        d_st, lt_st, _ = metrics_from_C(Cw, n)
        Xd = np.column_stack([np.ones(n), logA])
        beta = np.linalg.lstsq(Xd, X, rcond=None)[0]
        res = X - Xd @ beta
        Cr = (res - res.mean(axis=0, keepdims=True))
        Cr = Cr.T @ Cr / n
        d_re, lt_re, _ = metrics_from_C(Cr, n)
    else:
        d_st = lt_st = d_re = lt_re = float("nan")
    return dict(d=d_pr, lt=lt, lam=lam, d_strat=d_st, lt_strat=lt_st,
                d_resid=d_re, lt_resid=lt_re, n=n)


def metrics_from_C(C, n):
    lam = np.maximum(np.linalg.eigvalsh(C), 0.0)
    tr = float(lam.sum())
    if n < MIN_N or not np.isfinite(tr) or tr <= TR_FLOOR:
        return float("nan"), float("nan"), lam
    return float(tr * tr / float((lam ** 2).sum())), float(np.log(tr)), lam


# ============================================================
# 4. Recovery & decision helpers
# ============================================================
def recovery_vals(dD_curve, lt_curve, dq):
    """tau_1/2 (velocity PR) and tau_e-fold (logTr), point estimates."""
    dD_curve = np.asarray(dD_curve, dtype=float)
    lt_curve = np.asarray(lt_curve, dtype=float)
    if not np.isfinite(dD_curve[:6]).any() or not np.isfinite(dq):
        return None, None
    peak = float(np.nanmax(dD_curve[:6]))
    if not np.isfinite(peak):
        return None, None
    thr = dq + 0.5 * (peak - dq)
    th = None
    for tau in range(1, len(dD_curve)):
        if dD_curve[tau] <= thr:
            th = float(tau)
            break
    lp = int(np.nanargmax(lt_curve[:6])) if np.isfinite(
        lt_curve[:6]).any() else 0
    seg = lt_curve[lp + 1:]
    ef = None
    if len(seg) >= 3 and np.isfinite(seg).sum() == len(seg):
        slope = np.polyfit(np.arange(len(seg)), seg, 1)[0]
        if slope < 0 and lt_curve[lp] - seg[-1] >= MIN_TR_RECOV:
            ef = -1.0 / slope
    return th, ef


def tau_grid():
    return np.arange(TAU_MIN, TAU_MAX + 1)


# ============================================================
# 5. Figure construction (frozen design, pure functions of data)
# ============================================================
def _style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.titlesize": 10.5, "axes.labelsize": 10,
                         "legend.fontsize": 7.6, "xtick.labelsize": 8.5,
                         "ytick.labelsize": 8.5, "axes.unicode_minus": False,
                         "mathtext.fontset": "dejavusans"})


def plot_curves(ax, tg, y, color, label, ls="-"):
    ax.plot(tg, y, ls, color=color, lw=1.8, label=label)


def make_fig24(payload, fig_dir):
    """Event-locked traces: S, A, d_D(tau), logTr(tau) for B4 zF + CI."""
    _style()
    fig, axs = plt.subplots(2, 2, figsize=(11.8, 8.6), constrained_layout=True)
    tg = payload["tau"]
    ax = axs[0, 0]
    ax.plot(tg, payload["S_mean"], "-", color="#C0392B", lw=1.8,
            label="$\\langle S_t \\rangle$")
    ax.fill_between(tg, payload["S_lo"], payload["S_hi"], color="#C0392B",
                    alpha=0.15)
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_title("(a) surprise $S_t$ and gate current $A_t$ (event-locked, "
                 "Full TAN)", fontweight="bold")
    ax.set_ylabel("value")
    ax2 = ax.twinx()
    ax2.plot(tg, payload["A_mean"], "--", color="#1F4E79", lw=1.8,
             label="$\\langle A_t \\rangle$")
    ax2.fill_between(tg, payload["A_lo"], payload["A_hi"], color="#1F4E79",
                     alpha=0.12)
    ax2.set_ylabel("$A_t$")
    ax.legend(loc="upper left", fontsize=7.6)
    ax2.legend(loc="upper right", fontsize=7.6)
    ax.axvline(0, color="0.5", lw=0.8, ls="--")
    ax.grid(alpha=0.25, ls="--")

    ax = axs[0, 1]
    plot_curves(ax, tg, payload["dD"], "#C0392B", "velocity PR $d_D(\\tau)$")
    ax.fill_between(tg, payload["dD_lo"], payload["dD_hi"], color="#C0392B",
                    alpha=0.18)
    ax.axhline(payload["dDq"], color="0.4", ls=":", lw=1.2,
               label="quiet $d_D$ (%.2f)" % payload["dDq"])
    plot_curves(ax, tg, payload["dG"], "#2E86AB",
                "response PR $d_G(\\tau)$")
    ax.set_title("(b) effective rank of event responses: "
                 "$\\Delta d_D$ = %.2f" % payload["delta_dD"],
                 fontweight="bold")
    ax.set_ylabel("$d_{PR}$")
    ax.legend(loc="upper right", fontsize=7.6)
    ax.axvline(0, color="0.5", lw=0.8, ls="--")
    ax.grid(alpha=0.25, ls="--")

    ax = axs[1, 0]
    plot_curves(ax, tg, payload["lt"], "#C0392B",
                "$\\log\\mathrm{Tr}\\,C(\\tau)$")
    ax.fill_between(tg, payload["lt_lo"], payload["lt_hi"], color="#C0392B",
                    alpha=0.18)
    ax.axhline(payload["ltq"], color="0.4", ls=":", lw=1.2,
               label="quiet $\\log$Tr (%.2f)" % payload["ltq"])
    ax.set_title("(c) fluctuation magnitude $\\log\\mathrm{Tr}\\,C$ "
                 "(scale, PR-blind)", fontweight="bold")
    ax.set_xlabel("lag $\\tau = t - t_e$")
    ax.set_ylabel("$\\log\\mathrm{Tr}$")
    ax.legend(loc="lower right", fontsize=7.6)
    ax.axvline(0, color="0.5", lw=0.8, ls="--")
    ax.grid(alpha=0.25, ls="--")

    ax = axs[1, 1]
    txt = ("event value = mean over $\\tau\\in\\{0,1,2\\}$\n"
           "$\\Delta d_D = %.2f$   [$%.2f,\\ %.2f$]\n"
           "$\\Delta\\log\\mathrm{Tr} = %.2f$ nats\n"
           "$\\tau_{1/2} = %s$,  $\\tau_{e\\text{-fold}} = %s$"
           % (payload["delta_dD"], payload["ci_lo"], payload["ci_hi"],
              payload["delta_lt"], payload["tau_half"],
              payload["tau_efold"]))
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="#FBF3E4", ec="0.7"))
    # eigenvalue decay at event peak vs quiet (compact inset of fig26 idea)
    lam_e = payload["lam_event"]
    lam_q = payload["lam_quiet"]
    r = np.arange(1, len(lam_e) + 1)
    ax.semilogy(r, np.maximum(lam_e, 1e-12), "o-", color="#C0392B",
                ms=3.5, label="event $\\tau$=0")
    ax.semilogy(r, np.maximum(lam_q, 1e-12), "s--", color="0.5", ms=3,
                label="quiet")
    ax.set_xlabel("eigenvalue rank $i$")
    ax.set_ylabel("$\\lambda_i$ (log)")
    ax.set_title("(d) event vs quiet eigenvalue spectrum (B4, zF)",
                 fontweight="bold")
    ax.legend(fontsize=7.6)
    ax.grid(alpha=0.3, which="both", ls=":")

    fig.suptitle("fig24 - Event-locked response geometry, Full TAN (zF)",
                 fontsize=12, fontweight="bold")
    fig.savefig(fig_dir / "fig24_eventlocked.png", dpi=300,
                facecolor="white", bbox_inches="tight")
    fig.savefig(fig_dir / "fig24_eventlocked.pdf", facecolor="white",
                bbox_inches="tight")
    plt.close(fig)


def make_fig25(payload, fig_dir):
    _style()
    fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.9), constrained_layout=True)
    colors = {"B1": "#7F8C8D", "B3": "#16A085", "B4": "#C0392B",
              "C0": "#8E44AD"}
    tg = payload["tau"]
    for pr in payload["profiles"]:
        name, frame = pr["name"], pr["frame"]
        ls = "--" if frame == "zF" else (":" if frame == "zU" else "-")
        axs[0].plot(tg, pr["d"], ls, color=colors[name], lw=1.8,
                    label=f"{name} [{frame}]")
        axs[1].plot(tg, pr["lt"], ls, color=colors[name], lw=1.8,
                    label=f"{name} [{frame}]")
    for ax in axs:
        ax.axhline(0, color="0.5", lw=0.9)
        ax.axvline(0, color="0.5", lw=0.9, ls="--")
        ax.grid(alpha=0.25, ls="--")
        ax.set_xlabel("lag $\\tau = t - t_e$")
    axs[0].set_ylabel("$\\Delta d_D(\\tau) = d_D(\\tau) - d_D^{quiet}$")
    axs[0].set_title("(a) event-locked effective-rank expansion-recovery "
                     "(velocity)", fontweight="bold")
    axs[1].set_ylabel("$\\Delta\\log\\mathrm{Tr}\\,C(\\tau)$")
    axs[1].set_title("(b) fluctuation-scale expansion-recovery",
                     fontweight="bold")
    axs[0].legend(fontsize=7.0, ncol=3, loc="upper right")
    axs[1].legend(fontsize=7.0, ncol=3, loc="upper right")
    fig.suptitle("fig25 - Model comparison: LIF / TAN-noAttn / Full TAN / "
                 "input floor (solid zC, dashed zF, dotted zU)",
                 fontsize=12, fontweight="bold")
    fig.savefig(fig_dir / "fig25_recovery.png", dpi=300, facecolor="white",
                bbox_inches="tight")
    fig.savefig(fig_dir / "fig25_recovery.pdf", facecolor="white",
                bbox_inches="tight")
    plt.close(fig)


def make_fig26(payload, fig_dir):
    _style()
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 6.2), constrained_layout=True)
    for key, color, mk in (("B4", "#C0392B", "o"), ("B3", "#16A085", "s")):
        lam_e = payload[key]["lam_event"]
        lam_q = payload[key]["lam_quiet"]
        r = np.arange(1, len(lam_e) + 1)
        ax.semilogy(r, np.maximum(lam_e, 1e-12), f"{mk}-", color=color,
                    ms=4, lw=1.6, label=f"{key} event $\\tau$=0 (d={payload[key]['d_e']:.2f})")
        ax.semilogy(r, np.maximum(lam_q, 1e-12), f"{mk}--", color=color,
                    ms=3, alpha=0.7,
                    label=f"{key} quiet (d={payload[key]['d_q']:.2f})")
    ax.set_xlabel("eigenvalue rank $i$ of the response covariance (zF)")
    ax.set_ylabel("$\\lambda_i$ (log scale)")
    ax.set_title("fig26 - Quiet vs event spectra: new directions vs "
                 "amplification of existing ones", fontweight="bold")
    ax.grid(alpha=0.3, which="both", ls=":")
    ax.legend(fontsize=8.4)
    fig.savefig(fig_dir / "fig26_spectra.png", dpi=300, facecolor="white",
                bbox_inches="tight")
    fig.savefig(fig_dir / "fig26_spectra.pdf", facecolor="white",
                bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 6. Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--trial-length", type=int, default=T_DEFAULT)
    ap.add_argument("--burn-in", type=int, default=BURN)
    ap.add_argument("--out-root", type=str, default=None)
    ap.add_argument("--relax-asserts", action="store_true",
                    help="plumbing smoke only; NEVER for the full run")
    args = ap.parse_args()
    if args.seeds not in (3, 5):
        raise SystemExit("--seeds must be 3 or 5 (fixed seed order)")
    seeds = SEED_LIST[:args.seeds]

    root = Path(args.out_root) if args.out_root else Path(
        __file__).resolve().parents[2]
    fig_dir = root / "results" / "figures"
    tab_dir = root / "results" / "tables"
    log_dir = root / "results" / "logs"
    dat_dir = root / "data" / "experiment_results"
    ms_dir = root / "manuscript" / "figures"
    for d in (fig_dir, tab_dir, log_dir, dat_dir, ms_dir):
        d.mkdir(parents=True, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_dir / "effective_dimension.log",
                                      mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("phase2")
    relax = args.relax_asserts
    t0 = time.time()
    log.info("=" * 78)
    log.info("TAN Phase 2-Pilot - event-locked effective response geometry")
    log.info("=" * 78)
    log.info("Frozen protocol. T=%d burn=%d seeds=%s | tau [%d,%d] | "
             "eps=%.2f | W=%d lam=%.2f theta=%.2f Wq=%.2f Wk=Wv=%.1f "
             "beta=%.1f", args.trial_length, args.burn_in, seeds, TAU_MIN,
             TAU_MAX, EPS, W, LAMBDA, THETA, W_Q, W_K, BETA)
    if relax:
        log.warning("--relax-asserts: generator/calibration assertions "
                    "OFF (plumbing smoke only)")

    # ---------- parity vs code/model/tan.py ----------
    log.info("Audit: parity vs code/model/tan.py")
    if not parity_check(log):
        raise SystemExit("parity FAIL -> abort (no silent continuation)")

    # ---------- trials ----------
    trials = {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        tr = gen_trial(rng, args.trial_length)
        rep = generator_report(tr, args.trial_length)
        if not relax:
            ok = assert_generator(rep, log)
            if not ok:
                raise SystemExit("generator sanity FAIL -> abort")
        log.info("  seed %d: %s", seed, rep)
        x = tr["x"]
        pulse = tr["pulse"]
        ons = tr["onsets"]
        # valid events: within-burn + full tau range, clean pre-window
        valid = np.zeros(len(ons), dtype=bool)
        for j, te in enumerate(ons):
            if not (te >= args.burn_in + PRE_WIN and
                    te <= args.trial_length - 1 - TAU_MAX - 1):
                continue
            if pulse[te - PRE_WIN:te].any():
                continue
            valid[j] = True
        te_arr = ons[valid]
        # isolation flag (no onset within (te, te+TAU_MAX])
        iso = np.zeros(len(te_arr), dtype=bool)
        for j, te in enumerate(te_arr):
            nxt = ons[ons > te]
            iso[j] = True if len(nxt) == 0 else bool(nxt[0] > te + TAU_MAX)
        sims = {}
        for kind in ("lif", "noattn", "tan"):
            s = simulate_traces(x, kind)
            s["x"] = x
            sims[kind] = s
        frames = {}
        consts = {}
        for kind in ("lif", "noattn", "tan"):
            frames[kind] = {}
            consts[kind] = {}
            for fn, Z in build_frames(kind, sims[kind], x).items():
                Zs, nc = standardise(Z, args.burn_in)
                frames[kind][fn] = Zs
                consts[kind][fn] = nc
        # input-only frame zU (from input-derived mu,S of the TAN sim)
        Zu = np.column_stack([sims["tan"]["mu"], sims["tan"]["S"], x])
        Zu, nc_u = standardise(Zu, args.burn_in)
        trials[seed] = dict(x=x, pulse=pulse, te=te_arr, iso=iso,
                            first_amp=x[te_arr], frames=frames,
                            consts=consts, Zu=Zu, sims=sims,
                            const_u=nc_u)
        log.info("    valid events=%d (isolated=%d)", len(te_arr),
                 int(iso.sum()))
        for kind in ("lif", "noattn", "tan"):
            for fn, nc in consts[kind].items():
                if nc:
                    log.info("    %s %s: %d constant coord -> 0", kind, fn,
                             nc)

    # items to analyse: (kind, frame, pool-name)
    items = [("lif", "zC"), ("noattn", "zC"), ("noattn", "zF"),
             ("tan", "zC"), ("tan", "zF"), ("input", "zU")]

    def Z_of(seed, item):
        kind, fn = item
        if kind == "input":
            return trials[seed]["Zu"]
        return trials[seed]["frames"][kind][fn]

    # ---------- quiet geometry ----------
    # NOTE (estimator fix, logged 2026-09-04): per-seed global Z-scoring
    # assigns every coordinate an arbitrary per-seed level; pooling raw
    # standardized quiet samples across seeds would inject that
    # normalisation level as artificial between-seed variance.  Quiet
    # blocks are therefore centred per seed (per-column) before pooling.
    # (Velocity/Delta-z event quantities are unaffected: per-seed levels
    # cancel by construction.)  Same fix applied to the raw-state
    # secondary diagnostic below.
    log.info("Quiet reference (pooled over seeds, per-seed centred)")
    quiet = {}
    for item in items:
        Zs_all, V_all = [], []
        for seed in seeds:
            d = trials[seed]
            Z = Z_of(seed, item)
            pulse, T = d["pulse"], args.trial_length
            pi = np.where(pulse)[0]
            if len(pi) == 0:
                q = np.arange(args.burn_in, T)
            else:
                lo = np.searchsorted(pi, np.arange(T), "right") - 1
                hi = np.searchsorted(pi, np.arange(T), "left")
                hi = np.minimum(hi, len(pi) - 1)
                dl = np.where(lo >= 0, np.arange(T) - pi[np.maximum(lo, 0)],
                              10 ** 9)
                dr = np.where(hi >= 0, pi[hi] - np.arange(T), 10 ** 9)
                q = np.where((dl >= QUIET_DIST) & (dr >= QUIET_DIST))[0]
            q = q[q >= args.burn_in]
            keep = []
            last = -10 ** 9
            for tt in q:
                if tt - last >= QUIET_DIST:
                    keep.append(tt)
                    last = tt
            q = np.asarray(keep)
            qv = q[(q + 1 < T)]
            Zb = Z[q]
            Vb = Z[qv + 1] - Z[qv]
            Zs_all.append(Zb - Zb.mean(axis=0, keepdims=True))
            V_all.append(Vb - Vb.mean(axis=0, keepdims=True))
        Zq = np.concatenate(Zs_all)
        Vq = np.concatenate(V_all)
        dq, ltq, lamq, nq = pr_logtr(Zq)
        dqv, ltv, lamv, nv = pr_logtr(Vq)
        quiet[item] = dict(dGq=dq, ltq=ltq, dDq=dqv, n=nq, lam=lamq)
        log.info("  %-6s %s: quiet dGq=%s dDq=%s logTr=%s (n=%d)",
                 MODEL_NAME[item[0]], item[1],
                 "-" if not np.isfinite(dq) else f"{dq:.2f}",
                 "-" if not np.isfinite(dqv) else f"{dqv:.2f}",
                 "-" if not np.isfinite(ltq) else f"{ltq:.2f}", nq)

    # ---------- event pooling ----------
    log.info("Event-locked pooling + geometry")
    tg = tau_grid()
    geo = {}          # item -> per-tau geometry dicts
    iso_geo = {}
    S_trace, A_trace = [], []
    ev_counts = [len(trials[s]["te"]) for s in seeds]
    for item in items:
        kind, fn = item
        Dz = {t: [] for t in tg}
        V = {t: [] for t in tg}
        Rz = {t: [] for t in tg}
        A0 = {t: [] for t in tg}
        Dz_i = {t: [] for t in tg}
        V_i = {t: [] for t in tg}
        A0_i = {t: [] for t in tg}
        SA = {t: [] for t in tg}      # S and A traces for fig24 (B4 zF)
        AA = {t: [] for t in tg}
        for seed in seeds:
            d = trials[seed]
            Z = Z_of(seed, item)
            sim = d["sims"]["tan"] if kind != "lif" else d["sims"]["lif"]
            if kind == "input":
                sim = d["sims"]["tan"]
            for j, te in enumerate(d["te"]):
                pre = Z[te - PRE_WIN:te].mean(axis=0)
                iso_j = bool(d["iso"][j])
                for tau in tg:
                    t1 = te + tau
                    dz = Z[t1] - pre
                    v = Z[t1 + 1] - Z[t1]
                    Dz[tau].append(dz)
                    V[tau].append(v)
                    Rz[tau].append(Z[t1])
                    A0[tau].append(d["first_amp"][j])
                    if iso_j:
                        Dz_i[tau].append(dz)
                        V_i[tau].append(v)
                        A0_i[tau].append(d["first_amp"][j])
                if kind == "tan" and fn == "zF":
                    for tau in tg:
                        t1 = te + tau
                        SA[tau].append(float(sim["S"][t1]))
                        AA[tau].append(float(sim["A"][t1]))
        geo[item] = {}
        for tau in tg:
            dz = np.asarray(Dz[tau])
            v = np.asarray(V[tau])
            rz = np.asarray(Rz[tau])
            # normalisation-level fix (see quiet block): per-seed centre the
            # raw-state secondary matrix before pooling
            b0 = 0
            for c in ev_counts:
                rz[b0:b0 + c] -= rz[b0:b0 + c].mean(axis=0, keepdims=True)
                b0 += c
            lA = np.log10(np.maximum(np.asarray(A0[tau]), 1e-9))
            g = geometry_of(dz, lA)
            gv = geometry_of(v, lA)
            gr = geometry_of(rz, lA)
            geo[item][int(tau)] = dict(dG=g["d"], ltG=g["lt"], lam=g["lam"],
                                       dG_s=g["d_strat"],
                                       ltG_s=g["lt_strat"],
                                       dG_r=g["d_resid"],
                                       ltG_r=g["lt_resid"],
                                       dD=gv["d"], ltD=gv["lt"],
                                       dG_raw=gr["d"], n=g["n"])
        iso_geo[item] = {}
        for tau in tg:
            dz = np.asarray(Dz_i[tau])
            v = np.asarray(V_i[tau])
            if len(dz) == 0:
                iso_geo[item][int(tau)] = None
                continue
            lA = np.log10(np.maximum(np.asarray(A0_i[tau]), 1e-9))
            g = geometry_of(dz, lA)
            gv = geometry_of(v, lA)
            iso_geo[item][int(tau)] = dict(dG=g["d"], ltG=g["lt"],
                                           dD=gv["d"], n=g["n"])
        if kind == "tan" and fn == "zF":
            S_trace = {int(t): np.asarray(SA[t]) for t in tg}
            A_trace = {int(t): np.asarray(AA[t]) for t in tg}
        log.info("  %s %s pooled: n(events) = %d",
                 MODEL_NAME[kind], fn, geo[item][0]["n"])

    # ---------- summary, deltas, CIs ----------
    summary = []
    curves_rows = []
    for item in items:
        kind, fn = item
        g = geo[item]
        ev = EV_TAUS
        def mean_ev(key):
            vals = [g[int(t)][key] for t in ev]
            vals = [x for x in vals if np.isfinite(x)]
            return float(np.mean(vals)) if vals else float("nan")
        dDe, dGe = mean_ev("dD"), mean_ev("dG")
        dGs, dGr = mean_ev("dG_s"), mean_ev("dG_r")
        ltE = mean_ev("ltG")
        q = quiet[item]
        dD_delta = dDe - q["dDq"] if np.isfinite(dDe) and np.isfinite(
            q["dDq"]) else float("nan")
        lt_delta = ltE - q["ltq"] if np.isfinite(ltE) and np.isfinite(
            q["ltq"]) else float("nan")
        # recovery on isolated subset (tau >= 0)
        ig = iso_geo[item]
        dDc = np.array([ig[int(t)]["dD"] if ig[int(t)] else np.nan
                        for t in tg[tg >= 0]])
        lTc = np.array([ig[int(t)]["ltG"] if ig[int(t)] else np.nan
                        for t in tg[tg >= 0]])
        th, ef = recovery_vals(dDc, lTc, q["dDq"])
        summary.append(dict(model=MODEL_NAME[kind], frame=fn,
                            n_events=g[0]["n"],
                            dGq=q["dGq"], dDq=q["dDq"], ltq=q["ltq"],
                            dGe=dGe, dDe=dDe, dGs=dGs, dGr=dGr, ltE=ltE,
                            dD_delta=dD_delta, lt_delta=lt_delta,
                            tau_half=th, tau_efold=ef))
        log.info("  %s %s: n=%d dDe=%s dD_delta=%s dGe=%s logTr_delta=%s "
                 "tau_half=%s tau_efold=%s", MODEL_NAME[kind], fn,
                 g[0]["n"], "-" if not np.isfinite(dDe) else f"{dDe:.2f}",
                 "-" if not np.isfinite(dD_delta) else f"{dD_delta:.2f}",
                 "-" if not np.isfinite(dGe) else f"{dGe:.2f}",
                 "-" if not np.isfinite(lt_delta) else f"{lt_delta:.2f}",
                 "-" if th is None else f"{th:.0f}",
                 "-" if ef is None else f"{ef:.1f}")
        for tau in tg:
            gg = g[int(tau)]
            curves_rows.append(dict(model=MODEL_NAME[kind], frame=fn,
                                    tau=int(tau), n=gg["n"],
                                    dG=gg["dG"], dG_strat=gg["dG_s"],
                                    dG_resid=gg["dG_r"], dD=gg["dD"],
                                    logTr=gg["ltG"], dG_raw=gg["dG_raw"]))
    # bootstrap CI for event dD and logTr (primary frames)
    log.info("Event bootstrap (n=%d resamples)", N_BOOT_EV)
    rng_b = np.random.default_rng(424242)
    extra = {}
    for item in items:
        kind, fn = item
        # per-event velocity and response vectors at tau in EV_TAUS
        Vv = {t: [] for t in EV_TAUS}
        Dv = {t: [] for t in EV_TAUS}
        for seed in seeds:
            d = trials[seed]
            Z = Z_of(seed, item)
            for te in d["te"]:
                pre = Z[te - PRE_WIN:te].mean(axis=0)
                for t in EV_TAUS:
                    Vv[t].append(Z[te + t + 1] - Z[te + t])
                    Dv[t].append(Z[te + t] - pre)
        nb = len(Vv[0])
        Vv = {t: np.asarray(Vv[t]) for t in EV_TAUS}
        Dv = {t: np.asarray(Dv[t]) for t in EV_TAUS}
        boot_dD, boot_lt = [], []
        for _ in range(N_BOOT_EV):
            ix = rng_b.integers(0, nb, size=nb)
            dd = np.mean([pr_logtr(Vv[t][ix])[0] for t in EV_TAUS])
            lt = np.mean([pr_logtr(Dv[t][ix])[1] for t in EV_TAUS])
            boot_dD.append(dd)
            boot_lt.append(lt)
        ciD = (float(np.nanpercentile(boot_dD, 2.5)),
               float(np.nanpercentile(boot_dD, 97.5)))
        ciL = (float(np.nanpercentile(boot_lt, 2.5)),
               float(np.nanpercentile(boot_lt, 97.5)))
        for r in summary:
            if r["model"] == MODEL_NAME[kind] and r["frame"] == fn:
                r["dD_event_ci_lo"] = ciD[0]
                r["dD_event_ci_hi"] = ciD[1]
                r["lt_event_ci_lo"] = ciL[0]
                r["lt_event_ci_hi"] = ciL[1]
        extra[item] = (ciD, ciL)
        log.info("  %s %s: dD_event CI [%.2f, %.2f]", MODEL_NAME[kind], fn,
                 ciD[0], ciD[1])

    # recovery CI for B4 zF and B3 zF (isolated events)
    log.info("Recovery bootstrap (n=%d)", N_BOOT_REC)
    rng_r = np.random.default_rng(999)
    for item in (("tan", "zF"), ("noattn", "zF")):
        kind, fn = item
        tgp = tg[tg >= 0]
        Vv = {int(t): [] for t in tgp}
        Dv = {int(t): [] for t in tgp}
        for seed in seeds:
            d = trials[seed]
            Z = Z_of(seed, item)
            for j, te in enumerate(d["te"]):
                if not d["iso"][j]:
                    continue
                pre = Z[te - PRE_WIN:te].mean(axis=0)
                for t in tgp:
                    Vv[int(t)].append(Z[te + t + 1] - Z[te + t])
                    Dv[int(t)].append(Z[te + t] - pre)
        Vv = {t: np.asarray(Vv[t]) for t in Vv}
        Dv = {t: np.asarray(Dv[t]) for t in Dv}
        nb = len(Vv[0])
        q = quiet[item]
        boot_th, boot_ef = [], []
        for _ in range(N_BOOT_REC):
            ix = rng_r.integers(0, nb, size=nb)
            dDc = np.array([pr_logtr(Vv[int(t)][ix])[0] for t in tgp])
            lTc = np.array([pr_logtr(Dv[int(t)][ix])[1] for t in tgp])
            th, ef = recovery_vals(dDc, lTc, q["dDq"])
            boot_th.append(th if th is not None else float("nan"))
            boot_ef.append(ef if ef is not None else float("nan"))
        ci_th = (float(np.nanpercentile(boot_th, 2.5)),
                 float(np.nanpercentile(boot_th, 97.5)))
        ci_ef = (float(np.nanpercentile(boot_ef, 2.5)),
                 float(np.nanpercentile(boot_ef, 97.5)))
        for r in summary:
            if r["model"] == MODEL_NAME[kind] and r["frame"] == fn:
                r["tau_half_ci_lo"] = ci_th[0]
                r["tau_half_ci_hi"] = ci_th[1]
                r["tau_efold_ci_lo"] = ci_ef[0]
                r["tau_efold_ci_hi"] = ci_ef[1]
        log.info("  %s %s: tau_1/2 CI [%s, %s], tau_efold CI [%s, %s]",
                 MODEL_NAME[kind], fn, "-" if not np.isfinite(ci_th[0]) else
                 f"{ci_th[0]:.1f}", "-" if not np.isfinite(ci_th[1]) else
                 f"{ci_th[1]:.1f}", "-" if not np.isfinite(ci_ef[0]) else
                 f"{ci_ef[0]:.1f}", "-" if not np.isfinite(ci_ef[1]) else
                 f"{ci_ef[1]:.1f}")

    # ---------- decision gate (pre-registered) ----------
    def rw(model, frame):
        return next(r for r in summary if r["model"] == model and
                    r["frame"] == frame)
    b4f, b3f = rw("B4", "zF"), rw("B3", "zF")
    b4c, b1c, c0 = rw("B4", "zC"), rw("B1", "zC"), rw("C0", "zU")
    stable = bool(np.isfinite(b4f["dD_delta"]) and
                  b4f["dD_delta"] >= STABLE_EXPANSION and
                  b4f["dD_event_ci_lo"] > 0)
    c_ok = lambda a, b: np.isfinite(a) and np.isfinite(b)
    A = stable and c_ok(b4f["dD_event_ci_lo"], b3f["dD_event_ci_hi"]) and \
        b4f["dD_event_ci_lo"] > b3f["dD_event_ci_hi"] and \
        c_ok(b4c["dDe"], b1c["dDe"]) and b4c["dDe"] > b1c["dDe"] and \
        c_ok(b4c["dDe"], c0["dDe"]) and b4c["dDe"] > c0["dDe"]
    dlt = b4f["lt_delta"] - b3f["lt_delta"] if np.isfinite(
        b4f["lt_delta"]) and np.isfinite(b3f["lt_delta"]) else float("nan")
    B = (not A) and (not (c_ok(b4f["dD_delta"], b3f["dD_delta"]) and
                          b4f["dD_delta"] > b3f["dD_delta"])) and \
        c_ok(b4f["dDe"], b3f["dDe"]) and abs(b4f["dDe"] - b3f["dDe"]) < 0.5 \
        and c_ok(dlt, 0.0) and dlt >= MIN_LOGT_DIFF
    C = (not A and not B and c_ok(b4c["dDe"], b1c["dDe"]) and
         abs(b4c["dDe"] - b1c["dDe"]) < 0.5 and c_ok(b4c["dDe"], c0["dDe"])
         and abs(b4c["dDe"] - c0["dDe"]) < 0.5)
    outcome = "A" if A else ("B" if B else ("C" if C else "D"))
    log.info("-" * 78)
    log.info("DECISION GATE (pre-registered thresholds)")
    log.info("  B4 zF: dD_delta=%.2f CI[%.2f,%.2f] | B3 zF: dD_delta=%.2f "
             "CI[%.2f,%.2f] | event dD: B4zC=%.2f B1zC=%.2f C0zU=%.2f",
             b4f["dD_delta"], b4f["dD_event_ci_lo"], b4f["dD_event_ci_hi"],
             b3f["dD_delta"], b3f["dD_event_ci_lo"],
             b3f["dD_event_ci_hi"], b4c["dDe"], b1c["dDe"], c0["dDe"])
    log.info("  logTr delta B4-B3 (zF) = %.3f nats (>=%.2f for B)", dlt,
             MIN_LOGT_DIFF)
    log.info("  OUTCOME: %s", outcome)
    notes = {"A": "TAN-specific effective-rank expansion (working claim)",
             "B": "Attention acts through amplification / directional "
                  "reweighting, not effective rank",
             "C": "TAN-specific breathing not supported (expansion is "
                  "event-induced, not TAN-specific)",
             "D": "No stable event-locked expansion for B4 -> check "
                  "calibration, then pivot to anisotropy / curvature / "
                  "context coupling"}
    log.info("  -> %s", notes[outcome])

    # ---------- calibration asserts (full runs only) ----------
    if not relax:
        log.info("Calibration assertions")
        okc = True
        for r in summary:
            dim = FRAME_DIM[r["frame"]]
            for key in ("dGq", "dDq", "dGe", "dDe", "dGs", "dGr"):
                v = r[key]
                if np.isfinite(v) and not (0.8 - 1e-9 <= v <= dim + 1e-6):
                    log.error("  ASSERT FAIL: %s %s %s=%.3f outside "
                              "[0.8, %d]", r["model"], r["frame"], key, v,
                              dim)
                    okc = False
        lif_q = rw("B1", "zC")["dGq"]
        if np.isfinite(lif_q) and not (0.8 <= lif_q <= 2.0):
            log.error("  ASSERT FAIL: LIF quiet dGq = %.3f not in [0.8,2.0]",
                      lif_q)
            okc = False
        if not okc:
            raise SystemExit("calibration FAIL -> abort")
        log.info("  calibration OK")

    # ---------- figures ----------
    log.info("Figures (fig24/25/26)")
    # fig24 payload (B4 zF): per-tau curves + honest event-bootstrap bands
    it = ("tan", "zF")
    S_m = np.array([float(np.mean(S_trace[int(t)])) if len(S_trace[int(t)])
                    else np.nan for t in tg])
    S_l = np.array([float(np.percentile(S_trace[int(t)], 2.5)) for t in tg])
    S_h = np.array([float(np.percentile(S_trace[int(t)], 97.5))
                    for t in tg])
    A_m = np.array([float(np.mean(A_trace[int(t)])) if len(A_trace[int(t)])
                    else np.nan for t in tg])
    A_l = np.array([float(np.percentile(A_trace[int(t)], 2.5)) for t in tg])
    A_h = np.array([float(np.percentile(A_trace[int(t)], 97.5)) for t in tg])
    dDc = np.array([geo[it][int(t)]["dD"] for t in tg])
    dGc = np.array([geo[it][int(t)]["dG"] for t in tg])
    lTc = np.array([geo[it][int(t)]["ltG"] for t in tg])
    lam_e = geo[it][0]["lam"]
    lam_q = quiet[it]["lam"]
    # event bootstrap per tau (B4 zF): resample events, PR of pooled cov
    EvV = {int(t): [] for t in tg}
    EvD = {int(t): [] for t in tg}
    for seed in seeds:
        d = trials[seed]
        Z = Z_of(seed, it)
        for te in d["te"]:
            pre = Z[te - PRE_WIN:te].mean(axis=0)
            for t in tg:
                EvV[int(t)].append(Z[te + t + 1] - Z[te + t])
                EvD[int(t)].append(Z[te + t] - pre)
    EvV = {t: np.asarray(v) for t, v in EvV.items()}
    EvD = {t: np.asarray(v) for t, v in EvD.items()}
    nb = len(EvV[0])
    rng_f = np.random.default_rng(31337)
    N_BOOT_FIG = 300
    dD_lo, dD_hi = np.full(len(tg), np.nan), np.full(len(tg), np.nan)
    lt_lo, lt_hi = np.full(len(tg), np.nan), np.full(len(tg), np.nan)
    for i, t in enumerate(tg):
        bs_d, bs_l = [], []
        for _ in range(N_BOOT_FIG):
            ix = rng_f.integers(0, nb, size=nb)
            bs_d.append(pr_logtr(EvV[int(t)][ix])[0])
            bs_l.append(pr_logtr(EvD[int(t)][ix])[1])
        dD_lo[i], dD_hi[i] = np.nanpercentile(bs_d, 2.5), np.nanpercentile(
            bs_d, 97.5)
        lt_lo[i], lt_hi[i] = np.nanpercentile(bs_l, 2.5), np.nanpercentile(
            bs_l, 97.5)
    payload24 = dict(tau=tg, S_mean=S_m, S_lo=S_l, S_hi=S_h, A_mean=A_m,
                     A_lo=A_l, A_hi=A_h, dD=dDc, dD_lo=dD_lo, dD_hi=dD_hi,
                     dG=dGc, lt=lTc, lt_lo=lt_lo, lt_hi=lt_hi,
                     dDq=quiet[it]["dDq"], ltq=quiet[it]["ltq"],
                     delta_dD=rw("B4", "zF")["dD_delta"],
                     delta_lt=rw("B4", "zF")["lt_delta"],
                     ci_lo=rw("B4", "zF")["dD_event_ci_lo"],
                     ci_hi=rw("B4", "zF")["dD_event_ci_hi"],
                     tau_half=rw("B4", "zF")["tau_half"],
                     tau_efold=rw("B4", "zF")["tau_efold"],
                     lam_event=lam_e, lam_quiet=lam_q)
    make_fig24(payload24, fig_dir)

    profiles = []
    for name, item, frame in (("B1", ("lif", "zC"), "zC"),
                              ("B3", ("noattn", "zC"), "zC"),
                              ("B3", ("noattn", "zF"), "zF"),
                              ("B4", ("tan", "zC"), "zC"),
                              ("B4", ("tan", "zF"), "zF"),
                              ("C0", ("input", "zU"), "zU")):
        g = geo[item]
        dd = np.array([g[int(t)]["dD"] for t in tg])
        lt = np.array([g[int(t)]["ltG"] for t in tg])
        qq = quiet[item]
        base_d = qq["dDq"] if np.isfinite(qq["dDq"]) else 0.0
        base_lt = qq["ltq"] if np.isfinite(qq["ltq"]) else 0.0
        profiles.append(dict(name=name, frame=frame, d=dd - base_d,
                             lt=lt - base_lt))
    make_fig25(dict(tau=tg, profiles=profiles), fig_dir)

    sp = {}
    for name, item in (("B4", ("tan", "zF")), ("B3", ("noattn", "zF"))):
        sp[name] = dict(lam_event=geo[item][0]["lam"],
                        lam_quiet=quiet[item]["lam"],
                        d_e=rw(name, "zF")["dDe"], d_q=quiet[item]["dGq"])
    make_fig26(sp, fig_dir)
    for f in ("fig24_eventlocked", "fig25_recovery", "fig26_spectra"):
        for ext in ("png", "pdf"):
            (ms_dir / f"{f}.{ext}").write_bytes(
                (fig_dir / f"{f}.{ext}").read_bytes())

    # ---------- persistence ----------
    keys = sorted({k for r in summary for k in r.keys()})
    with open(tab_dir / "effective_dimension_summary.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in summary:
            w.writerow(r)
    ck = sorted({k for r in curves_rows for k in r.keys()})
    with open(tab_dir / "effective_dimension_curves.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=ck)
        w.writeheader()
        for r in curves_rows:
            w.writerow(r)
    for f in ("effective_dimension_summary.csv",
              "effective_dimension_curves.csv"):
        (dat_dir / f).write_bytes((tab_dir / f).read_bytes())

    # events npz: per-event meta + B4 zF response arrays at tau 0..2
    ev_meta = []
    V3, D3 = [], []
    for seed in seeds:
        d = trials[seed]
        Z = Z_of(seed, ("tan", "zF"))
        for j, te in enumerate(d["te"]):
            ev_meta.append([seed, int(te), float(d["first_amp"][j]),
                            int(d["iso"][j])])
            pre = Z[te - PRE_WIN:te].mean(axis=0)
            V3.append([Z[te + t + 1] - Z[te + t] for t in (0, 1, 2)])
            D3.append([Z[te + t] - pre for t in (0, 1, 2)])
    np.savez_compressed(
        dat_dir / "effective_dimension_events.npz",
        meta=np.asarray(ev_meta, dtype=np.float64),
        v_tau012=np.asarray(V3, dtype=np.float64),
        dz_tau012=np.asarray(D3, dtype=np.float64))

    # geometry npz
    geo_np = {"tau": tg}
    for item in items:
        g = geo[item]
        key = f"{MODEL_NAME[item[0]]}_{item[1]}"
        geo_np[key + "_dG"] = np.array([g[int(t)]["dG"] for t in tg])
        geo_np[key + "_dG_s"] = np.array([g[int(t)]["dG_s"] for t in tg])
        geo_np[key + "_dG_r"] = np.array([g[int(t)]["dG_r"] for t in tg])
        geo_np[key + "_dD"] = np.array([g[int(t)]["dD"] for t in tg])
        geo_np[key + "_logTr"] = np.array([g[int(t)]["ltG"] for t in tg])
        geo_np[key + "_dG_raw"] = np.array([g[int(t)]["dG_raw"] for t in tg])
        q = quiet[item]
        geo_np[key + "_quiet"] = np.asarray([q["dGq"], q["dDq"], q["ltq"]])
    np.savez_compressed(dat_dir / "effective_dimension_geometry.npz",
                        **geo_np)
    log.info("Persisted: tables, events npz, geometry npz, figures.")
    log.info("Phase 2-Pilot complete in %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
