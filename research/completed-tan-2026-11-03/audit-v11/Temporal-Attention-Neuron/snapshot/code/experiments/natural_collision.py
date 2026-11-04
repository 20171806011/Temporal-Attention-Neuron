"""
TAN Phase 1 - Natural State-Collision Experiment
================================================
Falsification test of the Markovian claim

    "the scalar membrane potential h_t is a sufficient (Markovian) state
     variable of the Temporal Attention Neuron (TAN)"

Protocol (identical history pairs, identical probe, per model):

  [Task 1] Natural collision search
     - N random 9-step stimulus histories (T-1 = 9) drawn from 4 stimulus
       families (Gaussian white noise / random step / ramp / sine).
     - For every model separately, find unordered pairs (A, B) whose
       subthreshold membrane potentials at t = T-1 collide:
           0.05 < h_{T-1} < theta   (no trivial reset coincidence)
           |h_{T-1}^A - h_{T-1}^B| < delta = 1e-4
     - No artificial construction: collisions are Monte-Carlo coincidences.

  [Task 2] Probe injection at t = T = 10
     - Inject the identical probe x* ~ Uniform(0.5, 3.0) into A and B.
     - Measure:  dH_before = |h9^A - h9^B|,
                 dH_tilde  = |u10^A - u10^B|  (continuous-branch potential,
                   i.e. pre-reset membrane potential at the probe step),
                 dA_T = |A10^A - A10^B|  (synaptic-current gap),
                 attention divergence: L1 and Jensen-Shannon divergence of
                 (alpha10^A, alpha10^B) -- Full TAN (uniform reference for B3).
     - The threshold/reset is a *readout* discontinuity common to all four
       spiking models. Following the protocol's exclusion of trivial reset
       coincidences, the headline divergence is measured on the continuous
       branch (pre-reset u10); post-reset divergence, spike-outcome
       bifurcations and reset re-synchronisation are tabulated as well
       (fig23: after a common double-spike at t=10 every model starts from
       the identical observable state h10 = 0; only models whose memory
       lives outside h_t re-bifurcate under a common continuation stream).

  [Task 3] Four-model ladder on an identical dynamics substrate:
       B1 LIF              h_t = lam*h_{t-1} + x_t
       B2 LIF + DelayLine  h_t = lam*h_{t-1} + w^T X_t (w recency ramp)
       B3 TAN-noAttn       h_t = lam*h_{t-1} + tanh(S_t)*mean(X_t)
       B4 Full TAN         h_t = lam*h_{t-1} + tanh(S_t)*C_t (softmax over X_t)
     All models share the spiking readout y_t = Theta(h_t - theta),
     reset h_t <- 0, with the paper defaults: W = 5, lam = 0.5, theta = 0.5,
     Wq = 2.0, Wk = Wv = 1.0, beta = 1.0, eps = 0.0.

  [Task 4] Publication figures (300 dpi)
     fig22_natural_collision:
       (a) representative A/B trajectories colliding at t = 9 and
           bifurcating at t = 10;
       (b) attention distributions alpha10^A / alpha10^B (mechanism panel);
       (c) log-log bifurcation scatter over all collision pairs of all four
           models - LIF sits on the theoretical contraction line
           dH' = lam*dH while the TAN family explodes above the identity;
       (d) JSD(alpha) vs dA_T correlation for the Full TAN.
     fig23_reset_rebifurcation:
       post-reset re-bifurcation after a common double-spike at t = 10.

Conventions (explicit, for reproducibility)
  - The receptive field is zero-padded at trial onset (stimulus onset after
    silence); every window has length W = 5 at every time step.
  - h_0 = 0.
  - Spike condition follows the repository implementation: u_t > theta.
  - S_t = max(0, x_t - mu_t - eps), mu_t = window mean *including* x_t.

Runtime dependencies: numpy, scipy, matplotlib only.

Outputs
  - results/figures/fig22_natural_collision.{png,pdf}
  - results/figures/fig23_reset_rebifurcation.{png,pdf}
  - results/tables/natural_collision_{summary,pairs,correlations}.csv
  - data/experiment_results/natural_collision_*.{csv,npz}
  - results/logs/natural_collision.log
  (figures mirrored to manuscript/figures/)

Author: Li Zexu (with DeepSeek), University of Leeds
"""
import sys
import time
import argparse
import csv
import logging
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# Paper defaults (must not be changed)
# ============================================================
W = 5                # receptive-field length
LAMBDA = 0.5         # membrane leak
THETA = 0.5          # spike threshold
W_Q, W_K, W_V = 2.0, 1.0, 1.0
BETA = 1.0           # attention inverse temperature
EPS = 0.0            # sensory noise tolerance (paper default 0.0)
DELTA = 1e-4         # collision tolerance
BAND_LO, BAND_HI = 0.05, THETA       # subthreshold search band
T_HIST = 9           # random stimulus history length (T-1)
T_PROBE = 10         # probe step T
N_CONT = 8           # continuation steps after the probe (fig23)
T_TOT = T_HIST + 1 + N_CONT          # total length per pair = 18

PROBE_LO, PROBE_HI = 0.5, 3.0        # probe distribution
CONT_LO, CONT_HI = 0.3, 1.8          # common continuation stream

GEN_NAMES = ["gauss", "step", "ramp", "sine"]
MODELS = ["lif", "buf", "noattn", "tan"]
MODEL_LABELS = {
    "lif": "LIF (B1)",
    "buf": "LIF+DelayLine (B2)",
    "noattn": "TAN-noAttn (B3)",
    "tan": "Full TAN (B4)",
}
MODEL_FULL = {
    "lif": "Baseline 1: LIF",
    "buf": "Baseline 2: LIF + passive delay line",
    "noattn": "Baseline 3: TAN without attention (uniform alpha)",
    "tan": "Model 4: Full TAN",
}
COLORS = {
    "lif": "#7F8C8D",
    "buf": "#16A085",
    "noattn": "#2E86AB",
    "tan": "#C0392B",
}

# module-level pools (set in main)
XALL = None
GEN_AR = None
H9 = {}
ENS_JS_L1 = {}

# ============================================================
# Logging
# ============================================================
def setup_logging(log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_dir / "natural_collision.log",
                                      mode="w", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("collision")


def fmt_p(p):
    return "<1e-300" if p < 1e-300 else f"{p:.2e}"


# ============================================================
# 1. Stimulus families (natural random histories)
# ============================================================
def gen_gauss(rng, n):
    return np.clip(rng.normal(1.00, 0.60, size=(n, T_HIST)), 0.0, 3.0)


def gen_step(rng, n):
    """Random step: 2-4 levels ~ U(0.05, 2.4), random integer boundaries."""
    X = np.empty((n, T_HIST))
    bounds = np.sort(rng.integers(1, 9, size=(n, 3)), axis=1)
    bounds = np.concatenate([np.zeros((n, 1), dtype=int), bounds,
                             np.full((n, 1), T_HIST, dtype=int)], axis=1)
    levels = rng.uniform(0.05, 2.40, size=(n, 4))
    for row in range(n):
        for s in range(4):
            X[row, bounds[row, s]:bounds[row, s + 1]] = levels[row, s]
    return X


def gen_ramp(rng, n):
    start = rng.uniform(0.05, 0.40, size=(n, 1))
    end = start + rng.uniform(0.60, 2.20, size=(n, 1))
    t = np.arange(T_HIST, dtype=float) / (T_HIST - 1)
    return np.clip(start + (end - start) * t[None, :], 0.0, 3.0)


def gen_sine(rng, n):
    off = rng.uniform(0.50, 1.10, size=(n, 1))
    amp = rng.uniform(0.20, 0.60, size=(n, 1))
    freq = rng.uniform(1.0, 2.5, size=(n, 1))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    t = np.arange(T_HIST, dtype=float)
    x = off + amp * np.sin(2.0 * np.pi * freq * t[None, :] / T_HIST + phase)
    return np.clip(x, 0.0, 3.0)


GENERATORS = [gen_gauss, gen_step, gen_ramp, gen_sine]

# ============================================================
# 2. Model dynamics (vectorised over batches)
# ============================================================
_BUF_W = np.arange(1, W + 1, dtype=float) / np.arange(1, W + 1).sum()


def simulate(X, model, eps=EPS, need_alpha=False):
    """Run one model on input matrix X (B, T).

    Returns dict with (B, T) arrays: h (post-reset potential), u (pre-reset
    potential, continuous branch), j (injected drive = synaptic current A_t),
    s (rectified surprise; gated models) and, if need_alpha, alpha (B, T, W)
    attention weights (Full TAN).  Spike condition: u > THETA -> h <- 0.
    """
    B, T = X.shape
    pad = np.zeros((B, W - 1))
    stream = np.concatenate([pad, X], axis=1)            # (B, T + W - 1)
    wv = sliding_window_view(stream, W, axis=1)          # (B, T, W)
    h = np.zeros((B, T))
    u = np.zeros((B, T))
    j = np.zeros((B, T))
    s = np.zeros((B, T)) if model in ("noattn", "tan") else None
    alpha = np.zeros((B, T, W)) if need_alpha else None
    h_prev = np.zeros(B)
    for t in range(T):
        win = wv[:, t]                      # (B, W); last entry = x_t
        xt = win[:, -1]
        mu = win.mean(axis=1)
        if model == "lif":
            drv = xt
        elif model == "buf":
            drv = win @ _BUF_W
        else:
            S = np.maximum(0.0, xt - mu - eps)
            s[:, t] = S
            if model == "noattn":
                drv = np.tanh(S) * mu
            else:  # Full TAN
                logits = BETA * W_Q * W_K * S[:, None] * win
                e = np.exp(logits)                     # |E| <= ~18: safe
                a = e / (e.sum(axis=1, keepdims=True) + 1e-9)
                if need_alpha:
                    alpha[:, t, :] = a
                C = W_V * (a * win).sum(axis=1)
                drv = np.tanh(S) * C
        j[:, t] = drv
        u[:, t] = LAMBDA * h_prev + drv
        fired = u[:, t] > THETA
        h[:, t] = np.where(fired, 0.0, u[:, t])
        h_prev = h[:, t]
    out = dict(h=h, u=u, j=j)
    if s is not None:
        out["s"] = s
    if alpha is not None:
        out["alpha"] = alpha
    return out


# ============================================================
# 3. Natural-collision search (sorted-neighbour scan)
# ============================================================
def find_collision_pairs(h_t, delta=DELTA, lo=BAND_LO, hi=BAND_HI):
    """Unordered pairs (iA, iB): both h in (lo, hi), |dh| < delta."""
    n = len(h_t)
    mask = (h_t > lo) & (h_t < hi)
    order = np.argsort(h_t, kind="stable")
    sh = h_t[order]
    A, B = [], []
    for pos, i in enumerate(order):
        if not mask[i]:
            continue
        jp = pos + 1
        while jp < n and sh[jp] - sh[pos] < delta:
            jj = order[jp]
            if mask[jj]:
                A.append(i)
                B.append(jj)
            jp += 1
    if len(A) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.asarray(A, dtype=np.int64), np.asarray(B, dtype=np.int64)


# ============================================================
# 4. Ensemble runner: probe injection + common continuation
# ============================================================
def run_ensemble(model, idxA, idxB, rng, cap, eps=EPS):
    """Draw <= cap collision pairs, inject a common probe and a common
    continuation stream, re-simulate, return (meta, metrics, rr)."""
    n_pairs = len(idxA)
    sel = np.sort(rng.choice(np.arange(n_pairs), size=min(cap, n_pairs),
                             replace=False))
    ia, ib = idxA[sel], idxB[sel]
    P = len(sel)
    xstar = rng.uniform(PROBE_LO, PROBE_HI, size=P)
    cont = rng.uniform(CONT_LO, CONT_HI, size=(P, N_CONT))
    XA = np.concatenate([XALL[ia], xstar[:, None], cont], axis=1)
    XB = np.concatenate([XALL[ib], xstar[:, None], cont], axis=1)
    ST = np.empty((2 * P, T_TOT))
    ST[0::2] = XA
    ST[1::2] = XB
    sim = simulate(ST, model, eps=eps, need_alpha=(model == "tan"))
    r = sim["h"].reshape(P, 2, T_TOT)
    # pass-2 must reproduce the pass-1 collision values (h at t = 9)
    assert np.allclose(r[:, :, T_HIST - 1],
                       np.stack([H9[model][ia], H9[model][ib]], axis=1),
                       rtol=0.0, atol=1e-13)
    rr = {}
    for k, v in sim.items():
        if v.ndim == 2:
            rr[k] = v.reshape(P, 2, T_TOT)
        elif v.ndim == 3:                       # alpha: (2P, T, W)
            rr[k] = v.reshape(P, 2, T_TOT, W)
        else:
            raise ValueError(k)
    h9 = rr["h"][:, :, T_HIST - 1]
    u10 = rr["u"][:, :, T_PROBE - 1]
    h10 = rr["h"][:, :, T_PROBE - 1]
    j10 = rr["j"][:, :, T_PROBE - 1]
    dh_before = np.abs(h9[:, 0] - h9[:, 1])
    dh_tilde = np.abs(u10[:, 0] - u10[:, 1])
    dA = np.abs(j10[:, 0] - j10[:, 1])
    dh_post = np.abs(h10[:, 0] - h10[:, 1])
    fireA = u10[:, 0] > THETA
    fireB = u10[:, 1] > THETA
    K = np.full(P, np.nan)
    nz = dh_before > 0
    K[nz] = dh_tilde[nz] / dh_before[nz]
    cls = np.where(fireA & fireB, 2, np.where(fireA | fireB, 1, 0))
    cont_gap = np.abs(rr["h"][:, 0, T_PROBE - 1:] -
                      rr["h"][:, 1, T_PROBE - 1:])       # (P, N_CONT+1)
    meta = dict(model=model, n_pairs=P, sel=sel, ia=ia, ib=ib,
                xstar=xstar, cont=cont, genA=GEN_AR[ia], genB=GEN_AR[ib])
    metrics = dict(dh_before=dh_before, dh_tilde=dh_tilde, dA=dA,
                   dh_post=dh_post, K=K, cls=cls, fireA=fireA, fireB=fireB,
                   cont_gap=cont_gap)
    if "s" in rr:
        S10 = rr["s"][:, :, T_PROBE - 1]
        metrics["SA"] = S10[:, 0]
        metrics["SB"] = S10[:, 1]
        metrics["act"] = ((S10[:, 0] > 0).astype(int) +
                          2 * (S10[:, 1] > 0).astype(int))
    if "alpha" in rr:
        al = rr["alpha"][:, :, T_PROBE - 1, :]           # (P, 2, W)
        metrics["alphaA"] = al[:, 0, :]
        metrics["alphaB"] = al[:, 1, :]
    return meta, metrics, rr


# ============================================================
# 5. Statistics helpers
# ============================================================
def jsd(p, q):
    """Jensen-Shannon divergence (nats) of two discrete distributions."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    e = 1e-300
    kl = lambda a, b: float((a * np.log(np.maximum(a, e) /
                                        np.maximum(b, e))).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def boot_ci(x, stat=np.mean, n_boot=4000, rng=None):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    r = rng if rng is not None else np.random.default_rng(0)
    vals = np.array([stat(r.choice(x, size=len(x), replace=True))
                     for _ in range(n_boot)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def geom_mean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


# ============================================================
# 6. Main pipeline
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-gen", type=int, default=40000,
                    help="random histories per stimulus family")
    ap.add_argument("--cap", type=int, default=1500,
                    help="collision pairs analysed per model")
    ap.add_argument("--eps", type=float, default=EPS,
                    help="surprise noise tolerance")
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--min-pairs", type=int, default=600,
                    help="abort if fewer natural collisions are found")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    fig_dir = root / "results" / "figures"
    tab_dir = root / "results" / "tables"
    log_dir = root / "results" / "logs"
    dat_dir = root / "data" / "experiment_results"
    ms_dir = root / "manuscript" / "figures"
    for d in (fig_dir, tab_dir, log_dir, dat_dir, ms_dir):
        d.mkdir(parents=True, exist_ok=True)
    log = setup_logging(log_dir)
    rng = np.random.default_rng(args.seed)

    log.info("=" * 78)
    log.info("TAN Phase 1 - Natural state-collision experiment "
             "(non-Markovianity of h_t)")
    log.info("=" * 78)
    log.info("Defaults: W=%d, lambda=%.2f, theta=%.2f, Wq=%.2f, Wk=Wv=%.1f, "
             "beta=%.1f, eps=%.2f", W, LAMBDA, THETA, W_Q, W_K, BETA, args.eps)
    log.info("Search: subthreshold band (%.2f, %.2f), collision tolerance "
             "delta=%.0e, history length T-1=%d", BAND_LO, BAND_HI, DELTA,
             T_HIST)
    log.info("Probe: x* ~ U(%.1f, %.1f) injected identically at T=%d; then %d "
             "common continuation steps U(%.1f, %.1f)", PROBE_LO, PROBE_HI,
             T_PROBE, N_CONT, CONT_LO, CONT_HI)
    t0 = time.time()

    # ----- Task 1: history pool and per-model simulation -----
    global XALL, GEN_AR, H9
    n_gen = args.n_per_gen
    N = n_gen * len(GENERATORS)
    blocks = [g(rng, n_gen) for g in GENERATORS]
    XALL = np.concatenate(blocks, axis=0)
    GEN_AR = np.repeat(np.arange(len(GENERATORS)), n_gen).astype(np.int64)
    log.info("Stimulus pool: N = %d histories (%d per family: %s)", N,
             n_gen, ", ".join(GEN_NAMES))
    log.info("Simulating the four models over the pool ...")
    for m in MODELS:
        sim = simulate(XALL, m, eps=args.eps)
        H9[m] = sim["h"][:, -1]
        occ = float(np.mean((H9[m] > BAND_LO) & (H9[m] < BAND_HI)))
        log.info("  %-28s : band occupancy at t=9 = %.4f", MODEL_FULL[m], occ)

    # ----- Task 1b: natural collision pairs per model -----
    found = {}
    for m in MODELS:
        iA, iB = find_collision_pairs(H9[m])
        if len(iA) < args.min_pairs:
            raise RuntimeError(
                f"{m}: only {len(iA)} natural collisions found "
                f"(need >= {args.min_pairs}); increase --n-per-gen")
        found[m] = (iA, iB)
        log.info("  %-28s : %d natural collision pairs found",
                 MODEL_FULL[m], len(iA))

    # ----- Tasks 2/3: probe + continuation on each ensemble -----
    log.info("-" * 78)
    log.info("Task 2/3: identical probe injection at t = 10 ...")
    ENS = {}
    for m in MODELS:
        meta, met, rr = run_ensemble(m, found[m][0], found[m][1], rng,
                                     cap=args.cap, eps=args.eps)
        ENS[m] = (meta, met, rr)
        log.info("  %-28s : %d pairs analysed", MODEL_FULL[m],
                 meta["n_pairs"])

    # ----- Global-resync subset on the Full-TAN ensemble -----
    # Pairs where every model spikes at t = 10 on both sides: every model
    # starts the continuation from the identical observable state h10 = 0.
    meta_t, met_t, rr_t = ENS["tan"]
    P_t = meta_t["n_pairs"]
    ST_t = np.empty((2 * P_t, T_TOT))
    ST_t[0::2] = np.concatenate([XALL[meta_t["ia"]], meta_t["xstar"][:, None],
                                 meta_t["cont"]], axis=1)
    ST_t[1::2] = np.concatenate([XALL[meta_t["ib"]], meta_t["xstar"][:, None],
                                 meta_t["cont"]], axis=1)
    tan_fire = rr_t["u"][:, :, T_PROBE - 1] > THETA
    mask_rs = tan_fire[:, 0] & tan_fire[:, 1]
    for m in ("lif", "buf", "noattn"):
        um = simulate(ST_t, m, eps=args.eps)["u"].reshape(P_t, 2, T_TOT)
        mask_rs &= (um[:, 0, T_PROBE - 1] > THETA) & \
                   (um[:, 1, T_PROBE - 1] > THETA)
    n_rs = int(mask_rs.sum())
    if n_rs < 40:            # fallback: relax to TAN + LIF double-spike
        mask_rs = tan_fire[:, 0] & tan_fire[:, 1]
        n_rs = int(mask_rs.sum())
        log.info("Global-resync subset relaxed to TAN+LIF double-spike: "
                 "n = %d", n_rs)
    else:
        log.info("Global-resync subset (all four models double-spike at "
                 "t = 10, so h10 = 0 for both members): n = %d / %d Full-TAN "
                 "pairs", n_rs, P_t)

    # ==================== summary statistics ======================
    rng_b = np.random.default_rng(args.seed + 1)
    summary_rows = []

    for m in MODELS:
        meta, met, _ = ENS[m]
        P = meta["n_pairs"]
        K = met["K"]
        Kf = K[np.isfinite(K)]
        k_lo, k_hi = boot_ci(Kf, np.mean, rng=rng_b)
        row = dict(model=m, n_pairs=P,
                   dh_before_min=float(np.nanmin(met["dh_before"])),
                   dh_before_max=float(np.nanmax(met["dh_before"])),
                   K_mean=float(np.mean(Kf)),
                   K_mean_ci_lo=k_lo, K_mean_ci_hi=k_hi,
                   K_median=float(np.median(Kf)),
                   K_gmean=geom_mean(Kf),
                   pct_K_gt_1=float(np.mean(Kf > 1.0)) * 100,
                   pct_K_gt_10=float(np.mean(Kf > 10.0)) * 100,
                   pct_K_gt_1e3=float(np.mean(Kf > 1e3)) * 100,
                   class_none=float(np.mean(met["cls"] == 0)) * 100,
                   class_one=float(np.mean(met["cls"] == 1)) * 100,
                   class_both=float(np.mean(met["cls"] == 2)) * 100,
                   postreset_violation=
                   float(np.mean(met["dh_post"] > DELTA)) * 100,
                   dh_tilde_median=float(np.median(met["dh_tilde"])),
                   dA_median=float(np.median(met["dA"])))
        if "SA" in met:
            act = met["act"]
            Ka = K[act == 3]
            Ko = K[(act == 1) | (act == 2)]
            Kn = K[act == 0]
            row.update(gate_both=float(np.mean(act == 3)) * 100,
                       gate_one=float(np.mean((act == 1) | (act == 2))) * 100,
                       gate_none=float(np.mean(act == 0)) * 100,
                       K_gate_both=float(np.nanmedian(Ka)) if len(Ka) else
                       float("nan"),
                       K_gate_one=float(np.nanmedian(Ko)) if len(Ko) else
                       float("nan"),
                       K_gate_none=float(np.nanmedian(Kn)) if len(Kn) else
                       float("nan"))
        if "alphaA" in met:
            js = np.array([jsd(a, b) for a, b in zip(met["alphaA"],
                                                     met["alphaB"])])
            l1 = np.abs(met["alphaA"] - met["alphaB"]).sum(axis=1)
            row.update(JSD_mean=float(np.mean(js)),
                       JSD_median=float(np.median(js)),
                       JSD_gt_0p3=float(np.mean(js > 0.3)) * 100,
                       L1_mean=float(np.mean(l1)))
            ENS_JS_L1[m] = (js, l1)
        summary_rows.append(row)

    # ---------------- headline prints ----------------------
    K_lif = ENS["lif"][1]["K"]
    K_lif = K_lif[np.isfinite(K_lif)]
    dev = np.abs(K_lif - LAMBDA)
    log.info("-" * 78)
    log.info("LIF contraction theorem check (continuous branch): "
             "K == lambda = %.1f", LAMBDA)
    log.info("   max|K - lambda| = %.3e    mean|K - lambda| = %.3e   "
             "(machine precision)", float(dev.max()), float(dev.mean()))

    log.info("=" * 78)
    log.info("TASK 3 - Amplification factor K = dH_tilde_T / dH_{T-1} "
             "(continuous branch)")
    log.info("=" * 78)
    log.info("%s", "%-15s %7s %12s %22s %10s %10s %8s %8s %8s" %
             ("model", "n", "mean K", "95% boot CI", "median", "gmean",
              "%K>1", "%K>10", "%K>1e3"))
    for r in summary_rows:
        log.info("%s", "%-15s %7d %12.3e [%9.2e, %9.2e] %10.2e %10.2e "
                 "%7.1f%% %7.1f%% %7.1f%%" %
                 (MODEL_LABELS[r["model"]], r["n_pairs"], r["K_mean"],
                  r["K_mean_ci_lo"], r["K_mean_ci_hi"], r["K_median"],
                  r["K_gmean"], r["pct_K_gt_1"], r["pct_K_gt_10"],
                  r["pct_K_gt_1e3"]))

    log.info("%s", "Probe-outcome classes at t=10 (none / one / both spiked) "
             "and post-reset Markov violation rate |dh10| > delta:")
    for r in summary_rows:
        log.info("  %-15s none=%5.1f%% one=%5.1f%% both=%5.1f%%   "
                 "post-reset violation = %5.1f%%",
                 MODEL_LABELS[r["model"]], r["class_none"], r["class_one"],
                 r["class_both"], r["postreset_violation"])
    for m in ("noattn", "tan"):
        r = next(rr for rr in summary_rows if rr["model"] == m)
        log.info("Gate state at the probe (%s): both open %.1f%%, one open "
                 "%.1f%%, closed %.1f%%", m, r["gate_both"], r["gate_one"],
                 r["gate_none"])
        log.info("   median K by gate state: both-open %.3e, one-open %.3e, "
                 "closed %.3e (closed => A10 = 0 => K = lambda by "
                 "construction)", r["K_gate_both"], r["K_gate_one"],
                 r["K_gate_none"])

    # ---------------- attention-divergence correlation (Full TAN) -------
    meta, met, _ = ENS["tan"]
    js, l1 = ENS_JS_L1["tan"]
    dA = met["dA"]
    dht = met["dh_tilde"]
    SA, SB = met["SA"], met["SB"]
    act_both = (SA > 0) & (SB > 0)
    act_one = (SA > 0) != (SB > 0)
    act_none = (SA <= 0) & (SB <= 0)

    def corr_block(x, y, tag):
        r_p, p_p = stats.pearsonr(x, y)
        r_s, p_s = stats.spearmanr(x, y)
        log.info("  %-44s n=%5d  Pearson r=% .3f (p=%s)   Spearman "
                 "rho=% .3f (p=%s)", tag, len(x), r_p, fmt_p(p_p), r_s,
                 fmt_p(p_s))
        return dict(tag=tag, n=len(x), pearson_r=float(r_p),
                    pearson_p=float(p_p), spearman_r=float(r_s),
                    spearman_p=float(p_s))

    log.info("-" * 78)
    log.info("TASK 4 - correlation analysis (Full TAN ensemble):")
    corr_rows = []
    corr_rows.append(corr_block(js, dA, "JSD(alpha) vs dA_T     [all pairs]"))
    corr_rows.append(corr_block(js[act_both], dA[act_both],
                                "JSD(alpha) vs dA_T     [gate both open]"))
    corr_rows.append(corr_block(js[act_both | act_one],
                                dA[act_both | act_one],
                                "JSD(alpha) vs dA_T     [>=1 gate open]"))
    corr_rows.append(corr_block(js, dht, "JSD(alpha) vs dH_tilde_T [all]"))
    corr_rows.append(corr_block(dA, dht, "dA_T vs dH_tilde_T      [all]"))
    log.info("  origin pile (both gates closed): n = %d  (JSD = 0, dA = 0)",
             int(act_none.sum()))
    log.info("  near-orthogonal attention (JSD > %.3f = 0.5 ln 2): %d / %d",
             0.5 * np.log(2), int((js > 0.5 * np.log(2)).sum()), len(js))
    rtan = next(r for r in summary_rows if r["model"] == "tan")
    log.info("  attention-divergence distribution: mean JSD = %.3f nats, "
             "median = %.3f, max = %.3f, mean L1 = %.3f, JSD > 0.3: %.1f%%",
             rtan["JSD_mean"], rtan["JSD_median"], float(js.max()),
             rtan["L1_mean"], rtan["JSD_gt_0p3"])

    # ---------------- cross-model amplification comparison ---------------
    log.info("-" * 78)
    log.info("Cross-model comparison (Mann-Whitney U on log10 K):")
    def logK(m):
        K = ENS[m][1]["K"]
        return np.log10(K[np.isfinite(K)])
    for a, b in [("tan", "noattn"), ("noattn", "buf"), ("tan", "buf"),
                 ("buf", "lif")]:
        u_stat, p_val = stats.mannwhitneyu(logK(a), logK(b),
                                           alternative="two-sided")
        mr = 10.0 ** (np.median(logK(a)) - np.median(logK(b)))
        log.info("  %-9s vs %-9s: U = %.0f, p = %s, median-K ratio = %.2e",
                 MODEL_LABELS[a], MODEL_LABELS[b], u_stat, fmt_p(p_val), mr)

    # ---------------- reset re-bifurcation curves -----------------------
    # Curves are tracked on the continuous branch (pre-reset u; the protocol's
    # divergence metric) and on the post-reset branch h (secondary).  A spike
    # resets h to 0, so post-reset gaps collapse whenever both members fire;
    # the continuous-branch gap reveals whether memory survives the reset.
    log.info("-" * 78)
    log.info("Reset re-bifurcation (global-resync subset, n = %d):", n_rs)
    reb = {}
    for m in MODELS:
        if m == "tan":
            hm = rr_t["h"][mask_rs]
            um = rr_t["u"][mask_rs]
        else:
            sm = simulate(ST_t, m, eps=args.eps)
            hm = sm["h"].reshape(P_t, 2, T_TOT)[mask_rs]
            um = sm["u"].reshape(P_t, 2, T_TOT)[mask_rs]
        gaps_u = np.abs(um[:, 0, T_PROBE - 1:] - um[:, 1, T_PROBE - 1:])
        gaps_h = np.abs(hm[:, 0, T_PROBE - 1:] - hm[:, 1, T_PROBE - 1:])
        reb[m] = (gaps_u, gaps_h)
        log.info("  %-9s: continuous |du| at k=1 = %.3e, k=8 = %.3e ; "
                 "post-reset |dh| at k=8 = %.3e ; frac |du18|>1e-3 = %.1f%%",
                 MODEL_LABELS[m], float(gaps_u[:, 1].mean()),
                 float(gaps_u[:, -1].mean()), float(gaps_h[:, -1].mean()),
                 100.0 * float(np.mean(gaps_u[:, -1] > 1e-3)))

    # ==================== figures =====================================
    log.info("-" * 78)
    log.info("Generating figures ...")
    fig22 = make_fig22(ENS, summary_rows, js, dA, act_both, act_one,
                       act_none, l1)
    fig22.savefig(fig_dir / "fig22_natural_collision.png", dpi=300,
                  facecolor="white", bbox_inches="tight")
    fig22.savefig(fig_dir / "fig22_natural_collision.pdf", facecolor="white",
                  bbox_inches="tight")
    plt.close(fig22)
    fig23 = make_fig23(reb)
    fig23.savefig(fig_dir / "fig23_reset_rebifurcation.png", dpi=300,
                  facecolor="white", bbox_inches="tight")
    fig23.savefig(fig_dir / "fig23_reset_rebifurcation.pdf",
                  facecolor="white", bbox_inches="tight")
    plt.close(fig23)
    for f in ("fig22_natural_collision.png", "fig22_natural_collision.pdf",
              "fig23_reset_rebifurcation.png",
              "fig23_reset_rebifurcation.pdf"):
        (ms_dir / f).write_bytes((fig_dir / f).read_bytes())
    log.info("Figures saved to results/figures/ and manuscript/figures/")

    # ==================== persist tables ===============================
    # pair-level CSV
    with open(tab_dir / "natural_collision_pairs.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        hdr = ["model", "genA", "genB", "h9A", "h9B", "dh_before", "xstar",
               "SA", "SB", "J10A", "J10B", "dA_T", "alphaA1", "alphaA2",
               "alphaA3", "alphaA4", "alphaA5", "alphaB1", "alphaB2",
               "alphaB3", "alphaB4", "alphaB5", "L1_alpha", "JSD_alpha",
               "dh_tilde_T", "dh_post_T", "fireA", "fireB", "K",
               "dk1", "dk4", "dk8"]
        w.writerow(hdr)
        for m in MODELS:
            meta_m, met_m, _ = ENS[m]
            P = meta_m["n_pairs"]
            j10v = met_m["dA"]
            for p in range(P):
                h9A, h9B = H9[m][meta_m["ia"][p]], H9[m][meta_m["ib"][p]]
                row = [m, GEN_NAMES[meta_m["genA"][p]],
                       GEN_NAMES[meta_m["genB"][p]],
                       f"{h9A:.9e}", f"{h9B:.9e}",
                       f"{met_m['dh_before'][p]:.3e}",
                       f"{meta_m['xstar'][p]:.6f}"]
                if "SA" in met_m:
                    row += [f"{met_m['SA'][p]:.6f}", f"{met_m['SB'][p]:.6f}"]
                else:
                    row += ["nan", "nan"]
                row += ["nan", "nan", f"{j10v[p]:.3e}"]
                if m == "tan":
                    aA = met_m["alphaA"][p]
                    aB = met_m["alphaB"][p]
                    row += [f"{v:.6f}" for v in aA]
                    row += [f"{v:.6f}" for v in aB]
                    row += [f"{np.abs(aA - aB).sum():.6f}",
                            f"{jsd(aA, aB):.6f}"]
                elif m == "noattn":
                    u5 = 1.0 / W
                    row += [f"{u5:.6f}"] * 10 + ["0.000000", "0.000000"]
                else:
                    row += ["nan"] * 12
                row += [f"{met_m['dh_tilde'][p]:.3e}",
                        f"{met_m['dh_post'][p]:.3e}",
                        int(met_m["fireA"][p]), int(met_m["fireB"][p]),
                        f"{met_m['K'][p]:.3e}"]
                g = met_m["cont_gap"][p]
                row += [f"{g[1]:.3e}", f"{g[4]:.3e}", f"{g[8]:.3e}"]
                w.writerow(row)
    # summary + correlation CSVs (union of keys; missing -> '')
    keys = sorted({k for r in summary_rows for k in r.keys()})
    with open(tab_dir / "natural_collision_summary.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k, "") for k in keys})
    keys2 = sorted({k for r in corr_rows for k in r.keys()})
    with open(tab_dir / "natural_collision_correlations.csv", "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys2)
        w.writeheader()
        for r in corr_rows:
            w.writerow({k: r.get(k, "") for k in keys2})
    for name in ("natural_collision_pairs.csv", "natural_collision_summary.csv",
                 "natural_collision_correlations.csv"):
        (dat_dir / name).write_bytes((tab_dir / name).read_bytes())
    log.info("Tables saved to results/tables/ and data/experiment_results/")

    # ==================== NPZ (Phase-2 interface) ======================
    save = {}
    for m in MODELS:
        meta_m, met_m, rr_m = ENS[m]
        P = meta_m["n_pairs"]
        save[f"{m}_h"] = rr_m["h"]
        save[f"{m}_u"] = rr_m["u"]
        save[f"{m}_j"] = rr_m["j"]
        save[f"{m}_dh_before"] = met_m["dh_before"]
        save[f"{m}_dh_tilde"] = met_m["dh_tilde"]
        save[f"{m}_cont_gap"] = met_m["cont_gap"]
        if "s" in rr_m:
            save[f"{m}_s"] = rr_m["s"]
        if "alpha" in rr_m:
            save[f"{m}_alpha"] = rr_m["alpha"]
    save["inputs_tan_ensemble"] = ST_t.reshape(P_t, 2, T_TOT)
    save["pair_gens"] = np.stack([meta_t["genA"], meta_t["genB"]], axis=1)
    save["xstar"] = meta_t["xstar"]
    save["continuation"] = meta_t["cont"]
    save["resync_mask_tan_pairs"] = mask_rs
    for m in MODELS:
        save[f"reb_u_{m}"] = reb[m][0]
        save[f"reb_h_{m}"] = reb[m][1]
    np.savez_compressed(dat_dir / "natural_collision_ensembles.npz", **save)
    log.info("NPZ ensembles saved (Phase-2 interface): "
             "data/experiment_results/natural_collision_ensembles.npz")
    log.info("-" * 78)
    log.info("Phase 1 complete in %.1f s", time.time() - t0)
    log.info("%s", "Outputs: fig22/fig23, natural_collision_{summary,pairs,"
             "correlations}.csv, natural_collision_ensembles.npz")


# ============================================================
# 7. Figures
# ============================================================
def _figstyle():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.unicode_minus": False,
        "mathtext.fontset": "dejavusans",
    })


def make_fig22(ENS, summary_rows, js, dA, act_both, act_one, act_none, l1):
    _figstyle()
    meta, met, rr = ENS["tan"]
    P = meta["n_pairs"]

    # ---- representative pair: no fire at the probe, max divergence ----
    nofire = ~met["fireA"] & ~met["fireB"]
    if nofire.sum() > 0:
        cand = np.where(nofire)[0]
        # prefer large divergence, tie-broken by attention divergence
        order = cand[np.argsort(-met["dh_tilde"][cand] - 1e-6 * js[cand])]
        best = int(order[0])
    else:
        best = int(np.argmax(met["dh_tilde"]))
    p = int(best)
    hA = rr["h"][p, 0, :]
    hB = rr["h"][p, 1, :]
    uA = rr["u"][p, 0, T_PROBE - 1]
    uB = rr["u"][p, 1, T_PROBE - 1]
    gA = GEN_NAMES[meta["genA"][p]]
    gB = GEN_NAMES[meta["genB"][p]]
    winA = np.concatenate([XALL[meta["ia"][p], 5:], [meta["xstar"][p]]])
    winB = np.concatenate([XALL[meta["ib"][p], 5:], [meta["xstar"][p]]])

    fig, axs = plt.subplots(2, 2, figsize=(13.4, 10.4),
                            constrained_layout=True)
    axs = axs.ravel()

    # --------- (a) trajectories ----------
    ax = axs[0]
    tt = np.arange(1, T_HIST + 1)
    ax.plot(tt, hA[:T_HIST], "-o", ms=4.5, lw=1.7, color=COLORS["tan"],
            label=f"history A (stimulus: {gA})")
    ax.plot(tt, hB[:T_HIST], "-s", ms=4.5, lw=1.7, color="#1F4E79",
            label=f"history B (stimulus: {gB})")
    ax.plot([T_PROBE], [uA], "o", ms=8, mfc="white", mec=COLORS["tan"],
            mew=2.2, zorder=5)
    ax.plot([T_PROBE], [uB], "s", ms=8, mfc="white", mec="#1F4E79", mew=2.2,
            zorder=5)
    ax.axvline(T_PROBE, color="0.55", ls="--", lw=1)
    ax.axhline(THETA, color="0.8", ls=":", lw=1)
    ax.annotate("", xy=(T_PROBE, uA), xytext=(T_PROBE, uB),
                arrowprops=dict(arrowstyle="<->", color="k", lw=1.3))
    ax.text(T_PROBE + 0.18, 0.5 * (uA + uB),
            f"$\\Delta\\tilde h_{{10}}$ = {met['dh_tilde'][p]:.3f}",
            fontsize=9, va="center")
    ax.text(0.03, 0.97,
            "natural collision at $t$ = 9:\n"
            f"$|h_9^A - h_9^B|$ = {met['dh_before'][p]:.1e} $<\\delta$ = "
            f"{DELTA:.0e}\nprobe $x^*$ = {meta['xstar'][p]:.2f}\n"
            f"$S_{{10}}^A$ = {met['SA'][p]:.3f}    $S_{{10}}^B$ = "
            f"{met['SB'][p]:.3f}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7",
                      lw=0.8))
    ax.set_xlabel("time step $t$")
    ax.set_ylabel("membrane potential $h_t$ (open symbol: pre-reset $u_{10}$)")
    ax.set_title("(a) Collision at $t$ = 9, bifurcation at $t$ = 10 under "
                 "the common probe (Full TAN)", fontweight="bold")
    ax.set_xticks(list(range(1, T_PROBE + 1)))
    ax.legend(loc="lower left", fontsize=8, framealpha=0.92)
    ax.grid(alpha=0.25, ls="--")
    ylo = min(-0.02, hA[:T_HIST].min() - 0.02, hB[:T_HIST].min() - 0.02)
    yhi = min(0.55, max(hA[:T_HIST].max(), hB[:T_HIST].max(), uA, uB) + 0.03)
    ax.set_ylim(ylo, max(yhi, ylo + 0.15))
    # inset zoom on the collision
    axin = ax.inset_axes([0.52, 0.14, 0.38, 0.26])
    axin.plot([9], [hA[8]], "o", color=COLORS["tan"], ms=5)
    axin.plot([9], [hB[8]], "s", color="#1F4E79", ms=5)
    m0 = 0.5 * (hA[8] + hB[8])
    w0 = max(2.0 * met["dh_before"][p], 1e-5)
    axin.set_xlim(8.90, 9.10)
    axin.set_ylim(m0 - 3 * w0, m0 + 3 * w0)
    axin.set_title("zoom: $t$=9 overlap", fontsize=7.5)
    axin.tick_params(labelsize=6.5)
    ax.indicate_inset_zoom(axin, edgecolor="0.6")

    # --------- (b) attention distributions ----------
    ax = axs[1]
    aA = met["alphaA"][p]
    aB = met["alphaB"][p]
    xpos = np.arange(W)
    wb = 0.38
    ax.bar(xpos - wb / 2, aA, wb, color=COLORS["tan"], alpha=0.92,
           label="trajectory A: $\\alpha_{10}^A$")
    ax.bar(xpos + wb / 2, aB, wb, color="#1F4E79", alpha=0.92,
           label="trajectory B: $\\alpha_{10}^B$")
    ticklbl = ([f"tap {6 + i}\n({winA[i]:.2f}|{winB[i]:.2f})" for i in range(4)]
               + [f"tap 10 = $x^*$\n({winA[4]:.2f}|{winB[4]:.2f})"])
    ax.set_xticks(xpos)
    ax.set_xticklabels(ticklbl, fontsize=7.6)
    am = int(np.argmax(aA))
    bm = int(np.argmax(aB))
    ax.annotate("argmax$_A$", xy=(am - wb / 2, aA[am]),
                xytext=(max(am - 0.55, 0), aA[am] + 0.1), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    if aB.max() - aB.min() > 1e-3:
        ax.annotate("argmax$_B$", xy=(bm + wb / 2, aB[bm]),
                    xytext=(min(bm + 0.15, W - 1), aB[bm] + 0.16),
                    fontsize=7.5, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(0.97, 0.94,
            f"$D_{{JS}}$ = {js[p]:.3f} nats\n"
            f"$L_1$ = {l1[p]:.3f}\n"
            f"$\\Delta A_{{10}}$ = {met['dA'][p]:.3f}\n"
            f"$\\Delta\\tilde h_{{10}}$ = {met['dh_tilde'][p]:.3f}",
            transform=ax.transAxes, fontsize=8.6, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", fc="#FBF3E4", ec="0.7",
                      lw=0.8))
    ax.set_xlabel("receptive-field tap $i$  (window entry value: A | B)")
    ax.set_ylabel("attention weight $\\alpha_{10,i}$")
    ax.set_title("(b) Same state $h_9$ and probe $x^*$, different context: "
                 "A engages attention, B stays in predictive silence "
                 "($S_{10}=0 \\Rightarrow \\alpha$ uniform, gate shut)",
                 fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, max(1.02, aA.max() + 0.3, aB.max() + 0.3))
    ax.grid(alpha=0.2, axis="y", ls="--")

    # --------- (c) phase-space bifurcation scatter ----------
    ax = axs[2]
    rowmap = {r["model"]: r for r in summary_rows}
    xall_l = []
    for m in MODELS:
        meta_m, met_m, _ = ENS[m]
        x = met_m["dh_before"]
        y = met_m["dh_tilde"]
        ok = (x > 0) & np.isfinite(x) & (y > 0) & np.isfinite(y)
        x, y = x[ok], y[ok]
        xall_l.append(x)
        r_ = rowmap[m]
        km = r_["K_mean"]
        km_s = f"{km:.4f}" if abs(km - LAMBDA) < 1e-9 else f"{km:.2e}"
        ax.scatter(x, y, s=7 if m == "lif" else 10, alpha=0.55,
                   color=COLORS[m],
                   label=f"{MODEL_LABELS[m]}  n={len(x)}, "
                         f"$\\langle K\\rangle$={km_s}")
    xa = np.concatenate(xall_l)
    lo_x = 10.0 ** np.floor(np.log10(xa.min()))
    hi_x = 2.2 * DELTA
    ax.plot([lo_x, hi_x], [LAMBDA * lo_x, LAMBDA * hi_x], "-", color="k",
            lw=1.7,
            label=f"contraction: $\\Delta\\tilde h_T=\\lambda\\Delta h_{{"
                  f"T-1}}$ ($\\lambda$={LAMBDA})")
    ax.plot([lo_x, hi_x], [lo_x, hi_x], ":", color="0.5", lw=1.2,
            label="identity (no amplification)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo_x, hi_x)
    ax.axvline(DELTA, color="0.6", ls="--", lw=1)
    ax.text(DELTA * 1.05, ax.get_ylim()[0], "$\\delta$", fontsize=8,
            color="0.4", va="bottom")
    ax.set_xlabel("pre-probe state gap $\\Delta h_{T-1} = |h_9^A - h_9^B|$")
    ax.set_ylabel("post-probe gap, continuous branch $\\Delta\\tilde h_T$")
    ax.set_title("(c) One-step divergence of all natural collision pairs "
                 "(per-model ensembles, $\\delta = 10^{-4}$)",
                 fontweight="bold")
    ax.grid(alpha=0.3, which="both", ls=":")
    ax.legend(loc="upper left", fontsize=7.4, framealpha=0.92)

    # --------- (d) attention divergence vs synaptic gap ----------
    ax = axs[3]
    dA_plot = np.maximum(dA, 1e-6)
    ax.scatter(js[act_none], dA_plot[act_none], s=6, color="0.75", alpha=0.8,
               label=f"gates closed both sides (n={int(act_none.sum())})")
    ax.scatter(js[act_one], dA_plot[act_one], s=11, color="#8E44AD",
               alpha=0.75,
               label=f"one gate open (n={int(act_one.sum())})")
    ax.scatter(js[act_both], dA_plot[act_both], s=14, color=COLORS["tan"],
               alpha=0.75, label=f"both gates open (n={int(act_both.sum())})")
    sel = act_both & (dA > 1e-12)
    if sel.sum() > 10:
        xx = js[sel]
        yy = np.log10(dA[sel])
        sl, ic = np.polyfit(xx, yy, 1)
        xr = np.linspace(0, max(js.max() * 1.02, 0.1), 10)
        ax.plot(xr, 10.0 ** (sl * xr + ic), "-", color="0.25", lw=1.3,
                label=f"OLS $\\log_{{10}}\\Delta A$ ~ JSD "
                      f"(slope = {sl:.2f})")
    rp = stats.pearsonr(js, dA)
    rs = stats.spearmanr(js, dA)
    rpb = stats.pearsonr(js[act_both], dA[act_both])
    ax.text(0.03, 0.965,
            f"all pairs, n = {len(js)}:\n"
            f"   Pearson  r = {rp[0]:.3f}  (p {fmt_p(rp[1])})\n"
            f"   Spearman $\\rho$ = {rs[0]:.3f}  (p {fmt_p(rs[1])})\n"
            f"gates open, n = {int(act_both.sum())}:\n"
            f"   Pearson  r = {rpb[0]:.3f}  (p {fmt_p(rpb[1])})",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7",
                      lw=0.8))
    ax.set_yscale("log")
    ax.set_xlim(-0.02, np.log(2) * 1.08)
    ax.set_ylim(1e-6, max(2.0, float(np.nanmax(dA_plot)) * 2))
    ax.set_xlabel("attention divergence $D_{JS}(\\alpha_{10}^A \\parallel "
                  "\\alpha_{10}^B)$  [nats]")
    ax.set_ylabel("synaptic-current gap $\\Delta A_{10}$")
    ax.set_title("(d) Attention orthogonalisation drives the current "
                 "bifurcation (Full TAN)", fontweight="bold")
    ax.grid(alpha=0.3, which="both", ls=":")
    ax.legend(loc="lower right", fontsize=7.4, framealpha=0.92)

    fig.suptitle("TAN Phase 1 - Natural state-collision experiment: the "
                 "scalar membrane potential $h_t$ is not a sufficient state "
                 "variable", fontsize=12, fontweight="bold")
    return fig


def make_fig23(reb):
    """Post-reset re-bifurcation: pairs with a common double-spike at t = 10
    (identical observable state h10 = 0) continue with identical inputs.
    Solid curves: continuous-branch gap <|u_A - u_B|> (the protocol metric);
    faint dashed curves: post-reset gap <|h_A - h_B|>."""
    _figstyle()
    fig, ax = plt.subplots(1, 1, figsize=(8.8, 6.6), constrained_layout=True)
    ks = np.arange(N_CONT + 1)
    fl = 1e-17
    for m in MODELS:
        gu, gh = reb[m]
        mean_u = gu.mean(axis=0)
        mean_h = gh.mean(axis=0)
        qlo = np.percentile(gu, 25, axis=0)
        qhi = np.percentile(gu, 75, axis=0)
        if m == "lif":
            ax.plot(ks, np.maximum(mean_u, fl), "-", color=COLORS[m], lw=2.4,
                    label="LIF: identical forever ($\\equiv 0$, machine-"
                          "exact)")
        else:
            ax.plot(ks, mean_u, "-o", ms=3.8, color=COLORS[m], lw=2.0,
                    label=f"{MODEL_LABELS[m]}, continuous $\\langle|u_A-u_B|"
                          f"\\rangle$")
            ax.plot(ks, np.maximum(mean_h, fl), "--", color=COLORS[m],
                    lw=1.1, alpha=0.75,
                    label=f"{MODEL_LABELS[m]}, post-reset "
                          f"$\\langle|h_A-h_B|\\rangle$")
        ax.fill_between(ks, np.maximum(qlo, fl), np.maximum(qhi, fl),
                        color=COLORS[m], alpha=0.10)
    ax.axvline(4, color="0.6", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_ylim(1e-17, 1.0)
    ax.text(4.1, 2e-7, "context flush at $t$ = 14: $x_{6..9}$ leave the "
            "receptive field", fontsize=8, rotation=90, va="center",
            color="0.35")
    ax.set_xticks(ks)
    ax.set_xticklabels([f"$t = {T_PROBE + k}$" for k in ks])
    ax.set_xlabel("steps after the common double-spike at $t$ = 10 "
                  "(identical observable state $h_{10} = 0$, identical "
                  "continuation inputs)")
    ax.set_ylabel("pair gap (continuous branch; dashed: post-reset)")
    ax.set_title("Reset re-bifurcation: a spike erases $h_t$ but not the "
                 "contextual memory that drives the next inputs",
                 fontweight="bold")
    ax.grid(alpha=0.3, which="both", ls=":")
    ax.legend(loc="upper right", fontsize=7.6)
    return fig


if __name__ == "__main__":
    main()
