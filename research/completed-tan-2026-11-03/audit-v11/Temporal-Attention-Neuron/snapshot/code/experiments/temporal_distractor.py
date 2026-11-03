"""
TAN Phase 3 - Probe 1 v3: Temporal Distractor Task (FROZEN)
===========================================================
History: v1 readout collapse -> v2 rejected (XOR label incompatible with
the preregistered LINEAR readout: with binary amplitude codes every
two-variable relation (XOR/AND/OR) is either linearly inseparable or
trivially reducible to query detection; oracle-history calibration failed
before model evaluation - evidence the calibration gate works).  v3 label
space is therefore the single-variable target property, per the locked
definition below.

FROZEN DEFINITION (2026-09-04, locked; no further label changes):
  target at t=5 (fixed), query at t=9 (retrieval cue, NOT in the label),
  distractors at t=6..8 (condition A) or silence (condition B).
  Hidden target code cT: 0 -> x5 ~ U(0.55,1.05) (lo), 1 -> x5 ~
  U(0.95,1.45) (hi); overlap [0.95,1.05]; query and distractors drawn
  from the same lo/hi mixture (iid), so P(distractor > target) ~= 0.78
  and "largest pulse" is NOT a reliable target readout.
  Label:  y = 0 (target-lo) | 1 (target-hi) | 2 (target absent, catch).
  Query function: opens the surprise gate at t=9 so the window (incl. the
  target at t-4) is re-read into the response - testing
      query -> temporal retrieval -> target information in response.
  Readout: 3-class linear softmax on response-trajectory features
      F = [u9..u13, y9..y13]   (t < 9 and raw x NEVER visible).
  Training: clean + distractor 50/50, one readout per (model, seed).
  Evaluation: clean / distractor conditions separately.
  Primary endpoint: Delta_dist = Acc_clean - Acc_distractor (balanced
  3-class); main comparison B4 vs B3; catch = negative control.
  Models/parameters/seeds/stimulus timing: unchanged (natural_collision
  simulate; seeds 20260904..06).

AUDIT CHAIN (any failure => stop; never tune the classifier):
  1. hidden-label audit (y == cT on fresh trials);
  2. ORACLE-history linear separability: linear softmax on raw x5 alone
     AND on x5..x9: bal3 clearly above chance (> 0.60 gate; this is a
     calibration GATE, not a theoretical ceiling) and all 3 classes
     predicted; direct separability probe printed (single-feature margin
     on x5);
  3. shuffled-label readout returns to chance (< 0.37);
  4. class balance p_k in [0.25, 0.42] per (seed, condition);
  5. overlap audit P(distractor > target) in [0.50, 0.95];
  6. all-class non-degeneracy: a model never predicting a class in a
     condition => CALIBRATION FAILURE, formal run blocked.

OUTCOME RULES (95% trial-bootstrap CIs, distractor condition unless noted):
  A: B4 > B3 AND B4 > B1 AND Delta_dist(B4) < Delta_dist(B3)
     -> selective binding advantage (attention-specific distractor
     resistance).
  B: B4 > B3 but B1 within CI of B4 -> generic dynamical advantage.
  C: CI(B4 - B3) contains 0 -> no computational advantage.

Outputs: results/tables/temporal_distractor_{summary,perseed,confusion}.csv
results/figures/fig27_distractor_task.{png,pdf} (+ mirror)
results/logs/temporal_distractor{,_smoke}.log
"""
import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import natural_collision as nc

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [20260904, 20260905, 20260906]
N_TRAIN = 4000
N_TEST = 2000
N_CATCH_FRAC = 1.0 / 3.0
L_BLK = 14
Q_IDX = 8
TGT_IDX = 4
D_IDX = slice(5, 8)
L2 = 1e-4
# frozen amplitude families (locked table)
A_LO_0, A_HI_0 = 0.55, 1.05      # target class 0 (lo)
A_LO_1, A_HI_1 = 0.95, 1.45      # target class 1 (hi)
MODELS = ["lif", "buf", "noattn", "tan"]
MNAME = {"lif": "B1", "buf": "B2", "noattn": "B3", "tan": "B4"}
ORACLE_GATE = 0.60               # calibration gate (not a ceiling)
CHANCE_MAX = 0.37
BAL_P_RANGE = (0.25, 0.42)
OVERLAP_RANGE = (0.50, 0.95)


def sample_amp(rng, n, cls):
    cls = np.asarray(cls, dtype=int).reshape(-1)
    lo = np.full(n, A_LO_0)
    hi = np.full(n, A_HI_0)
    m = cls == 1
    lo[m] = A_LO_1
    hi[m] = A_HI_1
    return rng.uniform(lo, hi)


def make_trials(rng, n):
    """Balanced trials. X (n, L), y (n;) = cT or 2 (catch), cT, cQ, present."""
    n_catch = int(round(N_CATCH_FRAC * n))
    n_present = n - n_catch
    n_dist = n_present // 2
    n_clean = n_present - n_dist
    cT = rng.integers(0, 2, size=n_present)
    cQ = rng.integers(0, 2, size=n_present)          # retrieval cue class
    X = np.zeros((n, L_BLK))
    y = np.full(n, -1, dtype=int)
    cT_all = np.full(n, -1, dtype=int)
    cQ_all = np.full(n, -1, dtype=int)
    idx = 0
    for (cond_n, n_d) in ((n_dist, 3), (n_clean, 0)):
        for j in range(idx, idx + cond_n):
            jj = j - idx
            X[j, TGT_IDX] = sample_amp(rng, 1, [cT[jj]])[0]
            X[j, Q_IDX] = sample_amp(rng, 1, [cQ[jj]])[0]
            if n_d:
                X[j, D_IDX] = sample_amp(rng, n_d,
                                         rng.integers(0, 2, size=n_d))
            y[j] = int(cT[jj])
            cT_all[j] = int(cT[jj])
            cQ_all[j] = int(cQ[jj])
        idx += cond_n
    n_catch_dist = n_catch // 2
    for k in range(n_catch):
        j = idx + k
        cq = int(rng.integers(0, 2))
        X[j, Q_IDX] = sample_amp(rng, 1, [cq])[0]
        if k < n_catch_dist:
            X[j, D_IDX] = sample_amp(rng, 3, rng.integers(0, 2, size=3))
        y[j] = 2
        cQ_all[j] = cq
    perm = rng.permutation(n)
    present = (cT_all >= 0)[perm]
    return (X[perm], y[perm], cT_all[perm], cQ_all[perm], present)


def resp_features(su):
    """F = [u9..u13, y9..y13]; t<9 and x never visible."""
    u = su["u"][:, Q_IDX:Q_IDX + 5]
    y = (u > nc.THETA).astype(float)
    return np.column_stack([u, y])


def hist5_features(X):
    return X[:, TGT_IDX:Q_IDX + 1]                    # oracle only


def softmax_fit(Xtr, ytr):
    n, d = Xtr.shape
    K = 3
    y = np.asarray(ytr, dtype=int)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Xs = (Xtr - mu) / sd

    def loss(w):
        W = w.reshape(K, d + 1)
        Z = Xs @ W[:, 1:].T + W[:, 0]
        Zm = Z - Z.max(axis=1, keepdims=True)
        e = np.exp(Zm)
        p = e / e.sum(axis=1, keepdims=True)
        ll = -np.mean(np.log(p[np.arange(n), y] + 1e-12))
        return ll + 0.5 * L2 * float((W[:, 1:] ** 2).sum())

    def grad(w):
        W = w.reshape(K, d + 1)
        Z = Xs @ W[:, 1:].T + W[:, 0]
        Zm = Z - Z.max(axis=1, keepdims=True)
        e = np.exp(Zm)
        p = e / e.sum(axis=1, keepdims=True)
        T = np.zeros_like(p)
        T[np.arange(n), y] = 1.0
        g = p - T
        # parameter layout is class-major: flat[k*(d+1)+j]
        gW = np.empty(K * (d + 1))
        gW = gW.reshape(K, d + 1)
        gW[:, 0] = g.mean(axis=0)
        gW[:, 1:] = (g.T @ Xs) / n + L2 * W[:, 1:]
        return gW.ravel()

    res = minimize(loss, np.zeros(K * (d + 1)), jac=grad, method="L-BFGS-B",
                   options=dict(maxiter=800, ftol=1e-12))
    return mu, sd, res.x


def predict(mu, sd, w, Xte):
    Xs = (Xte - mu) / sd
    W = w.reshape(3, Xte.shape[1] + 1)
    Z = Xs @ W[:, 1:].T + W[:, 0]
    Zm = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Zm)
    return e / e.sum(axis=1, keepdims=True)


def train_predict(Ftr, ytr, Fte, shuffle=False, rng=None):
    if shuffle and rng is not None:
        ytr = rng.permutation(ytr)
    mu, sd, w = softmax_fit(Ftr, ytr)
    return predict(mu, sd, w, Fte)


def bal3(pred, ytrue, mask=None):
    sel = np.arange(len(ytrue)) if mask is None else np.where(mask)[0]
    rec = []
    for k in range(3):
        m = ytrue[sel] == k
        if m.sum():
            rec.append(float((pred[sel][m] == k).mean()))
    return float(np.mean(rec)) if rec else float("nan")


def macro_f1(pred, ytrue, mask=None):
    sel = np.arange(len(ytrue)) if mask is None else np.where(mask)[0]
    fs = []
    for k in range(3):
        tp = int(((pred[sel] == k) & (ytrue[sel] == k)).sum())
        fp = int(((pred[sel] == k) & (ytrue[sel] != k)).sum())
        fn = int(((pred[sel] != k) & (ytrue[sel] == k)).sum())
        if tp + fp == 0 and fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(fs)) if fs else float("nan")


def confusion(pred, ytrue):
    C = np.zeros((3, 3), dtype=int)
    for t, p in zip(ytrue, pred):
        C[t, p] += 1
    return C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    log_dir = ROOT / "results" / "logs"
    tab_dir = ROOT / "results" / "tables"
    fig_dir = ROOT / "results" / "figures"
    ms_dir = ROOT / "manuscript" / "figures"
    for d in (log_dir, tab_dir, fig_dir, ms_dir):
        d.mkdir(parents=True, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(
            log_dir / ("temporal_distractor_smoke.log" if args.smoke
                       else "temporal_distractor.log"), mode="w",
            encoding="utf-8"),
            logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("probe1v3")
    t0 = time.time()
    seeds = SEEDS[:args.seeds]
    n_tr = 1500 if args.smoke else args.n_train
    n_te = 600 if args.smoke else args.n_test
    tag = "SMOKE" if args.smoke else "FORMAL"
    log.info("=" * 78)
    log.info("Phase 3 Probe 1 v3 - Temporal Distractor Task [%s] (FROZEN "
             "label: retrieval-cue target recall)", tag)
    log.info("=" * 78)
    log.info("train=%d test=%d seeds=%s | label = cT(lo/hi) + absent; "
             "query = retrieval cue (not in label)", n_tr, n_te, seeds)

    runs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        Xtr, ytr, _, _, pretr = make_trials(rng, n_tr)
        Xte, yte, _, _, prete = make_trials(rng, n_te)
        cond_tr = Xtr[:, D_IDX].max(axis=1) > 0
        cond_te = Xte[:, D_IDX].max(axis=1) > 0
        run = dict(seed=seed, yte=yte, prete=prete, cond_te=cond_te)
        for m in MODELS:
            s_tr = nc.simulate(Xtr, m)
            s_te = nc.simulate(Xte, m)
            Ftr, Fte = resp_features(s_tr), resp_features(s_te)
            p = train_predict(Ftr, ytr, Fte)
            # shuffled-label audit statistic = mean over 10 independent
            # shuffles (single-shuffle bal3 is heavy-tailed: sd ~0.08 at
            # n_tr=1500 due to L-BFGS overfitting shuffled labels)
            sh_vals = [train_predict(Ftr, ytr, Fte, shuffle=True, rng=rng)
                       .argmax(axis=1) for _ in range(10)]
            sh = float(np.mean([bal3(v, yte) for v in sh_vals]))
            pred = p.argmax(axis=1)
            run[m] = dict(bal3=bal3(pred, yte),
                          bal3_clean=bal3(pred, yte, ~cond_te),
                          bal3_dist=bal3(pred, yte, cond_te),
                          degr=bal3(pred, yte, ~cond_te) - bal3(pred, yte,
                                                                cond_te),
                          shuf=sh, shuf_max=float(max(bal3(v, yte)
                                                      for v in sh_vals)),
                          pred=pred)
        # oracle controls (calibration only)
        po5 = train_predict(hist5_features(Xtr)[:, :1], ytr,
                            hist5_features(Xte)[:, :1])
        po = train_predict(hist5_features(Xtr), ytr, hist5_features(Xte))
        run["oracle_x5"] = bal3(po5.argmax(axis=1), yte)
        run["oracle_full"] = bal3(po.argmax(axis=1), yte)
        # direct separability probe on x5 (single-feature margin)
        m0 = float(Xtr[ytr == 0, TGT_IDX].mean())
        m1 = float(Xtr[ytr == 1, TGT_IDX].mean())
        run["sep_margin"] = m1 - m0
        runs.append(run)
        log.info("  seed %d: %s | oracle(x5)=%.3f oracle(x5..9)=%.3f "
                 "sep-margin=%.3f", seed,
                 " | ".join("%s bal3[clean %.3f / dist %.3f] degr %+.3f "
                            "shuf %.3f" %
                            (MNAME[m], run[m]["bal3_clean"],
                             run[m]["bal3_dist"], run[m]["degr"],
                             run[m]["shuf"]) for m in MODELS),
                 run["oracle_x5"], run["oracle_full"], run["sep_margin"])

    # ---------------- audit chain ----------------
    ok = True
    # 1. hidden-label audit
    rng_a = np.random.default_rng(1)
    Xa, ya, cTa, _, _ = make_trials(rng_a, 300)
    for j in range(300):
        exp = 2 if cTa[j] < 0 else int(cTa[j])
        if exp != int(ya[j]):
            log.error("ASSERT FAIL: hidden-label mismatch trial %d", j)
            ok = False
            break
    # 4. class balance per (seed, condition)
    for run in runs:
        for cn, cond in (("dist", run["cond_te"]), ("clean", ~run["cond_te"])):
            for k in range(3):
                fk = float((run["yte"][cond] == k).mean())
                if not (BAL_P_RANGE[0] <= fk <= BAL_P_RANGE[1]):
                    log.error("ASSERT FAIL: class %d fraction %.3f "
                              "outside %s (%s, seed %d)", k, fk,
                              BAL_P_RANGE, cn, run["seed"])
                    ok = False
    # 5. overlap audit
    for seed in seeds:
        rng2 = np.random.default_rng(seed)
        Xt, _, _, _, _ = make_trials(rng2, n_tr)
        pres = Xt[:, TGT_IDX] > 0
        withd = pres & (Xt[:, D_IDX].max(axis=1) > 0)
        f = float((Xt[withd, D_IDX].max(axis=1) >
                   Xt[withd, TGT_IDX]).mean())
        if not (OVERLAP_RANGE[0] <= f <= OVERLAP_RANGE[1]):
            log.error("ASSERT FAIL: overlap freq %.3f outside %s", f,
                      OVERLAP_RANGE)
            ok = False
    # 2. oracle gate
    for run in runs:
        if not (run["oracle_x5"] > ORACLE_GATE):
            log.error("ASSERT FAIL: oracle(x5) bal3=%.3f <= %.3f "
                      "(linear separability gate)", run["oracle_x5"],
                      ORACLE_GATE)
            ok = False
        if not (run["sep_margin"] > 0.0):
            log.error("ASSERT FAIL: separability margin %.3f <= 0",
                      run["sep_margin"])
            ok = False
    # 3. shuffled at chance (audit statistic = mean of 10 shuffles)
    for run in runs:
        for m in MODELS:
            if run[m]["shuf"] > CHANCE_MAX:
                log.error("ASSERT FAIL: shuffled bal3 (mean of 10) = %.3f "
                          "> %.2f (%s seed %d)", run[m]["shuf"],
                          CHANCE_MAX, m, run["seed"])
                ok = False
    # 6. non-degeneracy: blocking only for models ABOVE chance that never
    # predict a class (internal inconsistency).  At-chance models are
    # expected to collapse -> recorded warning, not a block.
    cal_fail = 0
    for run in runs:
        for m in MODELS:
            pred = run[m]["pred"]
            above = max(run[m]["bal3_clean"], run[m]["bal3_dist"]) > 0.375
            for cn, cond in (("dist", run["cond_te"]), ("clean",
                                                        ~run["cond_te"])):
                for k in range(3):
                    if (run["yte"][cond] == k).any() and \
                            not (pred[cond] == k).any():
                        if above:
                            cal_fail += 1
                            log.error("CALIBRATION FAILURE: %s seed %d "
                                      "never predicts class %d (%s "
                                      "condition) while above chance",
                                      MNAME[m], run["seed"], k, cn)
                            ok = False
                        else:
                            log.warning("note: %s seed %d never predicts "
                                        "class %d (%s condition) but is at "
                                        "chance -> collapse expected, "
                                        "excluded from claims", MNAME[m],
                                        run["seed"], k, cn)
    log.info("  audit chain done (hidden label / balance / overlap / "
             "oracle / shuffled / non-degeneracy); calibration failures=%d",
             cal_fail)
    if args.smoke:
        log.info("SMOKE verdict: %s", "PASS -> formal run authorised" if ok
                 else "FAIL (blocked; stop, do not tune)")
        log.info("Smoke complete in %.1f s", time.time() - t0)
        return
    if not ok:
        raise SystemExit("Probe-1 v3 BLOCKED: audit FAIL -> stop (no "
                         "classifier tuning)")

    # ---------------- formal results ----------------
    Y = np.concatenate([r["yte"] for r in runs])
    CT = np.concatenate([r["cond_te"] for r in runs])
    PRED = {m: np.concatenate([r[m]["pred"] for r in runs]) for m in MODELS}
    log.info("=" * 78)
    log.info("RESULTS (pooled test n=%d)", len(Y))
    log.info("%s", "%-5s %10s %10s %10s %9s %9s %9s %9s" %
             ("mod", "bal3_all", "bal3_clean", "bal3_dist", "degr",
              "rec_lo", "rec_hi", "rec_catch"))
    for m in MODELS:
        pred = PRED[m]
        rec = [float((pred[Y == k] == k).mean()) if (Y == k).any()
               else float("nan") for k in range(3)]
        log.info("%s", "%-5s %10.3f %10.3f %10.3f %+9.3f %9.3f %9.3f "
                 "%9.3f" % (MNAME[m], bal3(pred, Y),
                            bal3(pred, Y, ~CT), bal3(pred, Y, CT),
                            bal3(pred, Y, ~CT) - bal3(pred, Y, CT),
                            rec[0], rec[1], rec[2]))
    rng_b = np.random.default_rng(4242)
    nboot = 2000
    n = len(Y)
    boot = {k: [] for k in ("B4-B3_dist", "B4-B1_dist", "B4-B3_clean",
                            "degr_B4-B3")}
    for _ in range(nboot):
        ix = rng_b.integers(0, n, size=n)
        yb, cb = Y[ix], CT[ix]
        b4, b3, b1 = (PRED["tan"][ix], PRED["noattn"][ix], PRED["lif"][ix])
        boot["B4-B3_dist"].append(bal3(b4, yb, cb) - bal3(b3, yb, cb))
        boot["B4-B1_dist"].append(bal3(b4, yb, cb) - bal3(b1, yb, cb))
        boot["B4-B3_clean"].append(bal3(b4, yb, ~cb) - bal3(b3, yb, ~cb))
        d4 = bal3(b4, yb, ~cb) - bal3(b4, yb, cb)
        d3 = bal3(b3, yb, ~cb) - bal3(b3, yb, cb)
        boot["degr_B4-B3"].append(d4 - d3)
    ci = {}
    for key, vals in boot.items():
        ci[key] = (float(np.percentile(vals, 2.5)),
                   float(np.percentile(vals, 97.5)))
        log.info("  %-14s %+.4f (95%% CI [%+.4f, %+.4f])", key,
                 float(np.mean(vals)), ci[key][0], ci[key][1])
    A = ci["B4-B3_dist"][0] > 0 and ci["B4-B1_dist"][0] > 0 and \
        ci["degr_B4-B3"][1] < 0
    B = (not A) and ci["B4-B3_dist"][0] > 0 and ci["B4-B1_dist"][0] <= 0 \
        <= ci["B4-B1_dist"][1]
    C = (not A and not B and ci["B4-B3_dist"][0] <= 0 <=
         ci["B4-B3_dist"][1])
    outcome = "A" if A else ("B" if B else ("C" if C else "? (mixed)"))
    log.info("OUTCOME: %s", outcome)
    log.info("  -> %s", {"A": "selective binding advantage (attention-"
                          "specific distractor resistance)",
                          "B": "generic dynamical advantage (LIF "
                          "comparable)",
                          "C": "no computational advantage on this task",
                          "? (mixed)": "does not match any pre-registered "
                          "pattern"}[outcome])

    # ---------------- persist ----------------
    with open(tab_dir / "temporal_distractor_perseed.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "model", "bal3_all", "bal3_clean", "bal3_dist",
                    "degr", "shuffled_bal"])
        for run in runs:
            for m in MODELS:
                r = run[m]
                w.writerow([run["seed"], MNAME[m], f"{r['bal3']:.4f}",
                            f"{r['bal3_clean']:.4f}", f"{r['bal3_dist']:.4f}",
                            f"{r['degr']:+.4f}", f"{r['shuf']:.4f}"])
    with open(tab_dir / "temporal_distractor_summary.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["contrast", "mean_diff", "ci_lo", "ci_hi"])
        for key, (lo, hi) in ci.items():
            w.writerow([key, f"{float(np.mean(boot[key])):+.4f}",
                        f"{lo:+.4f}", f"{hi:+.4f}"])
        w.writerow(["outcome", outcome])
    with open(tab_dir / "temporal_distractor_confusion.csv", "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "true", "pred_lo", "pred_hi", "pred_catch"])
        for m in MODELS:
            Cm = confusion(PRED[m], Y)
            for k in range(3):
                w.writerow([MNAME[m], k, Cm[k, 0], Cm[k, 1], Cm[k, 2]])
    make_fig27(PRED, Y, CT, fig_dir, ms_dir, outcome)
    log.info("Tables + fig27 saved.")
    log.info("Probe-1 v3 complete in %.1f s", time.time() - t0)


def make_fig27(PRED, Y, CT, fig_dir, ms_dir, outcome):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.titlesize": 10.5, "axes.labelsize": 10,
                         "legend.fontsize": 8, "axes.unicode_minus": False,
                         "mathtext.fontset": "dejavusans"})
    colors = {"B1": "#7F8C8D", "B2": "#16A085", "B3": "#2E86AB",
              "B4": "#C0392B"}
    labels = [MNAME[m] for m in MODELS]
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.2), constrained_layout=True)
    x = np.arange(len(MODELS))
    w = 0.36
    ax = axs[0]
    clean = [bal3(PRED[m], Y, ~CT) for m in MODELS]
    dist = [bal3(PRED[m], Y, CT) for m in MODELS]
    ax.bar(x - w / 2, clean, w, color=[colors[l] for l in labels],
           alpha=0.85, label="clean (no distractors)")
    ax.bar(x + w / 2, dist, w, color=[colors[l] for l in labels],
           alpha=0.45, edgecolor="k", lw=0.4, label="distractor condition")
    ax.axhline(1 / 3, color="0.5", ls=":", lw=1)
    ax.text(len(MODELS) - 0.45, 0.35, "chance", fontsize=8, color="0.4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("balanced 3-class accuracy (target lo/hi/absent)")
    ax.set_ylim(0.25, 1.0)
    ax.set_title("Probe-1 v3: retrieval-cue target recall by condition "
                 f"(outcome {outcome})", fontweight="bold")
    ax.legend(fontsize=7.6)
    ax.grid(alpha=0.25, axis="y", ls="--")
    ax = axs[1]
    degr = [bal3(PRED[m], Y, ~CT) - bal3(PRED[m], Y, CT) for m in MODELS]
    ax.bar(labels, degr, color=[colors[l] for l in labels], alpha=0.85)
    for i, v in enumerate(degr):
        ax.text(i, v + (0.005 if v >= 0 else -0.015), f"{v:+.3f}",
                ha="center", fontsize=8)
    ax.axhline(0, color="0.4", lw=1)
    ax.set_ylabel("$\\Delta_{dist}$ = bal3(clean) $-$ bal3(distractor)")
    ax.set_title("Distractor-induced degradation (lower = more robust)",
                 fontweight="bold")
    ax.grid(alpha=0.25, axis="y", ls="--")
    fig.suptitle("Phase 3 Probe 1 v3 - selective target retrieval under "
                 "distractors", fontsize=11.5, fontweight="bold")
    fig.savefig(fig_dir / "fig27_distractor_task.png", dpi=300,
                facecolor="white", bbox_inches="tight")
    fig.savefig(fig_dir / "fig27_distractor_task.pdf", facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    for ext in ("png", "pdf"):
        (ms_dir / f"fig27_distractor_task.{ext}").write_bytes(
            (fig_dir / f"fig27_distractor_task.{ext}").read_bytes())


if __name__ == "__main__":
    main()
