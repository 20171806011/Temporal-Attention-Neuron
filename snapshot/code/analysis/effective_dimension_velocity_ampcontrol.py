"""
Phase 2-Pilot compliance fill-in: velocity amplitude controls (dD)
==================================================================
Frozen-protocol compliance note (user-approved 2026-09-04): the frozen text
requires three amplitude versions per lag.  The main run reported pooled
velocity geometry only (stratified/residualised versions were applied to the
Delta-z state geometry).  This script fills the gap WITHOUT re-running the
experiment: it deterministically rebuilds the identical event ensembles
(same seeds / generator / models / standardisation / event filter as
code/experiments/effective_dimension.py) and computes, for every
(model, frame) item and every tau in [-6, 30]:

    dD_raw   (tau)  = PR(Cov_e v_{e,tau})                (pooled)
    dD_strat (tau)  = PR(pooled-within covariance,       median split on
                         log10 A_e)
    dD_resid (tau)  = PR(covariance of per-coordinate     OLS residual on
                         log10 A_e)

Audit gate: the recomputed pooled dD(tau) must match the persisted
results/tables/effective_dimension_curves.csv (dD column) to machine
precision; event counts must match (374 pooled, 119/132/123 per seed).
No existing artifact is modified; outputs are new files only:
    results/tables/effective_dimension_velocity_ampcontrol.csv
    results/logs/effective_dimension_velocity_control.log
"""
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import effective_dimension as ed

ROOT = Path(__file__).resolve().parents[2]
SEEDS = [20260904, 20260905, 20260906]
T, BURN = 5000, 500
TAUS = np.arange(ed.TAU_MIN, ed.TAU_MAX + 1)
EV = ed.EV_TAUS
MODEL_NAME = ed.MODEL_NAME
ITEMS = [("lif", "zC"), ("noattn", "zC"), ("noattn", "zF"),
         ("tan", "zC"), ("tan", "zF"), ("input", "zU")]
EXPECT_N = 374
EXPECT_SEED = {20260904: 119, 20260905: 132, 20260906: 123}


def pooled_within_pr(X, logA):
    med = float(np.median(logA))
    strata = (logA >= med).astype(int)
    Cw = ed.pooled_within_cov(X, strata)
    n = len(X)
    lam = np.maximum(np.linalg.eigvalsh(Cw), 0.0)
    tr = float(lam.sum())
    if n < ed.MIN_N or not np.isfinite(tr) or tr <= ed.TR_FLOOR:
        return float("nan")
    return float(tr * tr / float((lam ** 2).sum()))


def residualised_pr(X, logA):
    n = len(X)
    Xd = np.column_stack([np.ones(n), logA])
    beta = np.linalg.lstsq(Xd, X, rcond=None)[0]
    res = X - Xd @ beta
    res = res - res.mean(axis=0, keepdims=True)
    Cr = res.T @ res / n
    lam = np.maximum(np.linalg.eigvalsh(Cr), 0.0)
    tr = float(lam.sum())
    if n < ed.MIN_N or not np.isfinite(tr) or tr <= ed.TR_FLOOR:
        return float("nan")
    return float(tr * tr / float((lam ** 2).sum()))


def main():
    log_dir, tab_dir = ROOT / "results" / "logs", ROOT / "results" / "tables"
    log_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(
            log_dir / "effective_dimension_velocity_control.log", mode="w",
            encoding="utf-8"),
            logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("ampctrl")
    t0 = time.time()
    log.info("Velocity amplitude-control fill-in (no experiment re-run)")
    log.info("Seeds %s, T=%d burn=%d; items: %s", SEEDS, T, BURN,
             [(MODEL_NAME[k], f) for k, f in ITEMS])

    # ---- deterministic rebuild of the frozen event ensembles ----
    trials = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        tr = ed.gen_trial(rng, T)
        x, pulse = tr["x"], tr["pulse"]
        ons = tr["onsets"]
        valid = np.zeros(len(ons), dtype=bool)
        for j, te in enumerate(ons):
            if not (te >= BURN + ed.PRE_WIN and
                    te <= T - 1 - ed.TAU_MAX - 1):
                continue
            if pulse[te - ed.PRE_WIN:te].any():
                continue
            valid[j] = True
        te_arr = ons[valid]
        n_exp = EXPECT_SEED[seed]
        assert len(te_arr) == n_exp, (seed, len(te_arr), n_exp)
        sims = {}
        for kind in ("lif", "noattn", "tan"):
            s = ed.simulate_traces(x, kind)
            s["x"] = x
            sims[kind] = s
        frames = {}
        for kind in ("lif", "noattn", "tan"):
            for fn, Z in ed.build_frames(kind, sims[kind], x).items():
                frames[(kind, fn)] = ed.standardise(Z, BURN)[0]
        Zu = np.column_stack([sims["tan"]["mu"], sims["tan"]["S"], x])
        frames[("input", "zU")] = ed.standardise(Zu, BURN)[0]
        trials[seed] = dict(te=te_arr, first_amp=x[te_arr], frames=frames)
        log.info("  seed %d: %d valid events (expected %d)", seed,
                 len(te_arr), n_exp)

    # ---- per-item, per-tau velocity matrices, pooled across seeds ----
    rows = []
    pooled_audit = {}   # (item, tau) -> pooled dD (for audit vs curves.csv)
    for item in ITEMS:
        kind, fn = item
        for tau in TAUS:
            V_all, A_all = [], []
            for seed in SEEDS:
                d = trials[seed]
                Z = d["frames"][(kind, fn)]
                for j, te in enumerate(d["te"]):
                    t1 = te + int(tau)
                    V_all.append(Z[t1 + 1] - Z[t1])
                    A_all.append(d["first_amp"][j])
            V = np.asarray(V_all)
            logA = np.log10(np.maximum(np.asarray(A_all), 1e-9))
            assert len(V) == EXPECT_N
            d_raw, _, _, n = ed.pr_logtr(V)
            d_st = pooled_within_pr(V, logA)
            d_re = residualised_pr(V, logA)
            pooled_audit[(item, int(tau))] = d_raw
            rows.append(dict(model=MODEL_NAME[kind], frame=fn,
                             tau=int(tau), n=n,
                             dD_raw=d_raw, dD_strat=d_st, dD_resid=d_re))
            if int(tau) in (-1, 0, 1, 2, 4, 6, 8, 12):
                log.info("  %s %s tau=%3d: raw %.3f | strat %.3f | resid "
                         "%.3f", MODEL_NAME[kind], fn, int(tau),
                         d_raw if np.isfinite(d_raw) else float("nan"),
                         d_st if np.isfinite(d_st) else float("nan"),
                         d_re if np.isfinite(d_re) else float("nan"))

    # ---- audit gate: pooled dD must reproduce the persisted curves ----
    curves = {}
    with open(tab_dir / "effective_dimension_curves.csv", encoding="utf-8") \
            as fh:
        for r in csv.DictReader(fh):
            curves[(r["model"], r["frame"], int(r["tau"]))] = float(r["dD"])
    worst = 0.0
    for (item, tau), d_raw in pooled_audit.items():
        key = (MODEL_NAME[item[0]], item[1], tau)
        expect = curves.get(key)
        if expect is None or (not np.isfinite(expect) and
                              not np.isfinite(d_raw)):
            continue
        worst = max(worst, float(abs(d_raw - expect)))
    log.info("Audit gate: max |recomputed pooled dD - persisted curves| "
             "= %.3e", worst)
    assert worst < 1e-9, "audit gate FAIL - ensembles differ from main run"

    # ---- event values (mean tau in {0,1,2}) per version ----
    def ev_mean(rows_, model, frame, col):
        vals = [r[col] for r in rows_
                if r["model"] == model and r["frame"] == frame and
                r["tau"] in EV and np.isfinite(r[col])]
        return float(np.mean(vals)) if vals else float("nan")

    log.info("-" * 78)
    log.info("Event velocity effective rank (mean over tau in %s):", EV)
    log.info("%s", "%-10s %8s %10s %10s %10s" %
             ("item", "frame", "dD_raw", "dD_strat", "dD_resid"))
    ev_rows = []
    for item in ITEMS:
        name = MODEL_NAME[item[0]]
        raw = ev_mean(rows, name, item[1], "dD_raw")
        st = ev_mean(rows, name, item[1], "dD_strat")
        re = ev_mean(rows, name, item[1], "dD_resid")
        ev_rows.append(dict(model=name, frame=item[1], dD_raw=raw,
                            dD_strat=st, dD_resid=re))
        log.info("%s", "%-10s %8s %10.3f %10.3f %10.3f" %
                 (name, item[1], raw, st, re))

    b4 = next(r for r in ev_rows if r["model"] == "B4" and
              r["frame"] == "zF")
    b3 = next(r for r in ev_rows if r["model"] == "B3" and
              r["frame"] == "zF")
    for col in ("dD_raw", "dD_strat", "dD_resid"):
        log.info("B4 - B3 (zF, %s) = %+.3f", col, b4[col] - b3[col])

    # ---- event bootstrap CI per version (B4 zF, B3 zF) ----
    log.info("Event bootstrap per version (n=2000)")
    rng_b = np.random.default_rng(777)
    for item in (("tan", "zF"), ("noattn", "zF")):
        kind, fn = item
        mats = {}
        for t in EV:
            V_all = []
            for seed in SEEDS:
                d = trials[seed]
                Z = d["frames"][(kind, fn)]
                for te in d["te"]:
                    V_all.append(Z[te + t + 1] - Z[te + t])
            mats[t] = np.asarray(V_all)
        logA = np.log10(np.maximum(np.asarray(
            [a for seed in SEEDS for a in trials[seed]["first_amp"]]), 1e-9))
        nb = len(logA)
        cis = {}
        for col, fnc in (("dD_raw", lambda V, lA: ed.pr_logtr(V)[0]),
                         ("dD_strat", pooled_within_pr),
                         ("dD_resid", residualised_pr)):
            boot = []
            for _ in range(2000):
                ix = rng_b.integers(0, nb, size=nb)
                boot.append(np.mean([fnc(mats[t][ix], logA[ix])
                                     for t in EV]))
            boot = np.asarray(boot)
            cis[col] = (float(np.nanpercentile(boot, 2.5)),
                        float(np.nanpercentile(boot, 97.5)))
        log.info("  %s %s: raw CI [%.3f, %.3f] | strat CI [%.3f, %.3f] | "
                 "resid CI [%.3f, %.3f]", MODEL_NAME[kind], fn, cis["dD_raw"][0],
                 cis["dD_raw"][1], cis["dD_strat"][0], cis["dD_strat"][1],
                 cis["dD_resid"][0], cis["dD_resid"][1])
        for r in ev_rows:
            if r["model"] == MODEL_NAME[kind] and r["frame"] == fn:
                for col in ("dD_raw", "dD_strat", "dD_resid"):
                    r[f"{col}_ci_lo"], r[f"{col}_ci_hi"] = cis[col]

    # ---- persist (new file only) ----
    with open(tab_dir / "effective_dimension_velocity_ampcontrol.csv", "w",
              newline="", encoding="utf-8") as fh:
        keys = sorted({k for r in rows for k in r.keys()})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(tab_dir / "effective_dimension_velocity_eventvalues.csv", "w",
              newline="", encoding="utf-8") as fh:
        keys = sorted({k for r in ev_rows for k in r.keys()})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in ev_rows:
            w.writerow(r)
    log.info("Saved %s and %s (new files; nothing overwritten)",
             "effective_dimension_velocity_ampcontrol.csv",
             "effective_dimension_velocity_eventvalues.csv")
    log.info("Done in %.1f s", time.time() - t0)


if __name__ == "__main__":
    main()
