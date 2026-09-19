#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sprint4_2_vector_qk.py
======================
TAN-II Sprint 4.2-B v1 -- vector-QK mechanism isolation (iron gate).

Only the query/key representation changes; window, surprise gate, E-I
topology, membrane dynamics, readout are frozen (see
docs/TANII_SPRINT4_2B_PREREGISTRATION.md; Amendment 2 for the
unnormalized-control correction).

Models:
  M0  scalar TAN (positive query, TAN-I kernel)  E_i = c*S_t*x_i
  M1  scalar unconstrained control (sign s)      E_i = s*c*S_t*x_i
  M2  minimal vector d=2 (PRIMARY)               E_i = c*S_t*u(om*S_t).phi2(x_i)
  M3  d=3,4,8 moment-curve keys, 2-D query        E_i = c*S_t*u.phi_d(x_i)
  M4  E-I scalar (= TAN-II 4.1-C4)
  M5  E-I vector (both branches vector kernels)
  M2u M2 with UNNORMALIZED keys (control; Amendment 2: flip = 0)

All checks in the canonical fixed context [1,0,2,0,Q], Q_A=3.0, Q_B=4.0,
are DETERMINISTIC exact/closed-form assertions against pre-registered
predictions (P1-P13). Any FAIL -> exit 1 (STOP AND DEBUG).

Usage: python sprint4_2_vector_qk.py [--outdir DIR]
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

# ---- frozen heritage constants ----
W = 5
LAMBDA = 0.5
C = 2.0
EPS_E = 0.0
EPS_I = 0.3
TH_SP = 3.0
W_EI, W_IE, W_II = 0.8, 1.0, 0.2
OMEGA = 0.4
OMEGA_SWEEP = [0.2, 0.4, math.pi / 2.0, math.pi]
QA, QB = 3.0, 4.0
HIST = [1.0, 0.0, 2.0, 0.0]      # canonical fixed history (L1)

# ---- frozen pre-registered predictions (computed pre-run, bench script) ----
PRED = dict(
    M2_QA=dict(EA=3.59230429, EB=3.33356156, Eprobe=3.1078406574,
               diff=+0.25874272434782375, winner="A"),
    M2_QB=dict(EA=5.032371, EB=5.18828113, Eprobe=4.9890434862,
               diff=-0.15591013339559723, winner="B"),
    Qstar=3.707105,
    omega_sweep={(0.2,): ("A", "A"), (0.4,): ("A", "B"),
                 (math.pi / 2,): ("B", "B"), (math.pi,): ("A", "B")},
    M3_winners={"3": ("A", "A"), "4": ("A", "A"), "8": ("A", "A")},
    M4_fp=(1.359169371328, 1.639802644044),   # Amendment 3
    M5_fp=(0.892791780297, 1.151763092204),   # Amendment 3
    M5I_dots=(+0.108730, +0.008416),
    M2_fp_spiking=True,
)

CHECKS = []
N_FAIL = 0


def check(name, ok, detail=""):
    global N_FAIL
    if not ok:
        N_FAIL += 1
    CHECKS.append(dict(name=name, status="PASS" if ok else "FAIL",
                       detail=detail))


def softmax(lg):
    m = np.max(lg)
    e = np.exp(lg - m)
    return e / np.sum(e)


def phi(x, d, norm=True):
    p = np.array([x ** k for k in range(1, d + 1)], dtype=float)
    if x == 0.0 or not norm:
        return p if norm else p
    return p / np.linalg.norm(p)


def uhat(theta, d):
    v = np.zeros(d)
    v[0] = math.cos(theta)
    v[1] = math.sin(theta)
    return v


def jsd(p, q):
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    m = 0.5 * (p + q)
    return 0.5 * (kl(p, m) + kl(q, m))


def energies(model, Q, d=2, omega=OMEGA, sign=1.0, norm=True):
    """E/I branch logits/alpha/context/drives for window [1,0,2,0,Q]."""
    v = np.array(HIST + [Q], dtype=float)
    mu = float(v.mean())
    SE = max(0.0, Q - mu - EPS_E)
    SI = max(0.0, Q - mu - EPS_I)

    def key(x):
        return phi(x, d, norm)

    def qvec(S):
        return C * S * uhat(omega * S, d) if S > 0 else np.zeros(d)

    if model == "M0":
        logE = C * SE * v
    elif model == "M1":
        logE = sign * C * SE * v
    elif model in ("M2", "M3", "M5"):
        u = qvec(SE)
        logE = np.array([float(u @ key(x)) if x > 0 else 0.0 for x in v])
    elif model == "M4":
        logE = C * SE * v
    else:
        raise ValueError(model)
    alphaE = softmax(logE) if np.any(logE) else np.full(5, 0.2)
    CE = float(alphaE @ v)
    AE = math.tanh(SE) * CE

    logI = alphaI = CI = AI = None
    if model == "M4":
        logI = C * SI * v
        alphaI = softmax(logI) if np.any(logI) else np.full(5, 0.2)
        CI = float(alphaI @ v)
        AI = math.tanh(SI) * CI
    elif model == "M5":
        ui = qvec(SI)
        logI = (np.array([float(ui @ key(x)) if x > 0 else 0.0 for x in v])
                if SI > 0 else np.zeros(5))
        alphaI = softmax(logI) if np.any(logI) else np.full(5, 0.2)
        CI = float(alphaI @ v)
        AI = math.tanh(SI) * CI
    return dict(v=v, mu=mu, SE=SE, SI=SI, logE=logE, alphaE=alphaE, CE=CE,
                AE=AE, logI=logI, alphaI=alphaI, CI=CI, AI=AI)


def fp_of(model, res):
    if model in ("M4", "M5"):
        return (res["AE"] - W_EI * res["AI"]) / (1.0 - LAMBDA), \
               (W_IE * res["AE"] - W_II * res["AI"]) / (1.0 - LAMBDA)
    return res["AE"] / (1.0 - LAMBDA), None


def winner_history(logE):
    """argmax over history slots {0 (A), 2 (B)}."""
    return "A" if logE[0] > logE[2] else "B"


def run(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    out = {}

    # ================= canonical counterfactual (deterministic) =================
    for model in ("M0", "M1", "M2", "M3", "M4", "M5"):
        for d in ((2,) if model != "M3" else (3, 4, 8)):
            for sg in ((1.0,) if model != "M1" else (1.0, -1.0)):
                r3 = energies(model, QA, d=d, sign=sg)
                r4 = energies(model, QB, d=d, sign=sg)
                w3 = winner_history(r3["logE"])
                w4 = winner_history(r4["logE"])
                fp3, _ = fp_of(model, r3)
                fp4, _ = fp_of(model, r4)
                out.setdefault(model, {})[f"d{d}" + (f"_s{int(sg)}" if
                                     model == "M1" else "")] = dict(
                    w3=w3, w4=w4, fp3=fp3, fp4=fp4,
                    alpha3=r3["alphaE"].tolist(), alpha4=r4["alphaE"].tolist(),
                    probe3=float(r3["alphaE"][4]),
                    probe4=float(r4["alphaE"][4]),
                    pad3=float(r3["alphaE"][1] + r3["alphaE"][3]),
                    pad4=float(r4["alphaE"][1] + r4["alphaE"][3]),
                    jsd=float(jsd(r3["alphaE"], r4["alphaE"])),
                    EA3=float(r3["logE"][0]), EB3=float(r3["logE"][2]),
                    EA4=float(r4["logE"][0]), EB4=float(r4["logE"][2]))

    # ---- P2/P3/P4: M2 canonical margins, winner, probe below winner ----
    m2 = out["M2"]["d2"]
    pA, pB = PRED["M2_QA"], PRED["M2_QB"]
    check("P2: M2 Q=3 winner A, margin +0.258743",
          m2["w3"] == "A" and abs(m2["EA3"] - m2["EB3"] - pA["diff"]) < 1e-9)
    check("P3: M2 Q=4 winner B, margin -0.155910",
          m2["w4"] == "B" and abs(m2["EA4"] - m2["EB4"] - pB["diff"]) < 1e-9)
    r3 = energies("M2", QA)
    r4 = energies("M2", QB)
    probe3 = float(r3["logE"][4])
    probe4 = float(r4["logE"][4])
    check("P4: probe below winner at both queries",
          probe3 < r3["logE"][0] and probe4 < r4["logE"][2])
    check("P4b: probe energies match frozen values",
          abs(probe3 - pA["Eprobe"]) < 1e-9 and abs(probe4 - pB["Eprobe"]) < 1e-9)

    # ---- P5: M0 ----
    m0 = out["M0"]["d2"]
    check("P5: M0 history winner B at both queries (Flip=0)",
          m0["w3"] == "B" and m0["w4"] == "B")
    check("P5b: M0 probe dominates full window (amplitude-code leakage)",
          m0["probe3"] > max(out["M0"]["d2"]["alpha3"][0],
                             out["M0"]["d2"]["alpha3"][2]) and
          m0["probe4"] > max(out["M0"]["d2"]["alpha4"][0],
                             out["M0"]["d2"]["alpha4"][2]))

    # ---- P6: M1 ----
    m1p = out["M1"]["d2_s1"]
    m1n = out["M1"]["d2_s-1"]
    check("P6: M1 winner(+1)=B, winner(-1)=A (degenerate sign flip)",
          m1p["w3"] == "B" and m1n["w3"] == "A" and m1p["w4"] == "B"
          and m1n["w4"] == "A")
    r1n = energies("M1", QA, sign=-1.0)
    check("P6b: M1 s=-1 full-window winner is a pad (energy baseline 0)",
          int(np.argmax(r1n["alphaE"])) in (1, 3))

    # ---- P7: M3 no canonical crossing ----
    for d in (3, 4, 8):
        o = out["M3"][f"d{d}"]
        pr = PRED["M3_winners"][str(d)]
        check(f"P7: M3 d={d} winner (A,A) -> Flip=0",
              o["w3"] == pr[0] and o["w4"] == pr[1])
    check("P13: M3 d=8 at Q=4: B energy near baseline (mass to pads)",
          out["M3"]["d8"]["EB4"] < 0.1 and out["M3"]["d8"]["EA4"] > 2.0)

    # ---- P8: M4 ----
    m4 = out["M4"]["d2"]
    check("P8: M4 history winner B,B (Flip=0); E-I scalar keeps boundary",
          m4["w3"] == "B" and m4["w4"] == "B")
    check("P8b: M4 fp matches 4.1-C4 heritage (subthreshold)",
          abs(m4["fp3"] - PRED["M4_fp"][0]) < 1e-9 and
          abs(m4["fp4"] - PRED["M4_fp"][1]) < 1e-9 and
          m4["fp3"] < TH_SP and m4["fp4"] < TH_SP)

    # ---- P9: M5 ----
    m5 = out["M5"]["d2"]
    r5_3 = energies("M5", QA)
    r5_4 = energies("M5", QB)
    wI3 = winner_history(r5_3["logI"])
    wI4 = winner_history(r5_4["logI"])
    check("P9: M5 E-branch flips A->B (realized by E-I vector model)",
          m5["w3"] == "A" and m5["w4"] == "B")
    check("P9b: M5 I-branch stays A,A (branch divergence)",
          wI3 == "A" and wI4 == "A")
    check("P9c: M5 fp matches frozen values (subthreshold)",
          abs(m5["fp3"] - PRED["M5_fp"][0]) < 1e-9 and
          abs(m5["fp4"] - PRED["M5_fp"][1]) < 1e-9 and
          m5["fp3"] < TH_SP and m5["fp4"] < TH_SP)

    # ---- P10: JSD conjunction ----
    check("P10a: M0 JSD>0 but flip=0 (trap in-TAN)",
          m0["jsd"] > 1e-6 and m0["w3"] == m0["w4"])
    check("P10b: M2 JSD>0 AND flip=1",
          m2["jsd"] > 1e-6 and m2["w3"] != m2["w4"])
    check("P10c: M1 JSD>0 AND flip=1 (degenerate sign)",
          m1p["jsd"] > 1e-6 and m1p["w3"] != m1n["w3"])

    # ---- P11: M2 spiking ----
    check("P11: M2 fp(Q=3),fp(Q=4) >= spike threshold (spiking regime)",
          m2["fp3"] >= TH_SP and m2["fp4"] >= TH_SP)

    # ---- Q* crossing (closed form) ----
    dphi = phi(1.0, 2) - phi(2.0, 2)
    ph0 = math.atan2(dphi[1], dphi[0])
    th_s = ph0 + math.pi / 2.0
    Qstar = (th_s / OMEGA + 0.6) / 0.8
    check("P1: crossing Q* = 3.707105 (closed form)",
          abs(Qstar - PRED["Qstar"]) < 1e-6)

    # ---- P12: omega sweep (M2 canonical winners) ----
    sweep = {}
    for om in OMEGA_SWEEP:
        w = []
        for Q in (QA, QB):
            v = np.array(HIST + [Q])
            mu = float(v.mean())
            S = max(0.0, Q - mu - EPS_E)
            u = uhat(om * S, 2)
            lg = [float(u @ phi(x, 2)) if x > 0 else 0.0 for x in v]
            w.append(winner_history(np.array(lg)))
        sweep[om] = w
        pr = PRED["omega_sweep"][(om,)]
        check(f"P12: omega={om:.4f} winners {w} == prediction {list(pr)}",
              tuple(w) == pr)
    out["omega_sweep"] = {f"{om:.6f}": w for om, w in sweep.items()}

    # ---- Amendment 2: M2 unnormalized control ----
    ru3 = energies("M2", QA, norm=False)
    ru4 = energies("M2", QB, norm=False)
    wu3 = winner_history(ru3["logE"])
    wu4 = winner_history(ru4["logE"])
    check("AM2: unnormalized control winner B,B (Flip=0, exact)",
          wu3 == "B" and wu4 == "B" and ru3["logE"][0] - ru3["logE"][2] < 0
          and ru4["logE"][0] - ru4["logE"][2] < 0)
    out["M2_unnorm"] = dict(w3=wu3, w4=wu4,
                            diff3=float(ru3["logE"][0] - ru3["logE"][2]),
                            diff4=float(ru4["logE"][0] - ru4["logE"][2]))

    # ================= secondary: tuning scan (descriptive) =================
    xs = np.round(np.arange(0.2, 3.00001, 0.05), 10)
    curves = {}
    for model in ("M0", "M2", "M3", "M4", "M5"):
        for d in ((2,) if model != "M3" else (3, 8)):
            fpE = []
            for x in xs:
                r = energies(model, float(x), d=d)
                f, _ = fp_of(model, r)
                fpE.append(f)
            curves[f"{model}_d{d}"] = fpE
    out["tuning_x"] = xs.tolist()
    out["tuning"] = {k: [float(y) for y in v] for k, v in curves.items()}

    # ================= figures =================
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    labels = ["A", "B", "pad1", "pad2", "probe"]
    for qi, (r, title) in enumerate(((r3, "M2 @ Q=3 (winner A)"),
                                     (r4, "M2 @ Q=4 (winner B)"))):
        off = np.array([0, 1, 2, 3, 4]) + qi * 5.2
        ax.bar(off, r["alphaE"], color=["C1", "C2", "0.8", "0.8", "0.4"])
        ax.text(np.mean(off), 1.02, title, ha="center", fontsize=9)
    ax.set_xticks([0, 1, 2, 3, 4, 5.2, 6.2, 7.2, 8.2, 9.2])
    ax.set_xticklabels(labels + labels, fontsize=7)
    ax.set_title("M2 attention (vector QK): winner flips A->B")
    ax.set_ylim(0, 1.1)
    ax = axes[1]
    for qi, (r, title) in enumerate(((energies("M0", QA),
                                      "M0 @ Q=3 (winner B, probe dominant)"),
                                     (energies("M0", QB),
                                      "M0 @ Q=4 (winner B, probe dominant)"))):
        off = np.array([0, 1, 2, 3, 4]) + qi * 5.2
        ax.bar(off, r["alphaE"], color=["C1", "C2", "0.8", "0.8", "0.4"])
        ax.text(np.mean(off), 1.02, title, ha="center", fontsize=9)
    ax.set_xticks([0, 1, 2, 3, 4, 5.2, 6.2, 7.2, 8.2, 9.2])
    ax.set_xticklabels(labels + labels, fontsize=7)
    ax.set_title("M0 attention (scalar): no flip; amplitude-code leakage")
    ax.set_ylim(0, 1.1)
    ax = axes[2]
    mods = ["M0", "M1(+1)", "M2", "M2unnorm", "M3d3", "M4", "M5-E", "M5-I"]
    flips = []
    flips.append(1 if out["M0"]["d2"]["w3"] != out["M0"]["d2"]["w4"] else 0)
    flips.append(1 if m1p["w3"] != m1n["w3"] else 0)
    flips.append(1 if m2["w3"] != m2["w4"] else 0)
    flips.append(1 if wu3 != wu4 else 0)
    flips.append(1 if out["M3"]["d3"]["w3"] != out["M3"]["d3"]["w4"] else 0)
    flips.append(1 if m4["w3"] != m4["w4"] else 0)
    flips.append(1 if m5["w3"] != m5["w4"] else 0)
    flips.append(1 if wI3 != wI4 else 0)
    ax.bar(mods, flips, color=["0.6", "0.6", "C0", "0.8", "0.6", "0.6",
                               "C0", "0.75"])
    ax.set_title("Counterfactual winner flip (canonical history)")
    ax.set_ylim(0, 1.2)
    fig.suptitle("Sprint 4.2-B v1: vector-QK realizes query-conditioned "
                 "winner reordering")
    fig.tight_layout()
    fig.savefig(outdir / "sprint4_2b_canonical.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for k, v in curves.items():
        ax.plot(xs, v, lw=1.8, label=k)
    ax.axhline(TH_SP, color="r", ls=":", lw=1, label="spike threshold")
    ax.set_xlabel("probe x*")
    ax.set_ylabel("h_E* (conditional fp)")
    ax.set_title("Secondary: tuning curves (descriptive; no P1 verdicts)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "sprint4_2b_tuning.png", dpi=150)
    plt.close(fig)

    # ================= outputs =================
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary = dict(prereg="docs/TANII_SPRINT4_2B_PREREGISTRATION.md",
                   amendment2="docs/TANII_SPRINT4_2B_AMENDMENT_2.md",
                   params=dict(W=W, lambda_=LAMBDA, c=C, eps_E=EPS_E,
                               eps_I=EPS_I, th_sp=TH_SP, w_EI=W_EI,
                               w_IE=W_IE, w_II=W_II, omega=OMEGA,
                               qA=QA, qB=QB, hist=HIST),
                   checks=CHECKS, results=out, n_fail=N_FAIL,
                   script_sha256=sha)
    (outdir / "sprint4_2b_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    with open(outdir / "sprint4_2b_checks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "status", "detail"])
        for c in CHECKS:
            w.writerow([c["name"], c["status"], c["detail"]])
    with open(outdir / "sprint4_2b_tuning.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x"] + list(curves.keys()))
        for i, x in enumerate(xs):
            w.writerow([f"{x:.2f}"] + [f"{curves[k][i]:.6f}"
                                       for k in curves])
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default=str(
        Path(__file__).resolve().parents[3] / "results" / "sprint4_2"))
    args = ap.parse_args()
    summary = run(Path(args.outdir))
    print("=" * 92)
    print("TAN-II Sprint 4.2-B v1 canonical audit (deterministic)")
    print("=" * 92)
    for c in summary["checks"]:
        print(f"[{c['status']}] {c['name']}" + (f"  ({c['detail']})"
                                                if c["detail"] else ""))
    print("-" * 92)
    print(f"checks: {len(summary['checks'])} | FAIL: {summary['n_fail']}")
    if summary["n_fail"]:
        print("STOP AND DEBUG (pre-registered stop rule).")
    else:
        print("ALL CANONICAL CHECKS PASS: vector-QK realizes the "
              "counterfactual winner flip on the fixed history "
              "(M2/M5-E), scalar models do not (M0/M4), sign control is "
              "degenerate (M1), unnormalized control has no flip (AM2).")
    print(f"outputs written to: {args.outdir}")
    return 1 if summary["n_fail"] else 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    sys.exit(main())
