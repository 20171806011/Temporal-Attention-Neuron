#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py — Paper 2 publication figures (Fig. 1-6), generated
deterministically from the FROZEN sprint artifacts only. No new data,
no recomputation of experiments: all numbers are read from the frozen
JSON/CSV summaries under results/ (TAN-II sprints 4.1, 4.2, 4.3) and from
the frozen TAN-I values embedded in the manuscript text.

Output: paper2/figures/fig1_taxonomy.pdf ... fig6_ladder.pdf
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = Path(__file__).resolve().parents[1] / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 110,
})


def rpath(*parts):
    return ROOT.joinpath(*parts)


# ---------------------------------------------------------------------------
# Figure 1 — taxonomy + TAN architecture (schematic, frozen formulas)
# ---------------------------------------------------------------------------
def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    ax = axes[0]
    ax.axis("off")
    steps = [
        ("Memory", "history beyond state"),
        ("Saliency", "convex reweighting"),
        ("Routing", "argmax reorder"),
        ("Binding", "(K,V) transfer"),
        ("Composition", "algebraic f"),
    ]
    y = 0.95
    for name, sub in steps:
        ax.add_patch(plt.Rectangle((0.15, y - 0.13), 0.7, 0.11,
                                   facecolor="0.9", edgecolor="k"))
        ax.text(0.5, y - 0.055, name, ha="center", va="center",
                fontsize=11, fontweight="bold")
        ax.text(0.5, y - 0.115, sub, ha="center", va="center", fontsize=7)
        if y < 0.95:
            ax.annotate("", xy=(0.5, y - 0.02), xytext=(0.5, y + 0.13),
                        arrowprops=dict(arrowstyle="-|>", lw=1.2))
        y -= 0.19
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("(a) The five primitives", fontsize=10)

    ax = axes[1]
    ax.axis("off")
    chain = [r"$X_t$", r"$S_t=[x_t-\mu_t-\varepsilon]_+$",
             r"$\alpha_i=\mathrm{softmax}(\beta W_qW_k S_t x_i)$",
             r"$C_t=\sum_i\alpha_i x_i$", r"$A_t=\tanh(S_t)C_t$",
             r"$h_t=\lambda h_{t-1}+A_t$",
             r"$y_t=\Theta(h_t-\theta_{\mathrm{sp}})$"]
    x, y0 = 0.5, 0.98
    for i, s in enumerate(chain):
        ax.text(x, y0 - i * 0.13, s, ha="center", va="center", fontsize=9)
        if i < len(chain) - 1:
            ax.annotate("", xy=(x, y0 - (i + 1) * 0.13 + 0.055),
                        xytext=(x, y0 - i * 0.13 - 0.055),
                        arrowprops=dict(arrowstyle="-|>", lw=1.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("(b) The frozen TAN computation chain", fontsize=10)

    ax = axes[2]
    ax.axis("off")
    ax.annotate("", xy=(0.9, 0.5), xytext=(0.1, 0.5),
                arrowprops=dict(arrowstyle="-|>", lw=1.6))
    ax.plot([0.35, 0.65], [0.5, 0.5], "ko", ms=7)
    ax.text(0.35, 0.42, r"$K_A$", ha="center", fontsize=9)
    ax.text(0.65, 0.42, r"$K_B$", ha="center", fontsize=9)
    ax.text(0.5, 0.68, r"$q>0$: order locked",
            ha="center", fontsize=9, color="C0")
    ax.text(0.5, 0.30, r"$\mathrm{argmax}_i\,qK_i=\mathrm{argmax}_i K_i$",
            ha="center", fontsize=9, color="C0")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("(c) Scalar kernel: separable, order-locked", fontsize=10)
    fig.suptitle("Figure 1 — The Conceptual Taxonomy and the TAN "
                 "Architecture", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig1_taxonomy.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — Competition Without Routing (Sprint 4.1 frozen curves)
# ---------------------------------------------------------------------------
def fig2():
    csvp = rpath("results", "sprint4_1cd", "sprint4_1cd_curves.csv")
    rows = list(csv.reader(open(csvp, encoding="utf-8")))
    hdr = rows[0]
    data = rows[1:]
    xs = np.array([float(r[0]) for r in data])
    fp = {m: np.array([float(r[hdr.index(f"{m}_fpE")]) for r in data])
          for m in ("C1", "C2", "C3", "C4")}
    fpi = {m: np.array([float(r[hdr.index(f"{m}_fpI")]) for r in data])
           for m in ("C1", "C2", "C3", "C4")}

    fig = plt.figure(figsize=(13.5, 4.6))
    ax = fig.add_subplot(1, 3, 1)
    for m, ls in (("C1", "-"), ("C2", "--"), ("C3", "-."), ("C4", ":")):
        ax.plot(xs, fp[m], ls, lw=1.8, label=m)
    ax.plot(1.0, 0.2, "r*", ms=14)
    ax.axhline(3.0, color="r", ls=":", lw=1)
    ax.set_xlabel(r"probe $x^*$")
    ax.set_ylabel(r"$h_E^*$ (conditional fp)")
    ax.set_title("(a) C1–C4 tuning curves\n(C1 peak at $x^*=1.00$; "
                 "C3/C4 gain-only)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(1, 3, 2)
    panels = [("C1", 1.4), ("C2", 1.4), ("C3", 1.4),
              ("C4", 0.6), ("C4", 1.4), ("C4", 2.6)]
    for k, (m, xp) in enumerate(panels):
        a = fig.add_subplot(3, 6, 6 + k + 1)
        idx = int(np.argmin(np.abs(xs - xp)))
        fE, fI = fp[m][idx], fpi[m][idx]
        lim = min(max(3.0, abs(fE) * 1.25, abs(fI) * 1.25), 4.5)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            a.annotate("", xy=(fE, fI),
                       xytext=(fE + 0.7 * dx * (fE - (fE - 0.6) * 0),
                               fI + 0.7 * dy * 0.8),
                       arrowprops=dict(arrowstyle="-|>", color="0.6",
                                       lw=0.8))
        a.axvline(fE, color="C0", lw=1.2)
        a.axhline(fI, color="C3", lw=1.2)
        a.plot([fE], [fI], "ko", ms=4)
        a.set_xlim(-0.4, lim)
        a.set_ylim(-0.4, lim)
        a.set_xticks([])
        a.set_yticks([])
        a.set_title(f"{m}@{xp}", fontsize=7)
    fig.text(0.45, 0.02, "(b) Conditional Fixed-Context Phase Planes "
             "(nullclines = constant lines; NOT an autonomous 2-D portrait)",
             ha="center", fontsize=9)

    ax = fig.add_subplot(1, 3, 3)
    layouts = ["L01", "L02", "L03", "L12", "L13", "L23"]
    zeros = [0.0] * 6
    ax.bar(layouts, zeros, color="0.7", edgecolor="k")
    ax.set_ylim(-0.05, 1.0)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.text(2.5, 0.45, "FlipRate ≡ 0.000\n(six layouts × two variants,\n"
            "C3 & C4)", ha="center", fontsize=10, color="C0")
    ax.set_title("(c) Counterfactual flip rate", fontsize=10)
    fig.suptitle("Figure 2 — Competition Without Routing (Sprint 4.1)",
                 y=1.02)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(FIGDIR / "fig2_competition.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — Geometric phase transition (Sprint 4.2-A/B frozen data)
# ---------------------------------------------------------------------------
def fig3():
    # (a) schematic
    fig = plt.figure(figsize=(13.5, 4.4))
    ax = fig.add_subplot(1, 3, 1)
    ax.axis("off")
    ax.annotate("", xy=(0.95, 0.5), xytext=(0.05, 0.5),
                arrowprops=dict(arrowstyle="-|>", lw=1.6))
    ax.plot([0.3], [0.5], "ko", ms=5)
    ax.plot([0.7], [0.5], "ko", ms=5)
    ax.text(0.5, 0.62, r"$S^0$: two points", ha="center", fontsize=9)
    ax.text(0.5, 0.36, r"flip rate $\in\{0,1\}$", ha="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(r"(a) $d=1$: the signed ray", fontsize=10)
    ax = fig.add_subplot(1, 3, 2)
    ax.axis("off")
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), "k-", lw=1.4)
    ax.plot([-1.15, 1.15], [0.0, 0.0], "C0-", lw=1.6)
    ax.annotate("", xy=(0.92, 0.38), xytext=(0.25, 0.25),
                arrowprops=dict(arrowstyle="-|>", color="C3", lw=1.6))
    ax.annotate("", xy=(0.92, -0.38), xytext=(0.25, -0.25),
                arrowprops=dict(arrowstyle="-|>", color="C3", lw=1.6))
    ax.text(0.55, 0.55, r"$\theta$", fontsize=11, color="C3")
    ax.text(0.1, 1.2, r"$\mathcal{H}_0:\,Q^\top\Delta K=0$",
            fontsize=10, color="C0")
    ax.set_xlim(-1.25, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_title(r"(b) $d=2$: hyperplane bisection of $S^1$", fontsize=10)

    # (c) flip law MC vs analytic
    ax = fig.add_subplot(1, 3, 3)
    csvp = rpath("results", "sprint4_2", "sprint4_2_checks.csv")
    rows = list(csv.DictReader(open(csvp, encoding="utf-8")))
    pts = {}
    for r in rows:
        if r["experiment"] == "E3" and r["case"] == "iso":
            d = int(r["d"])
            pts.setdefault(d, {}).setdefault(float(r["theta_pi"]),
                                             []).append(float(r["estimate"]))
    ax.plot([0, 1], [0, 1], "k-", lw=2, label=r"analytic $\theta/\pi$")
    colors = {2: "C0", 3: "C1", 4: "C2", 8: "C3"}
    for d in (2, 3, 4, 8):
        ts = sorted(pts[d])
        ys = [np.mean(pts[d][t]) for t in ts]
        ax.plot(ts, ys, "o", ms=4, color=colors[d], label=f"MC d={d}")
    ax.plot([0, 1], [0, 1], "rv", ms=8,
            label=r"$d=1$ signed: $\{0,1\}$")
    ax.plot([0], [0], "b^", ms=8, label=r"$d=1$ positive: $0$")
    ax.set_xlabel(r"$\theta$ (units of $\pi$)")
    ax.set_ylabel(r"$\mathrm{FlipRate}_{\mathrm{CF}}$")
    ax.set_title(r"(c) $\mathrm{FlipRate}_{\mathrm{CF}}(\theta)=\theta/\pi$"
                 "\n(132/132 MC conditions)", fontsize=10)
    ax.legend(fontsize=7)
    fig.suptitle("Figure 3 — The Geometric Phase Transition in $d=2$ "
                 "(Sprint 4.2-A)", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGDIR / "fig3_phase.pdf")
    plt.close(fig)

    # (d) 4.2-B realization: winner vs query scan
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    summ = json.load(open(rpath("results", "sprint4_2",
                                "sprint4_2b_probe_summary.json"),
                         encoding="utf-8"))
    qg = summ["flip_q_grid"]
    scan = summ["flip_q_scan"]
    for name, ls, lw in (("M2", "-", 2.4), ("M5I", "--", 1.4),
                         ("M0", ":", 1.2)):
        wv = np.array([1.0 if w == "A" else -1.0 for w in scan[name]])
        ax.step(qg, wv, where="post", label=name, lw=lw, ls=ls)
    ax.axvline(3.0, color="k", ls=":", lw=1)
    ax.axvline(4.0, color="k", ls=":", lw=1)
    ax.axvline(3.7071, color="C0", ls="--", lw=1, alpha=0.7)
    ax.text(3.7071, 1.12, r"$Q^*=3.7071$", color="C0", fontsize=8)
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(["B", "A"])
    ax.set_xlabel("query amplitude Q (fixed history [1,0,2,0,Q])")
    ax.set_ylabel("history winner")
    ax.set_title("(d) The realized counterfactual flip (Sprint 4.2-B):\n"
                 r"margins $+0.2587\to-0.1559$", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(-1.3, 1.45)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig3d_realization.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — Binding vs discreteness (Sprint 4.2-C/D frozen data)
# ---------------------------------------------------------------------------
def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
    summ = json.load(open(rpath("results", "sprint4_2",
                                "sprint4_2c_summary.json"),
                         encoding="utf-8"))
    c = summ["canonical"]
    ax = axes[0]
    ax.bar(["Q=3", "Q=4", "swap Q=3", "swap Q=4"],
           [c["T"], 0, c["T_swap"], 0], color="0.75", edgecolor="k")
    ax.axhline(0, color="k", lw=0.8)
    ax.text(0, c["T"] + 0.03, f"T={c['T']:.4f}", ha="center", fontsize=8)
    ax.text(2, c["T_swap"] + 0.03, f"T_swap={c['T_swap']:.4f}",
            ha="center", fontsize=8)
    ax.set_ylabel(r"$C_{\mathrm{ev}}$ transfer")
    ax.set_title("(a) K-V soft binding: value-swap antisymmetry\n"
                 r"$T_{\mathrm{swap}}=-T$ ($10^{-12}$), $D\approx1.7$")
    ax = axes[1]
    dsum = json.load(open(rpath("results", "sprint4_2",
                                "sprint4_2d_summary.json"),
                         encoding="utf-8"))
    lad = dsum["ladder"]["M2_ladder"]
    gs = [1.0, 2.0, 5.0, 10.0, 100.0, 400.0]
    dev3 = [lad[str(g)]["dev"][0] for g in (1.0, 2.0, 5.0, 10.0, 100.0)] + \
        [0.0]
    ax.plot(gs, dev3, "o-", color="C1",
            label=r"$D_{\mathrm{ev}}(Q=3)$ (left axis)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\gamma$ (WTA sharpening)")
    ax.set_ylabel(r"$D$", color="C1")
    ax.tick_params(axis="y", labelcolor="C1")
    ax2 = ax.twinx()
    ax2.axhline(0.444527, color="C2", ls="--", lw=1.4,
                label=r"FlipRate $=0.444527$ flat (right axis)")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("FlipRate", color="C2")
    ax2.tick_params(axis="y", labelcolor="C2")
    ax.set_title("(b) The first orthogonal axis:\n"
                 "routing selectivity $\\perp$ readout discreteness",
                 fontsize=10)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=7,
              loc="center right")
    fig.suptitle("Figure 4 — Decoupling Binding from Discreteness "
                 "(Sprint 4.2-C/D)", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGDIR / "fig4_binding.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — Composition architecture & audit (Sprint 4.3-A frozen data)
# ---------------------------------------------------------------------------
def fig5():
    audit = json.load(open(rpath("results", "sprint4_3",
                                 "audit_summary.json"), encoding="utf-8"))
    canon = audit["canonical"]
    cf = audit["counterfactuals"]

    fig = plt.figure(figsize=(13.5, 5.4))
    ax = fig.add_subplot(2, 3, 1)
    ax.axis("off")
    for i, (name, sub) in enumerate([("M0", "single winner\n(one symbol)"),
                                     ("M1", "parallel binding\n(pair, no f)"),
                                     ("M2", "parallel binding\n+ combiner f")]):
        ax.add_patch(plt.Rectangle((0.08, 0.78 - i * 0.3), 0.84, 0.2,
                                   facecolor="0.9", edgecolor="k"))
        ax.text(0.5, 0.88 - i * 0.3, name, ha="center", va="center",
                fontweight="bold", fontsize=10)
        ax.text(0.5, 0.82 - i * 0.3, sub, ha="center", va="center",
                fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("(a) The M0–M1–M2 ladder", fontsize=10)

    ax = fig.add_subplot(2, 3, 2)
    scen = ["S1", "S2", "S3"]
    em0 = [canon[f"{s}_+"]["errors"]["E_comp_M0"] for s in scen]
    em1 = [canon[f"{s}_+"]["errors"]["E_comp_M1"] for s in scen]
    em2 = [canon[f"{s}_+"]["errors"]["E_comp_M2"] for s in scen]
    x = np.arange(3)
    ax.bar(x - 0.25, em0, 0.22, label="M0", color="0.6")
    ax.bar(x, em1, 0.22, label="M1 (projection)", color="0.75")
    ax.bar(x + 0.25, em2, 0.22, label="M2", color="C0")
    ax.set_xticks(x)
    ax.set_xticklabels(scen)
    ax.set_ylabel(r"$E_{\mathrm{comp}}(+)$")
    ax.set_title("(b) Three-error audit, canonical", fontsize=10)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(2, 3, 3)
    pairs = [("primary", -4.0), ("CF1 query swap", cf["1_query_swap"]
                                 ["y_minus"]), ("CF2 value swap",
                                                cf["2_value_swap"]
                                                ["y_minus"]),
             ("CF3 key swap", cf["3_key_swap"]["y_minus"]),
             ("CF4 channel swap", cf["4_channel_output_swap"]["y_minus"])]
    ax.bar([p[0] for p in pairs], [p[1] for p in pairs], color="C3")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel(r"$y(-)$ under the swap")
    ax.set_title("(c) Subtraction antisymmetry\n"
                 r"(primary $y(-)=-4$; swaps $=+4$)", fontsize=10)
    ax.tick_params(axis="x", rotation=25)

    ax = fig.add_subplot(2, 3, 4)
    am5 = cf["5_am5_trap"]["M0"]
    ax.bar(["M0 E_comp", "M0 E_bind,A", "M2 E_comp", "M2 E_bind,A"],
           [am5["E_comp"], am5["E_bindA"], 0.0, 0.0],
           color=["C3", "C3", "C0", "C0"])
    ax.set_ylabel("error (V_A=0)")
    ax.set_title("(d) AM5 zero-value trap:\njoint audit catches the "
                 "accidental zero", fontsize=10)

    ax = fig.add_subplot(2, 3, 5)
    b = audit["benchmarks"]
    seeds = list(audit["mc"].keys())
    ax.bar([r"$E[E_{\mathrm{comp}}^+]$", r"$E[E_{\mathrm{comp}}^-]$"],
           [b["E_comp_plus"], b["E_comp_minus"]], color="0.7",
           label="benchmark (quadrature)")
    ax.bar([r"$E[E_{\mathrm{comp}}^+]$", r"$E[E_{\mathrm{comp}}^-]$"],
           [np.mean([audit["mc"][s]["E_plus"] for s in seeds]),
            np.mean([audit["mc"][s]["E_minus"] for s in seeds])],
           color="C0", alpha=0.7, label="MC (3 seeds)")
    ax.set_title("(e) Unconditional error = routing error", fontsize=10)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(2, 3, 6)
    ax.bar(["max E|both-routed", "E|both (mean)"],
           [0.0, 0.0], color="C0")
    ax.set_ylim(-0.05, 0.15)
    ax.set_title("(f) Combiner zero-error\n"
                 r"$E_{\mathrm{comp}}|\mathrm{both}=0.0\mathrm{e}0$",
                 fontsize=10)
    fig.suptitle("Figure 5 — The Minimal Composition Architecture and "
                 "Audit (Sprint 4.3-A)", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGDIR / "fig5_composition.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6 — unified ladder
# ---------------------------------------------------------------------------
def fig6():
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.axis("off")
    rungs = [
        ("L0  Memory", "frozen baseline", "---"),
        ("L1  Saliency", "surprise-gated window", "order-locked: FlipRate 0"),
        ("L2  Geometry", "(measurement rung)",
         r"$d_D$ 2.089 vs 1.484; no selectivity"),
        ("L3  Competition", "O1 opponent E-I", r"$\not\Rightarrow$ Routing"),
        ("L4  Routing", "O2 vector Q-K", r"$\theta/\pi$; 0.4425/0.4445"),
        ("L5  Binding", "O3 K-V + O4 WTA", r"$T_{\mathrm{swap}}=-T$; D→0"),
        ("L6  Composition", "O5 parallel + combiner",
         r"$E|\mathrm{both}\equiv0$; 8 CFs"),
    ]
    y = 0.96
    for i, (name, dof, wall) in enumerate(rungs):
        ax.add_patch(plt.Rectangle((0.06, y - 0.115), 0.88, 0.1,
                                   facecolor="0.92", edgecolor="k"))
        ax.text(0.09, y - 0.055, name, fontsize=9, fontweight="bold",
                va="center")
        ax.text(0.32, y - 0.055, dof, fontsize=8, va="center")
        ax.text(0.93, y - 0.055, wall, fontsize=8, va="center",
                ha="right", style="italic")
        if i < len(rungs) - 1:
            ax.annotate("", xy=(0.5, y - 0.125), xytext=(0.5, y + 0.115),
                        arrowprops=dict(arrowstyle="-|>", lw=1.3))
        y -= 0.145
    ax.text(0.5, 0.005, "added organizational degree of freedom "
            "(italic: the boundary wall located at the rung)",
            ha="center", fontsize=8, color="0.35")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Figure 6 — The Unified Mechanistic Ladder of Dynamical "
                 "Computation", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_ladder.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print("figures written to", FIGDIR)
