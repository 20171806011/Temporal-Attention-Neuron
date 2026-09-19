#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_p1a_figure.py — Publication figure for Paper 2, Figure 7:
Address Identifiability and Multi-Candidate Routing Boundary (P1-A).
Generated deterministically from frozen P1-A summary artifacts.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGDIR = Path(__file__).resolve().parents[1] / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "figure.dpi": 300,
})

def make_fig7():
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.6))

    # -----------------------------------------------------------------------
    # Panel (a): Middle-Key Channel Hit Rate on N=3 (Condition C2)
    # -----------------------------------------------------------------------
    ax = axes[0]
    models_a = [
        "Blind",
        "Pos\nScalar",
        "Signed\nScalar",
        "Hist\n4.2-B",
        "Frozen\nRef",
        "1D\nMetric",
        "2D\nVector",
    ]
    rates_a = [33.75, 0.0, 0.0, 36.78, 18.43, 100.0, 100.0]
    colors_a = ["#999999", "#d95f02", "#e41a1c", "#7570b3", "#a6761d", "#2b83ba", "#1b7837"]

    ax.bar(models_a, rates_a, color=colors_a, width=0.62, edgecolor="black", linewidth=0.8)
    ax.axhline(33.33, color="gray", linestyle="--", linewidth=1.0, label="Chance (33.3%)")

    # Annotate 0/6705 failure and 100% hits
    ax.text(2, 5.0, "0/6,705\n(0.0%)", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#b2182b")
    ax.text(5, 102.0, "100%", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#2b83ba")
    ax.text(6, 102.0, "100%", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1b7837")

    ax.set_ylim(-2, 122)
    ax.set_ylabel("Middle-Key Hit Rate (%)")
    ax.set_title("(a) Intermediate-Key Hit Rate ($N=3$, C2)", fontsize=9.5)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=7.5, frameon=True)

    # -----------------------------------------------------------------------
    # Panel (b): Dual-Channel Both-Routed Accuracy across Conditions
    # -----------------------------------------------------------------------
    ax = axes[1]
    conditions = ["C1\n($N$=2)", "C2\n($N$=3)", "C3\n($\\sigma$=.01)", "C4\n($\\sigma$=.05)", "C5\n(OOD-L)", "C6\n(OOD-H)"]
    x = np.arange(len(conditions))
    width = 0.26

    signed_scalar = [52.14, 25.81, 25.96, 25.51, 0.0, 0.0]
    metric_1d = [100.0, 100.0, 98.76, 79.86, 100.0, 100.0]
    vector_2d = [100.0, 100.0, 98.76, 79.86, 100.0, 100.0]

    ax.bar(x - width, signed_scalar, width, label="SignedScalar", color="#e41a1c", edgecolor="black", linewidth=0.7)
    ax.bar(x, metric_1d, width, label="1D Metric Attention", color="#2b83ba", edgecolor="black", linewidth=0.7)
    ax.bar(x + width, vector_2d, width, label="2D Vector-QK ($S^1$)", color="#1b7837", edgecolor="black", linewidth=0.7, alpha=0.85, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=7.5)
    ax.set_ylim(-2, 122)
    ax.set_ylabel("Both-Routed Accuracy (%)")
    ax.set_title("(b) Dual-Channel Routing Accuracy (C1--C6)", fontsize=9.5)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=7.2, frameon=True)

    # -----------------------------------------------------------------------
    # Panel (c): Soft Readout Addition Error vs Hard Routing on C2
    # -----------------------------------------------------------------------
    ax = axes[2]
    models_c = ["Pos\nScalar", "Signed\nScalar", "Hist\n4.2-B", "Blind", "2D\nVector", "1D\nMetric"]
    soft_err = [2.4149, 1.4593, 1.2760, 1.1685, 0.9463, 0.4247]
    hard_err = [2.4772, 1.6230, 1.5077, 1.3335, 0.0000, 0.0000]

    xc = np.arange(len(models_c))
    w = 0.35

    ax.bar(xc - w/2, hard_err, w, label="Hard Add Error", color="#fc8d59", edgecolor="black", linewidth=0.7)
    ax.bar(xc + w/2, soft_err, w, label="Soft Add Error", color="#91bfdb", edgecolor="black", linewidth=0.7)

    ax.set_xticks(xc)
    ax.set_xticklabels(models_c, fontsize=7.5)
    ax.set_ylabel("Mean Addition Error $|\\hat{y} - (v_1+v_2)|$")
    ax.set_title("(c) Hard vs. Soft Readout Error ($N=3$, C2)", fontsize=9.5)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=7.2, frameon=True)

    fig.tight_layout()
    out_pdf = FIGDIR / "fig7_p1a_identifiability.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {out_pdf}")

if __name__ == "__main__":
    make_fig7()
