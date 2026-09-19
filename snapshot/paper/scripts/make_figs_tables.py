"""TAN-I paper artifacts: publication figures + LaTeX table fragments.

READ-ONLY over the frozen archive (no experiment re-runs).  New figure
PDFs are either pure schematics (no data) or bar charts drawn from values
read from frozen result files (natural_collision_summary.csv,
effective_dimension_summary.csv, effective_dimension_velocity_eventvalues.csv)
or transcribed from frozen audit documents (audit_v3_amended/*, Probe-2
audit output) - each transcription is marked with its frozen source.
Reused frozen experiment figures are copied verbatim by the caller.

Outputs: paper/figures/*.pdf (new), paper/tables/tab_*.tex (fragments).
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.titlesize": 9.5, "axes.labelsize": 9,
                     "legend.fontsize": 7.5, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "axes.unicode_minus": False,
                     "mathtext.fontset": "dejavusans"})


# ---------------------------------------------------------------- helpers
def load_csv(name):
    with open(ROOT / "results" / "tables" / name, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ================================================================ FIGURES
# fig1: TAN architecture schematic (no data)
def fig1():
    fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3f8", ec="#33506e", fs=8.4, bold=False):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.1,
                                    color="#222222"))

    ax.text(50, 59, "Temporal Attention Neuron (frozen equations)",
            ha="center", fontsize=11, fontweight="bold")
    box(2, 44, 18, 9, "input stream\n$x_{t-W+1..t}$", fc="#fdf6e3")
    box(24, 44, 20, 9, "window $X_t$,\n$\\mu_t=\\frac{1}{W}\\sum_i x_i$")
    box(48, 44, 22, 9, "surprise\n$S_t=[x_t-\\mu_t-\\varepsilon]_+$",
        fc="#fdecea")
    box(74, 44, 24, 9, "logits\n$E_{t,i}=\\beta W_q W_k\\,S_t\\,x_i$",
        fc="#fdecea")
    box(74, 30, 24, 9, "weights\n$\\alpha_{t,i}=\\mathrm{softmax}(E_{t,i})$")
    box(74, 16, 24, 9, "context\n$C_t=\\sum_i \\alpha_{t,i} W_v x_i$")
    box(48, 16, 22, 9, "gate\n$A_t=\\tanh(S_t)\\,C_t$", fc="#e8f5e9")
    box(24, 16, 20, 9, "membrane\n$h_t=\\lambda h_{t-1}+A_t$",
        fc="#e8f5e9")
    box(2, 16, 18, 9, "spike $y_t$ if\n$h_t>\\theta$, reset $h_t\\leftarrow 0$",
        fc="#fff8e1")
    arrow(20, 48.5, 24, 48.5)
    arrow(44, 48.5, 48, 48.5)
    arrow(70, 48.5, 74, 48.5)
    arrow(86, 44, 86, 39)
    arrow(86, 30, 86, 25)
    arrow(74, 20.5, 70, 20.5)
    arrow(48, 20.5, 44, 20.5)
    arrow(24, 20.5, 20, 20.5)
    # scalar multiplier annotation
    ax.text(86, 52.5, "scalar $S_t$ multiplies every tap", ha="center",
            fontsize=7.6, color="#8b0000", style="italic")
    fig.savefig(FIG / "fig1_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


# fig5: Probe-1 frozen pooled + per-seed (sources: POOLED_PROBE1_V3.md,
# PER_SEED_AUDIT.md values transcribed below from the frozen amended
# audit documents)
def fig5():
    models = ["B1", "B2", "B3", "B4"]
    clean = [0.331, 0.867, 0.375, 0.328]
    dist = [0.340, 0.843, 0.337, 0.389]
    per_seed_dist = {"B1": [0.352, 0.306, 0.361],
                     "B2": [0.847, 0.841, 0.840],
                     "B3": [0.371, 0.355, 0.285],
                     "B4": [0.427, 0.398, 0.342]}
    colors = {"B1": "#7F8C8D", "B2": "#16A085", "B3": "#2E86AB",
              "B4": "#C0392B"}
    fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.9), constrained_layout=True)
    x = np.arange(len(models))
    w = 0.36
    ax = axs[0]
    ax.bar(x - w / 2, clean, w, color=[colors[m] for m in models],
           alpha=0.9, label="clean")
    ax.bar(x + w / 2, dist, w, color=[colors[m] for m in models],
           alpha=0.5, edgecolor="k", lw=0.4, label="distractor")
    ax.axhline(1 / 3, color="0.5", ls=":", lw=1)
    ax.text(3.35, 0.345, "chance", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("balanced 3-class accuracy")
    ax.set_ylim(0.25, 0.95)
    ax.set_title("(a) pooled over three frozen seeds (means of seeds)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, axis="y", ls="--")
    ax = axs[1]
    for i, m in enumerate(models):
        vals = per_seed_dist[m]
        ax.scatter([i] * 3 + [i], vals + [np.mean(vals)],
                   s=[22] * 3 + [60], color=colors[m],
                   marker="o", zorder=3)
    ax.errorbar(x, [np.mean(per_seed_dist[m]) for m in models],
                yerr=[np.std(per_seed_dist[m]) for m in models],
                fmt="none", ecolor="0.3", capsize=2)
    ax.axhline(1 / 3, color="0.5", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("distractor-condition bal3 (per seed)")
    ax.set_ylim(0.2, 0.95)
    ax.set_title("(b) per-seed distractor condition (small dots: seeds; "
                 "large: mean)")
    ax.grid(alpha=0.25, axis="y", ls="--")
    fig.savefig(FIG / "fig5_probe1.pdf", bbox_inches="tight")
    plt.close(fig)


# fig6: Probe-2 identifiability audit (values from frozen
# AUDIT_REPORT.md / AUDIT_OUTPUT.txt; gates in red)
def fig6():
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.5), constrained_layout=True)
    ax = axs[0]
    ba = [0.779, 0.709]
    ci = [(0.724, 0.836), (0.644, 0.775)]
    ax.bar([0, 1], ba, color=["#16A085", "#2E86AB"], alpha=0.85, width=0.55)
    ax.errorbar([0, 1], ba,
                yerr=[[ba[i] - ci[i][0] for i in range(2)],
                      [ci[i][1] - ba[i] for i in range(2)]],
                fmt="none", ecolor="k", capsize=3)
    ax.axhline(0.58, color="#C0392B", ls="--", lw=1.2)
    ax.text(1.02, 0.585, "red line 0.58", fontsize=7, color="#C0392B")
    ax.axhline(0.5, color="0.6", ls=":", lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["B2 buffer\nlinear readout", "B3 no-attention\n"
                        "linear readout"])
    ax.set_ylim(0.4, 0.9)
    ax.set_ylabel("balanced accuracy")
    ax.set_title("(a) storage/integration shortcuts")
    ax.grid(alpha=0.25, axis="y", ls="--")
    ax = axs[1]
    bars = ["hit rate", "contrast $R$"]
    vals = [0.511, 0.034]
    reqs = [0.50, 2.0]
    ax.bar([0], [0.511], color="#C0392B", alpha=0.85, width=0.5)
    ax.errorbar([0], [0.511], yerr=[[0.511 - 0.481], [0.542 - 0.511]],
                fmt="none", ecolor="k", capsize=3)
    ax.axhline(0.5, color="0.5", ls="--", lw=1.1)
    ax.text(0.35, 0.505, "required $\\geq$0.5", fontsize=6.5)
    ax.set_xticks([0])
    ax.set_xticklabels(["attention hit rate\n(random = 0.25)"])
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("fraction")
    ax.set_title("(b) hit-rate gate")
    ax.grid(alpha=0.25, axis="y", ls="--")
    ax = axs[2]
    ax.bar([0, 1], [0.034, 0.0002], color=["#C0392B", "#C0392B"],
           alpha=0.8, width=0.5)
    ax.set_yscale("log")
    ax.axhline(2.0, color="0.5", ls="--", lw=1.1)
    ax.text(0.15, 2.4, "required contrast = 2", fontsize=6.5, rotation=90)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["contrast\n$\\mathbb{E}[\\alpha_{target}]/\\mathbb{E}"
                        "[\\alpha_{dist}]$", "routing JSD\n"
                        "(permutation $p$ = 0.92)"])
    ax.set_ylabel("value (log scale)")
    ax.set_title("(c) contrast and routing gates")
    ax.grid(alpha=0.25, which="both", ls=":")
    fig.savefig(FIG / "fig6_probe2.pdf", bbox_inches="tight")
    plt.close(fig)


# fig7: scalar-kernel mechanism schematic (no data)
def fig7():
    fig, ax = plt.subplots(figsize=(7.6, 4.9), constrained_layout=True)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")
    ax.text(50, 58, "Scalar-kernel mechanism:  $E_{t,i} = c\\,S_t\\,x_i$",
            ha="center", fontsize=11.5, fontweight="bold")

    def box(x, y, w, h, text, fc="#eef3f8", fs=8.2):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="#33506e",
                                   lw=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs)

    box(4, 40, 20, 10, "query $q$\n$x_t(q)\\in\\{3,4\\}$", fc="#fdecea")
    box(28, 40, 18, 10, "surprise\n$S_t=[x_t-\\mu_t]_+$", fc="#fdecea")
    box(50, 40, 34, 10, "common scalar multiplier\n$c=\\beta W_q W_k S_t$",
        fc="#fff3cd")
    for i, (y, xv) in enumerate([(26, "1.25 / 0.75 (A)"),
                                 (16, "1.75 / 2.25 (B)"),
                                 (6, "noise $\\sim\\mathcal{N}(0,.05)$")]):
        ax.add_patch(plt.Rectangle((4, y), 44, 7, fc="#f5f5f5",
                                   ec="0.6", lw=0.7))
        ax.text(26, y + 3.5, f"history tap $x_{{{i+1}}}$ = {xv}",
                ha="center", va="center", fontsize=7.6)
    ax.text(52, 26, "$E_i = c\\,x_i$  $\\Rightarrow$  ranking by $x_i$",
            fontsize=8.6)
    ax.text(52, 18, "$\\arg\\max_i E_i = \\arg\\max_i x_i$",
            fontsize=9.5, fontweight="bold", color="#8b0000")
    ax.text(52, 8.5, "query changes sharpness (|c|), not the winner",
            fontsize=8.0, style="italic")
    for yv in (43.5, 43.5, 43.5):
        ax.annotate("", xy=(28, 45), xytext=(24, 45),
                    arrowprops=dict(arrowstyle="-|>", lw=1.1))
        break
    for y1, y2 in ((40, 27), (40, 17), (40, 7)):
        ax.annotate("", xy=(26, y2 + 7), xytext=(26, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=0.8,
                                    color="0.4"))
    fig.savefig(FIG / "fig7_mechanism.pdf", bbox_inches="tight")
    plt.close(fig)


# fig8: conceptual decomposition (no data)
def fig8():
    fig, ax = plt.subplots(figsize=(8.4, 3.2), constrained_layout=True)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")
    items = [("History dependence", "ESTABLISHED (Phase 1)", "#e8f5e9"),
             ("Attention-like weighting", "ESTABLISHED\n(surprise-conditioned)",
              "#e8f5e9"),
             ("Dynamic amplification", "SUPPORTED (Phase 1/2)", "#fff3cd"),
             ("Query-conditioned routing", "STRUCTURALLY EXCLUDED\n"
              "(scalar kernel)", "#fdecea"),
             ("Semantic binding", "NOT IMPLEMENTED", "#fdecea")]
    w = 17
    for i, (t, s, c) in enumerate(items):
        x0 = 2 + i * 20
        ax.add_patch(plt.Rectangle((x0, 6), w, 14, fc=c, ec="#33506e",
                                   lw=1.1))
        ax.text(x0 + w / 2, 15.5, t, ha="center", va="center",
                fontsize=8.2, fontweight="bold")
        ax.text(x0 + w / 2, 9.5, s, ha="center", va="center", fontsize=6.6)
        if i < 4:
            ax.annotate("", xy=(x0 + w + 1, 13), xytext=(x0 + w - 0.6, 13),
                        arrowprops=dict(arrowstyle="-|>", lw=1.2))
    fig.savefig(FIG / "fig8_summary.pdf", bbox_inches="tight")
    plt.close(fig)


# ================================================================ TABLES
def num(v, fmt="%.4g"):
    try:
        return fmt % float(v)
    except Exception:
        return "--"


def tab_phase1():
    rows = load_csv("natural_collision_summary.csv")
    by = {r["model"]: r for r in rows}
    out = ["\\begin{tabular}{lcccccc}",
           "\\toprule",
           "Model & $n$ pairs & mean $K$ & 95\\% CI & median $K$ & "
           "\\% $K{>}10^3$ & post-reset viol. \\% \\\\",
           "\\midrule"]
    labels = {"lif": "B1 LIF", "buf": "B2 LIF+delay", "noattn":
              "B3 TAN-noAttn", "tan": "B4 Full TAN"}
    for m in ("lif", "buf", "noattn", "tan"):
        r = by[m]
        out.append(f"{labels[m]} & {num(r['n_pairs'], '%.0f')} & "
                   f"{num(r['K_mean'], '%.3e')} & "
                   f"[{num(r['K_mean_ci_lo'], '%.2e')}, "
                   f"{num(r['K_mean_ci_hi'], '%.2e')}] & "
                   f"{num(r['K_median'], '%.2e')} & "
                   f"{num(r['pct_K_gt_1e3'], '%.1f')} & "
                   f"{num(r['postreset_violation'], '%.1f')} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_phase1.tex").write_text("\n".join(out), encoding="utf-8")


def tab_phase2():
    rows = load_csv("effective_dimension_summary.csv")
    vel = load_csv("effective_dimension_velocity_eventvalues.csv")
    vby = {(r["model"], r["frame"]): r for r in vel}
    order = [("B1", "zC"), ("B3", "zC"), ("B3", "zF"), ("B4", "zC"),
             ("B4", "zF"), ("C0", "zU")]
    out = ["\\begin{tabular}{llcccccc}",
           "\\toprule",
           "Model & frame & $d_D^{event}$ & $\\Delta d_D$ & raw/strat/"
           "resid $d_D$ & event logTr (nats) & $\\tau_{e\\text{-fold}}$ "
           "& $\\tau_{1/2}$ \\\\",
           "\\midrule"]
    for m, f in order:
        r = next(x for x in rows if x["model"] == m and x["frame"] == f)
        v = vby[(m, f)]
        out.append(f"{m} & {f} & {num(r['dDe'], '%.3f')} & "
                   f"{num(r['dD_delta'], '%.3f')} & "
                   f"{num(v['dD_raw'], '%.3f')}/{num(v['dD_strat'], '%.3f')}"
                   f"/{num(v['dD_resid'], '%.3f')} & "
                   f"{num(r['ltE'], '%.2f')} & "
                   f"{num(r['tau_efold'], '%.1f')} & "
                   f"{num(r['tau_half'], '%.0f')} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_phase2.tex").write_text("\n".join(out), encoding="utf-8")


def tab_probe1():
    txt = (ROOT / "audit_v3_amended" / "PER_SEED_AUDIT.md").read_text(
        encoding="utf-8")
    out = ["\\begin{tabular}{cccccc}",
           "\\toprule",
           "seed & model & condition & bal3 & coverage status & final "
           "status \\\\",
           "\\midrule"]
    pat = re.compile(r"\|\s*(\d+)\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*"
                     r"([\d.]+)\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*"
                     r"(\w+)\s*\|")
    for line in txt.splitlines():
        m = pat.match(line)
        if m:
            seed, model, cond, bal, sh, cov, fin = m.groups()
            if sh == "PASS" or True:
                out.append(f"{seed} & {model} & {cond} & {bal} & {cov} & "
                           f"{fin} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_probe1.tex").write_text("\n".join(out), encoding="utf-8")


def tab_probe2():
    # values from frozen AUDIT_REPORT.md / AUDIT_OUTPUT.txt (2026-09-04)
    out = ["\\begin{tabular}{lcc}",
           "\\toprule",
           "audit & measured & requirement / gate \\\\",
           "\\midrule",
           "B2 linear readout BA & 0.779 [0.724, 0.836] & $\\leq 0.58$ "
           "(FAIL) \\\\",
           "B3 linear readout BA & 0.709 [0.644, 0.775] & $\\leq 0.58$ "
           "(FAIL) \\\\",
           "attention hit rate & 0.511 [0.481, 0.542] & $\\geq 0.50$ "
           "(FAIL); $q{=}A$: 0.000, $q{=}B$: 1.000 \\\\",
           "target/distractor contrast $R$ & 0.034 & $\\geq 2.0$ (FAIL) "
           "\\\\",
           "query-conditioned JSD & 0.0002 & permutation $p{=}0.92$ "
           "(FAIL) \\\\",
           "counterfactual argmax flip & 0.000 & $>0$ required (FAIL) "
           "\\\\",
           "single-tap conditional bias & up to 0.45--0.60 at B-codes & "
           "structural note \\\\",
           "\\bottomrule",
           "\\end{tabular}"]
    (TAB / "tab_probe2.tex").write_text("\n".join(out), encoding="utf-8")


def tab_params():
    out = ["\\begin{tabular}{lccl}",
           "\\toprule",
           "parameter & value & symbol & note \\\\",
           "\\midrule",
           "window length & 5 & $W$ & receptive field \\\\",
           "leak & 0.5 & $\\lambda$ & membrane decay \\\\",
           "threshold & 0.5 & $\\theta$ & spike threshold \\\\",
           "query weight & 2.0 & $W_q$ & linear map \\\\",
           "key weight & 1.0 & $W_k$ & linear map \\\\",
           "value weight & 1.0 & $W_v$ & linear map \\\\",
           "inverse temperature & 1.0 & $\\beta$ & softmax \\\\",
           "noise tolerance & 0.0 & $\\varepsilon$ & dead zone \\\\",
           "softmax offset & $10^{-9}$ & --- & denominator \\\\",
           "reset & $h_t \\leftarrow 0$ & --- & after spike \\\\",
           "\\bottomrule",
           "\\end{tabular}"]
    (TAB / "tab_params.tex").write_text("\n".join(out), encoding="utf-8")


def tab_claims():
    rows = [
        ("$h_t$ is a sufficient Markov state", "REJECTED", "Phase 1"),
        ("TAN is history-dependent", "ESTABLISHED", "Phase 1"),
        ("higher intrinsic dimension than LIF", "NOT SUPPORTED",
         "Phase 2"),
        ("event-locked geometry reorganises (rank/scale)",
         "SUPPORTED", "Phase 2"),
        ("fixed buffer can outperform TAN", "ESTABLISHED", "Probe 1"),
        ("improved retrieval in content-rich context (vs no-attn)",
         "LIMITED SUPPORT", "Probe 1"),
        ("distractor robustness", "NOT ESTABLISHED", "Probe 1"),
        ("genuine Q--K routing / query-conditioned selection",
         "REJECTED / STRUCTURALLY EXCLUDED", "Probe 2 + theory"),
        ("surprise-conditioned saliency weighting", "ESTABLISHED",
         "theory + all phases"),
    ]
    out = ["\\begin{tabular}{lcc}",
           "\\toprule",
           "claim & status & evidence \\\\",
           "\\midrule"]
    for c, s, e in rows:
        out.append(f"{c} & {s} & {e} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "tab_claims.tex").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    fig1()
    fig5()
    fig6()
    fig7()
    fig8()
    tab_params()
    tab_phase1()
    tab_phase2()
    tab_probe1()
    tab_probe2()
    tab_claims()
    print("artifacts written to", FIG, "and", TAB)
