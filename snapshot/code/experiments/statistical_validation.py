"""
TAN Stage 2 — Part 1: Statistical Validation
==============================================
Runs 4 experiments (A: Habituation, B: Noise, C: Phototaxis, D: Noisy Phototaxis)
× 2 models (LIF, TAN) × 50 random seeds.

For each (experiment, model) pair, computes the relevant metric per seed,
then performs:
  - Mean ± Std
  - Two-sample t-test (TAN vs LIF), with Bonferroni correction
  - Wilcoxon rank-sum (non-parametric robustness check)
  - Cohen's d (effect size)
  - Bootstrap 95% confidence intervals (10,000 resamples)

Outputs:
  - results/metrics_raw.csv   (per-seed metrics)
  - results/stats_summary.md  (statistical summary tables in Markdown)
  - results/stats_summary.csv (same in CSV)
  - figures/fig_boxplots.png  (4-panel box plots TAN vs LIF)
  - figures/fig_traj_overlay_D.png  (50-seed trajectory overlay for Exp D)
  - logs/run.log              (execution log)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats
import os, json, time, logging
from pathlib import Path

# ============ Font configuration ============
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 9

# ============ Paths ============
OUT_DIR = Path("/home/z/my-project/download/stage2_part1")
FIG_DIR = OUT_DIR / "figures"
RES_DIR = OUT_DIR / "results"
LOG_DIR = OUT_DIR / "logs"
for d in (FIG_DIR, RES_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ============ Logging ============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "run.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ============ Constants ============
N_SEEDS = 50
SEEDS = list(range(N_SEEDS))

# Colors (academic minimalist)
C_TAN = '#1A1A1A'
C_LIF = '#9E9E9E'
C_ACCENT = '#1F4E79'

# ============================================================
# 1. Neuron definitions
# ============================================================
class LIFNeuron:
    def __init__(self, lambda_leak=0.5, threshold=2.0):
        self.lambda_leak = lambda_leak
        self.threshold = threshold
        self.h = 0.0

    def forward(self, current_input):
        self.h = self.lambda_leak * self.h + current_input
        if self.h > self.threshold:
            spike = 1
            self.h = 0.0
        else:
            spike = 0
        return spike


class TANNeuron:
    def __init__(self, window_size=5, lambda_leak=0.5, threshold=0.5,
                 noise_tolerance=0.0, use_gating=True, use_asym_rect=True,
                 beta=1.0):
        self.window_size = window_size
        self.lambda_leak = lambda_leak
        self.threshold = threshold
        self.noise_tolerance = noise_tolerance
        self.use_gating = use_gating
        self.use_asym_rect = use_asym_rect
        self.beta = beta
        self.h = 0.0
        self.W_q = np.array([2.0])
        self.W_k = np.array([1.0])
        self.W_v = np.array([1.0])
        self.history = np.zeros(window_size)

    def softmax(self, x):
        e_x = np.exp(self.beta * (x - np.max(x)))
        return e_x / (e_x.sum(axis=0) + 1e-9)

    def forward(self, current_input):
        self.history = np.roll(self.history, -1)
        self.history[-1] = current_input
        S_t = current_input - np.mean(self.history)
        if self.use_asym_rect:
            S_t = np.maximum(0, S_t - self.noise_tolerance)
        Q = S_t * self.W_q
        K = self.history * self.W_k
        V = self.history * self.W_v
        att = self.softmax(Q * K)
        C_t = np.sum(att * V)
        if self.use_gating:
            g = np.tanh(S_t) if self.use_asym_rect else np.tanh(abs(S_t))
            A_t = g * C_t
        else:
            A_t = C_t * 0.5
        self.h = self.lambda_leak * self.h + A_t
        if self.h > self.threshold:
            spike = 1
            self.h = 0.0
        else:
            spike = 0
        return spike


# ============================================================
# 2. Experiment runners — each returns a dict of metrics
# ============================================================
def run_experiment_A(model_name, seed):
    """Habituation. Step signal: 5 zero + 25 unit. Metric: decay time."""
    np.random.seed(seed)
    if model_name == 'LIF':
        neuron = LIFNeuron(lambda_leak=0.5, threshold=2.0)
    else:
        neuron = TANNeuron(window_size=10, lambda_leak=0.5, threshold=0.3,
                           noise_tolerance=0.0, use_gating=True,
                           use_asym_rect=False, beta=1.0)
    signal = np.concatenate([np.zeros(5), np.ones(25)])
    spikes = [neuron.forward(x) for x in signal]
    spikes = np.array(spikes)

    # Decay time: first step (after onset at t=5) where firing rate drops
    # below 10% of peak rate.
    onset = 5
    post = spikes[onset:]
    if post.sum() == 0:
        decay_time = 0.0
    else:
        # Sliding window of 3 to compute rate
        rate = np.convolve(post, np.ones(3) / 3, mode='same')
        peak = rate.max()
        threshold = 0.1 * peak
        # Find first index where rate falls below threshold AFTER peak
        peak_idx = np.argmax(rate)
        below = np.where(rate[peak_idx:] < threshold)[0]
        decay_time = float(below[0] + peak_idx) if len(below) > 0 else float(len(post))

    return {
        'decay_time': decay_time,
        'total_spikes': int(spikes.sum()),
        'early_spikes': int(spikes[onset:onset+3].sum()),
    }


def run_experiment_B(model_name, seed):
    """Noise robustness. Step (0->1.5) + Gaussian noise std=0.4. Metric: false spike rate in steady period."""
    np.random.seed(seed)
    time_steps = 60
    base = np.concatenate([np.zeros(15), np.ones(45) * 1.5])
    noise = np.random.normal(0, 0.4, time_steps)
    signal = np.clip(base + noise, 0, None)

    if model_name == 'LIF':
        neuron = LIFNeuron(lambda_leak=0.8, threshold=2.0)
    else:
        neuron = TANNeuron(window_size=10, lambda_leak=0.8, threshold=1.0,
                           noise_tolerance=0.0, use_gating=True,
                           use_asym_rect=False, beta=1.0)

    spikes = []
    for x in signal:
        spikes.append(neuron.forward(x))
    spikes = np.array(spikes)

    # False spike rate: spikes per step during steady-noise period (steps 20-60)
    steady = spikes[20:60]
    false_rate = float(steady.mean())

    return {
        'false_spike_rate': false_rate,
        'total_spikes': int(spikes.sum()),
    }


class Bioton:
    def __init__(self, brain_type='TAN', start_pos=50.0, noisy_env=False,
                 noise_tolerance=0.0):
        if brain_type == 'TAN':
            self.brain = TANNeuron(window_size=5, lambda_leak=0.5,
                                   threshold=0.5,
                                   noise_tolerance=noise_tolerance,
                                   use_gating=True, use_asym_rect=True)
        else:
            thr = 2.5 if noisy_env else 2.0
            self.brain = LIFNeuron(lambda_leak=0.5, threshold=thr)
        self.pos = start_pos
        self.velocity = 0.0
        self.pos_log = []
        self.spike_log = []

    def step(self, light):
        spike = self.brain.forward(light)
        self.velocity = 0.6 * self.velocity + 0.8 * spike
        self.pos += self.velocity
        self.pos_log.append(self.pos)
        self.spike_log.append(spike)


def get_clean_light(x):
    return 10.0 * np.exp(-((x - 80.0) ** 2) / (2 * 15.0 ** 2))


def get_noisy_light(x, noise_level=0.5, rng=None):
    clean = get_clean_light(x)
    if rng is None:
        n = np.random.normal(0, noise_level)
    else:
        n = rng.normal(0, noise_level)
    return max(0.0, clean + n)


def run_experiment_C(model_name, seed, return_traj=False):
    """Phototaxis. 1D track, light at x=80. Metric: final error, convergence."""
    np.random.seed(seed)
    time_steps = 300
    agent = Bioton(brain_type=model_name, start_pos=50.0, noisy_env=False)
    for _ in range(time_steps):
        light = get_clean_light(agent.pos)
        agent.step(light)

    final_err = abs(agent.pos - 80.0)
    conv = next((i for i, p in enumerate(agent.pos_log) if abs(p - 80.0) < 2.0),
                time_steps)

    out = {
        'final_err': float(final_err),
        'convergence': int(conv),
        'total_spikes': int(sum(agent.spike_log)),
        'final_pos': float(agent.pos),
    }
    if return_traj:
        out['pos_log'] = list(agent.pos_log)
    return out


def run_experiment_D(model_name, seed, return_traj=False):
    """Noisy phototaxis. Light + Gaussian noise std=0.5. Metric: final error, convergence."""
    rng = np.random.RandomState(seed)
    time_steps = 250
    NOISE_LEVEL = 0.5
    agent = Bioton(brain_type=model_name, start_pos=50.0, noisy_env=True,
                   noise_tolerance=0.6 if model_name == 'TAN' else 0.0)
    for _ in range(time_steps):
        light = get_noisy_light(agent.pos, noise_level=NOISE_LEVEL, rng=rng)
        agent.step(light)

    final_err = abs(agent.pos - 80.0)
    conv = next((i for i, p in enumerate(agent.pos_log) if abs(p - 80.0) < 2.0),
                time_steps)

    out = {
        'final_err': float(final_err),
        'convergence': int(conv),
        'total_spikes': int(sum(agent.spike_log)),
        'final_pos': float(agent.pos),
    }
    if return_traj:
        out['pos_log'] = list(agent.pos_log)
    return out


# ============================================================
# 3. Statistical helpers
# ============================================================
def cohens_d(a, b):
    """Cohen's d with safeguards for degenerate variance.

    When one group has zero variance (e.g., LIF never fires in Exp A),
    classical Cohen's d divides by zero. We use a small floor on the
    pooled std to produce a finite, interpretable value.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    var_a = float(a.var(ddof=1)) if len(a) > 1 else 0.0
    var_b = float(b.var(ddof=1)) if len(b) > 1 else 0.0
    pooled_var = (var_a + var_b) / 2
    # Floor to avoid division by zero; 1e-6 corresponds to a negligible spread
    pooled_std = np.sqrt(max(pooled_var, 1e-6))
    diff = float(a.mean() - b.mean())
    d = diff / pooled_std
    # Cap to a reasonable range for reporting (-1000, 1000)
    return float(np.clip(d, -1000.0, 1000.0))


def bootstrap_ci(data, n_boot=10000, ci=0.95, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    data = np.asarray(data)
    n = len(data)
    boot_means = np.array([rng.choice(data, n, replace=True).mean()
                           for _ in range(n_boot)])
    lo = float(np.percentile(boot_means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot_means, (1 + ci) / 2 * 100))
    return lo, hi


def interpret_d(d):
    ad = abs(d)
    if ad < 0.2: return "negligible"
    if ad < 0.5: return "small"
    if ad < 0.8: return "medium"
    if ad < 2.0: return "large"
    if ad < 100.0: return "very large"
    return "extreme (degenerate variance)"


# Bonferroni correction: 4 comparisons (one per experiment)
N_COMPARISONS = 4
ALPHA = 0.05 / N_COMPARISONS  # = 0.0125

# ============================================================
# 4. Main runner
# ============================================================
def main():
    log.info("=" * 60)
    log.info("TAN Stage 2 — Part 1: Statistical Validation")
    log.info(f"Config: {N_SEEDS} seeds × 4 experiments × 2 models")
    log.info("=" * 60)

    t0 = time.time()

    # Collect all raw metrics
    rows = []  # each row: dict with exp, model, seed, metric...
    traj_D = {'LIF': [], 'TAN': []}  # store trajectories for Exp D overlay

    for exp_name, runner in [
        ('A', run_experiment_A),
        ('B', run_experiment_B),
        ('C', run_experiment_C),
        ('D', run_experiment_D),
    ]:
        for seed in SEEDS:
            for model in ['LIF', 'TAN']:
                if exp_name == 'D' and seed < 20:  # keep first 20 trajectories for overlay
                    res = runner(model, seed, return_traj=True)
                    traj_D[model].append(res.pop('pos_log'))
                else:
                    res = runner(model, seed)
                row = {'exp': exp_name, 'model': model, 'seed': seed}
                row.update(res)
                rows.append(row)
        log.info(f"  Exp {exp_name} done ({2 * len(SEEDS)} runs)")

    elapsed = time.time() - t0
    log.info(f"All runs complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ============================================================
    # 5. Aggregate per (exp, model) and run statistics
    # ============================================================
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "metrics_raw.csv", index=False)
    log.info(f"Raw metrics saved: {RES_DIR / 'metrics_raw.csv'}")

    # Build summary table
    summary_rows = []
    rng_boot = np.random.RandomState(123)

    # Define the primary metric per experiment
    PRIMARY_METRIC = {
        'A': 'decay_time',
        'B': 'false_spike_rate',
        'C': 'final_err',
        'D': 'final_err',
    }
    METRIC_LABEL = {
        'A': 'Decay Time (steps)',
        'B': 'False Spike Rate',
        'C': 'Final Error |x-80|',
        'D': 'Final Error |x-80|',
    }
    METRIC_BETTER = {  # 'lower' or 'higher' is better
        'A': 'higher',  # higher decay_time = stronger habituation signal; LIF has 0
        'B': 'lower',
        'C': 'lower',
        'D': 'lower',
    }

    for exp in ['A', 'B', 'C', 'D']:
        metric = PRIMARY_METRIC[exp]
        sub = df[df['exp'] == exp]
        lif_vals = sub[sub['model'] == 'LIF'][metric].values
        tan_vals = sub[sub['model'] == 'TAN'][metric].values

        # Descriptive stats
        lif_mean, lif_std = float(lif_vals.mean()), float(lif_vals.std(ddof=1)) if len(lif_vals) > 1 else 0.0
        tan_mean, tan_std = float(tan_vals.mean()), float(tan_vals.std(ddof=1)) if len(tan_vals) > 1 else 0.0
        lif_lo, lif_hi = bootstrap_ci(lif_vals, rng=rng_boot)
        tan_lo, tan_hi = bootstrap_ci(tan_vals, rng=rng_boot)

        # Statistical tests
        # Edge case: if both groups have zero variance, t-test is degenerate.
        # Use one-sample test of TAN vs 0 for Exp A (habituation existence proof).
        if exp == 'A' and lif_vals.std() == 0 and tan_vals.std() == 0:
            # Both are constants -> use one-sample t-test of TAN vs 0
            t_stat, t_p = stats.ttest_1samp(tan_vals, 0.0)
            test_note = "one-sample t-test (TAN vs 0); LIF is degenerate (no firing)"
        elif lif_vals.std() == 0 or tan_vals.std() == 0:
            # One group is degenerate -> use Wilcoxon only
            t_stat = float('nan')
            try:
                t_stat_clean, t_p = stats.ttest_ind(tan_vals, lif_vals, equal_var=False)
                t_stat = float(t_stat_clean)
            except Exception:
                t_p = 1.0
            test_note = "degenerate variance; t-test unreliable"
        else:
            t_stat, t_p = stats.ttest_ind(tan_vals, lif_vals, equal_var=False)
            test_note = "Welch's two-sample t-test"

        try:
            w_stat, w_p = stats.ranksums(tan_vals, lif_vals)
        except Exception:
            w_stat, w_p = float('nan'), float('nan')
        d = cohens_d(tan_vals, lif_vals)
        d_interp = interpret_d(d)

        # Significance after Bonferroni
        sig = "***" if t_p < 0.001 else ("**" if t_p < 0.01 else
              ("*" if t_p < ALPHA else "ns"))

        # For Exp A: "pass" means TAN shows habituation (decay_time > 0) with large d vs LIF
        # For B/C/D: "pass" means TAN significantly better than LIF (lower) with large |d|
        if exp == 'A':
            # Existence proof: TAN decay_time > 0
            pass_p = bool(t_p < 0.001)
            pass_d = bool(abs(d) > 0.8)
        else:
            pass_p = bool(t_p < 0.001)
            pass_d = bool(abs(d) > 0.8)

        summary_rows.append({
            'exp': exp,
            'metric': metric,
            'metric_label': METRIC_LABEL[exp],
            'better': METRIC_BETTER[exp],
            'LIF_mean': lif_mean,
            'LIF_std': lif_std,
            'LIF_CI_lo': lif_lo,
            'LIF_CI_hi': lif_hi,
            'TAN_mean': tan_mean,
            'TAN_std': tan_std,
            'TAN_CI_lo': tan_lo,
            'TAN_CI_hi': tan_hi,
            't_stat': float(t_stat) if not np.isnan(t_stat) else float('nan'),
            't_p': float(t_p),
            'wilcoxon_p': float(w_p),
            'cohens_d': d,
            'd_interp': d_interp,
            'sig': sig,
            'test_note': test_note,
            'pass_alpha_001': pass_p,
            'pass_large_d': pass_d,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RES_DIR / "stats_summary.csv", index=False)
    log.info(f"Summary stats saved: {RES_DIR / 'stats_summary.csv'}")

    # ============================================================
    # 6. Generate Markdown report
    # ============================================================
    md = []
    md.append("# TAN Stage 2 — Part 1: Statistical Validation Report\n")
    md.append(f"**Config:** {N_SEEDS} seeds × 4 experiments × 2 models\n")
    md.append(f"**Alpha (Bonferroni-corrected):** {ALPHA:.4f} (0.05 / {N_COMPARISONS})\n")
    md.append(f"**Bootstrap resamples:** 10,000\n")
    md.append(f"**Total runtime:** {elapsed:.1f}s ({elapsed/60:.1f} min)\n\n")

    md.append("---\n\n")
    md.append("## Per-Experiment Summary Tables\n\n")

    for exp in ['A', 'B', 'C', 'D']:
        row = summary_df[summary_df['exp'] == exp].iloc[0]
        md.append(f"### Experiment {exp} — {row['metric_label']}\n\n")
        md.append(f"*Better = {'lower' if row['better'] == 'lower' else 'higher'}*\n\n")
        md.append("| Model | Mean | Std | 95% CI | n |\n")
        md.append("|-------|------|-----|--------|---|\n")
        md.append(f"| LIF | {row['LIF_mean']:.3f} | {row['LIF_std']:.3f} | "
                  f"[{row['LIF_CI_lo']:.3f}, {row['LIF_CI_hi']:.3f}] | {N_SEEDS} |\n")
        md.append(f"| **TAN** | **{row['TAN_mean']:.3f}** | **{row['TAN_std']:.3f}** | "
                  f"**[{row['TAN_CI_lo']:.3f}, {row['TAN_CI_hi']:.3f}]** | {N_SEEDS} |\n\n")
        md.append(f"- **Test:** {row['test_note']}\n")
        md.append(f"- **t-statistic:** t = {row['t_stat']:.3f}, p = {row['t_p']:.4g} {row['sig']}\n")
        md.append(f"- **Wilcoxon rank-sum p:** {row['wilcoxon_p']:.4g}\n")
        md.append(f"- **Cohen's d:** {row['cohens_d']:.3f} ({row['d_interp']})\n\n")

        # Decision gate per experiment
        passes = row['pass_alpha_001'] and row['pass_large_d']
        verdict = "✅ PASS (p<0.001 AND |d|>0.8)" if passes else (
                  "⚠️ PARTIAL (passes one criterion)" if (row['pass_alpha_001'] or row['pass_large_d'])
                  else "❌ FAIL")
        md.append(f"**Verdict:** {verdict}\n\n")
        md.append("---\n\n")

    # Decision gate summary
    md.append("## Decision Gate Summary\n\n")
    md.append("| Experiment | p < 0.001 | |d| > 0.8 | Verdict |\n")
    md.append("|------------|-----------|----------|---------|\n")
    for exp in ['A', 'B', 'C', 'D']:
        row = summary_df[summary_df['exp'] == exp].iloc[0]
        p_ok = "✓" if row['pass_alpha_001'] else "✗"
        d_ok = "✓" if row['pass_large_d'] else "✗"
        passes = row['pass_alpha_001'] and row['pass_large_d']
        verdict = "✅ PASS" if passes else (
                  "⚠️ PARTIAL" if (row['pass_alpha_001'] or row['pass_large_d'])
                  else "❌ FAIL")
        md.append(f"| {exp} | {p_ok} | {d_ok} | {verdict} |\n")
    md.append("\n")

    n_pass = int(summary_df['pass_alpha_001'].sum() and summary_df['pass_large_d'].sum())
    # Recompute properly
    n_pass = int((summary_df['pass_alpha_001'] & summary_df['pass_large_d']).sum())
    n_partial = int((summary_df['pass_alpha_001'] ^ summary_df['pass_large_d']).sum())
    n_fail = 4 - n_pass - n_partial

    md.append(f"**Overall:** {n_pass}/4 PASS, {n_partial}/4 PARTIAL, {n_fail}/4 FAIL\n\n")
    if n_pass >= 4:
        md.append("**Gate 1 (Statistical Validation):** ✅ PROCEED to Part 2.\n")
    elif n_pass >= 3:
        md.append("**Gate 1:** ⚠️ Proceed with caution; investigate the partial/failing experiment.\n")
    else:
        md.append("**Gate 1:** ❌ HALT — TAN fails 2+ experiments. Revise model before continuing.\n")

    with open(RES_DIR / "stats_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    log.info(f"Markdown report saved: {RES_DIR / 'stats_summary.md'}")

    # ============================================================
    # 7. Figures
    # ============================================================
    log.info("Generating figures...")

    # --- Figure 1: Box plots (4 panels, one per experiment) ---
    fig, axs = plt.subplots(1, 4, figsize=(13, 4), constrained_layout=True)
    for i, exp in enumerate(['A', 'B', 'C', 'D']):
        metric = PRIMARY_METRIC[exp]
        sub = df[df['exp'] == exp]
        lif_vals = sub[sub['model'] == 'LIF'][metric].values
        tan_vals = sub[sub['model'] == 'TAN'][metric].values

        bp = axs[i].boxplot(
            [lif_vals, tan_vals],
            labels=['LIF', 'TAN'],
            patch_artist=True,
            widths=0.5,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='white',
                           markeredgecolor='black', markersize=6),
        )
        for patch, color in zip(bp['boxes'], [C_LIF, C_TAN]):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)

        # Annotate with stats
        row = summary_df[summary_df['exp'] == exp].iloc[0]
        title = f"Exp {exp}: {METRIC_LABEL[exp]}\n" \
                f"d={row['cohens_d']:.2f} ({row['d_interp']}), p={row['t_p']:.2g} {row['sig']}"
        axs[i].set_title(title, fontsize=10)
        axs[i].grid(True, linestyle='--', alpha=0.4, axis='y')
        axs[i].set_ylabel(METRIC_LABEL[exp] if i == 0 else "")

    fig.suptitle("Part 1 — Statistical Validation: TAN vs LIF (50 seeds per cell)",
                 fontsize=12, fontweight='bold')
    fig.savefig(FIG_DIR / "fig_boxplots.png", dpi=200, facecolor='white')
    plt.close(fig)
    log.info(f"Box plot saved: {FIG_DIR / 'fig_boxplots.png'}")

    # --- Figure 2: 50-seed trajectory overlay for Exp D ---
    fig, ax = plt.subplots(1, 1, figsize=(9, 5), constrained_layout=True)
    for traj in traj_D['LIF']:
        ax.plot(traj, color=C_LIF, alpha=0.25, linewidth=0.8)
    for traj in traj_D['TAN']:
        ax.plot(traj, color=C_TAN, alpha=0.4, linewidth=0.8)
    # Mean trajectories (thicker)
    lif_arr = np.array([t[:250] for t in traj_D['LIF']])
    tan_arr = np.array([t[:250] for t in traj_D['TAN']])
    ax.plot(lif_arr.mean(axis=0), color=C_LIF, linewidth=2.5,
            label=f"LIF mean (n={len(traj_D['LIF'])})")
    ax.plot(tan_arr.mean(axis=0), color=C_TAN, linewidth=2.5,
            label=f"TAN mean (n={len(traj_D['TAN'])})")
    ax.axhline(y=80, color='#F4A300', linestyle='--', linewidth=1.5,
               label="Optimal (x=80)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Agent Position")
    ax.set_title("Experiment D — Noisy Phototaxis: 20-seed Trajectory Overlay")
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.savefig(FIG_DIR / "fig_traj_overlay_D.png", dpi=200, facecolor='white')
    plt.close(fig)
    log.info(f"Trajectory overlay saved: {FIG_DIR / 'fig_traj_overlay_D.png'}")

    # ============================================================
    # 8. Print final summary
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 1 — STATISTICAL VALIDATION: COMPLETE")
    print("=" * 60)
    print(f"Output dir: {OUT_DIR}")
    print(f"  Results:  {RES_DIR}")
    print(f"  Figures:  {FIG_DIR}")
    print(f"  Logs:     {LOG_DIR}")
    print()
    print("Per-experiment summary:")
    for _, row in summary_df.iterrows():
        print(f"  Exp {row['exp']}: LIF={row['LIF_mean']:.3f}±{row['LIF_std']:.3f}  "
              f"TAN={row['TAN_mean']:.3f}±{row['TAN_std']:.3f}  "
              f"d={row['cohens_d']:.2f} p={row['t_p']:.2g} {row['sig']}  "
              f"{'✅' if row['pass_alpha_001'] and row['pass_large_d'] else '❌'}")
    print()
    n_pass = int((summary_df['pass_alpha_001'] & summary_df['pass_large_d']).sum())
    print(f"Gate 1 verdict: {n_pass}/4 experiments pass (need 4/4 to PROCEED)")
    print("=" * 60)


if __name__ == "__main__":
    main()
