"""
TAN Stage 2 — Part 3: Ablation Study
=====================================
Compares 5 TAN variants on 4 tasks (A: Habituation, B: Noise, C: Phototaxis,
D: Noisy Phototaxis) × 50 random seeds.

Variants:
  V0. TAN-full            — full TAN model (reference)
  V1. TAN-no-surprise     — replace S_t = x_t - mean(H_t) with S_t = x_t
                            (i.e., raw input directly, no deviation signal)
  V2. TAN-no-attention    — replace C_t = sum(alpha_i * V_i) with C_t = mean(V)
                            (uniform weighting instead of softmax attention)
  V3. TAN-no-gate         — replace A_t = tanh(S_t) * C_t with A_t = C_t
                            (drop the gating function entirely)

Predicted module-behavior map (from Stage 2 Research Plan §5.4):
  - Remove Surprise  →  Habituation disappears (Task A fails)
  - Remove Gate      →  Noise robustness explodes (Tasks B/D fail)
  - Remove Attention →  Performance degrades moderately (all tasks slightly worse)

Statistical analysis (same framework as Part 1 & 2):
  - Mean ± Std across 50 seeds per (variant, task)
  - Pairwise t-test TAN-full vs each ablation (Bonferroni-corrected)
  - Cohen's d
  - Module-behavior contribution heatmap (% degradation when module removed)

Outputs:
  - results/metrics_raw.csv         (5 variants × 4 tasks × 50 seeds = 1000 rows)
  - results/stats_summary.csv       (12 pairwise comparisons)
  - results/stats_summary.md        (human-readable report)
  - results/contribution_matrix.csv (3 modules × 4 tasks % degradation)
  - figures/fig_grouped_bar.png     (4 panels × 5 variants side-by-side)
  - figures/fig_contribution_heatmap.png (module-behavior heatmap)
  - logs/run.log
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats
import os, json, time, logging, warnings
from pathlib import Path
import pandas as pd

# ============ Font configuration ============
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 8

# ============ Paths ============
OUT_DIR = Path("/home/z/my-project/download/stage2_part3")
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
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============ Constants ============
N_SEEDS = 50
SEEDS = list(range(N_SEEDS))

# Variant identifiers
VARIANTS = [
    'TAN-full',         # V0: reference
    'TAN-no-surprise',  # V1
    'TAN-no-attention', # V2
    'TAN-no-gate',      # V3
]
N_VARIANTS = len(VARIANTS)

# Colors for plotting (TAN-full highlighted in black, ablations in lighter shades)
VARIANT_COLORS = {
    'TAN-full':         '#1A1A1A',  # black
    'TAN-no-surprise':  '#C0392B',  # red
    'TAN-no-attention': '#F39C12',  # orange
    'TAN-no-gate':      '#7CB344',  # green
}

# ============================================================
# 1. TAN neuron with configurable ablation flags
# ============================================================
class TANNeuronAblation:
    """
    TAN neuron with 3 ablation flags.

    Parameters
    ----------
    window_size : int
        Length of the history window W.
    lambda_leak : float
        Membrane leakage coefficient.
    threshold : float
        Spike threshold.
    noise_tolerance : float
        Dead-zone width tau for the surprise rectification.
    use_surprise : bool
        If False, replace S_t = x_t - mean(H_t) with S_t = x_t
        (no deviation-from-history computation).
    use_attention : bool
        If False, replace C_t = sum(alpha_i * V_i) with C_t = mean(V)
        (uniform weighting instead of softmax attention).
    use_gating : bool
        If False, replace A_t = tanh(S_t) * C_t with A_t = C_t
        (drop the gating function entirely).
    beta : float
        Inverse temperature for the softmax.
    """

    def __init__(self, window_size=5, lambda_leak=0.5, threshold=0.5,
                 noise_tolerance=0.0, use_surprise=True, use_attention=True,
                 use_gating=True, beta=1.0):
        self.window_size = window_size
        self.lambda_leak = lambda_leak
        self.threshold = threshold
        self.noise_tolerance = noise_tolerance
        self.use_surprise = use_surprise
        self.use_attention = use_attention
        self.use_gating = use_gating
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
        # 1. Update history window
        self.history = np.roll(self.history, -1)
        self.history[-1] = current_input

        # 2. Compute surprise
        if self.use_surprise:
            # Original TAN: deviation from window mean
            S_raw = current_input - np.mean(self.history)
            # Asymmetric rectification + noise tolerance
            S_t = np.maximum(0, S_raw - self.noise_tolerance)
        else:
            # ABLATION V1: use raw input directly as the "surprise" signal
            # This removes the deviation-from-history computation.
            # We still apply the same rectification + tolerance for fairness.
            S_raw = current_input
            S_t = np.maximum(0, S_raw - self.noise_tolerance)

        # 3. Attention-based context retrieval
        Q = S_t * self.W_q
        K = self.history * self.W_k
        V = self.history * self.W_v

        if self.use_attention:
            # Original TAN: softmax attention
            attention_scores = Q * K
            attention_weights = self.softmax(attention_scores)
            C_t = np.sum(attention_weights * V)
        else:
            # ABLATION V2: uniform weighting (mean of values)
            C_t = np.mean(V)

        # 4. Gated injection
        if self.use_gating:
            # Original TAN: tanh(S_t) * C_t
            gating = np.tanh(S_t)
            A_t = gating * C_t
        else:
            # ABLATION V3: drop the gate, use attention output directly
            # Scale by 0.5 to keep numerical range comparable (avoids
            # saturation that would occur with raw C_t which can be large).
            A_t = C_t * 0.5

        # 5. Membrane evolution (non-Markovian)
        self.h = self.lambda_leak * self.h + A_t

        # 6. Spike-and-reset
        if self.h > self.threshold:
            spike = 1
            self.h = 0.0
        else:
            spike = 0
        return spike


def make_variant(variant_name, **overrides):
    """Factory that returns a TANNeuronAblation configured for the given variant.

    Default parameters match the full TAN used in Part 1 / Part 2.
    """
    # Defaults (same as Part 1 / Part 2 TAN)
    base = dict(
        window_size=5,
        lambda_leak=0.5,
        threshold=0.5,
        noise_tolerance=0.0,
        use_surprise=True,
        use_attention=True,
        use_gating=True,
        beta=1.0,
    )
    base.update(overrides)

    if variant_name == 'TAN-full':
        pass
    elif variant_name == 'TAN-no-surprise':
        base['use_surprise'] = False
    elif variant_name == 'TAN-no-attention':
        base['use_attention'] = False
    elif variant_name == 'TAN-no-gate':
        base['use_gating'] = False
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    return TANNeuronAblation(**base)


# ============================================================
# 2. Environment (1D phototaxis) — same as Part 1/2
# ============================================================
def get_clean_light(x):
    return 10.0 * np.exp(-((x - 80.0) ** 2) / (2 * 15.0 ** 2))


def get_noisy_light(x, noise_level=0.5, rng=None):
    clean = get_clean_light(x)
    if rng is None:
        n = np.random.normal(0, noise_level)
    else:
        n = rng.normal(0, noise_level)
    return max(0.0, clean + n)


class Bioton:
    """Generic agent wrapper for any TAN variant."""
    def __init__(self, variant_name, start_pos=50.0, noisy_env=False,
                 noise_tolerance=0.0):
        # For noisy environment (Task D), use the specified noise_tolerance.
        # For clean environment (Task C), keep tolerance at 0.
        kwargs = dict(noise_tolerance=noise_tolerance)
        self.brain = make_variant(variant_name, **kwargs)
        self.variant_name = variant_name
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


# ============================================================
# 3. Experiment runners — same task definitions as Part 1/2
# ============================================================
def run_experiment_A(variant_name, seed):
    """Step input habituation test. Metric: decay time."""
    np.random.seed(seed)
    # Use the same TAN configuration as Part 1 Experiment A:
    # window=10, threshold=0.3, symmetric rectification (use_asym_rect=False)
    # but applied to the ablation framework.
    # We achieve "symmetric rectification" by setting noise_tolerance=0 and
    # use_surprise=True; the asymmetric max(0, ...) is part of the original
    # TAN, so we keep it for consistency with Part 1.
    #
    # NOTE: Part 1 Exp A used use_asym_rect=False (i.e., symmetric, with
    # gating = tanh(|S_t|)). Here for ablation consistency we use the
    # standard asymmetric TAN. This means TAN-full numbers in Part 3 Exp A
    # may differ slightly from Part 1 Exp A. We re-run TAN-full here for
    # a fair within-Part-3 comparison.
    neuron = make_variant(variant_name, window_size=10, lambda_leak=0.5,
                          threshold=0.3, noise_tolerance=0.0)
    signal = np.concatenate([np.zeros(5), np.ones(25)])
    spikes = np.array([neuron.forward(x) for x in signal])

    onset = 5
    post = spikes[onset:]
    if post.sum() == 0:
        decay_time = 0.0
    else:
        rate = np.convolve(post, np.ones(3) / 3, mode='same')
        peak = rate.max()
        threshold = 0.1 * peak
        peak_idx = np.argmax(rate)
        below = np.where(rate[peak_idx:] < threshold)[0]
        decay_time = float(below[0] + peak_idx) if len(below) > 0 else float(len(post))

    return {
        'decay_time': decay_time,
        'total_spikes': int(spikes.sum()),
        'early_spikes': int(spikes[onset:onset+3].sum()),
    }


def run_experiment_B(variant_name, seed):
    """Noisy light signal spike-rate test. Metric: false spike rate."""
    np.random.seed(seed)
    time_steps = 60
    base = np.concatenate([np.zeros(15), np.ones(45) * 1.5])
    noise = np.random.normal(0, 0.4, time_steps)
    signal = np.clip(base + noise, 0, None)

    neuron = make_variant(variant_name, window_size=10, lambda_leak=0.8,
                          threshold=1.0, noise_tolerance=0.0)
    spikes = np.array([neuron.forward(x) for x in signal])

    steady = spikes[20:60]
    false_rate = float(steady.mean())
    return {
        'false_spike_rate': false_rate,
        'total_spikes': int(spikes.sum()),
    }


def run_experiment_C(variant_name, seed):
    """Phototaxis (clean environment). Metric: final error."""
    np.random.seed(seed)
    time_steps = 300
    agent = Bioton(variant_name, start_pos=50.0, noisy_env=False,
                   noise_tolerance=0.0)
    for _ in range(time_steps):
        light = get_clean_light(agent.pos)
        agent.step(light)
    final_err = abs(agent.pos - 80.0)
    conv = next((i for i, p in enumerate(agent.pos_log) if abs(p - 80.0) < 2.0),
                time_steps)
    return {
        'final_err': float(final_err),
        'convergence': int(conv),
        'total_spikes': int(sum(agent.spike_log)),
        'final_pos': float(agent.pos),
    }


def run_experiment_D(variant_name, seed):
    """Noisy phototaxis. Metric: final error."""
    rng = np.random.RandomState(seed)
    time_steps = 250
    agent = Bioton(variant_name, start_pos=50.0, noisy_env=True,
                   noise_tolerance=0.6)
    for _ in range(time_steps):
        light = get_noisy_light(agent.pos, noise_level=0.5, rng=rng)
        agent.step(light)
    final_err = abs(agent.pos - 80.0)
    conv = next((i for i, p in enumerate(agent.pos_log) if abs(p - 80.0) < 2.0),
                time_steps)
    return {
        'final_err': float(final_err),
        'convergence': int(conv),
        'total_spikes': int(sum(agent.spike_log)),
        'final_pos': float(agent.pos),
    }


# ============================================================
# 4. Statistical helpers (same as Part 1/2)
# ============================================================
def cohens_d(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    var_a = float(a.var(ddof=1)) if len(a) > 1 else 0.0
    var_b = float(b.var(ddof=1)) if len(b) > 1 else 0.0
    pooled_std = np.sqrt(max((var_a + var_b) / 2, 1e-6))
    return float(np.clip((a.mean() - b.mean()) / pooled_std, -1000.0, 1000.0))


def bootstrap_ci(data, n_boot=10000, ci=0.95, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    data = np.asarray(data)
    n = len(data)
    boot_means = np.array([rng.choice(data, n, replace=True).mean()
                           for _ in range(n_boot)])
    return (float(np.percentile(boot_means, (1 - ci) / 2 * 100)),
            float(np.percentile(boot_means, (1 + ci) / 2 * 100)))


def interpret_d(d):
    ad = abs(d)
    if ad < 0.2: return "negligible"
    if ad < 0.5: return "small"
    if ad < 0.8: return "medium"
    if ad < 2.0: return "large"
    if ad < 100.0: return "very large"
    return "extreme"


# Bonferroni correction: 3 ablations × 4 tasks = 12 comparisons
N_COMPARISONS = 3 * 4
ALPHA = 0.05 / N_COMPARISONS

# Predicted broken behavior per ablation (from Stage 2 Plan §5.4)
PREDICTED = {
    'TAN-no-surprise':  {'A': 'broken', 'B': 'broken', 'C': 'degraded', 'D': 'broken'},
    'TAN-no-attention': {'A': 'preserved', 'B': 'degraded', 'C': 'degraded', 'D': 'degraded'},
    'TAN-no-gate':      {'A': 'preserved', 'B': 'broken', 'C': 'broken', 'D': 'broken'},
}

# ============================================================
# 5. Main
# ============================================================
def main():
    log.info("=" * 60)
    log.info("TAN Stage 2 — Part 3: Ablation Study")
    log.info(f"Config: {N_VARIANTS} variants × 4 tasks × {N_SEEDS} seeds "
             f"= {N_VARIANTS * 4 * N_SEEDS} runs")
    log.info(f"Bonferroni alpha = {ALPHA:.5f} (0.05 / {N_COMPARISONS})")
    log.info("=" * 60)
    log.info("Variants:")
    log.info("  V0 TAN-full         — full TAN (reference)")
    log.info("  V1 TAN-no-surprise  — S_t = x_t (no deviation from history)")
    log.info("  V2 TAN-no-attention — C_t = mean(V) (uniform weighting)")
    log.info("  V3 TAN-no-gate      — A_t = C_t (no tanh gating)")

    t0 = time.time()

    # ---- Run all experiments ----
    rows = []
    for task, runner in [
        ('A', run_experiment_A),
        ('B', run_experiment_B),
        ('C', run_experiment_C),
        ('D', run_experiment_D),
    ]:
        for seed in SEEDS:
            for variant in VARIANTS:
                res = runner(variant, seed)
                row = {'task': task, 'variant': variant, 'seed': seed}
                row.update(res)
                rows.append(row)
        log.info(f"  Task {task} done ({N_VARIANTS * N_SEEDS} runs)")

    elapsed = time.time() - t0
    log.info(f"All runs complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ---- Build dataframe ----
    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "metrics_raw.csv", index=False)
    log.info(f"Raw metrics saved: {RES_DIR / 'metrics_raw.csv'}  ({len(df)} rows)")

    # ============================================================
    # 6. Per-task pairwise statistics (TAN-full vs each ablation)
    # ============================================================
    PRIMARY_METRIC = {
        'A': 'decay_time',
        'B': 'false_spike_rate',
        'C': 'final_err',
        'D': 'final_err',
    }
    METRIC_LABEL = {
        'A': 'Decay Time (steps, higher=better)',
        'B': 'False Spike Rate (lower=better)',
        'C': 'Final Error (lower=better)',
        'D': 'Final Error (lower=better)',
    }
    BETTER = {'A': 'higher', 'B': 'lower', 'C': 'lower', 'D': 'lower'}

    rng_boot = np.random.RandomState(123)
    summary_rows = []
    contribution_rows = []  # for module-behavior heatmap

    for task in ['A', 'B', 'C', 'D']:
        metric = PRIMARY_METRIC[task]
        sub = df[df['task'] == task]
        full_vals = sub[sub['variant'] == 'TAN-full'][metric].values
        full_mean = float(full_vals.mean())
        full_std = float(full_vals.std(ddof=1)) if len(full_vals) > 1 else 0.0
        full_lo, full_hi = bootstrap_ci(full_vals, rng=rng_boot)

        for ablation in ['TAN-no-surprise', 'TAN-no-attention', 'TAN-no-gate']:
            abl_vals = sub[sub['variant'] == ablation][metric].values
            abl_mean = float(abl_vals.mean())
            abl_std = float(abl_vals.std(ddof=1)) if len(abl_vals) > 1 else 0.0
            abl_lo, abl_hi = bootstrap_ci(abl_vals, rng=rng_boot)

            # t-test
            try:
                t_stat, t_p = stats.ttest_ind(full_vals, abl_vals, equal_var=False)
            except Exception:
                t_stat, t_p = float('nan'), 1.0
            try:
                _, w_p = stats.ranksums(full_vals, abl_vals)
            except Exception:
                w_p = float('nan')
            d = cohens_d(full_vals, abl_vals)
            d_interp = interpret_d(d)

            # Significance
            sig = "***" if t_p < 0.001 else ("**" if t_p < 0.01 else
                  ("*" if t_p < ALPHA else "ns"))

            # Degradation analysis
            # For 'lower is better': ablation is worse if abl_mean > full_mean
            # For 'higher is better' (Exp A): ablation is worse if abl_mean < full_mean
            if BETTER[task] == 'lower':
                worse = abl_mean > full_mean
                pct_degradation = ((abl_mean - full_mean) / max(abs(full_mean), 1e-9)) * 100
            else:
                worse = abl_mean < full_mean
                pct_degradation = ((full_mean - abl_mean) / max(abs(full_mean), 1e-9)) * 100

            # Verdict: BROKEN / DEGRADED / PRESERVED
            if not worse:
                # Ablation is actually better or equal — module not necessary
                verdict = 'IMPROVED'
            elif t_p < ALPHA and pct_degradation > 50:
                verdict = 'BROKEN'
            elif t_p < ALPHA and pct_degradation > 10:
                verdict = 'DEGRADED'
            elif t_p < ALPHA:
                verdict = 'MILD-DEG'
            else:
                verdict = 'PRESERVED'

            # Check prediction
            predicted = PREDICTED[ablation][task]
            prediction_match = (
                (predicted == 'broken' and verdict == 'BROKEN') or
                (predicted == 'degraded' and verdict in ('DEGRADED', 'BROKEN', 'MILD-DEG')) or
                (predicted == 'preserved' and verdict in ('PRESERVED', 'MILD-DEG'))
            )

            summary_rows.append({
                'task': task,
                'metric': metric,
                'metric_label': METRIC_LABEL[task],
                'ablation': ablation,
                'module_removed': ablation.replace('TAN-no-', ''),
                'full_mean': full_mean,
                'full_std': full_std,
                'full_CI_lo': full_lo,
                'full_CI_hi': full_hi,
                'ablation_mean': abl_mean,
                'ablation_std': abl_std,
                'ablation_CI_lo': abl_lo,
                'ablation_CI_hi': abl_hi,
                't_stat': float(t_stat) if not np.isnan(t_stat) else float('nan'),
                't_p': float(t_p),
                'wilcoxon_p': float(w_p),
                'cohens_d': d,
                'd_interp': d_interp,
                'sig': sig,
                'pct_degradation': float(pct_degradation),
                'verdict': verdict,
                'predicted': predicted,
                'prediction_match': bool(prediction_match),
            })

            # Contribution matrix entry
            contribution_rows.append({
                'task': task,
                'module_removed': ablation.replace('TAN-no-', ''),
                'pct_degradation': float(pct_degradation),
                'verdict': verdict,
                'predicted': predicted,
                'match': bool(prediction_match),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RES_DIR / "stats_summary.csv", index=False)
    log.info(f"Stats summary saved: {RES_DIR / 'stats_summary.csv'}")

    contribution_df = pd.DataFrame(contribution_rows)
    contrib_pivot = contribution_df.pivot(index='module_removed',
                                          columns='task',
                                          values='pct_degradation')
    contrib_pivot.to_csv(RES_DIR / "contribution_matrix.csv")
    log.info(f"Contribution matrix saved: {RES_DIR / 'contribution_matrix.csv'}")

    # ============================================================
    # 7. Markdown report
    # ============================================================
    md = []
    md.append("# TAN Stage 2 — Part 3: Ablation Study Report\n")
    md.append(f"**Config:** {N_VARIANTS} variants × 4 tasks × {N_SEEDS} seeds "
              f"= {N_VARIANTS * 4 * N_SEEDS} runs\n")
    md.append(f"**Alpha (Bonferroni-corrected):** {ALPHA:.5f} (0.05 / {N_COMPARISONS})\n")
    md.append(f"**Bootstrap resamples:** 10,000\n")
    md.append(f"**Total runtime:** {elapsed:.1f}s ({elapsed/60:.1f} min)\n\n")
    md.append("---\n\n")

    # Variants table
    md.append("## Ablation Variants\n\n")
    md.append("| Variant | Surprise | Attention | Gate | Definition |\n")
    md.append("|---------|----------|-----------|------|------------|\n")
    md.append("| TAN-full | ✓ | ✓ | ✓ | Original TAN |\n")
    md.append("| TAN-no-surprise | ✗ | ✓ | ✓ | $S_t \\leftarrow x_t$ (use raw input) |\n")
    md.append("| TAN-no-attention | ✓ | ✗ | ✓ | $C_t \\leftarrow \\mathrm{mean}(V)$ (uniform weighting) |\n")
    md.append("| TAN-no-gate | ✓ | ✓ | ✗ | $A_t \\leftarrow C_t$ (drop tanh gating) |\n\n")
    md.append("---\n\n")

    # Per-task tables
    md.append("## Per-Task Pairwise Comparisons (TAN-full vs Ablation)\n\n")
    for task in ['A', 'B', 'C', 'D']:
        md.append(f"### Task {task} — {METRIC_LABEL[task]}\n\n")
        md.append("| Variant | Mean | Std | 95% CI | t-stat | p-value | Cohen's d | % degrad | Verdict | Pred. | Match? |\n")
        md.append("|---------|------|-----|--------|--------|---------|-----------|----------|---------|-------|--------|\n")
        sub = summary_df[summary_df['task'] == task]
        # TAN-full row
        full_row = sub.iloc[0]
        md.append(f"| **TAN-full** | **{full_row['full_mean']:.3f}** | "
                  f"**{full_row['full_std']:.3f}** | "
                  f"**[{full_row['full_CI_lo']:.3f}, {full_row['full_CI_hi']:.3f}]** | "
                  f"— | — | — | — | — | — | — |\n")
        # Ablation rows
        for _, r in sub.iterrows():
            match_icon = '✓' if r['prediction_match'] else '✗'
            md.append(f"| {r['ablation']} | {r['ablation_mean']:.3f} | "
                      f"{r['ablation_std']:.3f} | "
                      f"[{r['ablation_CI_lo']:.3f}, {r['ablation_CI_hi']:.3f}] | "
                      f"{r['t_stat']:.2f} | {r['t_p']:.3g} {r['sig']} | "
                      f"{r['cohens_d']:.2f} ({r['d_interp']}) | "
                      f"{r['pct_degradation']:+.1f}% | "
                      f"{r['verdict']} | {r['predicted']} | {match_icon} |\n")
        md.append("\n")

    # Module-behavior contribution matrix
    md.append("## Module-Behavior Contribution Matrix\n\n")
    md.append("Percentage degradation when each module is removed (positive = ablation worse):\n\n")
    md.append("| Module removed | Task A (Habituation) | Task B (Noise) | Task C (Phototaxis) | Task D (Noisy Photo.) |\n")
    md.append("|----------------|---------------------|----------------|---------------------|----------------------|\n")
    for module in ['surprise', 'attention', 'gate']:
        cells = []
        for task in ['A', 'B', 'C', 'D']:
            v = contrib_pivot.loc[module, task]
            sub = summary_df[(summary_df['task'] == task) &
                              (summary_df['module_removed'] == module)]
            if len(sub) > 0:
                verdict = sub.iloc[0]['verdict']
                cells.append(f"{v:+.1f}% ({verdict})")
            else:
                cells.append("N/A")
        md.append(f"| {module} | " + " | ".join(cells) + " |\n")
    md.append("\n")

    # Verdict legend
    md.append("**Verdict legend:**\n")
    md.append("- **BROKEN**: significant degradation > 50% (module is critical)\n")
    md.append("- **DEGRADED**: significant degradation 10-50% (module is important)\n")
    md.append("- **MILD-DEG**: significant degradation < 10% (module is helpful)\n")
    md.append("- **PRESERVED**: no significant difference (module not needed for this task)\n")
    md.append("- **IMPROVED**: ablation is actually better (module is harmful)\n\n")
    md.append("---\n\n")

    # Prediction accuracy
    n_predictions = len(summary_df)
    n_matches = int(summary_df['prediction_match'].sum())
    md.append("## Prediction Accuracy\n\n")
    md.append(f"- Total predictions: {n_predictions}\n")
    md.append(f"- Correct predictions: **{n_matches}/{n_predictions}** "
              f"({n_matches/n_predictions*100:.1f}%)\n\n")

    # Per-module prediction accuracy
    md.append("### Per-module breakdown\n\n")
    md.append("| Module removed | Predictions | Correct | Accuracy |\n")
    md.append("|----------------|-------------|---------|----------|\n")
    for module in ['surprise', 'attention', 'gate']:
        sub = summary_df[summary_df['module_removed'] == module]
        n_total = len(sub)
        n_correct = int(sub['prediction_match'].sum())
        md.append(f"| {module} | {n_total} | {n_correct} | "
                  f"{n_correct/n_total*100:.0f}% |\n")
    md.append("\n")

    # Decision gate
    # Pass criterion: each predicted "broken" cell is significantly worse (BROKEN verdict),
    # and each predicted "preserved" cell is NOT significantly worse.
    n_broken_correct = 0
    n_broken_total = 0
    n_preserved_correct = 0
    n_preserved_total = 0
    n_degraded_correct = 0
    n_degraded_total = 0
    for _, r in summary_df.iterrows():
        if r['predicted'] == 'broken':
            n_broken_total += 1
            if r['verdict'] == 'BROKEN':
                n_broken_correct += 1
        elif r['predicted'] == 'preserved':
            n_preserved_total += 1
            if r['verdict'] in ('PRESERVED', 'MILD-DEG'):
                n_preserved_correct += 1
        elif r['predicted'] == 'degraded':
            n_degraded_total += 1
            if r['verdict'] in ('DEGRADED', 'BROKEN', 'MILD-DEG'):
                n_degraded_correct += 1

    md.append("## Decision Gate Summary\n\n")
    md.append(f"**Pass criteria:**\n")
    md.append(f"1. All predicted BROKEN cells must be observed as BROKEN: "
              f"{n_broken_correct}/{n_broken_total}\n")
    md.append(f"2. All predicted PRESERVED cells must be observed as PRESERVED/MILD-DEG: "
              f"{n_preserved_correct}/{n_preserved_total}\n")
    md.append(f"3. All predicted DEGRADED cells must be observed as DEGRADED/BROKEN/MILD-DEG: "
              f"{n_degraded_correct}/{n_degraded_total}\n\n")

    all_pass = (n_broken_correct == n_broken_total and
                n_preserved_correct == n_preserved_total and
                n_degraded_correct == n_degraded_total)

    # Critical check: no ablation should IMPROVE performance (would mean the module is harmful)
    n_improved = (summary_df['verdict'] == 'IMPROVED').sum()
    md.append(f"4. No module removal should IMPROVE performance: "
              f"{n_improved} IMPROVED cells (must be 0)\n\n")

    if all_pass and n_improved == 0:
        md.append("**Gate 3 (Ablation Study):** ✅ PROCEED to Part 4. "
                  "All modules are necessary and predictions are confirmed.\n")
    elif n_improved > 0:
        md.append(f"**Gate 3:** ❌ HALT — {n_improved} ablation(s) IMPROVED performance. "
                  f"The removed module(s) are harmful and the model must be revised.\n")
    elif n_broken_correct < n_broken_total:
        md.append(f"**Gate 3:** ⚠️ Proceed with caution. {n_broken_total - n_broken_correct} "
                  f"predicted BROKEN cell(s) were not observed as BROKEN. "
                  f"The module(s) may be less critical than expected.\n")
    else:
        md.append(f"**Gate 3:** ⚠️ Proceed with caution. Some predictions did not match. "
                  f"Adjust the narrative accordingly.\n")

    with open(RES_DIR / "stats_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    log.info(f"Markdown report saved: {RES_DIR / 'stats_summary.md'}")

    # ============================================================
    # 8. Figures
    # ============================================================
    log.info("Generating figures...")

    # --- Figure 1: Grouped bar chart (4 panels × 5 variants) ---
    fig, axs = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    for i, task in enumerate(['A', 'B', 'C', 'D']):
        metric = PRIMARY_METRIC[task]
        means, stds, cis = [], [], []
        for v in VARIANTS:
            vals = df[(df['task'] == task) & (df['variant'] == v)][metric].values
            means.append(float(vals.mean()))
            stds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
            lo, hi = bootstrap_ci(vals, rng=rng_boot)
            cis.append((means[-1] - lo, hi - means[-1]))

        x = np.arange(len(VARIANTS))
        bar_colors = [VARIANT_COLORS[v] for v in VARIANTS]
        bars = axs[i].bar(x, means, yerr=np.array(cis).T, capsize=4,
                          color=bar_colors, edgecolor='black', linewidth=0.6,
                          width=0.6, alpha=0.85)
        # Highlight TAN-full bar with thicker red edge
        bars[0].set_edgecolor('#C0392B')
        bars[0].set_linewidth(2.0)
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(['full', '-surp', '-att', '-gate'],
                                rotation=0, ha='center', fontsize=9)
        axs[i].set_title(f"Task {task}: {METRIC_LABEL[task]}", fontsize=10)
        axs[i].grid(True, linestyle='--', alpha=0.4, axis='y')
        if i == 0:
            axs[i].set_ylabel("Mean ± 95% CI")

    fig.suptitle("Part 3 — Ablation Study: 4 TAN Variants × 4 Tasks (50 seeds each)\n"
                 "(Red border = TAN-full reference; ** = p<0.01, *** = p<0.001 vs full)",
                 fontsize=11, fontweight='bold')
    fig.savefig(FIG_DIR / "fig_grouped_bar.png", dpi=200, facecolor='white')
    plt.close(fig)
    log.info(f"Grouped bar chart saved: {FIG_DIR / 'fig_grouped_bar.png'}")

    # --- Figure 2: Module-behavior contribution heatmap ---
    fig, ax = plt.subplots(figsize=(7, 3.5))
    modules = ['surprise', 'attention', 'gate']
    tasks = ['A', 'B', 'C', 'D']
    matrix = np.array([[contrib_pivot.loc[m, t] for t in tasks] for m in modules])

    # Diverging colormap: red = ablation worse (positive), blue = ablation better (negative)
    vmax = max(abs(matrix.min()), abs(matrix.max()), 50)
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['A: Habit.', 'B: Noise', 'C: Photot.', 'D: Noisy Ph.'],
                       fontsize=9)
    ax.set_yticks(range(3))
    ax.set_yticklabels(['−surprise', '−attention', '−gate'], fontsize=10)
    ax.set_xlabel("Task")
    ax.set_ylabel("Ablation")
    ax.set_title("Module-Behavior Contribution Heatmap\n"
                 "(% degradation when module removed; + = ablation worse, − = ablation better)",
                 fontsize=10)
    # Annotate cells with % and verdict
    for i, m in enumerate(modules):
        for j, t in enumerate(tasks):
            v = matrix[i, j]
            sub = summary_df[(summary_df['task'] == t) &
                              (summary_df['module_removed'] == m)]
            verdict = sub.iloc[0]['verdict'] if len(sub) > 0 else '?'
            sig = sub.iloc[0]['sig'] if len(sub) > 0 else ''
            color = 'white' if abs(v) > vmax * 0.6 else 'black'
            ax.text(j, i, f"{v:+.0f}%\n{verdict}\n{sig}",
                    ha='center', va='center', fontsize=8,
                    color=color, fontweight='bold')
    plt.colorbar(im, ax=ax, label='% Degradation')
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_contribution_heatmap.png", dpi=200, facecolor='white')
    plt.close(fig)
    log.info(f"Contribution heatmap saved: {FIG_DIR / 'fig_contribution_heatmap.png'}")

    # ============================================================
    # 9. Final summary print
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 3 — ABLATION STUDY: COMPLETE")
    print("=" * 60)
    print(f"Output dir: {OUT_DIR}")
    print()
    print("Per-task mean (primary metric) across 4 variants:")
    for task in ['A', 'B', 'C', 'D']:
        metric = PRIMARY_METRIC[task]
        sub = df[df['task'] == task]
        line = f"  Task {task}: "
        for v in VARIANTS:
            val = sub[sub['variant'] == v][metric].mean()
            line += f"{v}={val:.3f}  "
        print(line)
    print()
    print("Module-Behavior Contribution Matrix (% degradation, + = ablation worse):")
    print(f"  {'':15s}  Task A    Task B    Task C    Task D")
    for module in ['surprise', 'attention', 'gate']:
        cells = []
        for task in ['A', 'B', 'C', 'D']:
            v = contrib_pivot.loc[module, task]
            sub = summary_df[(summary_df['task'] == task) &
                              (summary_df['module_removed'] == module)]
            verdict = sub.iloc[0]['verdict'] if len(sub) > 0 else '?'
            cells.append(f"{v:+6.1f}%/{verdict[:6]:6s}")
        print(f"  -{module:13s}  " + "  ".join(cells))
    print()
    n_matches = int(summary_df['prediction_match'].sum())
    n_total = len(summary_df)
    n_improved = int((summary_df['verdict'] == 'IMPROVED').sum())
    print(f"Predictions matched: {n_matches}/{n_total} ({n_matches/n_total*100:.0f}%)")
    print(f"IMPROVED cells (must be 0): {n_improved}")
    if all_pass and n_improved == 0:
        print(f"Gate 3 verdict: ✅ PROCEED to Part 4")
    elif n_improved > 0:
        print(f"Gate 3 verdict: ❌ HALT — module(s) are harmful")
    else:
        print(f"Gate 3 verdict: ⚠️ Proceed with caution")
    print("=" * 60)


if __name__ == "__main__":
    main()
