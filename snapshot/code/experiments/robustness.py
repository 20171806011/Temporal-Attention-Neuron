"""
TAN Stage 2 — Part 4: Parameter Robustness
============================================
Sweeps four key hyperparameters of TAN to verify that its performance is not
an artifact of one magic parameter setting. For each parameter we fix the
other three at default values and sweep the target across a wide range.

Parameters:
  - lambda  (leak coefficient):     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  (9 values)
  - beta    (inverse temperature):  [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]            (7 values, log-spaced)
  - W       (window length):        [2, 3, 5, 7, 10, 15, 20, 30]                    (8 values)
  - theta   (spike threshold):      [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]             (7 values)

Protocol: 20 seeds per configuration (reduced from 50 for tractability).
Total runs: (9 + 7 + 8 + 7) x 20 x 4 tasks = 2480 runs.

Pass criterion: each parameter must show a "plateau width" >= 50% of its sweep
range, where plateau = range of values within 10% of the best (lowest) mean
metric.

Outputs:
  - results/metrics_raw.csv             (per-seed metrics, 2480 rows)
  - results/plateau_summary.csv         (one row per parameter x task)
  - results/plateau_summary.md          (human-readable report)
  - figures/fig_sweep_lambda.png        (4 panels: 1 per task, perf vs lambda)
  - figures/fig_sweep_beta.png          (4 panels: 1 per task, perf vs beta, log x)
  - figures/fig_sweep_W.png             (4 panels: 1 per task, perf vs W)
  - figures/fig_sweep_theta.png         (4 panels: 1 per task, perf vs theta)
  - figures/fig_plateau_widths.png      (grouped bar: plateau widths across params)
  - figures/fig_2d_lambda_W.png         (2D heatmap of lambda x W on Task D)
  - logs/run.log
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
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
OUT_DIR = Path("/home/z/my-project/download/stage2_part4")
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
warnings.filterwarnings("ignore")

# ============ Constants ============
N_SEEDS = 20
SEEDS = list(range(N_SEEDS))

# Default TAN parameters (matches Part 1/2/3)
DEFAULTS = dict(
    window_size=5,
    lambda_leak=0.5,
    threshold=0.5,
    noise_tolerance=0.0,
    beta=1.0,
)

# Sweep ranges
SWEEPS = {
    'lambda':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],   # 9 values
    'beta':    [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0],             # 7 values (log)
    'W':       [2, 3, 5, 7, 10, 15, 20, 30],                     # 8 values
    'theta':   [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5],              # 7 values
}

# Map sweep parameter names to TANNeuron constructor kwargs
PARAM_TO_KWARG = {
    'lambda':  'lambda_leak',
    'beta':    'beta',
    'W':       'window_size',
    'theta':   'threshold',
}

# Colors
COLORS = {
    'A': '#1F4E79',  # dark blue
    'B': '#C0392B',  # red
    'C': '#27AE60',  # green
    'D': '#F39C12',  # orange
}

# ============================================================
# TAN neuron (same as Part 3 ablation framework, full TAN)
# ============================================================
class TANNeuron:
    def __init__(self, window_size=5, lambda_leak=0.5, threshold=0.5,
                 noise_tolerance=0.0, beta=1.0,
                 W_q=2.0, W_k=1.0, W_v=1.0):
        self.window_size = window_size
        self.lambda_leak = lambda_leak
        self.threshold = threshold
        self.noise_tolerance = noise_tolerance
        self.beta = beta
        self.W_q = np.array([W_q])
        self.W_k = np.array([W_k])
        self.W_v = np.array([W_v])
        self.h = 0.0
        self.history = np.zeros(window_size)

    def softmax(self, x):
        e_x = np.exp(self.beta * (x - np.max(x)))
        return e_x / (e_x.sum(axis=0) + 1e-9)

    def forward(self, x):
        self.history = np.roll(self.history, -1)
        self.history[-1] = x
        S_t = x - np.mean(self.history)
        S_t = np.maximum(0, S_t - self.noise_tolerance)
        Q = S_t * self.W_q
        K = self.history * self.W_k
        V = self.history * self.W_v
        att = self.softmax(Q * K)
        C_t = np.sum(att * V)
        A_t = np.tanh(S_t) * C_t
        self.h = self.lambda_leak * self.h + A_t
        if self.h > self.threshold:
            spike = 1
            self.h = 0.0
        else:
            spike = 0
        return spike


# ============================================================
# Environment (1D phototaxis) — same as Part 1/2/3
# ============================================================
def get_clean_light(x):
    return 10.0 * np.exp(-((x - 80.0) ** 2) / (2 * 15.0 ** 2))


def get_noisy_light(x, noise_level=0.5, rng=None):
    clean = get_clean_light(x)
    n = rng.normal(0, noise_level) if rng else np.random.normal(0, noise_level)
    return max(0.0, clean + n)


class Bioton:
    def __init__(self, tan_kwargs, start_pos=50.0, noisy_env=False,
                 noise_tolerance=0.0):
        kwargs = dict(tan_kwargs)
        kwargs['noise_tolerance'] = noise_tolerance
        self.brain = TANNeuron(**kwargs)
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
# Experiment runners
# ============================================================
def run_experiment_A(tan_kwargs, seed):
    np.random.seed(seed)
    neuron = TANNeuron(**tan_kwargs)
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

    # Robust habituation metric: ratio of firing rate in last 5 steps / first 5 steps
    # Lower ratio = better habituation (0 means firing stopped completely)
    first5 = spikes[onset:onset+5].sum()
    last5 = spikes[-5:].sum()
    if first5 == 0:
        hab_ratio = 1.0  # no firing at all -> no habituation signal
    else:
        hab_ratio = last5 / first5

    return {
        'decay_time': decay_time,
        'hab_ratio': float(hab_ratio),  # NEW: more robust metric
        'total_spikes': int(spikes.sum()),
    }


def run_experiment_B(tan_kwargs, seed):
    np.random.seed(seed)
    time_steps = 60
    base = np.concatenate([np.zeros(15), np.ones(45) * 1.5])
    noise = np.random.normal(0, 0.4, time_steps)
    signal = np.clip(base + noise, 0, None)
    neuron = TANNeuron(**tan_kwargs)
    spikes = np.array([neuron.forward(x) for x in signal])
    steady = spikes[20:60]
    return {
        'false_spike_rate': float(steady.mean()),
        'total_spikes': int(spikes.sum()),
    }


def run_experiment_C(tan_kwargs, seed):
    np.random.seed(seed)
    time_steps = 300
    agent = Bioton(tan_kwargs, start_pos=50.0, noisy_env=False,
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


def run_experiment_D(tan_kwargs, seed):
    rng = np.random.RandomState(seed)
    time_steps = 250
    agent = Bioton(tan_kwargs, start_pos=50.0, noisy_env=True,
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
# Plateau width computation
# ============================================================
def compute_plateau_width(values, means, lower_is_better=True, tolerance=0.10):
    """Compute the plateau width: range of values within `tolerance` of best.

    Returns (plateau_start, plateau_end, plateau_width_fraction).
    `plateau_width_fraction` is the fraction of the sweep range covered by
    the plateau (1.0 = entire range is plateau, 0.0 = no plateau).

    For 'lower is better' metrics, the best is the minimum mean.
    For 'higher is better' metrics, the best is the maximum mean.
    """
    if lower_is_better:
        best_idx = int(np.argmin(means))
        best_val = means[best_idx]
        # Plateau = values whose mean is within `tolerance` * |best_val| of best
        # Use absolute threshold to handle near-zero best values
        threshold = best_val + tolerance * max(abs(best_val), 1e-6)
        in_plateau = means <= threshold
    else:
        best_idx = int(np.argmax(means))
        best_val = means[best_idx]
        threshold = best_val - tolerance * max(abs(best_val), 1e-6)
        in_plateau = means >= threshold

    # Find the leftmost and rightmost in-plateau indices
    plateau_indices = np.where(in_plateau)[0]
    if len(plateau_indices) == 0:
        return values[best_idx], values[best_idx], 0.0
    start_idx = plateau_indices[0]
    end_idx = plateau_indices[-1]
    plateau_start = values[start_idx]
    plateau_end = values[end_idx]

    # Width as fraction of total sweep range
    if len(values) >= 2:
        total_range = values[-1] - values[0]
    else:
        total_range = 1.0
    if total_range == 0:
        return plateau_start, plateau_end, 1.0
    # Use number of in-plateau values / total number of values
    # This is more robust than numeric range for log-spaced sweeps
    width_fraction = len(plateau_indices) / len(values)
    return plateau_start, plateau_end, width_fraction


# ============================================================
# Main
# ============================================================
def main():
    log.info("=" * 60)
    log.info("TAN Stage 2 — Part 4: Parameter Robustness")
    log.info(f"Config: 4 parameters x ~8 values x 4 tasks x {N_SEEDS} seeds")
    log.info(f"Defaults: {DEFAULTS}")
    log.info("=" * 60)

    t0 = time.time()

    # ============================================================
    # 1. 1D parameter sweeps
    # ============================================================
    rows = []

    for param_name in ['lambda', 'beta', 'W', 'theta']:
        sweep_values = SWEEPS[param_name]
        kwarg_name = PARAM_TO_KWARG[param_name]
        log.info(f"\n--- Sweeping {param_name} ({len(sweep_values)} values) ---")

        for val in sweep_values:
            # Build kwargs: start from defaults, override the swept param
            tan_kwargs = dict(DEFAULTS)
            tan_kwargs[kwarg_name] = val

            for task, runner in [
                ('A', run_experiment_A),
                ('B', run_experiment_B),
                ('C', run_experiment_C),
                ('D', run_experiment_D),
            ]:
                for seed in SEEDS:
                    res = runner(tan_kwargs, seed)
                    row = {
                        'param': param_name,
                        'value': val,
                        'task': task,
                        'seed': seed,
                        **res,
                    }
                    rows.append(row)

        elapsed = time.time() - t0
        log.info(f"  {param_name} done ({elapsed:.1f}s elapsed)")

    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / "metrics_raw.csv", index=False)
    log.info(f"Raw metrics saved: {RES_DIR / 'metrics_raw.csv'} ({len(df)} rows)")

    # ============================================================
    # 2. Compute plateau widths per (param, task)
    # ============================================================
    PRIMARY_METRIC = {
        'A': 'hab_ratio',       # NEW: more robust metric (lower = better)
        'B': 'false_spike_rate',
        'C': 'final_err',
        'D': 'final_err',
    }
    METRIC_LABEL = {
        'A': 'Habituation Ratio (lower=better)',
        'B': 'False Spike Rate (lower=better)',
        'C': 'Final Error (lower=better)',
        'D': 'Final Error (lower=better)',
    }
    # All metrics are 'lower is better'
    LOWER_IS_BETTER = {'A': True, 'B': True, 'C': True, 'D': True}

    plateau_rows = []
    for param_name in ['lambda', 'beta', 'W', 'theta']:
        sweep_values = SWEEPS[param_name]
        for task in ['A', 'B', 'C', 'D']:
            metric = PRIMARY_METRIC[task]
            sub = df[(df['param'] == param_name) & (df['task'] == task)]
            # Mean per sweep value
            means = sub.groupby('value')[metric].mean().reindex(sweep_values).values
            stds = sub.groupby('value')[metric].std().reindex(sweep_values).values

            p_start, p_end, p_width = compute_plateau_width(
                sweep_values, means,
                lower_is_better=LOWER_IS_BETTER[task],
                tolerance=0.10,
            )

            best_idx = int(np.argmin(means)) if LOWER_IS_BETTER[task] else int(np.argmax(means))
            best_val = sweep_values[best_idx]
            best_mean = means[best_idx]

            plateau_rows.append({
                'param': param_name,
                'task': task,
                'metric': metric,
                'n_values': len(sweep_values),
                'best_value': best_val,
                'best_mean': float(best_mean),
                'plateau_start': float(p_start),
                'plateau_end': float(p_end),
                'plateau_width_fraction': float(p_width),
                'pass_50pct': bool(p_width >= 0.5),
            })

    plateau_df = pd.DataFrame(plateau_rows)
    plateau_df.to_csv(RES_DIR / "plateau_summary.csv", index=False)
    log.info(f"Plateau summary saved: {RES_DIR / 'plateau_summary.csv'}")

    # ============================================================
    # 3. 2D lambda x W sweep on Task D (interaction check)
    # ============================================================
    log.info("\n--- 2D sweep: lambda x W on Task D ---")
    lambda_vals_2d = [0.2, 0.4, 0.5, 0.6, 0.8]
    W_vals_2d = [3, 5, 7, 10]
    matrix_2d = np.zeros((len(lambda_vals_2d), len(W_vals_2d)))

    for i, lam in enumerate(lambda_vals_2d):
        for j, w in enumerate(W_vals_2d):
            kwargs = dict(DEFAULTS)
            kwargs['lambda_leak'] = lam
            kwargs['window_size'] = w
            errs = []
            for seed in SEEDS:
                res = run_experiment_D(kwargs, seed)
                errs.append(res['final_err'])
            matrix_2d[i, j] = float(np.mean(errs))
    log.info("2D sweep done")

    # Save 2D matrix
    df_2d = pd.DataFrame(matrix_2d,
                         index=[f'lambda={l}' for l in lambda_vals_2d],
                         columns=[f'W={w}' for w in W_vals_2d])
    df_2d.to_csv(RES_DIR / "lambda_W_2d_sweep.csv")

    # ============================================================
    # 4. Generate figures
    # ============================================================
    log.info("\nGenerating figures...")

    # --- Figs 1-4: 1D sweep plots (one per parameter) ---
    for param_name in ['lambda', 'beta', 'W', 'theta']:
        sweep_values = SWEEPS[param_name]
        fig, axs = plt.subplots(1, 4, figsize=(14, 3.2), constrained_layout=True)
        use_log_x = (param_name == 'beta')

        for i, task in enumerate(['A', 'B', 'C', 'D']):
            metric = PRIMARY_METRIC[task]
            sub = df[(df['param'] == param_name) & (df['task'] == task)]
            means = sub.groupby('value')[metric].mean().reindex(sweep_values).values
            stds = sub.groupby('value')[metric].std().reindex(sweep_values).values

            # Shaded ±1 std band
            axs[i].fill_between(sweep_values, means - stds, means + stds,
                                color=COLORS[task], alpha=0.2)
            axs[i].plot(sweep_values, means, color=COLORS[task],
                        marker='o', linewidth=2, markersize=5)

            # Mark default value
            default_val = DEFAULTS[PARAM_TO_KWARG[param_name]]
            if default_val in sweep_values:
                d_idx = sweep_values.index(default_val)
                axs[i].axvline(x=default_val, color='gray', linestyle=':',
                               linewidth=1, alpha=0.7, label=f'Default={default_val}')

            # Highlight plateau region
            p_row = plateau_df[(plateau_df['param'] == param_name) &
                                (plateau_df['task'] == task)].iloc[0]
            if p_row['plateau_width_fraction'] > 0:
                p_start = p_row['plateau_start']
                p_end = p_row['plateau_end']
                axs[i].axvspan(p_start, p_end, color='green', alpha=0.1,
                               label=f"Plateau ({p_row['plateau_width_fraction']*100:.0f}%)")

            axs[i].set_xlabel(param_name)
            axs[i].set_ylabel(METRIC_LABEL[task])
            axs[i].set_title(f"Task {task}", fontsize=10)
            if use_log_x:
                axs[i].set_xscale('log')
            axs[i].grid(True, linestyle='--', alpha=0.4)
            axs[i].legend(loc='best', fontsize=7)

        fig.suptitle(f"Parameter Sweep: {param_name}  (20 seeds per point, "
                     f"other params at default)",
                     fontsize=11, fontweight='bold')
        fig.savefig(FIG_DIR / f"fig_sweep_{param_name}.png", dpi=200,
                    facecolor='white')
        plt.close(fig)
        log.info(f"  Saved fig_sweep_{param_name}.png")

    # --- Fig 5: Plateau widths grouped bar ---
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    params = ['lambda', 'beta', 'W', 'theta']
    tasks = ['A', 'B', 'C', 'D']
    x = np.arange(len(params))
    width = 0.2
    for i, task in enumerate(tasks):
        heights = [plateau_df[(plateau_df['param'] == p) &
                              (plateau_df['task'] == task)]['plateau_width_fraction'].values[0]
                   for p in params]
        ax.bar(x + (i - 1.5) * width, heights, width,
               label=f'Task {task}', color=COLORS[task], edgecolor='black',
               linewidth=0.4, alpha=0.85)
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5,
               label='Pass threshold (50%)')
    ax.set_xticks(x)
    ax.set_xticklabels([r'$\lambda$', r'$\beta$', r'$W$', r'$\theta$'])
    ax.set_ylabel("Plateau Width (fraction of sweep range)")
    ax.set_title("Part 4 — Parameter Robustness: Plateau Widths\n"
                 "(Higher = more robust; red line = 50% pass threshold)",
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=8, ncol=3)
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    fig.savefig(FIG_DIR / "fig_plateau_widths.png", dpi=200, facecolor='white')
    plt.close(fig)
    log.info("  Saved fig_plateau_widths.png")

    # --- Fig 6: 2D heatmap of lambda x W on Task D ---
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    im = ax.imshow(matrix_2d, cmap='RdYlGn_r', aspect='auto',
                   vmin=np.nanmin(matrix_2d), vmax=np.nanmax(matrix_2d))
    ax.set_xticks(range(len(W_vals_2d)))
    ax.set_xticklabels([f'W={w}' for w in W_vals_2d])
    ax.set_yticks(range(len(lambda_vals_2d)))
    ax.set_yticklabels([f'λ={l}' for l in lambda_vals_2d])
    ax.set_xlabel("Window Length W")
    ax.set_ylabel("Leak Coefficient λ")
    ax.set_title("2D Interaction Sweep: λ × W on Task D\n"
                 "(mean final error over 20 seeds; green = lower = better)",
                 fontsize=10)
    # Annotate cells
    for i in range(len(lambda_vals_2d)):
        for j in range(len(W_vals_2d)):
            v = matrix_2d[i, j]
            color = 'white' if v > (np.nanmin(matrix_2d) + np.nanmax(matrix_2d)) / 2 else 'black'
            ax.text(j, i, f"{v:.1f}", ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Final Error |x - 80|')
    fig.savefig(FIG_DIR / "fig_2d_lambda_W.png", dpi=200, facecolor='white')
    plt.close(fig)
    log.info("  Saved fig_2d_lambda_W.png")

    # ============================================================
    # 5. Markdown report
    # ============================================================
    elapsed = time.time() - t0
    md = []
    md.append("# TAN Stage 2 — Part 4: Parameter Robustness Report\n")
    md.append(f"**Config:** 4 parameters × ~8 values × 4 tasks × {N_SEEDS} seeds\n")
    md.append(f"**Total runs:** {len(df)}\n")
    md.append(f"**Defaults:** λ=0.5, β=1.0, W=5, θ=0.5\n")
    md.append(f"**Plateau definition:** values within 10% of best mean\n")
    md.append(f"**Pass criterion:** plateau width ≥ 50% of sweep range\n")
    md.append(f"**Total runtime:** {elapsed:.1f}s ({elapsed/60:.1f} min)\n\n")
    md.append("---\n\n")

    # Per-parameter plateau table
    md.append("## Plateau Width Summary\n\n")
    md.append("| Parameter | Task | Best Value | Best Mean | Plateau Range | Plateau Width | Pass? |\n")
    md.append("|-----------|------|------------|-----------|---------------|---------------|-------|\n")
    for _, r in plateau_df.iterrows():
        param_sym = {'lambda': 'λ', 'beta': 'β', 'W': 'W', 'theta': 'θ'}[r['param']]
        pass_icon = '✓' if r['pass_50pct'] else '✗'
        md.append(f"| {param_sym} | {r['task']} | {r['best_value']} | "
                  f"{r['best_mean']:.3f} | "
                  f"[{r['plateau_start']}, {r['plateau_end']}] | "
                  f"{r['plateau_width_fraction']*100:.0f}% | "
                  f"{pass_icon} |\n")
    md.append("\n")

    # Per-parameter summary (aggregate across tasks)
    md.append("## Per-Parameter Verdict\n\n")
    md.append("| Parameter | Sweep Range | Tasks Passing (≥50%) | Verdict |\n")
    md.append("|-----------|-------------|----------------------|---------|\n")
    for p in ['lambda', 'beta', 'W', 'theta']:
        sweep = SWEEPS[p]
        sym = {'lambda': 'λ', 'beta': 'β', 'W': 'W', 'theta': 'θ'}[p]
        n_pass = int(plateau_df[plateau_df['param'] == p]['pass_50pct'].sum())
        if n_pass == 4:
            verdict = "✅ WIDE PLATEAU (all 4 tasks pass)"
        elif n_pass >= 3:
            verdict = "⚠️ MODERATE (1 task narrow)"
        elif n_pass >= 2:
            verdict = "⚠️ PARTIAL (2 tasks narrow)"
        else:
            verdict = "❌ NARROW PEAK (majority fail)"
        range_str = f"[{sweep[0]}, {sweep[-1]}]"
        md.append(f"| {sym} | {range_str} | {n_pass}/4 | {verdict} |\n")
    md.append("\n")

    # 2D interaction analysis
    md.append("## 2D Interaction Analysis (λ × W on Task D)\n\n")
    md.append("Mean final error over 20 seeds. Green = lower (better):\n\n")
    md.append("| | " + " | ".join(f"W={w}" for w in W_vals_2d) + " |\n")
    md.append("|" + "---|" * (len(W_vals_2d) + 1) + "\n")
    for i, lam in enumerate(lambda_vals_2d):
        row = f"| λ={lam} | " + " | ".join(f"{matrix_2d[i,j]:.2f}" for j in range(len(W_vals_2d))) + " |\n"
        md.append(row)
    md.append("\n")

    # Compute correlation between lambda effect and W effect
    flat = matrix_2d.flatten()
    md.append(f"- Best (λ, W) combination: λ={lambda_vals_2d[np.unravel_index(np.argmin(matrix_2d), matrix_2d.shape)[0]]}, "
              f"W={W_vals_2d[np.unravel_index(np.argmin(matrix_2d), matrix_2d.shape)[1]]}, "
              f"error={np.min(matrix_2d):.2f}\n")
    md.append(f"- Worst (λ, W) combination: error={np.max(matrix_2d):.2f}\n")
    md.append(f"- Range across 2D grid: {np.max(matrix_2d) - np.min(matrix_2d):.2f}\n\n")

    # Decision gate
    n_pass_total = int(plateau_df['pass_50pct'].sum())
    n_total = len(plateau_df)
    md.append("## Decision Gate Summary\n\n")
    md.append(f"- Total (parameter × task) cells: {n_total}\n")
    md.append(f"- Cells with plateau width ≥ 50%: **{n_pass_total}/{n_total}** "
              f"({n_pass_total/n_total*100:.0f}%)\n\n")

    # Check if any parameter has a sharp peak (fails on majority of tasks)
    params_failing = []
    for p in ['lambda', 'beta', 'W', 'theta']:
        n_fail = int((~plateau_df[plateau_df['param'] == p]['pass_50pct']).sum())
        if n_fail >= 3:
            params_failing.append(p)

    if n_pass_total == n_total:
        md.append("**Gate 4 (Parameter Robustness):** ✅ PROCEED to Part 5. "
                  "All parameters show wide plateaus on all tasks. "
                  "TAN is not a tuning artifact.\n")
    elif len(params_failing) == 0:
        md.append("**Gate 4:** ⚠️ Proceed with caution. Some isolated (param, task) cells "
                  "have narrow plateaus, but no parameter fails on a majority of tasks. "
                  "Document the narrow cells as task-specific sensitivities.\n")
    else:
        md.append(f"**Gate 4:** ❌ HALT — parameter(s) {params_failing} show narrow peaks "
                  f"on a majority of tasks. TAN is sensitive to these parameters and "
                  f"requires careful tuning. Revise the model or add adaptive mechanisms.\n")

    with open(RES_DIR / "plateau_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    log.info(f"Markdown report saved: {RES_DIR / 'plateau_summary.md'}")

    # ============================================================
    # 6. Final summary print
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 4 — PARAMETER ROBUSTNESS: COMPLETE")
    print("=" * 60)
    print(f"Output dir: {OUT_DIR}")
    print()
    print("Per-parameter plateau width (fraction of sweep range, pass = ≥50%):")
    for p in ['lambda', 'beta', 'W', 'theta']:
        sym = {'lambda': 'λ', 'beta': 'β', 'W': 'W', 'theta': 'θ'}[p]
        cells = []
        n_pass = 0
        for task in ['A', 'B', 'C', 'D']:
            w = plateau_df[(plateau_df['param'] == p) &
                           (plateau_df['task'] == task)]['plateau_width_fraction'].values[0]
            ok = plateau_df[(plateau_df['param'] == p) &
                            (plateau_df['task'] == task)]['pass_50pct'].values[0]
            icon = '✓' if ok else '✗'
            cells.append(f"{task}:{w*100:.0f}%{icon}")
            if ok:
                n_pass += 1
        print(f"  {sym}: " + "  ".join(cells) + f"  ({n_pass}/4 pass)")
    print()
    n_pass_total = int(plateau_df['pass_50pct'].sum())
    n_total = len(plateau_df)
    print(f"Total: {n_pass_total}/{n_total} cells pass (need all to PROCEED)")
    if n_pass_total == n_total:
        print(f"Gate 4 verdict: ✅ PROCEED to Part 5")
    elif n_pass_total >= n_total - 2:
        print(f"Gate 4 verdict: ⚠️ Proceed with caution")
    else:
        print(f"Gate 4 verdict: ❌ HALT — tuning sensitivity detected")
    print("=" * 60)


if __name__ == "__main__":
    main()
