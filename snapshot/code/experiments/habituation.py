"""
TAN Neuron Paper - All Experiments (English version)
====================================================
- Experiment A: Basic dynamics (step light signal)
- Experiment B: Noisy spiking comparison (TAN vs LIF)
- Experiment C: Phototactic navigation (1D track)
- Experiment D: Noisy-environment robustness
- Ablation: gating / asymmetric ReLU / window size
- Metrics: quantitative bar charts
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os, json

# ============ Font configuration ============
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# ============ Output directory ============
FIG_DIR = "/home/z/my-project/download/figures_en"
os.makedirs(FIG_DIR, exist_ok=True)

# ============ Color configuration (academic minimalist) ============
C_TAN = '#1A1A1A'        # Black - TAN
C_LIF = '#9E9E9E'        # Gray - LIF
C_INPUT = '#F4A300'      # Orange - input
C_THRESHOLD = '#C0392B'  # Red - threshold
C_ACTIVATION = '#1F4E79' # Deep blue - activation
C_ACCENT = '#2E86AB'     # Mid blue
C_LIGHT = '#A8C5DA'      # Light blue
C_OPTIMAL = '#F4A300'    # Orange - optimal

np.random.seed(42)

# ============================================================
# 1. Neuron definitions
# ============================================================
class LIFNeuron:
    """Classical Leaky Integrate-and-Fire neuron."""
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
    """
    Temporal Attention Neuron (TAN).
    - Surprise S_t = x_t - mean(H_t)
    - Asymmetric rectification S_t = max(0, S_t - tau)  (tau = noise_tolerance)
    - Attention alpha = softmax(beta * Q * K)
    - Gating A_t = tanh(S_t) * C_t
    - Non-Markovian membrane evolution h_t = lambda * h_{t-1} + A_t
    """
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
            # Asymmetric rectification + noise-tolerance dead zone
            S_t = np.maximum(0, S_t - self.noise_tolerance)
        else:
            # Symmetric (preserve sign)
            S_t = S_t - (np.sign(S_t) * self.noise_tolerance if abs(S_t) > self.noise_tolerance else 0)
            S_t = S_t

        Q = S_t * self.W_q
        K = self.history * self.W_k
        V = self.history * self.W_v

        attention_scores = Q * K
        attention_weights = self.softmax(attention_scores)

        C_t = np.sum(attention_weights * V)

        if self.use_gating:
            if self.use_asym_rect:
                gating = np.tanh(S_t)
            else:
                gating = np.tanh(abs(S_t))
            A_t = gating * C_t
        else:
            # No gating: directly use attention output
            A_t = C_t * 0.5

        self.h = self.lambda_leak * self.h + A_t

        if self.h > self.threshold:
            spike = 1
            self.h = 0.0
        else:
            spike = 0
        return spike


# ============================================================
# Experiment A: Basic dynamics (step light signal)
# ============================================================
def run_experiment_A():
    print("[Exp A] Basic dynamics ...")
    class TANBase:
        def __init__(self, window_size=10, threshold=0.3):
            self.window_size = window_size
            self.W_q = np.array([2.0])
            self.W_k = np.array([1.0])
            self.W_v = np.array([1.0])
            self.threshold = threshold
            self.history = np.zeros(window_size)
        def softmax(self, x):
            e_x = np.exp(x - np.max(x))
            return e_x / (e_x.sum(axis=0) + 1e-9)
        def forward(self, current_input):
            self.history = np.roll(self.history, -1)
            self.history[-1] = current_input
            surprise = current_input - np.mean(self.history)
            Q = surprise * self.W_q
            K = self.history * self.W_k
            V = self.history * self.W_v
            att = self.softmax(Q * K)
            base = np.sum(att * V)
            act = base * abs(surprise)
            spike = 1 if act > self.threshold else 0
            return spike, act, att, surprise

    neuron = TANBase(window_size=10)
    signal_input = np.concatenate([np.zeros(5), np.ones(25)])
    spikes, activations, surprises = [], [], []

    for x in signal_input:
        s, a, _, surp = neuron.forward(x)
        spikes.append(s); activations.append(a); surprises.append(surp)

    fig, axs = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True,
                            constrained_layout=True)
    axs[0].plot(signal_input, drawstyle='steps-post', color=C_INPUT,
                linewidth=2, label="Light Stimulus")
    axs[0].set_ylabel("Intensity")
    axs[0].set_title("Experiment A: TAN Neuron Response to Step Input")
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend(loc='upper right')

    axs[1].plot(activations, color=C_ACTIVATION, linewidth=2,
                label="Internal Activation $A_t$")
    axs[1].plot(surprises, color=C_ACCENT, linewidth=1.3, linestyle=':',
                alpha=0.8, label="Surprise $S_t$")
    axs[1].axhline(y=neuron.threshold, color=C_THRESHOLD, linestyle='--',
                   linewidth=1.3, label=f"Spike Threshold ({neuron.threshold})")
    axs[1].set_ylabel("Activation")
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend(loc='upper right')

    axs[2].stem(spikes, linefmt='k-', markerfmt='ko', basefmt=" ",
                label="Action Potentials (Spikes)")
    axs[2].set_xlabel("Time Step")
    axs[2].set_ylabel("Spike Output")
    axs[2].set_yticks([0, 1])
    axs[2].grid(True, linestyle='--', alpha=0.5)
    axs[2].legend(loc='upper right')

    out_path = os.path.join(FIG_DIR, "fig1_experiment_A.png")
    plt.savefig(out_path, dpi=200, facecolor='white')
    plt.close()

    spike_arr = np.array(spikes)
    transition_idx = 5
    early_spikes = spike_arr[transition_idx:transition_idx+3].sum()
    total_spikes = spike_arr.sum()
    habit_period = spike_arr[transition_idx+10:].sum()
    return {
        "total_spikes": int(total_spikes),
        "early_burst_spikes": int(early_spikes),
        "habituated_spikes": int(habit_period),
        "habituation_ratio": float(habit_period / max(total_spikes, 1))
    }


# ============================================================
# Experiment B: Noisy spiking comparison (TAN vs LIF)
# ============================================================
def run_experiment_B():
    print("[Exp B] Noisy spiking comparison ...")
    np.random.seed(42)
    time_steps = 60
    base_signal = np.concatenate([np.zeros(15), np.ones(45) * 1.5])
    noise = np.random.normal(0, 0.4, time_steps)
    noisy_signal = np.clip(base_signal + noise, 0, None)

    lif = LIFNeuron(lambda_leak=0.8, threshold=2.0)
    tan = TANNeuron(window_size=10, lambda_leak=0.8, threshold=1.0,
                    noise_tolerance=0.0, use_gating=True, use_asym_rect=False)

    lif_spikes, lif_h = [], []
    tan_spikes, tan_h = [], []

    for x in noisy_signal:
        ls = lif.forward(x)
        lif_spikes.append(ls); lif_h.append(lif.h)
        ts = tan.forward(x)
        tan_spikes.append(ts); tan_h.append(tan.h)

    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True,
                            constrained_layout=True)

    axs[0].plot(noisy_signal, color=C_INPUT, drawstyle='steps-post',
                linewidth=1.5, alpha=0.9, label="Noisy Light Stimulus")
    axs[0].plot(base_signal, color='gray', linestyle='--', alpha=0.6,
                linewidth=1.5, label="Clean Ground Truth")
    axs[0].set_title("Experiment B: TAN vs LIF Spiking under Noisy Stimulus")
    axs[0].set_ylabel("Intensity")
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend(loc='upper right')

    axs[1].plot(lif_h, color=C_LIF, alpha=0.7, linewidth=1.5,
                label="LIF Membrane Potential $h_t$")
    lif_spike_times = np.where(np.array(lif_spikes) == 1)[0]
    for t in lif_spike_times:
        axs[1].axvline(x=t, color=C_THRESHOLD, linestyle='-', alpha=0.6,
                       linewidth=1.0)
    axs[1].plot([], [], color=C_THRESHOLD, linestyle='-', label='LIF Spikes')
    axs[1].set_ylabel("LIF Dynamics")
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend(loc='upper right')

    axs[2].plot(tan_h, color=C_ACTIVATION, alpha=0.7, linewidth=1.5,
                label="TAN Membrane Potential $h_t$")
    tan_spike_times = np.where(np.array(tan_spikes) == 1)[0]
    for t in tan_spike_times:
        axs[2].axvline(x=t, color='black', linestyle='-', linewidth=1.5)
    axs[2].plot([], [], color='black', linewidth=1.5, label='TAN Spikes (Filtered)')
    axs[2].set_ylabel("TAN Dynamics")
    axs[2].set_xlabel("Time Step")
    axs[2].grid(True, linestyle='--', alpha=0.5)
    axs[2].legend(loc='upper right')

    out_path = os.path.join(FIG_DIR, "fig2_experiment_B.png")
    plt.savefig(out_path, dpi=200, facecolor='white')
    plt.close()

    lif_rate = np.mean(lif_spikes)
    tan_rate = np.mean(tan_spikes)
    return {
        "lif_spike_count": int(lif_spike_times.shape[0]),
        "tan_spike_count": int(tan_spike_times.shape[0]),
        "lif_rate": float(lif_rate),
        "tan_rate": float(tan_rate),
        "noise_suppression_ratio": float(lif_rate / max(tan_rate, 1e-9))
    }


# ============================================================
# Experiment C: Phototactic navigation (1D track)
# ============================================================
def run_experiment_C():
    print("[Exp C] Phototactic navigation ...")
    np.random.seed(42)
    time_steps = 300

    class Bioton:
        def __init__(self, brain_type='TAN', start_pos=50.0):
            if brain_type == 'TAN':
                self.brain = TANNeuron(window_size=5, lambda_leak=0.5,
                                       threshold=0.5, noise_tolerance=0.0,
                                       use_gating=True, use_asym_rect=True)
            else:
                self.brain = LIFNeuron(lambda_leak=0.5, threshold=2.0)
            self.pos = start_pos
            self.velocity = 0.0
            self.pos_log = []
            self.spike_log = []
        def step(self, light_intensity):
            spike = self.brain.forward(light_intensity)
            friction = 0.6
            boost = 0.8
            self.velocity = friction * self.velocity + boost * spike
            self.pos += self.velocity
            self.pos_log.append(self.pos)
            self.spike_log.append(spike)

    def get_light(x):
        return 10.0 * np.exp(-((x - 80.0)**2) / (2 * 15.0**2))

    agent_lif = Bioton(brain_type='LIF', start_pos=50.0)
    agent_tan = Bioton(brain_type='TAN', start_pos=50.0)

    for t in range(time_steps):
        agent_lif.step(get_light(agent_lif.pos))
        agent_tan.step(get_light(agent_tan.pos))

    fig, axs = plt.subplots(3, 1, figsize=(9, 9),
                            gridspec_kw={'height_ratios': [1, 2, 1]},
                            sharex=False, constrained_layout=True)

    x_space = np.linspace(0, 130, 500)
    axs[0].plot(x_space, get_light(x_space), color='gold', linewidth=2.5,
                label="Light Source Distribution")
    axs[0].axvline(x=80, color=C_OPTIMAL, linestyle='--', linewidth=1.5,
                   label="Optimal Zone (x=80)")
    axs[0].set_title("Experiment C: TAN vs LIF Phototactic Navigation")
    axs[0].set_ylabel("Light Intensity")
    axs[0].set_xlabel("1D Spatial Position")
    axs[0].legend(loc='upper left')
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].scatter([agent_lif.pos], [get_light(agent_lif.pos)], color=C_LIF,
                   s=80, zorder=5, label='LIF Final', marker='v')
    axs[0].scatter([agent_tan.pos], [get_light(agent_tan.pos)], color=C_TAN,
                   s=80, zorder=5, label='TAN Final')
    axs[0].legend(loc='upper left')

    time_axis = np.arange(time_steps)
    axs[1].plot(time_axis, agent_lif.pos_log, color=C_LIF, linewidth=1.8,
                label="LIF Agent Trajectory")
    axs[1].plot(time_axis, agent_tan.pos_log, color=C_TAN, linewidth=1.8,
                label="TAN Agent Trajectory")
    axs[1].axhline(y=80, color=C_OPTIMAL, linestyle='--', linewidth=1.5,
                   label="Optimal Zone (x=80)")
    axs[1].set_ylabel("Position")
    axs[1].set_xlabel("Time Step")
    axs[1].legend(loc='lower right')
    axs[1].grid(True, linestyle='--', alpha=0.5)

    axs[2].eventplot(np.where(np.array(agent_lif.spike_log) == 1)[0],
                     lineoffsets=1.5, linelengths=0.5, color=C_LIF,
                     label='LIF Motor Spikes')
    axs[2].eventplot(np.where(np.array(agent_tan.spike_log) == 1)[0],
                     lineoffsets=0.5, linelengths=0.5, color=C_TAN,
                     label='TAN Motor Spikes')
    axs[2].set_yticks([0.5, 1.5])
    axs[2].set_yticklabels(['TAN', 'LIF'])
    axs[2].set_xlabel("Time Step")
    axs[2].set_title("Motor Neuron Spiking Activity")
    axs[2].grid(True, linestyle='--', alpha=0.5)
    axs[2].legend(loc='upper right')

    out_path = os.path.join(FIG_DIR, "fig3_experiment_C.png")
    plt.savefig(out_path, dpi=200, facecolor='white')
    plt.close()

    def final_err(pos):
        return abs(pos - 80.0)
    def convergence_steps(pos_log, tol=2.0):
        for i, p in enumerate(pos_log):
            if abs(p - 80.0) < tol:
                return i
        return time_steps
    return {
        "lif_final_pos": float(agent_lif.pos),
        "tan_final_pos": float(agent_tan.pos),
        "lif_final_err": float(final_err(agent_lif.pos)),
        "tan_final_err": float(final_err(agent_tan.pos)),
        "lif_convergence": int(convergence_steps(agent_lif.pos_log)),
        "tan_convergence": int(convergence_steps(agent_tan.pos_log)),
        "lif_total_spikes": int(sum(agent_lif.spike_log)),
        "tan_total_spikes": int(sum(agent_tan.spike_log))
    }


# ============================================================
# Experiment D: Noisy-environment robustness
# ============================================================
def run_experiment_D():
    print("[Exp D] Noisy-environment robustness ...")
    np.random.seed(42)
    time_steps = 250
    NOISE_LEVEL = 0.5

    class BiotonRobust:
        def __init__(self, brain_type='TAN', start_pos=50.0):
            if brain_type == 'TAN':
                self.brain = TANNeuron(window_size=5, lambda_leak=0.5,
                                       threshold=0.5, noise_tolerance=0.6,
                                       use_gating=True, use_asym_rect=True)
            else:
                self.brain = LIFNeuron(lambda_leak=0.5, threshold=2.5)
            self.pos = start_pos
            self.velocity = 0.0
            self.pos_log = []
            self.spike_log = []
        def step(self, light_intensity):
            spike = self.brain.forward(light_intensity)
            friction = 0.6
            boost = 0.8
            self.velocity = friction * self.velocity + boost * spike
            self.pos += self.velocity
            self.pos_log.append(self.pos)
            self.spike_log.append(spike)

    def get_clean(x):
        return 10.0 * np.exp(-((x - 80.0)**2) / (2 * 15.0**2))

    def get_noisy(x, noise_level=0.5):
        return np.maximum(0, get_clean(x) + np.random.normal(0, noise_level))

    agent_lif = BiotonRobust(brain_type='LIF', start_pos=50.0)
    agent_tan = BiotonRobust(brain_type='TAN', start_pos=50.0)

    for t in range(time_steps):
        agent_lif.step(get_noisy(agent_lif.pos, NOISE_LEVEL))
        agent_tan.step(get_noisy(agent_tan.pos, NOISE_LEVEL))

    fig, axs = plt.subplots(3, 1, figsize=(9, 9),
                            gridspec_kw={'height_ratios': [1, 2, 1]},
                            sharex=False, constrained_layout=True)

    x_space = np.linspace(0, 130, 500)
    noisy_sample = [get_noisy(x, NOISE_LEVEL) for x in x_space]
    axs[0].plot(x_space, noisy_sample, color='gold', alpha=0.4, linewidth=1,
                label=f"Noisy Environment (std={NOISE_LEVEL})")
    axs[0].plot(x_space, get_clean(x_space), color=C_INPUT, linewidth=2,
                linestyle='--', label="Clean Underlying Gradient")
    axs[0].axvline(x=80, color=C_THRESHOLD, linestyle=':', linewidth=1.5,
                   label="Optimal Zone (x=80)")
    axs[0].set_title("Experiment D: Navigation Robustness in Noisy Environment")
    axs[0].set_ylabel("Sensory Input")
    axs[0].set_xlabel("1D Spatial Position")
    axs[0].legend(loc='upper left')
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].scatter([agent_lif.pos], [get_clean(agent_lif.pos)], color=C_LIF,
                   s=80, zorder=5, marker='v')
    axs[0].scatter([agent_tan.pos], [get_clean(agent_tan.pos)], color=C_TAN,
                   s=80, zorder=5)

    time_axis = np.arange(time_steps)
    axs[1].plot(time_axis, agent_lif.pos_log, color=C_LIF, linewidth=1.8,
                label="LIF Agent Trajectory")
    axs[1].plot(time_axis, agent_tan.pos_log, color=C_TAN, linewidth=1.8,
                label="TAN Agent Trajectory")
    axs[1].axhline(y=80, color=C_OPTIMAL, linestyle='--', linewidth=1.5,
                   label="Optimal Zone (x=80)")
    axs[1].set_ylabel("Position")
    axs[1].set_xlabel("Time Step")
    axs[1].legend(loc='lower right')
    axs[1].grid(True, linestyle='--', alpha=0.5)

    axs[2].eventplot(np.where(np.array(agent_lif.spike_log) == 1)[0],
                     lineoffsets=1.5, linelengths=0.5, color=C_LIF,
                     label='LIF Motor Spikes')
    axs[2].eventplot(np.where(np.array(agent_tan.spike_log) == 1)[0],
                     lineoffsets=0.5, linelengths=0.5, color=C_TAN,
                     label='TAN Motor Spikes')
    axs[2].set_yticks([0.5, 1.5])
    axs[2].set_yticklabels(['TAN', 'LIF'])
    axs[2].set_xlabel("Time Step")
    axs[2].set_title("Motor Neuron Spiking Activity")
    axs[2].grid(True, linestyle='--', alpha=0.5)
    axs[2].legend(loc='upper right')

    out_path = os.path.join(FIG_DIR, "fig4_experiment_D.png")
    plt.savefig(out_path, dpi=200, facecolor='white')
    plt.close()

    def convergence_steps(pos_log, tol=2.0):
        for i, p in enumerate(pos_log):
            if abs(p - 80.0) < tol:
                return i
        return time_steps
    return {
        "lif_final_pos": float(agent_lif.pos),
        "tan_final_pos": float(agent_tan.pos),
        "lif_final_err": float(abs(agent_lif.pos - 80.0)),
        "tan_final_err": float(abs(agent_tan.pos - 80.0)),
        "lif_convergence": int(convergence_steps(agent_lif.pos_log)),
        "tan_convergence": int(convergence_steps(agent_tan.pos_log)),
        "lif_total_spikes": int(sum(agent_lif.spike_log)),
        "tan_total_spikes": int(sum(agent_tan.spike_log))
    }


# ============================================================
# Ablation: gating / asymmetric rect / window size
# ============================================================
def run_ablation():
    print("[Ablation] Ablation experiments ...")
    np.random.seed(42)

    time_steps = 250
    NOISE_LEVEL = 0.5

    def get_clean(x):
        return 10.0 * np.exp(-((x - 80.0)**2) / (2 * 15.0**2))

    def get_noisy(x, noise_level=0.5):
        return np.maximum(0, get_clean(x) + np.random.normal(0, noise_level))

    class BiotonAblation:
        def __init__(self, tan_kwargs, start_pos=50.0, brain_type='TAN'):
            if brain_type == 'TAN':
                self.brain = TANNeuron(**tan_kwargs)
            else:
                self.brain = LIFNeuron(lambda_leak=0.5, threshold=2.5)
            self.pos = start_pos
            self.velocity = 0.0
            self.pos_log = []
            self.spike_log = []
        def step(self, light_intensity):
            spike = self.brain.forward(light_intensity)
            self.velocity = 0.6 * self.velocity + 0.8 * spike
            self.pos += self.velocity
            self.pos_log.append(self.pos)
            self.spike_log.append(spike)

    variants = {
        "Full TAN": dict(window_size=5, lambda_leak=0.5, threshold=0.5,
                          noise_tolerance=0.6, use_gating=True,
                          use_asym_rect=True),
        "No Gating": dict(window_size=5, lambda_leak=0.5, threshold=0.5,
                              noise_tolerance=0.6, use_gating=False,
                              use_asym_rect=True),
        "Symmetric Rect.": dict(window_size=5, lambda_leak=0.5,
                                threshold=0.5, noise_tolerance=0.6,
                                use_gating=True, use_asym_rect=False),
        "LIF (Baseline)": None,
    }

    results = {}
    trajectories = {}
    for name, kw in variants.items():
        np.random.seed(42)
        if kw is None:
            agent = BiotonAblation(None, brain_type='LIF')
        else:
            agent = BiotonAblation(kw, brain_type='TAN')
        for t in range(time_steps):
            agent.step(get_noisy(agent.pos, NOISE_LEVEL))
        trajectories[name] = agent.pos_log
        conv = next((i for i, p in enumerate(agent.pos_log) if abs(p-80)<2), time_steps)
        results[name] = {
            "final_err": float(abs(agent.pos - 80.0)),
            "convergence": int(conv),
            "total_spikes": int(sum(agent.spike_log))
        }

    fig, ax = plt.subplots(1, 1, figsize=(9, 5), constrained_layout=True)
    colors = [C_TAN, C_ACCENT, C_LIGHT, C_LIF]
    styles = ['-', '--', '-.', ':']
    for (name, traj), c, s in zip(trajectories.items(), colors, styles):
        ax.plot(np.arange(time_steps), traj, color=c, linestyle=s, linewidth=1.8,
                label=f"{name} (final err = {results[name]['final_err']:.1f})")
    ax.axhline(y=80, color=C_OPTIMAL, linestyle='--', linewidth=1.2, alpha=0.7,
               label="Optimal Zone (x=80)")
    ax.set_title("Ablation: Effect of Component Removal on Navigation (noisy env, std=0.5)")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Agent Position")
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    out_path = os.path.join(FIG_DIR, "fig5_ablation.png")
    plt.savefig(out_path, dpi=200, facecolor='white')
    plt.close()

    np.random.seed(42)
    window_sizes = [3, 5, 10, 15, 20]
    window_results = {}
    for w in window_sizes:
        np.random.seed(42)
        kw = dict(window_size=w, lambda_leak=0.5, threshold=0.5,
                  noise_tolerance=0.6, use_gating=True, use_asym_rect=True)
        agent = BiotonAblation(kw, brain_type='TAN')
        for t in range(time_steps):
            agent.step(get_noisy(agent.pos, NOISE_LEVEL))
        conv = next((i for i, p in enumerate(agent.pos_log) if abs(p-80)<2), time_steps)
        window_results[w] = {
            "final_err": float(abs(agent.pos - 80.0)),
            "convergence": int(conv),
            "total_spikes": int(sum(agent.spike_log))
        }
    return {"component_ablation": results, "window_ablation": window_results}


# ============================================================
# Quantitative metrics bar chart
# ============================================================
def plot_metrics(exp_b_metrics, exp_c_metrics, exp_d_metrics):
    print("[Metrics] Plotting quantitative metrics ...")
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)

    labels = ['LIF', 'TAN']
    rates_b = [exp_b_metrics['lif_rate']*100, exp_b_metrics['tan_rate']*100]
    bars = axs[0].bar(labels, rates_b, color=[C_LIF, C_TAN], width=0.5,
                      edgecolor='black', linewidth=0.7)
    axs[0].set_title("(a) Exp B Spike Rate")
    axs[0].set_ylabel("Spike Rate (%)")
    for bar, v in zip(bars, rates_b):
        axs[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{v:.1f}%', ha='center', fontsize=10)
    axs[0].grid(True, linestyle='--', alpha=0.4, axis='y')

    conv_c = [exp_c_metrics['lif_convergence'], exp_c_metrics['tan_convergence']]
    bars = axs[1].bar(labels, conv_c, color=[C_LIF, C_TAN], width=0.5,
                      edgecolor='black', linewidth=0.7)
    axs[1].set_title("(b) Exp C Convergence Steps")
    axs[1].set_ylabel("Steps (lower is better)")
    for bar, v in zip(bars, conv_c):
        axs[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{v}', ha='center', fontsize=10)
    axs[1].grid(True, linestyle='--', alpha=0.4, axis='y')

    err_d = [exp_d_metrics['lif_final_err'], exp_d_metrics['tan_final_err']]
    bars = axs[2].bar(labels, err_d, color=[C_LIF, C_TAN], width=0.5,
                      edgecolor='black', linewidth=0.7)
    axs[2].set_title("(c) Exp D Final-Position Error")
    axs[2].set_ylabel("|x - 80| (lower is better)")
    for bar, v in zip(bars, err_d):
        axs[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{v:.1f}', ha='center', fontsize=10)
    axs[2].grid(True, linestyle='--', alpha=0.4, axis='y')

    out_path = os.path.join(FIG_DIR, "fig6_metrics.png")
    plt.savefig(out_path, dpi=200, facecolor='white')
    plt.close()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    all_metrics = {}
    all_metrics['experiment_A'] = run_experiment_A()
    all_metrics['experiment_B'] = run_experiment_B()
    all_metrics['experiment_C'] = run_experiment_C()
    all_metrics['experiment_D'] = run_experiment_D()
    all_metrics['ablation'] = run_ablation()
    plot_metrics(all_metrics['experiment_B'],
                 all_metrics['experiment_C'],
                 all_metrics['experiment_D'])

    with open(os.path.join(FIG_DIR, "metrics.json"), 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    print("\n========== All experiments complete ==========")
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))
    print(f"\nFigures output directory: {FIG_DIR}")
