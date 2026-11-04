# Exact selected source bodies; original file is never imported.
import numpy as np

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
