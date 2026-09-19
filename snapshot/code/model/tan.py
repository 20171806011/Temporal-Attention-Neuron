"""TAN Neuron Model Implementation
"""
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


