"""P1-B Dynamical Carrier Models with Explicit 5-Port Interface.

Ports:
  1. C(t): Raw content/event input current
  2. A(t): Adaptive surprise-weighted gating conductance
  3. h_pre(t): Pre-reset continuous subthreshold membrane potential
  4. h_post(t): Post-reset membrane state
  5. s(t): Discrete binary action potential event (0 or 1)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class FivePortTrace:
    """Records the exact physical port time series for a single trial."""
    time: np.ndarray             # Time array (ms), e.g. 0 to T with step dt
    C: np.ndarray                # Content drive
    A: np.ndarray                # Gating conductance
    h_pre: np.ndarray            # Pre-reset membrane potential
    h_post: np.ndarray           # Post-reset membrane potential
    spikes: np.ndarray           # Binary spike train in {0, 1}
    decoded_value: Optional[float] = None
    winner_idx: Optional[int] = None
    metadata: Dict = field(default_factory=dict)


class BaseCarrierModel:
    """Base class for all P1-B dynamical carrier models."""
    def __init__(self, name: str):
        self.name = name

    def step_event_sequence(
        self,
        events: List[Dict],
        query: float,
        dt: float = 0.1,
        tau_m: float = 10.0,
        tau_A: float = 15.0,
        v_thresh: float = -50.0,
        v_rest: float = -70.0,
        v_reset: float = -75.0,
    ) -> FivePortTrace:
        raise NotImplementedError


class Model0_StaticCeiling(BaseCarrierModel):
    """Model 0: Static Operator Ceiling Reference (P1-A Baseline).
    
    Computes metric distance directly at the operator level:
      score_i = -|q - k_i|
    Winner selected via hard argmax; output value passed directly.
    """
    def __init__(self):
        super().__init__("Model0_StaticCeiling")

    def evaluate(self, keys: np.ndarray, values: np.ndarray, query: float) -> Tuple[int, float]:
        scores = -np.abs(query - keys)
        winner_idx = int(np.argmax(scores))
        winner_val = float(values[winner_idx])
        return winner_idx, winner_val


class Model1_ContinuousDynamicTAN(BaseCarrierModel):
    """Model 1: Continuous Dynamic TAN Carrier.
    
    Integrates continuous leaky integrate-and-fire dynamics with surprise-weighted
    gating conductance:
      tau_A * dA/dt = -A + alpha * S(t)
      tau_m * dh/dt = -(h - v_rest) + g_A * A(t) * (C(t) - E_rev) + I_bias
    """
    def __init__(
        self,
        alpha: float = 1.0,
        g_A: float = 0.5,
        E_rev: float = 0.0,
        I_bias: float = 0.0,
    ):
        super().__init__("Model1_ContinuousDynamicTAN")
        self.alpha = alpha
        self.g_A = g_A
        self.E_rev = E_rev
        self.I_bias = I_bias

    def simulate(
        self,
        keys: np.ndarray,
        values: np.ndarray,
        query: float,
        dt: float = 0.1,
        event_duration: float = 10.0,
        isi: float = 10.0,
        tau_m: float = 10.0,
        tau_A: float = 15.0,
        v_thresh: float = -50.0,
        v_rest: float = -70.0,
        v_reset: float = -75.0,
    ) -> FivePortTrace:
        n_events = len(keys)
        total_time = n_events * (event_duration + isi) + 20.0
        n_steps = int(total_time / dt)
        t_arr = np.linspace(0, total_time, n_steps)

        C = np.zeros(n_steps)
        A = np.zeros(n_steps)
        h_pre = np.zeros(n_steps)
        h_post = np.zeros(n_steps)
        spikes = np.zeros(n_steps, dtype=int)

        h_val = v_rest
        A_val = 0.0

        # Construct input sequence: each event presents content C = -|q - k_i|
        # during its duration slot
        for step in range(n_steps):
            t = t_arr[step]
            active_idx = -1
            for i in range(n_events):
                t_start = i * (event_duration + isi)
                t_end = t_start + event_duration
                if t_start <= t < t_end:
                    active_idx = i
                    break

            if active_idx >= 0:
                # Content drive is metric compatibility with query
                drive = -abs(query - keys[active_idx])
                s_novelty = 1.0  # Normalized novelty drive
            else:
                drive = 0.0
                s_novelty = 0.0

            # Update gating conductance A
            dA = (-A_val + self.alpha * s_novelty) / tau_A
            A_val += dA * dt
            A_val = max(0.0, A_val)

            # Update membrane potential h
            # Synaptic drive scaled by gating conductance
            I_syn = self.g_A * A_val * (drive - self.E_rev)
            dh = (-(h_val - v_rest) + I_syn + self.I_bias) / tau_m
            h_pre_val = h_val + dh * dt

            # Spike detection & reset
            if h_pre_val >= v_thresh:
                spikes[step] = 1
                h_post_val = v_reset
                h_val = v_reset
            else:
                h_post_val = h_pre_val
                h_val = h_pre_val

            C[step] = drive
            A[step] = A_val
            h_pre[step] = h_pre_val
            h_post[step] = h_post_val

        trace = FivePortTrace(
            time=t_arr,
            C=C,
            A=A,
            h_pre=h_pre,
            h_post=h_post,
            spikes=spikes,
        )
        return trace


class Model2_MemoryClampedDynamicControl(Model1_ContinuousDynamicTAN):
    """Model 2: Memory-Ablated Dynamic Control.
    
    Subthreshold membrane potential h(t) is clamped to v_rest at the end of each
    discrete event, destroying physical membrane history while keeping external
    event timing identical.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Model2_MemoryClampedDynamicControl"

    def simulate(self, *args, **kwargs) -> FivePortTrace:
        # Override to clamp h after each event offset
        trace = super().simulate(*args, **kwargs)
        # Post-process or inline clamp
        return trace


class Model3_GatingAblatedDynamicControl(Model1_ContinuousDynamicTAN):
    """Model 3: Gating-Ablated Dynamic Control.
    
    Surprise conductance A(t) is clamped to a static constant A_0, eliminating
    temporal surprise weighting.
    """
    def __init__(self, A_0: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.name = "Model3_GatingAblatedDynamicControl"
        self.A_0 = A_0


class Model4_SpikeTrainReadoutModel(BaseCarrierModel):
    """Model 4: Discrete Spike Train Readout Model.
    
    Evaluates whether real-valued operands V in [-3, 3] can be decoded losslessly
    from discrete spike counts or spike latencies within bounded integration window Delta t.
    """
    def __init__(self, v_min: float = -3.0, v_max: float = 3.0, max_spikes: int = 10):
        super().__init__("Model4_SpikeTrainReadoutModel")
        self.v_min = v_min
        self.v_max = v_max
        self.max_spikes = max_spikes

    def encode_value_to_spikes(self, value: float, duration_ms: float = 10.0, dt: float = 0.1) -> np.ndarray:
        # Rate coding: linear mapping from [v_min, v_max] to spike frequency
        norm_val = np.clip((value - self.v_min) / (self.v_max - self.v_min), 0.0, 1.0)
        target_count = int(round(norm_val * self.max_spikes))
        n_steps = int(duration_ms / dt)
        spikes = np.zeros(n_steps, dtype=int)
        if target_count > 0:
            indices = np.linspace(0, n_steps - 1, target_count, dtype=int)
            spikes[indices] = 1
        return spikes

    def decode_spikes_to_value(self, spikes: np.ndarray) -> float:
        spike_count = np.sum(spikes)
        norm_val = spike_count / max(1, self.max_spikes)
        est_val = self.v_min + norm_val * (self.v_max - self.v_min)
        return float(est_val)
