"""Conforming Implementation of Neural State Models for P1-B.
Governing Specification: P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md (SHA-256: 07D58A2DE773EA13E8FD5C16901CB8E8A370219D4793F4A7F9BACA23C747A82B)
Gate Status: G1-Design APPROVED by Codex on September 19, 2026.

Dynamically closed 3-variable carrier state:
    z(t) = [u(t), A(t), r(t)]^T in R x [0, 5.0] x [0, 2.0 ms]
Euler step size h = 0.1 ms.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
from scipy.stats import norm

# Physical Euler Carrier Constants (Protocol V2.6 Section 2)
H: float = 0.1          # ms
TAU_M: float = 10.0     # ms
TAU_A: float = 15.0     # ms
TAU_REF: float = 2.0    # ms
U_REST: float = 0.0
THETA_TH: float = 1.0
U_RESET: float = 0.0
ALPHA: float = 1.0
A_MAX: float = 5.0
W_K: float = 1.0
W_V: float = 1.0
G_A: float = 1.0
I_BIAS: float = 0.0
MU_BG: float = 1.4

# Timeline Constants (Section 3)
T_EVENT: float = 10.0   # ms (100 steps)
T_ISI: float = 10.0     # ms (100 steps)
T_SLOT: float = 20.0    # ms (200 steps)
W_SLOTS: int = 5        # slots
T_ENC: float = 100.0    # ms (1000 steps)
N_ENC_STEPS: int = 1000 # steps
T_CUE: float = 10.0     # ms (100 steps)
N_CUE_STEPS: int = 100  # steps

# Practical Delivery Tolerance
DELTA_V_TOL: float = 0.50

@dataclass
class CarrierState:
    u: float = 0.0
    A: float = 0.0
    r: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.u, self.A, self.r], dtype=np.float64)

def euler_step(
    u: float,
    A: float,
    r: float,
    I_content: float,
    novelty: float,
    I_query: float,
    h: float = H,
    tau_m: float = TAU_M,
    tau_A: float = TAU_A,
    tau_ref: float = TAU_REF,
    g_A: float = G_A,
    alpha: float = ALPHA,
    A_max: float = A_MAX,
    theta_th: float = THETA_TH,
    u_reset: float = U_RESET,
    u_rest: float = U_REST,
    I_bias: float = I_BIAS,
    clamp_A_const: Optional[float] = None
) -> Tuple[float, float, float, int, Optional[float]]:
    """Single continuous Euler recurrence step (Section 2.2).
    
    Returns:
        (u_next, A_next, r_next, spike, u_cand)
        where spike in {0, 1} and u_cand is the pre-reset candidate if r == 0, else None.
    """
    # Gating conductance evolution (or clamped constant)
    if clamp_A_const is not None:
        A_next = clamp_A_const
    else:
        dA = (-A + alpha * novelty) / tau_A * h
        A_next = min(A_max, max(0.0, A + dA))

    # Membrane & refractory evolution
    if r > 0.0:
        r_next = max(0.0, r - h)
        u_next = u_reset
        spike = 0
        u_cand = None
    else:
        I_syn = g_A * A * I_content
        du = (-(u - u_rest) + I_syn + I_query + I_bias) / tau_m * h
        u_cand = u + du
        if u_cand >= theta_th:
            spike = 1
            u_next = u_reset
            r_next = tau_ref
        else:
            spike = 0
            u_next = u_cand
            r_next = 0.0

    return u_next, A_next, r_next, spike, u_cand

def simulate_encoding(
    keys: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    model_type: str,
    A_0: float = 0.0,
    w_K: float = W_K,
    w_V: float = W_V,
    mu_bg: float = MU_BG,
    record_trace: bool = False
) -> Tuple[CarrierState, Optional[Dict[str, np.ndarray]], float]:
    """Simulate streaming encoding for 5 slots (100 ms total, 1000 steps).
    
    Intervention types (Section 4):
        'M1': Retained membrane, dynamic A
        'M2a': Clamped membrane (u<-0, r<-0 post-event), dynamic A
        'M2b': Retained membrane, constant A = A_0
        'M2c': Full-state reset (u<-0, A<-0, r<-0 post-event)
        'M2d': Clamped membrane (u<-0, r<-0 post-event), constant A = A_0
        'M4': Same physical dynamics as M1 during encoding
    
    Returns:
        (final_state, trace_dict_or_None, mean_A_encoding)
    """
    u, A, r = 0.0, (A_0 if model_type in ('M2b', 'M2d') else 0.0), 0.0
    clamp_A = A_0 if model_type in ('M2b', 'M2d') else None

    trace_u = [] if record_trace else None
    trace_A = [] if record_trace else None
    trace_r = [] if record_trace else None
    A_history = []

    steps_per_event = int(T_EVENT / H)  # 100 steps
    steps_per_isi = int(T_ISI / H)      # 100 steps

    for slot_idx in range(W_SLOTS):
        is_active = bool(mask[slot_idx])
        k_val = float(keys[slot_idx]) if is_active else 0.0
        v_val = float(values[slot_idx]) if is_active else 0.0

        # Event presentation phase [0, 10 ms)
        if is_active:
            I_content = w_K * k_val + w_V * v_val
            novelty = max(0.0, abs(k_val) - mu_bg)
        else:
            I_content = 0.0
            novelty = 0.0

        for _ in range(steps_per_event):
            A_history.append(A)
            if record_trace:
                trace_u.append(u)
                trace_A.append(A)
                trace_r.append(r)
            u, A, r, _, _ = euler_step(
                u, A, r, I_content, novelty, I_query=0.0,
                clamp_A_const=clamp_A
            )

        # Post-event intervention at active event offset
        if is_active:
            if model_type in ('M2a', 'M2d'):
                u = 0.0
                r = 0.0
            elif model_type == 'M2c':
                u = 0.0
                A = 0.0
                r = 0.0

        # ISI phase [10, 20 ms)
        for _ in range(steps_per_isi):
            A_history.append(A)
            if record_trace:
                trace_u.append(u)
                trace_A.append(A)
                trace_r.append(r)
            u, A, r, _, _ = euler_step(
                u, A, r, I_content=0.0, novelty=0.0, I_query=0.0,
                clamp_A_const=clamp_A
            )

    mean_A = float(np.mean(A_history))
    final_state = CarrierState(u=u, A=A, r=r)
    trace_dict = None
    if record_trace:
        trace_dict = {
            'u': np.array(trace_u, dtype=np.float64),
            'A': np.array(trace_A, dtype=np.float64),
            'r': np.array(trace_r, dtype=np.float64)
        }
    return final_state, trace_dict, mean_A

def simulate_delay(
    state: CarrierState,
    tau_delay: float,
    model_type: str,
    A_0: float = 0.0
) -> CarrierState:
    """Advance state through silence delay tau_delay in {0.0, 5.0, 10.0} ms."""
    if tau_delay <= 0.0:
        return CarrierState(u=state.u, A=state.A, r=state.r)

    n_steps = int(round(tau_delay / H))
    u, A, r = state.u, state.A, state.r
    clamp_A = A_0 if model_type in ('M2b', 'M2d') else None

    for _ in range(n_steps):
        u, A, r, _, _ = euler_step(
            u, A, r, I_content=0.0, novelty=0.0, I_query=0.0,
            clamp_A_const=clamp_A
        )
    return CarrierState(u=u, A=A, r=r)

def simulate_cue_and_readout(
    state_at_cue: CarrierState,
    query_q: float,
    model_type: str,
    g_dec: float = 1.0,
    b_dec: float = 0.0,
    A_0: float = 0.0
) -> Dict[str, Any]:
    """Simulate 10 ms cue window (100 steps) and evaluate readout.
    
    Total Readout Function (Section 5.1):
        At deadline step 100 (t = T_cue + 10.0 ms):
        u_readout = u_reset = 0.0 if r(T_deadline^-) > 0 else u_cand(T_deadline)
        V_hat = g_dec * u_readout + b_dec (query q excluded from affine map).
        
    Spiking Decoders (Section 5.2):
        Consumes strictly eligible spikes {s_1, ..., s_99}.
        s_100 (at t = 10.0 ms) is strictly excluded by half-open boundary.
        Count decoder: C in {1..5} -> {-2, -1, 0, 1, 2}. C=0 -> Silence/Abstain.
        Latency decoder: t_first in [0.1, 9.9] ms -> 20 bins over [-3, 3]. Silence -> Abstain.
    """
    u, A, r = state_at_cue.u, state_at_cue.A, state_at_cue.r
    clamp_A = A_0 if model_type in ('M2b', 'M2d') else None

    raw_spikes = []
    r_before_step = []
    u_cands = []

    for step_k in range(1, N_CUE_STEPS + 1):
        r_before_step.append(r)
        u, A, r, spike, u_cand = euler_step(
            u, A, r, I_content=0.0, novelty=0.0, I_query=1.0 * query_q,
            clamp_A_const=clamp_A
        )
        raw_spikes.append(spike)
        u_cands.append(u_cand)

    # 1. Analog Readout (M1, M2a, M2b, M2c, M2d)
    # Deadline step is 100 (index 99 in 0-based arrays)
    r_deadline_minus = r_before_step[99]
    u_cand_deadline = u_cands[99]

    if r_deadline_minus > 0.0:
        u_readout = U_RESET  # 0.0
    else:
        assert u_cand_deadline is not None, "Candidate voltage must be defined when r == 0"
        u_readout = float(u_cand_deadline)

    v_hat_analog = float(g_dec * u_readout + b_dec)

    # 2. Spiking Readout (M4)
    # Only steps 1..99 are eligible
    eligible_spikes = raw_spikes[:99]
    count = int(sum(eligible_spikes))

    # Count decoder
    count_map = {1: -2.0, 2: -1.0, 3: 0.0, 4: 1.0, 5: 2.0}
    if count == 0:
        abstained_count = True
        v_hat_count = 0.0
    else:
        abstained_count = False
        # Cap count at 5 if somehow exceeded
        v_hat_count = count_map.get(min(5, count), 2.0)

    # Latency decoder
    # 20 bins of width 0.5 ms over [0.0, 10.0) ms
    # 20 value reconstructions over [-3.0, 3.0] spaced by 0.30: -2.85 + k * 0.30
    first_spike_step = None
    for k_idx, spk in enumerate(eligible_spikes):
        if spk == 1:
            first_spike_step = k_idx + 1  # 1-indexed step in {1..99}
            break

    if first_spike_step is None:
        abstained_latency = True
        v_hat_latency = 0.0
        t_first = None
    else:
        abstained_latency = False
        t_first = first_spike_step * H  # in [0.1, 9.9] ms
        bin_idx = min(19, max(0, int(t_first / 0.5)))
        v_hat_latency = -2.85 + bin_idx * 0.30

    return {
        'u_readout': u_readout,
        'v_hat_analog': v_hat_analog,
        'v_hat_count': v_hat_count,
        'abstained_count': abstained_count,
        'v_hat_latency': v_hat_latency,
        'abstained_latency': abstained_latency,
        'spike_count': count,
        'raw_spikes': raw_spikes,
        'eligible_spikes': eligible_spikes,
        't_first': t_first,
        'r_deadline_minus': r_deadline_minus,
        'r_deadline_post': r
    }

def calibrate_m1_decoder_and_gate(
    calibration_trials: List[Any],
    rng_seed: int = 2026091999
) -> Tuple[float, float, float]:
    """Run calibration on 500 calibration trials (Section 5.1).
    
    Computes:
        1. A_0: unweighted mean of mean-encoding conductance across 500 trials
        2. (g_dec, b_dec): unweighted OLS regression on 1,000 M1 (u_readout, V_target) pairs
    
    Returns:
        (A_0, g_dec, b_dec)
    """
    delays = [0.0, 5.0, 10.0]
    A_means = []
    readouts = []
    targets = []

    for idx, trial in enumerate(calibration_trials):
        # Balanced round-robin delay
        delay = delays[idx % 3]

        # Simulate M1 encoding
        state_enc, _, mean_A = simulate_encoding(
            keys=trial.keys,
            values=trial.values,
            mask=trial.mask,
            model_type='M1'
        )
        A_means.append(mean_A)

        # Advance through delay
        state_cue = simulate_delay(state_enc, delay, model_type='M1')

        # Probe Role A
        q_A = float(trial.query_cues[0])
        v_target_A = float(trial.target_values[0])
        res_A = simulate_cue_and_readout(state_cue, q_A, model_type='M1')
        readouts.append(res_A['u_readout'])
        targets.append(v_target_A)

        # Probe Role B (independent repeated probe from identical encoded state)
        q_B = float(trial.query_cues[1])
        v_target_B = float(trial.target_values[1])
        res_B = simulate_cue_and_readout(state_cue, q_B, model_type='M1')
        readouts.append(res_B['u_readout'])
        targets.append(v_target_B)

    A_0 = float(np.mean(A_means))

    # Fit unweighted OLS: V_target = g_dec * u_readout + b_dec
    x = np.array(readouts, dtype=np.float64)
    y = np.array(targets, dtype=np.float64)

    # Standard OLS normal equations
    X = np.vstack([x, np.ones_like(x)]).T
    # Use lstsq for numerical stability
    sol, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    g_dec, b_dec = float(sol[0]), float(sol[1])

    return A_0, g_dec, b_dec

# Statistical Verification Functions (Section 7.4, 7.5)

def wilson_score_ci(x: int, n: int, alpha: float = 0.10) -> Tuple[float, float]:
    """Standard Wilson score confidence interval without continuity correction."""
    if n == 0:
        return 0.0, 0.0
    z = norm.ppf(1.0 - alpha / 2.0)
    p_hat = x / n
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2.0 * n)) / denom
    half = (z * math.sqrt(p_hat * (1.0 - p_hat) / n + (z**2) / (4.0 * n**2))) / denom
    return center - half, center + half

def newcombe_method10_paired_ci(
    n11: int,
    n10: int,
    n01: int,
    n00: int,
    alpha: float = 0.10
) -> Tuple[float, float]:
    """Newcombe Method 10 square-and-add confidence interval for paired proportions.
    
    Word-for-word matching to CRAN contingencytables 3.1.0:
    Newcombe_square_and_add_CI_paired_2x2.R
    """
    N_table = n11 + n10 + n01 + n00
    if N_table == 0:
        return 0.0, 0.0

    n1_plus = n11 + n10
    n_plus_1 = n11 + n01
    n2_plus = n01 + n00
    n_plus_2 = n10 + n00

    p1_hat = n1_plus / N_table
    p2_hat = n_plus_1 / N_table
    d_hat = p1_hat - p2_hat

    l1, u1 = wilson_score_ci(n1_plus, N_table, alpha)
    l2, u2 = wilson_score_ci(n_plus_1, N_table, alpha)

    n_prod = n1_plus * n2_plus * n_plus_1 * n_plus_2
    if n_prod == 0:
        psi = 0.0
    else:
        A = n11 * n00 - n10 * n01
        if A > N_table / 2.0:
            psi = (A - N_table / 2.0) / math.sqrt(n_prod)
        elif A < 0.0:
            psi = A / math.sqrt(n_prod)
        else:
            psi = 0.0

    theta_L = d_hat - math.sqrt((p1_hat - l1)**2 + (u2 - p2_hat)**2 - 2.0 * psi * (p1_hat - l1) * (u2 - p2_hat))
    theta_U = d_hat + math.sqrt((p2_hat - l2)**2 + (u1 - p1_hat)**2 - 2.0 * psi * (p2_hat - l2) * (u1 - p1_hat))
    return float(theta_L), float(theta_U)

def bootstrap_percentile_ci(
    errors: np.ndarray,
    B: int = 2000,
    seed: int = 2026091901,
    alpha: float = 0.05
) -> Tuple[float, float]:
    """Efron's non-parametric percentile bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    n = len(errors)
    boot_means = np.empty(B, dtype=np.float64)

    for b in range(B):
        sample_indices = rng.randint(0, n, size=n)
        boot_means[b] = np.mean(errors[sample_indices])

    boot_sorted = np.sort(boot_means)
    # Protocol V2.6 Section 7.5: 50th and 1950th sorted replicates for B=2000, alpha=0.05
    # In 0-based indexing: index 49 (50th element) and index 1949 (1950th element)
    idx_l = int(round((alpha / 2.0) * B)) - 1
    idx_u = int(round((1.0 - alpha / 2.0) * B)) - 1
    idx_l = max(0, min(idx_l, B - 1))
    idx_u = max(0, min(idx_u, B - 1))

    lcb = float(boot_sorted[idx_l])
    ucb = float(boot_sorted[idx_u])
    return lcb, ucb
