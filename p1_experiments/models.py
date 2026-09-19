"""Comparison Model Family for P1-A Address Identifiability Experiment (Revision 2).

All models evaluated on identical candidate events and queries:
1. QueryBlindControl: Pre-registered distinct candidate selection (Ch1 picks first active slot,
   Ch2 picks last active slot). Bounded by combinatorial chance level 1/(N*(N-1)).
2. PositiveScalarKernel: score = c * S * K_i (c=2.0, S=1.0). Monotonic amplitude ranking.
3. SignedScalarKernel: score = sign(q - 1.4) * K_i. Binary polar inversion; fails on middle keys.
4. HistoricalSprint4_2_KernelStaticControl: Static control adopting historical Sprint 4.2-B scoring kernel.
   Uses normalized polynomial moment curve phi(k) = (k, k^2)/||(k, k^2)|| from sprint4_2_vector_qk.py:90
   and fixed surprise drives S_E=1.8 (Ch1), S_E=2.6 (Ch2). Includes canonical regression verification.
5. FrozenSRotationHarmonicReference: Static control adopting harmonic S^1 scoring kernel
   phi(k) = (cos(0.4*k), sin(0.4*k)) with fixed surprise drives S=1.8, S=2.6 (no address port).
6. ScalarMetricAttention: score = -(q - K_i)^2. 1D distance metric compatibility.
7. VectorQKAddressAttention: score = cos(omega*(q - K_i)) on S^1 unit circle (omega=0.4).

Each model operates dual channels (Channel 1 for q1, Channel 2 for q2) and
feeds into an explicit algebraic combiner (add, sub).
"""

import math
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from task_generator import TrialData

class BaseModel:
    def __init__(self, name: str):
        self.name = name

    def forward_channel(self, keys: np.ndarray, mask: np.ndarray, query_cue: float, channel_idx: int) -> Tuple[int, np.ndarray]:
        """Returns (winner_idx, attention_probs)."""
        raise NotImplementedError

    def evaluate_trial(self, trial: TrialData, gamma: float = 10.0) -> Dict[str, Any]:
        W = trial.window_size
        keys = trial.keys
        values = trial.values
        mask = trial.mask
        q1, q2 = trial.query_cues
        t_idx1, t_idx2 = trial.target_idx

        # Channel 1
        w1, p1 = self.forward_channel(keys, mask, q1, channel_idx=1)
        # Channel 2
        w2, p2 = self.forward_channel(keys, mask, q2, channel_idx=2)

        # Hard readout values
        v1_hat = float(values[w1]) if w1 is not None else 0.0
        v2_hat = float(values[w2]) if w2 is not None else 0.0

        # Soft readout values
        v1_soft = float(np.sum(p1 * values))
        v2_soft = float(np.sum(p2 * values))

        # Check address routing correctness
        c1_correct = (w1 == t_idx1)
        c2_correct = (w2 == t_idx2)
        both_correct = (c1_correct and c2_correct)

        # Combined algebraic outputs
        y_add_hard = v1_hat + v2_hat
        y_sub_hard = v1_hat - v2_hat

        y_add_soft = v1_soft + v2_soft
        y_sub_soft = v1_soft - v2_soft

        # Ground truth targets
        gt_v1, gt_v2 = trial.target_values
        gt_add = trial.gt_add
        gt_sub = trial.gt_sub

        # Errors
        binding_err_c1 = abs(v1_hat - gt_v1)
        binding_err_c2 = abs(v2_hat - gt_v2)
        total_binding_err = binding_err_c1 + binding_err_c2

        err_add_hard = abs(y_add_hard - gt_add)
        err_sub_hard = abs(y_sub_hard - gt_sub)

        err_add_soft = abs(y_add_soft - gt_add)
        err_sub_soft = abs(y_sub_soft - gt_sub)

        return {
            'model_name': self.name,
            'trial_id': trial.trial_id,
            'seed': trial.seed,
            'N': trial.N,
            't1_is_middle': trial.t1_is_middle,
            't2_is_middle': trial.t2_is_middle,
            'is_middle_target': trial.is_middle_target,
            'c1_winner': w1,
            'c2_winner': w2,
            'c1_correct': c1_correct,
            'c2_correct': c2_correct,
            'both_correct': both_correct,
            'v1_hat': v1_hat,
            'v2_hat': v2_hat,
            'v1_soft': v1_soft,
            'v2_soft': v2_soft,
            'binding_err_total': total_binding_err,
            'err_add_hard': err_add_hard,
            'err_sub_hard': err_sub_hard,
            'err_add_soft': err_add_soft,
            'err_sub_soft': err_sub_soft
        }

def masked_softmax(scores: np.ndarray, mask: np.ndarray, gamma: float = 10.0) -> np.ndarray:
    """Stable max-subtracted softmax over active candidate mask."""
    active_scores = scores[mask]
    if len(active_scores) == 0:
        return np.zeros_like(scores)
    max_s = np.max(active_scores)
    exp_s = np.zeros_like(scores, dtype=np.float64)
    exp_s[mask] = np.exp(gamma * (active_scores - max_s))
    sum_exp = np.sum(exp_s[mask])
    if sum_exp > 0:
        return exp_s / sum_exp
    else:
        probs = np.zeros_like(scores)
        probs[mask] = 1.0 / len(active_scores)
        return probs

# 1. QueryBlindControl
class QueryBlindControl(BaseModel):
    """Negative control baseline: ignores query cue entirely.
    Pre-registered distinct candidate selection:
    Channel 1 selects the first active slot (active_indices[0]).
    Channel 2 selects the last active slot (active_indices[-1]).
    Because N >= 2, active_indices[0] != active_indices[-1].
    Combinatorial chance:
    P(both correct) = 1 / (N * (N - 1)) = 1/6 for N=3; 1/2 for N=2.
    P(c1 correct) = 1/N.
    P(c2 correct) = 1/N.
    """
    def __init__(self):
        super().__init__("QueryBlindControl")

    def forward_channel(self, keys: np.ndarray, mask: np.ndarray, query_cue: float, channel_idx: int) -> Tuple[int, np.ndarray]:
        scores = np.full_like(keys, -1e9, dtype=np.float64)
        scores[mask] = 0.0
        probs = np.zeros_like(keys, dtype=np.float64)
        active_indices = np.where(mask)[0]
        if len(active_indices) == 0:
            return 0, probs
        probs[mask] = 1.0 / len(active_indices)
        if channel_idx == 1:
            winner = active_indices[0]
        else:
            winner = active_indices[-1]
        return int(winner), probs

# 2. PositiveScalarKernel
class PositiveScalarKernel(BaseModel):
    """TAN-I / Sprint 4.2-B M0 positive scalar kernel:
    score = c * S * K_i (c=2.0, S=1.0).
    Since c > 0 and S > 0, always orders candidates monotonically by K_i ascending.
    """
    def __init__(self, c: float = 2.0, gamma: float = 10.0):
        super().__init__("PositiveScalarKernel")
        self.c = c
        self.gamma = gamma

    def forward_channel(self, keys: np.ndarray, mask: np.ndarray, query_cue: float, channel_idx: int) -> Tuple[int, np.ndarray]:
        S = 1.0
        scores = np.full_like(keys, -1e9, dtype=np.float64)
        scores[mask] = self.c * S * keys[mask]
        probs = masked_softmax(scores, mask, gamma=self.gamma)
        active_indices = np.where(mask)[0]
        winner = active_indices[np.argmax(scores[mask])]
        return int(winner), probs

# 3. SignedScalarKernel
class SignedScalarKernel(BaseModel):
    """Sprint 4.2-B M1 signed scalar control:
    score = sign(q - 1.4) * c * S * K_i.
    Flips between min and max keys; strictly incapable of addressing intermediate keys.
    """
    def __init__(self, midpoint: float = 1.4, gamma: float = 10.0):
        super().__init__("SignedScalarKernel")
        self.midpoint = midpoint
        self.gamma = gamma

    def forward_channel(self, keys: np.ndarray, mask: np.ndarray, query_cue: float, channel_idx: int) -> Tuple[int, np.ndarray]:
        sign_q = 1.0 if query_cue >= self.midpoint else -1.0
        scores = np.full_like(keys, -1e9, dtype=np.float64)
        scores[mask] = sign_q * keys[mask]
        probs = masked_softmax(scores, mask, gamma=self.gamma)
        active_indices = np.where(mask)[0]
        winner = active_indices[np.argmax(scores[mask])]
        return int(winner), probs

# 4. HistoricalSprint4_2_KernelStaticControl
class HistoricalSprint4_2_KernelStaticControl(BaseModel):
    """Static control adopting the historical Sprint 4.2-B scoring kernel.
    Uses normalized polynomial moment curve keys: phi(k) = (k, k^2) / ||(k, k^2)||
    from sprint4_2_vector_qk.py:90.
    Has no independent address port; query vector is driven by fixed historical surprise drives:
    u(S) = C * S * (cos(omega*S), sin(omega*S)) with C=2.0, omega=0.4.
    Channel 1 sets S_E = 1.8 (QA=3.0 equivalent); Channel 2 sets S_E = 2.6 (QB=4.0 equivalent).
    """
    def __init__(self, omega: float = 0.4, C: float = 2.0, gamma: float = 10.0):
        super().__init__("HistoricalSprint4_2_KernelStaticControl")
        self.omega = omega
        self.C = C
        self.gamma = gamma

    @staticmethod
    def phi_moment(k: float) -> np.ndarray:
        if k == 0.0:
            return np.array([0.0, 0.0], dtype=np.float64)
        norm = math.sqrt(k**2 + k**4)
        return np.array([k / norm, (k**2) / norm], dtype=np.float64)

    def forward_channel(self, keys: np.ndarray, mask: np.ndarray, query_cue: float, channel_idx: int) -> Tuple[int, np.ndarray]:
        S = 1.8 if channel_idx == 1 else 2.6
        theta = self.omega * S
        u = self.C * S * np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)

        scores = np.full_like(keys, -1e9, dtype=np.float64)
        active_indices = np.where(mask)[0]
        for idx in active_indices:
            k = float(keys[idx])
            p = self.phi_moment(k)
            scores[idx] = float(u @ p)

        probs = masked_softmax(scores, mask, gamma=self.gamma)
        winner = active_indices[np.argmax(scores[mask])]
        return int(winner), probs

    @classmethod
    def verify_canonical_historical_regression(cls) -> Dict[str, Any]:
        """Verifies bit-for-bit identity against sprint4_2_vector_qk.py:58-62 predictions:
        On fixed context [1, 0, 2, 0, Q]:
        For QA=3.0: SE=1.8, EA=3.59230429, EB=3.33356156, diff=+0.258742724, winner='A'
        For QB=4.0: SE=2.6, EA=5.032371, EB=5.18828113, diff=-0.155910133, winner='B'
        """
        omega = 0.4
        C = 2.0
        # QA = 3.0, mu = 1.2, SE = 1.8
        SE_A = 1.8
        u_A = C * SE_A * np.array([math.cos(omega * SE_A), math.sin(omega * SE_A)])
        phi_1 = cls.phi_moment(1.0)
        phi_2 = cls.phi_moment(2.0)
        EA_A = float(u_A @ phi_1)
        EB_A = float(u_A @ phi_2)
        diff_A = EA_A - EB_A

        # QB = 4.0, mu = 1.4, SE = 2.6
        SE_B = 2.6
        u_B = C * SE_B * np.array([math.cos(omega * SE_B), math.sin(omega * SE_B)])
        EA_B = float(u_B @ phi_1)
        EB_B = float(u_B @ phi_2)
        diff_B = EA_B - EB_B

        assert abs(EA_A - 3.59230429) < 1e-6, f"Regression fail EA_A={EA_A}"
        assert abs(EB_A - 3.33356156) < 1e-6, f"Regression fail EB_A={EB_A}"
        assert abs(diff_A - 0.258742724) < 1e-6, f"Regression fail diff_A={diff_A}"

        assert abs(EA_B - 5.032371) < 1e-6, f"Regression fail EA_B={EA_B}"
        assert abs(EB_B - 5.18828113) < 1e-6, f"Regression fail EB_B={EB_B}"
        assert abs(diff_B - (-0.155910133)) < 1e-6, f"Regression fail diff_B={diff_B}"

        return {
            'QA_3.0': {'EA': EA_A, 'EB': EB_A, 'diff': diff_A, 'winner': 'A'},
            'QB_4.0': {'EA': EA_B, 'EB': EB_B, 'diff': diff_B, 'winner': 'B'},
            'regression_passed': True
        }

# 5. FrozenSRotationHarmonicReference
class FrozenSRotationHarmonicReference(BaseModel):
    """Static control adopting harmonic S^1 scoring kernel.
    Keys on S^1 circle: phi(k) = (cos(omega*k), sin(omega*k)).
    Query driven by surprise S without independent address port:
    u(S) = (cos(omega*S), sin(omega*S)) with S=1.8 (Ch1) and S=2.6 (Ch2).
    """
    def __init__(self, omega: float = 0.4, gamma: float = 10.0):
        super().__init__("FrozenSRotationHarmonicReference")
        self.omega = omega
        self.gamma = gamma

    def forward_channel(self, keys: np.ndarray, mask: np.ndarray, query_cue: float, channel_idx: int) -> Tuple[int, np.ndarray]:
        S = 1.8 if channel_idx == 1 else 2.6
        theta_q = self.omega * S
        q_vec = np.array([math.cos(theta_q), math.sin(theta_q)])

        scores = np.full_like(keys, -1e9, dtype=np.float64)
        active_indices = np.where(mask)[0]
        for idx in active_indices:
            k = float(keys[idx])
            k_vec = np.array([math.cos(self.omega * k), math.sin(self.omega * k)])
            scores[idx] = float(q_vec @ k_vec)

        probs = masked_softmax(scores, mask, gamma=self.gamma)
        winner = active_indices[np.argmax(scores[mask])]
        return int(winner), probs

# 6. ScalarMetricAttention
class ScalarMetricAttention(BaseModel):
    """1D Scalar Metric Attention:
    score = - (q - K_i)^2.
    Proves whether distance metric compatibility in 1D achieves full addressability
    without requiring 2D orthogonal rotation.
    """
    def __init__(self, gamma: float = 10.0):
        super().__init__("ScalarMetricAttention")
        self.gamma = gamma

    def forward_channel(self, keys: np.ndarray, mask: np.ndarray, query_cue: float, channel_idx: int) -> Tuple[int, np.ndarray]:
        scores = np.full_like(keys, -1e9, dtype=np.float64)
        scores[mask] = - ((query_cue - keys[mask]) ** 2)
        probs = masked_softmax(scores, mask, gamma=self.gamma)
        active_indices = np.where(mask)[0]
        winner = active_indices[np.argmax(scores[mask])]
        return int(winner), probs

# 7. VectorQKAddressAttention
class VectorQKAddressAttention(BaseModel):
    """2D Vector QK Attention with Dedicated Address Port:
    score = cos(omega * (q - K_i)).
    Operates on S^1 unit circle with dedicated query address cue q (omega=0.4).
    """
    def __init__(self, omega: float = 0.4, gamma: float = 10.0):
        super().__init__("VectorQKAddressAttention")
        self.omega = omega
        self.gamma = gamma

    def forward_channel(self, keys: np.ndarray, mask: np.ndarray, query_cue: float, channel_idx: int) -> Tuple[int, np.ndarray]:
        theta_q = self.omega * query_cue
        q_vec = np.array([math.cos(theta_q), math.sin(theta_q)])

        scores = np.full_like(keys, -1e9, dtype=np.float64)
        active_indices = np.where(mask)[0]
        for idx in active_indices:
            k = float(keys[idx])
            k_vec = np.array([math.cos(self.omega * k), math.sin(self.omega * k)])
            scores[idx] = float(q_vec @ k_vec)

        probs = masked_softmax(scores, mask, gamma=self.gamma)
        winner = active_indices[np.argmax(scores[mask])]
        return int(winner), probs

# 8. LeakyCheatNegativeControl (Used exclusively for G1 fixture negative testing)
class LeakyCheatNegativeControl(BaseModel):
    """Negative fixture model that deliberately cheats by accessing target indices,
    ensuring that the G1 leakage check successfully raises an AssertionError.
    """
    def __init__(self, cheat_probability: float = 0.6):
        super().__init__("LeakyCheatNegativeControl")
        self.cheat_probability = cheat_probability

    def evaluate_trial(self, trial: TrialData, gamma: float = 10.0) -> Dict[str, Any]:
        # Deliberately output correct targets with high probability
        rng = np.random.RandomState(trial.seed + trial.trial_id)
        if rng.uniform(0, 1) < self.cheat_probability:
            w1, w2 = trial.target_idx
        else:
            active = np.where(trial.mask)[0]
            w1 = active[0]
            w2 = active[-1]
        c1_correct = (w1 == trial.target_idx[0])
        c2_correct = (w2 == trial.target_idx[1])
        both_correct = (c1_correct and c2_correct)
        return {
            'model_name': self.name,
            'c1_winner': w1,
            'c2_winner': w2,
            'c1_correct': c1_correct,
            'c2_correct': c2_correct,
            'both_correct': both_correct,
            'v1_hat': float(trial.values[w1]),
            'v2_hat': float(trial.values[w2]),
            'err_add_hard': 0.0,
            'err_sub_hard': 0.0
        }
