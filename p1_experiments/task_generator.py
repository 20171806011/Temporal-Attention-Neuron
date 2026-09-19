"""Task generator and data contract for P1-A Address Identifiability Experiment.

Design specifications derived from RESEARCH_AUDIT_AND_PLAN_2026-09-17 Section 9
and Codex handoff agreement:
1. Window size W=5. Candidate event count N in {2, 3}.
2. Candidate keys K_i in [0.3, 2.5], min spacing 0.05.
3. Candidate values V_i in [-3.0, 3.0], independent of keys and positions.
4. Target keys are explicitly queried via content cues (q1, q2).
5. Target values and hidden generator IDs are NEVER exposed to models.
6. Support for zero-noise and observation noise sigma in {0.0, 0.01, 0.05}.
7. Rejection sampling accounting and G1 adversarial validation suite.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

@dataclass
class TrialData:
    trial_id: int
    seed: int
    window_size: int
    N: int
    event_positions: List[int]
    keys: np.ndarray          # shape (W,)
    values: np.ndarray        # shape (W,)
    mask: np.ndarray          # shape (W,), boolean, True if candidate slot
    target_idx: Tuple[int, int]
    target_keys: Tuple[float, float]
    target_values: Tuple[float, float]
    query_cues: Tuple[float, float]
    gt_add: float
    gt_sub: float
    t1_is_middle: bool
    t2_is_middle: bool
    is_middle_target: bool    # True if either target is middle key (N=3)
    rejection_count: int

class TaskGenerator:
    def __init__(
        self,
        window_size: int = 5,
        key_range: Tuple[float, float] = (0.3, 2.5),
        min_key_spacing: float = 0.05,
        val_range: Tuple[float, float] = (-3.0, 3.0),
        noise_sigma: float = 0.0
    ):
        self.window_size = window_size
        self.key_range = key_range
        self.min_key_spacing = min_key_spacing
        self.val_range = val_range
        self.noise_sigma = noise_sigma

    def generate_single_trial(self, rng: np.random.RandomState, trial_id: int, seed: int, N: int = 3) -> TrialData:
        assert N in (2, 3), f"N must be 2 or 3, got {N}"
        assert N <= self.window_size, "N cannot exceed window size"

        # 1. Sample candidate keys with spacing constraint
        rejections = 0
        while True:
            raw_keys = rng.uniform(self.key_range[0], self.key_range[1], size=N)
            sorted_keys = np.sort(raw_keys)
            diffs = np.diff(sorted_keys)
            if np.all(diffs >= self.min_key_spacing):
                break
            rejections += 1
            if rejections > 20000:
                raise RuntimeError(f"Could not satisfy min_key_spacing={self.min_key_spacing} after {rejections} attempts")

        # Randomize key assignment order among candidates
        cand_keys = rng.permutation(sorted_keys)

        # 2. Sample independent candidate values
        cand_values = rng.uniform(self.val_range[0], self.val_range[1], size=N)

        # 3. Choose distinct event positions in window
        positions = sorted(rng.choice(self.window_size, size=N, replace=False).tolist())

        # Construct full window vectors
        keys = np.zeros(self.window_size, dtype=np.float64)
        values = np.zeros(self.window_size, dtype=np.float64)
        mask = np.zeros(self.window_size, dtype=bool)

        for i, pos in enumerate(positions):
            keys[pos] = cand_keys[i]
            values[pos] = cand_values[i]
            mask[pos] = True

        # 4. Select two distinct targets among the active positions
        target_indices = rng.choice(positions, size=2, replace=False)
        idx1, idx2 = int(target_indices[0]), int(target_indices[1])

        t_key1, t_key2 = float(keys[idx1]), float(keys[idx2])
        t_val1, t_val2 = float(values[idx1]), float(values[idx2])

        # 5. Determine whether targets are the intermediate (middle) key when N=3
        t1_is_mid = False
        t2_is_mid = False
        if N == 3:
            active_keys = [keys[p] for p in positions]
            sorted_active = sorted(active_keys)
            mid_key = sorted_active[1]
            if abs(t_key1 - mid_key) < 1e-9:
                t1_is_mid = True
            if abs(t_key2 - mid_key) < 1e-9:
                t2_is_mid = True
        is_middle = (t1_is_mid or t2_is_mid)

        # 6. Apply observation noise to query content cues: q_obs = q + N(0, (sigma * 2.2)^2)
        if self.noise_sigma > 0.0:
            std = self.noise_sigma * 2.2
            q1 = float(max(0.01, t_key1 + rng.normal(0.0, std)))
            q2 = float(max(0.01, t_key2 + rng.normal(0.0, std)))
        else:
            q1, q2 = t_key1, t_key2

        return TrialData(
            trial_id=trial_id,
            seed=seed,
            window_size=self.window_size,
            N=N,
            event_positions=positions,
            keys=keys,
            values=values,
            mask=mask,
            target_idx=(idx1, idx2),
            target_keys=(t_key1, t_key2),
            target_values=(t_val1, t_val2),
            query_cues=(q1, q2),
            gt_add=t_val1 + t_val2,
            gt_sub=t_val1 - t_val2,
            t1_is_middle=t1_is_mid,
            t2_is_middle=t2_is_mid,
            is_middle_target=is_middle,
            rejection_count=rejections
        )

    def generate_batch(self, seed: int, num_trials: int, N: int = 3) -> List[TrialData]:
        rng = np.random.RandomState(seed)
        return [self.generate_single_trial(rng, trial_id=i, seed=seed, N=N) for i in range(num_trials)]

    @staticmethod
    def create_query_swap_variant(trial: TrialData) -> TrialData:
        """Adversarial variant: swap query cues q1 <-> q2."""
        return TrialData(
            trial_id=trial.trial_id,
            seed=trial.seed,
            window_size=trial.window_size,
            N=trial.N,
            event_positions=list(trial.event_positions),
            keys=trial.keys.copy(),
            values=trial.values.copy(),
            mask=trial.mask.copy(),
            target_idx=(trial.target_idx[1], trial.target_idx[0]),
            target_keys=(trial.target_keys[1], trial.target_keys[0]),
            target_values=(trial.target_values[1], trial.target_values[0]),
            query_cues=(trial.query_cues[1], trial.query_cues[0]),
            gt_add=trial.gt_add,
            gt_sub=trial.target_values[1] - trial.target_values[0],
            t1_is_middle=trial.t2_is_middle,
            t2_is_middle=trial.t1_is_middle,
            is_middle_target=trial.is_middle_target,
            rejection_count=0
        )

    @staticmethod
    def create_value_swap_variant(trial: TrialData) -> TrialData:
        """Adversarial variant: swap candidate values V[idx1] <-> V[idx2] keeping keys and cues fixed."""
        new_values = trial.values.copy()
        idx1, idx2 = trial.target_idx
        new_values[idx1], new_values[idx2] = trial.values[idx2], trial.values[idx1]

        new_t_val1 = float(new_values[idx1])
        new_t_val2 = float(new_values[idx2])

        return TrialData(
            trial_id=trial.trial_id,
            seed=trial.seed,
            window_size=trial.window_size,
            N=trial.N,
            event_positions=list(trial.event_positions),
            keys=trial.keys.copy(),
            values=new_values,
            mask=trial.mask.copy(),
            target_idx=trial.target_idx,
            target_keys=trial.target_keys,
            target_values=(new_t_val1, new_t_val2),
            query_cues=trial.query_cues,
            gt_add=new_t_val1 + new_t_val2,
            gt_sub=new_t_val1 - new_t_val2,
            t1_is_middle=trial.t1_is_middle,
            t2_is_middle=trial.t2_is_middle,
            is_middle_target=trial.is_middle_target,
            rejection_count=0
        )

    @staticmethod
    def create_zero_value_variant(trial: TrialData) -> TrialData:
        """Adversarial variant: set target 1 value to exactly 0.0 to test zero-trap / cancellation."""
        new_values = trial.values.copy()
        idx1, idx2 = trial.target_idx
        new_values[idx1] = 0.0

        new_t_val1 = 0.0
        new_t_val2 = float(new_values[idx2])

        return TrialData(
            trial_id=trial.trial_id,
            seed=trial.seed,
            window_size=trial.window_size,
            N=trial.N,
            event_positions=list(trial.event_positions),
            keys=trial.keys.copy(),
            values=new_values,
            mask=trial.mask.copy(),
            target_idx=trial.target_idx,
            target_keys=trial.target_keys,
            target_values=(new_t_val1, new_t_val2),
            query_cues=trial.query_cues,
            gt_add=new_t_val1 + new_t_val2,
            gt_sub=new_t_val1 - new_t_val2,
            t1_is_middle=trial.t1_is_middle,
            t2_is_middle=trial.t2_is_middle,
            is_middle_target=trial.is_middle_target,
            rejection_count=0
        )

    @staticmethod
    def create_equal_value_variant(trial: TrialData, val: float = 1.75) -> TrialData:
        """Adversarial variant: set both target values to identical float val."""
        new_values = trial.values.copy()
        idx1, idx2 = trial.target_idx
        new_values[idx1] = val
        new_values[idx2] = val

        return TrialData(
            trial_id=trial.trial_id,
            seed=trial.seed,
            window_size=trial.window_size,
            N=trial.N,
            event_positions=list(trial.event_positions),
            keys=trial.keys.copy(),
            values=new_values,
            mask=trial.mask.copy(),
            target_idx=trial.target_idx,
            target_keys=trial.target_keys,
            target_values=(val, val),
            query_cues=trial.query_cues,
            gt_add=val + val,
            gt_sub=0.0,
            t1_is_middle=trial.t1_is_middle,
            t2_is_middle=trial.t2_is_middle,
            is_middle_target=trial.is_middle_target,
            rejection_count=0
        )

    @staticmethod
    def create_position_permutation_variant(trial: TrialData, rng: np.random.RandomState) -> TrialData:
        """Adversarial variant: permute the slot positions of the candidates in the window
        while strictly preserving the key-value pairs of each candidate.
        Tests position-invariance of content-addressable memory.
        """
        N = trial.N
        old_pos = trial.event_positions
        cand_keys = [trial.keys[p] for p in old_pos]
        cand_vals = [trial.values[p] for p in old_pos]

        # Pick new distinct positions
        new_pos = sorted(rng.choice(trial.window_size, size=N, replace=False).tolist())
        perm = rng.permutation(N)

        new_keys = np.zeros(trial.window_size, dtype=np.float64)
        new_values = np.zeros(trial.window_size, dtype=np.float64)
        new_mask = np.zeros(trial.window_size, dtype=bool)

        # Mapping from old position to new position
        pos_map = {}
        for i, new_p in enumerate(new_pos):
            src_idx = perm[i]
            new_keys[new_p] = cand_keys[src_idx]
            new_values[new_p] = cand_vals[src_idx]
            new_mask[new_p] = True
            pos_map[old_pos[src_idx]] = new_p

        new_idx1 = pos_map[trial.target_idx[0]]
        new_idx2 = pos_map[trial.target_idx[1]]

        return TrialData(
            trial_id=trial.trial_id,
            seed=trial.seed,
            window_size=trial.window_size,
            N=trial.N,
            event_positions=new_pos,
            keys=new_keys,
            values=new_values,
            mask=new_mask,
            target_idx=(new_idx1, new_idx2),
            target_keys=trial.target_keys,
            target_values=trial.target_values,
            query_cues=trial.query_cues,
            gt_add=trial.gt_add,
            gt_sub=trial.gt_sub,
            t1_is_middle=trial.t1_is_middle,
            t2_is_middle=trial.t2_is_middle,
            is_middle_target=trial.is_middle_target,
            rejection_count=0
        )
