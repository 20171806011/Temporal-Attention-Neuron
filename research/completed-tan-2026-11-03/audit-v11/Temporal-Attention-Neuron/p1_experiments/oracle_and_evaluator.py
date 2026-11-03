"""Oracle, G1 Identifiability Gate, and Metric Evaluator for P1-A (Revision 3).

Implements:
1. Ideal Associative Oracle: Theoretical upper bound with nearest-neighbor routing.
2. G1 Identifiability Gate:
   - G1.1: Oracle Soundness on N=2 and N=3 (500 trials each, 100.0% joint accuracy, 0.0 error).
   - G1.2: Information Leakage Check on QueryBlindControl (1,000 trials, point estimate <= chance + 0.05,
           Wilson 95% CI lower bound <= chance + 0.02).
   - G1.2b: Negative fixture test verifying that an intentionally leaky model triggers the EXACT
           production evaluate_leakage_compliance() assertion rejection path.
   - G1.3: Adversarial sensitivity & invariance suite (Query-swap, Value-swap, Zero-trap,
           Equal-val, Position-permutation; 200 trials each, 100% pass strictly asserted).
   - G1.4: Historical 4.2-B scoring kernel regression test against canonical fixed context.
3. Metric Evaluator: Computes disambiguated middle key channel hit rate, joint accuracy given
   middle key, hard routing accuracies, and soft vs. hard algebraic errors.
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional
from task_generator import TaskGenerator, TrialData
from models import BaseModel, masked_softmax, HistoricalSprint4_2_KernelStaticControl, LeakyCheatNegativeControl

class IdealOracle(BaseModel):
    """Theoretical upper bound: selects exact nearest candidate key in L1 distance."""
    def __init__(self):
        super().__init__("IdealOracle")

    def forward_channel(self, keys: np.ndarray, mask: np.ndarray, query_cue: float, channel_idx: int) -> tuple:
        active_indices = np.where(mask)[0]
        if len(active_indices) == 0:
            return 0, np.zeros_like(keys)
        diffs = np.abs(keys[active_indices] - query_cue)
        winner = active_indices[np.argmin(diffs)]
        probs = np.zeros_like(keys, dtype=np.float64)
        probs[winner] = 1.0
        return int(winner), probs

    def evaluate_trial(self, trial: TrialData, gamma: float = 10.0) -> Dict[str, Any]:
        keys = trial.keys
        values = trial.values
        mask = trial.mask
        q1, q2 = trial.query_cues
        t1, t2 = trial.target_idx

        active_indices = np.where(mask)[0]
        w1 = active_indices[np.argmin(np.abs(keys[active_indices] - q1))]
        w2 = active_indices[np.argmin(np.abs(keys[active_indices] - q2))]

        c1_correct = (w1 == t1)
        c2_correct = (w2 == t2)
        both_correct = (c1_correct and c2_correct)

        v1 = float(values[w1])
        v2 = float(values[w2])

        return {
            'model_name': self.name,
            'c1_winner': int(w1),
            'c2_winner': int(w2),
            'c1_correct': c1_correct,
            'c2_correct': c2_correct,
            'both_correct': both_correct,
            'v1_hat': v1,
            'v2_hat': v2,
            'v1_soft': v1,
            'v2_soft': v2,
            'binding_err_total': abs(v1 - trial.target_values[0]) + abs(v2 - trial.target_values[1]),
            'err_add_hard': abs((v1 + v2) - trial.gt_add),
            'err_sub_hard': abs((v1 - v2) - trial.gt_sub),
            'err_add_soft': abs((v1 + v2) - trial.gt_add),
            'err_sub_soft': abs((v1 - v2) - trial.gt_sub),
            't1_is_middle': trial.t1_is_middle,
            't2_is_middle': trial.t2_is_middle,
            'is_middle_target': trial.is_middle_target
        }

def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple:
    """Calculates Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # for 95%
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half_width = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))

def evaluate_leakage_compliance(
    evaluations: List[Dict[str, Any]],
    theoretical_chance: float = 1.0 / 6.0,
    delta_threshold: float = 0.05,
    delta_ci: float = 0.02
) -> Dict[str, Any]:
    """Single shared production compliance validator for information leakage.
    Used for both the production QueryBlind test and negative fixture testing.
    Raises AssertionError if leakage thresholds are violated.
    """
    total = len(evaluations)
    if total == 0:
        raise ValueError("Cannot evaluate leakage compliance on 0 evaluations.")

    both_count = int(np.sum([e['both_correct'] for e in evaluations]))
    c1_count = int(np.sum([e['c1_correct'] for e in evaluations]))
    c2_count = int(np.sum([e['c2_correct'] for e in evaluations]))

    both_acc = both_count / total
    c1_acc = c1_count / total
    c2_acc = c2_count / total

    ci_low, ci_high = wilson_score_interval(both_count, total, confidence=0.95)
    leak_threshold = theoretical_chance + delta_threshold
    ci_bound = theoretical_chance + delta_ci

    if both_acc > leak_threshold:
        raise AssertionError(
            f"LEAKAGE ASSERTION FAILURE: empirical accuracy {both_acc:.4f} exceeds threshold {leak_threshold:.4f} "
            f"(chance={theoretical_chance:.4f}, delta={delta_threshold:.4f})"
        )

    if ci_low > ci_bound:
        raise AssertionError(
            f"LEAKAGE CI ASSERTION FAILURE: 95% CI lower bound {ci_low:.4f} exceeds bound {ci_bound:.4f} "
            f"(chance={theoretical_chance:.4f}, delta_ci={delta_ci:.4f})"
        )

    return {
        'empirical_both_acc': both_acc,
        'both_count': both_count,
        'c1_acc': c1_acc,
        'c2_acc': c2_acc,
        'total_trials': total,
        'ci_95': [ci_low, ci_high],
        'theoretical_chance': theoretical_chance,
        'threshold': leak_threshold,
        'ci_bound': ci_bound,
        'passed': True
    }

def run_g1_identifiability_gate(generator: TaskGenerator, models: List[BaseModel], num_calibration_trials: int = 500) -> Dict[str, Any]:
    """Rigorous G1 Gate Verification:
    Asserts oracle perfection, tests leakage with Wilson CI, runs negative fixture leak tests
    through the identical production compliance path, executes 5-suite adversarial tests,
    and verifies historical 4.2-B regression.
    """
    print("--- Executing P1-A G1 Identifiability Gate Verification ---")
    gate_results = {
        'oracle_calibration': {},
        'adversarial_suites': {}
    }
    oracle = IdealOracle()

    # 1. Oracle Perfection Test on Zero-Noise Trials (N=2 and N=3)
    for N in (2, 3):
        calib_gen = TaskGenerator(window_size=5, noise_sigma=0.0)
        calib_trials = calib_gen.generate_batch(seed=99999 + N, num_trials=num_calibration_trials, N=N)
        oracle_evals = [oracle.evaluate_trial(t) for t in calib_trials]
        oracle_both_acc = float(np.mean([e['both_correct'] for e in oracle_evals]))
        oracle_err_add = float(np.mean([e['err_add_hard'] for e in oracle_evals]))

        assert oracle_both_acc == 1.0, f"Oracle failed on N={N} zero-noise calibration (acc={oracle_both_acc})!"
        assert oracle_err_add < 1e-12, f"Oracle non-zero add error on N={N} zero-noise calibration!"
        print(f"  [PASS] G1.1 Oracle Soundness on N={N}: {num_calibration_trials}/{num_calibration_trials} trials, 100.0% accuracy, 0.0 error")
        gate_results['oracle_calibration'][f'N{N}'] = {
            'trials': num_calibration_trials,
            'both_acc': oracle_both_acc,
            'mean_add_err': oracle_err_add,
            'passed': True
        }

    # 2. Leakage Test on QueryBlindControl
    test_gen = TaskGenerator(window_size=5, noise_sigma=0.0)
    leak_trials = test_gen.generate_batch(seed=88888, num_trials=1000, N=3)
    blind_model = [m for m in models if m.name == "QueryBlindControl"][0]
    blind_evals = [blind_model.evaluate_trial(t) for t in leak_trials]

    theoretical_chance_n3 = 1.0 / (3 * 2)  # 0.166667
    leak_result = evaluate_leakage_compliance(blind_evals, theoretical_chance=theoretical_chance_n3)
    print(f"  [CHECK] G1.2 Leakage Monitor: QueryBlind both-acc = {leak_result['empirical_both_acc']:.4f} (95% CI: [{leak_result['ci_95'][0]:.4f}, {leak_result['ci_95'][1]:.4f}], chance = {theoretical_chance_n3:.4f})")
    print(f"          QueryBlind Channel 1 acc = {leak_result['c1_acc']:.4f} (chance = 0.3333), Channel 2 acc = {leak_result['c2_acc']:.4f} (chance = 0.3333)")
    print(f"  [PASS] G1.2 Information Leakage Check Passed: empirical={leak_result['empirical_both_acc']:.4f} <= threshold={leak_result['threshold']:.4f}, CI_low={leak_result['ci_95'][0]:.4f} <= {leak_result['ci_bound']:.4f}")
    gate_results['leakage_check'] = leak_result

    # 2b. Negative Fixture Test: Verify that an intentionally leaky model is rejected by the EXACT SAME production check
    leaky_model = LeakyCheatNegativeControl(cheat_probability=0.6)
    leaky_evals = [leaky_model.evaluate_trial(t) for t in leak_trials]
    leaky_both_acc = float(np.mean([e['both_correct'] for e in leaky_evals]))

    try:
        evaluate_leakage_compliance(leaky_evals, theoretical_chance=theoretical_chance_n3)
        raise RuntimeError("CRITICAL FLAW: evaluate_leakage_compliance failed to reject an intentionally leaky cheat model!")
    except AssertionError as expected_rejection:
        print(f"  [PASS] G1.2b Negative Fixture Rejection Verified: LeakyCheat model triggered real production assertion rejection:")
        print(f"         \"{str(expected_rejection)}\"")
        gate_results['leakage_negative_fixture'] = {
            'leaky_both_acc': leaky_both_acc,
            'rejected_as_expected': True,
            'exception_caught': 'AssertionError',
            'rejection_message': str(expected_rejection)
        }

    # 3. Adversarial Invariance & Sensitivity Suite on Address-Aware Models
    address_models = [m for m in models if m.name in ("ScalarMetricAttention", "VectorQKAddressAttention")]
    num_adv_trials = 200
    swap_gen = TaskGenerator(window_size=5, noise_sigma=0.0)
    swap_trials = swap_gen.generate_batch(seed=77777, num_trials=num_adv_trials, N=3)
    adv_rng = np.random.RandomState(66666)

    for m in address_models:
        q_swap_ok = 0
        v_swap_ok = 0
        zero_val_ok = 0
        equal_val_ok = 0
        pos_perm_ok = 0

        for t in swap_trials:
            base_eval = m.evaluate_trial(t)
            assert base_eval['both_correct'], f"Base evaluation unexpectedly failed for {m.name} under zero noise!"

            # (a) Query swap: q1 <-> q2
            t_qswap = TaskGenerator.create_query_swap_variant(t)
            qswap_eval = m.evaluate_trial(t_qswap)
            if qswap_eval['c1_winner'] == base_eval['c2_winner'] and qswap_eval['c2_winner'] == base_eval['c1_winner']:
                sub_base = base_eval['v1_hat'] - base_eval['v2_hat']
                sub_qswap = qswap_eval['v1_hat'] - qswap_eval['v2_hat']
                if abs(sub_qswap - (-sub_base)) < 1e-6:
                    q_swap_ok += 1

            # (b) Value swap: V[idx1] <-> V[idx2]
            t_vswap = TaskGenerator.create_value_swap_variant(t)
            vswap_eval = m.evaluate_trial(t_vswap)
            if vswap_eval['both_correct']:
                if abs(vswap_eval['v1_hat'] - base_eval['v2_hat']) < 1e-6 and abs(vswap_eval['v2_hat'] - base_eval['v1_hat']) < 1e-6:
                    v_swap_ok += 1

            # (c) Zero value trap: V[idx1] = 0.0
            t_zero = TaskGenerator.create_zero_value_variant(t)
            zero_eval = m.evaluate_trial(t_zero)
            if zero_eval['both_correct']:
                if abs(zero_eval['v1_hat'] - 0.0) < 1e-6 and abs(zero_eval['v2_hat'] - base_eval['v2_hat']) < 1e-6:
                    if abs(zero_eval['err_add_hard']) < 1e-6 and abs(zero_eval['err_sub_hard']) < 1e-6:
                        zero_val_ok += 1

            # (d) Equal value test: V[idx1] = V[idx2] = 1.75
            t_eq = TaskGenerator.create_equal_value_variant(t, val=1.75)
            eq_eval = m.evaluate_trial(t_eq)
            if eq_eval['both_correct']:
                if abs(eq_eval['v1_hat'] - 1.75) < 1e-6 and abs(eq_eval['v2_hat'] - 1.75) < 1e-6:
                    sub_out = eq_eval['v1_hat'] - eq_eval['v2_hat']
                    add_out = eq_eval['v1_hat'] + eq_eval['v2_hat']
                    if abs(sub_out) < 1e-6 and abs(add_out - 3.5) < 1e-6:
                        equal_val_ok += 1

            # (e) Position permutation invariance
            t_pos = TaskGenerator.create_position_permutation_variant(t, adv_rng)
            pos_eval = m.evaluate_trial(t_pos)
            if pos_eval['both_correct']:
                if abs(pos_eval['v1_hat'] - base_eval['v1_hat']) < 1e-6 and abs(pos_eval['v2_hat'] - base_eval['v2_hat']) < 1e-6:
                    if abs(pos_eval['err_add_hard']) < 1e-6 and abs(pos_eval['err_sub_hard']) < 1e-6:
                        pos_perm_ok += 1

        print(f"  [VERIFY] {m.name} Adversarial Suite ({num_adv_trials} trials):")
        print(f"    - Query-swap: {q_swap_ok}/{num_adv_trials}")
        print(f"    - Value-swap: {v_swap_ok}/{num_adv_trials}")
        print(f"    - Zero-trap:  {zero_val_ok}/{num_adv_trials}")
        print(f"    - Equal-val:  {equal_val_ok}/{num_adv_trials}")
        print(f"    - Pos-perm:   {pos_perm_ok}/{num_adv_trials}")

        assert q_swap_ok == num_adv_trials, f"Query-swap check failed for {m.name}: {q_swap_ok}/{num_adv_trials}!"
        assert v_swap_ok == num_adv_trials, f"Value-swap check failed for {m.name}: {v_swap_ok}/{num_adv_trials}!"
        assert zero_val_ok == num_adv_trials, f"Zero-trap check failed for {m.name}: {zero_val_ok}/{num_adv_trials}!"
        assert equal_val_ok == num_adv_trials, f"Equal-val check failed for {m.name}: {equal_val_ok}/{num_adv_trials}!"
        assert pos_perm_ok == num_adv_trials, f"Pos-perm check failed for {m.name}: {pos_perm_ok}/{num_adv_trials}!"

        print(f"  [PASS] G1.3 Adversarial & Invariance Tests for {m.name}: All 5 suites 100% verified ({num_adv_trials}/{num_adv_trials})")
        gate_results['adversarial_suites'][m.name] = {
            'q_swap_ok': q_swap_ok,
            'v_swap_ok': v_swap_ok,
            'zero_val_ok': zero_val_ok,
            'equal_val_ok': equal_val_ok,
            'pos_perm_ok': pos_perm_ok,
            'total_adv_trials': num_adv_trials,
            'passed': True
        }

    # 4. Historical 4.2-B Scoring Kernel Canonical Regression Check
    hist_reg = HistoricalSprint4_2_KernelStaticControl.verify_canonical_historical_regression()
    assert hist_reg['regression_passed'] is True, "Historical 4.2-B canonical regression failed!"
    print("  [PASS] G1.4 Historical Sprint 4.2-B Scoring Kernel Regression Verified (+0.258742 on QA, -0.155910 on QB)")
    gate_results['historical_kernel_regression'] = hist_reg

    gate_results['g1_all_passed'] = True
    print(">>> ALL G1 IDENTIFIABILITY GATE CHECKS STRICTLY ASSERTED AND PASSED! <<<")
    return gate_results

def compute_condition_statistics(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes comprehensive statistics for a given model and condition."""
    n_trials = len(evaluations)
    if n_trials == 0:
        return {}

    c1_acc = float(np.mean([e['c1_correct'] for e in evaluations]))
    c2_acc = float(np.mean([e['c2_correct'] for e in evaluations]))
    both_acc = float(np.mean([e['both_correct'] for e in evaluations]))
    both_count = int(np.sum([e['both_correct'] for e in evaluations]))

    mean_binding_err = float(np.mean([e['binding_err_total'] for e in evaluations]))
    mean_err_add_hard = float(np.mean([e['err_add_hard'] for e in evaluations]))
    mean_err_sub_hard = float(np.mean([e['err_sub_hard'] for e in evaluations]))

    mean_err_add_soft = float(np.mean([e['err_add_soft'] for e in evaluations]))
    mean_err_sub_soft = float(np.mean([e['err_sub_soft'] for e in evaluations]))

    correct_evals = [e for e in evaluations if e['both_correct']]
    if len(correct_evals) > 0:
        cond_err_add_hard = float(np.mean([e['err_add_hard'] for e in correct_evals]))
        cond_err_sub_hard = float(np.mean([e['err_sub_hard'] for e in correct_evals]))
        cond_err_add_soft = float(np.mean([e['err_add_soft'] for e in correct_evals]))
        cond_err_sub_soft = float(np.mean([e['err_sub_soft'] for e in correct_evals]))
    else:
        cond_err_add_hard = None
        cond_err_sub_hard = None
        cond_err_add_soft = None
        cond_err_sub_soft = None

    # Middle key metrics (when N=3)
    middle_queries_c1 = [e['c1_correct'] for e in evaluations if e.get('t1_is_middle', False)]
    middle_queries_c2 = [e['c2_correct'] for e in evaluations if e.get('t2_is_middle', False)]
    total_middle_queries = len(middle_queries_c1) + len(middle_queries_c2)
    correct_middle_queries = sum(middle_queries_c1) + sum(middle_queries_c2)

    if total_middle_queries > 0:
        middle_key_channel_acc = float(correct_middle_queries / total_middle_queries)
    else:
        middle_key_channel_acc = None

    middle_trials = [e for e in evaluations if e.get('is_middle_target', False)]
    if len(middle_trials) > 0:
        joint_acc_given_middle = float(np.mean([e['both_correct'] for e in middle_trials]))
        middle_trials_count = len(middle_trials)
    else:
        joint_acc_given_middle = None
        middle_trials_count = 0

    return {
        'total_trials': n_trials,
        'c1_acc': c1_acc,
        'c2_acc': c2_acc,
        'both_routed_acc': both_acc,
        'both_routed_count': both_count,
        'both_routed_coverage_pct': both_acc * 100.0,
        'middle_key_channel_acc': middle_key_channel_acc,
        'middle_channel_queries_count': total_middle_queries,
        'middle_channel_correct_count': int(correct_middle_queries),
        'joint_acc_given_middle': joint_acc_given_middle,
        'middle_target_trials_count': middle_trials_count,
        'mean_binding_err': mean_binding_err,
        'unconditional_err_add_hard': mean_err_add_hard,
        'unconditional_err_sub_hard': mean_err_sub_hard,
        'unconditional_err_add_soft': mean_err_add_soft,
        'unconditional_err_sub_soft': mean_err_sub_soft,
        'conditional_err_add_hard': cond_err_add_hard,
        'conditional_err_sub_hard': cond_err_sub_hard,
        'conditional_err_add_soft': cond_err_add_soft,
        'conditional_err_sub_soft': cond_err_sub_soft
    }
