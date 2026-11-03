"""Production Evaluation Runner for P1-B Confirmatory Carrier Audit.
Governing Specification: P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md (SHA-256: 07D58A2DE773EA13E8FD5C16901CB8E8A370219D4793F4A7F9BACA23C747A82B)
Gate Status: G1-Design APPROVED by Codex on September 19, 2026.

Executes the 6 separate condition cells across 8 models:
    N in {2, 3}, tau_delay in {0.0, 5.0, 10.0} ms.
4,000 unique histories (2,000 per N), evaluated at all 3 delays and 2 role probes:
    12,000 role queries per N per model -> 24,000 role queries per model.
    Across 8 models: exactly 96,000 paired model-episodes and 192,000 role queries.
"""

import sys
import os
import csv
import json
import time
import hashlib
import psutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Add parent directory to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from task_generator import TaskGenerator, TrialData
from models_p1b_v2 import (
    H, TAU_M, TAU_A, TAU_REF, W_SLOTS, T_ENC, DELTA_V_TOL,
    CarrierState, simulate_encoding, simulate_delay, simulate_cue_and_readout,
    calibrate_m1_decoder_and_gate, wilson_score_ci, newcombe_method10_paired_ci,
    bootstrap_percentile_ci
)

ALL_MODELS = ['M0', 'M1', 'M2a', 'M2b', 'M2c', 'M2d', 'M4_Count', 'M4_Latency']
DELAYS = [0.0, 5.0, 10.0]

def generate_confirmatory_datasets() -> Tuple[List[TrialData], List[TrialData]]:
    """Generate 2,000 unique histories for N=2 and 2,000 for N=3 (Section 3.2)."""
    gen = TaskGenerator(window_size=W_SLOTS)
    
    # N=2: seed 2026091920
    rng_2 = np.random.RandomState(2026091920)
    trials_n2 = [
        gen.generate_single_trial(rng_2, trial_id=tid, seed=2026091920 + tid, N=2)
        for tid in range(2000)
    ]
    
    # N=3: seed 2026091930
    rng_3 = np.random.RandomState(2026091930)
    trials_n3 = [
        gen.generate_single_trial(rng_3, trial_id=tid, seed=2026091930 + tid, N=3)
        for tid in range(2000)
    ]
    
    return trials_n2, trials_n3

def generate_calibration_trials() -> List[TrialData]:
    """Generate 500 calibration trials: 250 N=2 and 250 N=3 (seed 2026091999)."""
    gen = TaskGenerator(window_size=W_SLOTS)
    rng = np.random.RandomState(2026091999)
    trials = []
    for tid in range(250):
        trials.append(gen.generate_single_trial(rng, trial_id=tid, seed=2026091999 + tid, N=2))
    for tid in range(250):
        trials.append(gen.generate_single_trial(rng, trial_id=250 + tid, seed=2026091999 + 250 + tid, N=3))
    return trials

def evaluate_episode(
    trial: TrialData,
    delay: float,
    model_name: str,
    g_dec: float,
    b_dec: float,
    A_0: float
) -> Dict[str, Any]:
    """Evaluate a single model on an episode across both Role A and Role B.
    
    Returns:
        Dictionary containing predictions, delivery successes, joint delivery,
        unconditional composition errors, and abstention status.
    """
    target_A = float(trial.target_values[0])
    target_B = float(trial.target_values[1])
    q_A = float(trial.query_cues[0])
    q_B = float(trial.query_cues[1])

    if model_name == 'M0':
        # Reference Static Model: Quadratic metric S(q, k) = -(q - k)^2 (models.py:296)
        def predict_m0(q_val: float) -> float:
            best_score = -float('inf')
            best_val = 0.0
            for pos in trial.event_positions:
                cand_k = float(trial.keys[pos])
                cand_v = float(trial.values[pos])
                score = -((q_val - cand_k) ** 2)
                if score > best_score:
                    best_score = score
                    best_val = cand_v
            return best_val

        v_hat_A = predict_m0(q_A)
        v_hat_B = predict_m0(q_B)
        abstained_A = False
        abstained_B = False

    else:
        # Dynamic Carrier Models
        base_model = 'M1' if model_name in ('M4_Count', 'M4_Latency') else model_name
        
        # 1. Simulate encoding
        state_enc, _, _ = simulate_encoding(
            keys=trial.keys,
            values=trial.values,
            mask=trial.mask,
            model_type=base_model,
            A_0=A_0
        )
        
        # 2. Simulate delay
        state_cue = simulate_delay(state_enc, tau_delay=delay, model_type=base_model, A_0=A_0)
        
        # 3. Independent repeated probing for Role A and Role B
        res_A = simulate_cue_and_readout(state_cue, query_q=q_A, model_type=base_model, g_dec=g_dec, b_dec=b_dec, A_0=A_0)
        res_B = simulate_cue_and_readout(state_cue, query_q=q_B, model_type=base_model, g_dec=g_dec, b_dec=b_dec, A_0=A_0)
        
        if model_name in ('M1', 'M2a', 'M2b', 'M2c', 'M2d'):
            v_hat_A = res_A['v_hat_analog']
            v_hat_B = res_B['v_hat_analog']
            abstained_A = False
            abstained_B = False
        elif model_name == 'M4_Count':
            v_hat_A = res_A['v_hat_count']
            v_hat_B = res_B['v_hat_count']
            abstained_A = res_A['abstained_count']
            abstained_B = res_B['abstained_count']
        elif model_name == 'M4_Latency':
            v_hat_A = res_A['v_hat_latency']
            v_hat_B = res_B['v_hat_latency']
            abstained_A = res_A['abstained_latency']
            abstained_B = res_B['abstained_latency']
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # Delivery success accounting (Section 7.1)
    succ_A = (abs(v_hat_A - target_A) < DELTA_V_TOL) and (not abstained_A)
    succ_B = (abs(v_hat_B - target_B) < DELTA_V_TOL) and (not abstained_B)
    joint_succ = succ_A and succ_B

    # Unconditional Composition Error via external linear node (Pathway A transport)
    y_hat_add = v_hat_A + v_hat_B
    y_hat_sub = v_hat_A - v_hat_B
    err_add = abs(y_hat_add - float(trial.gt_add))
    err_sub = abs(y_hat_sub - float(trial.gt_sub))
    e_comp = (err_add + err_sub) / 2.0

    return {
        'v_hat_A': v_hat_A,
        'v_hat_B': v_hat_B,
        'succ_A': succ_A,
        'succ_B': succ_B,
        'joint_succ': joint_succ,
        'abstained_A': abstained_A,
        'abstained_B': abstained_B,
        'err_A': abs(v_hat_A - target_A),
        'err_B': abs(v_hat_B - target_B),
        'e_comp': e_comp,
        'y_hat_add': y_hat_add,
        'y_hat_sub': y_hat_sub
    }

def evaluate_confirmatory_cell(
    trials: List[TrialData],
    N_val: int,
    delay: float,
    g_dec: float,
    b_dec: float,
    A_0: float,
    models: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Evaluate one of the 6 separate condition cells c = (N, tau_delay).
    
    Evaluates:
        1. Model-level metrics for all models (DeliveryAcc, JointDeliveryAcc, E_comp, Abstention).
        2. Paired 2x2 table for M1 vs M2a.
        3. Baseline prerequisite enforcement (0.80 for N=2, 0.40 for N=3).
        4. TOST Newcombe Method 10 equivalence interval.
        5. Spike channel bootstrap intervals (B=2000, seed 2026091901).
    """
    if models is None:
        models = ALL_MODELS

    n_trials = len(trials)
    results_by_model: Dict[str, List[Dict[str, Any]]] = {m: [] for m in models}

    for trial in trials:
        for m in models:
            res = evaluate_episode(trial, delay=delay, model_name=m, g_dec=g_dec, b_dec=b_dec, A_0=A_0)
            results_by_model[m].append(res)

    # Compute model summaries
    model_summaries = {}
    for m in models:
        m_res = results_by_model[m]
        succ_A_cnt = sum(1 for r in m_res if r['succ_A'])
        succ_B_cnt = sum(1 for r in m_res if r['succ_B'])
        joint_cnt = sum(1 for r in m_res if r['joint_succ'])
        abst_cnt = sum(1 for r in m_res if (r['abstained_A'] or r['abstained_B']))
        e_comps = np.array([r['e_comp'] for r in m_res], dtype=np.float64)

        model_summaries[m] = {
            'delivery_acc_A': succ_A_cnt / n_trials,
            'delivery_acc_B': succ_B_cnt / n_trials,
            'delivery_acc_overall': (succ_A_cnt + succ_B_cnt) / (2 * n_trials),
            'joint_delivery_acc': joint_cnt / n_trials,
            'abstention_rate': abst_cnt / n_trials,
            'mean_e_comp': float(np.mean(e_comps)),
            'std_e_comp': float(np.std(e_comps))
        }

    # Paired 2x2 table for M1 vs M2a (Section 7.4)
    m1_res = results_by_model['M1']
    m2a_res = results_by_model['M2a']

    n11 = sum(1 for i in range(n_trials) if m1_res[i]['joint_succ'] and m2a_res[i]['joint_succ'])
    n10 = sum(1 for i in range(n_trials) if m1_res[i]['joint_succ'] and not m2a_res[i]['joint_succ'])
    n01 = sum(1 for i in range(n_trials) if not m1_res[i]['joint_succ'] and m2a_res[i]['joint_succ'])
    n00 = sum(1 for i in range(n_trials) if not m1_res[i]['joint_succ'] and not m2a_res[i]['joint_succ'])

    # Baseline prerequisite check (Section 7.3)
    target_floor = 0.80 if N_val == 2 else 0.40
    m1_joint_acc = model_summaries['M1']['joint_delivery_acc']
    m1_passed_floor = bool(m1_joint_acc >= target_floor)

    # Newcombe Method 10 CI
    theta_L, theta_U = newcombe_method10_paired_ci(n11, n10, n01, n00, alpha=0.10)
    in_equivalence_margin = bool(-0.005 < theta_L and theta_U < 0.005)

    # Condition-Specific Decision Logic
    if not m1_passed_floor:
        decision = "FLOOR_LIMITED_COMPARISON"
        status_flag = "M1_CARRIER_INSUFFICIENT"
        redundancy_inferred = False
        decision_text = (
            f"M1 failed baseline floor ({m1_joint_acc:.4f} < {target_floor:.2f}). "
            f"Registered as FLOOR_LIMITED_COMPARISON; memory redundancy NOT inferred."
        )
    else:
        status_flag = "M1_BASELINE_MET"
        if in_equivalence_margin:
            decision = "EQUIVALENCE_ESTABLISHED"
            redundancy_inferred = True
            decision_text = (
                f"Equivalence established: 90% CI [{theta_L:.6f}, {theta_U:.6f}] subset (-0.005, 0.005). "
                f"Joint membrane-potential and refractory reset had no practically meaningful effect."
            )
        else:
            decision = "EQUIVALENCE_REJECTED"
            redundancy_inferred = False
            decision_text = (
                f"Equivalence rejected: 90% CI [{theta_L:.6f}, {theta_U:.6f}] outside (-0.005, 0.005)."
            )

    # Spike Channel Bootstrap Evaluations (Section 7.5)
    spike_channel_results = {}
    for spk_model in ('M4_Count', 'M4_Latency'):
        spk_errors = np.array([r['e_comp'] for r in results_by_model[spk_model]], dtype=np.float64)
        lcb, ucb = bootstrap_percentile_ci(spk_errors, B=2000, seed=2026091901)
        if ucb < 0.05:
            spk_decision = "DEMONSTRATED_CHANNEL_SUCCESS"
        elif lcb >= 0.05:
            spk_decision = "DEMONSTRATED_CHANNEL_LIMITATION"
        else:
            spk_decision = "INCONCLUSIVE"

        spike_channel_results[spk_model] = {
            'mean_e_comp': float(np.mean(spk_errors)),
            'lcb_95': lcb,
            'ucb_95': ucb,
            'decision': spk_decision
        }

    return {
        'N': N_val,
        'delay': delay,
        'n_trials': n_trials,
        'model_summaries': model_summaries,
        'table_2x2': {'n11': n11, 'n10': n10, 'n01': n01, 'n00': n00},
        'm1_joint_acc': m1_joint_acc,
        'target_floor': target_floor,
        'm1_passed_floor': m1_passed_floor,
        'theta_L': theta_L,
        'theta_U': theta_U,
        'in_equivalence_margin': in_equivalence_margin,
        'decision': decision,
        'status_flag': status_flag,
        'redundancy_inferred': redundancy_inferred,
        'decision_text': decision_text,
        'spike_channel_results': spike_channel_results,
        'raw_results': results_by_model
    }

def verify_manifest(manifest_path: Path) -> bool:
    """Read manifest file, rehash each referenced file on disk, and verify bit-for-bit SHA-256 integrity."""
    if not manifest_path.exists():
        return False
    manifest_dir = manifest_path.parent
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            return False
        for line in lines:
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                return False
            expected_hash, rel_filename = parts
            target_path = manifest_dir / rel_filename
            if not target_path.exists():
                return False
            with open(target_path, "rb") as f_in:
                actual_hash = hashlib.sha256(f_in.read()).hexdigest().upper()
            if actual_hash != expected_hash.upper():
                return False
        return True
    except Exception:
        return False

def run_full_confirmatory_workflow(
    trials_n2: List[TrialData],
    trials_n3: List[TrialData],
    g_dec: float,
    b_dec: float,
    A_0: float,
    output_dir: Path,
    is_smoke: bool = False
) -> Dict[str, Any]:
    """Orchestrate all 6 confirmatory cells and history-preserving delay-pooled bootstrap (Section 7.3-7.5)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process(os.getpid())
    peak_ram_mb = process.memory_info().rss / (1024 * 1024)

    # 1. Assert history uniqueness
    assert len(set(t.trial_id for t in trials_n2)) == len(trials_n2), "Duplicate trial IDs in N=2 dataset"
    assert len(set(t.trial_id for t in trials_n3)) == len(trials_n3), "Duplicate trial IDs in N=3 dataset"

    # Verify keys/values are not degenerate
    assert len(set(tuple(t.keys) for t in trials_n2)) == len(trials_n2), "Duplicate keys in N=2 dataset"
    assert len(set(tuple(t.keys) for t in trials_n3)) == len(trials_n3), "Duplicate keys in N=3 dataset"

    cell_reports: Dict[str, Dict[str, Any]] = {}
    total_episodes = 0
    total_queries = 0

    # 2. Evaluate all 6 separate cells
    for N_val, dataset in [(2, trials_n2), (3, trials_n3)]:
        for delay in DELAYS:
            cell_key = f"N{N_val}_delay{int(delay)}ms"
            report = evaluate_confirmatory_cell(
                trials=dataset,
                N_val=N_val,
                delay=delay,
                g_dec=g_dec,
                b_dec=b_dec,
                A_0=A_0
            )
            cell_reports[cell_key] = report
            total_episodes += len(dataset) * len(ALL_MODELS)
            total_queries += len(dataset) * len(ALL_MODELS) * 2
            cur_ram = process.memory_info().rss / (1024 * 1024)
            peak_ram_mb = max(peak_ram_mb, cur_ram)

    # 3. History-Preserving Pooled Bootstrap across delays within each N level (Section 7.5)
    pooled_bootstrap = {}
    for N_val, dataset in [(2, trials_n2), (3, trials_n3)]:
        n_histories = len(dataset)
        pooled_bootstrap[f"N{N_val}"] = {}
        for spk_model in ('M4_Count', 'M4_Latency'):
            # Collect error matrix: shape (n_histories, 3 delays)
            err_matrix = np.zeros((n_histories, 3), dtype=np.float64)
            for d_idx, delay in enumerate(DELAYS):
                c_key = f"N{N_val}_delay{int(delay)}ms"
                raw_cell = cell_reports[c_key]['raw_results'][spk_model]
                for h_idx in range(n_histories):
                    err_matrix[h_idx, d_idx] = raw_cell[h_idx]['e_comp']

            # History-level cluster resampling
            rng_boot = np.random.RandomState(2026091901)
            B = 2000
            boot_means = np.empty(B, dtype=np.float64)
            for b in range(B):
                sampled_histories = rng_boot.randint(0, n_histories, size=n_histories)
                boot_means[b] = np.mean(err_matrix[sampled_histories, :])

            boot_sorted = np.sort(boot_means)
            idx_l = int(round(0.025 * B)) - 1  # 49 (50th replicate)
            idx_u = int(round(0.975 * B)) - 1  # 1949 (1950th replicate)
            lcb = float(boot_sorted[idx_l])
            ucb = float(boot_sorted[idx_u])
            mean_pooled = float(np.mean(err_matrix))

            if ucb < 0.05:
                dec = "DEMONSTRATED_CHANNEL_SUCCESS"
            elif lcb >= 0.05:
                dec = "DEMONSTRATED_CHANNEL_LIMITATION"
            else:
                dec = "INCONCLUSIVE"

            pooled_bootstrap[f"N{N_val}"][spk_model] = {
                'mean_e_comp_pooled': mean_pooled,
                'lcb_95': lcb,
                'ucb_95': ucb,
                'decision': dec
            }

    # 4. Serialize outputs
    prefix = "smoke_" if is_smoke else ""
    summary_file = output_dir / f"{prefix}summary_p1b.json"
    csv_file = output_dir / f"{prefix}condition_summary.csv"
    manifest_file = output_dir / f"{prefix}manifest.sha256"

    # Build clean JSON summary
    clean_summary = {
        'protocol_version': '2.6',
        'is_smoke': is_smoke,
        'total_episodes_simulated': total_episodes,
        'total_role_queries': total_queries,
        'calibration': {'A_0': A_0, 'g_dec': g_dec, 'b_dec': b_dec},
        'cell_results': {},
        'pooled_bootstrap_across_delays': pooled_bootstrap
    }

    for c_key, c_rep in cell_reports.items():
        clean_summary['cell_results'][c_key] = {
            'N': c_rep['N'],
            'delay': c_rep['delay'],
            'n_trials': c_rep['n_trials'],
            'm1_joint_acc': c_rep['m1_joint_acc'],
            'target_floor': c_rep['target_floor'],
            'm1_passed_floor': c_rep['m1_passed_floor'],
            'table_2x2_m1_m2a': c_rep['table_2x2'],
            'theta_L_90': c_rep['theta_L'],
            'theta_U_90': c_rep['theta_U'],
            'in_equivalence_margin': c_rep['in_equivalence_margin'],
            'decision': c_rep['decision'],
            'status_flag': c_rep['status_flag'],
            'redundancy_inferred': c_rep['redundancy_inferred'],
            'decision_text': c_rep['decision_text'],
            'spike_channel': c_rep['spike_channel_results'],
            'model_summaries': c_rep['model_summaries']
        }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(clean_summary, f, indent=2)

    # Write CSV summary
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            'Cell', 'N', 'Delay_ms', 'Model', 'DeliveryAcc_A', 'DeliveryAcc_B',
            'DeliveryAcc_Overall', 'JointDeliveryAcc', 'AbstentionRate', 'Mean_E_comp'
        ])
        for c_key, c_rep in cell_reports.items():
            for m in ALL_MODELS:
                m_sum = c_rep['model_summaries'][m]
                writer.writerow([
                    c_key, c_rep['N'], c_rep['delay'], m,
                    f"{m_sum['delivery_acc_A']:.4f}", f"{m_sum['delivery_acc_B']:.4f}",
                    f"{m_sum['delivery_acc_overall']:.4f}", f"{m_sum['joint_delivery_acc']:.4f}",
                    f"{m_sum['abstention_rate']:.4f}", f"{m_sum['mean_e_comp']:.4f}"
                ])

    # Generate Manifest
    files_to_hash = [summary_file, csv_file]
    manifest_lines = []
    for fp in files_to_hash:
        with open(fp, "rb") as f_in:
            f_hash = hashlib.sha256(f_in.read()).hexdigest().upper()
        manifest_lines.append(f"{f_hash}  {fp.name}\n")

    with open(manifest_file, "w", encoding="utf-8") as f:
        f.writelines(manifest_lines)

    # Verify manifest
    manifest_ok = verify_manifest(manifest_file)
    assert manifest_ok is True, "Manifest verification failed immediately after creation"

    hw_peak_mb = getattr(process.memory_info(), 'peak_wset', 0) / (1024 * 1024)
    peak_ram_mb = max(peak_ram_mb, hw_peak_mb)

    return {
        'total_episodes': total_episodes,
        'total_queries': total_queries,
        'summary_file': summary_file,
        'csv_file': csv_file,
        'manifest_file': manifest_file,
        'manifest_ok': manifest_ok,
        'peak_ram_mb': peak_ram_mb,
        'clean_summary': clean_summary,
        'cell_reports': cell_reports,
        'pooled_bootstrap': pooled_bootstrap
    }

def run_smoke_benchmark(
    n_smoke_per_N: int = 10
) -> Dict[str, Any]:
    """Run full end-to-end smoke benchmark, resource accounting & tamper detection (G1.9)."""
    gen = TaskGenerator(window_size=W_SLOTS)
    rng_2 = np.random.RandomState(42)
    rng_3 = np.random.RandomState(43)
    smoke_n2 = [gen.generate_single_trial(rng_2, tid, 42 + tid, N=2) for tid in range(n_smoke_per_N)]
    smoke_n3 = [gen.generate_single_trial(rng_3, tid, 43 + tid, N=3) for tid in range(n_smoke_per_N)]

    # Calibrated values
    A_0 = 0.061886726
    g_dec = 0.132630817
    b_dec = -0.026816484

    smoke_artifacts_dir = SCRIPT_DIR / "smoke_artifacts"
    smoke_artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Time COMPLETE execution: evaluation + analysis + bootstrap + serialization + manifest
    t_wall_0 = time.perf_counter()
    t_cpu_0 = time.process_time()

    workflow_res = run_full_confirmatory_workflow(
        trials_n2=smoke_n2,
        trials_n3=smoke_n3,
        g_dec=g_dec,
        b_dec=b_dec,
        A_0=A_0,
        output_dir=smoke_artifacts_dir,
        is_smoke=True
    )

    t_wall_elapsed = time.perf_counter() - t_wall_0
    t_cpu_elapsed = time.process_time() - t_cpu_0

    total_episodes_run = workflow_res['total_episodes']  # 10 * 3 * 8 + 10 * 3 * 8 = 480
    cpu_sec_per_episode = t_cpu_elapsed / total_episodes_run
    projected_total_episodes = 96000
    projected_cpu_hours = (cpu_sec_per_episode * projected_total_episodes) / 3600.0
    projected_wall_minutes = ((t_wall_elapsed / total_episodes_run) * projected_total_episodes) / 60.0

    # Tamper Detection Test
    manifest_path = workflow_res['manifest_file']
    summary_path = workflow_res['summary_file']

    # 1. Clean verification must be True
    assert verify_manifest(manifest_path) is True, "Clean manifest must verify as True"

    # 2. Tamper injection: modify summary file on disk
    with open(summary_path, "r", encoding="utf-8") as f:
        orig_content = f.read()
    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n// TAMPER_INJECTION //\n")

    # Re-verify: MUST detect tamper and return False!
    tamper_detected = (verify_manifest(manifest_path) is False)
    assert tamper_detected is True, "Tamper was NOT detected by manifest verifier!"

    # 3. Revert tamper
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(orig_content)
    assert verify_manifest(manifest_path) is True, "Manifest must verify as True after revert"

    return {
        'total_episodes_run': total_episodes_run,
        'total_role_queries': workflow_res['total_queries'],
        't_wall_elapsed_sec': t_wall_elapsed,
        't_cpu_elapsed_sec': t_cpu_elapsed,
        'cpu_sec_per_episode': cpu_sec_per_episode,
        'projected_cpu_hours': projected_cpu_hours,
        'projected_wall_minutes': projected_wall_minutes,
        'peak_ram_mb': workflow_res['peak_ram_mb'],
        'manifest_verified': workflow_res['manifest_ok'],
        'tamper_detected_successfully': tamper_detected,
        'smoke_artifacts_dir': str(smoke_artifacts_dir)
    }
