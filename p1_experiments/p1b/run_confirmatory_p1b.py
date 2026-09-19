"""Authoritative Confirmatory Runner for P1-B.
Governing Specification: P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md
Gate Approval: G1-Executable APPROVED by Codex on September 19, 2026.

Executes the full 96,000 model-episodes (192,000 role queries) across:
- 4,000 unique histories (2,000 for N=2, 2,000 for N=3)
- 3 delays (0 ms, 5 ms, 10 ms)
- 8 models (M1, M2a, M2b, M2c, M2d, M3, M4_Count, M4_Latency)
- Seeded calibration (seed 2026091999, 500 trials)
- Condition-specific decision logic with cell-specific floors (0.80 for N=2, 0.40 for N=3)
- History-preserving delay-pooled bootstrap (B=2000, seed 2026091901)
- Bit-for-bit SHA-256 output manifest generation and independent disk verification.
"""

import os
import sys
import time
import json
import hashlib
import psutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from models_p1b_v2 import calibrate_m1_decoder_and_gate
from runner_p1b import (
    generate_confirmatory_datasets, generate_calibration_trials,
    run_full_confirmatory_workflow, verify_manifest
)

def main():
    print("=" * 80)
    print("STARTING FULL CONFIRMATORY SIMULATION: PHASE P1-B")
    print("Governing Protocol: P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md")
    print("Gate Status: G1-Design APPROVED, G1-Executable APPROVED by Codex")
    print("=" * 80)

    process = psutil.Process(os.getpid())
    t_wall_start = time.perf_counter()
    t_cpu_start = time.process_time()

    # Step 1: Calibration
    print("\n[Step 1/4] Executing Seeded M1 Calibration (500 trials, seed 2026091999)...")
    calib_trials = generate_calibration_trials()
    assert len(calib_trials) == 500, f"Expected 500 calibration trials, got {len(calib_trials)}"
    A_0, g_dec, b_dec = calibrate_m1_decoder_and_gate(calib_trials, rng_seed=2026091999)
    print(f"  Calibrated Parameters: A_0 = {A_0:.9f}, g_dec = {g_dec:.9f}, b_dec = {b_dec:.9f}")
    assert abs(A_0 - 0.061886726) < 1e-6
    assert abs(g_dec - 0.132630817) < 1e-6
    assert abs(b_dec - (-0.026816484)) < 1e-6

    # Step 2: Confirmatory Datasets
    print("\n[Step 2/4] Generating Confirmatory Evaluation Datasets (4,000 unique histories)...")
    trials_n2, trials_n3 = generate_confirmatory_datasets()
    assert len(trials_n2) == 2000, f"N=2 must have 2,000 trials, got {len(trials_n2)}"
    assert len(trials_n3) == 2000, f"N=3 must have 2,000 trials, got {len(trials_n3)}"
    print(f"  Generated {len(trials_n2)} N=2 histories and {len(trials_n3)} N=3 histories.")

    # Step 3: Full Confirmatory Simulation Workflow
    print("\n[Step 3/4] Executing Confirmatory Workflow across all 6 condition cells...")
    print("  Scale: 4,000 histories * 3 delays * 8 models = 96,000 model-episodes (192,000 queries)")
    output_dir = SCRIPT_DIR

    workflow_res = run_full_confirmatory_workflow(
        trials_n2=trials_n2,
        trials_n3=trials_n3,
        g_dec=g_dec,
        b_dec=b_dec,
        A_0=A_0,
        output_dir=output_dir,
        is_smoke=False
    )

    t_wall_elapsed = time.perf_counter() - t_wall_start
    t_cpu_elapsed = time.process_time() - t_cpu_start
    total_episodes = workflow_res['total_episodes']
    total_queries = workflow_res['total_queries']

    print(f"\n  Simulation Complete!")
    print(f"  Total model-episodes: {total_episodes}")
    print(f"  Total role queries: {total_queries}")
    print(f"  Wall time: {t_wall_elapsed:.2f}s ({t_wall_elapsed/60.0:.2f} min)")
    print(f"  CPU time: {t_cpu_elapsed:.2f}s ({t_cpu_elapsed/3600.0:.3f} CPU hours)")
    print(f"  Peak RAM: {workflow_res['peak_ram_mb']:.1f} MB")

    # Step 4: Verification of Manifest and Summary
    print("\n[Step 4/4] Verifying Output Artifacts and Cryptographic Manifest...")
    manifest_file = workflow_res['manifest_file']
    summary_file = workflow_res['summary_file']
    csv_file = workflow_res['csv_file']

    print(f"  Summary JSON: {summary_file}")
    print(f"  Summary CSV:  {csv_file}")
    print(f"  Manifest:     {manifest_file}")

    assert manifest_file.exists(), f"Manifest file missing: {manifest_file}"
    assert summary_file.exists(), f"Summary file missing: {summary_file}"
    assert csv_file.exists(), f"CSV file missing: {csv_file}"

    manifest_valid = verify_manifest(manifest_file)
    print(f"  Independent Manifest Rehash Verification: {manifest_valid}")
    assert manifest_valid is True, "Output manifest verification FAILED!"

    # Print Cell Summary Table
    print("\n" + "=" * 80)
    print("CONFIRMATORY RESULTS SUMMARY BY CELL")
    print("=" * 80)
    clean_summary = workflow_res['clean_summary']
    for c_key, c_data in clean_summary['cell_results'].items():
        print(f"\nCondition Cell: {c_key} (N={c_data['N']}, Delay={c_data['delay']} ms)")
        print(f"  M1 Joint Acc: {c_data['m1_joint_acc']:.4f} (Baseline Floor: {c_data['target_floor']:.2f})")
        print(f"  Floor Passed: {c_data['m1_passed_floor']}")
        print(f"  2x2 Table (n11, n10, n01, n00): {c_data['table_2x2_m1_m2a']}")
        print(f"  M1 vs M2a TOST 90% CI: [{c_data['theta_L_90']:.6f}, {c_data['theta_U_90']:.6f}]")
        print(f"  Cell Decision: {c_data['decision']} ({c_data['status_flag']})")
        print(f"  Memory Redundancy Inferred: {c_data['redundancy_inferred']}")
        for spk_m, spk_d in c_data['spike_channel'].items():
            print(f"  Spike Decoder {spk_m}: Mean E_comp={spk_d['mean_e_comp']:.4f}, 95% CI=[{spk_d['lcb_95']:.4f}, {spk_d['ucb_95']:.4f}] -> {spk_d['decision']}")

    print("\n" + "=" * 80)
    print("DELAY-POOLED BOOTSTRAP RESULTS (Section 7.5)")
    print("=" * 80)
    for N_key, p_data in clean_summary['pooled_bootstrap_across_delays'].items():
        print(f"\nDataset: {N_key} (Pooled across delays 0ms, 5ms, 10ms, 6,000 observations / 2,000 clusters)")
        for spk_m, spk_d in p_data.items():
            print(f"  {spk_m}: Pooled Mean={spk_d['mean_e_comp_pooled']:.4f}, 95% CI=[{spk_d['lcb_95']:.6f}, {spk_d['ucb_95']:.6f}] -> {spk_d['decision']}")

    print("\n" + "=" * 80)
    print("PHASE P1-B CONFIRMATORY SIMULATION SUCCESSFULLY EXECUTED AND VERIFIED")
    print("=" * 80)

if __name__ == "__main__":
    main()
