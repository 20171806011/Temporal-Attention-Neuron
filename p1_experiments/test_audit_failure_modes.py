"""Targeted Failure-Mode Regression Suite for P1-A Audit & Resource Checks.

Validates that verify_lossless_ledger_recomputation() and check_runtime_resource_budget()
CANNOT produce a false PASS when presented with corrupted data, missing coverage,
tampered summaries, or resource overruns.

Uses existing dry-run output data without generating new scientific trials.
"""

import os
import sys
import shutil
import tempfile
import json
import psutil
import pandas as pd
import numpy as np

from run_p1a_experiment import (
    verify_lossless_ledger_recomputation,
    check_runtime_resource_budget
)

def run_tests():
    print("================================================================================")
    print("      Running Targeted Failure-Mode Tests on P1-A Audit Verification           ")
    print("================================================================================")

    # Find the most recent dry_run directory to use as baseline
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    dry_runs = sorted([d for d in os.listdir(results_dir) if d.startswith('dry_run_')])
    assert len(dry_runs) > 0, "No dry_run directory found to test against!"
    baseline_dir = os.path.join(results_dir, dry_runs[-1])
    print(f"Using baseline directory: {baseline_dir}")

    # Load baseline summary
    with open(os.path.join(baseline_dir, 'summary_p1a.json'), 'r', encoding='utf-8') as f:
        baseline_summary = json.load(f)

    # Test 1: Duplicate Key in Inputs
    print("\n[Test 1] Testing Duplicate Keys in trial_inputs...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Copy baseline files
        for f in os.listdir(baseline_dir):
            shutil.copy(os.path.join(baseline_dir, f), os.path.join(tmp_dir, f))

        # Corrupt trial_inputs with a duplicate row
        df_in = pd.read_csv(os.path.join(tmp_dir, 'trial_inputs.csv.gz'), compression='gzip')
        df_corrupt = pd.concat([df_in, df_in.iloc[[0]]], ignore_index=True)
        df_corrupt.to_csv(os.path.join(tmp_dir, 'trial_inputs.csv.gz'), index=False, compression='gzip')

        try:
            verify_lossless_ledger_recomputation(tmp_dir, baseline_summary)
            raise RuntimeError("TEST 1 FAILED: Duplicate input key was NOT caught!")
        except AssertionError as e:
            assert "Duplicate join keys detected in trial_inputs" in str(e)
            print(f"  [PASS] Duplicate input key correctly caught: {e}")

    # Test 2: Incomplete Model Coverage in Outputs
    print("\n[Test 2] Testing Incomplete Model Coverage in model_outputs...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        for f in os.listdir(baseline_dir):
            shutil.copy(os.path.join(baseline_dir, f), os.path.join(tmp_dir, f))

        # Drop 1 model from 1 trial
        df_out = pd.read_csv(os.path.join(tmp_dir, 'model_outputs.csv.gz'), compression='gzip')
        df_corrupt = df_out.iloc[1:].copy()  # drop first row
        df_corrupt.to_csv(os.path.join(tmp_dir, 'model_outputs.csv.gz'), index=False, compression='gzip')

        try:
            verify_lossless_ledger_recomputation(tmp_dir, baseline_summary)
            raise RuntimeError("TEST 2 FAILED: Incomplete model coverage was NOT caught!")
        except AssertionError as e:
            assert "Incomplete model coverage" in str(e) or "Row count mismatch" in str(e)
            print(f"  [PASS] Incomplete model coverage correctly caught: {e}")

    # Test 3: Empty Zero-Error Set
    print("\n[Test 3] Testing Empty Zero-Error Target Set...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        for f in os.listdir(baseline_dir):
            shutil.copy(os.path.join(baseline_dir, f), os.path.join(tmp_dir, f))

        # Modify model_outputs so no trials have both_correct == True
        df_out = pd.read_csv(os.path.join(tmp_dir, 'model_outputs.csv.gz'), compression='gzip')
        df_out.loc[df_out['model'] == 'ScalarMetricAttention', 'both_correct'] = False
        df_out.to_csv(os.path.join(tmp_dir, 'model_outputs.csv.gz'), index=False, compression='gzip')

        try:
            verify_lossless_ledger_recomputation(tmp_dir, baseline_summary)
            raise RuntimeError("TEST 3 FAILED: Empty zero-error set was NOT caught!")
        except AssertionError as e:
            assert "Zero-error target evaluation set is empty" in str(e)
            print(f"  [PASS] Empty zero-error set correctly caught: {e}")

    # Test 4: Summary Discrepancy (Live Summary Tampering)
    print("\n[Test 4] Testing Summary Discrepancy Detection...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        for f in os.listdir(baseline_dir):
            shutil.copy(os.path.join(baseline_dir, f), os.path.join(tmp_dir, f))

        # Tamper with summary both_routed_acc for one model
        tampered_summary = json.loads(json.dumps(baseline_summary))
        tampered_summary['conditions']['C1_N2_zero_noise']['ScalarMetricAttention']['both_routed_acc'] = 0.95

        try:
            verify_lossless_ledger_recomputation(tmp_dir, tampered_summary)
            raise RuntimeError("TEST 4 FAILED: Tampered summary was NOT caught!")
        except AssertionError as e:
            assert "Summary discrepancy in both_routed_acc" in str(e)
            print(f"  [PASS] Summary discrepancy correctly caught: {e}")

    # Test 5: Resource Budget Exceeded Check
    print("\n[Test 5] Testing Runtime Resource Budget Monitor Abort...")
    proc = psutil.Process()
    cpu_start = proc.cpu_times()

    # Synthetic CPU overrun
    try:
        check_runtime_resource_budget(proc, cpu_start, max_cpu_sec=-1.0, stage_desc="Synthetic-Test")
        raise RuntimeError("TEST 5a FAILED: CPU overrun was NOT caught!")
    except RuntimeError as e:
        assert "RESOURCE BUDGET EXCEEDED: CPU time" in str(e)
        print(f"  [PASS] CPU budget overrun correctly raised RuntimeError: {e}")

    # Synthetic RAM overrun
    try:
        check_runtime_resource_budget(proc, cpu_start, max_ram_mb=0.001, stage_desc="Synthetic-Test")
        raise RuntimeError("TEST 5b FAILED: RAM overrun was NOT caught!")
    except RuntimeError as e:
        assert "RESOURCE BUDGET EXCEEDED: Peak memory" in str(e)
        print(f"  [PASS] RAM budget overrun correctly raised RuntimeError: {e}")

    print("\n================================================================================")
    print("ALL 5 TARGETED FAILURE-MODE TESTS PASSED SUCCESSFULLY!")
    print("The audit functions are proven to reject corrupted or out-of-budget inputs.")
    print("================================================================================")

if __name__ == '__main__':
    run_tests()
