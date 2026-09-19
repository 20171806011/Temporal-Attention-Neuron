"""
Plotting Utilities
===================
Wrapper script to regenerate all figures from experimental data.

Usage:
    python plotting.py

This script runs the experiment scripts which generate figures
into the results/figures/ directory.

Author: Li Zexu, University of Leeds
"""
import subprocess
import sys
import os

EXPERIMENT_SCRIPTS = [
    "habituation.py",
    "statistical_validation.py",
    "baseline_comparison.py",
    "ablation.py",
    "robustness.py",
    "information_analysis.py",
]

def main():
    exp_dir = os.path.join(os.path.dirname(__file__), "..", "experiments")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "figures")
    os.makedirs(output_dir, exist_ok=True)

    for script in EXPERIMENT_SCRIPTS:
        script_path = os.path.join(exp_dir, script)
        if not os.path.exists(script_path):
            print(f"WARNING: {script} not found, skipping")
            continue
        print(f"Running {script}...")
        result = subprocess.run([sys.executable, script_path],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR in {script}: {result.stderr[:500]}")
        else:
            print(f"  {script} completed")

    print("\nAll figures generated. Check results/figures/")

if __name__ == "__main__":
    main()
