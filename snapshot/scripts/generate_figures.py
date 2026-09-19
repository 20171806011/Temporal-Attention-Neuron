"""
Generate all paper figures from experimental data.
Usage: python scripts/generate_figures.py
"""
import subprocess
import sys
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    analysis_dir = os.path.join(project_root, "code", "analysis")
    plotting_script = os.path.join(analysis_dir, "plotting.py")

    if not os.path.exists(plotting_script):
        print(f"ERROR: {plotting_script} not found")
        sys.exit(1)

    print("Running plotting script...")
    result = subprocess.run([sys.executable, plotting_script])
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
