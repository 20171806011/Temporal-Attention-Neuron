#!/bin/bash
# Run all TAN experiments and generate figures
# Usage: bash scripts/run_all_experiments.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EXP_DIR="$PROJECT_ROOT/code/experiments"
RESULTS_DIR="$PROJECT_ROOT/results"
mkdir -p "$RESULTS_DIR/figures" "$RESULTS_DIR/tables" "$RESULTS_DIR/logs"

echo "========================================"
echo "TAN Experiment Suite"
echo "========================================"

echo ""
echo "[1/6] Stage 1: Single-seed phenomenology..."
cd "$EXP_DIR"
python habituation.py 2>&1 | tee "$RESULTS_DIR/logs/stage1.log"

echo ""
echo "[2/6] Stage 2 Part 1: Statistical validation (50 seeds)..."
python statistical_validation.py 2>&1 | tee "$RESULTS_DIR/logs/part1_stats.log"

echo ""
echo "[3/6] Stage 2 Part 2: Baseline comparison (5 models)..."
python baseline_comparison.py 2>&1 | tee "$RESULTS_DIR/logs/part2_baselines.log"

echo ""
echo "[4/6] Stage 2 Part 3: Ablation study..."
python ablation.py 2>&1 | tee "$RESULTS_DIR/logs/part3_ablation.log"

echo ""
echo "[5/6] Stage 2 Part 4: Parameter robustness..."
python robustness.py 2>&1 | tee "$RESULTS_DIR/logs/part4_robustness.log"

echo ""
echo "[6/6] Stage 2 Part 5: Dynamics analysis..."
python information_analysis.py 2>&1 | tee "$RESULTS_DIR/logs/part5_dynamics.log"

echo ""
echo "========================================"
echo "All experiments complete."
echo "Figures: $RESULTS_DIR/figures/"
echo "Tables:  $RESULTS_DIR/tables/"
echo "Logs:    $RESULTS_DIR/logs/"
echo "========================================"
