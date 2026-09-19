@echo off
REM Run all TAN experiments on Windows
REM Usage: scripts\run_all_experiments.bat

set EXP_DIR=%~dp0..\code\experiments
set RESULTS_DIR=%~dp0..\results
mkdir "%RESULTS_DIR%\figures" 2>nul
mkdir "%RESULTS_DIR%\tables" 2>nul
mkdir "%RESULTS_DIR%\logs" 2>nul

echo ========================================
echo TAN Experiment Suite
echo ========================================

echo.
echo [1/6] Stage 1: Single-seed phenomenology...
cd /d "%EXP_DIR%"
python habituation.py

echo.
echo [2/6] Stage 2 Part 1: Statistical validation (50 seeds)...
python statistical_validation.py

echo.
echo [3/6] Stage 2 Part 2: Baseline comparison (5 models)...
python baseline_comparison.py

echo.
echo [4/6] Stage 2 Part 3: Ablation study...
python ablation.py

echo.
echo [5/6] Stage 2 Part 4: Parameter robustness...
python robustness.py

echo.
echo [6/6] Stage 2 Part 5: Dynamics analysis...
python information_analysis.py

echo.
echo ========================================
echo All experiments complete.
echo ========================================
