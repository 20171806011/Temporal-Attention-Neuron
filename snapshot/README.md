# TAN: Temporal Attention Neuron — Final Submission Package

## Project Description

This package contains the complete research artefacts for the paper:

> **The Temporal Attention Neuron: A Minimalist Fusion of Transformer Attention and Spiking Neural Dynamics**
>
> Author: Li Zexu, University of Leeds, School of Physics and Astronomy

TAN is a single-neuron model that combines a Boltzmann-style temporal attention kernel with neuromodulatory surprise gating on top of a classical leaky integrate-and-fire (LIF) neuron. The model retains the local Markovian physical substrate of LIF while adding a non-Markovian temporal receptive field driven by a deviation-from-history signal. Experiments show that habituation, sparse sensory gating under noise, and adaptive niche localization emerge from the dynamics of this system.

### Scientific Motivation

Deep learning (Transformers) and computational neuroscience (spiking neurons) have evolved on disconnected tracks. TAN bridges them by implementing QKV-like temporal weighting at the single-neuron level, driven by a local deviation signal rather than raw input amplitude.

## Directory Structure

```
TAN_Final_Submission/
├── manuscript/          # LaTeX source + compiled PDF + figures
│   ├── main.tex
│   ├── main.pdf
│   ├── reference.bib
│   ├── sections/        # 7 section .tex files
│   └── figures/         # 21 figures (PNG + PDF)
├── code/                # All experiment and analysis code
│   ├── model/           # TAN, LIF, baselines
│   ├── experiments/     # 6 experiment scripts
│   ├── analysis/        # Statistics, plotting, MI
│   └── requirements.txt
├── data/                # Experimental data
│   └── experiment_results/  # 16 CSV/NPZ files
├── results/             # Generated outputs
│   ├── figures/
│   ├── tables/
│   └── logs/
├── scripts/             # One-click automation
│   ├── run_all_experiments.sh
│   ├── run_all_experiments.bat
│   ├── generate_figures.py
│   └── compile_pdf.sh
├── README.md
├── LICENSE
└── CHANGELOG.md
```

## Installation

### Prerequisites

- Python 3.8+
- LaTeX distribution (Tectonic recommended, or TeX Live/MiKTeX)

### Setup

```bash
# Clone or extract this package
cd TAN_Final_Submission

# Create virtual environment (optional but recommended)
conda create -n tan python=3.11
conda activate tan

# Install Python dependencies
pip install -r code/requirements.txt
```

## Reproduce Paper

### Step 1: Run Experiments

```bash
# Linux/Mac:
bash scripts/run_all_experiments.sh

# Windows:
scripts\run_all_experiments.bat

# Or run individually:
python code/experiments/habituation.py
python code/experiments/statistical_validation.py
python code/experiments/baseline_comparison.py
python code/experiments/ablation.py
python code/experiments/robustness.py
python code/experiments/information_analysis.py
python code/experiments/natural_collision.py   # Phase 1: state-collision / non-Markovianity test
```

This generates:
- Experimental data in `data/experiment_results/`
- Figures in `results/figures/` (incl. `fig22_natural_collision`, `fig23_reset_rebifurcation`)
- Logs in `results/logs/`

The Phase-1 experiment (natural state-collision test of the Markovian
claim for `h_t`) is documented in `PHASE1_NATURAL_COLLISION_REPORT.md`
(theory, protocol, statistics, and the Phase-2 PR/TwoNN interface).

### Step 2: Generate Figures (if needed)

```bash
python scripts/generate_figures.py
```

### Step 3: Compile PDF

```bash
bash scripts/compile_pdf.sh
```

Or manually:
```bash
cd manuscript/
tectonic main.tex
# OR:
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The output PDF is at `manuscript/main.pdf`.

## Data Availability

Pre-generated experimental data is included in `data/experiment_results/`:
- 16 CSV files containing raw metrics, statistical summaries, and contribution matrices
- 1 NPZ file containing state-space trajectory arrays

These results were generated with fixed random seeds (seeds 0-49 for 50-seed experiments, seeds 0-19 for 20-seed parameter sweeps). Re-running the scripts will reproduce identical results.

## Notes

- RNN and GRU baselines are **distillation/imitation baselines** trained on TAN-generated expert trajectories, not independent general-purpose trained models.
- The clean phototaxis task (Task C) is deterministic; no statistical test is applicable there.
- The LIF baseline in the habituation task (Task A) has a calibration limitation (threshold too high for stimulus amplitude). See manuscript Limitations section.
- A net energy advantage of TAN cannot be established without hardware-level energy measurements.
