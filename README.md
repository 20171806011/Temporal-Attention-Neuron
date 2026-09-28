# Temporal Attention Neuron (TAN): Complete Research Programme

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: Final Revised Manuscripts](https://img.shields.io/badge/Status-Complete%20Research%20Suite-success.svg)]()
[![Repository: Unified Monorepo](https://img.shields.io/badge/Repository-Unified%20Research%20Monorepo-blueviolet.svg)]()

> **Author**: **Li Zexu** (School of Physics and Astronomy, University of Leeds)  
> **Official Repository**: [https://github.com/20171806011/Temporal-Attention-Neuron](https://github.com/20171806011/Temporal-Attention-Neuron)  
> *(Note: The legacy repository `20171806011/-TAN-` has been formally deprecated and archived. All official code, experimental data, analytical benchmarks, and publication manuscripts across the entire research lifecycle are consolidated in this repository).*

---

## 🌟 Overview & Core Question

**How can minimal biological and dynamical mechanisms enable single spiking units, or compact neural circuits, to select, hold, and combine continuous representations across time?**

The **Temporal Attention Neuron (TAN)** is a bio-inspired computational framework that unifies the temporal expressiveness of Transformer self-attention with the continuous-time dynamics of Leaky Integrate-and-Fire (LIF) spiking neurons. By coupling a rectified surprise gating signal $S_t = [x_t - \mu_t - \varepsilon]_+$ with dynamic sliding-window attention, TAN achieves sensory habituation, noise filtering, and closed-loop adaptive navigation **without requiring synaptic weight updates**.

This repository contains the complete, reproducible research lineage—from the initial single-neuron prototype through exhaustive mechanistic boundary analysis to the multi-channel relational computation ladder.

---

## 👤 Author & Research Contributions

All theoretical formulations, computational models, experimental architectures, and manuscripts in this repository were conceived, designed, and authored by **Li Zexu**:

### 1. Conceptualization & Mathematical Formulation
* **The TAN Paradigm**: Conceived the integration of continuous-time biological leaky integration with dynamic, sliding-window temporal self-attention.
* **Rectified Surprise Gating**: Formulated the local deviation novelty signal $S_t = \max(0, x_t - \operatorname{mean}(H_t) - \varepsilon)$, enabling the neuron to autonomously gate information flow without gradient-based weight updates.
* **Mechanistic Boundary Theory (TAN-I)**: Discovered and mathematically formulated the *Natural State Collision* phenomenon—proving that the membrane potential $h_t$ is not Markov-sufficient, and establishing that the neuron retains rich historical trajectory memory.
* **Relational Architectural Ladder (TAN-II)**: Constructed the conceptual ladder charting the necessary architectural transitions from temporal saliency filtering to opponent E-I competition, vector-space metric addressing, Winner-Take-All symbol selection, and algebraic composition.

### 2. Algorithmic Implementation & Experimental Design
* **Model Implementation**: Developed the foundational TAN codebase (`tan.py`, `lif.py`), along with comparative biological and neural baselines (LIF, Adaptive LIF, RNN, GRU).
* **Sensorimotor Benchmarks**: Designed the 50-seed phototaxis navigation and sensory habituation task suites, demonstrating autonomous gradient ascent and noise suppression.
* **Large-Scale Empirical Verification**:
  * **Phase P1-A (60,000 trials, 420,000 evaluations)**: Discovered that 1D metric compatibility $-(q - k)^2$ is mathematically and empirically sufficient for 100% hard key selection, proving that higher-dimensional vector spaces are not strictly required for 1D selective addressing.
  * **Phase P1-B (96,000 episodes)**: Designed and executed the continuous dynamical carrier and spiking readout stress-testing suite, empirically characterizing the capacity limits of single-unit channels and establishing the necessity of distributed multi-channel circuits for multi-item binding.

### 3. Academic Manuscripts
* **Sole Lead Author** of all three comprehensive research manuscripts detailing the theoretical derivation, mechanistic limits, and architectural expansion of the TAN framework.

---

## 🧭 Complete Scientific Research Lineage

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Initial Formulation & Sensorimotor Dynamics (The Original TAN Neuron)                 │
│ • Model: TAN (LIF dynamics + surprise rectification + temporal self-attention kernel)           │
│ • Experiments: Sensory habituation, noise gating, closed-loop phototaxis (50-seed benchmarks)   │
│ • Manuscript: The Temporal Attention Neuron: A Minimalist Fusion of Transformer Attention & SNNs│
│ • Artifacts: snapshot/code/  |  snapshot/manuscript/  |  snapshot/data/  |  snapshot/results/   │
└────────────────────────────────┬────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Mechanistic Boundary Analysis (TAN-I Deep Dive)                                        │
│ • Geometry: Natural state collisions (membrane potential h_t exhibits non-Markovian memory)     │
│ • Proof: Order-locking theorem of positive scalar kernels (FlipRate ≡ 0.000, JSD analysis)      │
│ • Empirical: Event-locked response geometry, effective dimensionality, and distractor audits    │
│ • Manuscript 1: The Computational Boundary of Scalar Temporal Attention (Strictly 28 pages)     │
│ • Artifacts: snapshot/paper/  |  release/TAN_I_Mechanistic_Analysis.pdf                        │
└────────────────────────────────┬────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Relational Computation & Minimal Architectural Ladder (TAN-II & Phase P1)              │
│ • Sprint Ladder: E-I opponent competition → 2D vector QK → KV binding → WTA → parallel combiner │
│ • Phase P1-A: 1D metric compatibility sufficiency & intermediate addressing barrier (60k trials)│
│ • Phase P1-B: Continuous dynamical carrier & spiking readout characterization (96k episodes)   │
│ • Manuscript 2: The Minimal Architectural Ladder for Relational Computation (Strictly 28 pages) │
│ • Artifacts: p1_experiments/  |  snapshot/paper2/  |  release/TAN_II_...pdf                    │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Key Scientific Discoveries & Insights

### 1. Stage 1: Dynamic Sensory Gating Without Synaptic Plasticity
* Integrates a local sliding window $W=5$ with surprise rectification $S_t = \max(0, x_t - \mu_t - \varepsilon)$ to dynamically modulate attention weights.
* Achieves rapid habituation to repetitive stimuli and robust noise filtering without synaptic weight modification.
* Successfully drives closed-loop phototaxis navigation in simulated sensory environments, significantly outperforming standard LIF baselines across 50 independent runs.

### 2. Stage 2: Non-Markovian Memory & Theoretical Boundaries (TAN-I)
* **Natural State Collisions**: In 160,000 simulated histories, distinct input sequences reach identical membrane potentials $h_t$ (within numerical tolerance $10^{-4}$) and subsequently diverge by a mean expansion factor of $2.05 \times 10^4$. This mathematically proves that $h_t$ alone is not Markov-sufficient, demonstrating that the full state $(h_t, H_t)$ preserves genuine temporal trajectory information.
* **Scalar Kernel Boundary**: Proves both analytically and empirically (Theorem 5.1; 1,000 counterfactual trials, FlipRate $\equiv 0.000$) that positive scalar attention kernels preserve the rank order of input scores, establishing the constructive motivation for multi-channel and metric addressing.

### 3. Stage 3: The Relational Architecture Ladder (TAN-II & Phase P1)
* **1D Metric Compatibility Sufficiency (Phase P1-A)**: Across 60,000 empirical trials and 420,000 evaluations, demonstrated that a 1D metric distance kernel $-(q - k)^2$ under hard selection achieves 100% accuracy in selecting target keys—including intermediate keys—matching 2D vector rotation performance while requiring only a single scalar dimension.
* **Scalar Addressing Barrier**: Proved that monotonic dot-product attention cannot isolate intermediate keys ($0/6{,}705$ selections), rigorously delineating the boundary between monotonic scoring and metric compatibility.
* **Dynamical Carrier & Readout Limits (Phase P1-B)**: Conducted 96,000 adversarial test episodes evaluating continuous subthreshold carriers and single-unit spiking decoders, revealing the precise geometric operating conditions of single units and demonstrating why robust compositional binding naturally favors distributed multi-channel circuits.

---

## 📚 Publications & Manuscripts

The repository hosts full LaTeX sources and publication-grade precompiled PDFs:

1. **Stage 1 Manuscript**:  
   *The Temporal Attention Neuron: A Minimalist Fusion of Transformer Attention and Spiking Neural Dynamics*  
   - LaTeX Source: [`snapshot/manuscript/`](snapshot/manuscript/)
2. **Stage 2 Manuscript (Paper 1 — TAN-I Mechanistic Analysis)**:  
   *The Computational Boundary of Scalar Temporal Attention: A Mechanistic Analysis of the Temporal Attention Neuron*  
   - Precompiled PDF: [`release/TAN_I_Mechanistic_Analysis.pdf`](release/TAN_I_Mechanistic_Analysis.pdf) *(28 pages, complete citations, audited)*  
   - LaTeX Source: [`snapshot/paper/`](snapshot/paper/)
3. **Stage 3 Manuscript (Paper 2 — TAN-II Minimal Architectural Ladder)**:  
   *The Minimal Architectural Ladder for Relational Computation: From Temporal Saliency to Binding and Composition in a Dynamical Neuron Model*  
   - Precompiled PDF: [`release/TAN_II_Minimal_Architectural_Ladder.pdf`](release/TAN_II_Minimal_Architectural_Ladder.pdf) *(28 pages, complete citations, audited)*  
   - LaTeX Source: [`snapshot/paper2/`](snapshot/paper2/)

---

## 🚀 Quickstart & Reproduction Guide

All experiments, models, and test fixtures are fully reproducible out of the box.

### 1. Environment Setup

Python 3.10, 3.11, or 3.12 is recommended. Install dependencies via `requirements.txt`:

```bash
git clone https://github.com/20171806011/Temporal-Attention-Neuron.git
cd Temporal-Attention-Neuron
pip install -r requirements.txt
```

> [!TIP]
> **Cross-Platform Compatibility**: Repository files are tracked with `.gitattributes` to enforce uniform LF line endings. All cryptographic SHA-256 fixture checks pass identically on Windows, Linux, and macOS.

### 2. Running the Test Suites

Execute the Phase P1-B automated fixture tests (12 tests covering analytical collisions, metric routing, and ODE trajectories):

```bash
python -m pytest p1_experiments/p1b/test_p1b_g1_fixtures.py
```

### 3. Running Manuscript Quality & Verification Audits

Both compiled manuscripts feature automated text, citation, and layout QA validators:

```bash
# Verify Paper 1 (Strictly 28 pages, 0 missing refs, all structural checks pass)
cd snapshot && python paper/scripts/qa_pdf.py && cd ..

# Verify Paper 2 (Strictly 28 pages, 0 ??, all citations [1]-[13] verified)
cd snapshot/paper2 && python scripts/qa_paper2.py && cd ..
```

### 4. Running Stage 1 Experiments

Reproduce the original Stage 1 baseline benchmarks and generate figures:

```bash
# On Linux / macOS:
bash snapshot/scripts/run_all_experiments.sh

# On Windows:
snapshot\scripts\run_all_experiments.bat
```

> [!NOTE]
> PyTorch is only needed if executing the deep-learning baseline comparisons (RNN/GRU) in Stage 1; all core TAN neuron dynamics and theoretical benchmarks execute purely on standard `numpy` and `scipy`.

---

## 📂 Repository Directory Layout

```
Temporal-Attention-Neuron/
├── README.md                           # [This file] Complete research lineage and reproduction guide
├── requirements.txt                    # Python environment requirements
├── LICENSE                             # MIT License
│
├── release/                            # Precompiled manuscripts and release manifests
│   ├── TAN_I_Mechanistic_Analysis.pdf  # Paper 1 (28 pages, complete)
│   ├── TAN_II_Minimal_Architectural_Ladder.pdf # Paper 2 (28 pages, complete)
│   └── FINAL_MANUSCRIPT_RELEASE_MANIFEST.md
│
├── snapshot/                           # Working trees for Stage 1 & Stage 2
│   ├── code/                           # Core models (tan.py, lif.py, baselines.py) and experiments
│   ├── manuscript/                     # Stage 1 initial paper LaTeX sources
│   ├── paper/                          # Stage 2 revised Paper 1 LaTeX sources and figures
│   ├── paper2/                         # Stage 3 revised Paper 2 LaTeX sources and figures
│   ├── data/                           # Experimental data records (.csv, .npz)
│   └── results/                        # Generated figures (fig01–fig27)
│
├── p1_experiments/                     # Stage 3: Phase P1 experimental suites
│   ├── results/run_p1a_rev2/           # Phase P1-A final verified data (60,000 trials)
│   └── p1b/                            # Phase P1-B protocol, ODE carrier, and fixture suites
│
├── s41cd/ ... s43a/                    # Stage 3 TAN-II Sprint reproduction suites
├── scripts/                            # Verification tools, QA scripts, and ledger generators
├── CURRENT_RESEARCH_STATE.md           # Research state audit and boundary matrix
├── RESEARCH_CHRONOLOGY.md              # Complete experimental chronology
└── ERRATA.md                           # Technical errata and amendment history
```

---

## 📄 License & Open Science Attribution

* **Author**: Li Zexu (School of Physics and Astronomy, University of Leeds, 2026).
* **License**: Released under the [MIT License](LICENSE) for open academic study, replication, and scientific expansion.
* **Computational Tooling**: Implementation verification, regression testing harnesses, and documentation consistency checks were assisted by AI coding agents (OpenAI Codex, Google Antigravity) under the author's direct theoretical guidance and review.
