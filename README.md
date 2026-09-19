# Temporal Attention Neuron (TAN): Complete Research Programme

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Status: Final Revised Manuscripts](https://img.shields.io/badge/Status-Final%20Revised%20Manuscripts-success.svg)]()
[![Repository: Unified Monorepo](https://img.shields.io/badge/Repository-Unified%20Research%20Suite-blueviolet.svg)]()

> **Author**: Li Zexu  
> **Affiliation**: School of Physics and Astronomy, University of Leeds  
> **Official Repository**: [https://github.com/20171806011/Temporal-Attention-Neuron](https://github.com/20171806011/Temporal-Attention-Neuron)  
> *(Note: The earlier repository `20171806011/-TAN-` has been formally deprecated and archived. All official code, data, experiments, and publications across the entire research lifecycle are consolidated in this repository).*

---

## 🧭 Complete Scientific Research Lineage & Roadmap

This unified repository provides end-to-end provenance across all three evolutionary phases of the **Temporal Attention Neuron (TAN)** project:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 1: Initial Formulation & Baseline Experiments (The Original TAN Neuron)                   │
│ • Model: TANNeuronAblation (LIF + surprise gating + temporal self-attention kernel)              │
│ • Experiments: Habituation, noise gating, phototaxis navigation, 5-model baselines, ablations   │
│ • Paper: The Temporal Attention Neuron: A Minimalist Fusion of Transformer Attention & SNNs     │
│ • Artifacts: snapshot/code/  |  snapshot/manuscript/  |  snapshot/data/  |  results/fig01-fig21 │
└────────────────────────────────┬────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 2: Mechanistic Boundary Analysis (TAN-I Deep Dive)                                        │
│ • Phase 1: Natural state collisions (membrane potential h_t is not Markov-sufficient)          │
│ • Phase 2: Event-locked response geometry & effective dimension analysis (d_D)                 │
│ • Probe-1 & Probe-2: Proving scalar dot-product attention is order-locked & cannot route       │
│ • Paper 1: The Computational Boundary of Scalar Temporal Attention (Strictly 28 pages)          │
│ • Artifacts: snapshot/paper/  |  release/TAN_I_Mechanistic_Analysis.pdf                        │
└────────────────────────────────┬────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Stage 3: Relational Computation & Minimal Architectural Ladder (TAN-II & Phase P1)              │
│ • Sprint Ladder: E-I opponent competition → 2D vector QK → KV binding → WTA → parallel combiner │
│ • Phase P1-A: 1D metric compatibility sufficiency & intermediate addressing barrier (60k trials)│
│ • Phase P1-B: Continuous dynamical carrier & spiking readout audit (96k episodes, Protocol V2.6)│
│ • Paper 2: The Minimal Architectural Ladder for Relational Computation (Strictly 28 pages)      │
│ • Artifacts: p1_experiments/  |  snapshot/paper2/  |  release/TAN_II_...pdf                    │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Publications & Manuscripts

The repository hosts all manuscripts corresponding to each research phase:

1. **Stage 1 (Initial Paper)**:  
   *The Temporal Attention Neuron: A Minimalist Fusion of Transformer Attention and Spiking Neural Dynamics*  
   - LaTeX Source: [`snapshot/manuscript/`](snapshot/manuscript/)  
   - Core Theme: Original single-neuron model fusing temporal attention, local surprise rectification, and spiking dynamics.

2. **Stage 2 (Paper 1 — TAN-I Mechanistic Analysis)**:  
   *The Computational Boundary of Scalar Temporal Attention: A Mechanistic Analysis of the Temporal Attention Neuron*  
   - PDF: [`release/TAN_I_Mechanistic_Analysis.pdf`](release/TAN_I_Mechanistic_Analysis.pdf) *(Strictly 28 pages, 0 `??`, fully audited)*  
   - LaTeX Source: [`snapshot/paper/`](snapshot/paper/)  
   - Core Theme: Mathematical boundary proofs, natural state collisions, and order-locking under scalar attention kernels.

3. **Stage 3 (Paper 2 — TAN-II Minimal Architectural Ladder)**:  
   *The Minimal Architectural Ladder for Relational Computation: From Temporal Saliency to Binding and Composition in a Dynamical Neuron Model*  
   - PDF: [`release/TAN_II_Minimal_Architectural_Ladder.pdf`](release/TAN_II_Minimal_Architectural_Ladder.pdf) *(Strictly 28 pages, 0 `??`, fully audited)*  
   - LaTeX Source: [`snapshot/paper2/`](snapshot/paper2/)  
   - Core Theme: Constructive ascent along named organizational degrees of freedom, integrating the certified Phase P1-A and Phase P1-B audits.

---

## 🔬 Core Code Modules & Directory Layout

```
Temporal-Attention-Neuron/
├── README.md                           # [This file] Complete research lineage and reproduction guide
│
├── release/                            # Final compiled manuscripts and cryptographic manifest
│   ├── TAN_I_Mechanistic_Analysis.pdf  # Final revised Paper 1 (28 pages)
│   ├── TAN_II_Minimal_Architectural_Ladder.pdf # Final revised Paper 2 (28 pages)
│   └── FINAL_MANUSCRIPT_RELEASE_MANIFEST.md    # Cryptographic hashes and release records
│
├── snapshot/                           # Complete baseline and manuscript working trees
│   ├── code/                           # STAGE 1: Original TAN implementation
│   │   ├── model/                      # Core models: tan.py, lif.py, baselines.py
│   │   ├── experiments/                # Experiments: habituation, statistical_validation, ablation, etc.
│   │   └── analysis/                   # Analysis tools: statistics, plotting, mutual_information
│   ├── manuscript/                     # STAGE 1: Original TAN manuscript LaTeX sources
│   ├── paper/                          # STAGE 2: Revised Paper 1 LaTeX sources and figures
│   ├── paper2/                         # STAGE 3: Revised Paper 2 LaTeX sources and figures
│   ├── data/                           # Raw experimental datasets (.csv, .npz)
│   ├── results/                        # Generated figures (fig01–fig27) and statistical tables
│   └── scripts/                        # Automation scripts (run_all_experiments, compile_pdf)
│
├── p1_experiments/                     # STAGE 3: Phase P1 breakthrough experimental suites
│   ├── p1a/                            # Phase P1-A: 1D metric compatibility (60,000 trials, 420,000 evals)
│   └── p1b/                            # Phase P1-B: Protocol V2.6 ODE carrier & spiking audit (96,000 episodes)
│
├── s41cd/ ... s43a/                    # STAGE 3: TAN-II Sprint reproduction suites (4.1 to 4.3-A)
│
├── scripts/                            # Independent verification and QA test suites
│   ├── qa_pdf.py                       # Paper 1 layout, references, and text QA validator
│   ├── qa_paper2.py                    # Paper 2 layout, references, and text QA validator
│   └── generate_manifest_after.py      # Bidirectional cryptographic hash integrity verifier
│
├── MASTER_CLAIM_LEDGER.csv             # Full claim ledger across all rungs and audits
├── CURRENT_RESEARCH_STATE.md           # Research state audit and boundary matrix
├── RESEARCH_CHRONOLOGY.md              # Complete 25-round experimental chronology
└── ERRATA.md                           # Technical errata and amendment history
```

---

## 🧪 Scientific Highlights & Breakthrough Findings

### 1. The Original TAN Neuron (Stage 1)
- Combines a local deviation surprise signal $S_t = \max(0, x_t - \operatorname{mean}(H_t) - \tau_{\text{noise}})$ with QKV temporal attention over a history buffer $W$.
- Demonstrates sensory habituation, noise-gated stimulus response, and adaptive phototaxis navigation without synaptic weight updates.
- 50-seed statistical validation against LIF, adaptive-LIF, RNN, SNN, and Transformer baselines.

### 2. State Collisions & Saliency Locking (Stage 2 / TAN-I)
- **Theorem (Natural State Collisions)**: Constructive pairs of distinct input histories collapse to identical membrane potentials $h_t$ before diverging, proving that $h_t$ is not Markov-sufficient and establishing genuine temporal memory.
- **Probe-1 & Probe-2**: Proves that scalar positive attention kernels are order-locked ($\mathrm{FlipRate} \equiv 0.000$), demonstrating the mathematical impossibility of query-conditioned routing in scalar substrates.

### 3. The Relational Ladder & Phase P1 (Stage 3 / TAN-II)
- **Orthogonal Axis 1**: Routing Selectivity $\perp$ Readout Discreteness ($\gamma \to \infty$ WTA limit delivers lossless symbols while flip rates stay invariant at $0.4445$).
- **Orthogonal Axis 2**: Binding Fidelity $\perp$ Combiner Error (conditioned on correct binding, algebraic combiner error is identically $0.0\mathrm{e}0$).
- **Proposition 5.3 (*1D Metric Sufficiency*)**: Proves 1D distance compatibility $k(q, k_i) = -|q - k_i|$ achieves $100.00\%$ routing identical to 2D vector rotation under hard selection.
- **Proposition 5.4 (*Intermediate Addressing Impossibility*)**: Demonstrates the $0/6{,}705$ ($0.00\%$) intermediate key addressing barrier under monotonic scalar dot products.
- **Phase P1-B Carrier Audit**: Confirmatory audit across 96,000 episodes under Protocol V2.6 proves analytical subthreshold collisions (residual $< 10^{-10}$), baseline floor failure ($2.75\%$--$2.90\% < 80\%/40\%$, *memory redundancy unestablished*), and single-unit spiking channel breakdown ($E_{\text{comp}} > 2.2$).

---

## ⚡ Reproduction & Verification

### Prerequisites
- Python 3.10+ (`pip install numpy scipy matplotlib pandas pypdf pytest`)
- TeX distribution (MiKTeX or TeXLive) for LaTeX compilation

### Running Stage 1 Original Experiments
```bash
# Execute full Stage 1–5 pipeline
bash snapshot/scripts/run_all_experiments.sh
# or on Windows:
snapshot\scripts\run_all_experiments.bat
```

### Running Automated QA Validators
```bash
# Verify Paper 1 (Strictly 28 pages, 0 ??, 13/13 required checks)
python snapshot/paper/scripts/qa_pdf.py

# Verify Paper 2 (Strictly 28 pages, 0 ??, references [1]-[13] verified)
python snapshot/paper2/scripts/qa_paper2.py

# Verify bidirectional cryptographic hash match against baseline archive
python scripts/generate_manifest_after.py
```

### Running Phase P1-B Confirmatory Suite
```bash
python -m pytest p1_experiments/p1b/test_p1b_g1_fixtures.py
```

---

## 📄 License & Attribution

All models, experimental scripts, datasets, and manuscript sources are authored by **Li Zexu** (School of Physics and Astronomy, University of Leeds, 2026).  
Released for academic review and open scientific replication under the **MIT License**.
