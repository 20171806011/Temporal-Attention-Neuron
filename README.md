# Temporal Attention Neuron (TAN)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Status: Final Revised Manuscript](https://img.shields.io/badge/Status-Final%20Revised%20Manuscript-success.svg)]()

This repository contains the complete, authoritative research suite, experimental simulation code, cryptographic audit manifests, and publication-ready LaTeX manuscripts for the **Temporal Attention Neuron (TAN)** research programme by Li Zexu (School of Physics and Astronomy, University of Leeds).

---

## 📚 Publications & Manuscripts

The repository includes both companion manuscripts, compiled and strictly verified at **28 pages each**:

1. **Paper 1: TAN-I Mechanistic Analysis**  
   *The Computational Boundary of Scalar Temporal Attention: A Mechanistic Analysis of the Temporal Attention Neuron*  
   - PDF: [`release/TAN_I_Mechanistic_Analysis.pdf`](release/TAN_I_Mechanistic_Analysis.pdf)  
   - LaTeX Source: [`snapshot/paper/`](snapshot/paper/)  
   - Core Theme: Dynamical boundaries of scalar temporal attention, surprise-weighted filtering, and non-Markovian memory states.

2. **Paper 2: TAN-II Minimal Architectural Ladder**  
   *The Minimal Architectural Ladder for Relational Computation: From Temporal Saliency to Binding and Composition in a Dynamical Neuron Model*  
   - PDF: [`release/TAN_II_Minimal_Architectural_Ladder.pdf`](release/TAN_II_Minimal_Architectural_Ladder.pdf)  
   - LaTeX Source: [`snapshot/paper2/`](snapshot/paper2/)  
   - Core Theme: The constructive ascent from saliency to content routing, semantic binding, and algebraic composition, featuring the Phase P1-A and Phase P1-B breakthrough audits.

---

## 🔬 Key Experimental Phases & Scientific Findings

### Phase P1-A: Content Address Identifiability & 1D Metric Compatibility
- **Scope**: 60,000 Monte Carlo trials across 7 model variants (420,000 total model evaluations).
- **Key Breakthrough**:
  - **Proposition 5.3 (*1D Metric Sufficiency*)**: Proves that a 1D scalar key endowed with distance compatibility $k(q, k_i) = -|q - k_i|$ attains $100.00\%$ routing partitions identical to 2D rotational vector attention, disproving that multidimensional vector spaces are strictly required for content addressing.
  - **Proposition 5.4 (*Impossibility of Scalar Intermediate Addressing*)**: Demonstrates the intermediate-key addressing barrier ($0 / 6{,}705$ hits, $0.00\%$) under monotonic scalar dot products.

### Phase P1-B: Continuous Dynamical Carrier & Spiking Readout Audit
- **Scope**: 96,000 model-episodes (192,000 role queries) under pre-registered Protocol V2.6.
- **Key Breakthrough**:
  - **Analytical Collision Theorem**: Subthreshold inputs ($u(t) < \theta_A$) undergo deterministic state collisions (residual $< 10^{-10}$), proving history erasure before query arrival.
  - **Prerequisite Floor Evaluation**: The baseline dynamical carrier (M1) achieved joint delivery accuracy of only $2.75\%$--$2.90\% < 80\%/40\%$, classifying all conditions as `FLOOR_LIMITED_COMPARISON` and establishing that *memory redundancy is not established*.
  - **Spiking Readout Channel Limitation**: Single-unit LIF decoders exhibit severe transmission breakdown ($E_{\text{comp}} \in [2.21, 2.57]$, $82.1\%$--$83.0\%$ abstention), certifying `DEMONSTRATED_CHANNEL_LIMITATION`.

---

## 📁 Repository Structure

```
Temporal-Attention-Neuron/
├── README.md                     # [This file] Project overview and quick start
├── release/                      # Final publication PDFs and release manifest
│   ├── TAN_I_Mechanistic_Analysis.pdf
│   ├── TAN_II_Minimal_Architectural_Ladder.pdf
│   └── FINAL_MANUSCRIPT_RELEASE_MANIFEST.md
├── p1_experiments/               # Phase P1-A & Phase P1-B experimental suites
│   ├── p1a/                      # P1-A models, runner, and 60k evaluation data
│   └── p1b/                      # P1-B Protocol V2.6, Euler ODE models, and G1 fixtures
├── snapshot/                     # Manuscript source trees
│   ├── paper/                    # Paper 1 LaTeX source and figures
│   └── paper2/                   # Paper 2 LaTeX source and figures
├── scripts/                      # Independent audit, QA, and manifest verification scripts
│   ├── qa_pdf.py                 # Paper 1 automated QA validator
│   ├── qa_paper2.py              # Paper 2 automated QA validator
│   └── generate_manifest_after.py# Bidirectional cryptographic hash verifier
├── logs/                         # Sprint execution and audit logs
├── MASTER_CLAIM_LEDGER.csv       # Master claim ledger across all rungs
├── CURRENT_RESEARCH_STATE.md     # Full state audit and boundary matrix
└── RESEARCH_CHRONOLOGY.md        # Experimental chronology and amendments AM1–AM6
```

---

## ⚡ Reproduction & Verification

### Prerequisites
- Python 3.10+ (NumPy, SciPy, Matplotlib, pandas, pypdf)
- LaTeX environment (MiKTeX or TeXLive) for compiling manuscripts

### Running Automated QA Checks
```bash
# Verify Paper 1 (28 pages, required phrases, zero ??)
python snapshot/paper/scripts/qa_pdf.py

# Verify Paper 2 (28 pages, required phrases, zero ??)
python snapshot/paper2/scripts/qa_paper2.py

# Verify bidirectional cryptographic integrity against frozen baseline
python scripts/generate_manifest_after.py
```

### Running Phase P1-B Confirmatory Fixtures
```bash
python -m pytest p1_experiments/p1b/test_p1b_g1_fixtures.py
```

---

## 📄 License & Attribution

All models, experimental scripts, and manuscript sources in this repository are authored by **Li Zexu** (School of Physics and Astronomy, University of Leeds, 2026).  
Released for academic review and open scientific replication.
