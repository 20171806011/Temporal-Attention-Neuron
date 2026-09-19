# Final Scientific Manuscript Release Manifest

**Release Environment**: `C:\Users\李则徐\Downloads\TAN_Review\P0_20260917_181757\release`  
**Date**: September 19, 2026  
**Status**: Publication-Ready, Fully Audited, Visually Verified  

---

## 1. Release Deliverables

### Paper 1: TAN-I Mechanistic Analysis
- **File**: `TAN_I_Mechanistic_Analysis.pdf`
- **Full Path**: `C:\Users\李则徐\Downloads\TAN_Review\P0_20260917_181757\release\TAN_I_Mechanistic_Analysis.pdf`
- **Title**: *The Computational Boundary of Scalar Temporal Attention: A Mechanistic Analysis of the Temporal Attention Neuron*
- **Author**: Li Zexu (School of Physics and Astronomy, University of Leeds)
- **Date**: Final Revised Manuscript (September 2026)
- **Pages**: Exactly 28 pages
- **Size**: 879,480 bytes
- **SHA-256**: `73735523503C1115CA51074AAC35B87485B79730232714475D712BC5BBB354C0`
- **Source LaTeX Root**: `C:\Users\李则徐\Downloads\TAN_Review\P0_20260917_181757\snapshot\paper`
- **QA Verification**:
  - `python paper/scripts/qa_pdf.py` -> **PASS** (13/13 required checks FOUND)
  - Literal `??` count: 0
  - Banned phrases scan: 6/6 ABSENT
  - Bibliography: 58 references, clean layout, zero overfull hboxes

### Paper 2: TAN-II Minimal Architectural Ladder
- **File**: `TAN_II_Minimal_Architectural_Ladder.pdf`
- **Full Path**: `C:\Users\李则徐\Downloads\TAN_Review\P0_20260917_181757\release\TAN_II_Minimal_Architectural_Ladder.pdf`
- **Title**: *The Minimal Architectural Ladder for Relational Computation: From Temporal Saliency to Binding and Composition in a Dynamical Neuron Model*
- **Author**: Li Zexu (School of Physics and Astronomy, University of Leeds)
- **Date**: Final Revised Manuscript (September 2026)
- **Pages**: Exactly 28 pages
- **Size**: 821,088 bytes
- **SHA-256**: `D9FAE55C34DE38805B8FE505CD4C8C934CD29D0B3DD9119FB0B1516CB5A387F6`
- **Source LaTeX Root**: `C:\Users\李则徐\Downloads\TAN_Review\P0_20260917_181757\snapshot\paper2`
- **QA Verification**:
  - `python snapshot/paper2/scripts/qa_paper2.py` -> **PASS** (13/13 required checks FOUND)
  - Literal `??` count: 0
  - Bibliography: 13 references [1]–[13], perfectly fitted on page 28, zero orphan pages
  - Visual rendering inspection: Pages 1–28 inspected and confirmed publication quality

---

## 2. Scientific Content & P1-A / P1-B Breakthrough Integration in Paper 2

Paper 2 incorporates both certified P1-A and P1-B breakthrough results with full mathematical rigour:
1. **Mathematical Sufficiency of 1D Metric Compatibility (P1-A)**:
   - **Proposition 5.3** (*1D Metric Sufficiency under Hard Selection*): Formally establishes that a 1D scalar key endowed with distance compatibility $k(q, k_i) = -|q - k_i|$ attains identical discrete routing partitions ($\mathrm{FlipRate}_{\mathrm{CF}}(\Delta q) = \Delta q / \Delta_{\mathrm{span}}$) as 2D rotational vector attention, disproving that multidimensional vector spaces are strictly required for content addressing.
2. **Impossibility Boundary for Scalar Intermediate Keys (P1-A)**:
   - **Proposition 5.4** (*Impossibility of Scalar Intermediate Addressing*): Proves that monotonic dot products cannot address intermediate keys when $N \ge 3$.
   - **Empirical Confirmation**: 60,000 Monte Carlo trials across 7 model configurations (420,000 total model evaluations) demonstrate intermediate-key hit rate of **0 / 6,705 (0.00%)** under scalar dot product versus **100.00%** under 1D metric and 2D rotation.
3. **Publication Vector Figure (P1-A)**:
   - `figures/fig7_p1a_identifiability.pdf` embedded as Figure 4 with high-resolution vector layout (3 panels: hit rate vs candidate count, both-routed accuracy across candidate regimes C1–C6, and soft readout error bounds).
4. **Dynamical Carrier and Spiking Readout Audit (P1-B)**:
   - **Analytical Collision Theorem**: Subthreshold inputs ($u(t) < \theta_A$) undergo deterministic trajectory collapse (residual $< 10^{-10}$), erasing history prior to query arrival.
   - **Confirmatory Audit (96,000 episodes, 192,000 queries)**: Tested across 4,000 unique histories, 3 silence delays, and 8 models. Baseline carrier M1 fails prerequisite joint delivery accuracy floor ($2.75\%$--$2.90\% < 80\%/40\%$). Registered verdict: `FLOOR_LIMITED_COMPARISON`, with memory redundancy unestablished (`redundancy_inferred = False`).
   - **Spiking Readout Channel Breakdown**: Single-unit LIF decoders (rate and latency) fail severely ($E_{\mathrm{comp}} \in [2.21, 2.57]$, $82.1\%$--$83.0\%$ abstention), certifying `DEMONSTRATED_CHANNEL_LIMITATION`.
5. **Master Claim Ledger (Table 3)**:
   - Claim **C13** upgraded from `UNKNOWN` to `ESTABLISHED & BOUNDED` with exact statistics (60,000 trials, 420,000 evaluations).
   - Claim **C14** upgraded from `OPEN` to `BOUNDED / INSUFFICIENT` with exact confirmatory statistics (96,000 episodes, 192,000 queries, 6 cells at floor $<0.03$, spiking $E_{\mathrm{comp}}>2.2$).
6. **Abstract & Reproducibility Statement**:
   - Abstract updated with six principal results including Result (vi) for Phase P1-B.
   - Reproducibility statement updated with Protocol V2.6 dossier, 12-fixture G1 suite, and confirmatory output manifest hashes.

---

## 3. Preservation of Baseline Research Archive

- **Original Research Directory**: `C:\Users\李则徐\Downloads\TAN_Final_Submission\TAN_Final_Submission`
- **Bidirectional Verification Check**: `python scripts/generate_manifest_after.py`
  - Total controlled non-cache files: **368**
  - Missing files: **0**
  - Untracked new files: **0**
  - Modified files: **0**
  - Hash match rate: **100.00% bit-for-bit identical**

---

## 4. Quick Access & Transmission Instructions

To inspect, copy, or transmit the final manuscripts, use either the compiled PDF files directly:
```powershell
# Open Paper 1
Invoke-Item "C:\Users\李则徐\Downloads\TAN_Review\P0_20260917_181757\release\TAN_I_Mechanistic_Analysis.pdf"

# Open Paper 2
Invoke-Item "C:\Users\李则徐\Downloads\TAN_Review\P0_20260917_181757\release\TAN_II_Minimal_Architectural_Ladder.pdf"
```

Or recompile from LaTeX sources if needed:
```powershell
# Recompile Paper 1
cd "C:\Users\李则徐\Downloads\TAN_Review\P0_20260917_181757\snapshot\paper"
& "C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode main.tex

# Recompile Paper 2
cd "C:\Users\李则徐\Downloads\TAN_Review\P0_20260917_181757\snapshot\paper2"
& "C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe" -interaction=nonstopmode main.tex
```
