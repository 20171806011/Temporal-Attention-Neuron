# Comprehensive Validation Report: P0 Review of TAN Research

**Review Snapshot Date:** 2026-09-17  
**Review Directory:** `C:\Users\李则徐\Downloads\TAN_Review\P0_20260917_181757`  
**Execution Environment:** Python 3.12.7, NumPy 1.26.4, SciPy 1.13.1, Matplotlib 3.9.2, MiKTeX 24.1.0  
**Source Baseline:** `C:\Users\李则徐\Downloads\TAN_Final_Submission\TAN_Final_Submission` (368 files, 100% frozen)

---

## 1. Executive Verdict and Boundary Summary

The P0 phase has rigorously completed all mandate objectives:
1. Re-established the verifiable chronology and state ledger across TAN-I and TAN-II.
2. Independently diagnosed, numerically validated, and provided minimal errata for all blocker issues A01–A07.
3. Successfully replayed the entire TAN-II audit suite in an isolated snapshot without altering frozen history.
4. Corrected mathematical discrepancies in derived paper copies and recompiled the manuscripts.
5. Formally established that **existing composition results represent constructive operator-level sufficiency of a static retriever plus explicit arithmetic node**, rather than online neural dynamics or learned generalization.
6. Conducted literature positioning against foundational works (Charikar 2002, Martins & Astudillo 2016, Schlag et al. 2021, Ramsauer et al. 2020).
7. Flagged `VISUAL_QA_NOT_COMPLETED` and determined that manuscripts are **NOT READY FOR SUBMISSION**.

---

## 2. Independent Numerical Verifications

All checks were executed via independent Python scripts reading directly from the isolated snapshot data. Results are serialized in `NUMERICAL_CHECKS.json`.

### 2.1 Issue A01: Query Rotation Double \omega Multiplication
- **Code implementation:** $\mathbf{u}(\theta) = (\cos\theta, \sin\theta)$, $\mathbf{Q} = c S_E \mathbf{u}(\omega S_E)$. Evaluates angle $\theta = \omega S_E$.
- **Literal paper text:** $\mathbf{u}(s) = (\cos\omega s, \sin\omega s)$, $\mathbf{Q} = c S_E \mathbf{u}(\omega S_E)$. Evaluates angle $\theta = \omega^2 S_E$.
- **Independent Evaluation on Canonical History `[1, 0, 2, 0, Q]` ($\omega=0.4, c=2.0$):**
  - **Q = 3.0 ($S_E = 1.8$):**
    - Code Margin ($E_A - E_B$): `+0.2587427243` -> Winner: **A**
    - Literal Margin: `+0.7055409799` -> Winner: **A**
  - **Q = 4.0 ($S_E = 2.6$):**
    - Code Margin ($E_A - E_B$): `-0.1559101334` -> Winner: **B** (Counterfactual **FLIP**)
    - Literal Margin: `+0.8425586282` -> Winner: **A** (NO FLIP)
- **Verdict:** The literal paper formula prevents the counterfactual winner flip at $Q=4$. The code implementation is correct and matches line 63 of the preregistration. The text in `paper2/sections/s2_framework.tex` has been corrected in the derived copy.

### 2.2 Issue A02: Phase 2 Decision Gate Schema Gap
- **Input data:** `results/tables/effective_dimension_summary.csv` (374 events, 3 seeds).
- **Independent Evaluation:**
  - `stable = True`: $dD_{\text{delta}} = 1.088918 \ge 0.5$, $CI_{\text{lo}} = 1.873931 > 0$.
  - `Condition A = False`: failed because $B4_{zC} (1.5891) < B1_{zC} (1.6740)$.
  - `Condition B = False`: failed because $B4_{zF} (1.0889) > B3_{zF} (0.4835)$.
  - `Condition C = False`: failed because $|B4_{zC} - C0_{zU}| = 0.534135 > 0.5$.
  - `Outcome = D`: reached by fall-through in line 984.
- **Log-Trace Metrics Delineation:**
  - Event log-trace difference ($B4_{zF} - B3_{zF}$): `1.094463` nats.
  - Event-minus-quiet difference-in-differences ($[B4_{zF} - B4_{\text{quiet}}] - [B3_{zF} - B3_{\text{quiet}}]$): `1.354674` nats.
- **Verdict:** Outcome D occurred due to a classification schema gap (`UNCLASSIFIED_MIXED_CASE`), not because expansion was absent. The two log-trace metrics are distinct, valid statistics.

### 2.3 Issue A03: Statistical Metrics, Delineations, and Sample Sizes
- **Item 1 (K Mean vs Median):**
  - LIF (B1): Mean = $0.5000$, Median = $0.5000$
  - Buffer (B2): Mean = $4469.5365$ [95% CI $3.26$--$5.92\times 10^3$], Median = $895.5296$
  - TAN-noAttn (B3): Mean = $12347.8860$ [95% CI $7.78$--$18.3\times 10^3$], Median = $1369.0795$
  - Full TAN (B4): Mean = $20465.8556$ [95% CI $1.12$--$3.57\times 10^4$], Median = $2947.7244$
  - Membrane leak parameter in Phase 1: $\lambda = 0.5$ (`natural_collision.py:95`).
  - Verdict: Paper 2 Section 3 mislabeled `K_mean` as median. Corrected to present both metrics.
- **Item 2 (JSD Values by Setting):**
  - 2-candidate geometric toy example: Scalar JSD = $0.0751$ (flip 0); Vector JSD = $0.0647$ (flip 1).
  - Canonical full-window model (5 slots): M0 JSD = $0.009462259$; M2 JSD = $0.008726529$.
  - Verdict: Contexts clearly separated in text.
- **Item 3 (E3 Maximum Deviation & Units):**
  - Total checks = 140 (132 isotropic conditions + 8 control checks).
  - Single-seed maximum absolute deviation: `0.009266667` across all 132 conditions.
  - Seed-averaged maximum absolute deviation: `0.004666667` across 44 conditions.
  - Machine-readable field `theta_deg`: stores $\theta/\pi$ (e.g. $0.0833, 0.1667$), not degree values.
- **Item 4 (Probe-1 Sample Sizes):**
  - Total test trials: $6,000$ ($2,000$ per seed $\times 3$ seeds).
  - Generative derivation from `temporal_distractor.py`: $1/3$ catch ($n_{\text{catch}} = 667$), $n_{\text{present}} = 1333$; distractor present on $666$ and distractor catch on $333 \implies 999$ distractor trials/seed $\times 3 = 2,997$ distractor trials total.
- **Item 5 (Sprint 4.3-A Composition Errors & Coverage):**
  - Total histories: $30,000$ ($10,000$ per seed $\times 3$ seeds).
  - Both-routed subset: $6,608$ histories ($22.03\%$ coverage: $2,188$ / $2,258$ / $2,162$ per seed).
  - Conditional composition error on both-routed subset: `0.000000`.
  - Numerical quadrature benchmarks (integration over continuous uniform density): $E_{\text{comp}}^+ = 1.116099621$, $E_{\text{comp}}^- = 2.000999500$.
  - Pooled 30,000 Monte Carlo empirical means: $E_{\text{comp}}^+ = 1.107671595$, $E_{\text{comp}}^- = 1.975892767$.
  - Verdict: Zero error is strictly conditional on the $22.03\%$ both-routed subset; quadrature benchmarks and MC means are explicitly differentiated.

### 2.4 Issue A05: Softmax Numerical Conventions
- Sprint 4.2-B code: $\alpha = \exp(\text{lg} - \max) / \sum \exp(\text{lg} - \max)$, sum = `1.000000000000` exactly.
- Base `tan.py`: $\alpha = \exp(\beta(\text{lg} - \max)) / (\sum \exp + 10^{-9})$, sum = `0.999999999027`.
- Paper text: did not include $\max$ subtraction and asserted universal inheritance of $\delta=10^{-9}$.
- Numerical difference on non-overflow logits: $9.46\times 10^{-10}$. On large logits ($\ge 700$), max-subtraction prevents NaN overflow.
- Verdict: Paper text corrected to describe actual sprint implementations accurately.

### 2.5 Carrier Analysis of Sprint 4.3-A Composition Probe
- AST and data-flow analysis of `run_history` in `sprint4_3a_composition_probe.py`:
  - Step-by-step membrane potential update ($h$): **NOT PRESENT**.
  - Spike / reset integration: **NOT PRESENT**.
  - Arithmetic execution: **DIRECT EXPLICIT NODE** (`C1 + C2` / `C1 - C2`).
  - AST analysis via `check_composition_carrier.py` parses the AST tree and identifies all 5 `True` keyword arguments in `shortcut = dict(...)` (lines 92, 107, 137, 153, 179) plus 1 direct positional `True` call at line 198 (`check("...", True)`).
- Verdict: Proves constructive sufficiency of static retriever + explicit arithmetic node under controlled conditions. Does NOT demonstrate online neural dynamics or learned generalization.

---

## 3. Program Execution Checks & Bit-for-Bit Replay Suite

The 9 TAN-II scripts were executed in the isolated review directory `snapshot/` with explicit `--outdir` flags and `PYTHONDONTWRITEBYTECODE=1`:

| Job Name | Script Path | Expected Exit | Actual Exit | Runtime (s) | Status / Verdict |
|---|---|---|---|---|---|
| **s41cd** | `sprint4_1cd_ei_dynamics.py` | 0 | 0 | 13.30s | PASS: C1 non-monotonic peak verified; 0 flips in discrete dynamics |
| **s42a** | `code/experiments/sprint4_2/sprint4_2_geometry.py` | 0 | 0 | 7.58s | PASS: 316 PASS, 1 PROTOCOL-FAIL / NUMERICAL-PASS (E4-iii) |
| **s42_shortcuts** | `code/experiments/sprint4_2/sprint4_2_shortcut_audit.py` | 0 | 0 | 1.92s | PASS: 36/36 shortcut checks PASS |
| **s42b** | `code/experiments/sprint4_2/sprint4_2_vector_qk.py` | 0 | 0 | 2.30s | PASS: 27/27 canonical checks PASS; Q=3/4 flips verified |
| **s42b_probe** | `code/experiments/sprint4_2/sprint4_2_routing_probe.py` | 0 | 0 | 22.30s | PASS: 48/48 routing checks PASS |
| **s42c** | `code/experiments/sprint4_2/sprint4_2_binding_probe.py` | 0 | 0 | 3.77s | PASS: 42/42 binding checks PASS; soft blending verified |
| **s42d** | `code/experiments/sprint4_2/sprint4_2_hard_binding.py` | 0 | 0 | 18.14s | PASS: 77/77 hard binding checks PASS; $\gamma^* = 5.098015$ verified |
| **s43a** | `code/experiments/sprint4_3/sprint4_3a_composition_probe.py` | 0 | 0 | 6.50s | PASS: 79/79 composition checks PASS; coverage recorded |
| **s42_theory** | `code/experiments/sprint4_2/sprint4_2_theory_audit.py` | 1 | 1 | 6.43s | PASS: Exact expected exit=1; 1 failure: `[FAIL] E4(iii)` |

### Replay Output Byte-Level Integrity Verification
The script `run_audit_replay.py` systematically compared all 11 replay-generated JSON files against their corresponding original JSON files in `snapshot/results/`.
Results recorded in `replay_diff_report.json`:
- Total files compared: 11
- Total identical files: 11 (100% bit-for-bit identical, 0 byte differences)
- Total mismatched files: 0

### Error Fixture Verification
All independent check scripts (`check_a01_query_rotation.py`, `check_a02_phase2_branch.py`, `check_a03_statistics.py`, `check_a05_softmax.py`, `check_composition_carrier.py`, `run_audit_replay.py`) include dedicated `test_*_error_fixture()` routines that inject deliberate perturbations (e.g. faulty parameters, corrupted JSON, missing keywords) to verify that regressions are immediately caught. All 6 fixtures pass 100%.

---

## 4. Literature Matrix and Contribution Positioning

| Source Reference | Known Prior Theoretical Finding | Project Application / Replication | Mechanistic Facts Discovered in TAN | Unproven Conjectures / Prohibited Over-Claims |
|---|---|---|---|---|
| **Charikar (STOC 2002)** | Random hyperplane rounding produces collision probability $1 - \theta/\pi$ and flip probability $\theta/\pi$. | Applied as the theoretical flip law for 2D rotating queries under normalized moment keys. | Proved that the scalar TAN kernel is restricted to flip rate $0$, while 2D vector QK restores continuous $\theta/\pi$ flips. | Prohibits claiming $\theta/\pi$ is a novel mathematical law or that 2D is universally necessary for attention. |
| **Martins & Astudillo (ICML 2016)** | Softmax has full positive support over all finite logits; Sparsemax projects onto simplex to achieve exact sparse selections. | Explains why finite softmax cannot achieve exact one-hot binding without infinite gain ($\gamma \to \infty$). | Formalized the soft blending obstacle $D = |V_A - V_B|/(1 + e^{\gamma m})$ and established required gain $\gamma^*$. | Prohibits claiming that all finite selection mechanisms are incapable of exact binding. |
| **Schlag et al. (ICML 2021)** | Linear attention mechanisms are mathematically equivalent to fast weight memory programmers ($W_t = W_{t-1} + v_t k_t^\top$). | Connects windowed attention weighting to dynamic memory accumulation. | Showed that in TAN, memory resides in the window buffer, not exclusively in attention gating. | Prohibits conflating linear fast-weight updates with nonlinear Boltzmann attention. |
| **Ramsauer et al. (ICLR 2021)** | Continuous Hopfield networks with modern energy functions correspond to softmax attention with exponential storage capacity. | Connects TAN temporal attention retrieval to energy-based associative memory. | Demonstrated that single-winner associative retrieval requires key-value decoupling for value delivery. | Prohibits claiming TAN proves biological Hopfield memory without dynamical neuron-level proof. |

---

## 5. LaTeX Compilation and Manuscript Status

- **Paper 1 (`paper/`):** 28 pages. Compiles cleanly with `pdflatex` (exit code 0). Header updated to Draft v0.2.
- **Paper 2 (`paper2/`):** 26 pages. Compiles cleanly with `pdflatex` + `bibtex` (exit code 0). All citations (`charikar2002`, `martins2016`, `schlag2021`, `ramsauer2020`) resolved. Draft header updated.
- **Visual QA:** `VISUAL_QA_NOT_COMPLETED` due to local rendering inspection constraints.
- **Final Submission Readiness Verdict:** **NOT READY FOR SUBMISSION**.
  - Author correspondence contains unresolved placeholders.
  - Full-page visual typesetting and overflow QA has not been manually verified.
  - Manuscripts are certified as rigorously audited internal research technical baselines.

