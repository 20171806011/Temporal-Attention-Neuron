# TAN Research Chronology and Evidence Audit

**Review Baseline:** P0 Reworked Baseline (2026-09-17)  
**Review Directory:** C:\Users\李则徐\Downloads\TAN_Review\P0_20260917_181757  
**Status:** AUDITED AND RECONSTRUCTED

---

## 1. Overview of Timeline Reconstruction

The timeline of the Temporal Attention Neuron (TAN) project has been reconstructed by cross-referencing three independent sources:
1. **Operating system filesystem metadata** (mtime / ctime).
2. **Session transcript records** (25 session rounds from .dsh/sessions/ and JSONL logs).
3. **Internal document self-dated headers** and machine-readable execution logs.

### Key Chronological Clarifications

#### The Sept 4 vs Sept 8–9 Header Discrepancy
Multiple TAN-II documents and preregistrations (e.g., TANII_SPRINT4_2_PREREGISTRATION.md, TANII_SPRINT4_2B_PREREGISTRATION.md) contain self-dated headers of 2026-09-04. However, session transcripts and filesystem metadata show that:
- On **2026-09-04**, active development and runs were concentrated on **TAN-I** (Phase 1 natural collisions with membrane leak $\lambda=0.5$, Phase 2 response geometry, Probe-1 distractor analysis, and Probe-2 identifiability termination).
- Active execution of **TAN-II** (Sprints 4.1 to 4.3-A) commenced on **2026-09-08** and concluded on **2026-09-09**.
- **Assessment:** Template copying from the TAN-I milestone header is a plausible technical hypothesis explaining the 2026-09-04 datestamp across TAN-II documents. However, this is documented strictly as a plausible inference rather than an asserted absolute fact. The physical execution sequence is established by machine logs.

#### Sourced vs Unsourced Timestamps
Specific clock times without direct log or filesystem backing are explicitly designated as UNKNOWN rather than conjectured.

---

## 2. Granular Multi-Phase Chronology

| Phase / Sprint | Conceptual / Prereg Date | Deterministic Dev / Debug | Formal Monte Carlo Run | Results Log Timestamp | Amendments & Key Parameters | Manuscript Artifact |
|---|---|---|---|---|---|---|
| **Early Behavioral (Stage 2)** | Pre-Aug 2026 | Aug 2026 | Aug 16, 2026 | 2026-08-16 | Batch scripts, uncalibrated LIF | manuscript/ (39 pp PDF) |
| **TAN-I Phase 1 (Collisions)** | 2026-09-04 | 2026-09-04 | 2026-09-04 | UNKNOWN | Membrane leak $\lambda=0.5$ (`natural_collision.py:95`), $W=5$, 160k histories | paper/ Section 3 |
| **TAN-I Phase 2 (Geometry)** | 2026-09-04 | 2026-09-04 | 2026-09-04 | UNKNOWN | MIN_LOGT_DIFF = 0.2 (`effective_dimension.py:112`); outcome D branch gap | paper/ Section 4 |
| **TAN-I Probe-1 (Distractor)** | 2026-09-04 | 2026-09-04 | 2026-09-04 | UNKNOWN | Derived test sizes: 6,000 total test, 2,997 distractor ($N_{\text{test}}=2000$, 3 seeds) | paper/ Section 5 |
| **TAN-I Probe-2 (Identifiability)**| 2026-09-04 | 2026-09-04 | 2026-09-04 | UNKNOWN | TERMINATED / AUDIT_FAILED (0 argmax flips, Theorem 5.1) | paper/ Section 6 |
| **TAN-I Paper Compilation** | 2026-09-04 | 2026-09-04 | N/A | 2026-09-04 (filesystem) | 28 pp PDF compiled (`paper/main.pdf`) | paper/main.tex |
| **TAN-II Sprint 4.1-C/D (Opponent)**| 2026-09-08 | 2026-09-08 | 2026-09-08 | UNKNOWN | 6 opponent layouts tested, 0 flips | paper2/sections/s4_opponent.tex |
| **TAN-II Sprint 4.2-A (Geometry)** | 2026-09-09 | 2026-09-09 | 2026-09-09 | UNKNOWN | **AM1:** E4(iii) theory failure retained ($d=2, \theta=\pi/6$, JSD $> 5\times 10^{-4}$ vs 0.000454) | paper2/sections/s5_routing_geometry.tex |
| **TAN-II Sprint 4.2-B (Vector QK)**| 2026-09-09 | 2026-09-09 | 2026-09-09 | UNKNOWN | **AM2:** Unnormalized key control (FlipRate = 0 exact); **AM3:** Corrected fixed-point constants | paper2/sections/s5_routing_geometry.tex |
| **TAN-II Sprint 4.2-C (Soft Binding)**| 2026-09-09 | 2026-09-09 | 2026-09-09 | UNKNOWN | **AM4:** Pre-run correction of M5-E drive transfer ($T_{\text{drive}}$ from $+0.0985$ to $+0.236700$) | paper2/sections/s6_binding.tex |
| **TAN-II Sprint 4.2-D (WTA Limit)** | 2026-09-09 | 2026-09-09 | 2026-09-09 | UNKNOWN | **AM5:** Pre-run correction of $D_{\text{full}}$ non-monotonicity & cancellation root at $\gamma \approx 1.198$; $\gamma^* = 5.098015$ E-I crossing root | paper2/sections/s7_wta.tex |
| **TAN-II Sprint 4.3-A (Composition)**| 2026-09-09 | 2026-09-09 | 2026-09-09 | UNKNOWN | **AM6:** Runtime correction of rounding error in canonical margins ($0.210631 / 0.261879$ vs $0.2118 / 0.2637$) | paper2/sections/s8_composition.tex |
| **TAN-II Paper Compilation** | 2026-09-09 | 2026-09-09 | N/A | 2026-09-09 (filesystem) | 25 pp PDF compiled (original draft; revised copy 27 pp) | paper2/main.tex |
| **Comprehensive Audit** | 2026-09-17 | Read-only scan | None | 2026-09-17 17:42 | Handover task initialized | Handover documentation |
| **P0 Review Execution** | 2026-09-17 | Python checkers | Isolated Replay Suite | 2026-09-17 18:21 | Initial delivery (marked P0_REWORK_REQUIRED) | Initial audit artifacts |
| **P0 Rework Execution** | 2026-09-17 | Corrected checkers | 11 Replay JSON Diffs | 2026-09-17 (current) | All 8 audit points resolved, bidirectionally verified | Reworked deliverables |

---

## 3. Accurate Meanings and Categories of Amendments AM1–AM6

The six historical amendments of the TAN-II research programme are categorized by their exact scientific meanings:

1. **AM1 (Sprint 4.2-A / `sprint4_2_theory_audit.py`):**
   - *Meaning:* E4(iii) analytical tolerance failure retained. The test checked whether JSD between analytic and simulated query angles exceeded $5 \times 10^{-4}$ at $d=2, \theta=\pi/6$. The numerical value was $0.000454$ (which failed the strict threshold of $> 5 \times 10^{-4}$). Retained verbatim in code as an intentional failure documenting protocol adherence.
   - *Phase:* Deterministic theory audit.
2. **AM2 (Sprint 4.2-B / `sprint4_2_vector_qk.py`):**
   - *Meaning:* Unnormalized key control ($k_i = \varphi(x_i)$ without $S^1$ sphere projection). Disproved the hypothesis that vector representation alone suffices for routing; proved that equal-norm normalization is an indispensable mechanistic enabler (unnormalized keys yield $\mathrm{FlipRate} = 0.0$ exact).
   - *Phase:* Pre-run control formulation.
3. **AM3 (Sprint 4.2-B / `sprint4_2b_summary.json`):**
   - *Meaning:* Runtime correction of fixed-point constants for M4/M5/M2 conditions, formed after initial canonical audit and before formal Monte Carlo execution.
   - *Phase:* Pre-Monte Carlo freeze.
4. **AM4 (Sprint 4.2-C / `sprint4_2_binding_probe.py`):**
   - *Meaning:* Pre-run correction of the M5-E drive-transfer numerical prediction from $+0.0985$ to $+0.236700$. Discovered that the inhibitory branch re-reads the value channel and subtracts it, reversing net value transfer from negative to positive at $\gamma=1$.
   - *Phase:* Pre-run analytic correction.
5. **AM5 (Sprint 4.2-D / `sprint4_2_hard_binding.py` & `sprint4_3a_composition_probe.py`):**
   - *Meaning:* Pre-run correction documenting that full-window softness $D_{\text{full}}(Q=3)$ is non-monotonic across $\gamma$ and crosses $0$ at $\gamma \approx 1.198$. Identified this as a background dilution cancellation artifact (where silent slots compensate for loser weight), while event-normalized softness $D_{\text{ev}} \approx 1.5 > 0$ remains strictly positive. Later extended in 4.3-A as the CF5 zero-value trap test ($V_{\text{target}}=0$).
   - *Phase:* Pre-run geometric and metric correction.
6. **AM6 (Sprint 4.3-A / `sprint4_3a_composition_probe.py`):**
   - *Meaning:* Runtime correction of four-decimal rounding error in canonical routing margins. Replaced hand-preregistered values ($m_A=0.2118, m_B=0.2637$) with exact closed-form values ($m_A=0.210631046, m_B=0.261878523$), logging deviations of $0.00117$ and $0.00184$ without tuning models, seeds, or thresholds.
   - *Phase:* Runtime arithmetic calibration.

---

## 4. Verification and Integrity Invariants

- **Original Archive Invariant:** All 368 non-cache files in `C:\Users\李则徐\Downloads\TAN_Final_Submission\TAN_Final_Submission` are 100% bit-for-bit identical to baseline, verified bidirectionally (0 missing, 0 untracked, 0 modified).
- **Historical Script SHA-256 Hashes:** All 9 historical script hashes documented in summary files and reproducibility ledgers match their corresponding code files bit-for-bit (9/9 matches verified in `manifest_before.json` and documented in `manifest_before_correction_provenance.json`).
- **Replay JSON Invariant:** All 11 replay JSON output files in `P0_20260917_181757` match their frozen original JSON counterparts bit-for-bit and numerically (11/11 matches verified in `replay_diff_report.json`).
