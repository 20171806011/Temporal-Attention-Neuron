# Temporal Attention Neuron (TAN) Current Research State

**Audit & Review Snapshot Date:** 2026-09-17  
**Review Directory:** C:\\Users\\李则徐\\Downloads\\TAN_Review\\P0_20260917_181757  
**Source Archive:** C:\\Users\\李则徐\\Downloads\\TAN_Final_Submission\\TAN_Final_Submission (368 files, 100% frozen)

---

## 1. Executive Summary

This document establishes the verified baseline of the Temporal Attention Neuron (TAN) project. It distinguishes three independent entities:
1. **File existence and historical records**: What artifacts and logs exist in the repository.
2. **Experimental executions**: What was actually run, under what parameters, and what numerical outcomes were produced.
3. **Scientific claims**: What theoretical assertions are logically and empirically supported by the evidence, versus what claims exceed the evidence and must be narrowed, qualified, or withdrawn.

**Core Scientific Findings:**
- **TAN-I Mechanistic Baseline:** The 1D scalar membrane potential $h_t$ is not a state-sufficient representation for TAN (natural collision pairs with identical $h_t$ diverge under identical future inputs, $K_{\text{mean}} = 20465.86, K_{\text{median}} = 2947.72$ for B4, with membrane leak parameter $\lambda=0.5$ in `natural_collision.py:95`). Memory is provided by the temporal window buffer ($W=5$), rather than being an exclusive emergent property of attention. The positive scalar attention kernel failed the query-dependent routing audit (Probe-2: `TERMINATED / AUDIT_FAILED`; flip fraction 0.000, JSD 0.0002, permutation $p = 0.9201$); the general order-conservation result is Paper 2, Theorem 5.1.
- **TAN-II Architectural Ladder:** Opponent E-I coupling introduces non-monotonic tuning peaks without breaking scalar rank-order constraints (Sprint 4.1). A 2D vector query-key representation enables counterfactual winner flips: isotropic pairs follow the linear angular law $P(\text{flip}) = \theta / \pi$ on $S^1$ (Sprint 4.2-A), and the Sprint 4.2-B ensemble FlipRate is $0.4425$ vs the quadrature benchmark $0.4445$. Decoupled key-value pairs allow continuous value transmission via event-normalized readouts ($T = -0.4129, T_{\text{swap}} = +0.4129$), subject to the softmax mixing barrier ($D_{\text{ev}} = 1.7427 / 1.8444 > 0$, Sprint 4.2-C). Winner-take-all (WTA) argmax selection achieves exact symbolic delivery ($D=0$) in the $\gamma \to \infty$ limit; E-I net drive value transfer is reversed at low $\gamma$ ($T_{\text{drive}} = +0.2367$ at $\gamma = 1$) and returns to the value-aligned (negative-$T_{\text{drive}}$) direction above the bisection root $\gamma^* = 5.098015$ where $T_{\text{drive}} = 0$ (Sprint 4.2-D).
- **Current Front-line Boundary:** The composition result in Sprint 4.3-A is a **constructive proof of operator sufficiency** combining a static retriever with an explicit algebraic node ($C_1 + C_2$ or $C_1 - C_2$). It does not establish online continuous neural membrane dynamics, spike-based computation, or learned generalization. On the both-routed subset ($6{,}608 / 30{,}000 = 22.03\%$ coverage; quadrature benchmark $22.07\%$), composition error is identically zero ($E_{\text{comp}} = 0.0\mathrm{e}0$). The unconditional composition error is $E[E_{\text{comp}}^+] = 1.1077$ (quadrature benchmark $1.1161$) and $E[E_{\text{comp}}^-] = 1.9759$ (quadrature benchmark $2.0010$).
- **Manuscripts Status:** Both `paper/` (TAN-I, 28 pages) and `paper2/` (TAN-II, 28 pages) compile cleanly with zero errors. However, neither is submission-ready: visual rendering QA of figures cannot be verified autonomously in a headless terminal, and author correspondence metadata is pending.

---

## 2. Inventory and Structural Hierarchy

| Artifact Category | Primary Location | Scope / Nature | Status / Cautions |
|---|---|---|---|
| **Early Behavioral Archive** | `manuscript/` (39 pp), `results/` (early) | 6 behavioral experiments (habituation, noise, phototaxis) | Pre-dates mechanistic framework; uncalibrated LIF baselines. Do not confuse with current mechanistic papers. |
| **TAN-I Mechanistic Archive** | `paper/` (28 pp), `code/experiments/`, `audit_v3_amended/` | Phase 1 (collisions), Phase 2 (geometry), Probe-1 (distractor), Probe-2 (identifiability) | Frozen archive. Probe-2 permanently closed as negative boundary (cf. Paper 2, Theorem 5.1). |
| **TAN-II Sprint 4.1-C/D** | `results/sprint4_1cd/`, `sprint4_1cd_ei_dynamics.py` | E-I opponent dynamics ladder (C1–C4) | Confirmed non-monotonic tuning; 0 flip rate (does not break scalar sorting). |
| **TAN-II Sprint 4.2-A/B/C/D** | `results/sprint4_2/`, `code/experiments/sprint4_2/` | Vector QK geometry, counterfactual flips, KV decoupling, WTA limit | Replayed and verified. Double $\omega$ typo in text corrected in derived copy; code preserved. |
| **TAN-II Sprint 4.3-A** | `results/sprint4_3/`, `code/experiments/sprint4_3/` | Dual-channel binding + arithmetic combiner | 79 checks confirmed; static retriever + arithmetic node. Both-routed coverage $22.03\%$. |
| **Audit & Plan Snapshot** | `docs/RESEARCH_AUDIT_AND_PLAN_2026-09-17.md` | Comprehensive handover audit document | Baseline document for P0 execution. |

---

## 3. Strict Boundary Matrix of Existing Experiments

| Stage / Experiment | Verified Direct Evidence | Legitimate Scientific Claim | Prohibited Over-Claim |
|---|---|---|---|
| **Phase 1 (Natural Collision)** | 160,000 histories; 1,500 collision pairs/model; 843 common-reset subset; leak $\lambda=0.5$ | Membrane potential $h_t$ alone is an insufficient state descriptor; window provides history dependence. | Claiming all Markov systems contract, or that attention uniquely confers memory. |
| **Phase 2 (Effective Dimension)** | 3 seeds, 374 events, local response covariance; stable=True, outcome=D; `MIN_LOGT_DIFF = 0.2` | Local response covariance changes in specific coordinates; outcome D reflects decision tree classification gap. | Claiming higher intrinsic dimension than LIF or universal computational advantage. |
| **Probe-1 (Distractor Retrieval)** | 6,000 test trials (2,997 distractor trials derived from `temporal_distractor.py`); B2 acc $\approx 0.843$, B4 $\approx 0.389$ | Conditional advantage over compromised B3 baseline; B2 delay line is a strong positional baseline. | Claiming universal interference robustness or memory superiority. |
| **Probe-2 (Identifiability Audit)** | 0 argmax flips; JSD $= 0.0002$ (permutation $p = 0.9201$); `TERMINATED / AUDIT_FAILED` | Positive scalar kernel is monotonically constrained by amplitude; cannot perform content routing. | Treating Probe-2 as an incomplete experiment awaiting hyperparameter tuning. |
| **Sprint 4.1-C/D (Opponent)** | Non-monotonic peak in C1; C2–C4 no qualified peaks; 0 flips | Static opponent structure produces non-monotonic tuning; does not break kernel sorting barrier. | Claiming opponent competition achieves content routing. |
| **Sprint 4.2-A (Geometry)** | 140 checks (132 isotropic conditions + 8 controls); $\theta/\pi$ law; max single-seed diff 0.0093, max seed-averaged diff 0.0047 | Dot-product angular flip follows $\theta/\pi$ on $S^1$; minimal 2D sufficiency verified. | Claiming 2D is necessary across all attention families (e.g. scalar metric attention). |
| **Sprint 4.2-B (Vector QK)** | Energy margins $+0.2587 \to -0.1559$; ensemble flip $0.4425$ vs $0.4445$ | Normalized vector QK generates counterfactual winner flips ($Q=3 \to A, Q=4 \to B$). | Conflating winner flip rate with semantic address correctness. |
| **Sprint 4.2-C (Soft Binding)** | Continuous context $C_{\text{ev}} = 6.7427 / 7.1556$; soft error $D_{\text{ev}} = 1.7427 / 1.8444$ | KV decoupling enables continuous value transmission via event-normalized readout ($T_{\text{swap}} = -T$). | Claiming exact value binding under finite softmax gain. |
| **Sprint 4.2-D (WTA Limit)** | $\gamma \to \infty$ yields exact delivery ($D=0$); E-I direction reversal crossing root $\gamma^* = 5.098015$ ($T_{\text{drive}}=0$) | One-hot selection achieves exact value binding conditioned on correct upstream routing; WTA restores E-I direction. | Claiming hard selection cures upstream mis-routing errors or that $\gamma^*$ is an $\varepsilon=10^{-2}$ precision band. |
| **Sprint 4.3-A (Composition)** | 79 checks PASS; $E_{\text{comp}} \mid \text{both-routed} = 0.0\mathrm{e}0$ ($6{,}608 / 30{,}000 = 22.03\%$) | Dual parallel binding plus explicit arithmetic node computes correct sum/diff under correct routing. | Claiming online neural dynamics, spike-based composition, or learned generalization. |

---

## 4. Resolution of P0 Blockers (A01–A07 Summary)

1. **A01 (Query Rotation Double $\omega$):** Confirmed discrepancy between code ($\theta = \omega S$) and text ($u(s) = (\cos \omega s, \sin \omega s) \implies \theta = \omega^2 S$). Literal text fails to flip at $Q=4$ (margin $+0.8426$). Corrected in `paper2/sections/s2_framework.tex` to $\hat{u}(\theta) = (\cos\theta, \sin\theta)$, leaving frozen code untouched. Test fixture `test_a01_error_fixture()` confirms regression detection.
2. **A02 (Phase 2 Decision Tree Gap):** Verified `stable == True` ($d_{\text{delta}} = 1.0889 \ge 0.5$, $d_{\text{lo}} = 1.8739 > 0$). Outcome D was caused by falling through conditions A, B, and C in the decision tree. Corrected `MIN_LOGT_DIFF = 0.2` (from `effective_dimension.py:112`, was erroneously 0.5). Differentiated log-trace difference ($1.094463$) from difference-in-differences of event-minus-quiet changes ($1.354674$). Classified as `UNCLASSIFIED_MIXED_CASE`.
3. **A03 (Statistical Metrics & Units):** Corrected $K$ mean mislabeled as median in Paper 2; separated 2-candidate toy JSD ($0.0751 / 0.0647$) from full-window canonical JSD ($0.009462 / 0.008727$); distinguished single-seed max error ($0.009267$) from seed-averaged max error ($0.004667$); generatively derived Probe-1 sample sizes ($N_{\text{test}} = 2000$, $1/3$ catch $\implies 667$ catch, $1333$ present, $666$ distractor, $333$ catch-distractor $\implies 999/\text{seed} \times 3 = 2997$ distractor trials out of $6000$ total); resolved E3 check counts to 140 checks (132 isotropic $+ 8$ controls); distinguished Composition 4.3-A quadrature benchmarks ($1.116099621 / 2.000999500$) from 30,000 MC pooled empirical means ($1.107671595 / 1.975892767$), and reported both-routed coverage ($6608 / 30000 = 22.03\%$).
4. **A04 (Theoretical Scope & Minimality):** Refined mathematical definitions: Markovianity does not imply contraction; membrane potential $h_t$ alone is insufficient state descriptor, but complete state $(h_t, X_t)$ remains Markovian; finite softmax has full positive support across all non-infinite logits; scalar sorting handles $S=0$ and ties; soft blending requires distinct values $V_A \neq V_B$ and event-normalized readout $C_{\text{ev}}$; single-head delivery does not rule out symmetric pooling functions but cannot route two operands to downstream algebra; fixed $K=5$ does not imply invariant state space.
5. **A05 (Softmax Conventions):** Corrected Paper 2 Section 2 text claiming all sprints inherited $\delta=10^{-9}$. Sprint 4.2-B uses standard max-subtracted softmax without offset ($\sum \alpha_i = 1.0$ exactly); `tan.py` adds $10^{-9}$ in denominator.
6. **A06 (Chronology & Amendments):** Reconstructed full timeline from session logs and filesystem timestamps. Reconciled Sept 4 self-dates with Sept 8–9 execution. Explicitly marked unsourced times as `UNKNOWN` and template copying as plausible hypothesis. Clarified AM1–AM6 (AM4: M5-E drive transfer pre-run correction $+0.0985 \to +0.236700$; AM5: $D_{\text{full}}$ non-monotonicity and dilution cancellation artifact at $\gamma \approx 1.198$; AM6: runtime correction of rounding error in canonical margins $0.210631 / 0.261879$ vs $0.2118 / 0.2637$).
7. **A07 (Manuscript Quality & Readiness):** Removed obsolete draft headers (`Draft v0.1 — Sections 1–2 only`), updated correspondence placeholders, verified LaTeX compilation (clean exit code 0, Paper 1: 28 pages, Paper 2: 28 pages). Marked status as `PARTIALLY_RESOLVED (VISUAL_QA_PENDING / NOT_READY_FOR_SUBMISSION)` because autonomous text tools cannot perform visual rendering inspection of figures.

---

## 5. P1-A Address Identifiability Experiment & Formal Closure

Following authorization for autonomous multi-agent scientific execution, P1-A (Address Identifiability and Computational Carrier Audit) was formally pre-registered, implemented, audited across 5 review rounds, and executed at full scale in coordination with Codex CLI (`gpt-6-astra`, Conversation Thread `01a0b9ce-4527-7a11-9464-1ccd1a10c379`).

- **Execution Directory:** `p1_experiments/results/run_p1a_rev2/` (frozen, documented in `p1_experiments/P1A_FINAL_BASELINE_MANIFEST.md`).
- **Scale:** 60,000 synthetic trials across 6 conditions $\times$ 10 seeds (`2026091700–2026091709`); 420,000 total model evaluations across 7 model variants.
- **Resource Consumption:** 67.09s process CPU time (0.93% of 7,200s budget); 711.71 MB peak physical RAM (34.8% of 2,048 MB budget).
- **Post-Write Lossless Recomputation Audit:** 100% verified (`post_write_recomputation_audit.json`). Max numerical discrepancy between recomputed and stored errors was $3.55 \times 10^{-15} < 10^{-14}$; 20,000 zero-noise correct routing trials yielded max error $1.78 \times 10^{-15}$ (machine precision). All 42 condition $\times$ model summary aggregates matched live memory within machine precision.
- **Independent Peer Audit & Sign-off:** Codex independently verified the raw uncompressed trial datasets and issued the formal closure verdict:
  > **“最终裁决：本次正式重放验收通过；同意确立为 P1-A 的可信研究基线；同意在‘静态地址可识别性与检索算子边界’的限定范围内正式收口。”**

### Key Empirical & Theoretical Findings of P1-A:
1. **Sufficiency of 1D Metric Compatibility (Proposition 1):**
   A 1D distance-based compatibility kernel ($\text{score} = -(q - k)^2$) achieves **100.00% both-routed accuracy** on C1 ($N=2$) and C2 ($N=3$), with **100.00% middle-key hit rate** and zero hard binding error. Across all 60,000 trials, its hard argmax routing decisions are identical to 2D Vector-QK on $S^1$ ($\text{score} = \cos(0.4(q - k))$). This is inconsistent with the earlier conjecture that 2D orthogonal rotation is necessary for single-neuron content addressing (in this synthetic task).
2. **Failure of Scalar Dot-Products on Intermediate Keys (Proposition 2):**
   - Positive scalar kernel ($c S K_i$): 0.00% both-routed accuracy on C2 ($N=3$).
   - Signed scalar kernel ($\text{sign}(q - 1.4) K_i$): Drops from 52.14% on $N=2$ to 25.81% on $N=3$, with **exactly 0/6,705 middle-key channel hits (0.00% hit rate)**. Non-zero scalar multipliers can only preserve or invert rank order; they cannot make an intermediate key the unique maximum.
3. **Historical 4.2-B Kernel Demarcation:**
   The `HistoricalSprint4_2_KernelStaticControl` achieves only 13.68% accuracy on C2 (middle key hit rate 36.78%). Historical 4.2-B winner flips were fixed counterfactual switches across pre-set surprise drives, not semantic address retrieval.
4. **Hard Routing vs. Soft Readout Distinction:**
   On C2 noiseless, while hard routing accuracy is 100.00% for both, soft addition error differs: ScalarMetric = 0.4247, Vector-QK = 0.9463. Soft attention retains residual leakage bounded by kernel curvature and temperature $\gamma$.

---

## 6. Current Research State & Next Phase Pathways

- **P0 Rework Baseline:** Formally closed, verified in place, and fully auditable. The external source archive (368 files) was verified bit-for-bit at review time (`scripts/generate_manifest_after.py`, which hardcodes the author's local paths and cannot run from a clone). In this repository, `snapshot/` matches `manifest_before.json` for 337/368 files (17 revised paper/paper2 copies, 14 uncommitted build/log files).
- **P1-A Address Identifiability Baseline:** Formally established, audited, and closed with multi-agent consensus (`P1A_FINAL_BASELINE_MANIFEST.md`, `CLM-11` in `MASTER_CLAIM_LEDGER.csv`).
- **Strategic Pathways Available for User Decision:**
  - **Pathway A (Manuscript Formalization & Finalization):** Integrate P1-A findings into `paper2` (revising Section 2 and Section 10 to formalize the 1D Metric Sufficiency theorem and clarify the operator-level nature of content addressing), conduct visual inspection of compiled figures, and prepare the final unified publication bundle.
  - **Pathway B (P1-B Dynamical Carrier Investigation):** CLOSED (2026-09-27 update). See `p1_experiments/p1b/P1B_FINAL_SCIENTIFIC_CLOSURE_REPORT.md`: the pre-registered floors failed (M1 joint delivery 2.75–2.90% against floors of 80%/40%; spiking-decoder composition error $E_{\text{comp}} \ge 2.25$ in every cell).


