# P0 Handover and Research Transition Document (P0_HANDOFF.md)

**Handover Baseline Date:** 2026-09-17  
**Review Directory:** `C:\Users\李则徐\Downloads\TAN_Review\P0_20260917_181757`  
**Original Archive:** `C:\Users\李则徐\Downloads\TAN_Final_Submission\TAN_Final_Submission` (368 files, 100% frozen, 0 files modified)

---

## 1. P0 Execution and Delivery Summary

The P0 phase has established a verified, reproducible, and mathematically consistent scientific baseline for the Temporal Attention Neuron (TAN) project. All 10 mandatory deliverables are in place:
1. `REVIEW_README.md`: Workspace orientation and navigation guide.
2. `CURRENT_RESEARCH_STATE.md`: Comprehensive audit snapshot of scientific boundaries.
3. `RESEARCH_CHRONOLOGY.md`: 25-round reconstructed research timeline and amendment classification.
4. `manifest_before.json`: 368-entry baseline manifest with SHA-256 hashes and historical records.
5. `manifest_after.json`: Final post-execution manifest verifying source integrity and logging derived artifacts.
6. `MASTER_CLAIM_LEDGER.csv`: 10-claim master ledger mapping claims to evidence, counterexamples, and allowed wordings.
7. `FINDINGS.csv`: Granular registry of issues A01–A07 with severity, handling, and resolution status.
8. `ERRATA.md`: Detailed scientific errata documenting original vs revised formulations and rationales.
9. `NUMERICAL_CHECKS.json`: Consolidated machine-readable results of independent numerical verifications.
10. `VALIDATION_REPORT.md`: Multi-dimensional report covering numerical, replay, literature, compilation, and QA audits.

In addition, all independent check scripts are preserved in `scripts/`, replay logs in `logs/`, and revised, recompiled manuscripts in `snapshot/paper/` and `snapshot/paper2/`.

---

## 2. Claim Disposition Matrix

| Claim ID | Focus Area | Original Status | P0 Review Disposition | Handover Guidance & Allowed Scientific Wording |
|---|---|---|---|---|
| **CLM-01** | Phase 1 Membrane Sufficiency | CONFIRMED | **RETAINED (QUALIFIED)** | Membrane potential $h$ alone is insufficient state for TAN; history dependence is provided by the temporal window buffer. Complete state $(h_t, X_t)$ remains Markovian. Input domain $[0.0, 3.0]$. |
| **CLM-02** | Phase 2 Response Geometry | OUTCOME: D | **[AMENDED IN REWORK] DOWNGRADED (QUALIFIED)** | Stop condition decoupled from intrinsic dimension; local covariance divergence event-locked against quiet baseline ($B4 - B3 = 1.0945$ nats). Outcome D was an unhandled classification combination (`UNCLASSIFIED_MIXED_CASE`). Does not support higher intrinsic dimension. |
| **CLM-03** | Probe-1 Distractor Retrieval | CLAIMED_SUPERIOR | **[SUPERSEDED / DOWNGRADED IN REWORK] LIMITED SUPPORT** | Discarded unqualified superiority claim. B2 delay-line buffer is a strong baseline ($\approx 0.843$ vs B4 $\approx 0.849$). Advantage is conditional on specific distractor cues; Outcome B represents generic dynamical advantage. |
| **CLM-04** | Probe-2 Positive Scalar Routing | AUDIT_FAILED | **[TERMINATED / REFUTED IN REWORK] CLOSED BOUNDARY** | Positive scalar kernel is monotonically constrained by amplitude; cannot perform content routing (0 flips, Theorem 5.1). Permanently closed negative boundary. |
| **CLM-05** | Sprint 4.1 Opponent Tuning | CLAIMED_BREAKTHROUGH | **[AMENDED IN REWORK] ESTABLISHED TUNING / NEGATIVE ROUTING** | Opponent structure establishes non-monotonic tuning peaks; does NOT break scalar sorting barrier (flip rate = 0 across 6 layouts). |
| **CLM-06** | Sprint 4.2-A Query Geometry | THEORETICALLY_PROVEN | **RETAINED (REDUCED SCOPE)** | Dot-product angular flip follows $\theta/\pi$ on $S^1$ (Charikar 2002); minimal 2D sufficiency verified. 132 isotropic conditions + 8 controls (140 total checks); max deviation $0.004667$. Does not prove 2D is necessary across all attention families. |
| **CLM-07** | Sprint 4.2-B Vector QK Flips | CONFIRMED | **RETAINED (TYPO RESOLVED)** | Normalized vector QK produces counterfactual winner flips ($Q=3 \to A, Q=4 \to B$); paper typo (double $\omega$) corrected. Flip rate must not be conflated with address correctness. |
| **CLM-08** | Sprint 4.2-C Soft Value Binding | CONFIRMED | **RETAINED (BOUNDED)** | KV decoupling transmits bound continuous values via event-normalized readout ($D_{\text{ev}} \approx 1.5$). Soft blending obstacle prevents exact discrete value delivery under finite gain. |
| **CLM-09** | Sprint 4.2-D WTA Limit | CONFIRMED | **RETAINED (CONSTRUCTIVE)** | One-hot argmax selection delivers exact values conditioned on correct upstream routing; does not repair mis-routing. $\gamma^* = 5.098015$ crossing root restores E-dominant routing. |
| **CLM-10** | Sprint 4.3-A Composition | CLAIMED_COMPOSITION | **[SUPERSEDED / DOWNGRADED IN REWORK] CONSTRUCTIVE OPERATOR SUFFICIENCY** | Constructive operator sufficiency of static retrieval plus explicit arithmetic node ($y = C_1 \pm C_2$). Conditioned on correct dual routing ($6,608 / 30,000 = 22.03\%$ coverage), error is $0.0$; unconditional quadrature benchmarks are $1.116$ (add) / $2.001$ (sub), matching 30k MC empirical means $1.108$ / $1.976$. Does not prove online neural dynamics or learned generalization. |

---

## 3. Literature and Scientific Novelty Assessment

Following direct verification against foundational literature:
1. **$\theta/\pi$ Angular Flip Law:** Formally established by Charikar (STOC 2002) for random hyperplane rounding. In TAN-II, it represents an elegant mechanistic demonstration of a known geometric property in a single unit, rather than an independent mathematical discovery.
2. **Softmax Full Support & Sparsity:** Established by Martins & Astudillo (ICML 2016). Finite Boltzmann softmax cannot yield hard selection; alternative sparse projections exist.
3. **Attention as Dynamic Memory Programming:** Established by Schlag et al. (ICML 2021) and Ramsauer et al. (2020).
4. **Legitimate Core Novelty of TAN:**
   - A rigorous identifiability audit framework and negative control methodology for single-neuron attention.
   - Exact delineation of the computational boundary between scalar and vector query-key interactions.
   - Controlled demonstration of the minimal organizational degrees of freedom required to climb from memory to saliency, routing, binding, and composition.

---

## 4. Manuscript Publication Readiness

- **Status:** **NOT READY FOR SUBMISSION**.
- **Blockers:**
  1. Author and correspondence details contain placeholders (`[email placeholder --- see archive metadata]`).
  2. Full-page visual layout inspection (`VISUAL_QA_NOT_COMPLETED`) has not been conducted due to automated image rendering constraints.
  3. Related work and introduction sections must incorporate Charikar (2002), Martins & Astudillo (2016), Schlag et al. (2021), and Ramsauer et al. (2020) to frame contributions accurately.
- **Guidance:** Manuscripts in `snapshot/paper/` and `snapshot/paper2/` serve as internal, audited reference baselines. Submission actions require explicit user authorization.

---

## 5. Transition Recommendation: P1 GO / NO-GO

### Formal Recommendation: **CONDITIONAL GO FOR P1-A PROTOCOL DESIGN; NO-GO FOR IMMEDIATE RUNS**

**Scientific Justification:**
P0 has cleanly delineated existing findings. The immediate scientific hurdle is that **Sprint 4.3-A's 22% coverage is not a generalized addressable routing success**, because fixed queries $(4.5, 5.3)$ do not encode the identity of random keys $K_A, K_B$. 

Before any new experiments are executed, the protocol for **P1-A (Address Identifiability)** must be formally reviewed and approved by the user:
- **Task Identifiability Gate (G1):** The task must provide a legal input cue specifying which key is requested without leaking target value or using unobservable generator labels ($A, B$).
- **Minimal Comparison Suite:** Must test:
  1. Query-blind buffer control.
  2. Positive scalar product kernel.
  3. Signed scalar kernel.
  4. Frozen S-rotation reference.
  5. Scalar metric attention: $-(q - k)^2$.
  6. Explicit address vector-QK.
- **Resource Constraints:** $\le 2$ CPU hours, $\le 2$ GB RAM, 10 seeds ($2026091700$--$2026091709$), $1,000$ histories per condition.
- **Strict Prohibition:** **No new scientific runs shall be launched without explicit user approval.**
