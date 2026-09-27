# P1-A Address Identifiability and Computational Carrier Scientific Report

> **Status (corrected 2026-09-27):** This is the superseded Run 1 report (6 models, 360,000 evaluations), kept for the record. The authoritative P1-A report is `results/run_p1a_rev2/P1A_SCIENTIFIC_REPORT.md` (7 models, 420,000 evaluations). Figures in Sections 1, 3 and 4 that contradicted this file's own table have been corrected; see `ERRATA.md` ERR-17.

**Execution Date:** 2026-09-19 14:25:20  
**Protocol Basis:** `RESEARCH_AUDIT_AND_PLAN_2026-09-17.md` Section 9 & Codex Handover Protocol  
**Pre-registered Seeds:** 10 seeds (`2026091700`–`2026091709`), 1,000 trials per seed per condition  
**Total Histories Analyzed:** 60,000 trials (6 conditions × 10 seeds × 1,000 trials)  

---

## 1. Executive Scientific Summary

P1-A resolves the critical open question from Sprint 4.3-A: *Does the model select the requested target object when given a legal query address cue, rather than merely flipping winner order across fixed surprise amplitudes?*

### Key Findings Across the 6 Comparison Models:
1. **Positive Scalar Kernel is Completely Address-Incapable:**
   - Achieves only **0.00% both-routed accuracy** on Condition C2 (N=3), because score $c S K_i$ strictly maximizes the largest key amplitude regardless of which key is requested.
2. **Signed Scalar Kernel Fails on Intermediate Keys:**
   - On N=2, signed scalar achieves 52.14% both-routed accuracy: its sign threshold is fixed at q=1.4, so both channels are correct only when the two keys straddle 1.4 (5,214/10,000 trials).
   - On N=3, however, its both-routed accuracy collapses to **25.81%** and its middle-key channel hit rate is **0.00%** (0/6,705). A signed scalar product can only select the minimum or maximum key, so it cannot address an intermediate key.
3. **Frozen S-Rotation (4.2-B Baseline) is Address-Blind:**
   - Achieves only **6.55% both-routed accuracy** on N=3, below the query-blind chance level $1/(3\times 2) = 16.67%$ (QueryBlindControl: 17.20%). This confirms that 4.2-B's winner flip was a fixed counterfactual switch across two predefined surprise amplitudes, not content-addressable routing.
4. **1D Scalar Metric Attention is Sufficient for Content Addressing in this Task:**
   - Achieves **100.00% both-routed accuracy** and **0.0000 binding/algebraic error** on zero-noise N=2 and N=3, with **100.00% middle-key accuracy**.
   - **Scientific Impact:** Inconsistent with the conjecture that 2D orthogonal rotation is necessary for content addressing in this synthetic task. A 1D metric compatibility kernel (distance $- (q - k)^2$) selects the queried key in every zero-noise trial.
5. **2D Vector-QK Attention Performs Equally Well:**
   - Achieves **100.00% both-routed accuracy** under zero noise, matching the scalar metric model.
6. **Noise and Out-of-Distribution Robustness:**
   - Under query observation noise (relative sigma=0.01 and 0.05; query-noise SD = 0.022 and 0.11), both Metric Attention and Vector-QK reach 98.76% and 79.86% both-routed accuracy respectively. In the OOD amplitude domains ([0.05, 0.3] and [2.5, 3.5]) both reach 100.00%.

---

## 2. Overall Performance Matrix Across Conditions

| Condition | Model | Both-Routed Acc (%) | Middle Key Acc (%) | Uncond Add Err | Cond Add Err |
|---|---|---|---|---|---|
| C1_N2_zero_noise | QueryBlindControl | 49.19% | N/A | 0.0000 | 0.0000 |
| C1_N2_zero_noise | PositiveScalarKernel | 0.00% | N/A | 1.9982 | N/A |
| C1_N2_zero_noise | SignedScalarKernel | 52.14% | N/A | 0.9526 | 0.0000 |
| C1_N2_zero_noise | FrozenSRotationReference | 9.92% | N/A | 1.6139 | 0.0000 |
| C1_N2_zero_noise | ScalarMetricAttention | 100.00% | N/A | 0.0000 | 0.0000 |
| C1_N2_zero_noise | VectorQKAddressAttention | 100.00% | N/A | 0.0000 | 0.0000 |
| C2_N3_zero_noise | QueryBlindControl | 17.20% | 17.18% | 1.3335 | 0.0000 |
| C2_N3_zero_noise | PositiveScalarKernel | 0.00% | 0.0% | 2.4772 | N/A |
| C2_N3_zero_noise | SignedScalarKernel | 25.81% | 0.0% | 1.6230 | 0.0000 |
| C2_N3_zero_noise | FrozenSRotationReference | 6.55% | 8.67% | 1.9818 | 0.0000 |
| C2_N3_zero_noise | ScalarMetricAttention | 100.00% | 100.0% | 0.0000 | 0.0000 |
| C2_N3_zero_noise | VectorQKAddressAttention | 100.00% | 100.0% | 0.0000 | 0.0000 |
| C3_N3_low_noise | QueryBlindControl | 16.99% | 17.34% | 1.3313 | 0.0000 |
| C3_N3_low_noise | PositiveScalarKernel | 0.00% | 0.0% | 2.4921 | N/A |
| C3_N3_low_noise | SignedScalarKernel | 25.96% | 0.0% | 1.5971 | 0.0000 |
| C3_N3_low_noise | FrozenSRotationReference | 7.17% | 9.67% | 1.9968 | 0.0000 |
| C3_N3_low_noise | ScalarMetricAttention | 98.76% | 98.55% | 0.0256 | 0.0000 |
| C3_N3_low_noise | VectorQKAddressAttention | 98.76% | 98.55% | 0.0256 | 0.0000 |
| C4_N3_high_noise | QueryBlindControl | 16.99% | 17.34% | 1.3313 | 0.0000 |
| C4_N3_high_noise | PositiveScalarKernel | 0.00% | 0.0% | 2.4921 | N/A |
| C4_N3_high_noise | SignedScalarKernel | 25.51% | 0.0% | 1.6058 | 0.0000 |
| C4_N3_high_noise | FrozenSRotationReference | 7.17% | 9.67% | 1.9968 | 0.0000 |
| C4_N3_high_noise | ScalarMetricAttention | 79.86% | 77.79% | 0.3882 | 0.0000 |
| C4_N3_high_noise | VectorQKAddressAttention | 79.86% | 77.79% | 0.3882 | 0.0000 |
| C5_N3_low_amp_ood | QueryBlindControl | 16.34% | 16.5% | 1.3176 | 0.0000 |
| C5_N3_low_amp_ood | PositiveScalarKernel | 0.00% | 0.0% | 2.4888 | N/A |
| C5_N3_low_amp_ood | SignedScalarKernel | 0.00% | 0.0% | 2.5094 | N/A |
| C5_N3_low_amp_ood | FrozenSRotationReference | 0.00% | 0.0% | 2.4888 | N/A |
| C5_N3_low_amp_ood | ScalarMetricAttention | 100.00% | 100.0% | 0.0000 | 0.0000 |
| C5_N3_low_amp_ood | VectorQKAddressAttention | 100.00% | 100.0% | 0.0000 | 0.0000 |
| C6_N3_high_amp_ood | QueryBlindControl | 16.95% | 17.29% | 1.3329 | 0.0000 |
| C6_N3_high_amp_ood | PositiveScalarKernel | 0.00% | 0.0% | 2.5049 | N/A |
| C6_N3_high_amp_ood | SignedScalarKernel | 0.00% | 0.0% | 2.5049 | N/A |
| C6_N3_high_amp_ood | FrozenSRotationReference | 0.61% | 0.91% | 2.4693 | 0.0000 |
| C6_N3_high_amp_ood | ScalarMetricAttention | 100.00% | 100.0% | 0.0000 | 0.0000 |
| C6_N3_high_amp_ood | VectorQKAddressAttention | 100.00% | 100.0% | 0.0000 | 0.0000 |

---

## 3. G1 Identifiability Gate Verification
- **Oracle Soundness:** Verified 100.0% accuracy on N=2 and N=3 zero-noise calibration trials.
- **Leakage Check:** No G1 artifact records the 16.71% originally quoted here. The frozen rev2 G1 report (`results/run_p1a_rev2/g1_gate_report.json`) gives QueryBlindControl 17.20% both-routed accuracy (172/1,000; 95% CI 14.99–19.66%), below the chance + 5% threshold of 21.67%.
- **Adversarial Query/Value Swaps:** Confirmed that address-aware models invert output signs under query swap and track values under value swap.

---

## 4. Scientific Recommendations for Codex & Antigravity Joint Sign-off
1. **Acknowledge 1D Metric Sufficiency:** The paper's narrative must not claim that 2D vector representations are necessary for single-neuron content addressing; 1D metric attention makes identical hard routing decisions.
2. **Permanent Closure of Scalar Product Routing:** Positive and signed scalar dot products cannot address intermediate keys (Proposition 2).
3. **Prerequisite Met for Transition:** With G1 passed and address identifiability established, P1-A is verified and ready for formal sign-off.
