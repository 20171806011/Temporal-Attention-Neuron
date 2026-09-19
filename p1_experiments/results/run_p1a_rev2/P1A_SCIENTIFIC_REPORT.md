# P1-A Address Identifiability and Computational Carrier Scientific Report

**Run Identifier:** `run_p1a_rev2` (Full Replay)  
**Execution Timestamp:** 2026-09-19T14:04:31.758776+00:00  
**Platform:** Windows-11-10.0.26200-SP0 (Intel64 Family 6 Model 154 Stepping 4, GenuineIntel)  
**Wall Runtime:** 69.52 s | **CPU Time:** 67.09 s  
**Peak Memory:** 711.71 MB  
**Protocol Basis:** `RESEARCH_AUDIT_AND_PLAN_2026-09-17.md` Section 9 & Codex P1-A Amendment 1  

---

## 1. Executive Scientific Summary (100% Data-Driven)

P1-A experimentally addresses the foundational question: *Does the model select the requested target object when given a legal query address cue, rather than merely flipping winner order across fixed surprise amplitudes?*

### Key Measured Findings Across Evaluated Models:
1. **Positive Scalar Kernel is Completely Address-Incapable:**
   - Achieved exactly 0.00% both-routed accuracy on Condition C2 (N=3), because score $c S K_i$ strictly selects the maximum key amplitude regardless of query.
2. **Signed Scalar Kernel Strictly Fails on Intermediate Keys:**
   - On N=2 (C1), signed scalar achieved 52.14% both-routed accuracy.
   - On N=3 (C2), its both-routed accuracy dropped to 25.81%, and its **Middle Key Channel Hit Rate is exactly 0.00%**!
   - **Mechanistic Foundation:** Linear scalar multiplication can only preserve or invert key ordering; it cannot place an intermediate key at the maximum position among $\ge 3$ distinct items.
3. **Historical 4.2-B Reference Models are Address-Blind:**
   - `HistoricalSprint4_2_KernelStaticControl` (adopting historical Sprint 4.2-B polynomial moment curve kernel) achieved 13.68% both-routed accuracy on C2.
   - `FrozenSRotationHarmonicReference` (adopting harmonic unit-circle kernel) achieved 6.55% both-routed accuracy on C2.
   - This confirms that fixed surprise drives without an independent query address port cannot route memory requests.
4. **1D Scalar Metric Attention is Fully Sufficient for Content Addressing:**
   - On C1 (N=2) and C2 (N=3) zero noise, achieved 100.00% and 100.00% hard both-routed accuracy.
   - Its middle key channel hit rate on C2 is 100.00%.
   - **Theoretical & Empirical Impact:** Proves that 2D orthogonal rotation is not universally necessary for single-unit associative content addressing.
5. **Comparative Behavior: 1D Metric vs. 2D Vector-QK:**
   - Hard Routing: Both ScalarMetricAttention and VectorQKAddressAttention achieved identical hard both-routed accuracy of 100.00%.
   - Soft Attention Readout: Soft addition error differed between kernels: ScalarMetric = 0.4247, VectorQK = 0.9463.
6. **Noise and Out-of-Distribution Robustness:**
   - Under query noise (C3 $\sigma=0.01$, C4 $\sigma=0.05$), ScalarMetric hard accuracy: C3 = 98.76%, C4 = 79.86%.
   - Under OOD low keys [0.05, 0.3] (C5), ScalarMetric achieved 100.00% and VectorQK achieved 100.00%. Under OOD high keys [2.5, 3.5] (C6), ScalarMetric achieved 100.00% and VectorQK achieved 100.00%.

---

## 2. Overall Performance Matrix Across Conditions

| Condition | Model | Both-Routed Acc (%) | Middle Key Chan Acc (%) | Joint Acc Given Mid (%) | Hard Add Err | Soft Add Err |
|---|---|---|---|---|---|---|
| C1_N2_zero_noise | QueryBlindControl | 49.19% | N/A | N/A | 0.0000 | 0.0000 |
| C1_N2_zero_noise | PositiveScalarKernel | 0.00% | N/A | N/A | 1.9982 | 1.9396 |
| C1_N2_zero_noise | SignedScalarKernel | 52.14% | N/A | N/A | 0.9526 | 0.7957 |
| C1_N2_zero_noise | HistoricalSprint4_2_KernelStaticControl | 29.36% | N/A | N/A | 0.7957 | 0.5993 |
| C1_N2_zero_noise | FrozenSRotationHarmonicReference | 9.92% | N/A | N/A | 1.6139 | 0.7492 |
| C1_N2_zero_noise | ScalarMetricAttention | 100.00% | N/A | N/A | 0.0000 | 0.0000 |
| C1_N2_zero_noise | VectorQKAddressAttention | 100.00% | N/A | N/A | 0.0000 | 0.0000 |
| C2_N3_zero_noise | QueryBlindControl | 17.20% | 33.75% | 17.18% | 1.3335 | 1.1685 |
| C2_N3_zero_noise | PositiveScalarKernel | 0.00% | 0.0% | 0.0% | 2.4772 | 2.4149 |
| C2_N3_zero_noise | SignedScalarKernel | 25.81% | 0.0% | 0.0% | 1.6230 | 1.4593 |
| C2_N3_zero_noise | HistoricalSprint4_2_KernelStaticControl | 13.68% | 36.78% | 15.56% | 1.5077 | 1.2760 |
| C2_N3_zero_noise | FrozenSRotationHarmonicReference | 6.55% | 18.43% | 8.67% | 1.9818 | 1.3203 |
| C2_N3_zero_noise | ScalarMetricAttention | 100.00% | 100.0% | 100.0% | 0.0000 | 0.4247 |
| C2_N3_zero_noise | VectorQKAddressAttention | 100.00% | 100.0% | 100.0% | 0.0000 | 0.9463 |
| C3_N3_low_noise | QueryBlindControl | 16.99% | 33.82% | 17.34% | 1.3313 | 1.1719 |
| C3_N3_low_noise | PositiveScalarKernel | 0.00% | 0.0% | 0.0% | 2.4921 | 2.4281 |
| C3_N3_low_noise | SignedScalarKernel | 25.96% | 0.0% | 0.0% | 1.5971 | 1.4346 |
| C3_N3_low_noise | HistoricalSprint4_2_KernelStaticControl | 14.27% | 35.9% | 16.2% | 1.5091 | 1.2838 |
| C3_N3_low_noise | FrozenSRotationHarmonicReference | 7.17% | 19.51% | 9.67% | 1.9968 | 1.3302 |
| C3_N3_low_noise | ScalarMetricAttention | 98.76% | 99.11% | 98.55% | 0.0256 | 0.4444 |
| C3_N3_low_noise | VectorQKAddressAttention | 98.76% | 99.11% | 98.55% | 0.0256 | 0.9514 |
| C4_N3_high_noise | QueryBlindControl | 16.99% | 33.82% | 17.34% | 1.3313 | 1.1719 |
| C4_N3_high_noise | PositiveScalarKernel | 0.00% | 0.0% | 0.0% | 2.4921 | 2.4281 |
| C4_N3_high_noise | SignedScalarKernel | 25.51% | 0.0% | 0.0% | 1.6058 | 1.4421 |
| C4_N3_high_noise | HistoricalSprint4_2_KernelStaticControl | 14.27% | 35.9% | 16.2% | 1.5091 | 1.2838 |
| C4_N3_high_noise | FrozenSRotationHarmonicReference | 7.17% | 19.51% | 9.67% | 1.9968 | 1.3302 |
| C4_N3_high_noise | ScalarMetricAttention | 79.86% | 83.46% | 77.79% | 0.3882 | 0.5263 |
| C4_N3_high_noise | VectorQKAddressAttention | 79.86% | 83.46% | 77.79% | 0.3882 | 0.9547 |
| C5_N3_OOD_low_keys | QueryBlindControl | 16.34% | 32.89% | 16.5% | 1.3176 | 1.1618 |
| C5_N3_OOD_low_keys | PositiveScalarKernel | 0.00% | 0.0% | 0.0% | 2.4888 | 1.7639 |
| C5_N3_OOD_low_keys | SignedScalarKernel | 0.00% | 0.0% | 0.0% | 2.5094 | 1.4103 |
| C5_N3_OOD_low_keys | HistoricalSprint4_2_KernelStaticControl | 0.00% | 0.0% | 0.0% | 2.4888 | 1.8866 |
| C5_N3_OOD_low_keys | FrozenSRotationHarmonicReference | 0.00% | 0.0% | 0.0% | 2.4888 | 1.1734 |
| C5_N3_OOD_low_keys | ScalarMetricAttention | 100.00% | 100.0% | 100.0% | 0.0000 | 1.1143 |
| C5_N3_OOD_low_keys | VectorQKAddressAttention | 100.00% | 100.0% | 100.0% | 0.0000 | 1.1580 |
| C6_N3_OOD_high_keys | QueryBlindControl | 16.95% | 34.14% | 17.29% | 1.3329 | 1.1598 |
| C6_N3_OOD_high_keys | PositiveScalarKernel | 0.00% | 0.0% | 0.0% | 2.5049 | 2.3606 |
| C6_N3_OOD_high_keys | SignedScalarKernel | 0.00% | 0.0% | 0.0% | 2.5049 | 2.1129 |
| C6_N3_OOD_high_keys | HistoricalSprint4_2_KernelStaticControl | 0.00% | 0.0% | 0.0% | 2.5112 | 1.2402 |
| C6_N3_OOD_high_keys | FrozenSRotationHarmonicReference | 0.61% | 1.81% | 0.91% | 2.4693 | 1.2123 |
| C6_N3_OOD_high_keys | ScalarMetricAttention | 100.00% | 100.0% | 100.0% | 0.0000 | 0.7265 |
| C6_N3_OOD_high_keys | VectorQKAddressAttention | 100.00% | 100.0% | 100.0% | 0.0000 | 1.1033 |

---

## 3. G1 Identifiability Gate Audit Record (Extracted from Live Log)
- **G1.1 Oracle Soundness:**
  - N=2: 500 trials, both-acc = 100.00%, mean add err = 0.0
  - N=3: 500 trials, both-acc = 100.00%, mean add err = 0.0
- **G1.2 Information Leakage Monitor (QueryBlindControl):**
  - Empirical Joint Both-Acc: 17.20% (172/1000 trials)
  - Channel 1 Acc: 34.50% | Channel 2 Acc: 34.20%
  - Wilson 95% Confidence Interval: [14.99%, 19.66%]
  - Thresholds: Point estimate <= 21.67%; CI lower bound <= 18.67%
- **G1.2b Negative Fixture Check (Verified Production Rejection):**
  - LeakyCheat model achieved 66.10% and triggered production AssertionError: "LEAKAGE ASSERTION FAILURE: empirical accuracy 0.6610 exceeds threshold 0.2167 (chance=0.1667, delta=0.0500)"
- **G1.3 Adversarial Sensitivity & Invariance Suite:**
  - **ScalarMetricAttention:**
    - Query-swap: 200/200
    - Value-swap: 200/200
    - Zero-trap:  200/200
    - Equal-val:  200/200
    - Pos-perm:   200/200
  - **VectorQKAddressAttention:**
    - Query-swap: 200/200
    - Value-swap: 200/200
    - Zero-trap:  200/200
    - Equal-val:  200/200
    - Pos-perm:   200/200
- **G1.4 Historical Sprint 4.2-B Scoring Kernel Regression:**
  - QA=3.0: diff = 0.258742724, winner = A (Expected: +0.258742724, 'A')
  - QB=4.0: diff = -0.155910133, winner = B (Expected: -0.155910133, 'B')
  - Status: Passed = True

---

## 4. Formal Code Hashes for Provenance
```json
{
  "task_generator.py": "674cf9f49f5cd7f4975b87d29cd413e801fe40bd10fc56b966e663c9e6fc989d",
  "models.py": "3762cfb8627e7574dfbcbdc8db2470489471ad247832d03fa2dae916ce3f23e5",
  "oracle_and_evaluator.py": "13392549dcbddf5d6acb05721a57873714d2ca1a84e2ed712d1b8b5fe10d6fe3",
  "run_p1a_experiment.py": "256c8b777407e818c4c44ef7c8b0e10fceeabf498c661cc630a117e774f7bdb0"
}
```
