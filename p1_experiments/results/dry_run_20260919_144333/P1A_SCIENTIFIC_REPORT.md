# P1-A Address Identifiability and Computational Carrier Scientific Report

**Run Identifier:** `run_p1a_rev2` (Dry Run)  
**Execution Timestamp:** 2026-09-19T13:43:33.684734+00:00  
**Platform:** Windows-11-10.0.26200-SP0 (Intel64 Family 6 Model 154 Stepping 4, GenuineIntel)  
**Wall Runtime:** 1.34 s | **CPU Time:** 1.02 s  
**Peak Memory:** 91.37 MB  
**Protocol Basis:** `RESEARCH_AUDIT_AND_PLAN_2026-09-17.md` Section 9 & Codex P1-A Amendment 1  

---

## 1. Executive Scientific Summary (100% Data-Driven)

P1-A experimentally addresses the foundational question: *Does the model select the requested target object when given a legal query address cue, rather than merely flipping winner order across fixed surprise amplitudes?*

### Key Measured Findings Across Evaluated Models:
1. **Positive Scalar Kernel is Completely Address-Incapable:**
   - Achieved exactly 0.00% both-routed accuracy on Condition C2 (N=3), because score $c S K_i$ strictly selects the maximum key amplitude regardless of query.
2. **Signed Scalar Kernel Strictly Fails on Intermediate Keys:**
   - On N=2 (C1), signed scalar achieved 40.00% both-routed accuracy.
   - On N=3 (C2), its both-routed accuracy dropped to 0.00%, and its **Middle Key Channel Hit Rate is exactly 0.00%**!
   - **Mechanistic Foundation:** Linear scalar multiplication can only preserve or invert key ordering; it cannot place an intermediate key at the maximum position among $\ge 3$ distinct items.
3. **Historical 4.2-B Reference Models are Address-Blind:**
   - `HistoricalSprint4_2_KernelStaticControl` (adopting historical Sprint 4.2-B polynomial moment curve kernel) achieved 5.00% both-routed accuracy on C2.
   - `FrozenSRotationHarmonicReference` (adopting harmonic unit-circle kernel) achieved 10.00% both-routed accuracy on C2.
   - This confirms that fixed surprise drives without an independent query address port cannot route memory requests.
4. **1D Scalar Metric Attention is Fully Sufficient for Content Addressing:**
   - On C1 (N=2) and C2 (N=3) zero noise, achieved 100.00% and 100.00% hard both-routed accuracy.
   - Its middle key channel hit rate on C2 is 100.00%.
   - **Theoretical & Empirical Impact:** Proves that 2D orthogonal rotation is not universally necessary for single-unit associative content addressing.
5. **Comparative Behavior: 1D Metric vs. 2D Vector-QK:**
   - Hard Routing: Both ScalarMetricAttention and VectorQKAddressAttention achieved identical hard both-routed accuracy of 100.00%.
   - Soft Attention Readout: Soft addition error differed between kernels: ScalarMetric = 0.4386, VectorQK = 0.8734.
6. **Noise and Out-of-Distribution Robustness:**
   - Under query noise (C3 $\sigma=0.01$, C4 $\sigma=0.05$), ScalarMetric hard accuracy: C3 = 100.00%, C4 = 75.00%.
   - Under OOD low keys [0.05, 0.3] (C5), ScalarMetric achieved 100.00% and VectorQK achieved 100.00%. Under OOD high keys [2.5, 3.5] (C6), ScalarMetric achieved 100.00% and VectorQK achieved 100.00%.

---

## 2. Overall Performance Matrix Across Conditions

| Condition | Model | Both-Routed Acc (%) | Middle Key Chan Acc (%) | Joint Acc Given Mid (%) | Hard Add Err | Soft Add Err |
|---|---|---|---|---|---|---|
| C1_N2_zero_noise | QueryBlindControl | 45.00% | N/A | N/A | 0.0000 | 0.0000 |
| C1_N2_zero_noise | PositiveScalarKernel | 0.00% | N/A | N/A | 2.0824 | 2.0462 |
| C1_N2_zero_noise | SignedScalarKernel | 40.00% | N/A | N/A | 0.8043 | 0.7252 |
| C1_N2_zero_noise | HistoricalSprint4_2_KernelStaticControl | 25.00% | N/A | N/A | 0.6624 | 0.7215 |
| C1_N2_zero_noise | FrozenSRotationHarmonicReference | 10.00% | N/A | N/A | 1.9852 | 0.9018 |
| C1_N2_zero_noise | ScalarMetricAttention | 100.00% | N/A | N/A | 0.0000 | 0.0000 |
| C1_N2_zero_noise | VectorQKAddressAttention | 100.00% | N/A | N/A | 0.0000 | 0.0000 |
| C2_N3_zero_noise | QueryBlindControl | 10.00% | 26.67% | 0.0% | 1.8400 | 1.0976 |
| C2_N3_zero_noise | PositiveScalarKernel | 0.00% | 0.0% | 0.0% | 1.6216 | 1.6264 |
| C2_N3_zero_noise | SignedScalarKernel | 0.00% | 0.0% | 0.0% | 1.9873 | 1.9484 |
| C2_N3_zero_noise | HistoricalSprint4_2_KernelStaticControl | 5.00% | 20.0% | 6.67% | 1.2809 | 1.1278 |
| C2_N3_zero_noise | FrozenSRotationHarmonicReference | 10.00% | 26.67% | 13.33% | 1.1453 | 0.8049 |
| C2_N3_zero_noise | ScalarMetricAttention | 100.00% | 100.0% | 100.0% | 0.0000 | 0.4386 |
| C2_N3_zero_noise | VectorQKAddressAttention | 100.00% | 100.0% | 100.0% | 0.0000 | 0.8734 |
| C3_N3_low_noise | QueryBlindControl | 10.00% | 18.75% | 6.25% | 1.7170 | 1.2230 |
| C3_N3_low_noise | PositiveScalarKernel | 0.00% | 0.0% | 0.0% | 2.0618 | 2.0202 |
| C3_N3_low_noise | SignedScalarKernel | 15.00% | 0.0% | 0.0% | 1.3803 | 1.1007 |
| C3_N3_low_noise | HistoricalSprint4_2_KernelStaticControl | 25.00% | 43.75% | 18.75% | 1.2571 | 1.3780 |
| C3_N3_low_noise | FrozenSRotationHarmonicReference | 5.00% | 12.5% | 6.25% | 1.8379 | 1.5238 |
| C3_N3_low_noise | ScalarMetricAttention | 100.00% | 100.0% | 100.0% | 0.0000 | 0.4253 |
| C3_N3_low_noise | VectorQKAddressAttention | 100.00% | 100.0% | 100.0% | 0.0000 | 0.8945 |
| C4_N3_high_noise | QueryBlindControl | 10.00% | 18.75% | 6.25% | 1.7170 | 1.2230 |
| C4_N3_high_noise | PositiveScalarKernel | 0.00% | 0.0% | 0.0% | 2.0618 | 2.0202 |
| C4_N3_high_noise | SignedScalarKernel | 15.00% | 0.0% | 0.0% | 1.4315 | 1.1682 |
| C4_N3_high_noise | HistoricalSprint4_2_KernelStaticControl | 25.00% | 43.75% | 18.75% | 1.2571 | 1.3780 |
| C4_N3_high_noise | FrozenSRotationHarmonicReference | 5.00% | 12.5% | 6.25% | 1.8379 | 1.5238 |
| C4_N3_high_noise | ScalarMetricAttention | 75.00% | 75.0% | 75.0% | 0.4863 | 0.4393 |
| C4_N3_high_noise | VectorQKAddressAttention | 75.00% | 75.0% | 75.0% | 0.4863 | 0.8881 |
| C5_N3_OOD_low_keys | QueryBlindControl | 5.00% | 42.86% | 0.0% | 1.9252 | 0.9878 |
| C5_N3_OOD_low_keys | PositiveScalarKernel | 0.00% | 0.0% | 0.0% | 2.0468 | 1.2059 |
| C5_N3_OOD_low_keys | SignedScalarKernel | 0.00% | 0.0% | 0.0% | 3.0488 | 1.5318 |
| C5_N3_OOD_low_keys | HistoricalSprint4_2_KernelStaticControl | 0.00% | 0.0% | 0.0% | 2.0468 | 1.3152 |
| C5_N3_OOD_low_keys | FrozenSRotationHarmonicReference | 0.00% | 0.0% | 0.0% | 2.0468 | 0.9403 |
| C5_N3_OOD_low_keys | ScalarMetricAttention | 100.00% | 100.0% | 100.0% | 0.0000 | 0.9320 |
| C5_N3_OOD_low_keys | VectorQKAddressAttention | 100.00% | 100.0% | 100.0% | 0.0000 | 0.9832 |
| C6_N3_OOD_high_keys | QueryBlindControl | 10.00% | 40.0% | 6.67% | 1.6534 | 1.1422 |
| C6_N3_OOD_high_keys | PositiveScalarKernel | 0.00% | 0.0% | 0.0% | 1.9273 | 1.8334 |
| C6_N3_OOD_high_keys | SignedScalarKernel | 0.00% | 0.0% | 0.0% | 1.9273 | 1.5860 |
| C6_N3_OOD_high_keys | HistoricalSprint4_2_KernelStaticControl | 0.00% | 0.0% | 0.0% | 3.6467 | 1.4529 |
| C6_N3_OOD_high_keys | FrozenSRotationHarmonicReference | 0.00% | 0.0% | 0.0% | 3.6467 | 1.3673 |
| C6_N3_OOD_high_keys | ScalarMetricAttention | 100.00% | 100.0% | 100.0% | 0.0000 | 0.7087 |
| C6_N3_OOD_high_keys | VectorQKAddressAttention | 100.00% | 100.0% | 100.0% | 0.0000 | 1.0805 |

---

## 3. G1 Identifiability Gate Audit Record (Extracted from Live Log)
- **G1.1 Oracle Soundness:**
  - N=2: 50 trials, both-acc = 100.00%, mean add err = 0.0
  - N=3: 50 trials, both-acc = 100.00%, mean add err = 0.0
- **G1.2 Information Leakage Monitor (QueryBlindControl):**
  - Empirical Joint Both-Acc: 17.20% (172/1000 trials)
  - Channel 1 Acc: 34.50% | Channel 2 Acc: 34.20%
  - Wilson 95% Confidence Interval: [14.99%, 19.66%]
  - Thresholds: Point estimate <= 21.67%; CI lower bound <= 18.67%
- **G1.2b Negative Fixture Check:**
  - LeakyCheat model achieved 66.10% and was correctly REJECTED: True
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
  "oracle_and_evaluator.py": "1f20cbbe01c852fb4c4f65c18b337cde0507a2c452e7189cefaeefbdf0f91695",
  "run_p1a_experiment.py": "b2cbe737519f4d9fc43c4688bcc40be40ebdaae236efea513cea847d316758ab"
}
```
