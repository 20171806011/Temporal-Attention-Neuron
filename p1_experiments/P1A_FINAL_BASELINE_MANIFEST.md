# P1-A Final Baseline Manifest & Closure Sign-Off Record

**Experiment Title:** P1-A Address Identifiability and Computational Carrier Audit  
**Review Snapshot Base:** `C:\Users\李则徐\Downloads\TAN_Review\P0_20260917_181757`  
**Execution Output Directory:** `p1_experiments/results/run_p1a_rev2/`  
**Date of Baseline Freeze:** 2026-09-19  
**Multi-Agent Peer Reviewer:** Codex CLI (`gpt-6-astra`, Conversation Thread `01a0b9ce-4527-7a11-9464-1ccd1a10c379`)  
**Final Verdict:** **GO & FORMAL CLOSURE ACCEPTED**

---

## 1. Executive Summary & Closure Verdict

Following 5 rounds of rigorous multi-agent methodological review, pre-registration specification locking, failure-mode fixture testing, and full-scale execution across 60,000 trials (420,000 model evaluations), the P1-A experiment has achieved 100% verification across all G1 identifiability gates, resource budgets, and post-write lossless recomputation audits.

Codex issued the formal closure sign-off:
> **“最终裁决：本次正式重放验收通过；同意确立为 P1-A 的可信研究基线；同意在‘静态地址可识别性与检索算子边界’的限定范围内正式收口。”**

All artifacts in `results/run_p1a_rev2/` are now permanently frozen as the benchmark baseline for static address identifiability in the Temporal Attention Neuron (TAN) project.

---

## 2. Review Clarifications & Precision Audit Notes

During the final sign-off review, two precise methodological points were verified and recorded:

1. **Intermediate-Key Channel Query Denominator:**
   In Condition C2 ($N=3$, 10,000 trials $\times$ 2 channels = 20,000 channel queries), exactly **6,705** queries specifically targeted the intermediate (middle) key ($k_{\text{min}} < k_{\text{target}} < k_{\text{max}}$). The `SignedScalarKernel` model achieved exactly **0 hits out of 6,705 queries** (**0.00% hit rate**). (The preliminary dry-run text had referenced an approximate sample figure of 6,676; 6,705 is the exact unrounded integer denominator in the full dataset).

2. **Original Source Archive Cleanliness:**
   The original research repository (`C:\Users\李则徐\Downloads\TAN_Final_Submission\TAN_Final_Submission`) contains exactly **368 controlled non-cache files**, verified 100% bit-for-bit identical with zero missing, zero untracked, and zero modified files. The 39 pre-existing `__pycache__` files dated 2026-09-17 are deliberately excluded by cache-ignore rules.

---

## 3. Cryptographic Hash Manifest

### 3.1 Locked Source Code & Specification Files
All code was executed under strict SHA-256 hash verification matching Codex's execution authorization lock:

| File Name | SHA-256 Checksum | Size (Bytes) | Role |
|---|---|---|---|
| `run_p1a_experiment.py` | `256c8b777407e818c4c44ef7c8b0e10fceeabf498c661cc630a117e774f7bdb0` | 40,153 | Main orchestration & post-write audit |
| `task_generator.py` | `674cf9f49f5cd7f4975b87d29cd413e801fe40bd10fc56b966e663c9e6fc989d` | 11,881 | Synthetic trial generator & G1 suite |
| `models.py` | `3762cfb8627e7574dfbcbdc8db2470489471ad247832d03fa2dae916ce3f23e5` | 15,801 | 7 model implementations |
| `oracle_and_evaluator.py` | `13392549dcbddf5d6acb05721a57873714d2ca1a84e2ed712d1b8b5fe10d6fe3` | 18,349 | Ground truth evaluator & metrics |
| `REPLAY_PROTOCOL_AMENDMENT_1.md` | `7ececbc879c55c3ce0c9239f06a7baf922aec78c761e39e7718c3e3ff32ccd96` | 9,059 | Pre-registration protocol v2.0 |
| `THEORETICAL_PROPOSITION_1D_METRIC.md` | `19138da1a5b64eb70f9bc1e97b98c9481c0cbe16661d0c4aa86c9f79be20df94` | 7,234 | Formal propositions & soft bound |
| `test_audit_failure_modes.py` | `fdccdf1575bda77271a8234eb9472c90752cbee0aa34f8f878ba36e2f99d133f` | 6,691 | 5 audit failure-mode injection tests |

### 3.2 Certified Frozen Output Artifacts (`results/run_p1a_rev2/`)

| File Name | SHA-256 Checksum | Size (Bytes) | Contents & Description |
|---|---|---|---|
| `trial_inputs.csv.gz` | `5d3fa4fda5c36f9b1ac7c50f9784f58bfb74a6b5b5b659229abdab20ab7ae56e` | 6,318,566 | 60,000 raw unrounded float64 trial histories |
| `model_outputs.csv.gz` | `cf9767811a5fe5f4b2d91e37c3f6c25dbebb20a67125c00664e39efd9848adac` | 26,525,686 | 420,000 model evaluation inference rows |
| `per_seed_results.csv` | `9604fe33080e562a5a254a7843aa0798d03d607d9373626da58c6405d76c06df` | 94,796 | 420 rows (6 conditions $\times$ 7 models $\times$ 10 seeds) |
| `condition_summary.csv` | `77990921f04c3a3549ae56079fa0a1d753be29bb2dbffc4118f4820a215d58ea` | 4,197 | 42 rows aggregated condition statistics |
| `sample_trial_ledger.csv` | `3a86e87c397d1d4baff9124ab46990c963576a191e8f8ca46d59895483c462f2` | 58,659 | 420 sample rows for immediate human inspection |
| `g1_gate_report.json` | `cc56f0f2fbcdd1bbd649c2cd582a490b813e66ce6d4f543dffec104a804f4a60` | 1,813 | Complete G1 identifiability verification report |
| `post_write_recomputation_audit.json`| `a0c7f7f2cd894940ec450cd89c372ffc436a9105edf903ce50c162c358072c7d` | 558 | Lossless ledger recomputation audit certificate |
| `summary_p1a.json` | `3f043613c14a4d37526d883c1ac9aa4ff1bcaf2e0c0bdbd90c01c92bfc10d20c` | 43,426 | Full machine-readable summary & execution metadata |
| `P1A_SCIENTIFIC_REPORT.md` | `de71e4658ba77a3403d733420986b81cdcdebb983c233800b8b57c545a416f0c` | 8,969 | Data-driven scientific findings report |

---

## 4. Key Experimental Results Summary

### 4.1 Condition Performance Matrix

| Condition | Metric | QueryBlind | PosScalar | SignedScalar | Hist4.2B | FrozenS | ScalarMetric | VectorQK |
|---|---|---|---|---|---|---|---|---|
| **C1 ($N=2$, clean)** | Both-Routed Acc | 49.19% | 0.00% | 52.14% | 29.36% | 9.92% | **100.00%** | **100.00%** |
| | Hard Add Error | 0.0000 | 1.9982 | 0.9526 | 0.7957 | 1.6139 | **0.0000** | **0.0000** |
| **C2 ($N=3$, clean)** | Both-Routed Acc | 17.20% | 0.00% | 25.81% | 13.68% | 6.55% | **100.00%** | **100.00%** |
| | Middle Key Hit | 33.75% | 0.00% | **0.00% (0/6,705)** | 36.78% | 18.43% | **100.00%** | **100.00%** |
| | Soft Add Error | 2.1963 | 3.5186 | 2.6393 | 3.0134 | 3.2929 | **0.4247** | **0.9463** |
| **C3 ($\sigma=0.01$)** | Both-Routed Acc | 17.20% | 0.00% | 25.56% | 13.56% | 6.47% | **98.76%** | **98.76%** |
| | Middle Key Hit | 33.75% | 0.00% | 0.00% | 36.63% | 18.23% | **99.11%** | **99.11%** |
| **C4 ($\sigma=0.05$)** | Both-Routed Acc | 17.20% | 0.00% | 24.67% | 13.20% | 6.30% | **79.86%** | **79.86%** |
| | Middle Key Hit | 33.75% | 0.00% | 0.00% | 35.85% | 17.66% | **83.46%** | **83.46%** |
| **C5 (OOD Low)** | Both-Routed Acc | 17.20% | 0.00% | 0.00% | 0.00% | 3.56% | **100.00%** | **100.00%** |
| **C6 (OOD High)** | Both-Routed Acc | 17.20% | 0.00% | 0.00% | 10.74% | 3.79% | **100.00%** | **100.00%** |

### 4.2 Verified Core Scientific Facts
1. **Sufficiency of 1D Metric Compatibility:** Scalar metric attention (`score = -(q - k)^2`) achieves 100.00% both-routed accuracy on C1 and C2, with 100.00% middle-key hit rate. Hard argmax routing is mathematically identical to 2D Vector-QK on $S^1$, disproving the conjecture that 2D rotation is universally necessary for content addressing.
2. **Definitive Failure of Scalar Dot-Products on Intermediate Keys:** Signed scalar kernel achieves exactly 0/6,705 middle key hits (0.00%). A scalar multiplier cannot make an intermediate key the unique maximum.
3. **Historical 4.2-B Demarcation:** Sprint 4.2-B static control achieves only 13.68% accuracy on C2 (middle key hit rate 36.78%), confirming that historical winner flips were fixed counterfactual switches rather than semantic content addressing.
4. **Hard Routing vs. Soft Readout Distinction:** On C2 noiseless, while hard routing accuracy is 100.00% for both, soft addition error differs (ScalarMetric: 0.4247, VectorQK: 0.9463), bounded by kernel curvature and softmax temperature.

---

## 5. Audit Reproduction Instructions

To independently verify the integrity of the P1-A baseline:

```bash
# 1. Run all 5 failure-mode injection tests
python test_audit_failure_modes.py

# 2. Verify post-write recomputation audit certificate
python -c "import json; d = json.load(open('results/run_p1a_rev2/post_write_recomputation_audit.json')); print('Audit Passed:', d['audit_passed'])"

# 3. Verify bidirectional source integrity
python ../scripts/generate_manifest_after.py
```
