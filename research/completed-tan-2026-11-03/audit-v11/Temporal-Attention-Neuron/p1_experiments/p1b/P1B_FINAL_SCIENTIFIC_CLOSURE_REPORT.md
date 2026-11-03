# Phase P1-B Final Scientific Closure Report: Dynamical Carrier Investigation & Neural State Audit

- **Investigation Phase**: P1-B (Dynamical Carrier Investigation & Neural State Audit)
- **Governing Specification**: `P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md` (SHA-256: `07D58A2DE773EA13E8FD5C16901CB8E8A370219D4793F4A7F9BACA23C747A82B`)
- **Gate Approvals**: 
  - G1-Design: **APPROVED** by Codex (September 19, 2026)
  - G1-Executable: **APPROVED** by Codex (September 19, 2026)
- **Closure Review Date**: September 19, 2026
- **Auditing Agents**: Antigravity & Codex CLI (`gpt-6-astra-2`)

---

## 1. Executive Summary & Endorsed Scientific Determination

> **Endorsed Consensus Statement**:  
> Phase P1-B completed under frozen Protocol V2.6. M1 failed the baseline delivery prerequisite in every registered condition, rendering all M1–M2a comparisons floor-limited and leaving memory redundancy unestablished. Both specified spiking readouts met the registered criterion for demonstrated limitation of the tested end-to-end implementations. These findings establish model-specific insufficiency, not a universal biological impossibility result or a demonstrated requirement for population coding. Hypothesis H3 remains deferred.

---

## 2. Complete Cryptographic Provenance & Hash Table

All source artifacts, generated simulation outputs, and baseline repositories were independently audited and verified byte-for-byte in isolated replays:

### 2.1 Governing Source Artifacts (Frozen & Immutable)
| Source Artifact | SHA-256 Digest | Status | Role |
|---|---|---|---|
| `P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md` | `07D58A2DE773EA13E8FD5C16901CB8E8A370219D4793F4A7F9BACA23C747A82B` | Verified | Formal pre-registered specification |
| `models_p1b_v2.py` | `6B31FE0EF8B18ACDD1C7364879C18CC70C587F088841FA3AACF1F67B6771191F` | Verified | Production physical models and statistical methods |
| `runner_p1b.py` | `47609F1E70AB8DE2966EE8305757F44658EF799918D5F92108AC4EFE64304C26` | Verified | Confirmatory pipeline orchestrator |
| `test_p1b_g1_fixtures.py` | `8A56690F74EBD2E76DC39AB35B059DA9BEDA77A4671F59094993A12905D2FF23` | Verified | Authoritative 12-fixture validation suite |
| `task_generator.py` | `674CF9F49F5CD7F4975B87D29CD413E801FE40BD10FC56B966E663C9E6FC989D` | Verified | Upstream trial and task generator |

### 2.2 Generated Confirmatory Simulation Outputs
| Generated Artifact | Size | SHA-256 Digest |
|---|---|---|
| `summary_p1b.json` | 23,257 B | `C5A7D84C0D705453237D563B0F8FE54812C41505B7CCE0543C019A08868ABBE8` |
| `condition_summary.csv` | 3,327 B | `3BD40DD0C699EDC03DFAD29468C0402E42DB8F6DCAFF53049DBADA8E07A988B4` |
| `manifest.sha256` | 173 B | `08EC79B93C44150FC8E4251E15AC31DD3FF327D428D5849D0312009BEB16A667` |

- **Frozen Baseline Integrity**: `C:\Users\李则徐\Downloads\TAN_Final_Submission\TAN_Final_Submission` remains 100% bit-for-bit immutable (368 non-cache files verified).

---

## 3. Operational Performance & Resource Accounting

- **Confirmatory Scale**: 4,000 unique histories $\times$ 3 silence delays (0, 5, 10 ms) $\times$ 8 models = **96,000 model-episodes** (**192,000 role queries**).
- **Execution Wall Time**: 83.25 seconds (1.39 minutes).
- **CPU Resource Usage**: 80.34 process seconds (**0.022 CPU hours**, well below the registered 2.0 CPU hour ceiling).
- **Peak RAM / Working Set**: **144.5 MB** (well below the registered 2,000 MB ceiling).
- **Codex Independent Replay**: 151.84 wall seconds, 146.34 CPU seconds (0.041 CPU hours), 162.4 MiB peak working set; reproduced all JSON and CSV outputs byte-for-byte.

---

## 4. Methodological & Metric Definitions

To ensure complete reporting precision:
1. **Calibrated Constant Gate $A_0$**: $A_0 = 0.061886726$ is the frozen constant gate parameter used by constant-gate control arms M2b and M2d. Model M1 dynamically evolves from initial state $A(0) = 0.0$. The frozen affine readout parameters are $g_{\text{dec}} = 0.132630817$ and $b_{\text{dec}} = -0.026816484$.
2. **Abstention Reporting**: The reported abstention rates of $82.1\% - 83.0\%$ represent **episode-level abstention** (episodes where at least one role query abstained/silenced). Individual-query silence rates were $57.15\% - 58.60\%$.
3. **Composition Error ($E_{\text{comp}}$)**: Values reflect **unconditional composition errors** across all episodes (including silence fallback $0.0$), averaging addition error $|(v_A + v_B) - (t_A + t_B)|$ and subtraction error $|(v_A - v_B) - (t_A - t_B)|$. For episodes where both role queries spiked, conditional means were $2.9829 - 3.1685$ for Count and $2.5165 - 2.6207$ for Latency.

---

## 5. Confirmatory Experimental Findings

### 5.1 Condition-Specific Cell Evaluations & Decision Rule Enforcement
Under Protocol V2.6 Section 7.3, baseline prerequisite joint delivery accuracy floors were registered at **0.80** for $N=2$ and **0.40** for $N=3$.

| Condition Cell | $N$ | Delay | M1 Joint Acc | Baseline Floor | Floor Met? | M1 vs M2a TOST 90% CI | Registered Cell Decision | Memory Redundancy Inferred? |
|---|---|---|---|---|---|---|---|---|
| `N2_delay0ms` | 2 | 0.0 ms | 0.0290 (58/2000) | 0.80 | **No** | `[-0.001515, 0.001515]` | **`FLOOR_LIMITED_COMPARISON`** | **Not established** |
| `N2_delay5ms` | 2 | 5.0 ms | 0.0290 (58/2000) | 0.80 | **No** | `[-0.001515, 0.001515]` | **`FLOOR_LIMITED_COMPARISON`** | **Not established** |
| `N2_delay10ms` | 2 | 10.0 ms | 0.0290 (58/2000) | 0.80 | **No** | `[-0.001515, 0.001515]` | **`FLOOR_LIMITED_COMPARISON`** | **Not established** |
| `N3_delay0ms` | 3 | 0.0 ms | 0.0275 (55/2000) | 0.40 | **No** | `[-0.002978, 0.000841]` | **`FLOOR_LIMITED_COMPARISON`** | **Not established** |
| `N3_delay5ms` | 3 | 5.0 ms | 0.0280 (56/2000) | 0.40 | **No** | `[-0.002262, 0.001186]` | **`FLOOR_LIMITED_COMPARISON`** | **Not established** |
| `N3_delay10ms` | 3 | 10.0 ms | 0.0280 (56/2000) | 0.40 | **No** | `[-0.002262, 0.001186]` | **`FLOOR_LIMITED_COMPARISON`** | **Not established** |

**Interpretation**:
Because M1 joint delivery accuracy ($\sim 2.75\% - 2.90\%$) falls drastically below the condition floors, every single cell is registered as **`FLOOR_LIMITED_COMPARISON`** (`status_flag: M1_CARRIER_INSUFFICIENT`). Under Protocol §7.3–7.4, this does not authorize confirmatory equivalence of a functioning memory mechanism. Thus, memory redundancy is **not established**. The observed statistical equivalence is a floor artifact of mutual failure.

### 5.2 Spike Channel Evaluators (M4_Count and M4_Latency)
Evaluated via Pathway A transport with registered limitation threshold $E_{\text{comp}} \ge 0.05$:

| Condition Cell | Model | Unconditional Mean $E_{\text{comp}}$ | 95% Bootstrap CI (Exact JSON) | Registered Decision |
|---|---|---|---|---|
| $N=2$, 0 ms | M4_Count | 2.4521 | `[2.397318, 2.506076]` | **`DEMONSTRATED_CHANNEL_LIMITATION`** |
| $N=2$, 0 ms | M4_Latency | 2.2894 | `[2.245344, 2.332977]` | **`DEMONSTRATED_CHANNEL_LIMITATION`** |
| $N=2$, 5 ms | M4_Count | 2.4470 | `[2.392478, 2.500591]` | **`DEMONSTRATED_CHANNEL_LIMITATION`** |
| $N=2$, 5 ms | M4_Latency | 2.2870 | `[2.242402, 2.331462]` | **`DEMONSTRATED_CHANNEL_LIMITATION`** |
| $N=2$, 10 ms | M4_Count | 2.4453 | `[2.391100, 2.499126]` | **`DEMONSTRATED_CHANNEL_LIMITATION`** |
| $N=2$, 10 ms | M4_Latency | 2.2873 | `[2.241896, 2.331667]` | **`DEMONSTRATED_CHANNEL_LIMITATION`** |
| **$N=2$ Delay-Pooled** | **M4_Count** | **2.4482** | **`[2.393533, 2.502027]`** | **`DEMONSTRATED_CHANNEL_LIMITATION`** |
| **$N=2$ Delay-Pooled** | **M4_Latency** | **2.2879** | **`[2.243582, 2.332606]`** | **`DEMONSTRATED_CHANNEL_LIMITATION`** |
| $N=3$, 0 ms | M4_Count | 2.5286 | `[2.478280, 2.578247]` | **`DEMONSTRATED_CHANNEL_LIMITATION`** |
| $N=3$, 0 ms | M4_Latency | 2.2558 | `[2.212683, 2.297804]` | **`DEMONSTRATED_CHANNEL_LIMITATION`** |
| $N=3$, 5 ms | M4_Count | 2.5195 | `[2.470695, 2.569154]` | **`DEMONSTRATED_CHANNEL_LIMITATION`** |
| $N=3$, 5 ms | M4_Latency | 2.2544 | `[2.212309, 2.297139]` | **`DEMONSTRATED_CHANNEL_LIMITATION`** |
| $N=3$, 10 ms | M4_Count | 2.5188 | `[2.469123, 2.568488]` | **`DEMONSTRATED_CHANNEL_LIMITATION`** |
| $N=3$, 10 ms | M4_Latency | 2.2547 | `[2.212511, 2.296804]` | **`DEMONSTRATED_CHANNEL_LIMITATION`** |
| **$N=3$ Delay-Pooled** | **M4_Count** | **2.5223** | **`[2.473506, 2.571918]`** | **`DEMONSTRATED_CHANNEL_LIMITATION`** |
| **$N=3$ Delay-Pooled** | **M4_Latency** | **2.2550** | **`[2.212398, 2.296450]`** | **`DEMONSTRATED_CHANNEL_LIMITATION`** |

---

## 6. Scientific Bounds and Scope Limits

1. **Model-Specific Carrier Insufficiency**:
   The demonstrated delivery failure applies specifically to the registered M1 continuous carrier dynamics and frozen linear readout under Protocol V2.6. Subthreshold state collisions (fixture G1.5) prove information loss for this specific architecture, but do not constitute a universal mathematical impossibility proof for all 3-variable dynamical systems or all neural architectures.
2. **Channel Limitation Scope**:
   The `DEMONSTRATED_CHANNEL_LIMITATION` outcome applies strictly to the tested single-unit rate and latency point-process decoders operating on this specific carrier under Pathway A. Because the upstream analog carrier already fails its baseline delivery benchmark, these results demonstrate limitation of the tested end-to-end implementations, without isolating spike discretization as the exclusive bottleneck or demonstrating a universal requirement for population coding.
3. **Hypothesis H3**:
   Hypothesis H3 (Internal Composition Dynamics) remains formally deferred.

---

## 7. Formal Gate Status & Closure Conclusion

- **Gate G1-Design**: 100% APPROVED.
- **Gate G1-Executable**: 100% APPROVED.
- **Gate G2 (Confirmatory Simulation & Independent Audit)**: **PASSED & APPROVED**.
- **Phase P1-B**: Concluded with full cryptographic reproducibility and agreement between Antigravity and Codex CLI (internal AI-assisted review, not independent certification; wording corrected 2026-09-27).
