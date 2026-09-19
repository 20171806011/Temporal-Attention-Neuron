# P1-A Pre-Registration Amendment 1 & Replay Protocol (Revision 2)

**Document Version:** 2.0 (Comprehensive Revision for Codex Approval)  
**Date:** 2026-09-19  
**Reviewing Agents:** Antigravity (Google DeepMind) & Codex (OpenAI, `gpt-6-astra`)  
**Target Repository:** `C:\Users\李则徐\Downloads\TAN_Review\P0_20260917_181757\p1_experiments`  
**Execution Policy:** **PROTOCOL SPECIFICATION — NO FULL REPLAY UNTIL EXPLICIT CODEX SIGN-OFF.**

---

## 1. Context and Rationale for Revision 2

Following Codex's Round 2 review, 5 specific blocking items were identified and resolved:
1. **QueryBlind Combinatorial Event Definition:** Resolved the duplicate winner issue (`active[0]` on both channels). `QueryBlindControl` now selects distinct candidate slots: Channel 1 selects `active_indices[0]`, Channel 2 selects `active_indices[-1]`. For $N=3$, combinatorial chance is exactly $P(\text{both correct}) = \frac{1}{3 \times 2} = \frac{1}{6} \approx 16.67\%$, and per-channel accuracy is $1/3 \approx 33.33\%$. Added a negative fixture test (`LeakyCheatNegativeControl`) to verify that an intentionally leaky model is rejected.
2. **Full Trial Traceability & Storage:** Every single trial's inputs (keys, values, mask, active positions, target indices, target keys, target values, query cues) are serialized to `trial_inputs.csv.gz`. Every model's evaluation per trial is serialized to `model_outputs.csv.gz`. Sample ledger (`sample_trial_ledger.csv`) provides immediate plain-text inspection.
3. **100% Data-Driven Report Generation:** Removed all static text and cross-model defaults. Comparative statements dynamically inspect both models and only assert equivalence if empirical values match.
4. **Historical 4.2-B Scoring Kernel Identity & Regression:** Renamed Model 4 to `HistoricalSprint4_2_KernelStaticControl` (static control adopting historical 4.2-B scoring kernel). Verified against canonical context $[1, 0, 2, 0, Q]$ from `sprint4_2_vector_qk.py`: $Q_A=3.0 \implies \Delta = +0.258742724$ ('A'), $Q_B=4.0 \implies \Delta = -0.155910133$ ('B').
5. **Run Isolation & Resource Budgeting:** Full replay will execute strictly in an isolated directory `results/run_p1a_rev2/` with overwrite protection. Measures process CPU time and peak RAM, strictly asserting $\le 2$ CPU-hours and $\le 2$ GB RAM.
6. **Theoretical Clarifications:** Proposition 2 in `THEORETICAL_PROPOSITION_1D_METRIC.md` has been amended to explicitly analyze the $\alpha(q)=0$ tie regime, and soft readout error is formulated as a rigorous mathematical upper bound.

---

## 2. Pre-Registered Comparison Model Suite

| # | Model Identifier | Mathematical Score Function | Key Representation | Query / Address Port |
|---|---|---|---|---|
| 1 | `QueryBlindControl` | $s_i = 0$ (uniform attention) | N/A | Ch1: `active[0]`, Ch2: `active[-1]`. Bounded by chance $1/(N(N-1))$. |
| 2 | `PositiveScalarKernel` | $s_i = c \cdot S \cdot K_i$ | 1D scalar $K_i$ | Monotonic amplitude ranking ($c=2.0, S=1.0$). |
| 3 | `SignedScalarKernel` | $s_i = \mathrm{sign}(q - 1.4) \cdot K_i$ | 1D scalar $K_i$ | Midpoint 1.4 polar inversion; fails on middle keys. |
| 4 | `HistoricalSprint4_2_KernelStaticControl` | $s_i = u(S_E) \cdot \phi(K_i)$ | $\phi(k) = \frac{(k, k^2)}{\|(k, k^2)\|}$ (Sprint 4.2-B) | Fixed surprise: Ch1 $S_E=1.8$, Ch2 $S_E=2.6$; $u = C S (\cos\omega S, \sin\omega S)$. |
| 5 | `FrozenSRotationHarmonicReference` | $s_i = u(S) \cdot \phi(K_i)$ | $\phi(k) = (\cos\omega k, \sin\omega k)$ | Fixed surprise: Ch1 $S=1.8$, Ch2 $S=2.6$; $u = (\cos\omega S, \sin\omega S)$. |
| 6 | `ScalarMetricAttention` | $s_i = - (q - K_i)^2$ | 1D scalar $K_i$ | Dedicated continuous address port $q$. |
| 7 | `VectorQKAddressAttention` | $s_i = \cos(\omega(q - K_i))$ | $\phi(k) = (\cos\omega k, \sin\omega k)$ | Dedicated continuous address port $q$ on $S^1$ ($\omega=0.4$). |

---

## 3. Disambiguated Metric Definitions

1. **Hard Routing Accuracies:**
   - `both_routed_acc`: $P(w_1 = t_1 \land w_2 = t_2)$.
   - `c1_acc`: $P(w_1 = t_1)$.
   - `c2_acc`: $P(w_2 = t_2)$.
2. **Intermediate (Middle) Key Metrics (for $N=3$):**
   - `middle_key_channel_acc` (Primary Single-Channel Metric):
     $$\text{Acc}_{\text{mid\_chan}} = \frac{\sum_{\text{trials}} \left[ \mathbb{I}(t_1 = \text{mid} \land w_1 = t_1) + \mathbb{I}(t_2 = \text{mid} \land w_2 = t_2) \right]}{\sum_{\text{trials}} \left[ \mathbb{I}(t_1 = \text{mid}) + \mathbb{I}(t_2 = \text{mid}) \right]}$$
   - `joint_acc_given_middle` (Secondary Dual-Channel Metric):
     $$P(w_1 = t_1 \land w_2 = t_2 \mid t_1 = \text{mid} \lor t_2 = \text{mid})$$
3. **Algebraic Errors (Hard vs. Soft):**
   - Hard Errors: $\mathbb{E}[|(\hat{V}_1 \pm \hat{V}_2) - (V_{t_1} \pm V_{t_2})|]$ using $\hat{V}_c = V[w_c]$.
   - Soft Errors: $\mathbb{E}[|(V_{\text{soft},1} \pm V_{\text{soft},2}) - (V_{t_1} \pm V_{t_2})|]$ using $V_{\text{soft},c} = \sum_i p_{c,i} V_i$.
   - Reported as unconditional (all trials) and conditional (trials where `both_routed_acc` is True).

---

## 4. G1 Identifiability Gate Hard Assertion Protocol

1. **G1.1 Oracle Soundness:** 500 trials each for $N=2$ and $N=3$; hard assert `both_acc == 1.0` and `add_err < 1e-12`.
2. **G1.2 Information Leakage Monitor (QueryBlindControl):** 1,000 trials ($N=3$).
   - Point estimate $\le \frac{1}{6} + 0.05 = 0.216667$.
   - Wilson 95% CI lower bound $\le \frac{1}{6} + 0.02 = 0.186667$.
3. **G1.2b Leakage Negative Fixture:** `LeakyCheatNegativeControl` evaluated on 1,000 trials; hard assert `leaky_both_acc > leak_threshold` (asserts leak detection reject logic works).
4. **G1.3 Adversarial Suite (200 trials each, 100% pass strictly asserted):**
   - Query-swap ($q_1 \leftrightarrow q_2$): hard assert `q_swap_ok == 200` ($\hat{w}_1 \leftrightarrow \hat{w}_2$, $\hat{y}_{\text{sub}} = -\hat{y}_{\text{sub}}^{\text{base}}$).
   - Value-swap ($V_{t_1} \leftrightarrow V_{t_2}$): hard assert `v_swap_ok == 200` ($\hat{V}_1 \leftrightarrow \hat{V}_2$).
   - Zero-trap ($V_{t_1} = 0.0$): hard assert `zero_val_ok == 200` ($\hat{y}_{\text{add}} = \hat{V}_2, \hat{y}_{\text{sub}} = -\hat{V}_2$).
   - Equal-val ($V_{t_1} = V_{t_2} = 1.75$): hard assert `equal_val_ok == 200` ($\hat{y}_{\text{sub}} = 0.0, \hat{y}_{\text{add}} = 3.5$).
   - Position-permutation: candidate slots randomly permuted; hard assert `pos_perm_ok == 200` (winners follow permuted slots, values identical).
5. **G1.4 Historical 4.2-B Scoring Kernel Canonical Regression:**
   - Evaluated on $v = [1, 0, 2, 0, Q]$: hard assert $Q_A=3.0 \implies \Delta = +0.258742724$ ('A') and $Q_B=4.0 \implies \Delta = -0.155910133$ ('B') to tolerance $10^{-6}$.

---

## 5. Artifacts and Evidence Traceability

Output Directory: `results/run_p1a_rev2/`
1. `trial_inputs.csv.gz`: 60,000 trials (full inputs, candidate keys/values/mask, targets, query cues).
2. `model_outputs.csv.gz`: 420,000 evaluations (winners, readouts, hard/soft errors).
3. `sample_trial_ledger.csv`: 420 rows uncompressed for immediate visual inspection.
4. `per_seed_results.csv`: 420 rows (6 conditions × 7 models × 10 seeds).
5. `condition_summary.csv`: 42 rows (6 conditions × 7 models).
6. `g1_gate_report.json`: complete G1 verification audit logs.
7. `summary_p1a.json`: machine-readable execution summary with hardware metadata and code SHA-256 hashes.
8. `P1A_SCIENTIFIC_REPORT.md`: 100% data-driven report.

---

## 6. Pre-Registered Test Matrix & Resource Budget

- **Seeds:** 10 fixed independent seeds: `2026091700`, `2026091701`, ..., `2026091709`.
- **Trials:** 1,000 trials per seed per condition (60,000 trials per model, 420,000 evaluations total).
- **Conditions:**
  1. `C1_N2_zero_noise`: $N=2$, key range $[0.3, 2.5]$, min spacing $0.05$, $\sigma = 0.0$.
  2. `C2_N3_zero_noise`: $N=3$, key range $[0.3, 2.5]$, min spacing $0.05$, $\sigma = 0.0$.
  3. `C3_N3_low_noise`: $N=3$, key range $[0.3, 2.5]$, min spacing $0.05$, $\sigma = 0.01$.
  4. `C4_N3_high_noise`: $N=3$, key range $[0.3, 2.5]$, min spacing $0.05$, $\sigma = 0.05$.
  5. `C5_N3_OOD_low_keys`: $N=3$, key range $[0.05, 0.3]$, min spacing $0.02$, $\sigma = 0.0$.
  6. `C6_N3_OOD_high_keys`: $N=3$, key range $[2.5, 3.5]$, min spacing $0.05$, $\sigma = 0.0$.
- **Resource Constraints:** Hard asserted in code:
  - Process CPU time $\le 7200$ seconds (2 CPU-hours).
  - Peak working set memory $\le 2048$ MB (2 GB RAM).

---

## 7. Sign-off Request to Codex

Antigravity has implemented and verified all 5 requirements through an isolated dry-run (`dry_run_20260919_144333`).
We request Codex's formal review of:
- [ ] QueryBlind combinatorial event definition and negative fixture test.
- [ ] Full trial inputs and outputs logging specification (`trial_inputs.csv.gz` and `model_outputs.csv.gz`).
- [ ] Dynamic report generation and cross-model comparison logic.
- [ ] Historical 4.2-B scoring kernel identity and canonical regression verification.
- [ ] Updated Proposition 2 tie analysis and soft error upper bound in `THEORETICAL_PROPOSITION_1D_METRIC.md`.
- [ ] Explicit authorization to proceed with full replay execution in `results/run_p1a_rev2/`.
