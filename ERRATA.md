# Scientific Errata and Technical Amendments (TAN Project)

**Review Baseline:** P0 Review (2026-09-17)  
**Review Directory:** `C:\Users\李则徐\Downloads\TAN_Review\P0_20260917_181757`  
**Scope:** Formal record of discrepancies, original statements, verified revisions, mathematical rationales, and affected artifact scopes.  
**Post-review corrections:** 2026-09-27 (new items ERR-17–ERR-20; status updates to ERR-01, 06, 07, 08, 10, 12, 15, 16 and AM1, AM2, AM4, AM5). The review directory above is the historical local path of the original review; this repository is the public copy. Paths beginning `paper/`, `paper2/`, `manuscript/`, `docs/`, `code/`, `results/` or `audit_v3_amended/` are inside `snapshot/`; `p1_experiments/`, `release/`, `s41cd/`, `s42c/` and top-level files are at the repository root. Manuscript bodies (`snapshot/paper/`, `snapshot/paper2/`, `snapshot/manuscript/`, `release/`) were deliberately not edited in the post-review pass; their remaining issues are recorded in ERR-18–ERR-20.

---

## Summary of Errata Items

| Errata ID | Topic | Severity | Affected Artifacts | Status |
|---|---|---|---|---|
| **ERR-01** | Query rotation double \omega multiplication | P0 BLOCKER | `paper2/sections/s2_framework.tex:211-212` (fixed), `docs/TANII_SPRINT4_2B_PREREGISTRATION.md:38` (frozen verbatim; superseded by :63) | RESOLVED |
| **ERR-02** | Phase 2 decision tree classification gap | P0 BLOCKER | `code/experiments/effective_dimension.py:984-1000`, `results/logs/effective_dimension.log` | RESOLVED |
| **ERR-03** | K mean mislabeled as median in Paper 2 | P0 BLOCKER | `paper2/sections/s3_state_geometry.tex:29` | RESOLVED |
| **ERR-04** | JSD 2-candidate toy example conflated with full model | P0 BLOCKER | `paper2/sections/s5_routing_geometry.tex:185` | RESOLVED |
| **ERR-05** | E3 maximum deviation aggregation and units | P0 BLOCKER | `paper2/sections/s5_routing_geometry.tex:118`, `results/sprint4_2/sprint4_2_summary.json` | RESOLVED |
| **ERR-06** | Probe-1 total vs conditional distractor sample sizes | P0 BLOCKER | `code/experiments/temporal_distractor.py`, `audit_v3_amended/POOLED_PROBE1_V3.md` | RESOLVED in top-level docs; frozen sources still say n=6000 (see note) |
| **ERR-07** | Composition conditional zero error vs unconditional error | P0 BLOCKER | `paper2/sections/s8_composition.tex:86-97`, `paper2/sections/s9_ladder.tex:90`, `results/sprint4_3/monte_carlo_results.json` | PARTIAL (see ERR-18) |
| **ERR-08** | Markovianity vs contraction claim scope | P0 BLOCKER | `paper/sections/methods.tex:37`, `paper2/sections/s3_state_geometry.tex:18` | PARTIAL (residue at `s3_state_geometry.tex:15,36-37`; see ERR-18) |
| **ERR-09** | Finite softmax support set and routing operationalization | P0 BLOCKER | `paper2/sections/s2_framework.tex:38` | RESOLVED |
| **ERR-10** | Scalar sorting theorem edge cases (S=0, ties) | P0 BLOCKER | `paper2/sections/s2_framework.tex:66`, `paper2/sections/s5_routing_geometry.tex:253`, `docs/TANII_SPRINT4_2_THEORY.md` | RESOLVED |
| **ERR-11** | Soft blending obstacle necessary conditions | P0 BLOCKER | `paper2/sections/s6_binding.tex` | RESOLVED |
| **ERR-12** | Single-winner head vs functional two-value computation | P0 BLOCKER | `paper2/sections/s7_wta.tex:111-121` | RESOLVED |
| **ERR-13** | Architectural ladder resource invariance claim | P0 BLOCKER | `paper2/sections/s9_ladder.tex:141` | RESOLVED |
| **ERR-14** | Softmax offset \delta=10^{-9} inheritance claim | P0 BLOCKER | `paper2/sections/s2_framework.tex:25` | RESOLVED |
| **ERR-15** | Documentation template dates (Sept 4 vs Sept 8–9) | P0 BLOCKER | `docs/TANII_SPRINT4_*.md` | DISCLOSED (frozen 2026-09-04 headers unchanged) |
| **ERR-16** | Manuscript submission readiness and visual QA caveat | P0 BLOCKER | `paper/main.tex`, `paper2/main.tex` | PARTIALLY_RESOLVED / DISCLOSED (NOT READY FOR SUBMISSION) |
| **ERR-17** | P1-A documents mixed superseded Run 1 figures with rev2 data | P1 MAJOR | `p1_experiments/P1A_SCIENTIFIC_REPORT.md`, `p1_experiments/P1A_FINAL_BASELINE_MANIFEST.md`, `p1_experiments/feedback_to_codex*.txt`, `MASTER_CLAIM_LEDGER.csv` (CLM-11), `CURRENT_RESEARCH_STATE.md`; hash-locked P1-A files | RESOLVED in editable documents; hash-locked residue DISCLOSED |
| **ERR-18** | Paper 2 (TAN-II) manuscript: numbers, equations and claims that do not match code or data | P0 BLOCKER (submission) | `paper2/main.tex`, `paper2/sections/s3,s4,s5,s6,s7,s8,s9,s10` | DISCLOSED (manuscript not edited; fix before any submission) |
| **ERR-19** | TAN-I paper and Stage 1 manuscript: numbers and claims that do not match data or code | P0 BLOCKER (submission) | `paper/sections/results.tex`, `manuscript/main.tex`, `manuscript/sections/{abstract,experiments,conclusion}.tex` | DISCLOSED (manuscripts not edited; fix before any submission) |
| **ERR-20** | Frozen snapshot documents and release manifest: stale inventories, superseded values, overclaiming wording | P1 MAJOR | `snapshot/README.md`, `docs/TANII_SPRINT4_*.md`, `results/sprint4_2/sprint4_2c_checks.csv`, `release/FINAL_MANUSCRIPT_RELEASE_MANIFEST.md`, `manifest_after.json` | DISCLOSED (frozen records not edited) |

---

## Detailed Errata Records

### ERR-01: Query Rotation Double \omega Multiplication
- **Original Formulation:**
  `the query is the rotating vector $Q_t=c\,S_t\,\hat u(\omega S_t)$ with $\hat u(s)=(\cos\omega s,\sin\omega s,0,\ldots)$` (`paper2/sections/s2_framework.tex:199`, `docs/TANII_SPRINT4_2B_PREREGISTRATION.md:38`).
- **Revised Formulation:**
  `the query is the rotating vector $Q_t=c\,S_t\,\hat u(\omega S_t)$ with $\hat u(\theta)=(\cos\theta,\sin\theta,0,\ldots)$` where $\theta = \omega S_t$.
- **Rationale & Verification:**
  Under the literal text formula, substituting $s = \omega S_t$ yields an angle $\theta_{\text{literal}} = \omega^2 S_t = 0.16 S_t$.
  - At $Q=3.0$ ($S_E = 1.8$): code angle $\theta = 0.72$ rad, literal angle $\theta = 0.288$ rad.
    - Code energy margin: $+0.2587427243$ (Winner: A)
    - Literal energy margin: $+0.7055409799$ (Winner: A)
  - At $Q=4.0$ ($S_E = 2.6$): code angle $\theta = 1.04$ rad, literal angle $\theta = 0.416$ rad.
    - Code energy margin: $-0.1559101334$ (Winner: B -> **FLIP**)
    - Literal energy margin: $+0.8425586282$ (Winner: A -> **NO FLIP**)
  The literal formula prevents the counterfactual winner flip at $Q=4$. The actual implementation in `sprint4_2_vector_qk.py` and line 63 of the preregistration used $\theta = \omega S$. The paper text has been corrected in the derived copy; the frozen code is correct.

---

#### ERR-02: Phase 2 Decision Gate Classification Gap
- **Original Formulation:**
  Outcome `D` in `results/logs/effective_dimension.log` annotated as:
  `"D": "No stable event-locked expansion for B4 -> check calibration, then pivot to anisotropy / curvature / context coupling"`
- **Revised Formulation:**
  Outcome `D` is reclassified as `UNCLASSIFIED_MIXED_CASE`. Note that the calibration threshold in `effective_dimension.py:112` is `MIN_LOGT_DIFF = 0.2` (not 0.5).
- **Rationale & Verification:**
  In `effective_dimension.py:967`, `stable` evaluates to `True` ($dD_{\text{delta}} = 1.088918 \ge 0.5$, $CI_{\text{lo}} = 1.873931 > 0$).
  However:
  - Condition A failed because $B4_{zC} (1.5891) < B1_{zC} (1.6740)$.
  - Condition B failed because $B4_{zF} (1.0889) > B3_{zF} (0.4835)$ (i.e. $\text{not}(B4 > B3)$ is `False`).
  - Condition C failed because $|B4_{zC} - C0_{zU}| = |1.589085 - 1.054950| = 0.534135 > 0.5$.
  Line 984 assigned all remaining cases to `D`. Thus, `outcome == D` occurred because the data fell into an unhandled classification combination, NOT because expansion was unstable.
  - Additionally: Event log-trace difference ($\log \mathrm{Tr}_E$, $z_F$: $3.577357 - 2.482894 = 1.094463$ nats) is distinguished from the difference-in-differences of event-minus-quiet changes ($1.354674$ nats). Both numbers are mathematically correct metrics of different quantities.

---

### ERR-03: K Mean Mislabeled as Median in Paper 2 Section 3
- **Original Formulation:**
  `median $K$ is $0.5$ \emph{exactly} for B1 ..., $4.47\times10^{3}$ [95\% CI $3.26$--$5.92\times10^{3}$] for B2, $1.235\times10^{4}$ [$7.78\times10^{3}$--$1.83\times10^{4}$] for B3, and $2.047\times10^{4}$ [$1.12$--$3.57\times10^{4}$] for B4` (`paper2/sections/s3_state_geometry.tex:29`).
- **Revised Formulation:**
  `mean $K$ is $0.5$ for B1, $4.47\times10^{3}$ [95\% CI $3.26$--$5.92\times10^{3}$] for B2, $1.235\times10^{4}$ [$7.78\times10^{3}$--$1.83\times10^{4}$] for B3, and $2.047\times10^{4}$ [$1.12$--$3.57\times10^{4}$] for B4 (corresponding medians: $0.5$, $895.5$, $1369.1$, and $2947.7$)`. Membrane leak parameter in Phase 1 is verified as $\lambda = 0.5$ (`natural_collision.py:95`).
- **Rationale & Verification:**
  The values reported in `natural_collision_summary.csv` are:
  - B1 (LIF): `K_mean = 0.500000`, `K_median = 0.500000`
  - B2 (Buffer): `K_mean = 4469.536489`, `K_median = 895.529570`
  - B3 (TAN-noAttn): `K_mean = 12347.886007`, `K_median = 1369.079508`
  - B4 (Full TAN): `K_mean = 20465.855578`, `K_median = 2947.724430`
  The LaTeX source copied `K_mean` but labeled it as `median`. Both statistics are now explicitly reported.

---

### ERR-04: Conflation of 2-Candidate Toy JSD with Full-Window Model JSD
- **Original Formulation:**
  `The scalar trap re-demonstrates inside the neuron (JSD$=0.0751$ with flip $0$...; the vector model satisfies JSD$>0$ \emph{and} winner flip (flip case JSD$=0.0647$).` (`paper2/sections/s5_routing_geometry.tex:185`).
- **Revised Formulation:**
  `In a 2-candidate geometric toy example, the scalar trap exhibits JSD$=0.0751$ with zero flip, while the vector model exhibits JSD$=0.0647$ with a winner flip. In the canonical 5-slot full-window model ($[1, 0, 2, 0, Q]$), the full-window distribution shifts yield M0 JSD$=0.009462$ and M2 JSD$=0.008727$.`
- **Rationale & Verification:**
  $0.0751$ and $0.0647$ are JSDs calculated on an isolated 2-event probability vector. The canonical full-window model M0 and M2 in `sprint4_2b_summary.json` record JSDs of $0.009462259$ and $0.008726529$ across all 5 slots. The revision clearly distinguishes the two settings.

---

### ERR-05: Maximum Deviation Metric and Units in E3 Analysis
- **Original Formulation:**
  `maximum deviation $\approx0.004$` (`paper2/sections/s5_routing_geometry.tex:118`).
- **Revised Formulation:**
  `maximum deviation across 3-seed averages is $0.004667 \approx 0.005$ (single-seed maximum deviation $0.009267 \approx 0.009$) across 140 checks (132 isotropic conditions + 8 control checks); machine-readable grid variable \texttt{theta\_deg} records $\theta/\pi$, not degrees.`
- **Rationale & Verification:**
  In `sprint4_2_summary.json` (E3 checks):
  - Total checks = 140 (132 isotropic $+ 8$ controls).
  - Maximum absolute difference across all individual seed checks: $0.009266667$.
  - Maximum absolute difference after averaging across the 3 seeds: $0.004666667$.
  The paper rounded $0.004667$ down to $0.004$ without stating the aggregation unit. Additionally, `theta_deg` values ($0.0833, 0.1667, \ldots$) represent $\theta/\pi$.

---

### ERR-06: Sample Size Stratification in Probe-1
- **Original Formulation:**
  Presenting $N=6,000$ trials without differentiating the distractor subpopulation.
- **Revised Formulation:**
  Total test set comprises $6,000$ trials ($2,000$ per seed $\times 3$ seeds). Generative derivation from `temporal_distractor.py` logic: out of $2,000$ trials/seed, $1/3$ are catch trials ($n_{\text{catch}} = 667$), $n_{\text{present}} = 1333$; distractor present on $n_{\text{dist}} = 666$ and distractor catch on $n_{\text{catch\_dist}} = 333$. Thus, distractor condition contains $666 + 333 = 999$ trials/seed, yielding exactly $2,997$ distractor trials across 3 seeds. Bootstrap resampling samples from the $6,000$ test set then filters for the distractor condition.
- **Status note (2026-09-27):** Top-level documents now describe the contrast as scored on the $2,997$ distractor trials. Frozen sources still say "n = 6000" (meaning the pooled trials that are resampled, not the scored subpopulation): `audit_v3_amended/POOLED_PROBE1_V3.md:12`, `paper/sections/results.tex:76`, `docs/REPRODUCIBILITY_LEDGER.md:59`, `docs/TAN_RESEARCH_DOSSIER.md:220`, `snapshot/CHANGELOG.md:30`. Read each of these as "6,000 pooled trials, of which 2,997 are distractor trials".

---

### ERR-07: Composition Conditional Zero Error vs Unconditional Error
- **Original Formulation:**
  `conditioned on correct binding, the combiner error is exactly $0.0\mathrm{e}0$` (`paper2/sections/s8_composition.tex:87-88`; the maximum over $3\times10^4$ histories is claimed at `:96-97` and `paper2/sections/s9_ladder.tex:90`) without highlighting the both-routed coverage rate or distinguishing quadrature from MC means.
- **Revised Formulation:**
  Conditioned on correct dual routing (both channels routed to targets, $6,608 / 30,000 = 22.03\%$ coverage: $2,188 / 2,258 / 2,162$ per seed), combiner error is $0.0\mathrm{e}0$.
  Across the full ensemble of $30,000$ histories:
  - Numerical quadrature benchmarks (integration over continuous uniform density): $E_{\text{comp}}^+ = 1.116099621$, $E_{\text{comp}}^- = 2.000999500$.
  - Pooled 30,000 Monte Carlo empirical means: $E_{\text{comp}}^+ = 1.107671595$, $E_{\text{comp}}^- = 1.975892767$.
  Both metrics are explicitly reported and differentiated in Paper 2 Section 8 and Section 10 (Table 1).
- **Status note (2026-09-27):** Only partly carried through. `paper2/sections/s8_composition.tex:96-97` and `paper2/sections/s9_ladder.tex:90` still quote a maximum error without saying it is taken over the $6,608$ both-routed histories only; the same omission is in `docs/TANII_SPRINT4_3A_RESULTS.md:26,74` and `docs/TANII_SPRINT4_3A_CLAIM_LEDGER.md:18` (see ERR-18 item M5 and ERR-20).

---

### ERR-08: Non-Contractivity vs Non-Markovianity Claim Scope
- **Original Formulation:**
  `paper/sections/methods.tex:37` states that a Markovian hypothesis predicts contraction ($K \le 1$), and that $K > 1$ refutes Markovianity.
- **Revised Formulation:**
  Markovianity does not imply contraction ($h_{t+1} = 2 h_t$ is strictly Markovian and expansive). The valid finding is that the 1D membrane potential $h_t$ alone is an insufficient state descriptor for TAN; the complete state $(h_t, X_t)$ remains Markovian.
- **Status note (2026-09-27):** `paper2/sections/s3_state_geometry.tex:15` and `:36-37` still use "non-Markovianity" and "Markov-violation rate" for what is measured as non-contraction ($K > 1$) of the 1D membrane map. Read these as "insufficiency of $h_t$ as a state descriptor" (see ERR-18).

---

### ERR-09: Softmax Support Set and Operational Primitive Definitions
- **Original Formulation:**
  Inferring content routing from changes in the candidate support set.
- **Revised Formulation:**
  Finite Boltzmann softmax assigns strictly positive probability to all finite logits (full support). Support set changes cannot define routing. Routing is defined by rank reordering, winner flips, and value delivery (citing Martins & Astudillo 2016 for sparse alternatives).

---

### ERR-10: Scalar Sorting Theorem Edge Cases (S=0 and Ties)
- **Original Formulation:**
  Scalar sorting theorem asserted universal rank-order invariance without explicit degenerate handling.
- **Revised Formulation:**
  When surprise $S=0$, all logits collapse to zero, producing uniform attention (a degenerate tie). The rank-order invariance theorem strictly holds for all $S > 0$.

---

### ERR-11: Soft Blending Obstacle Necessary Conditions
- **Original Formulation:**
  Claiming finite softmax blending always incurs error.
- **Revised Formulation:**
  Soft blending error $D = |V_A - V_B| / (1 + \exp(\gamma \cdot \text{margin}))$ is strictly positive if and only if $V_A \neq V_B$. If $V_A = V_B$, blending error trivially vanishes regardless of $\gamma$. This numerical cancellation must not be mistaken for exact structural binding.

---

### ERR-12: Single-Winner Head vs Functional Two-Value Computation
- **Original Formulation:**
  Claiming a single attention head cannot compute any function of two values.
- **Revised Formulation:**
  A single attention head with a single winner cannot simultaneously deliver two distinct bound operands to downstream modules. However, it can compute specific symmetric functions (e.g. uniform pooling multiplied by 2 computes $V_A + V_B$). Mechanism operand delivery is distinguished from functional arithmetic.

---

### ERR-13: Architectural Ladder Resource Invariance Claim
- **Original Formulation:**
  `paper2/sections/s9_ladder.tex:141` claimed that the architectural ladder added no state space or non-linearities because window size $W=5$ was held constant.
- **Revised Formulation:**
  While $W=5$ remains fixed, the ladder introduces vector embedding dimensions ($d=2$), E-I opponent units ($h_E, h_I$), and dual attention channels, which represent additional physical and computational degrees of freedom.

---

### ERR-14: Softmax Numerical Conventions Across Sprints
- **Original Formulation:**
  `paper2/sections/s2_framework.tex:25` claimed all sprints inherited a denominator offset $\delta=10^{-9}$.
- **Revised Formulation:**
  Sprint 4.2-B uses standard max-subtracted softmax without offset ($\sum \alpha_i = 1.0$ exactly); `tan.py` adds $10^{-9}$ in the denominator after max-subtraction. The text has been corrected to reflect actual sprint implementations.

---

### ERR-15: Documentation Template Dates (Sept 4 vs Sept 8–9)
- **Clarification:**
  TAN-II documents self-dated `2026-09-04` were created and executed on `2026-09-08` and `2026-09-09`. This was caused by template copying from the TAN-I archive milestone. Times lacking primary execution timestamps are marked as `UNKNOWN`.

---

### ERR-16: Manuscript Submission Readiness and Visual QA Caveat
- **Clarification:**
  Both `paper/` and `paper2/` drafts contain author correspondence placeholders (`[correspondence email pending submission]`). Under current environment constraints, automated visual inspection of rendered PDF pages (`VISUAL_QA_NOT_COMPLETED`) is unavailable. The papers are NOT READY FOR SUBMISSION.

---

### ERR-17: P1-A Documents Mixed Superseded Run 1 Figures with rev2 Data
- **Issue:** P1-A was run twice. Run 1 (6 models, 360,000 evaluations) is archived in `p1_experiments/archive_run1/`; rev2 (7 models, 420,000 evaluations, `p1_experiments/results/run_p1a_rev2/`) is authoritative. Several top-level P1-A documents quoted Run 1 numbers, or numbers that match no data file, next to rev2 numbers.
- **Corrected on 2026-09-27 (checked against `run_p1a_rev2/condition_summary.csv` and `summary_p1a.json`):**
  - `p1_experiments/P1A_SCIENTIFIC_REPORT.md` is now labelled as the superseded Run 1 report. Its prose contradicted its own table (Run 1 and rev2 agree for these models) and has been corrected: signed scalar on N=2 is $52.14\%$ both-routed (not $100\%$; the sign threshold is fixed at $q = 1.4$, so both channels succeed only when the two keys straddle $1.4$); signed scalar on N=3 (C2) is $25.81\%$ (not $27.64\%$); FrozenS on C2 is $6.55\%$ (not $16.66\%$), below the query-blind level of $17.20\%$.
  - `p1_experiments/P1A_FINAL_BASELINE_MANIFEST.md`: the C2 soft-add-error row now reads $1.1685 / 2.4149 / 1.4593 / 1.2760 / 1.3203 / 0.4247 / 0.9463$, and every C3–C6 cell now matches `condition_summary.csv`. C3 and C4 share histories, so query-independent models give identical C3 and C4 values.
  - Middle-key query count: $6,705$ everywhere (`feedback_to_codex_final_closure.txt` said $6,676$, which matches no data file). The G1 leakage figure is $172/1000 = 17.20\%$ (95% CI $14.99$–$19.66\%$, threshold $21.67\%$), not $16.71\%$.
  - `p1_experiments/feedback_to_codex.txt`: "中间键正确率" (middle-key accuracy) is relabelled as the joint accuracy on trials whose target set contains the middle key; the per-channel middle-key hit rates are now given as well (QueryBlind $33.75\%$, FrozenS $18.43\%$, Metric/VectorQK $99.11\%$ at $\sigma = 0.01$ and $83.46\%$ at $\sigma = 0.05$).
  - CLM-11 (generator and `MASTER_CLAIM_LEDGER.csv`): the noise $\sigma$ is relative (query-noise SD $= 2.2\sigma$); $100\%$ holds only in the zero-noise conditions; Metric and VectorQK make identical routing decisions, and under noise both reach $98.76\%$ (C3) and $79.86\%$ (C4), not $100\%$.
  - `CURRENT_RESEARCH_STATE.md:72-73`: peak RAM is $34.8\%$ of the budget ($711.71 / 2,048$ MB; was $34.7\%$), and the zero-noise routing error is reported as a maximum of $1.78\times10^{-15}$ (machine precision) rather than "zero".
- **Residue in hash-locked files (not edited, because doing so would break the P1-A replay hashes):**
  - `p1_experiments/REPLAY_PROTOCOL_AMENDMENT_1.md:13,106` says "5" items but lists 6.
  - `p1_experiments/task_generator.py:109-111`: the configured $\sigma$ is relative; the query-noise SD actually applied is $2.2\sigma$.
  - `p1_experiments/models.py:174`: the docstring gives the signed score as $\mathrm{sign}(q-1.4)\,c\,S\,K_i$, but the code at `:185` computes $\mathrm{sign}_q \cdot K_i$ (no $c\,S$ factor).
  - `p1_experiments/results/run_p1a_rev2/P1A_SCIENTIFIC_REPORT.md:30` says "Proves" for an empirical result on one synthetic task.
  - `p1_experiments/archive_run1/` keeps the old Run 1 numbers by design.
  - `p1_experiments/results/condition_summary.csv`, `per_seed_results.csv` and `summary_p1a.json` (top level of `results/`) are Run 1 outputs (6 models, 360,000 evaluations). Use `results/run_p1a_rev2/`.
- **Metric kernel:** the P1-A code uses $-(q-k)^2$ (`p1_experiments/models.py:309`). The form $-|q-k|$ appears only in rejected P1-B drafts (`p1_experiments/p1b/codex_response_round1.txt`, `p1_experiments/p1b/models_p1b.py`, `P1B_PRE_REGISTRATION_PROTOCOL.md` and `_V2.md`) and in `release/FINAL_MANUSCRIPT_RELEASE_MANIFEST.md:49` (see ERR-20).

---

### ERR-18: Paper 2 (TAN-II) Manuscript Issues (Not Edited)
The manuscript body was not edited in the post-review pass. Each item below must be fixed before any submission.
- **Abstract (`paper2/main.tex:73`):** "N=3, 60,000 trials": 10,000 of the 60,000 trials are N=2 (C1). The $100\%$ result needs a zero-noise qualifier ($98.76\%$ at $\sigma = 0.01$, $79.86\%$ at $\sigma = 0.05$). "proves" should be "shows" for this empirical result.
- **`s5_routing_geometry.tex:236`:** "60,000 histories": 60,000 trials over 50,000 distinct histories (C3 and C4 share histories).
- **`s5_routing_geometry.tex:265` (figure caption):** $\sigma$ is given without saying that the applied query-noise SD is $2.2\sigma$.
- **`s5_routing_geometry.tex:271`:** "proves"; $100\%$ holds only in C1, C2, C5 and C6.
- **M1, equation vs code (`s8_composition.tex:164-173`, eq. 9 and the decoder):** The implemented model differs from the equation. The gate $A$ is novelty-driven, $\max(0, |K|-1.4)$ clipped to $[0, 5]$; $r$ is a 2 ms refractory countdown; $u$ is a leaky integrate-and-fire membrane; the readout is $u$ at cue step 100; there is no $\sigma_A$ term. Sources: `p1_experiments/p1b/models_p1b_v2.py:84-104,150-151,260-269`; `P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md:35-52,114-117`.
- **M2, G1.5 (`s8_composition.tex:174-176`, `main.tex:74`, `s10_ledger.tex:55`):** G1.5 is a single constructed compensation pair ($K = 1.8$ vs $2.2$; $V_1 = -1.5$ vs $1.0$; $V_2' = -0.130767$; residual $2\times10^{-20}$). It shows that the subthreshold map is not injective; it does not show that information is erased in general. The closure report says so (`P1B_FINAL_SCIENTIFIC_CLOSURE_REPORT.md` §6.1, line 107). Genuine erasure is Proposition 6.1 (keys $K_1 \le 1.4$).
- **M3 (`s8_composition.tex:182-183`):** The E_bind criterion is $|\hat V - V| < 0.50$ (not $\le 0.05$). The TOST margin is $\pm 0.005$ (not $\pm 0.05$). M2a is the membrane-clamp control; the constant gate $A \equiv A_0$ is M2b/M2d.
- **M4 (`s8_composition.tex:186`, `main.tex:74`, `s10_ledger.tex:55`):** $[2.21, 2.57]$ is the envelope of the per-cell confidence intervals ($2.212$–$2.572$). The cell means of $E_{\text{comp}}$ range over $2.254$–$2.529$.
- **M5 (`s8_composition.tex:96-97`, `s9_ladder.tex:90`):** the maximum error is taken over the $6,608$ both-routed histories only, not all $30,000$ (ERR-07 remains partial).
- **M6 (`s8_composition.tex:73` and caption at `:86`):** "matching quadrature benchmarks within sampling variance" / "equals the routing-error benchmark": the pooled Monte Carlo $E^-_{\text{comp}} = 1.975893$ lies about two standard errors below the quadrature value $2.000999$ ($z = -2.02$), and all 3 seeds ($1.989$, $1.960$, $1.979$) are below it. The gap should be reported, not called a match.
- **M7 (`s6_binding.tex:97`, `s7_wta.tex:145`):** "$D_{\text{ev}} \approx 1.5$" should be $D_{\text{ev}}(3) = 1.6925$ at $\gamma \approx 1.198$ ($1.49$ is the $\gamma = 2$ value).
- **M8 (`s7_wta.tex:95-98`):** "restoring positive … for all $\gamma > \gamma^*$" is wrong in sign and scope. Above $\gamma^* = 5.098014585$ the E-I drive transfer is negative (value-aligned; $T_{\text{drive}} = +0.2367, +0.2516, +0.0157, -1.0541$ at $\gamma = 1, 2, 5, 10$), and only these few $\gamma$ values were sampled.
- **M9 (`s4_competition.tex:85`):** C1 curvature $-0.11$ should be $-0.0800$ (`results/sprint4_1cd/sprint4_1cd_summary.json`, C1 `p1_fp` `c2`).
- **M10 (`s10_ledger.tex:77-78`):** the seed list ($20260904$–$20260906$) omits P1-B. P1-B uses $2026091920 + \text{tid}$, $2026091930 + \text{tid}$ and $2026091999$ (`p1_experiments/p1b/runner_p1b.py:44-61`).
- **M11, wording:** "certifies" (`main.tex:74`, `s8_composition.tex:186`) should be "is consistent with" / "was internally audited". The generalisation from one single-unit model to all single-unit spiking decoders (`s8_composition.tex:186`) is not supported.
- **ERR-08 residue:** `s3_state_geometry.tex:15,36-37` (see ERR-08).
- **Figure script:** `paper2/scripts/make_p1a_figure.py` hard-codes the plotted values rather than reading them. The values match `run_p1a_rev2`, so this is a reproducibility note only.
- **Stale PDF:** `paper2/The Minimal Architectural Ladder for Relational.pdf` is an older 25-page build; the current build is `paper2/main.pdf` (28 pages).

---

### ERR-19: TAN-I Paper and Stage 1 Manuscript Issues (Not Edited)
The manuscript bodies were not edited in the post-review pass.
- **TAN-I `paper/sections/results.tex:17`:** the B3 CI is typeset as "$[7.78, 1.83]\times10^4$"; it should be $[0.778, 1.83]\times10^4$ (data: $7,781$–$18,327$).
- **TAN-I `paper/sections/results.tex:76`:** "n = 6000 pooled trials": the contrast is scored on $2,997$ distractor trials (see ERR-06).
- **Stage 1 `manuscript/sections/experiments.tex` (phototaxis):**
  - `:72` says the agent halts "near x = 80"; the data give $x = 86$ (error $6.0$), which is what the `:79` caption says.
  - `:89` says "lost far beyond x > 120"; the maximum in the data is $120$ (mean $117.84$).
  - `:91` gives the hovering band as $[80, 90]$, while `:98` and `:135` give $[80, 95]$; the recorded final positions range over $56$–$104$ (mean $90.6$).
  - Hovering cannot occur. The motor rule is $v \leftarrow 0.6\,v + 0.8\,\text{spike}$ (`code/experiments/baseline_comparison.py:306,398`), so $v \ge 0$ and $x$ never decreases. The agent stops; it does not hover. The E. coli run-and-tumble analogy (`experiments.tex:93`, `abstract.tex:7`, `conclusion.tex:48`) is unsupported.
- **`experiments.tex:123`:** CI $[11.347, 14.612]$ vs $[11.400, 14.629]$ in `results/tables/stats_summary.csv`.
- **`experiments.tex:128` (caption):** "lower error than LIF on all four tasks ($p < 10^{-30}$ …)" contradicts `:121`. There is no test for clean Task C, and Task A measures decay time, not error.
- **Baseline fairness:** The LIF threshold is hand-set (2.5 noisy / 2.0 clean; `baseline_comparison.py:374,377`), while TAN gets a 0.6 dead-zone (`:503`). The baselines are LIF, adaptive LIF and Elman-RNN/GRU imitators (hidden size 8, trained once on seed-0 TAN output). No other SNN or Transformer baseline was run.
- **Title (`manuscript/main.tex:68,93`):** "A Minimalist Fusion of Transformer Attention and Spiking Neural Dynamics" overstates the link: the model has one softmax attention step inside a single neuron, and no Transformer is implemented or used as a baseline.

---

### ERR-20: Frozen Snapshot Documents and Release Manifest (Not Edited)
These are frozen records. They were not edited, so they keep the discrepancies below.
- **`snapshot/README.md`:** it names `main.pdf`, but the file is `Temporal_Attention_Neuron_Manuscript.pdf`. Its inventory counts are stale: 7 sections (actual 8), 21 figures (27), 6 scripts (9 plus 3 subdirectories), 16 CSV / 1 NPZ (20 / 4).
- **`docs/TANII_SPRINT4_3A_RESULTS.md:26,74` and `docs/TANII_SPRINT4_3A_CLAIM_LEDGER.md:18`:** the maximum error is over the $6,608$ both-routed histories only (see ERR-07).
- **`docs/TANII_SPRINT4_2D_AMENDMENT_5.md:22` and `docs/TANII_SPRINT4_2D_CLAIM_LEDGER.md:21` (H7):** $D_{\text{events}}(3) \approx 1.5$ should be $1.69$ ($D_{\text{ev}}(3) = 1.6925$ at $\gamma \approx 1.198$; see ERR-18 M7).
- **`docs/TANII_SPRINT4_1CD_RESULTS.md:56`:** $\kappa = -0.11$ should be $-0.08$. At `:194`, fp(3.0) $0.56$ should be $0.53$.
- **`docs/TANII_SPRINT4_2_CLAIM_LEDGER.md:19`:** "MC 316 项一致" (316 Monte Carlo checks agree) is the total number of passing checks in `results/sprint4_2/sprint4_2_checks.csv` (316 of 317; the one failure is the AM1 check), not the number of $d$/$\theta$-grid Monte Carlo checks behind claim B.
- **`results/sprint4_2/sprint4_2c_checks.csv:11`** (and the copy in `s42c/`): the label reads $+0.0985$; the tested value is $+0.236700$ (see AM4).
- **`docs/TANII_SPRINT4_2B_PREREGISTRATION.md:38`:** double-$\omega$ formula kept verbatim (see ERR-01).
- **`release/FINAL_MANUSCRIPT_RELEASE_MANIFEST.md`:**
  - `:5` "Publication-Ready, Fully Audited, Visually Verified" contradicts ERR-16 (NOT READY FOR SUBMISSION; visual QA not completed).
  - `:45,47` "breakthrough" / "certified" should read "confirmatory" / "internally audited".
  - `:49` gives the metric as $-|q-k_i|$; the code uses $-(q-k_i)^2$. The FlipRate formula $\Delta q/\Delta\text{span}$ quoted there does not exist in the code.
  - `:56` calls G1.5 an "Analytical Collision Theorem" with "deterministic trajectory collapse … erasing history". G1.5 is one constructed non-injective pair (see ERR-18 M2).
  - `:58` $[2.21, 2.57]$ is a CI envelope, not a range of means (see ERR-18 M4), and "certifying" should read "consistent with".
- **`manifest_after.json`:** a historical record of the 2026-09-19 tree (586 entries). It no longer matches the repository (6 hash mismatches, 18 listed files absent). No script reads it.
- **Missing logs:** snapshot log files are git-ignored (`.gitignore:24`, `snapshot/**/*.log`) but some documents cite them. Those citations cannot be checked from a clone.

---

## Technical Amendments Record (AM1–AM6)

| Amendment | Scope | Description & Rationale | Verified Parameter / Value |
|---|---|---|---|
| **AM1** | Sprint 4.2-A | The pre-registered E4(iii) prediction $\mathrm{JSD} \approx 0.00103$ (asserted $> 5\times 10^{-4}$) was a hand-calculation error; the exact closed-form value is $0.000454$ with flip $= 0$. The check is kept as a logged FAIL (protocol fail, numerically correct), and the audit script exits 1 by design | $d=2, \theta=\pi/6$, $\mathrm{JSD} = 0.000454$ ($< 5\times 10^{-4}$) |
| **AM2** | Sprint 4.2-B | Unnormalized key control prediction error; correct expectation is zero flips. At $\omega = 0.4$ (all query angles in $(0, \pi/2)$), unnormalized keys $p(x) = (x, x^2)$ give $\mathrm{FlipRate} = 0.0$ exactly, so key normalization is the enabling condition in the tested query range | $k_i = \varphi(x_i)$, $\mathrm{FlipRate}=0.0$ exact |
| **AM3** | Sprint 4.2-B | Runtime correction of numerical fixed-point constants for M4/M5/M2 conditions, formed after canonical audit and before formal Monte Carlo execution | M4/M5/M2 numerical fixed-point constants |
| **AM4** | Sprint 4.2-C | Pre-run analytical drive-transfer value correction for M5-E (inhibitory branch value subtraction discovered at $\gamma=1$) | $T_{\text{drive}}$ corrected from $+0.0985$ to $+0.236700$; the frozen check label in `results/sprint4_2/sprint4_2c_checks.csv:11` still reads $+0.0985$ (the tested value is $+0.236700$) |
| **AM5** | Sprint 4.2-D | Identification of $D_{\text{full}}$ non-monotonicity and cancellation zero at $\gamma \approx 1.198$ due to silent slot background dilution (while event-normalized softness $D_{\text{ev}}(3) \approx 1.69$ at that $\gamma$ remains strictly positive) | $D_{\text{full}}=0$ cancellation root at $\gamma \approx 1.198$ |
| **AM6** | Sprint 4.3-A | Runtime precision correction for canonical routing margins (replaced hand-preregistered four-decimal values with exact closed-form values) | Canonical margins corrected from $0.2118 / 0.2637$ to exact $0.210631 / 0.261879$ |
| **$\gamma^*$ Definition** | Sprint 4.2-D | Definition of $\gamma^* = 5.098015$ as exact bisection root where net E-I drive transfer reverses ($T_{\text{drive}} = 0$) | $\gamma^* = 5.098014585$ (bisection tolerance $10^{-6}$); above it the E-I drive transfer is value-aligned (routing winners unchanged) |

