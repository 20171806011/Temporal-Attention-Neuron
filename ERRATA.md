# Scientific Errata and Technical Amendments (TAN Project)

**Review Baseline:** P0 Review (2026-09-17)  
**Review Directory:** `C:\Users\李则徐\Downloads\TAN_Review\P0_20260917_181757`  
**Scope:** Formal record of discrepancies, original statements, verified revisions, mathematical rationales, and affected artifact scopes.

---

## Summary of Errata Items

| Errata ID | Topic | Severity | Affected Artifacts | Status |
|---|---|---|---|---|
| **ERR-01** | Query rotation double \omega multiplication | P0 BLOCKER | `paper2/sections/s2_framework.tex:199`, `docs/TANII_SPRINT4_2B_PREREGISTRATION.md:38` | RESOLVED |
| **ERR-02** | Phase 2 decision tree classification gap | P0 BLOCKER | `code/experiments/effective_dimension.py:984-1000`, `results/logs/effective_dimension.log` | RESOLVED |
| **ERR-03** | K mean mislabeled as median in Paper 2 | P0 BLOCKER | `paper2/sections/s3_state_geometry.tex:29` | RESOLVED |
| **ERR-04** | JSD 2-candidate toy example conflated with full model | P0 BLOCKER | `paper2/sections/s5_routing_geometry.tex:185` | RESOLVED |
| **ERR-05** | E3 maximum deviation aggregation and units | P0 BLOCKER | `paper2/sections/s5_routing_geometry.tex:118`, `results/sprint4_2/sprint4_2_summary.json` | RESOLVED |
| **ERR-06** | Probe-1 total vs conditional distractor sample sizes | P0 BLOCKER | `code/experiments/temporal_distractor.py`, `audit_v3_amended/POOLED_PROBE1_V3.md` | RESOLVED |
| **ERR-07** | Composition conditional zero error vs unconditional error | P0 BLOCKER | `paper2/sections/s8_composition.tex:136`, `results/sprint4_3/monte_carlo_results.json` | RESOLVED |
| **ERR-08** | Markovianity vs contraction claim scope | P0 BLOCKER | `paper/sections/methods.tex:37`, `paper2/sections/s3_state_geometry.tex:18` | RESOLVED |
| **ERR-09** | Finite softmax support set and routing operationalization | P0 BLOCKER | `paper2/sections/s2_framework.tex:38` | RESOLVED |
| **ERR-10** | Scalar sorting theorem edge cases (S=0, ties) | P0 BLOCKER | `paper2/sections/s4_opponent.tex`, `docs/TANII_SPRINT4_2_THEORY.md` | RESOLVED |
| **ERR-11** | Soft blending obstacle necessary conditions | P0 BLOCKER | `paper2/sections/s6_binding.tex` | RESOLVED |
| **ERR-12** | Single-winner head vs functional two-value computation | P0 BLOCKER | `paper2/sections/s8_composition.tex` | RESOLVED |
| **ERR-13** | Architectural ladder resource invariance claim | P0 BLOCKER | `paper2/sections/s9_ladder.tex:141` | RESOLVED |
| **ERR-14** | Softmax offset \delta=10^{-9} inheritance claim | P0 BLOCKER | `paper2/sections/s2_framework.tex:25` | RESOLVED |
| **ERR-15** | Documentation template dates (Sept 4 vs Sept 8–9) | P0 BLOCKER | `docs/TANII_SPRINT4_*.md` | RESOLVED |
| **ERR-16** | Manuscript submission readiness and visual QA caveat | P0 BLOCKER | `paper/main.tex`, `paper2/main.tex` | RESOLVED |

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
  - Additionally: Event log-trace difference ($B4_{zF} - B3_{zF} = 1.094463$ nats) is distinguished from the difference-in-differences of event-minus-quiet changes ($1.354674$ nats). Both numbers are mathematically correct metrics of different quantities.

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

---

### ERR-07: Composition Conditional Zero Error vs Unconditional Error
- **Original Formulation:**
  `conditioned on correct binding, the combiner error is exactly $0.0\mathrm{e}0$` (`paper2/sections/s8_composition.tex:136`) without highlighting the both-routed coverage rate or distinguishing quadrature from MC means.
- **Revised Formulation:**
  Conditioned on correct dual routing (both channels routed to targets, $6,608 / 30,000 = 22.03\%$ coverage: $2,188 / 2,258 / 2,162$ per seed), combiner error is $0.0\mathrm{e}0$.
  Across the full ensemble of $30,000$ histories:
  - Numerical quadrature benchmarks (integration over continuous uniform density): $E_{\text{comp}}^+ = 1.116099621$, $E_{\text{comp}}^- = 2.000999500$.
  - Pooled 30,000 Monte Carlo empirical means: $E_{\text{comp}}^+ = 1.107671595$, $E_{\text{comp}}^- = 1.975892767$.
  Both metrics are explicitly reported and differentiated in Paper 2 Section 8 and Section 10 (Table 1).

---

### ERR-08: Non-Contractivity vs Non-Markovianity Claim Scope
- **Original Formulation:**
  `paper/sections/methods.tex:37` states that a Markovian hypothesis predicts contraction ($K \le 1$), and that $K > 1$ refutes Markovianity.
- **Revised Formulation:**
  Markovianity does not imply contraction ($h_{t+1} = 2 h_t$ is strictly Markovian and expansive). The valid finding is that the 1D membrane potential $h_t$ alone is an insufficient state descriptor for TAN; the complete state $(h_t, X_t)$ remains Markovian.

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
  Both `paper/` and `paper2/` drafts contain author correspondence placeholders (`[email placeholder --- see archive metadata]`). Under current environment constraints, automated visual inspection of rendered PDF pages (`VISUAL_QA_NOT_COMPLETED`) is unavailable. The papers are NOT READY FOR SUBMISSION.

---

## Technical Amendments Record (AM1–AM6)

| Amendment | Scope | Description & Rationale | Verified Parameter / Value |
|---|---|---|---|
| **AM1** | Sprint 4.2-A | E4(iii) analytical tolerance failure retained (test required $\mathrm{JSD} > 5\times 10^{-4}$ at $d=2, \theta=\pi/6$; actual simulated value $0.000454$ failed strict threshold; retained verbatim in code as intentional negative control) | $d=2, \theta=\pi/6$, $\mathrm{JSD} = 0.000454$ ($< 5\times 10^{-4}$) |
| **AM2** | Sprint 4.2-B | Unnormalized key control prediction error; correct expectation is zero flips (unnormalized keys without $S^1$ projection yield $\mathrm{FlipRate}=0.0$ exact; proves equal-norm normalization is indispensable mechanistic enabler) | $k_i = \varphi(x_i)$, $\mathrm{FlipRate}=0.0$ exact |
| **AM3** | Sprint 4.2-B | Runtime correction of numerical fixed-point constants for M4/M5/M2 conditions, formed after canonical audit and before formal Monte Carlo execution | M4/M5/M2 numerical fixed-point constants |
| **AM4** | Sprint 4.2-C | Pre-run analytical drive-transfer value correction for M5-E (inhibitory branch value subtraction discovered at $\gamma=1$) | $T_{\text{drive}}$ corrected from $+0.0985$ to $+0.236700$ |
| **AM5** | Sprint 4.2-D | Identification of $D_{\text{full}}$ non-monotonicity and cancellation zero at $\gamma \approx 1.198$ due to silent slot background dilution (while event-normalized softness $D_{\text{ev}} \approx 1.5$ remains strictly positive) | $D_{\text{full}}=0$ cancellation root at $\gamma \approx 1.198$ |
| **AM6** | Sprint 4.3-A | Runtime precision correction for canonical routing margins (replaced hand-preregistered four-decimal values with exact closed-form values) | Canonical margins corrected from $0.2118 / 0.2637$ to exact $0.210631 / 0.261879$ |
| **$\gamma^*$ Definition** | Sprint 4.2-D | Definition of $\gamma^* = 5.098015$ as exact bisection root where net E-I drive transfer reverses ($T_{\text{drive}} = 0$) | $\gamma^* = 5.09801533$ (bisection tolerance $10^{-6}$), restoring E-dominant routing |

