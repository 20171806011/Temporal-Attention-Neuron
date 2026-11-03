# Pre-execution design review: original-carrier stage localization

Prepared 2026-11-02 before any new data generation, fit or outcome inspection by this reviewer. The selected design uses the unchanged original P1B carrier and one fixed external diagnostic family. It does not continue the enriched interpolation series.

## Why a learned probe is useful here

The frozen equations already give an exact structural observation: during silence and query, I_content=0, so A does not enter the u/r recursion. With the same initial u,r and the same current q, changing only A leaves every future membrane candidate, refractory state, spike and native readout unchanged. This follows from `models_p1b_v2.py:78–107,200–267` and holds through threshold/reset branches, not only subthreshold linearization.

That fact alone does not show that useful target information was lost: original A is driven by keys/timing rather than values. It can nevertheless help interpret the value-dependent u coordinate. A limited, properly held-out decoder comparison is therefore more informative than an A-only intervention alone. The intervention remains a useful implementation check, not the primary recovery claim.

At the cue endpoint the registered final encoding ISI and subsequent silence are longer than the refractory period, so r is expected to be zero. Verify this on every generated case before omitting r from learned features; an unexpected nonzero value is a contract/implementation issue, not permission to add another feature after outcomes.

## One fixed learner, five prespecified views

Use total-degree-four polynomial ridge regression, with a shared target/solver specification and no degree, regularization or architecture search:

| External diagnostic | Features supplied | Polynomial terms including intercept |
|---|---|---:|
| Query-only negative control | q | 5 |
| Native-channel probe | exact registered u_readout, q | 15 |
| Pre-query membrane probe | u at cue, q | 15 |
| Full endpoint probe | u and A at cue, q | 35 |
| Shuffled-state negative control | donor-history cue u,A, current-history q | 35 |

The deadline feature must be the actual `u_readout` returned by the registered query function: pre-reset candidate when r at the deadline's left limit is zero, otherwise reset voltage. Substituting post-step membrane u would change the feature and confound the query/readout comparison. The two roles always start from the same cue checkpoint; their resulting states are not chained.

The native frozen affine decoder, a zero predictor, and an M0 raw-correct-history oracle accompany these learned views. The oracle explicitly has privileged key/value history and is a data/task positive control, not an endpoint-state decoder or free neural memory. No oracle outputs or targets enter learned input features.

The fixed regularizer is lambda=0.001 in mean squared error, with no intercept penalty. Fit raw-feature standardization and expanded-column standardization on training data only, recording both transformations and any constant-column handling. The fixed degree yields unequal numbers of active terms across views; this is a nested representation diagnostic, not an equal-total-capacity or architecture-optimality comparison. Polynomial extrapolation and the chosen regularization can limit performance. A negative result cannot establish absence of information.

Predetermined clipping to [-3,3] uses the already known target support and is legitimate as a decoder policy. It must be identical across learned views, applied without truth-dependent decisions, and accompanied by raw predictions, raw metrics and clipping counts. Do not select clipped versus raw conclusions after seeing results. Native labels and native predictions remain unchanged.

## Data separation and negative controls

The proposed allocation is, per N=2 and N=3, 1,024 training histories, 256 validation histories and 512 test histories: 3,584 unique histories total. Each history contributes both role queries at all three original delays. The complete six-row history block belongs to exactly one split. No same-history query, delay, endpoint, target or derived trajectory may cross splits.

The chosen seed contract is original-generator streams 2026110220 and2026110221, split seed2026110222, shuffle seed2026110223, and test paired-history bootstrap seed2026110224. Save the split assignments and full input ledger before model evaluation. Split by immutable generated-history ID, not by rows after expansion. The same exact train/validation/test histories serve every view.

Validation is descriptive and may catch implementation failures, but must not choose degree, lambda, clipping, features or a model. If scientific validation performance is poor, keep the prespecified model and report that limitation. If an actual code defect requires replay, preserve the failed record and distinguish correction from tuning.

Shuffle donor states within each N/split, using one whole-history donor mapping shared across all delays and both roles. A random cycle or other declared derangement avoids accidental self-donors. Current q and targets stay with the recipient history. Fit the shuffled-state model separately on shuffled training data, and use independently shuffled validation/test maps. This is a negative control for recoverable history association, not a clean causal intervention on A: shuffling also breaks the original q–state joint distribution. N/delay may define matching blocks for the control but must not enter learned features.

Forbidden learned features include true keys other than the current q, values, event positions/masks, target indices, history IDs, split IDs, delay labels, N labels, trajectories, second-role queries and any oracle-derived answer. Feature builders should accept only the declared numeric state/readout values plus current q. Evaluator metadata can still define reporting strata and reconstruction checks.

## Metrics and inference rules

The primary test contrast is full-endpoint diagnostic minus frozen-native Ecomp, after averaging the three delay-specific Ecomp values within each held-out history. Use the fixed 2,000-replicate paired-history bootstrap; resample whole histories, preserving both queries and all delays. Report every original N/delay cell, full-distribution results and the all-positive-novelty stratum, without refitting on that stratum. At 512 test histories per N, the all-positive subgroup may contain only roughly50–120 histories per N; its uncertainty and denominators must remain visible.

Secondary contrasts have narrower meanings:

- Native-channel probe versus frozen native: the learned external map plus direct q access can recover more from the already-produced channel statistic, if held-out performance improves. This does not isolate the affine function alone, because the external probe also receives q directly.
- Pre-query u probe versus native-channel probe: evidence that retaining pre-query u helps this fixed learner compared with the emitted deadline statistic. Differences can reflect noninjective query/reset transformations, conditioning and model-class approximation; they are not by themselves an information-theoretic loss theorem.
- Full u,A probe versus pre-query u probe: A-assisted recovery within this fixed supervised family. Since A can be correlated with keys/timing and degree-four interactions use that representation, a gain is not automatically a claim of conditional Shannon information or minimal state dimension.
- Query-only and shuffled-state controls: comparable gains in these controls would weaken a memory-specific interpretation. They are not licenses to retune the primary learner until the controls look favorable.

All-positive-novelty reporting removes the particular flat-input ambiguity identified in the previous audit, but it also conditions key amplitudes and query currents. It does not convert subgroup risk differences into a causal effect of novelty. The state probes cannot separate every encoding, retention and readout mechanism conclusively; a modest positive result is useful evidence of externally recoverable target signal, while failure remains absence of evidence for this learner.

The structural A-invisibility check can set A to0 and5 at a fixed held-out cue u,r and current q and compare the complete native query outputs. These are state interventions for equation validation; the altered states need not be reachable from the task generator. They do not become training examples or enlarge the model's query-time access. Equality supports the exact structural statement, not recovery sufficiency.

## Resource and stopping scope

These decoders require supervised data, learned parameters, training-only statistics, polynomial feature computation, numerical solve and direct endpoint/current-query access. Their success is an external diagnostic, not a learned neuronal implementation or a modification to original P1B. A full-state probe has a richer measurement contract than the registered scalar/spike decoder even though the carrier itself is unchanged.

Stop after the fixed fit, all validation/test tables, controls, clustered comparisons, exact structural check and independent ledger verification. Do not add hidden states, new readout samples, trajectories, parameter sweeps or a second decoder family. Preserve original confirmatory labels, previous negative findings and repository files. No new model result was produced in this design review.
