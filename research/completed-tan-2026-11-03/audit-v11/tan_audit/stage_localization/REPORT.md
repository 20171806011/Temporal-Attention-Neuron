# Original-carrier stage localization and manuscript corrections

2026-11-02. Frozen repository commit `85213f0d417d54e4f48b301ed1c0e4200ef3c421`. This continuation preserves all previous experiments, repository bytes and historical P1B labels.

The original carrier retains some target-predictive information that the registered readout does not exploit as effectively as the fixed external probe. On 1,024 held-out histories, the full endpoint probe reduces mean composition error from 1.977919 to 1.890126; query-only and shuffled-state controls remain near 1.977. This is a modest predictive gain, not successful relational memory: strict joint operand accuracy is only 4.13%, and the experiment does not establish query-specific key binding. The separate manuscript patches correct source-contract mismatches and narrow interpretive overclaims without changing historical results.

## Completed documentation deliverable

[CORRECTION_DRAFT.md](CORRECTION_DRAFT.md) gives source-cited errors, proposed text and provenance. [manuscript_corrections.patch](manuscript_corrections.patch) corrects Paper2 section 8 and its matching abstract clause: the actual hybrid Euler equations, old-A membrane current, refractory-countdown meaning of r, deadline candidate readout, positive-novelty G1.5 fixture scope, dynamic-gate M2a reset intervention, strict operand tolerance .50, TOST margin .005, and M4's same-carrier spike readout/interval criterion. Recorded counts, calibration, intervals and scientific decision labels are preserved.

[readme_scope_corrections.patch](readme_scope_corrections.patch) is separate and interpretive: the tested sufficient mechanisms and model-specific failures do not establish general distributed-coding necessity. Neither patch is applied. Both passed `git apply --check --whitespace=error-all`; an isolated replacement-paragraph LaTeX build passed. A full manuscript build/layout review was not performed. [PATCH_VALIDATION.json](PATCH_VALIDATION.json), [PATCH_MANIFEST.json](PATCH_MANIFEST.json) and the separate [patch_review.md](patch_review.md) record these checks.

## Frozen experiment and access contract

[PREREGISTRATION.md](PREREGISTRATION.md) was saved before new inputs or outcomes; SHA256 `8ffcda523dfc042f9d64473c361ecc39c6f69b1f9d2b2af7994bbd35dea6c1ff`. No architecture, parameter, noise or feature search occurred. [DESIGN_REVIEW.md](DESIGN_REVIEW.md) explains why the learned diagnostic adds useful evidence beyond an A-only intervention.

The original clean generator produced 1,792 histories each for N=2 and N=3, using continuous RandomState streams 2026110220/2026110221. Split seed 2026110222 assigns, within each N, 1,024 training, 256 validation and 512 test histories. Total: 3,584 histories; 2,048/512/1,024 in train/validation/test. Both role queries and all three delays (0, 5, 10 ms) stay together. The 21,504 query rows form 10,752 paired rows; the test has 3,072 pairs but only 1,024 independent histories. No exclusions occurred. Validation was reported without selection/refitting; test outcomes did not alter the frozen models.

M1 encoding, delay, query dynamics and historical calibration are unchanged. The source is [models_p1b_v2.py](../../Temporal-Attention-Neuron/p1_experiments/p1b/models_p1b_v2.py), SHA256 `6b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f`. Calibration remains g=0.1326308169898006, b=-0.02681648355488859, A0=0.06188672587424413; M1 retains its registered initialization. Queries start from independent copies of each delayed checkpoint. All r at cue onset are zero, so omitting this constant feature does not omit a varying state.

Each learned view uses total-degree-four polynomial ridge regression, lambda=.001, with an unpenalized intercept. Raw and polynomial scaling use training-only moments. Deterministic linear solves fit one scalar predictor per view across both N and roles; no N, delay, role, true keys/values, event positions, history ID or trajectory enters the features. The external learner sees training labels, direct q and the declared state/readout values. Predictions use the preregistered support clip [-3,3], with raw predictions and clip counts retained.

| View | Accessible features | Fitted coefficients |
|---|---|---:|
| QUERY_ONLY | Current q | 5 |
| NATIVE_OUTPUT_Q | Exact returned deadline u_readout, q | 15 |
| CUE_U_Q | Cue-onset u, q | 15 |
| CUE_U_A_Q | Cue-onset u, A, q | 35 |
| SHUFFLED_STATE_Q | Another history's u,A at the same delay; recipient q | 35 |

Shuffle seed 2026110223 constructs one no-self-donor cycle per N/split, reused for both roles and all delays. It never crosses splits. This control is fit separately on shuffled training features. The zero baseline has no input. The correct-history oracle uses M0 nearest-key lookup over the actual raw records; this is privileged evaluator memory, never a learned input or a neural resource claim.

Input SHA256: `3233ad20c7cc95c7581a9756fedb1a5669409a1fe9be5968dd341d7e8ea693ed`. Fitted-model SHA256, saved before validation/test prediction: `6c51d7552cfa8554fe9db4fbdfbdedd0f47c6d3bfb4a93bcf18d82acc36cb660`.

## Held-out outcomes

Ecomp is the original algebraic composition error, equal to max(|eA|,|eB|). The accuracy column uses original strict operand error <.5 for both roles; it does not create new P1B pass/closure labels. Each history is equally weighted; the table averages its three delays. The all-positive subset conditions on every active key having positive novelty and is not a different fitted model or a causal novelty intervention.

| Arm | Full Ecomp, 1,024 histories | Full joint accuracy | All-positive Ecomp, 173 histories | All-positive joint accuracy |
|---|---:|---:|---:|---:|
| Frozen native | 1.977919 | 3.2552% | 1.974595 | 4.6243% |
| Query only | 1.976712 | 3.0273% | 1.965334 | 4.6243% |
| Native output + q | 1.953903 | 3.1576% | 1.912613 | 3.8536% |
| Cue u + q | 1.931632 | 3.2878% | 1.880722 | 4.0462% |
| Cue u,A + q | 1.890126 | 4.1341% | 1.780266 | 5.9730% |
| Shuffled state + q | 1.977327 | 3.1250% | 1.956032 | 4.8170% |
| Zero | 1.978480 | 3.6133% | 1.970829 | 4.6243% |
| Correct-history oracle | 0 | 100% | 0 | 100% |

The primary full-state-minus-native mean difference is -0.087793; paired history-bootstrap percentile interval [-0.111071,-0.063862]. For the all-positive stratum it is -0.194330 [-0.270234,-0.119664]. These intervals condition on the fitted training sample and use 2,000 replicates, resampling within N after averaging delays within each history. Seed 2026110224 is fixed. There is no correction for multiple comparisons or claim that the experiment is the original confirmatory run.

All prespecified contrasts are retained:

| Left minus right | Full mean [95% percentile interval] | All-positive mean [95% percentile interval] |
|---|---|---|
| Full endpoint − native | -.087793 [-.111071,-.063862] | -.194330 [-.270234,-.119664] |
| Native-output probe − native | -.024017 [-.037319,-.010607] | -.061982 [-.104686,-.020054] |
| Cue-u probe − native-output probe | -.022270 [-.034587,-.009540] | -.031891 [-.069727,.005952] |
| Full endpoint − cue-u probe | -.041506 [-.060361,-.023832] | -.100457 [-.153150,-.049715] |
| Shuffled state − full endpoint | .087201 [.063371,.110623] | .175766 [.100647,.255408] |
| Query only − full endpoint | .086586 [.063003,.109139] | .185069 [.110786,.261375] |

The shuffled comparison intervals are descriptive summaries only: recipient resampling does not preserve the dependence induced by donor cycles. Also donors match N/split/delay, not recipient novelty stratum: 144 of 173 all-positive test recipients receive non-positive donor histories (94 for N2, 50 for N3). That subgroup shuffle contrast changes the state marginal as well as alignment. Neither caveat affects the primary full-state-minus-native comparison. The all-positive cue-u-minus-native-output interval crosses zero; this intermediate-stage contrast is inconclusive in that stratum.

| Stratum / N | Histories | Native Ecomp | Full endpoint Ecomp | Full endpoint joint accuracy |
|---|---:|---:|---:|---:|
| Full / 2 | 512 | 1.994055 | 1.897413 | 4.2318% |
| Full / 3 | 512 | 1.961783 | 1.882839 | 4.0365% |
| All-positive / 2 | 119 | 2.034578 | 1.846655 | 4.2017% |
| All-positive / 3 | 54 | 1.842410 | 1.633963 | 9.8765% |

The small all-positive N=3 denominator must accompany its apparently larger accuracy. Full-state train/validation/test Ecomp is 1.874117/1.940623/1.890126. No validation or subgroup score was used for model selection. One test pair is clipped in each of CUE_U_Q and CUE_U_A_Q; full-state raw Ecomp is 1.890424, so the principal gain is not an artifact of clipping. No all-positive test output is clipped. All 576 cells, including all delays, raw errors, per-role MAE/RMSE and unfavorable accuracies, remain in [results.json](results.json); [metric_cells.csv](metric_cells.csv) is a readable tabular export.

## What is localized, and what is not

The query sets content current to zero. In the frozen update, u and r therefore evolve independently of A throughout the query. Replacing A by 0 or 5 at 96 declared cue checkpoints leaves the complete returned native result, including spike sequence, exactly unchanged (192 intervention windows). These are within-bound state interventions, not claims of naturally reachable encoded histories. [A_invisibility_checks.json](A_invisibility_checks.json) records them.

The combined evidence supports a limited conclusion: original state plus query has target-predictive value for a small external learner beyond the registered readout policy; holding cue u,r fixed, the native query is insensitive to cue A. A can already have affected retained u during encoding. In the original model A is driven by keys/timing, not values; A-assisted decoding can use that information to interpret u without A itself storing value content. The learner's gain does not show that the native carrier is a sufficient representation of all requested targets.

The feature views have different capacities and scaling/regularization geometry. Direct q, nonlinear arithmetic, supervised parameters, state sensors and output clipping are added diagnostic resources. Thus the staged gains do not uniquely assign physical causal blame to affine calibration, query evolution or A omission. A worse native-deadline probe need not imply irreversible loss: conditioning and finite polynomial approximation can also matter. No failed probe establishes information absence.

In particular, predictive gain is not demonstrated key binding. A state can predict a shared history-level value statistic without selecting the requested item. This study has neither a state-only nor a query-permutation control that identifies query-specific benefit. The correct-history oracle validates the task/target pairing and attainable answer with privileged memory; it does not validate compressed-state sufficiency. The absolute errors and low strict success remain the main negative finding.

## Resource accounting and next decision

The physical carrier remains the original three-coordinate M1; its timing, currents, reset semantics and initialization are unchanged. The five fitted diagnostics add 105 total coefficients across independent arms, training-only preprocessing constants, up to 35 monomials per output and access to one, two or three exact scalar features. Each arm has its own learner; 105 is not a single proposed neuronal decoder. The full endpoint is measured before the native 10 ms query window, whereas the native-output feature is measured at its deadline. These are different observation contracts, not matched implementations or equal-delay hardware.

There are 2,048 training histories and 12,288 dependent query training rows. Raw histories are retained only by the evaluator/oracle and for audit provenance. No raw-history cache enters learned features. Exact floating state access and stored learned weights have no measured biological energy or precision budget here. Main execution recorded 7.739 s and peak RSS 213,600 KiB in this environment; these are software observations, not neural costs.

The justified project decision is to keep the original failure labels, correct the manuscript contract, and close this fixed probe run without enlarging the decoder. The next discriminating question is whether the observed gain depends on the correct query identity rather than a shared content statistic. A fresh, separately preregistered query-swap/state-only diagnostic would address that narrow issue before designing an A-sensitive physical readout. Such an extension was not executed in this study. Opening a new native input or readout pathway would need to be named as a contract change, not a repair of the historical experiment.

## Evidence and reproducibility

The main command completed with exit 0:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/stage_localization/run_probe.py
```

[run.log](run.log) records frozen input/model hashes before metrics. [inputs.jsonl](inputs.jsonl), [split_and_donor_ledger.json](split_and_donor_ledger.json), [native_observations.jsonl](native_observations.jsonl), [allowed_features.npz](allowed_features.npz), [frozen_models.json](frozen_models.json), [predictions.jsonl](predictions.jsonl) and [test_history_losses.json](test_history_losses.json) contain the full reproducible chain. Predictions retain all eight arms, raw/clipped outputs and every attempted history/delay. There are no discarded failures.

The independent command `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python tan_audit/stage_localization/independent_review.py` completed with exit 0. It regenerated all 3,584 histories, six donor cycles, 21,504 native query rows, five feature matrices, 10,752 paired prediction rows, 576 metric cells and 12 bootstrap contrasts. Native states/readouts and metric/bootstrap recomputations matched exactly. An alternate augmented least-squares solver matched coefficients within 8.33e-13 and independently refitted predictions within 9.08e-13; saved-prediction arithmetic differences were at most 5.33e-15. All 96 A-invisibility checkpoints passed. The ledger includes 103 deadline spikes and 2,237 pre-deadline refractory cases, so correct candidate/reset semantics were exercised rather than checked only on a silent subthreshold branch.

The checker and its final disposition are in [independent_review.py](independent_review.py), [INDEPENDENT_REVIEW_CHECKS.json](INDEPENDENT_REVIEW_CHECKS.json), [independent_review.log](independent_review.log) and [REVIEW.md](REVIEW.md). A separate [LEAKAGE_SCOPE_REVIEW.md](LEAKAGE_SCOPE_REVIEW.md) documents feature/split checks and the donor caveats. Source-flow inspection and saved hashes support the stated execution order; they are not an external timestamp certification. Primary artifacts and repository status remained unchanged during review. The patch review is separate from the experiment review.

 The documentation and fixed experiment are complete. Full manuscript release-layout checking and a future query-specific diagnostic are explicitly outside this completed scope, not completed work.
