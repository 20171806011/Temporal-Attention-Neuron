# Independent review of the original-carrier stage-localization experiment

2026-11-02. **PASS for the frozen implementation and all reported numerical outcomes, with the interpretation limits below.** The primary artifacts and original repository were preserved. The numerical checker imports neither `run_probe.py` nor the original model/generator modules; it independently reconstructs their declared operations. A second reviewer checked leakage, feature access and interpretation in [LEAKAGE_SCOPE_REVIEW.md](LEAKAGE_SCOPE_REVIEW.md).

The reviewed contract is [PREREGISTRATION.md](PREREGISTRATION.md), SHA256 `8ffcda523dfc042f9d64473c361ecc39c6f69b1f9d2b2af7994bbd35dea6c1ff`; the primary source is [run_probe.py](run_probe.py), SHA256 `31021b95d5bc76f9a0798c406e63337f4a2d9056bd9fbac8dd6c8ae20eb6f5f2`. The frozen native source remains SHA256 `6b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f`. All reviewed primary hashes appear in [INDEPENDENT_REVIEW_CHECKS.json](INDEPENDENT_REVIEW_CHECKS.json).

## Numerical reconstruction

[independent_review.py](independent_review.py) completed with exit code **0** using:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python tan_audit/stage_localization/independent_review.py > tan_audit/stage_localization/independent_review.log 2>&1
```

The final independent run, after the archive-portability change described below, took 1.488 s and reported peak RSS 192,412 KiB. [independent_review.log](independent_review.log) records the completed assertions. These software resource observations are separate from the primary experiment's resources and are not biological costs.

| Check | Independently verified scope | Largest discrepancy |
|---|---|---:|
| Generator and split RNG | All 3,584 histories, complete values/keys/masks/positions/queries/labels/rejection counts and splits | Exact record equality |
| Donor RNG | All six N-by-split cycles, no self-donor, permutation, no cross-split transfer | Exact mapping equality |
| Registered Euler dynamics | All 21,504 native query rows, including all cue states, deadline readouts, refractory state and eligible spike counts | 0 |
| Allowed features | All five saved arrays reconstructed from independently generated state/current query | 0 |
| Preprocessing | Training-only raw and expanded moments, deterministic polynomial powers | 2.23e-16 |
| Ridge coefficients | All five fits checked with augmented least squares, independent of the primary normal-equation solve | 8.33e-13 |
| Alternate-solver predictions | All query rows using independently refitted coefficients | 9.08e-13 |
| Saved predictions | 10,752 paired rows, eight arms; 344,064 raw-and-clipped scalar outputs | 5.33e-15 for reconstructed raw outputs; clipping exact |
| Metric cells | All 576 cells, all fields and denominators | 0 |
| Uncertainty arithmetic | All 1,024 test-history losses and 12 declared 2,000-replicate contrasts | 0 |

The original generator's RNG call order was reconstructed without importing it: rejection-sampled keys, key permutation, independent values, event positions, then target choice. Seed metadata never reset the continuous RNG. Per N there are exactly 1,024/256/512 training/validation/test histories; both roles and all three delays share each history's split. The five fitting problems use 12,288 training rows from 2,048 histories. Validation/test labels do not enter the reconstructed transforms or regression solve.

The independent vector recurrence respects the original old-A update, refractory countdown and pre-reset candidate. Its scalar-equivalent calculations match the saved native observations exactly. The readout distinction is consequential in these data: across all splits, 103 query windows spike exactly at the deadline and 2,237 enter the deadline step refractory. The checker uses the candidate for the former and reset voltage for the latter, and excludes deadline spikes from eligible counts. It does not substitute post-step u for `u_readout`.

All cue r values are zero. The 96 declared checkpoints and 192 A interventions were reconstructed independently: replacing A by 0 or 5 leaves membrane outputs, refractory values and complete spike sequences exactly unchanged. The frozen equation gives the structural reason: during the query, content current is zero, so current cue A has no effect on u/r when their initial values and q are fixed. This does not erase A's earlier effect on u during encoding, and these interventions need not correspond to reachable histories.

## Leakage and solver review

No forbidden learned feature or history leakage was found. Exact saved feature arrays contain only the declared q, cue state coordinates or native deadline readout. IDs, masks, positions, N and delay are grouping/indexing metadata, not fitted inputs. M0's raw-history answer is a privileged evaluator control; it does not feed any feature. All-positive status is used for reporting only, without subgroup refitting.

The ridge objective, intercept exemption and lambda=.001 scaling are correct. The alternate augmented least-squares solve avoids relying solely on the primary normal-equation implementation. Its maximum coefficient difference is 8.33e-13; the largest residual of a frozen coefficient in the independently formed normal equations is 1.16e-15. Raw and clipped predictions are both retained; clipping is the frozen support policy rather than target-dependent correction.

Source flow writes inputs and fitted models before outcome summaries. Source inspection and artifact hashes support the declared execution path; the files alone cannot independently certify historical timestamps or exclude an unrecorded prior run. This is a verification limitation, not evidence of such a run. No performance-driven branch, tuning or validation selection appears in the inspected script.

## Scientific interpretation

The primary full-state-minus-native Ecomp difference is −0.087793, with the declared paired percentile interval [−0.111071, −0.063862]. On 1,024 independent test histories, full-state Ecomp is 1.890126 versus native 1.977919, query-only 1.976712 and shuffled-state 1.977327. This establishes modest target-predictive signal accessible to the fixed external learner. Full-state strict joint accuracy is still only 4.13%, and the original failure labels remain unchanged.

All-positive test results condition on 173 histories, comprising N2=119 and N3=54. Full-state Ecomp is 1.780266 versus native 1.974595; strict joint accuracy is 5.97%. This is a conditioned analysis of the same fitted learner, not a causal novelty manipulation. Its cue-u-minus-native-output interval crosses zero [−0.069727, 0.005952], so not every intermediate contrast is resolved even descriptively.

Retain these limitations when shortening the report:

1. The five feature views have 5/15/15/35/35 terms, different scaling and regularization geometry, direct access to q, external learned parameters and exact state measurements. Their gains do not uniquely locate a physical loss mechanism. Full endpoint versus cue u is not a capacity-matched intervention on A alone.
2. Predictive target information is not demonstrated query-specific key binding. A carrier can predict shared history-level content without selecting the queried item. No state-only or shuffled-query control was run. Probe failure likewise cannot establish information absence.
3. The native query's A blindness is exact; the empirical full-state gain is a separate finite-family result. Together they support further investigation of readout access, not sufficiency of the carrier or an implementable neuronal decoder.
4. Of 173 all-positive test recipients, **144 have donors outside that stratum** (94/119 for N2 and 50/54 for N3). The shuffled subgroup changes state/novelty composition as well as alignment. This follows the frozen design and is not leakage.
5. Shuffled recipient losses share underlying histories through the donor cycle. Recipient-history resampling does not preserve that dependence, so the shuffled contrast intervals remain descriptive and must not be presented as valid independent-history population confidence intervals for the randomized control. This does not affect the primary full-state-minus-native contrast. All intervals condition on the one fitted learner and omit training-sample uncertainty.
6. The correct-history oracle's 100% score confirms the target pairing and raw-record task; it does not validate state compression or add a neural resource for free. Fixed clipping affects only one full-state test pair; raw Ecomp 1.890424 supports the same modest-gain interpretation.

The main [REPORT.md](REPORT.md) was read through its conclusions, resource accounting and next-step decision. Its present distinction between predictive improvement and demonstrated binding is appropriate. Wording about native A access should remain conditional on fixed cue u,r: prior A can already have influenced encoded u. Closing this bounded run and preserving the original negative result is supported; a future query-specific diagnostic would require its own preregistration.

## Reproduction and disposition

The checker locates the repository from its own path: `HERE.parent.parent / 'Temporal-Attention-Neuron'`. Replay therefore requires sibling `tan_audit/` and `Temporal-Attention-Neuron/` directories, with this stage under `tan_audit/stage_localization/`. It reads the existing primary artifacts and writes only independent review outputs. The source hashes remain mandatory provenance checks. Git status is checked when `.git` is present; an archive without `.git` explicitly reports repository-status verification unavailable and records `repository_status_unchanged=null`, never a claimed verified result.

The final checker was also executed in an isolated temporary snapshot with the declared sibling layout and no `.git`, using [verify_archive_review.py](verify_archive_review.py). This exercised the actual archive branch and completed exit 0, including all generator, recurrence, fit, metric and bootstrap checks. [ARCHIVE_REPLAY_REVIEW.json](ARCHIVE_REPLAY_REVIEW.json) and [archive_replay_review.log](archive_replay_review.log) retain the evidence. Temporary copies were removed afterwards; primary experiment execution was not repeated.

No primary experimental artifact changed during verification. The original repository was clean before and after.   The separate manuscript-patch audit is outside this numerical review and is not silently represented as independently rechecked here.
