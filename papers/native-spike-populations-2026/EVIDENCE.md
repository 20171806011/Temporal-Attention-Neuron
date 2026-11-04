# Evidence ledger for the V6–V10 population manuscript

Canonical source snapshot: `c75814eac5202e6a265ca42e78805f906d302249`. All file paths and SHA-256 hashes are in [evidence.json](evidence.json).

This publication reorganizes saved scientific results. It introduces no new training runs, task data, held-out evaluation or numerical-model replay. Original phase reports retain their execution-time status statements. Study cohorts and estimands are not pooled.

## Study inventory and commit gates

| Phase | Paired blocks | New fits | Derived cuts | Held-out model-episode evaluations | Protocol commit | Checkpoint commit |
|---|---:|---:|---:|---:|---|---|
| V6 | 20 | 80 | 0 | 40,960 | `26d85e9d6176f7b0d335d55527cd0c8049f55a75` | `0cb7d16aad5fe897406110fc5c79fd483347aab3` |
| V7 | 20 | 160 | 0 | 81,920 | `8ad0b69e65074563929dff905acc0067837e3778` | `041693b7297b888ed31ac78f8d5677408669f1fe` |
| V8 | 104 | 416 | 0 | 212,992 | `cd5cd4ef1cc65fd098b3a73d36d68e7a49307d17` | `3ec9ce117ad3ce1a20d6e5ef4f2395916f120821` |
| V9 | 5 | 20 | 10 | 15,360 | `ad1d807b1d48be559fd67d4f6d196b8693b4da90` | `d4fde77eb49e21c00950cb93050819ab72032e90` |
| V10A | 5 | 40 | 0 | 20,480 | `4dd342161cee0d2d0f0a8383cec6aba5758d514d` | `e77dd9555ce15f94b9f5bcba12650c48aa9ce19d` |

The 716 fits yield 726 held-out model conditions because V9 derives ten inference cuts without training. There are 371,712 model-episode evaluations, not 371,712 independent training replications. Each study uses 128 training and 512 held-out episodes per paired block, with 200 updates per trained model.

## Claim-to-source map

| Claim | Saved source and exact field | Scope |
|---|---|---|
| V6 DEND bundled-recipe contrast | [TEST_SUMMARY.json](../../research/dendritic-population-v6/results/TEST_SUMMARY.json), `statistics.intervals.accuracy_DEND_CE_minus_RATE` | −13.564453 pp; pointwise bootstrap95 [−28.750000,+0.136719]; changes loss and LR together. |
| V7 POINT factorial interaction | [TEST_SUMMARY.json](../../research/dendritic-population-v7/results/TEST_SUMMARY.json), `statistics.intervals.accuracy_POINT_loss_lr_interaction` | −33.837891 pp; six-primary-family adjusted percentile interval [−64.482666,−5.843058]; approximate exploratory coverage. |
| V8 primary POINT interaction | [TEST_SUMMARY.json](../../research/dendritic-population-v8/results/TEST_SUMMARY.json), `statistics.intervals.accuracy_POINT_loss_lr_interaction` | −14.105694 pp; primary approximate Student-t95 [−22.603812,−5.607576],104 fresh blocks; negative direction repeats but the earlier crossover of simple-effect point estimates does not. |
| V9 dependence versus retrainable alternatives | [TEST_SUMMARY.json](../../research/dendritic-population-v9/results/TEST_SUMMARY.json), `statistics.cells`; `rows` | DEND intact/retrained/cut passes4/5,5/5,0/5; no-cross success concerns finite task/gate and retains multiple cells and state. |
| V10A fixed-recipe size comparison | [TEST_SUMMARY.json](../../research/dendritic-population-v10/results/TEST_SUMMARY.json), `statistics.cells`; `statistics.intervals` | DEND-minus-POINT BA contrasts at h6/4/2/1 are +9.82/+12.81/+19.58/+16.74pp; four unadjusted intervals, five blocks; not a pure representation bound. |
| V10A minimum-size POINT passing case | [TEST_SUMMARY.json](../../research/dendritic-population-v10/results/TEST_SUMMARY.json), `rows[id=s0_POINT_h1]` | BA91.195437%,coverage97.0703125%; two branch cells+one hidden soma+two output cells; five cells, not one. |

V7/V8 within-architecture loss-by-learning-rate contrasts use `(CE/.03 − CE/.01) − (RATE/.03 − RATE/.01)`. V8 uses a sole prespecified primary Student-t interval; its bootstrap interval is supplementary. V7 uses approximate Bonferroni-adjusted bootstrap intervals for six primary contrasts. V6, V9 and V10 intervals are pointwise exploratory. These methods must not be silently exchanged.

## Complete cell means and failures

| Phase | Architecture | Condition | n | Accuracy % | Balanced accuracy % | Coverage % | Failed | Fully silent | Fully abstaining |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| V6 | DEND | RATE_0.01 | 20 | 99.746094 | 99.746094 | 99.794922 | 0 | 0 | 0 |
| V6 | DEND | CE_0.03 | 20 | 86.181641 | 86.181641 | 86.230469 | 3 | 0 | 2 |
| V6 | POINT | RATE_0.01 | 20 | 59.560547 | 59.560547 | 79.326172 | 17 | 0 | 0 |
| V6 | POINT | CE_0.03 | 20 | 74.238281 | 74.238281 | 79.492188 | 9 | 2 | 2 |
| V7 | DEND | RATE_0.01 | 20 | 99.941406 | 99.941406 | 99.970703 | 0 | 0 | 0 |
| V7 | DEND | RATE_0.03 | 20 | 99.960938 | 99.960938 | 99.960938 | 0 | 0 | 0 |
| V7 | DEND | CE_0.01 | 20 | 96.191406 | 96.191406 | 96.220703 | 3 | 0 | 0 |
| V7 | DEND | CE_0.03 | 20 | 82.490234 | 82.490234 | 82.490234 | 4 | 2 | 3 |
| V7 | POINT | RATE_0.01 | 20 | 71.796875 | 71.796875 | 86.611328 | 15 | 0 | 0 |
| V7 | POINT | RATE_0.03 | 20 | 75.000000 | 75.000000 | 88.144531 | 11 | 0 | 0 |
| V7 | POINT | CE_0.01 | 20 | 87.734375 | 87.734375 | 95.927734 | 7 | 0 | 0 |
| V7 | POINT | CE_0.03 | 20 | 57.099609 | 57.099609 | 60.732422 | 13 | 5 | 5 |
| V8 | POINT | RATE_0.01 | 104 | 66.062575 | 66.062575 | 80.496920 | 87 | 0 | 1 |
| V8 | POINT | RATE_0.03 | 104 | 68.134014 | 68.134014 | 84.707407 | 79 | 0 | 1 |
| V8 | POINT | CE_0.01 | 104 | 85.163762 | 85.163762 | 92.685171 | 38 | 2 | 2 |
| V8 | POINT | CE_0.03 | 104 | 73.129507 | 73.129507 | 78.566331 | 47 | 5 | 7 |
| V9 | DEND | intact | 5 | 90.351562 | 90.351562 | 92.070312 | 1 | 0 | 0 |
| V9 | DEND | retrained | 5 | 98.867188 | 98.867188 | 99.453125 | 0 | 0 | 0 |
| V9 | DEND | cut | 5 | 55.898438 | 55.898438 | 92.148438 | 5 | 0 | 0 |
| V9 | POINT | intact | 5 | 53.593750 | 53.593750 | 70.664062 | 5 | 0 | 0 |
| V9 | POINT | retrained | 5 | 44.335938 | 44.335938 | 67.500000 | 5 | 0 | 0 |
| V9 | POINT | cut | 5 | 51.171875 | 51.171875 | 80.156250 | 5 | 0 | 0 |
| V10A | DEND | h=6 | 5 | 99.609375 | 99.573413 | 99.726562 | 0 | 0 | 0 |
| V10A | POINT | h=6 | 5 | 89.179688 | 89.756944 | 96.367188 | 2 | 0 | 0 |
| V10A | DEND | h=4 | 5 | 97.031250 | 97.043651 | 97.812500 | 0 | 0 | 0 |
| V10A | POINT | h=4 | 5 | 84.531250 | 84.236111 | 92.968750 | 4 | 0 | 0 |
| V10A | DEND | h=2 | 5 | 89.218750 | 88.501984 | 94.531250 | 2 | 0 | 0 |
| V10A | POINT | h=2 | 5 | 71.679688 | 68.923611 | 94.804688 | 5 | 0 | 0 |
| V10A | DEND | h=1 | 5 | 88.359375 | 88.343254 | 96.601562 | 3 | 0 | 0 |
| V10A | POINT | h=1 | 5 | 74.023438 | 71.602183 | 91.640625 | 4 | 0 | 0 |

Delayed XOR has balanced class counts, so ordinary and balanced accuracy coincide in V6–V9. V10 uniformly samples16 patterns with7 positive and9 negative patterns; balanced accuracy is the mean of class recalls, with abstentions wrong. Failure means either accuracy<90% or coverage<90% in V6–V9; balanced accuracy replaces accuracy in V10. Means do not determine the number of passing fits.

## Saved numerical audits and execution costs

| Phase | Checker assertions | Adam updates | Checkpoints | Final proxy derivatives | Sampled gradients | Training/artifact seconds | Test/artifact seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| V6 | 107,001 | 16,000 | 720 | 14,080 | 240 | 72.271 | 12.279 |
| V7 | 202,055 | 32,000 | 1,440 | 28,160 | 480 | 146.784 | 24.733 |
| V8 | 555,877 | 83,200 | 3,744 | 73,216 | 1,248 | 398.360 | 77.126 |
| V9 | 44,671 | 4,000 | 180 | 2,920 | 60 | 18.400 | 4.890 |
| V10A | 68,836 | 8,000 | 360 | 3,430 | 200 | 32.398 | 3.964 |

Across the saved audits:978,440 assertions,143,200 reconstructed updates,6,444 checkpoints,121,806 final anchored-surrogate derivative checks and2,228 sampled gradients. Maximum native state/candidate residual is3.552713678800501e−15; maximum final local surrogate-chain-rule residual is7.041099786553318e−10. Spikes, counts and predictions match exactly in the saved full-trajectory replays. These are internal same-workflow checks, not external independent replication or peer review. Derivative checks validate a local surrogate graph, not the exact derivative of a hard-spike function.

All per-phase dense matrix MAC accounting, separate diagnostic passes, test work and audit timing are copied into evidence.json from the original COST/RESOURCE_COST records. Wall time is CPU/environment dependent; overlapping audit wall times cannot be added as elapsed workflow time. Neither MACs nor model events measure joules, ATP or hardware energy. V6 scientific sources were frozen before training data; its separately written native verifier was added with the trained-checkpoint commit before held-out data, not before training. The phase-specific freeze records remain authoritative.

## Publication figures

- [Optimization and interaction figure](figures/optimization_interactions.png): V6 paired bundled recipes, V7/V8 separate POINT factorial means and their original primary interval conventions.
- [Recurrence controls](figures/recurrence_controls.png): all five V9 seed outcomes for both architectures, including failed intact, retrained and cut conditions.
- [Binding scale curves](figures/binding_scaling.png): all V10 outcomes, mean balanced accuracy and five-block paired contrasts at every size.

Each figure is also supplied as vector PDF and SVG. All are generated from saved JSON by prepare_evidence.py; figures do not imply pooled cohorts or equal biological/energy resources.

## Evidence boundaries and prior context

V5 independently selected POINT CE/.003; V6–V10 fixed grids did not retest that selected configuration. The uncertain V5 independently tuned comparison is not replaced by favorable fixed-recipe comparisons. Its original [report](../../research/dendritic-population-v5/REPORT.md) and [protocol](../../research/dendritic-population-v5/PROTOCOL.md) are separately hashed as context, not counted among the new V6–V10 fits.

V9 removes60 cross-cell weights (176→116 effective trainable parameters), changes gradient normalization and retains self recurrence, branch states and output integration. Inference cuts concern an already trained solution; finite retraining failure cannot establish physical necessity. V10 matches parameters/state at each size but not cell counts, scalar nonlinearities, reset derivatives or effective optimization. Size changes initialization drive variance and per-coordinate mean-L2 strength. Its h=1 POINT success rules out interpreting this pilot as impossibility at the tested finite gate.

## Validate the publication extraction

Run `python validate_evidence.py` from this directory or any other working directory. It checks source SHA-256 identities against the pinned Git tree, headline arithmetic, paired contrast means, the V8 primary Student-t formula, complete fit retention and figure presence. It does not import scientific models or rerun experiments. Original experiment audit scripts may rewrite their logs; run those only in a separate copy.
