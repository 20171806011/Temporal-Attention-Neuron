# V7 resource and compute accounting

Counts below cover the fixed 160-run factorial. Dense matrix multiply-accumulate (MAC) terms are an algorithmic count, not measured FLOPs, energy, or hardware efficiency.

| Quantity | DEND | POINT |
|---|---:|---:|
| Trainable float64 parameters | 176 | 176 |
| Continuous state variables | 20 | 20 |
| Previous hidden spikes / output counters | 6 / 2 | 6 / 2 |
| Branch computation | 12 graded compartments | 12 native-spiking cells |
| Hidden somata / output cells | 6 / 2 | 6 / 2 |
| Batch / sequence steps / updates | 128 / 24 / 200 | 128 / 24 / 200 |
| Fixed loss–learning-rate cells / seeds per cell | 4 / 20 | 4 / 20 |
| Forward / backward dense MAC terms per episode-step | 156 / 240 | 156 / 240 |

| Cell | Runs | Updates | Forward MAC | Backward MAC | Diagnostic MAC | Test MAC | Recorded train seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| DEND RATE_0.01 | 20 | 4,000 | 1,916,928,000 | 2,949,120,000 | 86,261,760 | 38,338,560 | 17.858 |
| DEND RATE_0.03 | 20 | 4,000 | 1,916,928,000 | 2,949,120,000 | 86,261,760 | 38,338,560 | 17.627 |
| DEND CE_0.01 | 20 | 4,000 | 1,916,928,000 | 2,949,120,000 | 86,261,760 | 38,338,560 | 17.779 |
| DEND CE_0.03 | 20 | 4,000 | 1,916,928,000 | 2,949,120,000 | 86,261,760 | 38,338,560 | 17.996 |
| POINT RATE_0.01 | 20 | 4,000 | 1,916,928,000 | 2,949,120,000 | 86,261,760 | 38,338,560 | 18.621 |
| POINT RATE_0.03 | 20 | 4,000 | 1,916,928,000 | 2,949,120,000 | 86,261,760 | 38,338,560 | 18.717 |
| POINT CE_0.01 | 20 | 4,000 | 1,916,928,000 | 2,949,120,000 | 86,261,760 | 38,338,560 | 18.848 |
| POINT CE_0.03 | 20 | 4,000 | 1,916,928,000 | 2,949,120,000 | 86,261,760 | 38,338,560 | 18.511 |

Total: 32,000 parameter updates; 15,335,424,000 training-forward MAC terms and 23,592,960,000 training-backward MAC terms. The 1,440 additional training-diagnostic forwards contribute 690,094,080 terms; testing contributes 306,708,480. Nine diagnostics include update 200, so final training metrics are not counted twice.

There are 2,560 unique training histories and 10,240 unique held-out histories, reused across eight models within each paired block. The 81,920 test model-episodes are not 81,920 independent training units; the inferential unit remains 20 seed/data blocks.

Training and artifact phase: 146.784 s. Held-out evaluation and artifact phase: 24.733 s. The native test model calls themselves total 1.041 s; the rest includes compression, metrics and I/O.

| Separate audit | Recorded wall seconds | Status |
|---|---:|---|
| verification | 40.288 | PASS |
| gradient_audit | 73.706 | PASS |

Dense matrix MAC counts exclude scalar nonlinearities, resets, loss computation, Adam, data generation, I/O and numerical audits.
Recorded wall times depend on CPU contention and file compression; separate audit times may overlap and must not be summed as elapsed workflow time.
DEND has 12 graded compartments while POINT has 12 additional spiking branch cells; scalar work, activation patterns and reset derivatives differ.
Matching matrix terms and nominal learning rates does not establish equal effective updates, energy or hardware cost.
Checkpoint forward counts include the final training diagnostic; no validation or hyperparameter selection runs were performed.

Numerical checking is extra work outside the training/test matrix budget: 480 sampled backward replays, 28,160 final-checkpoint local proxy-derivative checks, full Adam reconstruction, native trajectory replay, and statistical regeneration. See the audit JSON for the complete counts and residuals.

CPU-only NumPy execution used one BLAS/OpenMP thread. Each training, test/artifact and audit stage had a 300-second cap. No paid compute, architecture search, external output pulse generator, or analog decoder was introduced.
