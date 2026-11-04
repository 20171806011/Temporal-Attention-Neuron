# V8 resource and compute accounting

Counts cover the fixed 416-run POINT factorial replication. Dense matrix multiply-accumulate (MAC) terms are an algorithmic count, not measured FLOPs, energy, or hardware efficiency.

| Quantity | Each POINT configuration |
|---|---:|
| Trainable float64 parameters | 176 |
| Continuous state values | 20 |
| Previous hidden spikes / output counters | 6 / 2 |
| Native branch cells / hidden somata / output cells | 12 / 6 / 2 |
| Batch / sequence steps / updates | 128 / 24 / 200 |
| Paired seed/data blocks | 104 |
| Forward / backward dense MAC terms per episode-step | 156 / 240 |

| Recipe | Runs | Updates | Forward MAC | Backward MAC | Diagnostic MAC | Test MAC | Recorded train seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| RATE_0.01 | 104 | 20,800 | 9,968,025,600 | 15,335,424,000 | 448,561,152 | 199,360,512 | 97.960 |
| RATE_0.03 | 104 | 20,800 | 9,968,025,600 | 15,335,424,000 | 448,561,152 | 199,360,512 | 98.477 |
| CE_0.01 | 104 | 20,800 | 9,968,025,600 | 15,335,424,000 | 448,561,152 | 199,360,512 | 98.324 |
| CE_0.03 | 104 | 20,800 | 9,968,025,600 | 15,335,424,000 | 448,561,152 | 199,360,512 | 98.499 |

Total: 83,200 parameter updates; 39,872,102,400 training-forward MAC terms and 61,341,696,000 training-backward MAC terms. The 3,744 additional training-diagnostic forward batches contribute 1,794,244,608 terms; held-out inference contributes 797,442,048. Nine diagnostics include update 200, so final training metrics are not counted twice.

There are 13,312 unique training histories and 53,248 unique held-out histories, reused across four models within each paired block. The 212,992 test model-episodes are not 212,992 independent training units; the inferential unit is 104 seed/data blocks.

Training and artifact phase: 398.360 s. Held-out evaluation and artifact phase: 77.126 s. Native test model calls total 3.248 s; the remaining time includes metrics, compression and I/O.

| Separate audit | Recorded wall seconds | Status |
|---|---:|---|
| verification | 107.831 | PASS |
| gradient_audit | 191.860 | PASS |

Dense matrix MAC counts exclude scalar nonlinearities, resets, loss computation, Adam, data generation, I/O and numerical audits.
Recorded wall times depend on CPU contention and file compression; audit times may overlap and must not be summed as elapsed workflow time.
All four cells are POINT; this experiment cannot compare architecture resources or structural superiority.
Matching matrix terms and nominal learning rates does not establish equal effective updates, energy or hardware cost.
Checkpoint forward counts include the final training diagnostic; no validation or hyperparameter selection runs were performed.

Numerical checking is additional work outside the training/test matrix budget: 1,248 sampled backward replays, 73,216 final-checkpoint local proxy-derivative checks, complete Adam reconstruction, native trajectory replay, and statistical regeneration. Audit JSON files retain complete counts and residuals.

CPU-only NumPy execution uses one BLAS/OpenMP thread. Fixed caps are 900 seconds for training plus diagnostics, 600 seconds for held-out evaluation plus artifacts, and 900 seconds per audit. No paid compute, architecture search, external output pulse generator, or analog decoder was introduced.

V7 appears in one interval figure solely as historical pilot context. Its 20 blocks are not pooled with the 104 new V8 blocks, and that plot is not a formal between-phase test.
