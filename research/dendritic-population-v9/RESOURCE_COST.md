# V9 resource and computation accounting

The fixed experiment has 20 trained models and 10 inference-only cuts. Five seed/data blocks are the statistical units. Matrix multiply-accumulate (MAC) terms describe dense arithmetic, not measured hardware operations or energy.

| Resource | Intact DEND / POINT | Retrained no-cross DEND / POINT | Inference cut |
|---|---:|---:|---:|
| Stored float64 parameters | 176 | 176 | 176 |
| Effective trainable parameters during fitting | 176 | 116 | No additional fitting |
| Cross-cell / self recurrent slots active | 60 / 12 | 0 / 12 | 0 / 12 |
| Continuous states / previous hidden spikes / output counters | 20 / 6 / 2 | 20 / 6 / 2 | 20 / 6 / 2 |
| Branch units / hidden somata / outputs | 12 / 6 / 2 | 12 / 6 / 2 | 12 / 6 / 2 |
| Training batch / sequence steps / updates | 128 / 24 / 200 | 128 / 24 / 200 | No additional fitting |
| Forward / backward dense MAC terms per episode-step | 156 / 240 | 156 / 240 | 156 / no backward |

DEND branches retain graded states and have no branch spike events; POINT branch cells spike and reset. Scalar nonlinearities, reset derivatives, active states and gradient patterns differ. The 60 removed weights still occupy dense matrix positions. The original L2 regularizer denominator remains 176.

| Arm | Condition | Fits | Updates | Training forward MAC | Training backward MAC | Diagnostic forward MAC | Test forward MAC | Training seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DEND | intact | 5 | 1,000 | 479,232,000 | 737,280,000 | 21,565,440 | 9,584,640 | 4.530 |
| DEND | retrained | 5 | 1,000 | 479,232,000 | 737,280,000 | 21,565,440 | 9,584,640 | 4.536 |
| DEND | cut | 0 | 0 | 0 | 0 | 0 | 9,584,640 | 0.000 |
| POINT | intact | 5 | 1,000 | 479,232,000 | 737,280,000 | 21,565,440 | 9,584,640 | 4.609 |
| POINT | retrained | 5 | 1,000 | 479,232,000 | 737,280,000 | 21,565,440 | 9,584,640 | 4.646 |
| POINT | cut | 0 | 0 | 0 | 0 | 0 | 9,584,640 | 0.000 |

Totals: 4,000 parameter updates, 1,916,928,000 training-forward MAC terms, 2,949,120,000 training-backward MAC terms, 86,261,760 diagnostic-forward terms and 57,507,840 test-forward terms. Nine saved diagnostics per fit include the final training diagnostic, which is not counted again.

There are 640 unique training histories and 2,560 unique held-out histories. Inputs are shared across architectures/conditions within each block. The 15,360 test model-episodes are not independent training replicates.

Recorded training and artifact phase: 18.400 s. Held-out evaluation and artifact phase: 4.890 s. Native test calls alone: 0.264 s. CPU-only NumPy, one BLAS/OpenMP thread; 300-second scientific stage caps.

| Additional verification | Recorded wall seconds | Status |
|---|---:|---|
| verification | 5.800 | PASS |
| gradient_audit | 7.527 | PASS |

Audit work is outside the training/test matrix counts: full Adam reconstruction, scalar-equation native replay, 60 sampled full-batch gradient replays, and 2,920 final-checkpoint local surrogate finite differences. Pre-data fixtures include 704 intact and 232 masked local derivative checks. Audit files retain exact counts, timing and residuals.

- Dense matrix MAC terms exclude scalar nonlinearities, reset, losses, optimizer, generation, I/O and numerical verification.
- Intact and retrained have unequal effective trainable parameter counts (176 versus 116), despite equal storage and dense matrix work.
- DEND and POINT have unequal scalar nonlinearities, reset derivatives, branch-event semantics and realized activation/gradient patterns.
- Recorded wall times depend on machine contention and artifact compression; audits may overlap and must not be added as total elapsed workflow time.
- Spike totals and state L1 variation are computational activity proxies, not physical energy, ATP, biological metabolism or hardware efficiency.
- The inference-cut condition has no additional fit; its training origin is the corresponding intact model.
