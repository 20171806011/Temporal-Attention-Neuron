# V10 resource and compute accounting

All numbers refer to the frozen 40-run RATE/.01/200-update study; each arm/size has 5 independent paired seed/data blocks.

| h | Arm | Parameters | Float states | Cells | Graded branches | Spiking branches | Forward/backward MAC per episode-step |
|---:|---|---:|---:|---:|---:|---:|---:|
| 6 | DEND | 176 | 20 | 8 | 12 | 0 | 156 / 240 |
| 6 | POINT | 176 | 20 | 20 | 0 | 12 | 156 / 240 |
| 4 | DEND | 102 | 14 | 6 | 8 | 0 | 88 / 128 |
| 4 | POINT | 102 | 14 | 14 | 0 | 8 | 88 / 128 |
| 2 | DEND | 44 | 8 | 4 | 4 | 0 | 36 / 48 |
| 2 | POINT | 44 | 8 | 8 | 0 | 4 | 36 / 48 |
| 1 | DEND | 21 | 5 | 3 | 2 | 0 | 16 / 20 |
| 1 | POINT | 21 | 5 | 5 | 0 | 2 | 16 / 20 |

Parameters: 12h input weights +2h² recurrent weights +2h branch biases +h soma biases +2h output weights +2 output biases =2h²+17h+2. Continuous states: 2h branches +h somata +2 outputs; separately store h previous binary spikes and 2 count accumulators.

Forward dense MAC terms: 12h+2h²+2h. Backward terms: output-weight gradient 2h + previous-hidden gradient 2h + input-weight gradient 12h + recurrent-weight gradient 2h² + recurrent-hidden gradient 2h². No input gradient is needed.

| h | Arm | Updates | Training forward MAC | Training backward MAC | Nine-diagnostic MAC | Test MAC | Train wall seconds |
|---:|---|---:|---:|---:|---:|---:|---:|
| 6 | DEND | 1,000 | 479,232,000 | 737,280,000 | 21,565,440 | 9,584,640 | 4.703 |
| 6 | POINT | 1,000 | 479,232,000 | 737,280,000 | 21,565,440 | 9,584,640 | 4.848 |
| 4 | DEND | 1,000 | 270,336,000 | 393,216,000 | 12,165,120 | 5,406,720 | 4.274 |
| 4 | POINT | 1,000 | 270,336,000 | 393,216,000 | 12,165,120 | 5,406,720 | 4.320 |
| 2 | DEND | 1,000 | 110,592,000 | 147,456,000 | 4,976,640 | 2,211,840 | 3.652 |
| 2 | POINT | 1,000 | 110,592,000 | 147,456,000 | 4,976,640 | 2,211,840 | 3.799 |
| 1 | DEND | 1,000 | 49,152,000 | 61,440,000 | 2,211,840 | 983,040 | 3.261 |
| 1 | POINT | 1,000 | 49,152,000 | 61,440,000 | 2,211,840 | 983,040 | 3.401 |

Totals: 8,000 parameter updates; 1,818,624,000 training-forward and 2,678,784,000 training-backward dense MAC terms. Nine diagnostics per fit contribute 81,838,080; testing contributes 36,372,480. Final training metrics are already included in the ninth diagnostic.

Unique training episodes 640; unique held-out episodes 2,560; held-out model-episodes 20,480. The independent inferential units remain five seed/data blocks, not 20,480 independently trained models.

Training with artifacts: 32.398s. Test with artifacts: 3.964s. Test model calls: 0.243s.

Audits: verification: 8.469s, gradient_audit: 9.868s.

Limits:

- Matrix MAC counts exclude scalar nonlinearities, resets, loss, Adam, I/O and audit work.
- Same h matches parameters, continuous states, dense matrix terms, batch, optimizer and decoder between arms; represented cells, nonlinearities/reset derivatives and activation patterns differ.
- h counts hidden somata, not total neurons; POINT has 2h additional native branch cells and DEND has 2h graded compartments.
- Per-weight initialization distributions remain fixed; aggregate drive variance and per-coordinate mean-L2 strength change with size. This is a complete recipe/size comparison, not pure capacity isolation.
- Spike counts, dense event-delivery proxy, state variation and dimensionless latency are not energy, ATP, milliseconds or physical hardware efficiency.
- Wall times depend on CPU contention/compression; overlapping audit times are not added as elapsed workflow time.
