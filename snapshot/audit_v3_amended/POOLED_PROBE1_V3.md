# Pooled Probe-1 v3 (amended audit passed)

Pooling was executed after the amended audit table (see PER_SEED_AUDIT.md).  Per-seed detail and flags are preserved; pooled means are equal-weight means over seeds with seed variability reported; no cell was deleted.

| model | mean clean | mean distractor | per-seed distractor | notes |
| --- | ---: | ---: | --- | --- |
| B1 | 0.331 | 0.340 | [0.352, 0.306, 0.361] | approx. chance |
| B2 | 0.867 | 0.843 | [0.847, 0.841, 0.84] | strong stable positional-memory baseline |
| B3 | 0.375 | 0.337 | [0.371, 0.355, 0.285] | lo-blind in seed1 overall & seed3-clean cell (INVALID there); nominal bal3 not treated as healthy 3-class decoding |
| B4 | 0.328 | 0.389 | [0.427, 0.398, 0.342] | context-dependent; absolute elevation attenuates in seed3 (dist ~ chance there) |

## B4 vs B3, distractor condition (pooled trials, n=6000)
- pooled difference: **+0.0570** (95% CI [+0.0387, +0.0758])
- per-seed differences: [0.0561, 0.0425, 0.0562]
- B4 absolute distractor-condition bal3 across seeds: 0.427 / 0.398 / 0.342
  (attenuates; seed 20260906 ~ chance); the B4-B3 contrast itself is
  consistent across seeds (+0.043 .. +0.056).
- interpretation (constrained): Full TAN shows evidence of improved
  target-relevant retrieval under content-rich temporal context relative
  to the no-attention TAN control; the absolute effect is partially
  replicated across seeds, with substantially weaker evidence in seed
  20260906.
- NOT claimed: distractor robustness / interference resistance / noise
  robustness; B3 is not a clean baseline (lo-blind cells); B2 remains the
  strongest model on this fixed-position retrieval task.
