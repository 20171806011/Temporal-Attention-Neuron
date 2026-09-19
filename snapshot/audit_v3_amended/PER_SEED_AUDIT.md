# Per-seed amended audit table

| seed | model | condition | bal3 | shuffle status | coverage status | final |
| --- | --- | --- | ---: | --- | --- | --- |
| 20260904 | B1 | clean | 0.346 | PASS | PASS_WITH_FLAG | PASS_WITH_FLAG |
| 20260904 | B1 | dist | 0.352 | PASS | PASS_WITH_FLAG | PASS_WITH_FLAG |
| 20260904 | B2 | clean | 0.868 | PASS | PASS | PASS |
| 20260904 | B2 | dist | 0.847 | PASS | PASS | PASS |
| 20260904 | B3 | clean | 0.333 | PASS | PASS_WITH_FLAG | PASS_WITH_FLAG |
| 20260904 | B3 | dist | 0.371 | PASS | PASS_WITH_FLAG | PASS_WITH_FLAG |
| 20260904 | B4 | clean | 0.328 | PASS | PASS | PASS |
| 20260904 | B4 | dist | 0.427 | PASS | PASS | PASS |
| 20260905 | B1 | clean | 0.318 | PASS | PASS | PASS |
| 20260905 | B1 | dist | 0.306 | PASS | PASS | PASS |
| 20260905 | B2 | clean | 0.857 | PASS | PASS | PASS |
| 20260905 | B2 | dist | 0.841 | PASS | PASS | PASS |
| 20260905 | B3 | clean | 0.322 | PASS | PASS | PASS |
| 20260905 | B3 | dist | 0.355 | PASS | PASS | PASS |
| 20260905 | B4 | clean | 0.329 | PASS | PASS | PASS |
| 20260905 | B4 | dist | 0.398 | PASS | PASS | PASS |
| 20260906 | B1 | clean | 0.329 | PASS | PASS_WITH_FLAG | PASS_WITH_FLAG |
| 20260906 | B1 | dist | 0.361 | PASS | PASS_WITH_FLAG | PASS_WITH_FLAG |
| 20260906 | B2 | clean | 0.877 | PASS | PASS | PASS |
| 20260906 | B2 | dist | 0.840 | PASS | PASS | PASS |
| 20260906 | B3 | clean | 0.470 | PASS | INVALID_FOR_INTERPRETATION | INVALID_FOR_INTERPRETATION |
| 20260906 | B3 | dist | 0.285 | PASS | PASS | PASS |
| 20260906 | B4 | clean | 0.329 | PASS | PASS | PASS |
| 20260906 | B4 | dist | 0.342 | PASS | PASS | PASS |

- B3 lo-blind evidence: overall never-predicts-lo in 1/3 seeds (seed1 overall; seed3 clean cell INVALID_FOR_INTERPRETATION); not a >=2/3 persistent model-level flag by the overall-present criterion
- B4 seed-wise distractor-minus-clean: +0.099 / +0.069 / +0.013 (absolute elevation attenuates in seed 20260906, where B4 dist ~ chance)
- B1 cells at chance with collapse -> PASS_WITH_FLAG, excluded from claims
- B2 cells healthy (all classes predicted); shuffled nulls carry overfit tail -> QC limitation where flagged