# Independent cross-check of source-contract errata and research-map scope

2026-11-02. This is a read-only cross-check against the frozen repository, independent of the author of PROJECT_MAP.md and MANUSCRIPT_CONTRACT_ERRATA.md. No model experiment, historical replay, publication or repository edit was performed.

The material errata are confirmed:

| Claim checked | Direct supporting source locations | Assessment |
|---|---|---|
| Paper2 describes a different P1-B state equation and readout | `snapshot/paper2/sections/s8_composition.tex:165–176` versus protocol `P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md:28–56,112–124` and `models_p1b_v2.py:78–107,249–267` | Confirmed. Registered A is novelty-driven; old A multiplies content in the membrane update; r is a refractory countdown, not an accumulating surprise-weighted carrier. Readout uses deadline membrane candidate/reset, not r at query onset. |
| Paper2 misdescribes G1.5 | Paper2 `s8_composition.tex:174–176`; protocol `:229–247` | Confirmed. The registered fixture uses positive-novelty keys 1.8/2.2, content weights scaled to .1, and compensated values. It is not a statement that all subthreshold differences are erased by a voltage-dependent gate. |
| Operand and equivalence tolerances are conflated | Paper2 `s8_composition.tex:182–183`; protocol `:173–176,213,221–223` | Confirmed. Operand success is strict error <.50 without abstention; equivalence margin is ±.005; .05 is the separate channel-composition threshold. |
| M2a is incorrectly called constant-gate | Paper2 `s8_composition.tex:183`; protocol `:93–106`; code `models_p1b_v2.py:121–133,167–175` | Confirmed. M2a keeps dynamic A and resets u/r at active-event offsets. M2b/M2d have constant A0. |
| README necessity wording exceeds the closure | `README.md:37`; closure report `:16,104–111`; Paper2 `s9_ladder.tex:20–25` and `s8_composition.tex:193` | Confirmed. Closure establishes model-specific insufficiency, leaves population-coding necessity unproved and H3 deferred; the ladder is explicitly sufficient, not necessary. README line30 additionally calls the ladder transitions necessary, another instance of the same overstatement. |

The source-level mismatch does not by itself invalidate the saved numerical outcomes. A separately reviewed documentation correction should preserve original data and labels. Neutral wording such as “inconsistent equations” or “a different model description” is preferable to implying knowledge of how the mismatch arose.

The research map correctly distinguishes the original window-based TAN from the closed P1-B carrier. `snapshot/manuscript/sections/model.tex:3,18–42` explicitly retains the input window and constructs query/key/value features from it. P1-B instead declares the complete recurrent state as (u,A,r) in protocol `:26–31`. Therefore the new P1-B transduction claim must not be generalized to all original TAN implementations or to the whole raw-array software API.

The map also points to the correct P1-A authority: `p1_experiments/results/run_p1a_rev2/P1A_SCIENTIFIC_REPORT.md` and the final baseline manifest. The header of the root `p1_experiments/P1A_SCIENTIFIC_REPORT.md` explicitly marks that file as superseded Run1. The map does not mistake that superseded report for the authoritative seven-model, 420,000-evaluation run. Its scalar-metric/vector-QK hard-choice statement is scoped to static addressing; it does not claim that this solves an online dynamical carrier.

No overclaim of new whole-suite replication was found in PROJECT_MAP.md. It explicitly says the current audit did not rerun the full original sensory/phototaxis suites, the full 60,000-trial P1-A ledger or a full historical 96,000-episode byte replay. The novelty-stratified M1 analysis is identified as a separate current diagnostic, whose outcomes belong in its own report. Earlier supplied-source and enriched-control results are labeled by their distinct assumptions and provenance.

This source crosscheck did not modify original repository files or historical numerical outputs.
