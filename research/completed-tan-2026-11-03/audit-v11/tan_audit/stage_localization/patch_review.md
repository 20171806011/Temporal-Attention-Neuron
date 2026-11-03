# Independent manuscript-patch source review

2026-11-02. Scientific review of the proposed correction by a separate reviewer. Frozen repository commit: `85213f0d417d54e4f48b301ed1c0e4200ef3c421`. The reviewed patches remain unapplied to the repository.

**Disposition: no blocking source-contract mismatch found in the reviewed patch versions.** Both patches passed an independent `git apply --check` against the frozen repository (exit 0); `git status --short` remained empty. The main patch corrects the manuscript contract and its abstract summary. The README change is separately reviewable and explicitly interpretive.

Reviewed patch SHA256:

- `manuscript_corrections.patch`: `39320870b516e99f74460410d5a8953d8ceb7e013d5039b008fe79dfe3b1803e`
- `readme_scope_corrections.patch`: `f7fab604304533229c48cc95ed89ee6966751b54ed20d0854d47dd14a3f2ea9c`

## Source-contract checks

Source paths below are relative to `Temporal-Attention-Neuron`; `p1b/` abbreviates `p1_experiments/p1b/`. These locations refer to the frozen originals, not shifted preview line numbers.

| Checked point | Independent source evidence | Assessment |
|---|---|---|
| Hybrid Euler update and state meaning | `p1b/models_p1b_v2.py:54–106`; protocol V2.6 `:24–56` | Corrected equations use old A for membrane current, update A even during refractory time, implement threshold/reset, and identify r as a refractory countdown. Unit coefficients and rest/bias terms are specialized consistently with the frozen constants. |
| Query and analog deadline | `p1b/models_p1b_v2.py:221–269`; protocol `:112–124` | Correct: query drives the membrane for 100 steps/10 ms. Cue-local r99 decides whether readout is zero; otherwise readout is the pre-reset deadline candidate, including a threshold crossing on step100. Query is excluded as a direct affine-decoder argument. |
| Spike eligibility and source | `p1b/models_p1b_v2.py:271–303`; protocol `:126–143` | Correct: same physical carrier/query simulation, indicators 1–99 only, no analog bypass. Silence retains zero fallback and abstention. The patch avoids implying a new downstream LIF simulation. |
| G1.5 scope | protocol `:229–247`; `p1b/test_p1b_g1_fixtures.py:563–625` | Correct: keys 1.8/2.2 have positive novelty; fixture content weights are .1, separate from main weights 1. Fixed-key gate yields an affine voltage in values on a no-spike branch. Compensated changes give ideal state equality; executable floating residual <1e-10 is stated separately. Target gap 2.5 implies ideal common-output worst-case error >=1.25. No universal all-subthreshold erasure or population prevalence is asserted. |
| M2a intervention | `p1b/models_p1b_v2.py:121–133,167–175`; protocol `:93–106` | Correct: M2a keeps the dynamic gate and resets u/r after active events. Constant A0 belongs to M2b/M2d; M2d additionally resets u/r. |
| Operand tolerance, comparison margin and decision order | protocol `:171–176,186–214`; `p1b/runner_p1b.py:232–259` | Correct: strict operand error <.50 without abstention, both roles for joint success; TOST 90% interval inside (-.005,.005) is only numerical evidence here because failed baseline takes precedence. All cells remain FLOOR_LIMITED_COMPARISON with redundancy_inferred=False. |
| Channel limitation and interpretation | protocol `:216–223`; `p1b/runner_p1b.py:266–276`; closure `:104–111` | Correct: lower 95% interval bound >=.05 is the limitation rule. Findings apply to tested end-to-end carrier/readout implementations and do not isolate spike discretization as the exclusive bottleneck. |
| Historical values and scope | original `snapshot/paper2/sections/s8_composition.tex:179–193`; closure `:68–100` | Displayed counts, coefficients, accuracy/CI values, abstention ranges and original decision labels are retained. The earlier static-combiner paragraph is preserved. A separate check found all 39 independently selected historical strings in the preview. |
| Optional README correction | original `README.md:30,37`; closure `:16,104–111`; `snapshot/paper2/sections/s9_ladder.tex:20–25` | Appropriately separated interpretive narrowing: tested sufficient ladder and model-specific failures do not establish general distributed-coding necessity. |

## Provenance and validation limits

The reviewer directly read the original model, relevant fixture and runner code, Protocol V2.6, closure text, original paper paragraphs, prior errata/cross-check, both generated patches and patch manifest. This review does not claim to rerun the historical 96,000 episodes or independently reproduce their numbers. The numbers are preserved and cross-checked against their recorded source, not newly estimated.

Independent completed commands:

```text
git -C Temporal-Attention-Neuron apply --check tan_audit/stage_localization/manuscript_corrections.patch
git -C Temporal-Attention-Neuron apply --check tan_audit/stage_localization/readme_scope_corrections.patch
```

Both returned exit 0. A Python read-only check confirmed the 39 selected historical strings and balanced equation/aligned/cases delimiters in the preview. No PDF compilation or visual layout review was performed; syntax delimiter checks are not a substitute for a release build. No repository, historical output or proposed-patch mutation was performed by this reviewer. 

Frozen source SHA256:

| Source | SHA256 |
|---|---|
| `p1b/models_p1b_v2.py` | `6b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f` |
| `p1b/P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md` | `07d58a2de773ea13e8fd5c16901cb8e8a370219d4793f4a7f9baca23c747a82b` |
| `p1b/P1B_FINAL_SCIENTIFIC_CLOSURE_REPORT.md` | `b9830aef1eebe517524df637657b3ab6b216ec78783275990a5b137a1f0a9f3e` |
| `p1b/test_p1b_g1_fixtures.py` | `8a56690f74ebd2e76dc39ab35b059da9beda77a4671f59094993a12905d2ff23` |
| `p1b/runner_p1b.py` | `47609f1e70ab8de2966ee8305757f44658ef799918d5f92108ac4efe64304c26` |
| `snapshot/paper2/sections/s8_composition.tex` | `ed692809df97f6f627d33ba421c74ed30f1aa192e7fca5af7bc04071dc869a95` |

This disposition is tied to the patch hashes above. Any later patch change should receive a correspondingly scoped recheck.

## Narrow follow-up recheck

The author subsequently clarified that the cell-specific TOST comparison is **M1 versus M2a**. The revised manuscript patch hash recorded above is `39320870b516e99f74460410d5a8953d8ceb7e013d5039b008fe79dfe3b1803e`. The reviewer reversed only that wording and the resulting hunk-line count in memory and recovered the previously reviewed patch SHA256 `e062b9372c69f80f4a52e2b0246a1702bf185fe05c7227f7bfcab85f1f5c8d53`; this verifies there were no other patch changes. The comparison identification matches `runner_p1b.py:225–239`. A fresh independent `git apply --check` returned exit 0 and repository status stayed clean. The historical-string check count is **39**, consistently recorded above. README patch is unchanged. The reviewer has not inspected a compiled PDF; any author-side fragment compilation is separately documented.
