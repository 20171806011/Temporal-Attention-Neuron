# P1B manuscript correction draft

2026-11-02. Prepared against frozen repository commit `85213f0d417d54e4f48b301ed1c0e4200ef3c421`.

The P1B manuscript currently describes a different dynamical system and readout from Protocol V2.6 and its implementation. The proposed correction restores the registered contract while preserving the recorded results, counts, intervals and decision labels. This is a documentation correction, not a new experiment or retrospective change to the protocol.

Two independent proposals are supplied:

- [manuscript_corrections.patch](manuscript_corrections.patch) changes Paper2 section 8 and the matching overbroad collision sentence in its abstract. It corrects equations, readout, fixture scope, interventions and decision rules.
- [readme_scope_corrections.patch](readme_scope_corrections.patch) separately narrows the README's architectural-necessity interpretation. It is an optional interpretive revision, not a numerical or implementation correction.

Neither patch has been applied to the repository. Preview text is under [manuscript_preview](manuscript_preview/). Source paths and line numbers below refer to the original frozen files; the patch necessarily changes preview line numbers.

## Corrections to the registered contract

| Original manuscript location and issue | Proposed correction | Frozen primary source |
|---|---|---|
| [s8_composition.tex:161](../../Temporal-Attention-Neuron/snapshot/paper2/sections/s8_composition.tex#L161): an ODE with a voltage-driven gate and an accumulating `r` state | Give the actual hybrid Euler update: novelty drives the clipped gate; the membrane uses old `A`; `r` is a refractory countdown; threshold crossings reset voltage and set `r`. State the registered constants and M1's zero initialization. | [models_p1b_v2.py:54](../../Temporal-Attention-Neuron/p1_experiments/p1b/models_p1b_v2.py#L54), especially lines 80–106; Protocol V2.6 §§2–3, lines 24–56 |
| [s8_composition.tex:172](../../Temporal-Attention-Neuron/snapshot/paper2/sections/s8_composition.tex#L172): an affine readout of `r` at query arrival | Read out after 100 cue steps/10 ms: zero if refractory immediately before the deadline step; otherwise the pre-reset deadline candidate voltage, even when that step emits a spike. Query affects membrane dynamics but is not a direct affine-decoder argument. The two roles probe independent copies of the same checkpoint. | [models_p1b_v2.py:249](../../Temporal-Attention-Neuron/p1_experiments/p1b/models_p1b_v2.py#L249), lines 249–269; Protocol §§3.3,5.1, lines 67–74,112–124 |
| [s8_composition.tex:174](../../Temporal-Attention-Neuron/snapshot/paper2/sections/s8_composition.tex#L174): all subthreshold histories claimed erased because a voltage-threshold gate is inactive | Describe G1.5's particular fixed-key affine fiber. Both keys have positive novelty, and the fixture uses `wK=wV=.1`, unlike main-experiment unit weights. Compensation in the second value preserves ideal terminal state while changing the first target by 2.5. This is separate from the first zero-novelty event's erasure. Narrow the abstract accordingly. | [test_p1b_g1_fixtures.py:563](../../Temporal-Attention-Neuron/p1_experiments/p1b/test_p1b_g1_fixtures.py#L563), lines 563–625; [Protocol §8.1](../../Temporal-Attention-Neuron/p1_experiments/p1b/P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md#L229) |
| [s8_composition.tex:182](../../Temporal-Attention-Neuron/snapshot/paper2/sections/s8_composition.tex#L182): operand tolerance .05, with non-strict inequality | Restore strict operand error `< .50` with no abstention; joint success requires both operands. Keep the original .80/.40 prerequisite floors. Composition-risk threshold .05 is a distinct endpoint. | [Protocol §7.1](../../Temporal-Attention-Neuron/p1_experiments/p1b/P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md#L171), lines 171–176,186–197 |
| [s8_composition.tex:183](../../Temporal-Attention-Neuron/snapshot/paper2/sections/s8_composition.tex#L183): M2a described as a constant gate | M2a retains dynamic novelty-driven `A` and resets only `u,r` at active-event offsets. Constant `A0` belongs to M2b/M2d; M2d additionally resets `u,r`. | [Protocol §4](../../Temporal-Attention-Neuron/p1_experiments/p1b/P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md#L93), lines 93–106; model lines 121–133,167–175 |
| Same paragraph: TOST margin ±.05 and a claimed equivalence despite baseline failure | Restore the M1/M2a margin ±.005. State that the recorded 90% intervals are numerically inside this margin, but failed baseline prerequisites prevent confirmatory equivalence or functioning-memory redundancy. Preserve `FLOOR_LIMITED_COMPARISON` and `redundancy_inferred = False`. | [Protocol §7.4](../../Temporal-Attention-Neuron/p1_experiments/p1b/P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md#L199), lines 199–214; [runner_p1b.py:232](../../Temporal-Attention-Neuron/p1_experiments/p1b/runner_p1b.py#L232), lines 232–259 |
| [s8_composition.tex:186](../../Temporal-Attention-Neuron/snapshot/paper2/sections/s8_composition.tex#L186): a separate LIF conversion implied; limitation criterion given only as a mean error | M4 consumes the same physical carrier's cue-window spike indicators 1–99, excluding step100 and analog bypass. Silence is abstention with zero fallback. Registered limitation requires the lower bound of the central 95% interval to be ≥.05. Keep every reported CI and the original limitation labels. | [models_p1b_v2.py:271](../../Temporal-Attention-Neuron/p1_experiments/p1b/models_p1b_v2.py#L271), lines 271–303; Protocol §§5.2,7.5; [runner_p1b.py:266](../../Temporal-Attention-Neuron/p1_experiments/p1b/runner_p1b.py#L266), lines 266–276 |

For G1.5, the ideal-state equality argument and floating-point verification remain explicitly distinct. The implementation verifies a residual below `1e-10`; that numerical inequality alone is not literal state equality and does not prove that an arbitrarily precise decoder cannot distinguish floating states. The ideal recurrence's exact compensated fiber, the common query and the target gap support the stated ideal pairwise worst-case error of at least 1.25. This construction is not a prevalence estimate over randomly drawn task histories.

The revised paragraph retains the original displayed calibration coefficients as rounded values, not newly estimated constants. It likewise preserves 4,000 histories, 96,000 model episodes, 192,000 role queries, all reported accuracy and interval values, and original scientific labels. The builder checks selected historical strings; the independent reviewer separately cross-checked their recorded sources. No historical run was rerun for this documentation task.

## Separate interpretive README proposal

[README.md:30](../../Temporal-Attention-Neuron/README.md#L30) describes the ladder as necessary transitions, and [README.md:37](../../Temporal-Attention-Neuron/README.md#L37) says P1B establishes the necessity of distributed circuits. The optional patch instead says the project constructs sufficient mechanisms within the tested families and characterizes model-specific carrier/readout failures. The scientific closure expressly excludes universal impossibility and population-coding necessity, while H3 remains deferred: [closure §6](../../Temporal-Attention-Neuron/p1_experiments/p1b/P1B_FINAL_SCIENTIFIC_CLOSURE_REPORT.md#L104), lines 104–111. These are scope corrections; they do not negate the reported negative results or establish that another architecture will work.

## Provenance

Frozen source SHA256 values:

| Source relative to repository | SHA256 |
|---|---|
| `snapshot/paper2/sections/s8_composition.tex` | `ed692809df97f6f627d33ba421c74ed30f1aa192e7fca5af7bc04071dc869a95` |
| `snapshot/paper2/main.tex` | `b10865a3ecec8f2b48897dfab13d1a01062fcbb41ff26c7a8e6a447b8dfdb575` |
| `README.md` | `bb195ff9334b003490e045815ef7c9969763d9d610119c1da418f2588bbf80e8` |
| `p1_experiments/p1b/models_p1b_v2.py` | `6b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f` |
| `p1_experiments/p1b/P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md` | `07d58a2de773ea13e8fd5c16901cb8e8a370219d4793f4a7f9baca23c747a82b` |
| `p1_experiments/p1b/runner_p1b.py` | `47609f1e70ab8de2966ee8305757f44658ef799918d5f92108ac4efe64304c26` |
| `p1_experiments/p1b/test_p1b_g1_fixtures.py` | `8a56690f74ebd2e76dc39ab35b059da9beda77a4671f59094993a12905d2ff23` |
| `p1_experiments/p1b/P1B_FINAL_SCIENTIFIC_CLOSURE_REPORT.md` | `b9830aef1eebe517524df637657b3ab6b216ec78783275990a5b137a1f0a9f3e` |

Patch SHA256:

- Manuscript: `39320870b516e99f74460410d5a8953d8ceb7e013d5039b008fe79dfe3b1803e`.
- Optional README: `f7fab604304533229c48cc95ed89ee6966751b54ed20d0854d47dd14a3f2ea9c`.

## Completed validation and reproducibility

From the frozen repository, both commands completed with exit 0 and no diagnostics:

```sh
git apply --check --whitespace=error-all tan_audit/stage_localization/manuscript_corrections.patch
git apply --check --whitespace=error-all tan_audit/stage_localization/readme_scope_corrections.patch
```

The replacement paragraphs compiled with installed `pdflatex -interaction=nonstopmode -halt-on-error` in an isolated `/tmp` output directory, exit 0. This verifies the replacement text and equation syntax, not the full manuscript layout or a release PDF. The expanded explanation may change pagination; full manuscript rendering and page-count review remain release-stage work, outside this patch assessment.

[PATCH_MANIFEST.json](PATCH_MANIFEST.json) records source/preview/patch hashes and preserved-string checks. [PATCH_VALIDATION.json](PATCH_VALIDATION.json) records actual commands, exit codes, before/after source hashes and a clean `git status --short`. [patch_review.md](patch_review.md) is the independent source review. [manuscript_latex_validation.log](manuscript_latex_validation.log) is the completed syntax-build log.

To regenerate the proposals and their validation in this audit workspace:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/stage_localization/prepare_manuscript_patch.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/stage_localization/validate_manuscript_patch.py
```

The patches, draft, previews and scientific validation evidence are complete. Full release-layout validation remains outside this scope; original repository and historical numerical outputs were not modified.
