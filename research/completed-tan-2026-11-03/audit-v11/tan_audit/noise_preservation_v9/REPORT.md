# Original Task B: preserved step evidence, limited noise attribution

Scientific continuation, 2026-11-02. Original repository commit `85213f0d417d54e4f48b301ed1c0e4200ef3c421`. Original files, historical numerical results and labels remain unchanged. This report adds one preregistered control study to the original noisy-step experiment; it does not change P1A, P1B or the neuronal parameters.

**Result:** the original TAN response is not explained by complete output collapse in this task. On 512 fresh paired noise histories, it detects every step and makes no catch false alarm under the fixed five-step readout. Native LIF also detects every step, with five catch false alarms. TAN's lower late output therefore coexists with preserved evidence of this particular transition. However, the historical `false_spike_rate` is a plateau activity score, not an identification of noise-caused errors. Its clean-input baseline, the silent null, and the actual output-budget mismatch are essential to interpretation.

## What was measured and reproduced

The authoritative [original runner](../../Temporal-Attention-Neuron/snapshot/code/experiments/statistical_validation.py) uses 60 discrete updates: base zero for indices0–14 and1.5 for15–59, Gaussian noise SD.4 added before lower clipping at0. LIF uses leak.8 and threshold2. TAN uses window10, leak.8, threshold1, signed surprise in attention and `tanh(abs(surprise))` gating. The current sample enters the rolling window before the surprise is computed. All states start at zero. This is the original signed-surprise branch, not the rectified branch used in the module ablation.

The exact original endpoint is `mean(spikes[20:60])`. It includes four mixed pre/post-onset history windows, indices20–23. It supplies no desired spike train or label identifying which spike was caused by noise. Deleting outputs can only improve it, and complete silence attains its exact optimum0.

Phase A ran the exact extracted source bodies at every original seed0–49. **All100 per-model/seed records, including both rate and total spikes, match the frozen raw table exactly.** Model means/sample SD reproduce as LIF.5065/.0256596642 and TAN.2645/.0339154110. `PHASE_A_COMPLETE.json` was written before fresh inputs and supplementary diagnostics. The historical48% reduction remains valid as a reduction in this endpoint. A separately labeled whole-seed paired bootstrap gives TAN−LIF mean−.2420, pointwise95% interval[−.2525,−.2310]; historical tests and labels were not replaced. The source already paired models through identical seeds, while the original reported tests treated samples as independent; the manuscript had acknowledged that limitation.

## Frozen preservation control

[PREREGISTRATION.md](PREREGISTRATION.md) was saved before execution; SHA256 `70a9fcebb77a223e321e34fa11eb18f1228431c3d3ef8ac40cfc1a4706eb87e1`. Exactly512 new independent Gaussian paths use RandomState2026110226. Each path is delivered both with the original step and with an all-zero no-step base, using the same noise before clipping. Both native models see both waveforms from reset. Inputs were saved and hashed before evaluation. No thresholds, windows, amplitude, noise level or parameters were tuned.

The new external readout predicts a step if at least one spike occurs in `[15:20)`, the five onset samples excluded by the old endpoint. It is granted the known onset clock and a five-step counter. It does not recover amplitude or an unknown event time. Two extra observed-output controls are an all-zero trace and native LIF with independent constant output thinning. The keep probability529/1013 comes from the exact saved historical means. Thinning is label-blind, applies at every time, and preserves native LIF's internal updates/reset/generated spikes; the independent uniforms are shared between each history's step/catch counterparts. There are no outcome-dependent exclusions.

| Observed output | Noisy-step late rate R | Clean-step R | Step misses /512 | Catch false alarms /512 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Original TAN | .272900391 | .025000000 | 0 | 0 | 1.000000000 |
| Original LIF | .503759766 | .500000000 | 0 | 5 | .995117188 |
| LIF output thinning | .264453125 | .261105627* | 88 | 3 | .911132813 |
| Zero output | .000000000 | .000000000 | 512 | 0 | .500000000 |

*Thinned clean quantities are exact expectations under the frozen deletion probability, not replicated clean trajectories. Native clean controls were run once per model/condition. TAN clean onset contains4 spikes and full trace5; LIF clean onset contains2 and full trace22. Both clean catch outputs are silent. TAN's noisy catch late rate is.000390625, so no onset false alarms does not mean the entire catch trace is silent. All catch/full-trial counts and resources are included in the complete ledger.

The observed TAN accuracy1 is a finite-sample result, not proof of zero population error. Degenerate empirical-bootstrap intervals do not address unseen failure probabilities. This known-onset, one-amplitude/one-noise task is narrow and nearly saturated for both native models.

## All frozen contrasts, including unfavorable ones

Each of2000 bootstrap replicates resamples whole noise-history groups, keeping both classes, all arms and the associated thinning mask together. The following are **pointwise** descriptive95% percentile intervals, without simultaneous/familywise coverage or multiplicity-adjusted claims. The draws, all replicate values and all per-history values are saved.

| TAN comparison | Mean difference | Pointwise95% interval |
|---|---:|---:|
| Balanced accuracy minus zero output | +.500000000 | [.500000000,.500000000] |
| Balanced accuracy minus native LIF | +.004882813 | [.000976563,.009765625] |
| Balanced accuracy minus thinned LIF | +.088867188 | [.072265625,.105468750] |
| Noisy-step late R minus native LIF | −.230859375 | [−.234765625,−.226855469] |
| Noisy-step late R minus thinned LIF | +.008447266 | [.002537842,.014843750] |
| Noisy-minus-clean R change minus native LIF's change | +.244140625 | [.240234375,.248144531] |

The thinning comparison rejects complete suppression as a description of native TAN and shows better onset preservation than this particular indiscriminate deletion rule. **It is not an equal-fresh-budget superiority result.** TAN uses more late observed events than the thinned control: .272900391 versus.264453125, a difference.008447266 (about3.19% of the thinned rate). Mean full-trial observed counts also differ:13.837891 for TAN,22.699219 for LIF and11.900391 for thinned LIF. The frozen rule matched the historical expected late event budget only; it was not re-fitted to fresh outcomes or full-trial totals.

## Why the original score is not a noise-error measure

For clean LIF, the native recurrence produces candidate1.5, then2.7, then reset, repeating after onset. Its spikes are exactly16,18,…,58;20 occur in the old scoring window. Thus R=.5 even without injected noise. The simulation and an algebraic certificate agree. For clean TAN, its filled constant window gives surprise/gate/current0 from index24; the single clean spike inside the old scoring interval is part of the adaptation transient.

Adding noise changes mean step R by **+.247900391 for TAN** and **+.003759766 for LIF** relative to their own clean references. The much lower noisy TAN rate therefore must not be described as an equivalent fraction of independently identified noise errors removed. The clean models use different response strategies: sparse transition response versus tonic plateau firing. The noisy-minus-clean quantity is itself only a signed, model-specific change in rate, incorporating clipping and nonlinear dynamics; it is not a decomposition labeling individual spikes.

The supported benefit is selective signaling of this transition with less late activity under the frozen settings. The study does not establish signal-amplitude fidelity, preservation of arbitrary waveforms, unknown-onset detection, an optimal decoder, attention's isolated causal necessity or general noise robustness. Native LIF's strong detection is retained as positive comparator evidence.

## Resources and source fidelity

The same external stimulus and leak do not match internal gain or physical energy. TAN stores ten history samples plus its membrane and performs attention arithmetic; LIF stores its membrane. Thresholds differ, and TAN transforms the input into an effective current. No hardware or physical timestep/energy model is supplied. Per-trial current/state proxy sums, peaks, generated events and observed events are saved, with no ATP or joule interpretation.

For example, on noisy step histories, mean `sum(abs(current))` is22.72384 for TAN versus69.92294 for LIF; mean `sum(current²)` is17.81114 versus109.79001, and mean `sum(candidate²)` is38.83340 versus225.96107. These differently scaled internal quantities do not measure total energy. Thinned LIF retains the LIF computational/physical trace and generated22.699219 spikes on average while exposing11.900391. Zero output is a mathematical null with no assigned physical-energy implementation.

[ORIGINAL_CLAIM_CORRECTIONS.md](ORIGINAL_CLAIM_CORRECTIONS.md) supplies source-cited corrections. The original Task B signed branch and rectified ablation branch have saved full-model rates.2645 and.1415 respectively; their module effects must remain branch-specific. The no-gate ablation injects.5 times the context, not unscaled context. `stats.ranksums` is correctly called a Wilcoxon rank-sum test in the manuscript; it is not a paired signed-rank test. The zero-output limitation and standard signal-detection control are not new general theory. This continuation's contribution is the frozen-task reproduction and completed controlled evidence. P1A's separate positive addressing evidence remains intact.

## Execution, independent validation and deliverables

Primary completed command: `PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/noise_preservation_v9/run_noise_preservation.py`. Runtime approximately4.55seconds; Python3.12.14, NumPy2.3.5. Exact-class and separately instrumented states/history/spikes agree on100 original,2048 fresh and4 clean native trajectories:2152 trajectories and129120 discrete updates, checked against an additional exact-source execution. Original calls add100 direct metric-run trajectories. Fresh observed output ledger has4096 rows; original and clean native ledgers add100 and4. All outputs and controls remain in the denominator.

Independent validation **passed** on its first trajectory-validation attempt:3295 checks and825 numerical comparisons, covering all2152 native trajectories,4096 fresh observed outputs/ledger rows, all4200 ledger rows overall, and all6×2000 fresh plus2000 historical contrast replicates. Every spike matches exactly; maximum floating residual is4.718447854656915e−16. The independent checker reconstructed equations, noise, masks, every native trajectory, observed controls, ledger fields and paired statistics without importing the primary simulator. Its completed command is `PYTHONDONTWRITEBYTECODE=1 python tan_audit/noise_preservation_v9/independent_review.py`. Primary artifacts and the clean source repository remained unchanged. REVIEW.md and INDEPENDENT_REVIEW_CHECKS.json preserve checks, hashes, initial scalar-only harness clarifications and execution evidence. Archive source snapshots omit.git and therefore report Git provenance as unavailable; source-hash checks remain mandatory. A separate isolated archive-style replay without.git passed both the copied checker and a fresh-directory primary rerun. All4485366 numeric array elements,532 JSON numeric values, bootstrap draws and scientific CSV bytes reproduced exactly; original primary/review hashes stayed unchanged. ARCHIVE_REPLAY_REVIEW.json records the commands and unavailable Git provenance.

Read [RESEARCH_AGENDA.md](RESEARCH_AGENDA.md) for the completed decision and next unresolved measurement question. Primary results are primary_results.json, original_results.json and clean_results.json. Full inputs/trajectories are the three input NPZs and corresponding trajectory NPZs; fresh_samples.npz, three CSV ledgers, original_call_results.csv, both bootstrap draw arrays and replicate CSVs preserve complete outcomes. PRIMARY_MANIFEST.json supplies hashes/schema and phase ordering. Fresh input SHA256 `4d94d62d97f672478d62baaf859437ee2f3a1fa8e73fcc7593f41541564d45d9`.

 The study stops at this fixed table and independent review. The older initial earlier archive transfer limitation is documented in prior reports; the sources needed for this study are local and verified, so it is not a blocker for this result.
