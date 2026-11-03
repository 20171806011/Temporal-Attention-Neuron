# Original Task B source and provenance review

Source-only audit of frozen repository `85213f0d417d54e4f48b301ed1c0e4200ef3c421`. No neuron simulation, new metric, or significance test was executed for this review. Source/table inspection, hashing, and static AST extraction were completed. Existing result labels are preserved.

The original measurable endpoint is **spikes per step during a fixed part of a noisy constant-light plateau**. Calling that number a false-spike rate does not make it a selective-noise-rejection measure. The implementation contains no ground-truth false-event labels, clean-reference subtraction, absent-step control, or signal-recovery score. An always-silent output has the optimal score of zero and preserves no signal. This logical limitation does not negate the observed lower spiking or establish that TAN fails to preserve its onset response; that question requires the new control.

## Exact original contract

The authoritative runner is [statistical_validation.py:174](../../Temporal-Attention-Neuron/snapshot/code/experiments/statistical_validation.py#L174), with neuron definitions at [73](../../Temporal-Attention-Neuron/snapshot/code/experiments/statistical_validation.py#L73) and [89](../../Temporal-Attention-Neuron/snapshot/code/experiments/statistical_validation.py#L89).

| Component | Frozen implementation |
|---|---|
| Replicates | 50 seeds, integers 0–49. Each model call resets NumPy's legacy global RNG with that seed. |
| Time and signal | 60 discrete steps; base is zero at indices 0–14 and 1.5 at 15–59. No physical timestep is specified here. |
| Noise and clipping | 60 independent Gaussian draws with standard deviation 0.4; add to base, then clip the resulting input below at zero. The pre-step input therefore has rectified noise with positive mean; the actually delivered noise is not zero-mean additive Gaussian after clipping. |
| LIF | Initial membrane 0; leak 0.8, threshold 2; update `h = .8*h + input`, strict `h > 2` spike and reset to 0. |
| TAN | Initial membrane 0 and ten-zero history; window 10, leak .8, threshold 1, noise tolerance 0, beta 1, gating enabled, asymmetric rectification **disabled**. |
| TAN update | Shift history and include the current input; signed surprise is current input minus this updated window mean. Query is twice signed surprise; keys/values are the history. Stable softmax has denominator `sum(exp)+1e-9`. Current is `tanh(abs(surprise))*context`. Strict threshold crossing resets membrane to 0. |
| Primary endpoint | `mean(spikes[20:60])`, 40 steps, historically named `false_spike_rate`; lower is labeled better. Secondary returned field is total spikes over all 60 steps. |

The window first contains only post-onset samples at index 24. The scored interval includes four mixed pre/post-onset windows, indices 20–23. It therefore cannot simply be described as excluding all history-window adaptation. This indexing statement does not bound subsequent membrane settling.

These parameters differ from the later rectified Probe 1 and from the navigation models. Neither the later W=5, threshold=.5 model nor the navigation dead-zone .6 should be substituted. At a noiseless constant input 1.5, the Task B LIF's post-reset values are 1.5 then 2.7; its two-step tonic-spiking cycle is an algebraic prediction. Thus plateau spiking need not be noise-induced even for the baseline. A clean-reference replay can test this prediction without recalibrating it.

## Existing tables and inference

`NOISE_SOURCE_MANIFEST.json` records hashes, table coverage, original source-node line ranges, and exact extraction hashes. All six `snapshot/data/experiment_results` versus `snapshot/results/tables` pairs are byte-identical: statistical-validation, baseline, and ablation metrics and summaries. Duplicate tables are duplicate evidence, not independent replications.

The statistical-validation raw table has exactly 100 Task B rows: two models × each seed 0–49, with no missing or repeated model/seed pair. The saved summary reports:

| Existing saved field | LIF | TAN |
|---|---:|---:|
| Mean `false_spike_rate` | 0.5065 | 0.2645 |
| Sample standard deviation | 0.0256596642 | 0.0339154110 |
| Reported bootstrap interval | [0.4995, 0.5140] | [0.2555, 0.2740] |

These are table reads, not newly reproduced simulations. Saved Welch p is `7.225941479519349e-60`; saved Cohen d is `-8.047301626612212`. The field called `wilcoxon_p` is calculated using **`scipy.stats.ranksums`**, not Wilcoxon signed-rank ([statistical_validation.py:419](../../Temporal-Attention-Neuron/snapshot/code/experiments/statistical_validation.py#L419)). Model pairs receive identical noise because the runner reseeds before each model call. A new inference should respect seed-level pairing; timepoints within a trajectory are not independent replicates. Existing per-model bootstrap intervals are not paired confidence intervals for the difference. The manuscript already acknowledges unperformed paired analysis.

The saved raw-table SHA256 is `9709f92bdfdafcc55fe7f2109dfa0e3e9f0ea8ba383bb8563a6de3ee17bb52e9`; summary SHA256 is `c106904ab62bea991b6b2a355e57caac519ba21967408aa3a2564ab07af162c0`.

## Calibration and ablation limits

The baseline-comparison Task B runner preserves the LIF and TAN parameters above ([baseline_comparison.py:447](../../Temporal-Attention-Neuron/snapshot/code/experiments/baseline_comparison.py#L447)). Its RNN/GRU comparisons imitate a TAN trajectory ([279](../../Temporal-Attention-Neuron/snapshot/code/experiments/baseline_comparison.py#L279)); a smaller plateau spike count alone does not establish preserved imitation fidelity or better denoising.

The Task B ablation baseline is a **different TAN branch**: [ablation.py:155](../../Temporal-Attention-Neuron/snapshot/code/experiments/ablation.py#L155) always rectifies surprise to `max(0, S_raw − tolerance)`. There is no `use_asym_rect=False` route. Its Task B runner uses the same nominal W/leak/threshold ([317](../../Temporal-Attention-Neuron/snapshot/code/experiments/ablation.py#L317)), but the saved `ablation_summary.csv` full mean is **0.1415**, compared with **0.2645** for the original signed-surprise branch. The factory comment saying its defaults match Parts 1/2 does not make these Task B implementations equal. Ablation effects cannot directly localize the original Task B mechanism without accounting for this branch change. Its no-gate variant also scales context by .5 ([187](../../Temporal-Attention-Neuron/snapshot/code/experiments/ablation.py#L187)); it is not a clean removal with all injection gains fixed.

Task B thresholds are hand-set and the effective inputs differ: LIF receives raw input, whereas TAN receives a transformed attention current. Shared leak and shared raw stimulus do not establish matched input gain, membrane excursion, output rate, computation, state count, or energy. TAN additionally stores ten history samples and performs softmax. Spike count can be an output-activity proxy, but should not be called measured total energy. Task A's separately acknowledged non-firing LIF calibration problem must not be transferred unmodified to Task B.

## Intended benefit, controls, and failure interpretations

The manuscript explicitly describes preserving the true transition while suppressing stationary fluctuations; see the separately completed [NOISE_CLAIM_SCOPE_REVIEW.md](NOISE_CLAIM_SCOPE_REVIEW.md). That makes **step/onset evidence retained in the output** the relevant companion to plateau suppression, not an unrelated retrieval benchmark. Within the reviewed runners, associated summaries, manuscript sections, ERRATA, and claim ledger, no explicit Task B zero-output null or multiseed signal-preservation metric was found. The prior v8 agenda already identified the silent-output loophole; it is not a new discovery of this review.

A narrow preregistered control can retain the exact signal scale, noise law, timing, model thresholds, and initialization while adding an absent-step counterpart using the same noise draw. A fixed onset-window output score then tests whether the physical step remains distinguishable; a threshold-free score avoids selecting a favored detection threshold. The always-zero output should have chance discrimination and zero original plateau score. Clean-reference trajectories distinguish deterministic response from effects introduced by noise. Any gain/output-budget control needs a declared calibration partition or a clearly labeled oracle/descriptive role; matching numerically equal currents or noise units is insufficient across transformed state coordinates.

Interpretation limits: lower original rate plus retained detection supports suppression with a measurable retained signal **under this fixed stimulus**. Lower rate plus lost detection supports gain-collapse as an adequate explanation. A successful gain-matched simpler control weakens uniqueness claims. Failure on a fixed readout does not prove the full spike train or state lacks information. No outcome from these controls establishes broad noise rejection, optimal energy efficiency, or general task-relevant memory.

## Safe extraction and completion evidence

Importing the whole original file triggers font setup, directory creation under `outputs/stage2_part1`, and log writes before any runner is called ([statistical_validation.py:32](../../Temporal-Attention-Neuron/snapshot/code/experiments/statistical_validation.py#L32)). The audit therefore parsed it as text and copied only `LIFNeuron`, `TANNeuron`, and `run_experiment_B`. Their only external runtime dependency is NumPy (`np`), plus Python builtins. Original RNG semantics and softmax epsilon are preserved; no model was invoked.

Completed command: `python tan_audit/noise_preservation_v9/build_source_manifest.py`, exit code 0. It statically compiled selected AST nodes, wrote `taskb_original_extracted.py`, and confirmed exact 50×2 Task B coverage plus all six duplicate-table checks. It did not execute selected definitions, model calls, or statistical routines. 

No access denials or unresolved blockers. Earlier optional file searches encountered nonexistent guessed paths; the actual source was found through `rg --files`. Only audit-directory files were written.
