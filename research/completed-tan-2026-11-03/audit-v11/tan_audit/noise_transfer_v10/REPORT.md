# Finite-output signal and noise accounting on original Task B

Scientific version10 continuation, 2026-11-02. Frozen source commit `85213f0d417d54e4f48b301ed1c0e4200ef3c421`. This is one new preregistered batch with unchanged original neurons and v9 controls. All historical labels, earlier results and P1A positive evidence remain preserved.

**The completed result is a bounded readout finding.** For this fixed step/catch task, simplifying TAN's five-step spike count to a single “any spike” decision can discard substantial dependence on the Gaussian input path while retaining almost all of the binary target information. A conservative joint population certificate—not only a plug-in entropy estimate—supports the following statements simultaneously for both native models:

| Native model | Signal retained by D, at least | Signal lost from C to D, at most | Conditional path detail discarded, at least |
|---|---:|---:|---:|
| Native TAN | 0.970645 | 0.029355 | 0.503421 |
| Native LIF | 0.882846 | 0.117154 | 0.124389 |

Units are bits per modeled trial, under the imposed balanced presence/absence prior; the target contains at most1 bit. The simultaneous confidence is at least95% under the declared iid sampling model. These are bounds, not the empirical point estimates. Both native models admit nontrivial bounds. The coarsening is an external readout operation, not evidence that a neuronal threshold or attention module performed this information selection internally.

## Exact task and observation contract

The [original Task B source](../../Temporal-Attention-Neuron/snapshot/code/experiments/statistical_validation.py) uses a60-step waveform, a1.5 step beginning at index15, and independent Gaussian noise SD.4 added before lower clipping at0. Original LIF uses leak.8 and threshold2; original TAN uses window10, leak.8, threshold1, signed surprise in attention, magnitude surprise gating, and strict threshold/reset. Initial states are zero. Exact original source nodes and the completed v9 100-record replay were hash-verified; the historical replay was not relabeled as a new experiment.

The original task always presented the step. Here `S~Bernoulli(1/2)` specifies the **v9 diagnostic step/catch ensemble**, with no-step catch base0. It is not an estimated natural event prior or capacity-maximizing distribution. `N` is the entire60-coordinate pre-clipping Gaussian path, independent of S. Each of2048 fresh noise rows was run under both classes and both native models. The two counterfactual outputs are paired for inference only; a decoder sees one trial, never the pair, label, noise, state or thinning mask.

The observations are fixed: `C=sum(spikes[15:20])` on alphabet0..5, and `D=1[C>=1]`. The known onset clock and external counter are explicit resources. The possible onset is fixed at15; conditional on a step being present, its timing entropy is0. A variable defined as “no event versus event at15” would simply recode presence, not test variable timing. Count C also discards within-window spike timing. This study measures neither unknown-onset localization nor clean waveform/amplitude recovery.

Fresh inputs used RandomState2026110240 and were saved and hashed before model evaluation. Native controls include one clean step/catch trajectory per model. The fixed v9 LIF deletion probability remains529/1013. Its count law is integrated exactly as `Binomial(K,p)` conditional on native LIF count K; no new deletion masks or parameters were selected. This evaluates the same stochastic output channel using its marginalized distributions. Entropies of those distributions are not averages of mask-conditioned MI, and are not generally expected finite-sample MI over realized masks. The marginalization reduces evaluator Monte Carlo variation; it does not physically improve the control or expose a probability vector to the decoder.

## Finite-alphabet information, without estimating a noise density

For an observed output O, compute `I_S=I(S;O)`, `I_N=I(N;O|S)`, and `U_O=H(O|S,N)`. Native dynamics are deterministic given S and N, so their `U_O=0`. For thinned LIF, U is the known conditional entropy of the deletion channel. In all arms,

`H(O) = I_S + I_N + U_O`.

Only the six-bin count distribution and two-bin decision distribution are estimated. No60-dimensional density, KSG estimator, differential entropy, adaptive binning, smoothing, learned decoder or hyperparameter search is used. The identities are standard information theory, not information created by a deterministic neuron. [Shannon's original formulation](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) supplies the entropy framework; [the mathematical note](POPULATION_CERTIFICATE_PROOF.md) gives the self-contained confidence argument.

The allocation orders signal first and path dependence conditional on signal. It includes interactions and is not a unique causal allocation of signal-origin and noise-origin spikes. It does not estimate a percentage of continuous input noise removed, noise power, harmful events, or physical energy. The full noise path includes retained past noise; future coordinates after index19 are causally irrelevant to these outputs. Conditioning only on an instantaneous noise sample would omit relevant state history.

The complete empirical-law estimates are:

| Arm | Observed output | Signal I_S | Conditional path I_N | Deletion residual U |
|---|---|---:|---:|---:|
| Native TAN | Count C | 0.998221 | 0.918076 | 0.000000 |
| Native TAN | Decision D | 0.996962 | 0.003038 | 0.000000 |
| Native LIF | Count C | 0.996142 | 0.626843 | 0.000000 |
| Native LIF | Decision D | 0.928936 | 0.070775 | 0.000000 |
| LIF output thinning | Count C | 0.629695 | 0.087461 | 0.848635 |
| LIF output thinning | Decision D | 0.624460 | 0.041998 | 0.317303 |
| Zero output | Count C | 0.000000 | 0.000000 | 0.000000 |
| Zero output | Decision D | 0.000000 | 0.000000 | 0.000000 |

The count observation alone has **more** conditional path dependence for TAN than LIF (.918076 versus.626843 bits); the binary decision has **less** (.003038 versus.070775). These are descriptive point estimates for different declared observations, not an architecture-wide noise-filter ranking. The retained signal information must accompany either comparison.

For thinned LIF, conditional count entropy.936095 splits into.087461 bits of conditional input-path dependence and.848635 bits of deletion residual. Assigning the whole conditional entropy to environmental noise would conflate two sources. Silence has zero path information but also zero target information. Thus minimizing an isolated noise-information number does not establish useful filtering.

## Count detail, uncertainty and the population bounds

The new quantities are `Delta_signal=I(S;C)-I(S;D)`, `Delta_path=I(N;C|S)-I(N;D|S)`, and the corresponding deletion-residual difference. They sum to `H(C)-H(D)`. For deterministic native arms, `Delta_path=H(C|S,D)`. Re-expressing v9's binary confusion rates as MI alone would not have been a new result; this study instead measures what extra count detail contains.

| Arm | Signal detail lost | Conditional path detail discarded | Deletion detail discarded |
|---|---:|---:|---:|
| Native TAN | 0.001259 [0.000000, 0.003893] | 0.915038 [0.894466, 0.933190] | 0.000000 [0.000000, 0.000000] |
| Native LIF | 0.067207 [0.050928, 0.084103] | 0.556068 [0.540021, 0.571666] | 0.000000 [0.000000, 0.000000] |
| LIF output thinning | 0.005236 [0.003706, 0.006852] | 0.045462 [0.043622, 0.047296] | 0.531331 [0.524731, 0.537877] |
| Zero output | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] | 0.000000 [0.000000, 0.000000] |

Brackets here are **pointwise descriptive95% percentile bootstrap intervals**, from2000 whole-history resamples with seed2026110241. Every resample keeps both classes and all model laws together and reapplies the analytic deletion kernel. All52 information terms/differences and bootstrap bias diagnostics are saved. The diagnostics do not replace the plug-in estimator. An interval endpoint of−2.22e−16 is displayed as0 after rounding; every raw residual remains saved. Empty bins and sparse catch counts are disclosed, and resampling cannot discover unseen outcomes. General finite-sample entropy problems are established methodological issues ([Paninski, original analysis](https://www.cns.nyu.edu/pub/lcv/paninski-infoEst-2003.pdf)).

The leading certificate is separate from those bootstrap intervals. It allocates failure probability.025 to all four native model/class count histograms, using a union over62 nontrivial subsets per six-state alphabet. This gives total-variation radius.0473989164 and entropy-continuity allowance.3853001401 bits. A further.025 covers four exact one-sided binomial miss/false-alarm upper limits. With balanced error upper bound eU and `b=h2(eU)`, the error-variable identities bound both `H(S|D)` and `H(D|S)` by b. Combining the events gives simultaneous confidence at least.95:

`I(S;D)>=1-b`,

`0<=Delta_signal<=b`,

`Delta_path>=max(0,Hhat(C|S)-.3853001401-b)`.

[POPULATION_CERTIFICATE_PROOF.md](POPULATION_CERTIFICATE_PROOF.md) provides the subset-union, entropy-continuity and binomial-inversion derivations and primary references. The calculation accounts for unseen categories and does not require unbiased plug-in entropy or nondegenerate bootstrap intervals. The native-only certificate is not applied to fractionally integrated thinning probabilities. It supports a population statement about these fixed observables/prior; it does not certify biological noise robustness, a new mechanism or universal performance.

## Detection, timing, firing and resources remain distinct

Fresh TAN had2048/2048 step hits and **1/2048 catch false alarms**; native LIF had2048/2048 hits and41/2048 false alarms. Balanced accuracies were.999756 and.989990. This new TAN false alarm directly illustrates why v9's0/512 observation was not a population-perfection claim. The integrated thinned control has expected TPR.839790, FPR.010454 and balanced accuracy.914668; its fractional probabilities are not observed Bernoulli counts. Zero-output accuracy is.5.

Timing remains fixed and untested as a variable. Original late-rate R remains activity per step: TAN.268616, LIF.504541, thinned expectation.263477 and zero0. Relative to their own clean references, TAN rises by.243616 and LIF by.004541. These signed changes are not individual noise-caused spike labels. Fresh TAN late rate exceeds the thinned expectation by.005139; mean full-trial counts are13.732422 versus11.902745 for that comparison. The historical expected late budget was retained, not re-matched to fresh data.

Current/state sums and generated/observed counts are saved separately. For example, mean full-trial absolute-current sum is22.68918 for native TAN and69.97256 for LIF; thinned LIF retains its native internal computation. Different thresholds, transformed currents, ten TAN history samples plus membrane, attention arithmetic, and an external readout prevent a matched physical-gain/energy claim. No ATP, joules, bits-per-energy, or new biological state-count equivalence is inferred.

## Completed verification, artifacts and scope

Primary command `PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/noise_transfer_v10/run_noise_transfer.py` completed with exit0 in approximately14.79seconds. It retained8192 fresh native trajectories plus4 clean trajectories,491760 discrete updates, and16384 fresh observed-law ledger rows. The primary instrumentation checks exact frozen source classes at every update. Four clean trajectories match v9. No observations were excluded.

Independent command `PYTHONDONTWRITEBYTECODE=1 python tan_audit/noise_transfer_v10/independent_review.py` completed with exit0. **4341 checks and1163 comparisons passed:** all8196 native trajectories, full histories/attention/states/spikes, all ledger/resource laws,4096000 bootstrap indices,104000 bootstrap metric values, pmfs and native confidence certificates. Native float states and spikes matched exactly; the largest numerical discrepancy across all statistics/resources was5.33e−14, below the frozen1e−12 tolerance. All primary files and the clean source checkout remained unchanged. A separate archive-style replay without `.git` passed both the copied independent checker and a fresh-directory primary run. All16,159,416 numeric array elements,994 JSON numeric values, kernels, pmfs, bootstrap/certificate values and scientific CSV bytes reproduced exactly. Source/v9 dependency hashes stayed mandatory and Git provenance was explicitly unavailable. ARCHIVE_REPLAY_REVIEW.json records these commands and preserves original-file integrity. [REVIEW.md](REVIEW.md) and its JSON preserve the complete evidence.

Read [PREREGISTRATION.md](PREREGISTRATION.md), [CONTRACT_SCOPE_REVIEW.md](CONTRACT_SCOPE_REVIEW.md), [primary_results.json](primary_results.json), [population_certificate.json](population_certificate.json), [bootstrap_summary.json](bootstrap_summary.json), and [RESEARCH_AGENDA.md](RESEARCH_AGENDA.md). PRIMARY_MANIFEST.json supplies complete schemas/hashes. Inputs, full trajectories, per-history count laws, raw integer frequencies, all bootstrap draws/metrics/pmfs, resource ledgers, formula-only oracle/silent/kernel checks and independent scripts are included.

Protocol SHA256:`14ef73c1418b4dcea2a5a55de20391e3a8c82925bef70741ee3033ebcbfca179`. Fresh input SHA256:`9ea84952c119b8083e93d0ca7265e3cbc9d0927cb022a11cb9ade0b2c7bbfc15`. Python3.12.14, NumPy2.3.5, SciPy1.17.0; installed dependencies only. The separately saved W5 sinusoidal MI/cascade analysis is a different model/task and was not rerun or declared corrected here. No original labels, repository files or remote publication were changed. No current execution/input blocker remains.
