# Broader manuscript review and proposed next check

2026-11-02; frozen commit `85213f0d417d54e4f48b301ed1c0e4200ef3c421`. This is source inspection and experimental design, not a new model/estimator run. No repository or prior result was changed.

## Recommendation

For the current next step, prioritize an exact target/distractor permutation witness in the original Probe-1 task, as proposed by the natural-collision review. It directly tests whether the native permitted response can expose the task's target-slot content, despite an ordered history window retaining that content. This is more directly connected to the project's retrieval boundary than further P1B fitting or an expanded estimator study. Root will choose and preregister the exact admissible histories and verification budget; this review has not run or certified the witness.

A separate unresolved source issue concerns the original manuscript's information-theoretic noise-filtering account. Its implementation mixes information units, has a questionable discrete bias correction and bin mapping, and treats same-time variables as a cascade despite omitted temporal state. This belongs in the deferred measurement agenda. The contribution of auditing it would be project-specific reproducibility, not a new information-theory result. It must not be presented as an empirically completed correction.

Natural collisions already answer a narrower state question: membrane alone is not sufficient when an explicit input window remains. The later paper correctly distinguishes geometric response changes from content selectivity. Repeating that generic distinction, extending collision-ratio searches, or fitting another slightly better P1B decoder is lower priority.

## Broader evidence and causal boundaries

1. `snapshot/paper2/sections/s3_state_geometry.tex:15–63` reports natural collisions, closed-gate and common-reset controls. Its lemma concerns the projection `(h,X) -> h`; the complete `(h,X)` remains a Markov state. This establishes omitted-history dependence, not robust retention of a desired variable or recovery of its value. The paper itself says this section proves nothing new and reruns nothing. The Phase-2 discussion at `:67–108` explicitly says covariance effective rank is not intrinsic dimension/capacity. The terminology **Phase 1 natural collision** should not be conflated with **P1-A Rev2 static address-identifiability**.
2. `CURRENT_RESEARCH_STATE.md:41–46` and `P0_HANDOFF.md:32–34` already record that B2 delay-line decoding is much stronger than B4 in the distractor task, that positive-scalar content routing failed its audit, and that phase-space changes do not establish routing. No new claim of attention-specific robust memory is justified from those facts.
3. The original manuscript makes a stronger noise-filtering claim: `snapshot/manuscript/sections/analysis.tex:68–118` compares internal-current and spike-output mutual information and calls thresholding the noise filter. Its later caution against exclusive causality does not repair invalid cross-estimator comparisons. Its table also reports more noise information in TAN spikes than LIF spikes; this is not itself a superiority measure.
4. Task B's endpoint is only the steady-period spike rate (`snapshot/code/experiments/statistical_validation.py:175–204`). A permanently silent output scores perfectly on this endpoint. The reported reduction alone therefore does not establish preservation of a signal relevant to a task. This is distinct from the already-disclosed Task-A decay-time flaw and LIF calibration limitation.
5. Existing P1B manuscript discrepancies are already present in repository `ERRATA.md:230` and neighboring ERR-18 entries. The v7 patch is useful implementation of an existing recorded correction, not a novel discovery of those discrepancies. The new agenda should distinguish that from genuinely unaddressed checks.

## Candidate 1: original task-observability witness

The frozen Probe-1 contract is explicit in `snapshot/code/experiments/temporal_distractor.py:13–26,77–95,139–144`: target at array index4 (paper step5), three distractors at indices5–7, query at index8 (step9), followed by zeros; the only allowed features are five pre-reset voltages and their spike indicators at indices8–12. The hidden class controls the target amplitude family; the current query is not the class label. Choose target values outside the overlapping amplitude interval [.95,1.05] so a low/high label collision does not merely reproduce generator label ambiguity.

At the query, a permutation of the four prior amplitudes leaves their sum/mean unchanged. With fixed current q it therefore leaves S unchanged. The original B4 context is the symmetric sum of elementwise exponentially weighted amplitudes divided by the sum of those weights, so it too is unchanged in ideal arithmetic; B3's mean context has the same symmetry. Attention weights themselves are permuted, not generally componentwise equal. Once pre-query h is genuinely matched by a common native reset, the query candidate and spike agree. Thereafter each zero input closes the rectified gate while the remaining window values are nonnegative, so the entire permitted post-query response can agree even while the ordered windows differ. These conditions need exact verification in the selected fixture, including reset reachability, positive query novelty, permitted target/distractor supports, strict spike semantics and floating residuals.

The full ordered state is not claimed equal: it contains the target in its designated slot. The proposed obstruction is to the declared response observation under this future stimulus, not to all state-only decoders or all possible later probing. Native B2's position-weighted delay-line current and the privileged raw-target-tap oracle are useful positive controls; symmetric B3 and full B4 test the observation obstruction. Artificially resetting h would be an intervention with extra control, so the strongest contract-preserving version verifies a reset that occurs under the original dynamics in both admissible histories. A constructed equal-prior pair permits a 1/2 classification-error lower bound for its common observation, not a prevalence or population-risk estimate for the original generator.

Permutation-invariant aggregation is standard; [Zaheer et al., Deep Sets, 2017](https://arxiv.org/abs/1703.06114) studies functions of sets and their invariance to input order. The project-specific contribution would be a task-valid, common-reset witness covering the complete allowed native response and useful controls. It is not a new general symmetry theorem, and it should not be confused with the already-known positive-scalar rank-order obstruction.

## Deferred information-analysis source path

The original executable is `snapshot/code/experiments/information_analysis.py`:

- `:452–491`: `mutual_information_ksg` returns a digamma expression without division by `ln(2)`.
- `:494–542`: `mutual_information_discrete` uses `log2`, then adds `(|X|-1)(|Y|-1)/(2n)` and clips below zero.
- `:572–614`: the principal Layer-4 trace has 3,000 steps; RandomState/global seed 1; fixed clean multi-frequency input; noise SD .4; clipped input; TAN W=5, leak .5, threshold .5, novelty tolerance0; LIF leak .5 and threshold2. The arrays are generated inside the function, not loaded from a hidden dataset.
- `:618–659`: KSG is applied to TAN S and A and **post-reset LIF h**; the discrete estimator is applied to spikes. Every result is written under `mi_bits`, with no conversion.
- `:718`: the assembled table is written as `mutual_info_summary.csv`; `:830–843` subtracts the unlike-unit A and Y estimates for headline “noise filtering”.
- `:1037–1046`: the downstream pass gate uses that same comparison.

The archived `snapshot/data/experiment_results/dynamics_mutual_info.csv` and `snapshot/results/tables/dynamics_mutual_info.csv` are byte-identical (SHA256 `9be71f3b3a9a461693115fa6fdcb3cf954f8f163b1d3980d8e946f8897f475a9`). Their first 15 rows agree with the manuscript's printed rounding; five more rows are the original noise-SD series. This establishes a consistent source/table path but is not yet numerical regeneration of the table.

No original 3,000-step Layer-4 input/state ledger was found. The similarly named `dynamics_state_space_trajectories.npz` contains only the separate Layer-3 arrays: habituation `(35,2)`, noise `(60,2)`, navigation `(150,2)`. A future replay must therefore be labeled reconstruction under the frozen source and installed numerical libraries, followed by comparison to the archived 20-row table; it cannot claim historical trace-byte identity.

The reusable module `snapshot/code/analysis/mutual_information.py` repeats the same unit, radius-count, bin and bias formulas. It is not a corrected alternative. Searches of the full repository's Markdown/TeX/Python and specifically `ERRATA.md`, `VALIDATION_REPORT.md`, `P0_HANDOFF.md` and the TAN research dossier found no recorded correction of these MI issues. This is a bounded search statement, not a guarantee about unseen conversations or archives.

## Mathematical issues to check, not retrofit

**Units.** The KSG paper defines natural-log units and Eq.8 uses the digamma expression implemented here; conversion to bits requires division by `ln(2)`. Thus the code's A/S values and Y values are not in the same units. This invalidates the numerical subtraction, but does not imply that every inequality changes sign after correction. [Kraskov, Stögbauer and Grassberger, 2004, primary journal text](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.69.066138).

**KSG radius convention.** That paper's first estimator uses strict marginal-neighbor counts at the joint kth-neighbor distance; the repository counts points within an inclusive radius while retaining the Eq.8 digamma formula. A deterministic neighbor-count fixture can expose the convention mismatch without treating one finite-sample calibration as a theorem about estimator quality. The spike/gate arrays also contain atoms; small jitter does not silently turn a mixed distribution into the smooth-density setting.

**Discrete correction.** Miller–Madow corrects each plug-in entropy by `(m-1)/(2N)` in nats. Combining corrected entropies gives, in bits, `I_MM = I_plugin + (mX+mY-mXY-1)/(2N ln2)`, where support counts must be stated. Under full Cartesian occupancy this becomes a *negative* correction `-(mX-1)(mY-1)/(2N ln2)`. The code instead always adds the latter magnitude without the bit conversion. These are standard formulas and finite-sample approximations, not guarantees of unbiasedness. [Paninski, 2003, author-hosted primary paper, pp.1197 and1203](https://www.cns.nyu.edu/pub/lcv/paninski-infoEst-2003.pdf).

**Bins.** `digitize(x, edges[:-1])`, followed by clipping to `0..n_bins-1`, leaves bin0 unused for inputs within the sample's min/max and merges the last two intervals. With `n_bins=2`, every in-range x maps to bin1. A balanced perfectly associated binary fixture is therefore a decisive positive control for the bin mapping; do not choose a replacement bin count based on neuronal outcomes.

**Temporal conditioning.** `y_t = 1[lambda*h_{t-1}+A_t > theta]`; A_t alone is not generally sufficient to determine y_t. Therefore `z_t -> A_t -> y_t` is not automatically a Markov chain, and a data-processing argument cannot attribute a difference of same-time MI values to thresholding. For LIF, the recorded h is after reset and is downstream of the spike decision; treating it as a pre-spike internal layer is also incorrect. A complete pre-threshold candidate and prior state must be distinguished from post-reset h. Correct units/estimators do not solve this causal problem.

## Candidate 2: deferred bounded calibration and frozen-trace audit

Before execution, root should freeze the exact seeds, controls, formulas and stopping rule. Suggested minimal scope:

1. Extract only the original class/function AST definitions needed for recurrence/estimators, avoiding top-level output paths under the original output directory; retain source hash and an independent implementation for comparison. Do not import the whole experiment, which creates directories and logging outputs on import.
2. Predeclare exact finite contingency-table controls: independent balanced binary variables, perfectly associated balanced binary variables, and constant output. Use fixed bin edges or the declared original/repair mapping. Separately test original mapping at `n_bins=2` to expose its collapse, and check general occupancy-count Miller–Madow algebra. Negative finite-sample corrected estimates must be saved before any optional nonnegative presentation clipping.
3. One modest fixed Gaussian calibration for the KSG unit/radius issue, with analytic target `I=-.5*ln(1-rho^2)`; e.g. rho0 and .8, fixed seed and sample size chosen before running. Report deviations; do not optimize k, jitter, normalization or sample count until a desired result appears. This validates implementation, not robustness of mixed-atom neural estimates.
4. Reconstruct the principal 3,000-step original TAN/LIF trace, save full pre/post-reset state and inputs, and reproduce all 15 principal original estimator outputs before applying separately named unit/bias/bin fixes. The five historical sigma points may be reconstructed solely for faithful saved-table verification if the budget allows; this is a fixed old table, not a new sweep. Preserve the original estimates and labels in separate files.
5. Save the causal dependency calculation for A_t, h_{t-1}, candidate, y_t and post-reset h. An exact same-A/different-prior-h fixture can refute the alleged current-only deterministic channel; it must be labeled a state intervention unless reachable from actual histories. Correct deterministic input/output positive control uses the *pre-reset candidate*, with threshold mapping verified exactly.

Prediction: source-level unit/bin/correction mismatches will be confirmed; the numerical direction and magnitude of corrected neural MI differences must be reported without assuming they will reverse. Inference: establish which reported comparisons are reproducible and well-defined, and whether they license selective-noise or threshold-causal language. Do not conclude that TAN has no useful behavior, that lower MI is always better, or that the repaired estimator is a complete biological noise-robustness test. Stop after this one fixed audit and independent verification.

## Provenance and completion

Read-only file hashes:

| Source | SHA256 |
|---|---|
| `snapshot/code/experiments/information_analysis.py` | `423f45fda1cc26b6405ee8eb963fd23ac0a4dc6d912f66e98f108a85f0881190` |
| `snapshot/code/analysis/mutual_information.py` | `f9a7c1e4ecffb72e79c9c713947951f9cc511e47adf6d120b1a4075ba5c4f250` |
| `snapshot/manuscript/sections/analysis.tex` | `34d886b12b97591b99314a01f7dd8194be3105c8fe1d0b2a879b4db9d79647b0` |
| `snapshot/paper2/sections/s3_state_geometry.tex` | `4a85efd07b7aa44d42704c79b126c4e78ac3e1fae17a8a0bdefca43c48a3ac16` |
| `ERRATA.md` | `dc140fb2a1c8e0c174b9444c8d3d5fa4e3a448d760dc7811b4fb59983987b72f` |

Primary literature pages were accessed on2026-11-02. No new model/estimator run, fitting, installation, repository modification or remote publication was performed for this review.  Historical Layer-4 trace bytes are absent; deterministic reconstruction and archived-table comparison are available if that deferred audit is later selected. Neither proposed experiment is represented as completed here.
