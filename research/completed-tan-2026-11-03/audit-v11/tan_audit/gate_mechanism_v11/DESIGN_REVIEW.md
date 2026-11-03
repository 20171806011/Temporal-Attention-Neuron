# Final-round v11 design proposal

2026-11-02. Prospective design only: no calibration samples, model trajectories, or new outcomes had been generated. The frozen protocol governs execution; no further research family or sweep is proposed.

## Selected contrast

Use one temporal-correlation change, iid noise versus stationary Gaussian AR(1) with correlation parameter `.9`, and compare frozen native TAN against a simple, independently calibrated causal adaptive-filter/LIF baseline. Keep the original amplitude1.5, noise marginal SD.4, onset15,60 updates, lower clipping, zero initial model states, and five-step onset detector. There is no change-time variable or new timing-localization task.

This tests whether native TAN retains an advantage over ordinary adaptive filtering when nuisance persistence changes. It is not a causal isolation of TAN's gate: the two mechanisms still differ in context weighting, filtering dynamics and retained state. A clear non-superiority result is useful. If the ordinary filter matches or exceeds TAN, or the contrast remains unresolved, do not tune another baseline or make a new challenge.

## Source audit and why not calibrate the repository ALIF here

The actual repository AdaptiveLIF in `snapshot/code/experiments/baseline_comparison.py:107–136,460–462` uses membrane leak.8, initial/base threshold2, spike-triggered threshold increment.3 and threshold recovery.05. It compares the new candidate to the current threshold, resets/increments after a spike, then recovers threshold every update. It retains a membrane and adaptive threshold, so it does have temporal state. The original Task B TAN is the signed-surprise/magnitude-gated W10 implementation, not the separately rectified branch.

A one-parameter ALIF threshold calibration is possible on training data, but finite-window spike/reset/adaptation feedback does not justify assuming a monotonic late-rate objective. A threshold grid or optimizer would require a new search budget and a convention for possible calibration failure. For this final bounded round, prefer the following ordinary adaptive-filter control with a closed-form gain fit. **It is a new diagnostic baseline, not the repository's ALIF class or a historical registered result.** No original model or label is changed.

## Ordinary adaptive-filter baseline and calibration

The preferred baseline has exactly two retained scalar states, an EMA reference `m` and LIF membrane `u`, both initially zero. At each delivered input `x_t`:

`m_t = (1-alpha)*m_(t-1) + alpha*x_t`,

`r_t = abs(x_t-m_t)`,

`J_t = g*r_t`,

`u_candidate = .8*u_(t-1) + J_t`.

It spikes iff `u_candidate>1`, then resets membrane to0, otherwise retains the candidate. Update the EMA before computing the residual, matching TAN's current-inclusive reference convention. Full-wave rectification preserves the original Task B gate's magnitude symmetry; a positive-part rectifier would introduce a further positive-edge-only prior. There is no threshold adaptation, learned decoder, history cache, or access to the class/noise path at inference.

Set `alpha=2/(10+1)=2/11` before any data. An EMA with this parameter has mean sample lag `(1-alpha)/alpha=4.5`, matching the W10 uniform window's mean lag. For stationary unrectified iid input, its reference variance factor `alpha/(2-alpha)=1/10` also matches that of a ten-point mean. These facts justify one ordinary reference timescale; they do not make the complete filters equivalent. Current-sample weights, innovation variance, startup behavior, clipping, and frequency responses still differ.

Use512 independent iid training noise histories, each with the same step/catch pairing as v9. Run native TAN once to obtain its effective currents `A_t`, and compute baseline unscaled `r_t`. Define the one fitted, globally shared gain by

`g = sum_(all training histories, both classes, all60steps) abs(A_t)
     / sum_(same training histories/classes/steps) r_t`.

Both currents are nonnegative, so this matches mean integrated absolute effective current on the balanced iid training ensemble. It uses no detection/latency/test objective. Freeze full-precision `g`, training-input hash, numerator/denominator and calibration artifact before generating or evaluating held-out data. A zero denominator is a failure, not a cue to change the model. Do not fit a second gain, class-dependent gain, threshold, EMA time constant, or AR-specific calibration.

TAN and the baseline then have the same membrane leak/threshold and matched training current integral in model units. This is a meaningful narrow charging-current normalization. It does **not** match squared current, held-out current, output spike counts, retained memory, computation, or physical energy. Report each of those distinctions. Matching only the training expectation deliberately leaves budget drift under the correlation shift observable.

## One fixed held-out distribution change

Recommend1024 fresh independent innovation histories `Z[i,t]~Normal(0,1)`, using a separate frozen seed. For each row construct:

`N_iid,t = .4*Z_t`;

`N_AR,0 = .4*Z_0`,

`N_AR,t = .9*N_AR,t-1 + .4*sqrt(1-.9^2)*Z_t` for t>=1.

The AR initialization is stationary; starting its noise at0 would reduce early variance and confound the comparison. Both regimes have exactly the same Gaussian marginal SD.4. The `.9` coefficient corresponds to an approximately ten-sample correlation timescale, chosen once from the existing W10 horizon, not outcomes. Both models receive identical delivered waveforms within each regime/class. Couple iid/AR through the same `Z` for paired comparisons, and pair the step/catch waveforms before clipping.

The step amplitude, duration, per-time input marginal and expected total input squared amplitude remain unchanged between regimes. Realized per-history amplitude/energy integrals and temporal spectra differ; do not normalize individual test paths to erase or distort the intended dependence change.

Use exactly two simulated models, TAN and the calibrated ordinary filter. A mathematical zero-output control can retain the v9 chance reference without additional neuron simulation. Clean step/catch controls are deterministic sanity checks, not extra stochastic conditions or calibration targets. No AR coefficient, input amplitude, detector threshold, observation window or test sample-size sweep is allowed.

## Endpoints, inference and non-superiority

Keep the fixed detection rule: at least one spike in indices15–19. Report TPR, catch FPR, specificity and balanced accuracy for each model×noise regime. The primary dependence-shift contrast should be fixed as

`[BA_TAN,AR - BA_baseline,AR] - [BA_TAN,iid - BA_baseline,iid]`.

Also report its two absolute model contrasts. A positive interaction does not by itself establish that TAN is better on AR: it could reflect a worse iid starting point. An AR-superiority statement requires its prespecified absolute AR contrast to support that statement as well. If either challenge is at ceiling, retain the resulting null/limited resolution and stop.

Latency is a secondary all-trial endpoint using the same existing deadline. Let `T` be the first onset-window spike offset0–4; set `L=5` for a miss. Report mean capped delay and the full cumulative detection profile by offsets0–4, together with deadline TPR. Do not use responder-only latency averages or discard misses. The known time15 remains externally supplied; this measures response delay after a known signal, not inference of an unknown onset time.

Retain original late rate on20–59, total/onset/late generated spikes, and the native noisy-minus-own-clean rate as separate activity endpoints. Save effective current, candidate/post-reset membrane, EMA reference or full TAN history/attention/gate, and per-trial absolute-current, squared-current and squared-candidate sums and peaks. Report actual iid/AR current and spike-budget mismatches. No ATP, energy-efficiency, or equal-state-resource claim follows: TAN stores ten input samples plus membrane; the baseline stores reference plus membrane and one fixed calibrated gain.

Use2000 whole-held-out-history paired bootstrap resamples preserving both regimes, both classes and both models. Freeze exact contrasts before evaluation and save full per-history outcomes. Intervals are conditional on the trained-and-frozen gain; they do not measure uncertainty over repeated calibration experiments. Calibration/test seeds and complete histories must remain separate. No significance-driven stopping or calibration revision is allowed.

Possible outcomes are deliberately symmetric: a TAN advantage is evidence for this fixed model's correlation-transfer behavior under the declared normalization; baseline parity/superiority shows ordinary adaptive filtering suffices for this test; a gain-budget mismatch or ceiling can limit the conclusion. None uniquely attributes behavior to attention or proves all adaptive-threshold models inferior. Independent source-equation, trajectory, calibration and paired-statistic verification should precede interpretation.

The scientific scope is one frozen comparison and its separate computational validation. No additional ablation, noise family, decoder, calibration search, or follow-up experiment is represented as completed here.
