# Final bounded round: one nuisance-correlation contrast against ordinary adaptation

Frozen 2026-11-02 before any v11 calibration/test inputs or model execution. The registered scope is the single comparison below and its separate computational validation.

## Scientific question and interpretation

Does the frozen original Task B TAN show an advantageous response to a single change in nuisance temporal correlation beyond a gain-calibrated ordinary adaptive filter? Ordinary adaptation already separates transients from slower backgrounds; this is not a novel generic property. Primary literature and repository equations are reviewed in SOURCE_AND_BASELINE_REVIEW.md, PRIMARY_ADAPTATION_NOTES.md and MECHANISM_PROPOSAL.md.

Select exactly two arms and one environmental contrast: original TAN and an ordinary leaky-subtraction/full-wave-rectifier/LIF baseline; iid noise versus stationary AR(1) noise with rho=.9. No alternative baseline, rectifier, rho, rise time, learned decoder or threshold will be tried. Changed noise correlation is a NEW stimulus distribution, not original-task replication. This is a model comparison, not isolated causal identification of TAN's gate: attention/context, nonlinear gain and memory structure also differ.

Prediction is deliberately non-superiority-compatible. For stationary unclipped linear signals, both contrast filters below have white-noise variance .9 sigma². At rho=.9 TAN's boxcar deviation variance is .42497807238 sigma², while the ordinary filter has .32142857143 sigma². Thus correlation attenuation is not uniquely or automatically better for TAN. Clipping, finite initialization, rectification, context and spikes can change behavioral results. The empirical question remains open.

## Models, fixed before calibration

Original TAN is the exact verified Task B source from snapshot/code/experiments/statistical_validation.py, commit85213f0d417d54e4f48b301ed1c0e4200ef3c421: W10, membrane leak.8, threshold1, signed surprise, tanh(abs(surprise)) gate, weights(2,1,1), beta1, current-inclusive window, softmax denominator epsilon1e-9, strict > spike/reset0, zero initial window/membrane. Reuse hash-verified v9 source extraction and native instrumentation. All original labels and P1A evidence remain unchanged.

Ordinary adaptive baseline (ADAPT): retained scalars m,u, initially0. For every delivered nonnegative x:

m_new=.8*m+.2*x;
d=(9/8)*(x-m_new);
J=kappa*abs(d);
u_candidate=.8*u+J;
y=1[u_candidate>1]; u_new=0 if y else u_candidate.

Use updated m for d and always retain m_new, including after a spike. Full-wave rectification matches the original Task B gate's magnitude treatment; it is not the positive-part alternative in a prospective note. No attention, input array, adaptive spike threshold or additional state. This is an ordinary input-adaptation/filter comparator, not the repository's spike-triggered AdaptiveLIF and not a validated biophysical model.

Why this pole/scale: for a clean step Delta, d starts at .9Delta and has infinite transient sum4.5Delta, matching W10 boxcar surprise; its stationary white-noise variance also matches .9sigma². The normalization9/8 is fixed analytically. This targets the actual deviation filter rather than matching only the mean estimator's lag (the alternative2/11 EMA proposal is not executed). These matches concern the signed linear contrast before the full-wave current and scalar calibration. The temporal kernels are not identical. Verify the algebra/finite clean trajectories, but do not choose a different pole afterward.

## One closed-form calibration; no search

Exactly512 fresh iid Gaussian paths from RandomState2026110250.normal(0,.4,size=(512,60)). Each path creates the original step and no-step catch, both lower-clipped at0. Save/hash training inputs first. Original base is0 at0..14 and1.5 at15..59; catch base0. Both classes have equal weight and start each model from reset.

Run native TAN to obtain its input-driven effective current A_t, and compute the baseline's unscaled abs(d_t). Freeze

kappa = sum_{histories,classes,t} abs(A_t) / sum_{histories,classes,t} abs(d_t).

Require finite positive numerator/denominator/gain; otherwise stop and report the failed calibration. This single ratio matches TOTAL ABSOLUTE EFFECTIVE CURRENT on the balanced iid calibration ensemble, including all60 steps. It is not fit to spike accuracy, false alarms, latency or any test result, and not a physical-energy/rate/conditional-class matching guarantee. Save training trajectories, currents, ratio and exact budget residual in CALIBRATION_FROZEN.json BEFORE generating/evaluating test inputs. No gain clipping, alternate fit or calibration retry. Test inference is conditional on this one frozen calibration; do not refit kappa in test bootstraps.

## One fixed held-out test

Exactly1024 independent innovation rows Z from RandomState2026110251.standard_normal((1024,60)). For each row:

N_iid,t=.4*Z_t;
N_AR,0=.4*Z_0;
N_AR,t=.9*N_AR,t-1+.4*sqrt(1-.9²)*Z_t for t>=1.

The AR process starts in its stationary Gaussian marginal. Both regimes have marginal variance .16 at every time; using shared innovations pairs them without making their delivered paths equal. Noise is added before lower clipping. Each row supplies both regimes, both step/catch conditions and both models. Save/hash Z, noise paths, deterministic bases and delivered waveforms before test evaluation. No per-path energy normalization, selection or exclusions.

Signal amplitude1.5, onset15, duration45, nominal signal sum67.5 and squared sum101.25 are identical across regimes. Expected raw noise squared sum is9.6 in both; finite realized noise and clipped-input energies differ and must be reported. Equal Gaussian marginals imply equal expected delivered per-time input moments within each signal class, not exact samplewise energy equality. The models still start at0; their finite adaptation transients are not replaced by stationary model states. Four deterministic clean model/class traces are separate checks, not statistical replicates.

## Observation and fixed metrics

Keep v9's shared observation budget: D=1[any spike at indices15..19]. Report per regime/model TPR, catch-window FPR, specificity and balanced accuracy BA=.5*(TPR+1-FPR). The known onset clock is an external readout assumption. This is not unknown-onset localization.

Report first response delay t-15 within15..19 on step trials. Use unconditional capped latency L=min(first delay,5), with miss assigned5. Save the complete delay/miss distribution; also report conditional-on-hit latency, always with coverage. No posthoc window or latency threshold.

Also report probability of any pre-onset spike0..14 (identical across paired step/catch prefixes, counted once per independent history/model/regime), original late R=mean(spikes[20:60]), total/onset/late counts and noisy-minus-own-clean late R. Pre-onset alarms and catch-window false alarms are distinct. Preserve all unfavorable effects and signal/false-alarm tradeoffs.

Define per-history BA contribution as .5*(D_step+1-D_catch). Let Delta_r=BA_TAN,r-BA_ADAPT,r. Exactly two primary contrasts:

1. Delta_AR: absolute balanced-accuracy difference in correlated noise;
2. Interaction=Delta_AR-Delta_iid: change in relative performance under correlation.

Support for the specific claimed correlation advantage requires BOTH lower endpoints of their predeclared97.5% two-sided paired-bootstrap intervals to exceed0. This is a conservative two-comparison reporting rule, not an exact finite-sample coverage theorem. If either fails, superiority for this hypothesis is not established; if an upper endpoint is<=0, the corresponding direction is disfavored. Crossing0 is inconclusive, not equivalence. No data-driven margin, alternative primary score or significance stopping. TPR/FPR, latency and resource tradeoffs limit any interpretation even when the two criteria pass. No gate-only causal, matched-energy or universal adaptive-baseline superiority claim follows.

## Paired uncertainty and resources

Exactly2000 resamples of whole1024-row innovation groups with RandomState2026110252. Every draw preserves both regimes/classes/models. Save indices, all primary replicate values and all fixed metric replicates. Primary intervals use quantiles.0125/.9875; remaining metric/contrast intervals are pointwise descriptive95% (.025/.975), with no simultaneous guarantee. The training gain stays fixed. Empirical zero errors and degenerate percentile intervals are not population perfection. Do not resample timepoints or treat four counterfactual waveforms as independent histories.

Report each model/regime/class's effective-current sum abs(I), sum I², sum pre-reset candidate², current/candidate/state peaks and generated spikes; also raw-input/noise squared sums and their realized mismatch. Report training budget equality and fresh budget drift, not assumed equality. TAN retains10 history scalars plus membrane and attention/gating arithmetic; ADAPT retains2 scalars plus subtraction, magnitude, gain and common spike generator. Equal calibrated currentL1 is a dimensionless proxy, not total neural energy or equal computational resource. No ATP/joules or resource-normalized superiority inferred.

## Independent verification, preservation and stop

Save full calibration/test/clean trajectories, native histories/surprise/gate/context/attention/current/candidate/state/spikes, ADAPT m,d,J,candidate/state/spikes, all ledgers, inputs, gain, resources, bootstrap replicates and hashes. Verify source extraction and exact native outputs at each update. Independent code must regenerate the baseline recurrence, ARstationary initialization, gain and budget matching, all trajectories/metrics/paired statistics without importing primary simulation/metric helpers. Exact spike/label identity; absolute float tolerance1e-12 with raw residuals retained. Repo source/earlier artifacts unchanged; source hashes mandatory, absent.git provenance unavailable in archive replay.

The registered stopping point is the fixed experiment and separate computational validation. No added arm, parameter sweep, decoder improvement, or additional experiment is part of this protocol.
