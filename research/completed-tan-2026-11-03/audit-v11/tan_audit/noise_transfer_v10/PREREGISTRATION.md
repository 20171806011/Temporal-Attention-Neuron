# One prospective finite-output transfer audit: count detail versus fixed decision

2026-11-02, frozen before v10 input generation, neuron execution or information estimates. Follows completed v9, which is motivation rather than an untouched discovery dataset. Exactly one fresh batch; no model/threshold/decoder/estimator search, installation, spending, remote edits or publication. Preserve all original labels, P1A positive evidence and prior outputs.

## Question and prediction

How much additional step information is in the onset spike count C beyond the already-fixed decision D=1[C>=1], versus how much conditional dependence on the Gaussian input path that count detail contains? Binary-decision MI alone merely re-expresses v9 detection rates and is not the new finding. Predict that native count detail retains appreciable conditional path dependence while adding little step-label information, for both native models; no architecture winner is prespecified. The target is one rigorously bounded diagnostic result, including an informative negative or inconclusive bound. This is standard information accounting applied to the frozen task, not new information theory or proof of an internal neuronal computation.

## Variables and available information

S has imposed balanced prior P(S=0)=P(S=1)=1/2. S=1 is the original Task B step, S=0 the v9 no-step catch. Original Task B itself always presented the step; class information is defined on the added balanced v9 ensemble. N is the entire independent 60-coordinate pre-clipping Gaussian path, each coordinate SD.4, independent of S. Delivered input X_t=max(0,1.5*S*1[t>=15]+N_t), t=0..59. Pairing both S values for a noise history is an experimental/inference device; no decoder sees the counterfactual pair, true S, noise path, model states or thinning mask.

The single-trial observation C=sum(spikes[15:20]) has the fixed alphabet{0,1,2,3,4,5}; D=1[C>=1] has alphabet{0,1}. The known onset clock and external five-step counter are readout resources. No fit, learned features or adaptive timing. Onset time is the constant15 and has entropy0, so unknown-change timing information/localization is not measured. C omits within-window spike timing and is not the complete spike-pattern information. Original late rate R=sum(spikes[20:60])/40, full-trial counts, and current/state proxies remain separate activity/resource endpoints; no41-bin late-count MI is added.

The frozen native source is snapshot/code/experiments/statistical_validation.py at commit85213f0d417d54e4f48b301ed1c0e4200ef3c421. Reuse the exact verified source-node extraction and manifest in ../noise_preservation_v9. LIF leak.8/threshold2; TAN W10/leak.8/threshold1, signed surprise, tanh(abs(Surprise)) gate, current-inclusive history, beta1, weights(2,1,1), softmax epsilon1e−9, strict > threshold and reset0, zero initial states. No rectified-branch substitution. The original100-record replication is completed in v9 and is not repeated as a new discovery; verify its hashes and source provenance before the new study.

## Fixed sample and controls

Exactly2048 iid noise rows from continuous NumPy RandomState2026110240.normal(0,.4,size=(2048,60)). Each row supplies both classes and both native models from reset. Save/hash all Gaussian noise and delivered waveforms BEFORE model evaluation. No exclusions, seed changes or sample-size extensions.

Four observed-output arms: native TAN, native LIF, fixed LIF_THINNED, ZERO_OUTPUT. Thinning uses the unchanged rational p=529/1013 from v9 historical calibration. Conditional on native LIF count k, its observed count law is B[k,j]=choose(k,j)*p^j*(1-p)^(k-j), j<=k, zero otherwise. Integrate this law exactly; do not draw new U. For D use success probability q(k)=1-(1-p)^k. This marginalizes the same v9 label-blind independent-deletion channel, reducing evaluator Monte Carlo variance; it does not improve the physical control or grant its native count to a decoder. The mask remains unobserved. Pairwise shared masks in v9 do not change these single-trial marginal laws. Analytic results are expectations, not v9's realized thinned traces. ZERO_OUTPUT has exact all-zero law and no assigned physical energy.

Run four native clean step/catch trajectories once, compare their exact source outputs and v9 clean traces. Compute thinned clean laws analytically. Clean outputs are deterministic checks, not2048 independent samples. For every history/class/arm retain native count, observed-output count probability vector, D positive probability, expected or actual late/full-trial counts, generated spike counts and resource proxies. Save every full native trajectory, including surprise/history/attention/current/candidate/post-reset state/spikes, or exact numeric arrays sufficient to reconstruct these without fitting.

## Information calculations and scope

Use base2 logs, 0log0=0, fixed alphabets, no smoothing, pseudocounts, adaptive bins, KSG, differential entropy or termwise bias correction. For each arm and O in{C,D}, estimate the two conditional pmfs by averaging native one-hot laws or integrated kernels, and their balanced marginal. Report all bins including empty/singleton native-count cells.

Compute H(O), H(O|S), I_S=I(S;O)=H(O)-H(O|S), residual U_O=H(O|S,N), and I_N=I(N;O|S)=H(O|S)-U_O. For native arms U_O=0; for thinning U_C=mean_s,n H(B[k_s,n,:]) and U_D=mean_s,n h2(q(k_s,n)). These identities are exact for the empirical law and population identities under the declared model; the estimated pmfs are not exact population probabilities.

The allocation is ordered conditional information, not a unique causal division of signal-origin/noise-origin spikes. Full-path conditioning includes retained past noise and signal–noise interactions; no instantaneous-noise cascade or percentage of continuous noise removed is claimed. The extra U term is external output deletion randomness conditional on inputs, not intrinsic neuronal noise. Information is bits/trial, not energy, noise power, information created by deterministic processing, or channel capacity optimized over priors.

Primary within-arm count-to-decision differences: Delta_signal=I_S(C)-I_S(D), Delta_path=I_N(C)-I_N(D), Delta_deletion=U_C-U_D. For deterministic natives, Delta_path=H(C|S)-H(D|S)=H(C|S,D). Report these for all four arms. Data processing implies nonnegative differences; coarsening is external and must not be attributed to the native spike threshold's causal role. Do not add between-architecture MI optimization or signal/noise ratios.

## Uncertainty and a simultaneous population certificate

Exactly2000 whole-history bootstrap draws with RandomState2026110241, each sampling2048 row indices with replacement. Preserve both classes, models and all count laws; recompute all pmfs/kernels inside each replicate. Save indices and all metric replicates. Give pointwise descriptive95% percentile intervals for every reported information term and three count-to-decision differences for each arm. Save bootstrap mean minus original estimate as a bias diagnostic only; do not substitute a corrected estimator or interpret bootstrap as discovering unseen bins. No significance stopping or simultaneous-coverage claim for these intervals.

Separately construct the following conservative, simultaneous >=95% population certificate for BOTH NATIVE models. Allocate alpha_TV=.025 to the four native model×class count distributions (alphabet size6). For n=2048 set

delta=sqrt(log(4*(2^6-2)/.025)/(2*n)),

F6(delta)=h2(delta)+delta*log2(5).

TV=max_A(pHat(A)-p(A)); a union over62 nonempty proper subsets and four distributions with one-sided Hoeffding gives joint TV<=delta. Complement subsets already cover both signs, no extra factor2. Entropy continuity gives |H(pHat)-H(p)|<=F6(delta), since delta<=5/6. Paired classes/models need not be independent for the union bound.

Allocate alpha_CP=.025 to the four native model×{miss,FPR} Bernoulli cells, alpha_cell=.00625. For k errors/n use one-sided exact upper qU: k=0 gives -expm1(log(alpha_cell)/n); k=n gives1; otherwise beta.ppf(1-alpha_cell,k+1,n-k). Verify nonboundary quantiles by binomial-CDF inversion. Let eU=(missU+FPRU)/2 and b=h2(eU) if eU<=.5, otherwise1. With E=S xor D, both H(S|D)<=h2(Perror) and H(D|S)<=h2(Perror). Thus on the combined event, simultaneously for both native arms:

I(S;D)>=1-b;
0<=Delta_signal<=b;
Delta_path>=max(0,Hhat(C|S)-F6(delta)-b).

The confidence allocation is .025+.025=.05 across all statements, even with correlations. This protects against finite-sample entropy bias/unseen outcomes and zero empirical errors. It is a certificate for these observables/prior, not perfect detection or harmful-noise removal. Record exact bounds even if the lower bound is0; do not widen the sample or change observation. Do NOT apply CP to analytically integrated fractional thinning probabilities. Bootstrap estimates for that control remain descriptive.

## Activity, resources and validation

Keep v9 late-rate scoring unchanged. Report native noisy minus own clean late rate; thinned expected rates are p times native LIF rates. Different thresholds/current transforms and TAN ten stored history samples plus membrane versus LIF membrane preclude matched gain/physical energy claims. Report native per-history sum abs(current), sum current², sum candidate² and peaks, generated versus observed counts; thinned control retains LIF internal computation. Historical expected late output budget is unchanged, actual fresh budget mismatch is disclosed. No ATP/joule or bits-per-energy claim.

Before interpreting results, check kernel normalization, clean native laws, exact silent null, algebraic label-oracle law C=S (formula-only check, not a neuron), algebraic p=0 and p=1 kernel endpoints (not new empirical arms/sweep), exact finite binary MI formula from TPR/FPR, entropy chain-rule closure, native residual0, and data-processing inequalities for coarsening and thinning. Preserve raw tiny residuals; absolute tolerance1e−12 for float comparisons/roundoff, exact spike decisions. All negative values beyond tolerance are failures to diagnose, not silently clipped. Compare independent implementation on all native trajectories, input/draw arrays, kernel pmfs, ledger rows, bootstrap terms and confidence certificates, without importing primary simulation/statistic helpers. Source hashes mandatory, clean live Git checkout unchanged; archive without.git must explicitly record provenance unavailable.

Stop after this one fresh batch, complete fixed tables/certificates, independent review, source-cited interpretation and same scientific report update. If both natives benefit from the same external coarsening, report that shared finding. If no nontrivial certificate emerges, report the limitation. No original W5 sinusoid estimator rerun, changing-onset benchmark, learned decoder, new neuronal mechanism, publication or paid service in this step.
