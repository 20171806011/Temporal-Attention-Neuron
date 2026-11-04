# Prospective v10 design review: finite-alphabet noise transfer

2026-11-02. Design review only; no v10 native trajectories, random inputs, information estimates, or new outcomes were generated or inspected. This review is advice for a separate preregistration. The completed v9 result motivates the question and is not an untouched discovery dataset.

## Decision: one narrow count-versus-decision audit is useful

Keep the exact original Task B native models and v9 step/catch contract. Let `S` be the balanced binary step label, `N` the original 60-sample pre-clipping Gaussian path, `Y=sum(spikes[15:20])` on the fixed alphabet `{0,1,2,3,4,5}`, and `D=1[Y>=1]`. All initial states, parameters, clipping, time windows, and thresholds remain frozen. `S` is independent of `N`; the two conditions for a history share `N`.

The useful new question is **how much extra step information and noise-dependent variability reside in the count detail that the already-fixed binary decision discards**. Merely computing `I(S;D)` would re-express v9's TPR/FPR, with no new independent behavioral capability. The more informative pair is:

`extra_count_signal = I(S;Y) - I(S;D) = I(S;Y|D)`;

for native deterministic models,

`discarded_count_noise = I(N;Y|S) - I(N;D|S) = H(Y|S,D)`.

These quantities distinguish useful count detail from conditional count variability at this operating point. They do not establish attention's causal role, amplitude reconstruction, general denoising, or superior biological efficiency. The identities themselves are standard information theory, not a novel scientific discovery. Any contribution is a reproducible diagnosis of these frozen models and task.

The onset time never varies, so neither timing information nor unknown-onset detection can be measured here. `Y` itself discards within-window spike timing. Entropy units are bits per modeled trial, not bits per second; no physical timestep or measured energy is supplied. The external known-onset clock and five-step counter remain explicit readout resources.

## Exact estimands and the stochastic-control correction

For either native model, `Y=f(S,N)` and `H(Y|S,N)=0`. Hence:

`H(Y) = I(S;Y) + I(N;Y|S)`, with `I(N;Y|S)=H(Y|S)`.

The full `N` includes noise before onset retained in the carrier. Future noise after index 19 cannot affect this causal onset count, but including it in `N` does not change the identity. No binning of the continuous noise path, nearest-neighbor MI estimator, or differential entropy estimate is needed: only the finite output conditional distributions are estimated.

For `LIF_THINNED`, write `K` for the native LIF onset count and keep `p=529/1013`. Given `(S,N)`, the retained count obeys the exact kernel

`P(Y=j | K=k) = binom(k,j) p^j (1-p)^(k-j)`.

Integrate this six-bin kernel analytically rather than sampling another deletion sequence. That preserves the specified thinning channel while removing avoidable Monte Carlo variability from the additional random source `U`. The unconditional model law changes neither its native LIF trajectories nor its output budget; the estimator now marginalizes `U` rather than using one realized deletion path per trial as v9 did.

The correct decomposition is then

`H(Y) = I(S;Y) + I(N;Y|S) + H(Y|S,N)`,

`H(Y|S,N) = E[h(Binomial(K,p))]`,

`I(N;Y|S) = H(Y|S) - E[h(Binomial(K,p))]`.

The last entropy equals `I(U;Y|S,N)` because the full output is deterministic after conditioning on the deletion randomness. Calling all of `H(Y|S)` environmental-noise information would incorrectly charge random deletion to the environment. For the binary decision use `q_k=1-(1-p)^k`, replacing binomial entropy by `h2(q_k)` in the corresponding residual.

These are ordered conditional information quantities. They are not a unique additive causal allocation of physical signal and noise: conditioning order matters when the input sources interact. In particular, `I(N;Y|S)` must not be silently relabeled unconditional `I(N;Y)` or a measure of noise amplitude/energy.

## Recommended fixed execution

Use exactly **2048 fresh independent Gaussian paths**, one continuous recorded legacy NumPy stream, with both step/catch conditions per path. Root should freeze unused input/bootstrap seeds and hash the input arrays before evaluating models. Do not reuse the v9 sample for parameter selection or call this an independent confirmation of a prespecified v9 information hypothesis.

Run only native TAN and LIF; derive the thinned count/decision laws analytically and the zero-output law exactly. Retain the original late spike rate as unchanged context, not another optimized objective. Keep all trajectories or exact reproducible native counts/current resource records under the same contract and preserve all observations.

For each arm and each output `Y,D`, report:

- Conditional distributions for `S=0,1`, their balanced marginal, and all fixed-alphabet empty/singleton native-count cells.
- `H(output)`, `I(S;output)`, `I(N;output|S)`, and the residual `H(output|S,N)` separately.
- Count-to-decision differences in retained signal information and environmental-noise information. For thinning, retain its deletion-entropy reduction too; do not use the native shortcut.
- Original event-budget/resource accounting and fixed-detector TPR/FPR as context. No new fitted decoder, metric-dependent threshold, bin width, onset, or architecture change.

Compute all terms from the same estimated probability laws, with log base two and `0 log 0=0`. Preserve the original six-value alphabet even if some cells are empty. Avoid ad hoc pseudocounts or termwise bias corrections that break exact chain-rule identities or data-processing inequalities.

## Uncertainty and finite-sample limits

Use exactly 2000 bootstrap resamples of complete independent `N` rows. Each resample preserves both conditions and all arms; recompute native PMFs and their analytical thinning mixtures. The two condition rows are not independent replications. Report prespecified contrasts and intervals even if unfavorable; do not increase sample size or change estimators after outcomes.

The plug-in entropy and derived MI are population estimates, although their algebraic identities hold exactly for the empirical law. Empty observed cells do not certify impossible population events. Ordinary bootstrap cannot generate unseen categories and may give degenerate intervals at a perfect empirical detector. Label these intervals descriptive, report occupied/empty cells, and do not treat small between-arm differences as established when estimator bias or boundary behavior could explain them. A bootstrap bias diagnostic can be recorded if frozen, but should not silently replace the primary estimator. These estimation limitations are established in [Paninski's primary analysis of entropy and MI estimation](https://direct.mit.edu/neco/article/15/6/1191/6731/Estimation-of-Entropy-and-Mutual-Information), also described on the [author's research page](https://sites.stat.columbia.edu/liam/research/abstracts/info_est-nc-abs.html).

Add one small rigorous safeguard for the native decision information. Freeze the simultaneous family as **TAN/LIF × miss/FPR**, four Bernoulli error probabilities. For each, obtain a one-sided exact binomial upper bound at `alpha_cell=.05/4`. There are 2048 independent noise paths for each cell despite the cross-condition pairing; the Bonferroni guarantee requires no independence among the four cells. For `k=0` errors, the upper bound is `1-alpha_cell^(1/n)`; for `0<k<n`, invert the binomial lower tail, equivalently the beta quantile with parameters `(k+1,n-k)` at `1-alpha_cell`; for `k=n`, use one.

Let `e_U=(miss_U+FPR_U)/2`. With simultaneous confidence at least 95%,

`I(S;Y) >= I(S;D) >= 1-h2(e_U)` if `e_U<=.5`; otherwise report lower bound zero.

The second inequality follows directly by defining `E=S xor D`: `H(S|D)=H(E|D)<=H(E)=h2(P(E))`. This guards against interpreting a zero-error sample as population perfection. It certifies only retained binary-label information, not the nuisance entropy or between-arm ranking. **Do not apply exact binomial bounds to the analytically integrated fractional thinning error probabilities.** A more elaborate multinomial entropy confidence construction is unnecessary for this bounded follow-up.

## Independent validation and stop

Independently regenerate exact native trajectories and both conditional count tables; reconstruct all integrated kernels and statistics without importing primary metric helpers. Mathematical checks should include:

- Each PMF sums to one; all entropies/nonconditional MI are within their fixed-alphabet limits; all declared conditional information terms are nonnegative up to a frozen numerical tolerance.
- `0 <= I(S;D) <= I(S;Y) <= 1` and each complete entropy decomposition closes to numerical tolerance.
- Native residual entropy is exactly zero. Thinned residual entropy matches the expected binomial/binary kernel entropy.
- Thinning obeys `I(S;Y_thin)<=I(S;K_LIF)`; environmental conditional information likewise cannot exceed the native count's under the Markov kernel.
- The always-silent control has all information/entropy terms zero. An algebraic one-bit oracle `Y=S` has one signal bit and zero nuisance entropy; this is a formula check, not a competing neuron with label access.
- Kernel endpoint checks at `k=0` and optional algebraic `p=0,1` cases validate formulas; they are not new model arms or parameter sweeps.

Stop after the fixed PMFs, entropy partitions, count-versus-decision contrasts, native finite-sample lower bounds, and independent verification. If count detail adds little signal and substantial conditional variability, conclude that the already-fixed coarse decision discards that detail while preserving the measured label information at this operating point. If it adds useful label information, retain that tradeoff. If estimation limits leave a contrast uncertain, report uncertainty and close the test rather than launching a decoder or estimator search. Do not revise historical Task B labels or the independent P1A findings.

The independent statistical-design reviewer agreed with the paired bootstrap, exact thinning integration, and four-cell binomial safeguard.  experiment execution awaits study's frozen preregistration.

## Subsequent root decision before execution

Root selected exactly 2048 fresh histories and a stronger joint native certificate: allocate total error probability `.025` to the four conditional native count distributions using a finite-alphabet total-variation bound, and `.025` to the four native miss/FPR exact-binomial bounds. Thus the earlier standalone `.05/4` binomial allocation above is a design option, **superseded by `.025/4` under the selected combined certificate**. The frozen preregistration will determine the precise entropy-continuity formulas and report bounds on both extra count signal and discarded conditional noise. The same data suffice; no new model arm, latency task, threshold, or tuning search is added. The onset time remains constant with zero entropy, and the external readout uses that known time. No v10 execution has occurred.
