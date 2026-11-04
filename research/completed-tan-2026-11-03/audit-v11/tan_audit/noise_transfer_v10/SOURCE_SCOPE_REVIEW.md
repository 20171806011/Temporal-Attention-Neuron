# Prospective Task B information-transfer scope review

2026-11-02. Source review and mathematical derivation only: no new trajectories, entropy/MI estimates, estimator fixtures, or result tables were computed. Repository `85213f0d417d54e4f48b301ed1c0e4200ef3c421`; prior outputs remain unchanged. The study will freeze the exact v10 design after these reviews.

## Distinguish the two experimental sources

The original manuscript's MI experiment is **not Task B**. [information_analysis.py:571](../../Temporal-Attention-Neuron/snapshot/code/experiments/information_analysis.py#L571) builds 3,000 samples of a deterministic three-frequency sinusoid, seeded Gaussian noise SD .4, then lower-clips the sum. Its TAN uses W=5, leak=.5, threshold=.5 ([586](../../Temporal-Attention-Neuron/snapshot/code/experiments/information_analysis.py#L586)) and always rectifies surprise ([109](../../Temporal-Attention-Neuron/snapshot/code/experiments/information_analysis.py#L109)). LIF uses leak=.5, threshold2 ([602](../../Temporal-Attention-Neuron/snapshot/code/experiments/information_analysis.py#L602)). The reported quantities compare contemporaneous clean amplitude/change-rate/noise against surprise, current or post-reset membrane, and spikes ([628](../../Temporal-Attention-Neuron/snapshot/code/experiments/information_analysis.py#L628)). They are time-sample associations on that construction.

Task B instead has a 60-step fixed upward step; TAN W10, leak.8, threshold1, signed-surprise attention and magnitude gating, and LIF leak.8/threshold2 ([statistical_validation.py:174](../../Temporal-Attention-Neuron/snapshot/code/experiments/statistical_validation.py#L174)). Its original scalar endpoint is late spike rate. The v9 extension added paired step/catch histories and the known-onset external readout ([v9 protocol:19](../noise_preservation_v9/PREREGISTRATION.md#L19), [27](../noise_preservation_v9/PREREGISTRATION.md#L27)); its exact observed outputs and count fields are implemented at [run_noise_preservation.py:218](../noise_preservation_v9/run_noise_preservation.py#L218) and [237](../noise_preservation_v9/run_noise_preservation.py#L237).

A finite-output Task B diagnostic would be a **new, separately scoped analysis of the preserved Task B/v9 contract**. It would not reproduce, repair, validate, or refute the original sinusoidal MI table. The v9 agenda had proposed a separate original-estimator audit; the study's current direction selects the narrower Task B continuation instead.

## What the original information claims already say

[analysis.tex:69](../../Temporal-Attention-Neuron/snapshot/manuscript/sections/analysis.tex#L69) titles thresholding as the noise filter; [105](../../Temporal-Attention-Neuron/snapshot/manuscript/sections/analysis.tex#L105) says it filters most noise information, while [108](../../Temporal-Attention-Neuron/snapshot/manuscript/sections/analysis.tex#L108) explicitly disclaims exclusive causal attribution. [110](../../Temporal-Attention-Neuron/snapshot/manuscript/sections/analysis.tex#L110) separately reports clean change-rate information. Thus comparing useful-signal information and nuisance information is already part of the project; it is not a new conceptual proposal.

Existing v8 source review identified mixed KSG/log2 units, a discrete correction/bin convention problem, and an incomplete same-time causal chain. The primary source indeed returns an unconverted digamma expression ([information_analysis.py:490](../../Temporal-Attention-Neuron/snapshot/code/experiments/information_analysis.py#L490)), separately uses log2 plus a stated correction ([532](../../Temporal-Attention-Neuron/snapshot/code/experiments/information_analysis.py#L532)), labels all results `mi_bits` ([659](../../Temporal-Attention-Neuron/snapshot/code/experiments/information_analysis.py#L659)), and subtracts estimates for a filtering headline ([829](../../Temporal-Attention-Neuron/snapshot/code/experiments/information_analysis.py#L829)). These are existing source findings, not executed v10 corrections. The convenience module [mutual_information.py:65](../../Temporal-Attention-Neuron/snapshot/code/analysis/mutual_information.py#L65) repeats those formulas; it should not be silently reused as a validated finite-category estimator.

The same-time quantity `A_t` alone does not determine the spike because the update also uses the previous membrane ([information_analysis.py:116](../../Temporal-Attention-Neuron/snapshot/code/experiments/information_analysis.py#L116)). The LIF `h_t` used for its MI comparison is post-reset, not the pre-threshold candidate. A Task B count-to-detector data-processing identity cannot cure or establish the original internal-stage causal claim.

## A precise prospective finite-output contract

Use `S` only for the **binary step-presence label** in these formulas; it is not the source code's surprise `S_t`. State the imposed prior `P(S=0)=P(S=1)=1/2`. The original step-only experiment has constant S and cannot by itself measure class information. This balanced ensemble is supplied by v9's paired catch control.

Let N be the entire pre-clipping Gaussian noise path, independent of S; fixed initial states and fixed native dynamics are part of the contract. For the onset output, only its prefix through index19 can matter. Let C be the count of observed spikes at indices15–19, supported on the fixed alphabet `{0,1,2,3,4,5}`; D=`1[C≥1]` is the already declared detector. Analyze the output of **one actual trial**, never the paired counterfactual `(C_step,C_catch)` as though the receiver observed both. Preserve pairs for statistical resampling only.

Onset time τ=15 is a constant, so `H(τ)=I(τ;C)=0`. This design measures step-presence information at a known time; it does not demonstrate learning or recovering an unknown onset. A separately sampled time-index variable could encode protocol phase, but would be a different estimand. Count and detector also discard details of within-window spike timing.

For native deterministic outputs `Y=C` or `Y=D`:

`H(Y | S,N)=0`,

`I(N;Y | S)=H(Y | S)`,

`H(Y)=I(S;Y)+I(N;Y | S)`.

These are standard Shannon identities, including the finite-output/continuous-conditioning case ([Gray, §5.5, Lemma5.5.6](https://ee.stanford.edu/~gray/it.pdf#page=143)). They avoid estimating a high-dimensional continuous-noise density, but empirical output frequencies still estimate a population distribution. `I(N;Y|S)` means **stimulus-conditioned information about nuisance in this finite output**. It is not information created by the neuron, a uniquely attributable noise component, or a percentage of the continuous input noise removed. Conditioning can include signal–noise interaction: the elementary XOR example has zero marginal information about either independent bit and nonzero conditional information. No partial-information decomposition has been established here.

For the count/detector comparison, since D is a deterministic function of C:

`I(S;C)−I(S;D)=I(S;C|D) ≥ 0`,

`I(N;C|S)−I(N;D|S)=I(N;C|S,D) ≥ 0`.

This is the information discarded by the **external count-to-detection readout**, not a causal localization to TAN's neuronal threshold. The chain rule and data-processing inequality are standard ([Cover and Thomas, §2.8](https://www.cs.columbia.edu/~vh/courses/LexicalSemantics/Association/Cover%26Thomas-Ch2.pdf#page=21)). Higher within-class count entropy can coexist with robust detection; it is not automatically task failure. Native bounds include `I(S;C)≤1 bit`, `H(C)≤log2(6)`, and `H(D)≤1 bit`. Zero output has zero class information and zero nuisance information, so nuisance minimization alone remains insufficient.

## Separate stimulus noise from output-deletion randomness

v9 LIF_THINNED deletes each generated LIF spike independently with keep probability `p=529/1013`. Its uniforms U are independent of stimulus noise and shared between counterfactual conditions for pairing ([v9 protocol:21](../noise_preservation_v9/PREREGISTRATION.md#L21)). They are **external observation-channel randomness**, not intrinsic neuronal noise; native LIF state/reset/generated spikes are unchanged.

Write C0 for the native LIF onset count and C′ for the retained count. Given S,N, the exact kernel is `C′ ~ Binomial(C0,p)`. Therefore:

`I(N;C′ | S)=H(C′ | S)−E[H(Binomial(C0,p))]`.

For D′=`1[C′≥1]`, the analogous conditional kernel is Bernoulli with success probability `1−(1−p)^C0`. Its entropy is zero on zero-count histories; the randomized channel need not add entropy on every history.

With Y′ equal to C′ or D′, chain-rule decomposition in this specified order gives

`H(Y′|S)=I(N;Y′|S)+I(U;Y′|S,N)`.

The second term is `H(Y′|S,N)` because Y′ is deterministic given S,N,U. Order matters: this is not a claim that the two terms are unique, context-free information sources. Simply comparing total `H(Y′|S)` with native conditional entropy would conflate retained stimulus-noise information with the added deletion randomness.

Conditioning on the actual mask gives the valid but **different** quantity `I(N;Y′|S,U)=H(Y′|S,U)`. It grants mask knowledge and should not silently replace the mask-unobserved comparison. For count marginals, the binomial kernel is sufficient; it integrates the known thinning mechanism without new trajectories or fitted parameters. A realized-mask estimate from v9 outputs and an analytically marginalized-kernel estimate are distinct finite-sample analyses and must be labeled if both are selected. Shared masks affect paired covariance, not the single-condition binomial kernel.

## Estimation and novelty boundaries for preregistration

Freeze fixed output categories, base-2 units, zero-probability convention, estimators, any null/analytic-kernel treatment, and uncertainty before new v10 estimates. Retain seed/history groups when resampling; do not regard the two counterfactual classes or 60 timepoints as independent histories. An empirical zero conditional entropy or maximal class information does not prove the population value; an all-correct empirical bootstrap can remain degenerate. Plug-in finite-sample bias and bootstrap coverage require explicit limits rather than an automatic “unbiased” label ([Paninski 2003, §§3–4 and9](https://www.cns.nyu.edu/pub/lcv/paninski-infoEst-2003.pdf)). In particular, treating each unique continuous noise path as a categorical sample ID would mechanically make empirical `H(Y|S,N)` zero even for one realization of a randomized thinning channel; that is not evidence of population determinism. Use the known model/kernel assumptions rather than that sample-ID artifact.

No new general theorem or information-processing principle is proposed. The potential contribution is a reproducible, controlled diagnosis on the frozen original Task B dynamics: how much of the preserved output is associated with the prescribed binary stimulus, how much stimulus-conditioned variability remains, and how an explicitly external readout/randomizer changes those quantities. It does not establish total neural information capacity, biological equivalence, physical energy efficiency, arbitrary waveform preservation, or attention's unique causal role. Existing P1A evidence is unaffected.

Additional primary-source support for the standard identities, mixed continuous/discrete conditioning, data processing, and estimator limitations is supplied in [PRIMARY_IDENTITY_NOTES.md](PRIMARY_IDENTITY_NOTES.md). `SOURCE_PATHS.json` lists the inspected local sources and hashes for scientific archive inclusion. No model or estimator execution; no blocker.
