# Boundary supplement to the archived TAN synthesis

Date: 2026-11-04. This is a scoped archival supplement, not a new experiment or a replacement manuscript. Source baseline: `9cddbbad7f1722d8a116c0a8245fc40d3a5e97ca`. Exact source identities are in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json).

## 1. Scalar ordering and static addressing

For fixed distinct scalar keys and score `E_i(q)=q k_i`, multiplication by a strictly positive `q` preserves every key ordering; multiplication by a negative `q` reverses every ordering. At `q=0` all keys tie. The unique winner is therefore constant **within each nonzero sign class**, assuming the extreme key is unique. This corrects any reading of the historical theorem that would assign a unique winner to a query at zero.

The frozen positive-query TAN score has `q=β W_q W_k S`, with positive coefficients and nonnegative rectified surprise `S`. Its active queries cannot change the winning key for a fixed candidate array. This is the order-conservation result labeled `thm:order` in historical paper 2 (Theorem 5.1). It is elementary scalar multiplication, scoped to this separable score. It is not a statement about positive-definite kernels.

A different scalar score, `E_i(q)=−(q−k_i)²`, selects a requested distinct key `k_t` at `q=k_t`, with margin `min_{i≠t}(k_i−k_t)²>0`. No increase in query dimension is needed for this static selection. P1A Rev2 records 6,705/6,705 intermediate-key role queries for the metric score and zero for the signed scalar-product score in its noiseless three-key condition. The full key, value, and validity arrays are given to those computations. This is evidence about access to an available array; it is not a learned-writing mechanism or proof that a compact neuronal state stores that array.

Sources: `snapshot/paper2/sections/s5_routing_geometry.tex`; E1's `THEORETICAL_PROPOSITION_1D_METRIC.md`, `P1A_EVIDENCE_REVIEW.md`, and `condition_summary.csv`.

## 2. Exact fixed-alphabet composition floors

Let targets `A,B` be independent Uniform[−3,3], and let numerical decoded outputs belong to a fixed alphabet `C`. Write `e_A=decoded_A−A`, `e_B=decoded_B−B`. The archived composition score is

`L = (|e_A+e_B| + |e_A−e_B|)/2 = max(|e_A|, |e_B|)`.

With nearest-alphabet distance `d_C(z)`, every such decoder satisfies the **pointwise** bound `L ≥ max(d_C(A),d_C(B))`, including coupled or randomized decoders. Independent nearest-codeword choices attain equality in the relaxed oracle problem. If `F(t)=P(d_C(A)≤t)`, target independence gives

`E[L] ≥ ∫₀^∞ [1−F(t)²] dt`.

For Count, `C={−2,−1,0,1,2}`. The residual CDF is `5t/3` on `[0,1/2]`, `(2+t)/3` on `[1/2,1]`, and one thereafter. Its exact expected composition floor is `25/54 ≈ 0.462962963`; per-role mean absolute error is `1/3`.

For Latency, the alphabet consists of the 20 values `−2.85+0.30k`, `k=0,…,19`, plus the zero fallback. The residual CDF is `7t` on `[0,3/40]`, `1/20+19t/3` on `[3/40,3/20]`, and one thereafter. Its exact expected composition floor is `3139/32000 = 0.09809375`; per-role mean absolute error is `0.073125`. Removing the zero fallback would instead give `1/10` and `0.075`, respectively.

These constants are not probabilities of misclassification. Their units are the target's numeric units. They are population expectations, not deterministic bounds on a finite sample mean. For a fixed delivered dataset, the correct deterministic oracle floor is the average of its actual per-episode nearest-alphabet distances under the max score. The relaxed oracle also grants every output code regardless of whether the neuronal dynamics can realize it; reachability or delivery restrictions can make actual achievable error larger.

Both actual-alphabet population floors exceed the historical `0.05` composition-error target under these assumptions. They do not constrain alternative output alphabets, continuous-time codes, different target distributions, internal analog readouts, or spiking neurons generally.

The retained oracle report derives these values with exact rational integration and separately records Monte Carlo checks. Its one-million-pair run extends the same seeded stream as the earlier 200,000-pair run; they are not independent replications. This archive cites those saved results and did not rerun them.

Sources: `research/completed-tan-2026-11-03/native-eight-stage/oracle-validation/REPORT.md`, `validate_oracle.py`, `results.json`, and `independent-review.md`.

## 3. Detection tradeoffs and the adaptation comparator

This is the historical **W10 detection** model, not the W5 rectified-surprise collision model. The comparator implements

`m_t=.8 m_(t−1)+.2 x_t`,

`I_t=κ |(9/8)(x_t−m_t)|`,

`u_t=.8 u_(t−1)^post+I_t`, followed by the shared strict threshold `u_t>1` and zero reset.

A separate 512-history iid calibration fixed `κ=0.840938900477691` by matching pooled absolute current, not test accuracy. It did not match all moments, compute, storage, or physical energy. TAN retains ten history coordinates plus membrane; this comparator retains running mean and membrane.

The frozen test used 1,024 paired innovation histories across iid/AR(0.9), step/catch, and both models. Each waveform had 60 updates, signal onset 15, and detection window 15–19. This supplies onset information to the readout. Misses have capped delay five, so the delay includes failures.

| Noise/model | Misses / 1,024 steps | False alarms / 1,024 catches | Balanced accuracy | Mean capped delay |
| --- | ---: | ---: | ---: | ---: |
| iid TAN | 0 | 1 | 0.999512 | 0.560547 |
| iid ADAPT | 0 | 137 | 0.933105 | 0.103516 |
| AR(0.9) TAN | 62 | 6 | 0.966797 | 0.930664 |
| AR(0.9) ADAPT | 7 | 40 | 0.977051 | 0.301758 |

The AR balanced-accuracy difference TAN−ADAPT is −1.0254 percentage points, with the preregistered 97.5% paired percentile interval [−2.0020, −0.0488]. The change in this difference from iid to AR is −7.6660 points, interval [−9.0820, −6.2988]. These frozen comparisons support a ranking reversal and a false-alarm/miss tradeoff, not equivalence or universal domination. The paired unit is an innovation history; calibration remains fixed during bootstrap.

Benda, Longtin and Maler (2005), *Spike-frequency adaptation separates transient communication signals from background oscillations*, provides primary-literature precedent for adaptation-based signal separation ([PubMed record](https://pubmed.ncbi.nlm.nih.gov/15745957/), DOI `10.1523/JNEUROSCI.4795-04.2005`). The engineering comparator above is not presented as an implementation of that paper's biological model. This archive checked bibliographic/abstract-level indexing; primary full text was not newly retrieved. No new biological inference is made.

The saved report's separately written replay verified spike decisions and numerical records within the same AI-assisted workflow. It is internal computational verification, not an external experiment.

Sources: E4's `REPORT.md`, `PREREGISTRATION.md`, and `metric_summary.json`.

## 4. Habituation and native higher-moment sensitivity

In the fixed-parameter rectified-surprise implementation, a sustained constant input eventually fills the finite window with identical samples. Then current minus window mean is zero, the surprise gate closes, and the attention contribution vanishes. With no additional drive and a leak magnitude below one, post-reset membrane state decays rather than sustaining input-driven firing. No synaptic-weight update is required for this behavior; changing history state supplies the adaptation. This supports a bounded model-level habituation description, not synaptic learning or biological identity.

Historical demonstrations use multiple implementations, including simplified demonstration cells. The archived manuscript also acknowledges that a reported LIF threshold was mismatched to the step amplitude. Its wording about calibrated baselines elsewhere must not override that documented limitation. Those older demonstrations are retained as history; the calibrated detection comparison in §3 has its own stronger and different contract.

For ideal exponential normalization without the denominator epsilon, `C(q)=d log Σ_i exp(qH_i)/dq` has cumulant expansion `μ+qκ₂+q²κ₃/2+…`. Thus attention can depend on moments beyond mean and variance. This is a standard exponential-family identity, not proof of an unprecedented computation or of task improvement. Fixed-query pooling remains invariant to permutations of a given window. The deployed epsilon also prevents treating every ideal-normalization equality as an exact implementation equality.

Sources: `snapshot/code/experiments/habituation.py`, `snapshot/manuscript/sections/experiments.tex`, canonical manuscript and E3.

## 5. What the collision and training records establish

The distractor-swap proof concerns a fixed W5 model, specified amplitude boxes, a shared reset, and a fixed continuation. Its complete allowed observation comprises five pre-reset membrane voltages and five native spikes. A distinct internal ordered window can coexist with an identical permitted response. The equal paired-label mixture therefore imposes a one-half classification error floor for response-only decoding, whereas the lower bound induced in the original continuous generator is tiny. It does not explain the entire empirical failure rate.

The clean-only supplemental certificate establishes exact-real response collisions separately, including a branch without target/query reset. A reduction of the **allowed response map** to a scalar carrier does not say that the entire dynamical state is one-dimensional. Historical unshifted exponential normalization and stabilized normalization differ slightly because both retain an additive denominator epsilon; they were checked separately.

The saved E8 check covers four collision brackets, reset branches, and 64 trajectories. It does not independently rederive every global monotonicity or local derivative certificate. Its maximum saved-scalar difference is approximately `3.33e−16`, which is numerical trajectory agreement, not a universal finite-precision theorem.

The clean-only training intervention uses three historical seeds, 1,500 class-balanced training examples per arm, a fixed 33-parameter linear readout, and 1,001 clean evaluation examples per seed. The mean balanced accuracy rises from 32.5881% to 38.1199% when the training composition changes. This modest gain is conditional on that protocol and does not invert the collision map. Its test-resampling intervals do not integrate over all training or seed-selection uncertainty.

Sources: canonical manuscript, E2, E3, and E8.

## What this does not establish

This archive does not establish a universal hierarchy of all memory systems; universally optimal neuron structure; physical energy or metabolic savings; biological implementation; a general resource-matched benchmark victory; external replication; peer review; or completed journal submission. Later population experiments remain a separate line and do not retroactively transform these single-unit contracts.

## Limitations

This is a review and preservation operation on selected retained artifacts. It does not re-execute all code, audit every historical file, or conduct a comprehensive novelty search. Older source drafts may contain superseded wording. Saved checks and exact arguments have explicitly limited domains; finite observations cannot certify universal behavior. The current closure version should be amended transparently if a later review finds a material error.
