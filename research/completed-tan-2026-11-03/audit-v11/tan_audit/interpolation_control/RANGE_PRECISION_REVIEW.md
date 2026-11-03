# Independent range, precision and leak interpretation check

This review reads the frozen protocol, input ledger, predictions and exact/precision checks. It does not change primary files or run another parameter experiment. The separate ledger-only script `range_precision_check.py` recomputes raw versus scored errors and checks3,180 raw perturbation/leak-bound rows. All checks pass; the largest residual in the common/differential leak identities is4.22e−14.

## Dynamic-range rejection is different from arithmetic error

The preregistered guard treats coefficients beyond±120/±297 (relative allowance1e−10) as an invalid device state and emits zero. This is an explicit finite-range device policy; it is not a statement that the unguarded interpolation arithmetic failed to retrieve the target. Both errors are correctly retained in the ledger.

| Original-history group/condition | Rejected | Raw mean Ecomp | Raw maximum error | Scored mean Ecomp with fallback |
|---|---:|---:|---:|---:|
| IID interpolation float32 |0/512|2.93049e−7|3.96399e−6|2.93049e−7|
| Adversarial interpolation float32 |16/18|3.94746e−16|4.44089e−16|2.66666667|
| Adversarial interpolation coefficient noise |9/18|.15524445|.52777419|1.56430824|
| Adversarial interpolation write/query key noise |8/18|.01255399|.02303969|1.33951780|
| Adversarial M0 cache state noise |16/18|.00203196|.00281732|2.66687945|
| Adversarial M0 write/query key noise |10/18|0|0|1.66666667|

All16 minimum-gap float32 interpolation cases exceed the slope limit, by4.5776367e−5 at the low-key edge or1.14440918e−4 at the high-key edge. High-key cases also exceed the intercept limit by2.74658203e−4. These overshoots exceed the specified numerical allowance, so their status is correctly RANGE_EXCEEDED. Yet their actual binary32 query computation returns±3, matching the stored targets up to4.44e−16. Hence the2.6667 fallback score must not be described as a raw float32 reconstruction error or an inherent inability of float32 to implement interpolation.

This endpoint coincidence does not mean the rounded coefficient line is the exact ideal line. For example, `adv_8` has coefficients(120.00011444091797,−297.0002746582031). Evaluating those coefficients as exact reals at the rounded query gives an approximately1.1444e−5 error; binary32 multiply/add rounding cancels it at the tested endpoint. The precision ledger correctly separates coefficient errors from final readout arithmetic. Such cancellation is fixture-specific, not a robustness theorem for nearby queries.

M0's adversarial noise failures also arise chiefly from guards at nominal key/value endpoints. Its unguarded nearest-key retrieval remains correct under the stated small key perturbations, and its value error stays below.003 under state noise. A perturbed stored key just below.3 or a value just beyond±3 is rejected by the frozen range policy. Comparisons should therefore show raw and fallback metrics together and should not attribute every adversarial failure to key-binding loss. No posthoc guard relaxation is recommended for this registered run; additional dynamic-range headroom would be a separately declared engineering assumption.

The noise comparison uses different physical coordinate assumptions: interpolation perturbs raw slope/intercept by(.12,.297), while M0 perturbs key/value by(.0011,.003). Equal fractional fullscale is transparently specified, but neither equal raw noise, equal signal-to-noise ratio nor equal physical noise/energy follows. A fixed centering of the affine representation changes these sensitivities without changing ideal expressive capacity.

## Perturbation formulas

For an ideal valid query q with v=aq+b, let stored coefficients become a+δa,b+δb and measured query q+δq. Exact output error is

`q*δa + δb + a*δq + δa*δq`.

Thus `|e| <= |q|εa + εb + (|a|+εa)εq`, and the generator support supplies the uniform bound`2.5εa+εb+(120+εa)εq`. The fixed fullscale perturbation gives.597 without query noise and.609012 with query bound.0001. These are bounds on raw affine outputs before range rejection, with floating evaluation accounted separately. They are not bounds on the zero-fallback scored error after invalidation.

For measured write keys ki+δki with |δki|<=εk and exact values, the fitted line passes through the two measured keys. Its denominator magnitude is at least d−2εk, so for d>=.05 and εk<.025,

`|a_tilde| <= 6/(.05−2εk)` and `|error at a true target/query| <= 6(εq+εk)/(.05−2εk)`.

For εk=εq=.0001 this is.02409638554. It holds for all530 raw write/query-noise cases. It does not guarantee the perturbed coefficients remain within the unperturbed support ranges120/297, which explains some independent range failures. Values were not perturbed in this write-noise scenario.

## Leak mechanisms

For common post-encoding coefficient leak a→λa,b→λb and a clean true query, the entire interpolant scales: error=(λ−1)v. Hence the bound3|1−λ| is valid. With the registered10ms/150ms choice it is.1934790449; adversarial finite results agree at the boundary within4.22e−14 floating residual.

For differential post-encoding leak a→λa*a,b→λb*b,

`error=(λa−λb)*a*q + (λb−1)*v`,

so the uniform bound is`300|λa−λb|+3|1−λb|`. Loss of cancellation can therefore cause large errors even though each coordinate's magnitude decreases. In the registered adversarial histories, the maximum raw/scored error is9.30234044. This does not contradict the common-leak bound because the two mechanisms differ.

Leakage before the second write acts on the first stored key/value record, not on final coefficients. It changes the denominator and can make two measured keys coincide. The designed valid-input example yields WRITE_SINGULAR as intended. The IID interwrite-leak run has raw mean Ecomp2.96080 and raw maximum527.67888; after three range rejections, scored Ecomp is1.69115 and its maximum51.21965. Smaller post-guard mean error is a fallback effect, not evidence the underlying erroneous interpolants are more accurate. The adversarial raw average excludes the one undefined singular division and must keep that exclusion explicit; its scored result includes the singular fallback.

The nonzero delays here are labeled mechanisms, not physically calibrated equivalence experiments. Perfect holding assumes no state decay. Common and differential coefficient leak test specified coordinate dynamics, while interwrite leak additionally depends on the known event schedule and phase controller. None supplies a biophysical neuron implementation or an N=3 claim.

Detailed numerical summaries and every adversarial float32 payload/overshoot are in `RANGE_PRECISION_CHECKS.json`.
