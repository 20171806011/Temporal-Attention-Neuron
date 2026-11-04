# Independent pre-execution assessment of the proposed Probe1 swap family

2026-11-02. This is an analytical design assessment prepared prospectively, before any new fixture/model execution by this reviewer. It supplements `NATURAL_COLLISION_REVIEW.md`. No proposed history was passed to a simulator.

Use the original Probe1 present-distractor timeline, zero outside t5–9, and paired histories

`H=(a,b,c,d,q)`, `H'=(b,a,c,d,q)`

at t5–9, with `a,c ∈ [123/200,127/200]` and `b,d,q ∈ [273/200,277/200]`. Thus low coordinates lie in [.615,.635], strictly inside lo-only support, and high coordinates lie in [1.365,1.385], strictly inside hi-only support. The histories carry opposite target labels while sharing the same current query.

## Uniform branch proof obligations

At t8, the window is a permutation of `(0,a,b,c,d)` and

`S8 = (4d-a-b-c)/5 ∈ [.561,.589]`.

The study's more conservative independent-variable bound `[.557,.593]` is also valid. Set `g=2S8 ∈ [1.114,1.186]` and δ=10^-9. For the unshifted exponential normalization in `natural_collision.py`, context C8>1 is equivalent to

`(a-1)e^(ga)+(b-1)e^(gb)+(c-1)e^(gc)+(d-1)e^(gd) > 1+δ`.

A uniform lower bound is exactly

`2(.365) exp(1.114·1.365) − 2(.385) exp(1.186·.635)`.

Both sign choices are conservative: the positive high-value terms use their smallest coefficient and exponent, while negative low-value terms use their largest magnitude and exponent. The source's δ belongs on the right-hand side. The intended rational certificate can bound exponentials using a positive Taylor lower sum and an upper remainder: for `0≤x<n+2`,

`T_n(x) ≤ exp(x) ≤ T_n(x) + [x^(n+1)/(n+1)!]/[1−x/(n+2)]`.

This avoids treating a floating exponential as an exact proof. A separate positive fourth-order Taylor lower bound for exp(1.114)>3 proves tanh(.557)>.5. Together, C8>1 and tanh(S8)>.5 imply current A8>.5, and both histories reset to h8=0 regardless of their earlier nonnegative h7. The simulation must still log the complete branch and source implementation rather than silently assume this reset.

At t9, `S9=(4q-a-b-c-d)/5 ∈ [.284,.316]`. The conservative `(0,.32]` bound is valid. All window values are ≤1.385, and the denominator offset makes context no larger than the maximum. Thus with h8=0:

`0<u9=tanh(S9) C9 < .32·1.385 = .4432 < .5`.

The window multiset and the query coincide across the pair; ideal u9 therefore coincides by permutation symmetry. Query gating is strictly positive and the query readout is strictly subthreshold. After t9 the native generator supplies zeros, so S10..13=0 and the entire permitted response equals

`[u9,u9/2,u9/4,u9/8,u9/16,0,0,0,0,0]`.

This is an ideal response identity, not merely close h matching. Actual floating differences must be logged separately, since summation order can create ULP differences. The full ordered window remains different and contains the target amplitude at its known slot.

## Controls

The ordered target-tap oracle distinguishes opposite classes without ambiguity in these nonoverlap support intervals. Original B2 has j8≥.974 for H and ≥.924 for H', so its h8 resets in both cases. Its existing query pre-reset feature differs by `(b-a)/15 ∈ [.048666..., .051333...]`. This is a positive control for the existing observation/position distinction, not an assertion that a particular B2 fitted classifier achieves perfect population performance.

Original LIF provides a negative control: active pulses exceed threshold and its query output is determined by the common q. B3 must be reported without presupposing its outcome. The design proposes a single t8 reset-suppression B4 control; it must be clearly identified as an intervention, with all other update rules fixed and a precise definition of whether the t8 spike flag is preserved. Its outcome is not part of the original-model theorem and must not be assumed before execution.

## Density and inference

Conditional on a present-distractor trial in the original generator, the target, three distractors and query are independent draws from the same fair lo/hi mixture. Each component is uniform on an interval of length .5. On the selected nonoverlap low or high intervals, the mixture density is exactly `.5/.5=1`. Thus each orientation box has conditional probability mass

`(.02)^5 = 1/312500000 = 3.2e-9`.

Swapping the target and first distractor is measure preserving. The mirrored boxes are disjoint because target ranges are disjoint. Their union mass is 6.4e-9. Any classifier restricted to the identical ideal response has classification error at least one half conditional on this union, and hence at least 3.2e-9 over the original **present-distractor conditional distribution**. Randomized classifiers obey the same paired bound. This tiny analytic population bound is not an empirical prevalence estimate from the planned 512 constructed pairs and does not explain most of the observed task error.

Do not silently relabel this as a lower bound on the paper's balanced-three-class metric. Nor automatically multiply by 1/3 for the exact finite dataset: `make_trials` fixes rounded catch/present/distractor counts (for n=2000, present-distractor count is 666), although the conditional density statement is unaffected. An explicitly idealized population condition mixture could be specified separately if needed.

The fixed box tests validity across admissible values while maintaining the paired permutation. It does not establish robustness to independently injected noise, finite sensor precision, altered timing, arbitrary future probes or a different observation contract. The general mathematical ingredients are standard symmetry and reset facts; the project-specific diagnostic is their implication for the existing labelled Probe1 task and all of its permitted response features.
