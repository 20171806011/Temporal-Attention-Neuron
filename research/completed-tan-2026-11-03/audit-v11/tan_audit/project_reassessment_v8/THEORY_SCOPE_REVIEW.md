# Independent proof, admissibility and scope review

2026-11-02. **PASS for the ideal paired-family theorem and its explicitly conditional probability bound.** No new model trajectory, fixture or decoder was run by this reviewer. The review directly inspected the frozen Probe1 generator, its response-feature function and the original four-model simulator. An independent exact-rational interval calculation completed with exit 0; its values are in [THEORY_SCOPE_CHECKS.json](THEORY_SCOPE_CHECKS.json).

Reviewed preregistration SHA256: `d8f0b97300f7fd1d3e3b0c07cd5ca17c3cfb4b40b20efa25792e6ab3dd78df23`. The numerical trajectory reconstruction is a separate review, and this note does not claim its outcome.

## 1. Original task and admissibility

`snapshot/code/experiments/temporal_distractor.py:72–89,103–146` fixes target t5, distractors t6–8, query t9, zeros elsewhere, and fourteen-step blocks. The label is the target's latent low/high class or absent; the query class is not the label. Its allowed feature is the ten-vector of pre-reset u9–u13 followed by the corresponding threshold events. The code does not grant the readout raw history or pre-query observations.

The chosen low interval [.615,.635] lies strictly below the high-class support's lower bound .95; the high interval [1.365,1.385] lies strictly above the low-class support's upper bound1.05. Both are interior subsets of their own original class supports. Consequently the paired histories `(a,b,c,d,q)` and `(b,a,c,d,q)` have different, unambiguous target labels, compatible distractor/query latent classes, and a common query. There is no overlap-label ambiguity in this family. The full-window target oracle's correctness is restricted to this family; it is not an oracle for latent class within the original overlap region.

## 2. Exact common-reset and positive-query proof

The source `natural_collision.py:203–252` uses unshifted exponentials, denominator `sum(exp)+1e-9`, old post-reset h in the leak term, strict candidate `u>.5`, and reset to zero. These details matter: replacing the task feature by post-reset h or changing the softmax normalization would require a new verification.

At t8, both windows are permutations of `(0,a,b,c,d)`, and

`S8=(4d-a-b-c)/5 ∈ [561/1000,589/1000]`.

Thus the preregistered looser bounds [.557,.593] are valid. Put γ=2S8 and δ=10^-9. Let D=sum(exp(γx)) and N=sum(x exp(γx)). The correct exact identity is

`(D+δ)(C8−1)=N−D−δ`.

The protocol's informal expression `(sum exp)*(C8−1)` omits δ from this prefactor. This is a notation clarification, not a failure of its sign argument: the sign of C8−1 is still the sign of the stated numerator. The frozen protocol need not be edited; the certificate/report should use D+δ explicitly.

The numerator has the following uniform lower bound:

`L=.73 exp(1.114·1.365) − .77 exp(1.186·.635) −1−δ`.

The two high terms have positive coefficients, so their smallest coefficients and exponents give lower bounds. The two low terms have negative coefficients, so the largest magnitude and exponent give the required lower bound. The zero tap contributes −1. Independent Fraction arithmetic, with degree40 positive Taylor sums and a geometric upper tail, certifies **L>0.7045935403361287**. No sampled floating exponential is used to establish the sign.

The same calculation certifies `exp(1.114)−3 > .0465201352794557`; therefore `tanh(.557)>.5`. Hence C8>1 and A8>.5 uniformly. Nonnegative inputs and zero initialization give h≥0 by induction, so u8≥A8>.5 and both native histories reset h8=0 regardless of their different earlier trajectories. A slightly stronger independent conservative combination gives A8>.528613674287212, but the theorem only needs the strict .5 comparison.

At t9,

`S9=(4q-a-b-c-d)/5 ∈ [71/250,79/250]=[.284,.316]`.

The gate is therefore strictly open. The context is positive and at most the maximum window amplitude1.385; the positive denominator offset only reduces the sum of attention weights. With h8=0 and tanh(S)<S for S>0,

`0<u9<(.32)(1.385)=277/625=.4432<.5`.

The two query windows have identical multisets and common current q. Ideal S9, context, current and u9 are consequently identical. Ordered attention vectors can permute; the scalar aggregate does not.

After query, the original generator supplies zero inputs. Window means remain nonnegative, so rectified surprise and gated current are zero. The complete allowed feature is therefore exactly

`[u9,u9/2,u9/4,u9/8,u9/16,0,0,0,0,0]`

for both histories. This proves equality of the entire permitted observation, rather than just one chosen projection. The ordered windows still differ at t9 and t10, and become equal after t11 under the declared zero continuation. At query, the full ordered state still contains the target amplitude in its known position; full-system state equality is neither assumed nor claimed.

The proof is for ideal arithmetic. The independent numerical review must separately report literal floating equality and residuals. A tiny numerical gap alone cannot exclude a decoder with arbitrary floating precision. The strict analytical margins are large relative to usual arithmetic roundoff, but that observation is not a universal finite-precision or noise theorem.

## 3. Positive and negative controls

B1's active pulses all exceed threshold, so its scalar state resets and its query response depends on the common q. B2's recency weights retain order; its t8 current lower bounds exceed .5 on both orientations and its existing query u difference is `(b−a)/15`, within [.048666..., .051333...]. It is a positive control for permitted-observation separation, not proof that the previously trained B2 classifier is perfect.

Original B3 has a different history evolution and must be reported without imposing a desired result. Comparing native B3/B4 is not a matched single-step intervention on attention alone. The t8-reset-suppression B4 arm is a clearly named changed update: it preserves the event flag while retaining candidate voltage for one step. It can test whether that reset destroys an order-dependent membrane residue in this construction. It is not a native biological implementation, and the response-collision theorem does not depend on that intervention producing the hoped-for separation.

## 4. Probability statement

Conditional on an original **present-distractor** trial, target class, distractor classes and query class are independent fair choices, and their amplitudes are independent uniforms over length-.5 class intervals. On each selected nonoverlap interval, the unconditional mixture density is `.5/.5=1`. Each of the five coordinates has interval width .02, giving orientation-box mass

`m=(1/50)^5=1/312500000=3.2×10^-9`.

The target/distractor swap is a unit-Jacobian involution between disjoint boxes, preserves density, reverses target label and preserves ideal F. For any randomized F-only classifier, the two paired probabilities of correct classification sum to at most one; a third absent response cannot improve this bound. Thus average error conditional on the mirrored union is at least1/2, and the contribution to classification error over the original present-distractor conditional population is at least m. The union mass is2m=6.4×10^-9.

This is an analytical lower bound from the specified generator, not an empirical prevalence estimate from 512 constructed draws. It is extremely small and cannot explain the observed large average error. It is not directly a bound on the reported balanced-three-class metric. The finite generator fixes rounded condition counts, so an overall-population multiplier must not be silently assumed to be1/3. Correlation between its reused clean/distractor class arrays does not change the marginal conditional density used here, but no independence of all finite-dataset trials is being claimed.

## 5. Novelty and limits

Existing Phase1 already establishes h-only insufficiency, including an exact common-reset control; existing Probe2 already diagnoses scalar attention's order-preserving selection. The revised manuscript also distinguishes representation complexity from selectivity. The new check is narrower and complementary: a valid opposite-label family whose *whole frozen Probe1 response* is identical while full ordered state contains the answer. General permutation invariance, threshold reset and paired risk arguments are standard mathematics, not new theoretical discoveries.

The fixed box demonstrates that the branch and collision are not confined to one numerical point while preserving the paired exchange structure. It does not establish robustness under independently perturbed histories, sensor/process noise, altered timing, an extra query, a different observation contract or arbitrary future inputs. In particular, it does not prove universal absence of memory, insufficiency of the complete ordered state, or a state-count lower bound for neurons.

No proof gap was found beyond the harmless denominator-prefactor clarification above. No original sources, primary outputs or protocol bytes were changed. No model execution, fitting, publication, installation or spending was performed by this reviewer.
