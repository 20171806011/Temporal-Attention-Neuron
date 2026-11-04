# Finite-time residual writing: source justification and mechanism scope

Prepared 2026-11-02 before inspecting any finite-write experiment outcomes. No model experiment or parameter search was performed for this review. This is a narrow source and interface review, not an exhaustive literature survey or a claim that a new biological circuit has been found.

## Primary-source justification

Schlag, Irie and Schmidhuber's ICML 2021 paper gives a fast-weight update proportional to the difference between an incoming value and the value retrieved with the current key. Equation 20 performs retrieval; equations 21–24 specify write strength and a residual-times-key update. The slow network generates keys, values and write strengths. Their normalization discussion explicitly distinguishes additional accumulator state from normalizing feature vectors. These operations motivate examining a residual-based writer while counting its controller, memory and normalization resources. They do not establish a 10 ms neuronal implementation or the proposed scalar anchor/slope circuit. [Primary paper, §4.2 and equations 20–29](https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf), [proceedings record](https://proceedings.mlr.press/v139/schlag21a.html).

Read scope: the primary PDF's abstract, fast-weight background and §4.2, including equations 20–29 and surrounding normalization discussion, were inspected directly. The prior interpolation review also inspected other portions, but those earlier observations are not a substitute for this source-specific read. No empirical timing, analog-energy or biological claim is imported from the paper.

The original Widrow–Hoff 1960 paper was located in the author's Stanford archive, but this browser interface exposed no readable text from its scanned pages. It is therefore **not used as a content source** in this report, nor is the 2021 paper's historical attribution treated as an independent inspection of that original paper. The single directly inspected primary research source suffices to justify this candidate family. The local calculus below is our derivation, not a result attributed to that source.

## Candidate is scalar residual descent, not hidden algebraic inversion

Let three retained analog variables be K, V and S. K and V retain the first event's key and value, and S starts at zero. During the second event define the instantaneous combinational quantities

`d = k2 - K`, `e = v2 - V`,

and evolve

`dS/dt = gamma * d * (e - d*S)`.

The proposed query is `V + S*(q-K)`. For fixed held K,V and current k2,v2, this is gradient descent on the scalar quadratic `0.5*(e-d*S)^2`. Its stationary value is S=e/d for d nonzero, but the differential update itself needs multiplication and subtraction, not a runtime division. A simulator must actually integrate the declared update; assigning the stationary value, or using a closed-form exponential as the implemented writer, would silently grant a different operation. The closed form is appropriate as an independent mathematical reference.

This is a constrained one-dimensional least-squares correction anchored at the first stored point. It is not identical to the 2021 matrix fast-weight algorithm. It retains a first key/value record after encoding, unlike the previous two-coefficient representation. That additional retained anchor is an explicit resource, not an external cache disguised as a helper function. It is also not an N=3 solution or a test of membership for an arbitrary absent key.

## Compatibility with the actual repository

The inspected frozen files are `p1_experiments/p1b/models_p1b_v2.py` and `p1_experiments/task_generator.py` in commit `85213f0d417d54e4f48b301ed1c0e4200ef3c421`.

| Resource or operation | Frozen TAN implementation | Candidate requirement |
|---|---|---|
| Input | Content current k+v and novelty max(0,k-1.4) | Separate current raw k and v ports |
| Retained state | u, nonnegative A bounded by 5, refractory timer r in [0,2] ms | Held K and V plus a signed slope S |
| Gate update | A relaxes toward novelty | Store and preserve a content-dependent key/value anchor |
| Interaction | Old A multiplies content current | Reconstruct residual e-dS, then multiply it by d |
| Query | q enters membrane current; fixed affine map of deadline u, or spike decoder | Multiply query difference by S and add V |
| Controller | Event timeline, specified reset/refractory rules | Distinguish capture, hold, slope-write and query phases |

These are substantive changes. In particular, low keys have zero novelty, and k+v alone cannot identify separate raw k and v. The candidate cannot claim the same accessible input interface. Nor does naming three analog coordinates make them equivalent to TAN's two continuous state coordinates plus a constrained refractory timer. The query is also an enriched algebraic readout. A native-TAN comparator may preserve its own contract, but its score must not be described as an equal-operation comparison.

## Selected narrow comparison and its justification

First isolate second-write convergence with ideal held anchors as a declared favorable control. For exact first anchors and gap d, the residual after T is

`e(T) = e(0)*exp(-gamma*d^2*T)`.

At the smallest legal gap 0.05 and maximal value difference approaching 6, achieving a uniform error below 0.5 within 10 ms requires `gamma >= log(12)/(10*0.05^2)`, approximately 99.4 in raw key-coordinate units. With the original explicit Euler interval h=0.1 ms, stable contraction for the largest gap approaching 2.2 requires `gamma < 2/(h*2.2^2)`, approximately 4.13. These inequalities expose an incompatibility for this **fixed-gain explicit-Euler recurrence**, not for all continuous, adaptive-gain or hybrid circuits.

A fixed-gain minimax recurrence provides a stronger quantified certificate: on squared-gap interval [m,M]=[0.05²,2.2²], the optimal constant `h*gamma` is `2/(M+m)`, with worst endpoint residual factor `(M-m)/(M+m)` per step. The selected first arm uses this analytically predetermined gain and the original 100 Euler updates over 10 ms. Legal near-boundary gap controls test the certificate, rather than selecting a gain from observed success.

The selected positive control changes the response-bandwidth budget explicitly: set `gamma_fast=log(24)/(10*0.05^2)` and use `n=ceil(10*gamma_fast*2.2^2)` equal Euler substeps within the same 10 ms event. This ensures a nonnegative residual factor for every legal gap, while the exponential upper bound gives residual at most 6/24=0.25 with ideal capture. The gain and update count are selected from the domain and a predetermined factor-two margin to the 0.5 delivery tolerance, not fitted to samples. The two arms form one fixed resource comparison, not a gain or architecture sweep.

The fast control corresponds to a much faster effective pole gamma*d_max². That response speed is a resource; numerical integration substeps are not automatically biological clock cycles. A smaller integration step can establish that the declared continuous proposal works mathematically, but does not establish its physical bandwidth or energy. A gain normalized by d² could remove the stiffness disparity only by supplying an inversion/normalization operation that must be counted or dynamically implemented; it is not included in this comparison.

## Capture, hold and precision cannot be omitted

Ordinary first-order charging from zero for a finite event gives K=(1-exp(-T/tau))*k1 and the analogous V. The legal second key can equal this imperfectly captured K even when its distance from the true k1 exceeds 0.05; then d=0 and this update cannot correct the second value. An optimistic perfect-anchor test must therefore be identified as a control, not the complete online mechanism.

A known-duration static drive gain can compensate ideal linear charging. The selected implementation uses 100 actual Euler steps with h=0.1 ms and tau=10 ms, shared gain `G=1/(1-0.99^100)`, and zero initial states. K and V are not assigned their target values at the event offset. G is an offline fixed calibration constant, not a per-history division, but it requires known event duration, gain precision, initial reset and a phase switch. Duration and gain errors can spoil the compensation and are not tested by merely matching the nominal recurrence. Subsequent gated perfect hold is explicitly enriched; it is not the native TAN relaxation law.

In raw coordinates gamma has units 1/(key-unit² ms), whereas 1/tau_m has units 1/ms. A numerical equality between them is not a native-gain comparison. The effective pole gamma*d² has units 1/ms. Any rescaling of keys, values or S changes gain and noise units and must be recorded together with amplitude range and readout precision.

Count three persistent analog states, controller bits/phase inputs, temporary arithmetic operands, raw-input precision, two multiplications per residual evaluation, gain multiplication, subtraction/addition, and query arithmetic. If the implementation stores d or e across steps, those are extra retained registers; recomputing them from held states and current event inputs avoids that storage but requires the corresponding combinational circuitry. An arithmetic schedule and operand width are not a measured transistor, energy or latency budget. S can approach magnitude 120 in ideal full-domain interpolation, so it cannot be identified with a bounded nonnegative gate or a 2 ms timer without an explicit conversion.

## Interpretation rule

A decisive negative fixed-recurrence result identifies a needed operation or resource: faster effective response, adaptive normalization, additional write time, changed state representation, or a different hybrid update. It does not extend the prior exact-flow obstruction to approximate behavior. A constructive enriched result must report all its interface and resource changes and retain the registered TAN outcomes unchanged. The strongest useful next step is a finite-tolerance convergence/resource certificate for one declared mechanism, followed by a bounded validation of that certificate; an architecture sweep is unnecessary.

 
