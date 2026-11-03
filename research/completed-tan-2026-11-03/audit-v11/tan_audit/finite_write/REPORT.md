# Finite-time binding needs a counted write mechanism

Completed 2026-11-02. This study directly tests finite-error behavior. A division-free residual writer can meet the N=2 operand tolerance in a declared 10 ms event, but the successful implementation requires a substantially faster feedback/update budget and several explicit resources absent from native TAN. The exact-overwrite obstruction from the previous study is not used as a finite-error impossibility argument.

The fixed 100-update version fails the uniform tolerance even under favorable, accurately captured first anchors. The positive control executes 6,153 literal updates and passes all 512 fresh histories and 25 fixed fixtures. Both are enriched models: three payload states, three temporary registers, a staged controller, key-preserving inputs, calibrated first capture, gated perfect retention and a different query readout. No algebraic interpolation solution is substituted into the implemented write.

## Selected mechanism and its relation to prior work

The candidate follows a residual-times-key update, a familiar fast-weight/error-correction mechanism. The directly inspected primary paper specifies such updates and their controller/normalization machinery; it does not provide a 10 ms biological realization of this scalar candidate. Our contribution is a task-specific finite-tolerance and resource certificate, not a new learning rule. [Schlag, Irie and Schmidhuber (2021), §4.2](https://proceedings.mlr.press/v139/schlag21a/schlag21a.pdf). The [source review](LITERATURE.md) records exactly what was read.

Retain the first key/value in K,V and a slope in S. During the second event, use current inputs k,v to perform

`S <- S + h*gamma*(k-K)*[v-V-(k-K)*S]`.

After encoding, query q returns `V+S*(q-K)`. This anchored representation retains the first record; it is not the earlier two-coefficient overwrite. No second record remains after writing. Subtractions and products gradually correct the residual, so runtime data-dependent division is absent. The equilibrium ratio is a mathematical analysis tool, not a write helper.

The first anchor is also written dynamically. K and V start at zero and undergo 100 Euler charging updates with tau=10 ms and fixed input gain `G=1/(1-.99^100)=1.5773675301`. There is no exact-value assignment at offset. That known-duration calibration reaches the desired first anchor within floating roundoff. Hold switches subsequently set the relevant derivatives to zero. Gain precision, exact capture time, initial reset and ideal holding are explicit conditions, not properties established for original TAN.

## Finite-tolerance certificate, before execution

For exact held anchors, define d=k2-k1 and e=v2-v1. The second-target residual after N updates is

`R_N=e*(1-h*gamma*d^2)^N`.

The original generator has |d| between 0.05 and 2.2, and |e| at most 6 at support closure. At h=0.1 ms the optimal uniform constant Euler gain is `gamma=8000/1937`. Its worst contraction factor is `rho=1935/1937`; after 100 steps the worst closure error is **6*rho^100=5.411106902**, far above 0.5. Exact rational checks show at least **2,406** such minimax updates are needed to bring the uniform bound below 0.5. At the old h this would require 240.6 ms, not 10 ms. Legal interior fixtures approach the bound; excluded distribution endpoints are not essential to the failure.

This is a bound for this constant-gain Euler residual family, not all neurons, all smooth flows or adaptive/hybrid writers. Rescaling keys alone leaves the squared-gap condition ratio 1,936 unchanged. Adaptive normalization could alter it but would introduce a new inverse/normalization operation to implement and count.

The continuous reference has residual `e*exp(-gamma*d^2*T)`. Reaching 0.5 uniformly in 10 ms requires gamma greater than about 99.396 in the declared key units. Raising that gain while retaining h=0.1 violates explicit-Euler stability for large gaps. This separates a discrete-update constraint from a physical continuous-time constraint; an integration step is not itself a biological clock.

The frozen FAST10 control uses `gamma=ln(24)/(10*.05^2)=127.1221532`, with a predetermined factor-two tolerance margin, and `N=ceil(10*gamma*2.2^2)=6153`. Its h is 0.0016252235 ms. Every residual factor is nonnegative, and the worst Euler bound is **0.2497948286**, compared with the continuous bound 0.25. This fixes both gain and update count analytically, without a sweep. The largest effective pole is 615.271 per ms, corresponding to about 1.6253 microseconds; no physical feedback path at that speed was built or measured.

## Completed test

The unchanged generator produced 512 fresh N=2 histories using continuous RandomState(2026110217). Twenty-five fixed fixtures comprise 18 reused earlier support cases, four legal maximum-gap cases, two native-interface collisions and one uncalibrated-capture case. Every history has original, query-swap and re-encoded value-swap variants. All 1,611 inputs were saved and hashed before execution. Both arms produced **3,222 rows and 6,444 operand outputs**, with no exclusions or invalid states.

Fresh original histories, n=512:

| Write budget | Mean Ecomp | Maximum error | Strict ±0.5 joint tolerance |
|---|---:|---:|---:|
| A100, 100 updates | 0.163840971 | 4.435647501 | 457/512 |
| FAST10, 6,153 updates | 0.000812449 | 0.143239546 | 512/512 |

On the 25 fixed original fixtures, A100 maximum error is 5.411106902 with 5/25 within tolerance; FAST10 maximum is 0.249794829 with 25/25 within tolerance. Constructed cases and dependent swaps are not additional IID prevalence evidence. All 12 arm/group/variant metric cells, including MAE/RMSE and continuous-reference diagnostics, are saved. These are new algorithm diagnostics, not replacement P1B pass labels.

Capture endpoint error was at most 3.56e-15. Literal iterative slope endpoints differ from the independent geometric solution by at most 1.44e-12. Query swapping exactly permutes outputs while preserving stored states. All state guards pass without clipping or numerical boundary-band events. Full active-phase trajectories for the 25 original fixtures are saved, with exact constant-hold segments specified by durations; fixed snapshots and complete resource records are saved for every row.

The continuous reference is important for interpretation. It integrates only the second slope stage while granting the actual calibrated Euler anchor; it is not a full continuous K,V,S circuit simulation. A100 mean IID error is 0.164814925 in that reference, close to the actual discrete 0.163840971, so small-gap failure is not merely numerical instability. However, A100's maximum-gap oscillatory recurrence differs from its continuous reference by up to 5.4111 on the fixtures. That branch must not be described as a faithful continuous neuronal trajectory. FAST10's largest fixture output difference from the continuous reference is about 0.0002052. The candidate executes the literal recurrence; neither closed form is used as its writer.

The independent review identified a further numerical-to-physical calibration gap. Keeping the Euler-calibrated G unchanged in an actual continuous tau=10 ms capture produces relative undercharge `1-G*(1-exp(-1))=0.0029135554`. The corresponding converged worst first-query error can reach 1.01288 under the analytic closure formula, above 0.5. Continuous capture would instead require the different fixed calibration `1/(1-exp(-1))=1.5819767069`. That alternative was not added or tested. Thus the positive result is for the declared discrete mechanism, and its second-stage continuous reference does not validate a complete continuous leaky circuit.

## Two controls expose resources that speed alone cannot supply

**First-capture calibration.** With G=1, tau=10 ms and the same 100 capture steps, a legal first key k1=2 is stored as K=2*(1-.99^100). Choosing the second key equal to that value, with values (0,2), leaves effective d=0 even though the true keys are well separated. The actual floating test also gives d exactly zero, S remains zero, and the second-value error is 2 despite the FAST10 gain. The calibrated primary arms handle the same input. This is one fixed diagnostic, not a gain sweep.

Calibration is fragile to timing. Keeping G fixed but using 99 or 101 capture updates changes the anchor gain by about -0.005832 or +0.005774. The independent analytic review gives minimum-spacing examples whose converged first-target error then exceeds 2.3. These are sensitivity calculations, not extra simulated timing arms or a probability estimate. Nominal success does not establish robustness to unknown duration, gain drift or retention noise.

**Native input interface.** Two legal histories have keys/values `[(.8,1),(1,-1)]` and `[(1,.8),(.8,-.8)]`, the same active slots, and common queries (.8,1). Both present native content currents (1.8,0) and zero novelty, but their target vectors are (1,-1) and (-.8,.8). Frozen native M1 code produces exactly identical encoding traces, terminal states and query outputs. For any decoder receiving only that same port history/state and query, at least one target error on this pair is at least **0.9**. This is a finite-tolerance information obstruction for the declared interface, not a population estimate or universal neuron limitation.

The enriched candidate sees separate raw keys and values and distinguishes the pair. A key-preserving input representation, such as raw key alongside the content sum, would remove this particular ambiguity but is a changed interface. More internal settling speed cannot recover information already merged by the native inputs. The native comparator uses its own readout contract; its numerical score is not an equal-operation contest with the enriched readout.

## Complete resource interpretation

- **Stored payload:** K,V,S, three signed/typed analog coordinates, represented as 192 binary64 bits. Their ranges and roles differ from native u,A and the restricted refractory timer r.
- **Temporary storage:** three explicit binary64 scratch registers, another 192 allocated bits, cleared between updates. These are counted even though no history survives in them. Evaluator traces never feed back into the writer/query.
- **Control and clocks:** two stage bits plus external event/reset/clock signals. A standalone ordinary step counter would add 7 bits for A100 or 13 for FAST10, giving 9 or 15 digital bits with stage control, before oscillator/timing hardware. The supplied external timing is an assumption, not a complete two-bit physical controller.
- **Arithmetic:** each second-write step uses three subtractions, three multiplications including constant gain, and one addition. There are zero runtime data-dependent divisions. FAST10 uses 18,459 write multiplications and the same number of subtractions per history, versus 300 each for A100. High-precision current input and query/output arithmetic remain extra requirements; their physical latency is not simulated.
- **Retention and gains:** first-capture G, gamma, switched perfect hold, reset and known durations are fixed resources. No autonomous slope leak, uncertain initial state, process noise, sensor noise or output quantization is tested. An added ordinary slope leak would leave a bias, as derived separately in THEORY.md.
- **Energy:** no joule/ATP/area claim is made. Per-coordinate square, squared-update-rate and total-variation integrals retain their distinct units. For fresh originals the mean slope squared-rate proxy is 14.7691 for A100 and 396.4221 for FAST10. The oscillatory A100 maximum-gap fixture accumulates about 518.226 units of slope variation, versus 2.7273 for FAST10; this is not a physical energy ranking. Resource allocation/holding durations and all proxy values are preserved per row.

The [resource clarification](RESOURCE_CLARIFICATION.md) distinguishes provisioned scratch capacity, external clock assumptions and a possible internal counter implementation. No equal-state-count, equal-energy or full-hardware equivalence follows from these numerical models.

## Conclusion and evidence

Approximate finite-time binding is achievable for this explicitly enriched residual mechanism under its declared gain, input, precision and retention assumptions. The registered 100-step constant-gain budget cannot provide its uniform accuracy guarantee. The concrete missing resources are a suitable input representation, accurate capture/holding and sufficiently fast or differently normalized residual correction. This is a narrower and more useful conclusion than extending an exact-flow obstruction to all approximate computation. No N=3 result or general minimal-state theorem is claimed.

The main command `PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/finite_write/run_finite_write.py` completed with exit 0, recorded CPU wall time 2.073 s. That wall time is not neuronal write latency. Exact certificates completed separately, and independent implementation/ledger review is recorded in REVIEW.md and its artifacts.

Protocol SHA256: `01a94f178132b6b6210a8609dfcd4393c22f8c73f847cfddbc23fd410513a7a1`. Input SHA256: `b9c9b03937463bd519e3ada94bcec5c2c8551b48791d31641b2737dff9887100`. Main source SHA256: `df38bebc8f12c012829294e1decb88b040aa45b83233f38f326c803ab66846a2`.

All previous results and original repository labels remain preserved.  the absent historical whole-archive bytes remain outside the scope of source audits already disclosed.
