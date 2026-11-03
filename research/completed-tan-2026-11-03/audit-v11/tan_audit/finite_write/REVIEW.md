# Independent review: finite-time anchored residual writer

Reviewed 2026-11-02. **Pass for the declared discrete, enriched algorithm and its candidate-specific approximation bound.** The result does not validate a complete continuous neuronal circuit or establish a general obstruction to approximate binding. No primary protocol, input, script, result or historical repository file was modified by this review.

Executed verification:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/finite_write/independent_review.py > tan_audit/finite_write/independent_review.log 2>&1
```

The command completed with **exit 0**, in 2.23 seconds. The checker does not import the experimental writer or repository generator. `INDEPENDENT_REVIEW_CHECKS.json` and `independent_review.log` retain its results.

## Independently verified evidence

The checker reconstructed the exact generator random-call order from the inspected specification, including 30 rejected key draws, all 512 fresh histories, their dependent query/value variants and all 25 fixture variants. All 1,611 input records match. Every history obeys the numerical half-open key/value ranges, key separation, slot and valid-query contract.

All 3,222 prediction rows and 6,444 operand outputs match an independently evaluated geometric recurrence. All 12 arm/group/variant metric cells, resource aggregates, query swaps and saved hashes were checked. There are no invalid rows, excluded histories or numerical boundary-band cases. Query-swapping exactly reverses predictions without changing the final payload. The adversarial cases, including reused conditions with different timings, remain support checks rather than a population prevalence estimate.

The reviewer checked 25,776 stored snapshots and the complete first/second-event trajectory arrays for all 25 original fixtures in both arms. The constant holding intervals are represented by their exact durations and unchanged endpoints; the saved event-local arrays plus this metadata reconstruct the entire episode. Maximum final-slope discrepancy from the geometric reference is 1.44e-12; maximum full fixture-slope discrepancy is 3.77e-12.

Resource checking covered 135,324 fields. Independent geometric-series formulas reproduced coordinate extrema, total variation, coordinate-square integrals, squared-rate integrals and holding contributions. The maximum absolute resource-formula discrepancy was 7.28e-9, within the declared absolute/relative checking tolerances. Large-valued derivative integrals make an absolute discrepancy alone an inappropriate relative-accuracy measure. These separately dimensioned proxies are not physical energy and must not be summed as if their units matched.

Exact Fraction comparisons confirm `6*(1935/1937)^2405 > 0.5` and `6*(1935/1937)^2406 < 0.5`. This is the fixed-gain Euler-family certificate, not a statement about every possible adaptive, normalized, hybrid or continuous mechanism.

## Implementation and retained state

The writer actually performs the registered capture and residual updates. It does not assign an ideal first anchor at the capture offset, call a data-dependent division during writing, jump to the stationary slope, or use the closed-form propagator as its implementation. The geometric and continuous formulas appear only in evaluator code after the state evolution.

`Writer` retains K, V, S, phase and three explicitly allocated scratch arrays. The scratch arrays execute subtraction/product operations and are zeroed between updates. They are additional computational storage: three payload reals plus three temporary reals, not a three-register total implementation. The scalar interpretation is 192 payload bits plus 192 scratch bits at binary64 width, apart from stage bits, external input/output precision and control infrastructure. Actual Python/NumPy storage and evaluator histories are larger; the report does not claim these logical widths measure host memory or a transistor-level circuit.

The query method accepts only `self,q` and uses K,V,S plus scratch arithmetic. It has no access to targets, event IDs, stored positions, the second event record or evaluator traces. The retained first anchor is explicit and remains after encoding. The algorithm therefore uses a different three-coordinate representation from the earlier two-coefficient writer.

Capture and readout also require separate raw key/value inputs, event timing/reset, gating and arithmetic absent from the native TAN contract. The native refractory timer is not an unrestricted signed slope register. An equal number of named payload coordinates does not establish equal physical state ranges or dynamics.

## Resource accounting qualifications

The logged two stage bits describe the candidate's finite event-stage state. The high-rate clock and event-active/onset/offset/read signals are external inputs. They do not constitute a complete physical controller implementation. If a sequential digital implementation internally counts the local write steps, at least 7 counter bits for 100 steps or 13 for 6,153 steps are needed in addition to stage bits; other oscillator/timer implementations have different unmodeled resources. A source of exact event timing also has costs. These distinctions should accompany the stage-bit count rather than be hidden in a host loop.

The 60 scratch-real-ms field denotes provisioned scratch capacity during the two 10 ms active windows: three allocated registers times 20 ms. It is not a measurement of instantaneous nonzero scratch occupancy. Capture uses one scratch register in its arithmetic sequence, whereas the slope update uses all three; the internal arithmetic-operation durations were not modeled. Total allocated scratch capacity is 300 real-ms over the episode. Query arithmetic latency is likewise not simulated, so no physical 10 ms end-to-end query guarantee follows.

A100 and FAST10 use the same payload, nominal event duration and holding assumptions, but different feedback gains and update counts. FAST10 performs 61.53 times as many second-write updates. Its largest effective slope pole is approximately 615.27/ms, corresponding to a 1.6253 microsecond relaxation time. These are distinct gain and numerical/discrete update resources; a smaller numerical integration step alone is not evidence that a biological implementation achieves that bandwidth. The units of gamma include inverse squared key units and should not be equated numerically to the native membrane leak rate.

## Results and continuous-reference scope

On fresh original histories, A100 mean Ecomp is 0.163840971 and 457/512 histories meet strict 0.5 joint operand tolerance. FAST10 mean Ecomp is 0.000812449 and all 512 meet that tolerance. All 25 original fixtures pass in FAST10, with maximum error 0.249794829. A100 fixture maximum is 5.411106902. These are outcomes of the new model and do not replace original P1B labels or its complete gate criteria.

The A100 large-gap branch is close to the oscillatory Euler stability boundary. On maximum-gap fixtures, its answer differs from the continuous second-stage reference by up to 5.4111. That large-gap failure is specific to this discrete approximation and must not be described as the same failure of the continuous flow. The minimum-gap continuous residual is still approximately 5.4114, so the finite-time slow-convergence limitation remains independently meaningful. On the fresh sample the maximum A100 Euler/reference prediction difference is 0.013423; FAST10's is 0.0001321.

**The saved `continuous_reference_predictions` validates only the second slope-writing stage while granting the captured Euler anchor. It is not a full continuous K,V,S simulation.** The first-stage gain is calibrated to exactly 100 Euler updates:

`G_Euler = 1/(1-0.99^100) = 1.5773675301`.

A continuous tau=10 ms charging stage would instead require

`G_continuous = 1/(1-exp(-1)) = 1.5819767069`

for exact endpoint capture. Keeping the implemented Euler gain unchanged in that continuous capture model gives relative undercharge `lambda = 1-G_Euler*(1-exp(-1)) = 0.0029135554`. The theory's converged first-query support-closure formula then reaches `14.85*lambda/(0.05-2.5*lambda) = 1.0128800643`. This is an analytic calibration-sensitivity example, not another empirical arm or an estimate of its prevalence. It does not invalidate the discrete result; it prevents treating that result and the partial continuous reference as full physical circuit validation.

The preregistered uncalibrated-capture diagnostic also independently reproduces error 2.0. Its effective gap is exactly zero in the ideal algebra and can be ULP-small in the floating recurrence. Calibration, exact initial reset, exact event duration and perfect holding are substantive assumptions; nominal success does not establish robustness to their errors.

## Native interface control and conclusion

The two native-control histories have exactly equal ordered content/novelty currents, event timing, encoding traces and future query inputs/results. Their queried target vectors differ by 1.8 in maximum norm. A common response therefore has minimax error at least 0.9 on this pair. This is a direct finite-tolerance input-information bound, not an exact-flow argument or a population risk estimate. The enriched candidate can distinguish these histories because it receives separate raw k and v ports.

The primary literature report correctly frames the residual-times-key update as an established learning mechanism and the anchored scalar specialization as an enriched proposal. This reviewer inspected that report and the local equations; independent external-paper retrieval was performed by the literature reviewer, not repeated here.

The study establishes one useful mechanism/resource tradeoff: a declared division-free iterative writer can meet the finite tolerance in a mathematical discrete implementation with additional raw inputs, calibrated capture, perfect gated holding, scratch storage, high feedback bandwidth and enriched readout. Its fixed 100-update counterpart cannot provide the uniform guarantee. It does not establish a faithful TAN implementation, optimal circuit, equal-energy comparison, universal neural obstruction or N=3 solution.

 The physical timing, calibration, retention, arithmetic and readout mechanisms remain scientific implementation gaps, not silently solved components.
