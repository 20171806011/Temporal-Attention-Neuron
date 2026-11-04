# Executed nonlinear held-out decoding on a fixed bounded domain

2026-11-02.4096 new targets, exact equation-derived decoder, no fitting. Fixed keys(.8,1.8),slots(0,4),content-gateη=.1; domainD=[-.5,.5]². This is **conditional decoding with known keys/timing**, not original variable-key TAN retrieval or an architectural success claim.

## Main result

The exact bounded inverse eliminates the meaningless off-domain tangent estimates in noiseless data. Noise still limits useful decoding. Under sensor-only noise the later sample improves aggregate composition loss but **does not improve the first-role RMSE over zero**. Membrane-process and combined-noise cases show little/no composition improvement over zero once all abstentions are counted.

All numbers below include4096histories and declared fallback(0,0) whenever no unique admissible root exists. Local jointsuccess requires valid decode and both errors<.05; this is a **new local diagnostic**, not P1-B's criterion.

| Scenario | m | Coverage | RMSE role1 | RMSE role2 | Mean E_comp | Local jointsuccess |
|---|---:|---:|---:|---:|---:|---:|
| Always zero baseline | — | 100% | 0.287739 | 0.290334 | 0.334215 | 1.050% |
| noiseless | 1 | 100.000% | 1.42728e-10 | 3.19999e-13 | 1.18419e-10 | 100.000% |
| noiseless | 100 | 100.000% | 1.04068e-10 | 2.31697e-13 | 9.09007e-11 | 100.000% |
| sensor_only | 1 | 1.831% | 0.290367 | 0.287342 | 0.333828 | 0.171% |
| sensor_only | 100 | 69.214% | 0.302198 | 0.161717 | 0.268165 | 10.352% |
| membrane_process_only | 1 | 2.368% | 0.291818 | 0.286903 | 0.334568 | 0.220% |
| membrane_process_only | 100 | 17.041% | 0.31172 | 0.264928 | 0.334076 | 1.855% |
| sensor_and_membrane | 1 | 1.440% | 0.290083 | 0.288289 | 0.334667 | 0.122% |
| sensor_and_membrane | 100 | 16.309% | 0.308966 | 0.265197 | 0.331982 | 1.611% |
| gate_process_only | 1 | 100.000% | 1.42728e-10 | 3.19999e-13 | 1.18419e-10 | 100.000% |
| gate_process_only | 100 | 91.113% | 0.165394 | 0.0874633 | 0.117351 | 33.276% |

NoiseSD=1e-6 for each active component in its declared units. Both timing designs have identical100step current/probe duration and2samples; decision latency differs0.1ms versus10ms.

- Noiseless: all 8192 schedule/history inversions valid; first-role RMSE~1e-10. This is numerical reconstruction accuracy, not exact float-state invertibility proof.
- Sensor-only m100:69.21% coverage, Ecomp0.26817 versuszero0.33421. First-roleRMSE0.30220 is **worse** thanzero0.28774; role2 improves. Conditional-valid role2RMSE0.000643 would misleadingly hide unconditional0.161717 caused largely by fallback; both are saved, never substituted.
- Membrane-process m100 and combined m100: coverage~16–17%; Ecomp~0.332–0.334. Descriptive paired95%normal intervals for Ecomp difference versuszero include0. These are not established improvements.
- Gate-process m1 remains noiseless because old-A update ordering prevents the first gate innovation from reaching the first membrane sample. This timing artifact is not generally better memory. Atm100 gate innovations accumulate.
- Original tolerance0.5 would make zero succeed on every interior target in this narrow domain; it is deliberately **not** used as success evidence.

## Global uniqueness, not merely local rank

For the ideal decimal-constant subthreshold Euler equations, endpointA is affine and endpointu quadratic:

`A=d+f1*x+f2*y`;
`u=b+c1*x+c2*y+q11*x²+q12*x*y+q22*y²`.

Coefficients are regenerated from exact Fraction recurrence with oldA, not fitted to targets. The affine determinant's four corner values are[-4.110453430218917e-06, -3.992463766317291e-06, -4.003973246249194e-06, -3.885983582347569e-06]. All are strictly negative; f2>0. Every constant-A fiber intersects convexD in a connected interval, and u varies strictly monotonically along that fiber since derivative=detJ/f2. Thus the mathematical map is globally injective **onD**. A nonzero determinant alone would not establish global injectivity; the connected-fiber argument does.

Both event contents remain positive onD, and gate/input recurrence is coordinatewise monotone. Upper corner(.5,.5) has maximal pre-reset candidate 0.338425325 and maximal gate0.366605287, strictly below thresholds1/5. This establishes the polynomial branch acrossD; it is not extrapolated to[-3,3]. Finite precision is separately checked, not conflated with the ideal algebraic proof.

Actual vectorized Euler endpoints for all 4096 held-out targets match polynomial endpoints within4.17e-16. Vector stepping matches frozen scalar euler_step exactly on first 8 held-out targets and 4 corners. Actual spike returns are checked in scalar validation; no reset was hidden by checking only post-resetu. All stochastic query paths remained in the valid branch.

## Frozen solver and abstention accounting

Given two observations, invert known O_m to estimate endpointstate. Substitute y=(Ahat-d)/f2-(f1/f2)x into the quadraticu equation. Solve using stable quadratic roots (linear fallback separately tested). Negative discriminant is NO_REAL_ROOT without clipping. Both candidate coordinates must lie inD; only1e-10boundary roundoff can snap, counted separately; forward residual must be≤1e-10state units. Exactly1valid root permitsdecode; all other outcomes fallback0. Roots are never selected using truth.

| Scenario | m | Valid | No real root | Out of domain | Ambiguous/residualfailure | Boundary snaps |
|---|---:|---:|---:|---:|---:|---:|
| noiseless | 1 | 4096 | 0 | 0 | 0 | 0 |
| noiseless | 100 | 4096 | 0 | 0 | 0 | 0 |
| sensor_only | 1 | 75 | 798 | 3223 | 0 | 0 |
| sensor_only | 100 | 2835 | 0 | 1261 | 0 | 0 |
| membrane_process_only | 1 | 97 | 502 | 3497 | 0 | 0 |
| membrane_process_only | 100 | 698 | 0 | 3398 | 0 | 0 |
| sensor_and_membrane | 1 | 59 | 971 | 3066 | 0 | 0 |
| sensor_and_membrane | 100 | 668 | 0 | 3428 | 0 | 0 |
| gate_process_only | 1 | 4096 | 0 | 0 | 0 | 0 |
| gate_process_only | 100 | 3732 | 0 | 364 | 0 | 0 |

All physical branch departure counts0. Boundary sanity checks are separate from heldout results:8 of 9 grid points needed only the prescribed roundoff-band snap; **no heldout prediction was snapped**. Noisy no-real/out-of-domain cases do not indicate noiseless global folding. Every per-history CSV records estimated state, target, prediction,status, bothrealcandidatevaluepairs and acceptedresidual for audit.

## Sampling and diagnostics

Targets use RandomState20261103; noises use20261104 in preregistered draw order. Physical innovations are afterdrift and recurrent; schedules share each complete100step path. ε0shared; ε1/ε100independent. No initial physical uncertainty was substituted for sensornoise. Noisecomponents have different units and no equal-energy interpretation.

`paired_uncertainty.json` provides **post-execution descriptive**, unadjusted approximate95%normal intervals over all histories for every cell's Ecomp difference fromzero. Sensor-onlym100 difference−0.06605, interval[−0.07151,−0.06059]; membrane-only m100−0.000139, interval[−0.003194,0.002916]; combined m100−0.002233, interval[−0.005231,0.000766]. These intervals were not a preregistered hypothesis test and do not correct for multiple comparisons.

The fixed local-linear inverse is retained in JSON as an **unbounded sensitivity diagnostic**, not behavioralprediction: its huge values outside D show failure of linearization/conditioning, not actual bounded decoder losses. For example sensor-only m100 has linear-diagnostic first-role RMSE~5.03 versusbounded exact0.3022. This reinforces why prior tangentRMSE22.5 and0.385 were not valid nonlinear-retrieval performance estimates.

## Reproduction and scope

```sh
python nonlinear-heldout/run_heldout.py
python nonlinear-heldout/summarize_uncertainty.py
```

Source frozen SHA2566b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f checked before import; runtime versions/targetCSVhash/prereg SHA saved in results. Protocol was written before execution. No training, set selection,η/domain/noise tuning or broad clipping. Runtime~2.8seconds. Repeat result/target/prediction hashes match. Exact source and production runner were not modified; runner not executed. No installs, public writes or spending.

Artifacts: protocol, script,resultsJSON,targetCSV,10predictionCSVs,paireduncertaintyJSON,report,SHA256SUMS. Noise-free polynomialinverse is verified on this bounded known-key domain; robustness depends strongly on noise/censoring/fallback. This does not establish unknown-key addressing, full P1-B success, or a need for additional neurons. Fixed question complete; no automatic domain expansion follows.
