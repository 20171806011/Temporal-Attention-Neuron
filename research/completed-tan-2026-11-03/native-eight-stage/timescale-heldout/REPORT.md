# Final bounded timescale experiment: held-out results

2026-11-02. Three predeclared arms, 4096 fresh paired histories, no fitting or parameter search. All use the proposed content-dependent A coupling and known fixed keys/timing. This is **not a rerun or success claim for original TAN/P1-B**.

## Main finding

Selectively lengthening the gate time constant improves this conditional two-value task substantially. Uniform slowing is less effective, especially with membrane-process noise. The intervention changes retention **and finite-window charging**, not state count alone; it does not isolate decay causally from input gain.

All arms share targets, raw per-step noise realizations, 100 ms encoding, 10 ms query, added query content 0.1, query 0.8, two analog samples at 0/10 ms, exact bounded nonlinear inverse, and abstention-to-zero policy. Changed constants:

| Arm | tau_m ms | tau_A ms | Composed OJ condition | Retained gate coefficient f1/f2 |
|---|---:|---:|---:|---:|
| baseline | 10 | 15 | 26042.6299 | 0.004743 |
| gate_only | 10 | 150 | 78.1985 | 0.586542 |
| uniform | 100 | 150 | 464.1902 | 0.586542 |

## Full-denominator held-out performance

Each cell includes all 4096 histories, including zero fallbacks. Local joint success requires a valid inverse and both errors <0.05. This local threshold is **not** the original P1-B criterion; original tolerance 0.5 would make zero trivial on this narrow box.

| Arm | Noise | Coverage | RMSE role 1 | RMSE role 2 | Mean E_comp | Local joint success |
|---|---|---:|---:|---:|---:|---:|
| Zero baseline | — | 100% | 0.288717 | 0.285625 | 0.332726 | 1.025% |
| baseline | noiseless | 100.000% | 1.02811e-10 | 2.28749e-13 | 9.00258e-11 | 100.000% |
| baseline | sensor_only | 69.116% | 0.303696 | 0.158411 | 0.270803 | 10.254% |
| baseline | membrane_process_only | 16.040% | 0.310101 | 0.262203 | 0.332441 | 1.636% |
| baseline | sensor_and_membrane | 16.138% | 0.310767 | 0.26174 | 0.333122 | 1.514% |
| baseline | gate_process_only | 90.649% | 0.167793 | 0.0842051 | 0.119362 | 32.495% |
| gate_only | noiseless | 100.000% | 1.96698e-12 | 3.10953e-13 | 1.82076e-12 | 100.000% |
| gate_only | sensor_only | 99.463% | 0.0348394 | 0.0215305 | 0.00809317 | 99.463% |
| gate_only | membrane_process_only | 96.582% | 0.0931262 | 0.0600277 | 0.0481046 | 74.707% |
| gate_only | sensor_and_membrane | 96.362% | 0.0956853 | 0.0613661 | 0.0495144 | 73.877% |
| gate_only | gate_process_only | 99.854% | 0.01688 | 0.0135498 | 0.00276519 | 99.854% |
| uniform | noiseless | 100.000% | 6.49716e-12 | 2.03575e-12 | 6.27675e-12 | 100.000% |
| uniform | sensor_only | 91.284% | 0.148905 | 0.102598 | 0.102618 | 40.137% |
| uniform | membrane_process_only | 48.218% | 0.321953 | 0.229339 | 0.312925 | 5.420% |
| uniform | sensor_and_membrane | 47.412% | 0.322913 | 0.232383 | 0.314282 | 5.908% |
| uniform | gate_process_only | 99.731% | 0.0232768 | 0.0185326 | 0.00417709 | 99.731% |

Each active noise component has SD 1e-6 in its declared sensor-voltage or state-per-update units. These are not physically equal-energy noise models across time constants. All noiseless inverses succeed with errors near floating-point precision. Every rejected noisy observation is out of domain; there are no ambiguous roots, residual failures, boundary snaps, or physical branch departures in the held-out runs. Exact rejection counts and conditional-valid errors are retained in JSON and all 15 prediction ledgers; conditional metrics are never substituted for full-denominator results.

Primary sensor-only paired history-bootstrap comparisons (2000 replicates, fixed seed, percentile 95% intervals):

| Comparison | Mean E_comp difference | 95% interval |
|---|---:|---|
| baseline_minus_zero | -0.061923 | [-0.067654, -0.056496] |
| gate_only_minus_zero | -0.324633 | [-0.328407, -0.321189] |
| uniform_minus_zero | -0.230109 | [-0.234950, -0.225884] |
| gate_only_minus_baseline | -0.262710 | [-0.267771, -0.257262] |
| uniform_minus_baseline | -0.168186 | [-0.173066, -0.163330] |

No arm/noise/domain selection followed these intervals, and no multiplicity-adjusted general claim is made. The fixed comparison supports a remediable timescale/charging bottleneck for this modified, known-key task—not general learned content addressing.

## Analytic and implementation checks

Each arm regenerates its endpoint affine-A/quadratic-u coefficients from exact Fraction arithmetic. All four determinant corners are strictly negative. Together with connected constant-A fibers in the convex domain [-0.5,0.5]², this proves global uniqueness for the ideal subthreshold map. Nonzero rank alone is not used as a global-invertibility argument.

Whole-box branch validity follows coordinatewise monotonicity: event contents and gate drives are nonnegative; both Euler decay coefficients are positive; the upper corner (0.5,0.5) bounds every encoding trajectory. The maximum-corner candidate/gate pairs are approximately baseline (0.33843,0.36661), gate-only (0.06099,0.06163), uniform (0.00903,0.06163), safely below reset/clipping thresholds.

All 4096 deterministic endpoints per arm agree with independently propagated polynomials to <3.7e-16. Scalar frozen euler_step checks on the first eight targets and four corners match vector encoding exactly, and explicit scalar query checks agree with O100 within 1e-13. Actual returned spikes, refractory values and pre-reset candidates are checked; no reset is hidden by post-reset state reporting. Old A enters every membrane update. Both leak and forcing coefficients change with h/tau, as preregistered.

Stable quadratic inversion retains the prior fixed tolerances: 1e-10 boundary-roundoff band and state residual. No broad clipping, projection or truth-based root choice. Mathematical ideal-map injectivity and observed floating-point reconstruction are distinguished.

## Noise geometry, not observation matrix alone

`noise_geometry.json` evaluates the full composite Sigma^(-1/2) O J Dv at the basepoint, with Dv=0.5 I from the actual domain half-width. It includes right singular vectors, so weak value directions are explicit. This is post-execution analytic reporting, not another simulation or tuning step.

Sensor-only weakest whitened singular values are approximately 1.30, 72.51 and 5.37 for baseline, gate-only and uniform. With sensor plus membrane-process noise they are 0.208, 11.70 and 0.759. The baseline weak direction is almost entirely the first value; gate-only substantially improves that direction. Pure process-noise covariance is singular because the initial sample is exact: it is treated as a noiseless constraint plus a noisy remaining direction, never silently regularized. These local noise geometries explain sensitivity but do not replace the bounded nonlinear held-out risks above.

## Resource accounting

Mean deterministic proxy integrals across held-out histories:

| Arm | Encoding ∫A dt | Encoding ∫abs(Isyn) dt | Encoding ∫u² dt | Query ∫A dt | Query ∫abs(Isyn) dt | Query ∫u² dt |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 7.368324 | 4.072810 | 0.447422 | 1.280802 | 0.128080 | 1.437546 |
| gate_only | 2.070171 | 0.698586 | 0.013990 | 0.511221 | 0.051122 | 1.136172 |
| uniform | 2.070171 | 0.698586 | 0.000507 | 0.511221 | 0.051122 | 0.024211 |

These are model-unit proxies, **not ATP or measured physical energy**. Slower A charges less during brief events while retaining the old event better. Equal external probe/current duration does not imply equal internal gated currents. Longer intrinsic relaxation is a changed resource despite the same three coordinates and external deadline. Reset/carryover energy, throughput and between-trial memory are not simulated and cannot be claimed free. Offset/content equality in the earlier test held only at its basepoint, not over perturbed histories.

## Reproduce and stop

```sh
python timescale-heldout/run_timescales.py
python timescale-heldout/analyze_noise_geometry.py
```

Targets seed 20261105; noise 20261106; bootstrap 20261107. Target ledger/hash was saved before metrics. Protocol hash, frozen model hash, environment versions, coefficients, branch checks, full metrics, resource proxies and root/status ledgers are saved. Source hash 6b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f at commit 85213f0d417d54e4f48b301ed1c0e4200ef3c421. No source edits, installs, production-runner execution, public writes or spending. Runtime about five seconds; repeat hashes match.

This final bounded comparison closes the conditional series. No further parameter sweep was launched. The original model's fixed decoder ceilings and rank/readout defects remain; the successful conditional arm required content-dependent gating, a longer gate timescale, added query coupling, two analog samples, and a known-equation decoder. Unknown/random keys and timings, N3, noisy encoding, finite precision, biological implementation, and resource-budgeted general retrieval remain untested.
