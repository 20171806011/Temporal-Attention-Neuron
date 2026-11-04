# Executed two-sample timing comparison: same physics, later observation

2026-11-02. Fixed comparison m1 versus m100; no optimization sweep. Both schedules apply queryq=.8 and added contentIprobe=.1 for the same100steps/10ms and take exactly2analog samples, sharing identical physical trajectories. Second-sample availability is0.1ms versus10ms: **longer decision latency is a cost**. No extra probe duration, amplitude or sample count is given to the later schedule.

## Result

| Second sample | d_m | Observation singular values | Composed encoder-observer singular values | Composed condition |
|---|---:|---|---|---:|
| 1 | 0.001 | [1.4071604368284119, 0.000710651020187785] | [0.08882145364930398, 4.501410798869301e-08] | 1.97319e+06 |
| 100 | 0.04387200979 | [1.0649917489080862, 0.04119469454978729] | [0.0675879769801868, 2.5952823165408837e-06] | 26042.6 |

The later sample materially improves numerical conditioning but the composed map remains ill-conditioned (~26,043). Sensor-noise amplification for A0 drops1407.16→24.27 (~58×). This is not a capacity or successful TAN retrieval result.

First-value local-diagnostic RMSE at prespecified noiseSD1e-6 in each scenario's stated units (4096paired paths):

| Scenario | Sample1 | Sample100 |
|---|---:|---:|
| sensor_only | 22.5009593 | 0.385054537 |
| membrane_process_only | 15.8147929 | 2.41788224 |
| sensor_and_membrane | 27.2762111 | 2.44449407 |
| gate_process_only | 4.42211997e-05 | 0.113206277 |

**Countervailing result preserved:** gate-process noise is invisible to u1 because the first gate innovation occurs after that step's old-A-driven membrane update; it accumulates by u100, worsening that scenario. Later sampling is therefore not universally better across noise models. Large remaining errors are reported without clipping predictions or tuning parameters. **RMSE22.5 exceeds the entire bounded target domain[-3,3]: it is an unregularized tangent-inverse sensitivity diagnostic, not behavior/performance of a valid bounded retrieval system. Even RMSE0.385 may lie outside the locally validated inverse radius and must not be interpreted as established nonlinear decoding accuracy.** Gate-process invisibility at m1 is specifically an old-A/update-order timing artifact, not evidence of better memory: the newly injected gate noise has not yet reached the membrane sample.

## Equations and covariance verification

The query uses old A, as frozen Euler requires. With a=.99,b=149/150,k=.001:

`d_m=k*(b^m-a^m)/(b-a)`;
`[u0,u_m] = O_m*[u0,A0] + [0,.8*(1-a^m)]`;
`O_m=[[1,0],[a^m,d_m]]`.

Exact noiseless predictions agree with actual frozen euler_step trajectories within4.45e-16 across the previously fixed9-value grid. At all endpoints A0<0.18781<2 and u0<1; query candidate-voltage threshold margin>0.4392. Actual returned spikes and refractory values are checked, not inferred from post-reset voltages.

The four noise scenarios were prespecified: sensor only, additive membrane innovation only, both, and additive gate innovation only; each activeSD=1e-6. Physical innovations enter updatedu/A **after** each deterministic drift and recur through later steps. Prequery physical state is fixed. The same complete100step physical path is observed by both schedules; ε0 is shared and ε1/ε100 are independent. No restart after sample1 and no independent-noisy-snapshot substitution.

For independent innovations:

`Var(Ahat-A0)=[σs²(1+a^(2m))+σu²Σa^(2j)+σA²Σd_j²]/d_m²`;
`Cov(uhat-u0,Ahat-A0)=-a^m*σs²/d_m`;
`Var(uhat-u0)=σs²`.

All positive-variance empirical estimates are within1.72 MonteCarlo standard errors of their exact predictions. Full covariance matrices and propagation through the frozen local J inverse are in JSON. Zero predicted variance is reported without inventing a z-score. Covariance agreement checks numerical implementation, not a biological noise assumption.

All4096paths per scenario remain below threshold and within gate bounds throughout100steps; both pre-innovation candidate and post-innovation state thresholds are monitored. Gaussian noise is unbounded, so this observed safety is not a global guarantee. No clipping/reset branch was silently applied to preserve the analytic formula.

## Diagnostic interpretation and resources

The content-dependent A encoder, keys(.8,1.8),slots(0,4),η=.1 and frozen local Jacobian are unchanged from the prior experiment. The actual test perturbation isδv=(.001,-.001). A basepoint linear inverse is an **external diagnostic**, not a trained or implemented original retrieval mechanism. Noise-free first-value local-linearization bias remains about-4.4221e-5 and is separated from noise covariance.

Sensor voltage units, membrane-state innovations per step and gate-state innovations per step are distinct noise models; identical numericSD does not mean equal physical energy. Raw and dimensionless composed sensitivities are saved; no bit-capacity statement follows. Endpoint gate state and observation-matrix inversion do not add an encoder coordinate, but the query current and two analog samples are added readout resources relative to original TAN. Original query remains A-blind.

## Provenance and reproduction

`PROTOCOL.md` written before execution; hash in results. Model source85213f0d417d54e4f48b301ed1c0e4200ef3c421/SHA2566b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f checked before import. Prior rank-results hash and runtime versions recorded. Deterministic trajectories use frozen euler_step; vectorized stochastic recurrence preserves oldA and update order, adding only documented after-drift innovations. Production runner not imported; no source modifications, installs, training, public writes or spending.

```sh
python two-sample-timing/run_timing.py
```

Artifacts: protocol, script, resultsJSON with9grid checks and8paired schedule/scenario summaries, full deterministic100step perturbed query trace, report and SHA256SUMS. Seed20261102,4096paths. Repeat execution reproduces JSON/trace hashes. Runtime~0.8seconds.

Bounded question resolved: later timing helps specified sensor/membrane noise and hurts the gate-noise control, with unchanged physics but later decision time. No broader retrieval claim or automatic timing search follows.
