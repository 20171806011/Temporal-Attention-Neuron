# Executed same-state content-gate rank and observability ablation

2026-11-02. Prespecified bounded experiment, no training/parameter search. Fixed keys(.8,1.8), slots(0,4), same3 state coordinates(u,A,r), η=.1,27 grid encodings plus derivative/local-diagnostic checks. Completed with all prespecified outcomes retained.

## Findings

- Changing A's active-event drive from key-only novelty to `novelty+η(v+3)` makes the two-value endpoint Jacobian **rank2 locally at v=(0,0)**. A content-blind matched-offset control `novelty+3η` remains rank1.
- Offset and content arms have **bitwise-identical complete baseline trajectories**, mean gate and input proxy. Their rank difference therefore is not merely baseline current gain.
- The unchanged actual query is still **A-blind**: even with rank2 encoding its observable derivative has rank at most1. At fixed u, changing A among0,.1,1,5 gives identical actual cue result dictionaries.
- A separately added query current0.1 and **two analog time samples** makes the observable map locally rank2, but very poorly conditioned. This is an added readout coupling/resource, not original TAN retrieval or a free extra encoder state.
- Noise and local-linearization diagnostics expose weaknesses; no retrieval success, general key binding or robust capacity is claimed.

## Encoding Jacobians

J=∂(u_T,A_T)/∂(v1,v2), raw model units:

**original**
```text
J = [[0.0, 0.02858676872060995], [0.0, 0.0]]
singular values = [0.02858676872060995, 0.0]
determinant = 0.0
```
**offset**
```text
J = [[6.908472123199264e-06, 0.05024437174996322], [0.0, 0.0]]
singular values = [0.05024437222491181, 0.0]
determinant = 0.0
```
**content**
```text
J = [[0.00013926662469358346, 0.06310841767423776], [0.00011849111241245314, 0.024984938883834358]]
singular values = [0.06787451302064998, 5.8906035982411584e-05]
determinant = -3.998218506283065e-06
```

Content raw condition number≈1152.25. Offset/original analytical second rows are exactly0, hence rank1; the ~1e-22 second singular value arising for the composed offset probe is floating SVD roundoff, **not** rank2. Content determinant≈-3.99822e-6 is nonzero. Full column rank provides only local inversion with known fixed keys/timing; the result says nothing about variable-key binding across the original task.

Normalized coordinates also reported: input v/3, output(u/1,A/5), using target half-range, membrane threshold and gate bound as declared scales. These conventional scales are not a measured noise model; normalization changes conditioning and cannot establish physical capacity.

| Arm | Dimensionless state singular values | Dimensionless state condition |
|---|---|---:|
| original | [0.08576030616182985, 0.0] | None |
| offset | [0.15073311667473543, 0.0] | None |
| content | [0.18991829450820252, 3.789415511520773e-05] | 5011.809708668879 |

## Verification and branch margins

Analytic tangent uses **old A**, matching production Euler: `Du_next=.99 Du+.01*(I_content DA+A indicator)`; `DA_next=(149/150)DA+(.1/15)*η indicator` for content, without the η term for controls. Central differences use prespecified1e-4 and1e-5; maximum Jacobian residual<7.6e-12. Actual affine query derivatives also agree with `g_dec*.99^100*J_u` within1e-9 (full residuals in JSON).

All27 fixed grid encodings v∈{-.5,0,.5}² have zero spikes. Worst threshold margin>0.6615; maximal A<0.3667, far below clip5. Original first-event A=0 is a key-only lower-bound state, but value changes do not affect A there; the derivative is exactly0. Content/offset gate is strictly positive after the first step. Original custom stepping matches frozen simulate_encoding endpoints bitwise at all9 grid points.

Atbase, original meanA≈0.025109 versus0.073888 for both offset/content. Integrated absolute synaptic-input proxy rises1.93254→4.04397. This proxy is **not physical energy**, and unchanged coordinate count does not mean unchanged physiology/current cost.

## Readout resource and noise results

Original query uses I_content=0; its raw readout is `.99^100*u_T+known_constant`, so A remains unobservable. Modified diagnostic reads u0 and u1 after one step with I_content=.1, q=.8:

`y = [[1,0],[.99,.001]] [u_T,A_T] + [0,.008]`.

Observation determinant=.001; for content encoding the composed singular values are approximately[.08882145,4.50141e-8], condition≈1.97319e6. One additional analog time sample and new query-time current coupling are explicit resources. No persistent hidden state was added to the encoder. Analytical tangents stored by the experiment are diagnostic calculations, not neuron state.

Local inverse diagnostics use actual perturbationδv=(.001,-.001), base-J pseudoinverse, and2000 Gaussian draws/level, seed20261102. This is an external local full-state/measurement observer, **not** an implemented or trained retrieval decoder. Values are not clipped. Coordinate-wise independent additive endpoint noise and voltage-sample noise are different, uncalibrated noise/resource models; their same numericσ is not a matched physical comparison. JSON includes covariance propagation and singular values whitened by each declaredσ.

Content-arm first-value RMSE:

| Measurement | σ=0 | σ=1e-8 | σ=1e-6 | σ=1e-4 |
|---|---:|---:|---:|---:|
| state | 4.4221087e-05 | 0.00017542631 | 0.017031259 | 1.7031751 |
| original_raw_readout | 0.0010022017 | 0.0010022017 | 0.0010022018 | 0.0010022688 |
| modified_probe | 4.4221079e-05 | 0.22309747 | 22.30955 | 2230.9549 |

The rank2 state inverse already has~4.42e-5 first-value bias at zero noise for this smallδv, from local linearization amplified by conditioning. Original readout loses the first-value perturbation in its nullspace. Two-sample inversion magnifies independent sample noise severely, reflecting subtraction of almost equal samples. Large RMSE outcomes are preserved rather than tuned away; this does not prove every readout scheme must fail.

## Provenance, reproducibility and limits

Source models_p1b_v2.py SHA2566b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f at commit85213f0d417d54e4f48b301ed1c0e4200ef3c421; reviewed before import and checked. Modified encoding uses frozen euler_step with only the explicitly changed gate drive; no source edits. Saved calibration unchanged. Production runner not imported. Python/NumPy/SciPy versions in JSON; no installs, random model training, public writes or spending.

`PROTOCOL.md` was written before first execution; its SHA256 is stored in results. Tool chronology places design communication at18:16:56UTC and first completed results at18:18:17UTC. This is a local recorded prespecification, not an externally registered protocol. Dimensionless reporting was added afterward for unit transparency without changing experiment design or outcomes.

```sh
python content-gate-rank/run_rank.py
```

`results.json` includes all27 branch checks, Jacobians, finite differences, spectra, actual readouts, probe checks and36 noise-condition summaries. Three CSVs store complete baseline traces. Repeat run reproduces final results/traces exactly; SHA256SUMS fingerprints artifacts. Runtime~1.4seconds.

Stop condition reached: local encoding-rank escape verified, unchanged readout obstruction retained, added readout severely ill-conditioned under declared noise. This is not evidence that more neurons are necessary, nor that modifying A alone solves TAN retrieval.
