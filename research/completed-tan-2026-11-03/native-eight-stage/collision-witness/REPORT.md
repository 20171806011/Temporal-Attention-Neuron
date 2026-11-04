# Executed production-default collision witnesses

2026-11-02. Four controlled encodings with frozen model code and saved full-precision calibration; no fit, randomness, training, parameter sweep or population inference. Existing conceptual collision fixtures are adapted to default weights/timing, not presented as a novel general theorem.

## Result

| Model | First/second keys | Compensation R | Float final delta-u | Float residual ULPs | Max exact encoding synaptic current | Encoding spikes | Cue spikes per member |
|---|---|---:|---:|---:|---:|---:|---:|
| M2b | 0.8,1.8 | 0.000322222362880236 | 3.12250226e-17 | 9 | 0.111416048 | 0 | 0 |
| M1 | 1.8,2.2 | 0.000160298006578493 | 2.77555756e-17 | 1 | 0.854461017 | 0 | 1 |

For each model, slots0,4 contain values(-1,+R) versus(+1,-R). Their first target values differ by2; all values and keys lie within task ranges. M1's keys both produce positive novelty, avoiding the low-key zero-novelty erasure mechanism. M1 still starts A=0 at the very first time step, but develops a positive gate during the first event.

**Mathematical result:** exact rational arithmetic verifies equal final carrier states under the ideal decimal-constant Euler recurrence. **Finite simulation:** final states are not bitwise equal; they differ by the tiny residuals above. Do not turn a numerical tolerance into an exact equality claim.

Actual frozen readouts are identical between the two members in both cases: M2b prediction0.041705547807934666; M1 prediction-0.012845719483768992. Both predictions fail strict0.5 tolerance for **both** target signs. No approximate equality was used to obtain this output equality. The current script reports delta-prediction0 exactly; this equality of implemented readout outputs does not imply equality for every conceivable decoder of the distinct float states.

## Analytic construction

With no spikes and fixed keys, A_n depends only on keys, not values, and r=0. Define n=0..999 before-step indices; event1 is n0..99 and event2 n800..899. The production membrane recurrence is

`u_(n+1) = (99/100)u_n + (1/100) A_n (key_n+value_n)`.

At encoding completion:

`u_1000 = b + c1*v1 + c2*v2`,

`c_i = (1/100) sum_(n in event_i) A_n*(99/100)^(999-n)`.

For M2b, A_n=A0 and c1/c2=(99/100)^800. For M1, the exact key-only gate recurrence is A_(n+1)=(149/150)A_n+novelty_n/150; gate clipping never activates. Both coefficients are positive. Setting R=c1/c2 yields

`u_minus=b-c1+c2*R=b`, `u_plus=b+c1-c2*R=b`.

Same key sequences give identical A; r=0 by the verified subthreshold condition. Thus ideal full endpoint states match exactly. The two first targets differ by2, so any deterministic decoder of that identical mathematical state and identical query has absolute error at least1 on at least one member. This is a witness of non-injectivity for these fixed model/regime maps, not a theorem against general three-state systems.

The code computes all coefficients and R as `fractions.Fraction`, checks cancellation by exact equality, and saves R as exact hexadecimal numerator/denominator. Subthreshold validity is not inferred from near-zero residuals: exact current maxima for both value signs are <1, and u0=0 with stable convex update implies u_n<1. Both actual encoding trajectories also have zero spikes. Exact rational arithmetic idealizes decimal constants; float64 model evaluation and conversion of R introduce rounding, fully exposed above.

## Query behavior

Actual first-key queries are used: q0.8 for M2b and q1.8 for M1, zero additional delay. M2b cue remains subthreshold. M1 cues each spike at step74, followed by refractory evolution; both have one eligible spike. Encoding-subthreshold, not cue-subthreshold, is required by the collision construction. Once mathematical final states coincide, any identical deterministic cue dynamics preserve indistinguishability, including spike/reset branches.

## Limits and provenance

- Four fixed synthetic histories have zero probability as exact points under continuous sampling; neighborhoods do not automatically inherit the exact impossibility conclusion for discontinuous/high-precision decoders. No prevalence claim is made.
- Floating endpoints differ by1–9ULP; an arbitrary highly sensitive/discontinuous float-state decoder could distinguish them. Do not claim the numerical comparison alone rules that out.
- Conceptual collision argument already exists in repository G1.5 with changed weights. This is a default-parameter witness extension, not a newly discovered principle or universal impossibility theorem.
- Model commit85213f0d417d54e4f48b301ed1c0e4200ef3c421, SHA2566b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f. Entire source safety-reviewed in preceding fixture; hash checked before import. Production runner not imported or executed.
- Saved calibration remains A0=0.06188672587424413, g=0.1326308169898006, b=-0.02681648355488859. Python/NumPy/SciPy versions recorded in JSON; no dependencies installed.
- New local script only, no repo modifications, external publication or spending. Runtime~0.8seconds. Prespecified protocol saved before run.

## Reproduction

```sh
python collision-witness/run_witness.py
```

`results.json`: coefficients, exact-proof flags, numerical residues, phase results and readouts. Four trace CSVs: every encoding/cue step. Two ratio JSON files: exact rational compensation ratios. `SHA256SUMS`: artifact fingerprints. Repeat execution reproduces JSON, ratios and trace hashes exactly.
