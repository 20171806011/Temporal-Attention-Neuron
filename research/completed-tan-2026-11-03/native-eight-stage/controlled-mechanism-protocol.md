# Minimal controlled state-sensitivity test (proposed, not executed)

Pinned source:85213f0d417d54e4f48b301ed1c0e4200ef3c421.
Scope: distinguish an exact zero-input gate erasure from a nonzero but exponentially fading value sensitivity, and test whether retained A is causally visible during cue presentation. This does NOT establish successful content retrieval, biological plausibility, or necessity of extra dimensions.

## Source and safety review

Exact shared copies `models_p1b_v2.py` (SHA2566b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f) and `runner_p1b.py` (47609f1e70ab8de2966ee8305757f44658ef799918d5f92108ac4efe64304c26) match closure frozen hashes. Model imports math, numpy, dataclasses, typing, scipy.stats.norm; top level defines constants/classes/functions only, no I/O. Reviewed relevant functions euler_step, simulate_encoding, simulate_delay and simulate_cue_and_readout have no I/O. Runner includes output-writing and smoke tamper-test functions; **do not invoke runner** for this test. NumPy2.3.5 and SciPy1.17.0 are available. No repository code executed by this reviewer.

Stored JSON has aggregate M1/M2b performance and calibration A0 only, not state traces or gate distributions. M2b N2delay0 joint success0.0295 vs M1 .0290, but their frozen decoder calibration is inherited from M1; it is not evidence that M2b's information is optimally accessible.

## Primary exact pair

- W5, occupied positions0 and4.
- key array[0.8,0,0,0,1.8], mask[True,False,False,False,True].
- history minus values[-0.5,0,0,0,0]; history plus[+0.5,0,0,0,0].
- Model arms M1 and existing M2b with A0=.061886726; no parameters tuned.
- Encode100ms and probe q=.8 for10ms; optional registered silence delays0,5,10ms.
- Compare paired state vectors at encoding completion and paired output values after cue. Trace option can verify subthreshold branches, but no need to save1000-step traces if bounds checked analytically.

### Analytic pre-registered assertions

M1: first key.8<1.4 yields novelty0 and A0=0; first input value never enters membrane. Identical later inputs imply **identical state and readout** across pair. This is a fixture of known Proposition6.1, not novel theory.

M2b: positive fixed gate introduces value sensitivity. Every input current is subthreshold: first maximum A0*(.8+.5)<.081; second A0*1.8<.112. Initial u0; stable Euler coefficient.99 means no threshold reset or refractory branch. With deltaV=1, after first100 presentation steps deltaU=A0*(1-.99^100). After remaining900 steps:

`deltaU_enc=A0*(1-.99^100)*.99^900=4.627428839279313e-6`.

At q=.8, the10ms cue also remains subthreshold. Thus

`deltaU_readout=.99^100*deltaU_enc=1.693788612116668e-6`.

Frozen affine g=.132630817 yields prediction difference only2.246485674503298e-7. Added delay d multiplies all downstream paired differences by.99^(10*d).

This proves local first-value sensitivity under a constant gate, but not distinguishability under noise or fixed tolerance. Do not label any nonzero difference a retrieval success. The two target values differ by1, and strict tolerance<.5 makes their acceptable intervals disjoint; report actual role errors and margins as well.

## One small extension that adds mechanistic value

Repeat the pair with first event position p in{0,1,2,3}, second fixed at4 (total16 encodes across2 histories×2 models×4positions). No N or task-resource escalation. M1 always erases the first event. M2b expected final sensitivity is:

`deltaU_enc(p)=A0*(1-.99^100)*.99^(900-200*p)`.

Report ratio between neighboring positions=.99^-200 and compare with measured paired differences. This isolates event-age attenuation rather than mistaking restored input gain for multi-item addressing. The gate offset is not a new state dimension, but changes current/energy; report that difference. It also differs from adaptive gate retention; M2b already exists and needs no architecture invention.

## Independent cue-time A visibility control

Choose matched u=.1,r=0, query q=.8, and A in{0,.061886726,.4,5}; call simulate_cue_and_readout for M1. During cue `I_content=0`; euler_step's only gate term is `g_A*A*I_content`, so membrane/spikes/readout must be identical for all A even though A decays differently. This establishes a readout-path disconnection in this model, not lack of information in A during encoding. It is a different, sharper control than first-event erasure.

## Minimal executable interface (reviewed source subset)

Use only `simulate_encoding`, `simulate_delay`, `simulate_cue_and_readout`, `CarrierState` from the exact model copy, numpy to construct5-element arrays, and an independently written analytic formula. Do not import runner or execute main functions. Assert finite states, r0 for subthreshold examples, exact M1 pair equality, and M2b prediction agreement with analytic values to absolute tolerance1e-14 (relative1e-8 where appropriate). Use float64. Save JSON rows with pinned source hash, environment versions, arm,p,delay,paired states, measured/analytic differences, readout values and target errors. No training, random draws or full confirmatory run needed. These are deterministic mechanism fixtures, so no bootstrap or p-values.

Stop when the fixtures agree or an implementation/analytic discrepancy is diagnosed; do not escalate to broader simulation automatically.

## Proposed follow-on: production-default collision, not an unconstrained decoder sweep

Existing `test_p1b_g1_fixtures.py:563–622` already demonstrates a subthreshold collision, but changes weights to wK=wV=.1. A useful narrowly incremental next result is a witness at production defaults wK=wV=1, with positive gates, separating state loss from affine-decoder limitations.

**Closed-form M2b witness:** retain positions0,4 and keys.8,1.8. Let R=.99^800=.0003222223628802337. Choose history Hminus values(-1,+R) and Hplus(+1,-R). Both remain subthreshold. The final paired voltage difference is c1*2+c2*(-2R)=0 because c1/c2=R. A is identical fixedA0, refractory state0, so full final states match in exact arithmetic. The first target differs2. Any deterministic decoder using only this state and identical first-key cue must return the same output, so at least one history incurs absolute first-role error>=1. Small float rounding residual is numerical verification only, not itself a proof against arbitrarily discontinuous decoders; the exact linear recurrence supplies the mathematical collision.

**M1 no-zero-gate witness:** positions0,4, keys1.8 and2.2 (both novelty-positive), initialA0=0 with production gating dynamics. Compute A_n independently of values from the fixed key sequence. In a verified subthreshold trajectory, value sensitivities at the100ms endpoint are c_i=.01*sum_{n in event_i} A_n*.99^(999-n), where event1 n0..99, event2 n800..899. Set R=c1/c2 and use values(-1,+R) vs(+1,-R). Assert R finite and in range, all candidate voltages<1, no spikes and r0; then exact algebra produces the same collision without modifying weights. Trace maxA and input-current bounds so threshold crossing is excluded, rather than inferring it from small output difference. This is proposed, not yet executed.

These tests are stronger than finding no gain from a fitted nonlinear readout: a constructive collision proves an information-loss witness for the fixed carrier irrespective of decoder richness. It is still model/regime-specific, not universal3-state impossibility, and does not quantify prevalence under the random task distribution. Existing G1.5 already contains the conceptual collision argument; describe any verified default-parameter witness as scope extension, not a newly discovered principle.
