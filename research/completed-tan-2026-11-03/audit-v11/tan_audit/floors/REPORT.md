# Independent P1B metric, alphabet, and native-cue audit

Frozen repository: `85213f0d417d54e4f48b301ed1c0e4200ef3c421`.
The prior earlier archive was unavailable in this executor. This is an independent reconstruction from repository source, not direct verification of the eight archived experimental stages. 

## Confirmed metric and general codebook floors

`p1_experiments/p1b/runner_p1b.py:147–157` constructs both sum and difference from the same two operand predictions, then averages their absolute errors. With signed operand errors eA and eB,

`E_comp = (|eA+eB| + |eA-eB|)/2 = max(|eA|, |eB|)`.

The task generator samples independent uniform values on [-3,3], independently of candidate keys, positions and target selection (`task_generator.py:74–96`). Thus the two selected targets are independent uniform operands. For any fixed finite output alphabet S, nearest-symbol decoding independently minimizes both absolute operand errors and hence their maximum, even if arbitrary coupled access to both true targets is granted. If D is the distance of a uniform target to S, the relaxed optimal population loss is `integral (1-F_D(t)^2) dt`. This relaxation grants true targets, arbitrary codeword selection and the use of zero fallback without delivery penalty. Actual dynamics can perform worse.

| Alphabet relaxation | Exact population E_comp floor | Decimal |
| --- | --- | --- |
| Count {-2,-1,0,1,2}; zero fallback adds no symbol | 25/54 | 0.462962963 |
| Twenty latency midpoints, without fallback | 1/10 | 0.100000000 |
| Twenty latency midpoints plus zero fallback | 3139/32000 | 0.098093750 |

For count, `F_D(t)=5t/3` on [0,1/2] and `(t+2)/3` on [1/2,1]. For latency plus zero, `F_D(t)=7t` on [0,3/40] and `19t/3+1/20` on [3/40,3/20]. Integrating these exact rational pieces produces the stated values.

These are population bounds under the specified uniform independent targets, fixed alphabet, paired sum/difference metric, and no rescaling. They are not deterministic bounds on each finite-sample mean. They do not apply to analog observations or newly calibrated codebooks.

## Stronger bounds for the original clean-task cue

The frozen model always ends encoding with at least 100 silent steps (`models_p1b_v2.py:178–189`). Before that silence, a non-refractory stored voltage is below threshold 1, and a refractory state is reset to zero. Consequently at query onset `r=0` and `u<0.99^100=0.3660323413`; registered delays only decrease this upper bound. The clean task restricts query q to [0.3,2.5]. During cue, `I_content=0` and `I_query=q` (`models_p1b_v2.py:249–254`), so A does not affect voltage.

Even under the conservative upper-envelope state `u=0.99^100`, `q=2.5`, the first spike is at step 36. A second spike requires 20 refractory steps and at least 51 integration steps from zero: its earliest possible step is 107, outside the 99-step eligible window. Therefore the clean registered task produces at most one eligible spike. The fixture's examples for counts 2–5 use q=3.5,7,15,40, explicitly outside the clean query range (`test_p1b_g1_fixtures.py:640–648`). Those fixtures verify decoder mechanics, not reachability within the task.

The native count output is consequently restricted to {-2,0 fallback}. Native latency first-spike time is at least 3.6 ms, so only bins 7–19 can be eligible. A conservative native latency output superset is `{-0.75,-0.45,-0.15,0,0.15,0.45,...,2.85}`. No claim that every symbol in this superset is reachable for every history is required for the following relaxation.

| Native-task output superset | Exact relaxed population E_comp floor | Decimal |
| --- | --- | --- |
| Count {-2,0 fallback} | 13/9 | 1.444444444 |
| Latency bins 7–19 plus zero fallback | 1243/1600 | 0.776875000 |

The count CDF is `2t/3` on [0,1] and `(t+3)/6` on [1,3]. The latency-superset CDF is `14t/3` on [0,3/40], `4t+1/20` on [3/40,3/20], and `t/6+5/8` on [3/20,9/4]. Exact integration is implemented generically using clipped Voronoi half-cell lengths, rather than hardcoding the answers.

These sharper bounds require the frozen membrane time constant 10 ms, threshold/reset, final 10 ms silence, clean bounded cue, and zero cue content input. They must not be carried over without a new derivation to slower membrane variants, a nonzero content probe, changed sampling, or extra analog readout. Mathematical rational alphabets idealize code constants; floating evaluation introduces only machine-precision changes to reconstruction values.

The count channel's only non-abstaining value is -2. Therefore joint delivery success at the registered strict tolerance |error|<0.50 additionally implies both targets lie in (-2.5,-1.5). Its population joint delivery rate is at most `(1/6)^2=1/36`, regardless of how count occurrence depends on the history. This is an upper bound, not a claim it is attained.

## Interpretation and existing limitations

Protocol V2.6 already discusses ideal-quantizer references and first-event erasure. Neither is a new discovery here. The explicit alphabet-constrained and native-cue reachability bounds sharpen that accounting. The original decision threshold 0.05 and historical `DEMONSTRATED_CHANNEL_LIMITATION` labels are unchanged. The unreachable success threshold for these frozen output sets prevents assigning the failure specifically to a general inability of one neuron to carry information. The final scientific closure already limits its claims to the tested end-to-end implementation (`P1B_FINAL_SCIENTIFIC_CLOSURE_REPORT.md:106–112`); README's statement of distributed-circuit necessity is broader than those results support.

Other bounded checks confirmed old A is used for the present-step current, that query dynamics is A-blind when content input is zero, and that the registered pre-reset candidate/refractory semantics and step-100 exclusion operate as written. The scaled collision fixture reruns with residual `2.03e-20`, not literal bitwise-equal states. It uses wK=wV=0.1, so its empirical example should not silently be called a production-weight history. Its analytic fixed-branch argument and numerical tolerance are separate claims.

## Actual validation

Command: `PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/floors/audit_floors.py`.

- Exact rational metric identity checked on 9,409 signed-error pairs.
- Exact Voronoi integration independently confirms all five alphabet results.
- For each alphabet, 262,144 fresh independent uniform pairs (fixed seed 2026110251) numerically cross-check the exact result; each estimate lies within six estimated standard errors. This Monte Carlo check is validation, not the evidence for exactness.
- Five bounded registered fixtures passed: G1.1 state closure/refractory deadline, G1.5 scaled collision, G1.6a arithmetic references, G1.6b actual spike decoding, G1.7 Euler convergence. G1.6a alone is only an arithmetic assertion; the independent integration is the substantive oracle check.
- Production cue at the conservative upper-envelope state has first spike 36 and eligible count 1; changing A between 0 and 5 leaves all observed cue outputs identical.

Deliverables: audit_floors.py and results.json (exact pieces, Monte Carlo checks, versions and source SHA-256). The earlier whole-archive bytes were unavailable and are outside this completed audit scope.
