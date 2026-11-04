# Input-boundary audit: rectifier ambiguity is not a positive-novelty theorem

Prepared from the frozen repository before the new native-model replay. This is an exact input-transduction and query-identifiability analysis; it does not claim that an injective input stream remains recoverable from the final neuronal state or registered readout.

## 1. Identify the actual boundary before calling two histories identical

The Python presentation function `simulate_encoding(keys, values, mask, ...)` receives raw arrays. It is therefore incorrect to say that the entire Python API, generator, runner or original TAN project is given identical inputs in the paired examples below. The raw arrays differ and remain available to the experiment presenter. The registered physical carrier sees their prescribed transduction, not unrestricted access to those arrays.

The registered active-event boundary is

`I_content = k+v`, `nu = max(0,abs(k)-1.4)`, `I_query=0`.

Keys are strictly positive in the generator, so `abs(k)=k`. During blank slots and ISIs both content and novelty are zero. At query, content and novelty are zero and `I_query=q`. The state update also receives fixed shared parameters; its previous u,A,r are feedback state, not another exogenous key/value channel. Event masks control stimulus presentation and the post-event reset interventions; the same masks and onset/offset schedule must therefore be included when comparing complete input/control paths.

Source boundary: `P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md:47–74,100–106`; `models_p1b_v2.py:112–198,200–219,242–253`; `runner_p1b.py:109–127`. The presenter computes currents from raw arrays, then calls `euler_step`. No historical array, target index, target value, trial ID, role ID or second query is passed into that recurrent step or the registered decoder.

Initial state is u=A=r=0 for M1, M2a, M2c and M4 encoding. Constant-gate controls M2b/M2d start with A=A0 and impose the same independently calibrated, globally frozen A0. Post-event resets differ by model but not by history within a pair. All paired-indistinguishability statements require the same model, calibration constants, initial condition and external control schedule on both histories. They hold for any shared admissible initial state and shared constants, not only a zero initial gate.

Queries are clean in the registered P1B generator: q is exactly the requested key. Role A and B are independent repeated probes of the same encoded/delayed state, not sequential probes through a mutated state. Target indices remain evaluator metadata. The distinct-candidate requirement is satisfied by all examples. The static M0 reference is an explicit exception: it reads raw key/value arrays and event positions at query time (`runner_p1b.py:90–105`), so the interface-collision bound does not apply to M0.

The full original TAN model with its history buffer is another model/contract, not this closed P1B carrier. No claim about all original TAN inputs follows from this P1B-specific transduction audit.

## 2. Strictly positive novelty preserves the complete event pair

For nu>0, the registered domain gives k>1.4 and the exact inverse

`k = nu+1.4`, `v = I_content-k`.

Consequently, with the same complete input/control waveform, every positive-novelty active event has the same key and value at the same time. If all active events have positive novelty, two different task histories cannot have the same exact full input path and same queries while requiring different targets. In particular, **there is no nontrivial exact N=2 collision with both events positive novelty** at this boundary.

Equal content sums alone are insufficient. The planned positive-novelty comparison is:

| History | Ordered (key,value) events | Content | Novelty | Common queries | Target vector |
|---|---|---|---|---|---|
| H0 | (1.6,1), (2,-1) | (2.6,1) | (.2,.6) | (1.6,2) | (1,-1) |
| H1 | (2,.6), (1.6,-.6) | (2.6,1) | (.6,.2) | (1.6,2) | (-.6,.6) |

Both histories have positive novelty at both events, but their novelty waveforms differ. They are a test against a projected-scalar mistake, not a counterexample to full-input injectivity. Even if terminal states or a restricted readout later coincide, that would be a carrier/readout collision after distinct inputs.

Machine arithmetic needs a separate qualification. With binary64 k and the actual binary64 MU_BG, positive-domain subtraction k−MU_BG is exact by the Sterbenz condition: k/MU_BG<2 and MU_BG/k<2. Thus distinct representable positive-novelty keys cannot collapse through this subtraction. The rounded content addition can merge nearby values at a fixed key, so the full floating map is not asserted to be literally injective. The maximum content-sum ULP over the registered range is 2^-50≈8.88e−16, bounding this value ambiguity at machine scale. It cannot support a 0.5-scale target ambiguity. This does not excuse larger numerical state or readout errors downstream.

## 3. A stronger query-aware sufficient-identifiability statement

Positive novelty is sufficient but not necessary for task-relevant identifiability from the complete input path and a valid exact query. Suppose there is at most one zero-novelty active event in the history.

1. Decode all positive-novelty event keys and values using the inverse above.
2. If q matches one of those distinct keys, its value is uniquely identified.
3. Otherwise the promise that q requests an active candidate means q is the unique zero-novelty event's key. Its value is `I_content_at_that_event−q`.

Thus histories with zero or one zero-novelty event have uniquely determined requested values under this full-path oracle. This is true for either N=2 or N=3 and does not require giving the oracle the other role's query. If the sole low event has zero content and is indistinguishable from a blank at the current boundary, its value is nevertheless −q; event position is then irrelevant to value recovery. Granting the known presenter mask makes the proof especially direct and cannot weaken any collision lower bound below.

The theorem depends on clean valid queries, distinct candidate keys, the declared channel map and complete input access. It is not a decoder implemented inside M1 and does not grant model access to old waveforms. It also does not imply that every history with two or more zero-novelty events is ambiguous; that is a necessary location for the constructed ambiguity, not a complete ambiguity classifier.

In particular, an N=2 history with one high-novelty and one low-novelty event is task-identifiable from the full input and query, although its low event's isolated per-event map is many-to-one. This is why a broad claim that every rectified low input destroys task information would be incorrect.

## 4. Exact paired histories on the flat novelty branch

Use slots (0,4), the same active mask, the same 10 ms events/10 ms ISIs and queries q=(.8,1):

| History | Ordered events | Content | Novelty | Targets |
|---|---|---|---|---|
| L0 | (.8,1), (1,-1) | (1.8,0) | (0,0) | (1,-1) |
| L1 | (1,.8), (.8,-.8) | (1.8,0) | (0,0) | (-.8,.8) |

All keys are in [0.3,2.5), values in [-3,3), key separation is .2>.05, and both queries request distinct actual events. Target indices swap, but those indices are not model inputs. Content, novelty, query, masks, event boundaries and time parameters all agree at the carrier boundary. Raw key/value arrays do not agree.

For a deterministic carrier, equality of the complete exogenous input/control path and initial state implies equality of the whole state/output trajectory by induction, including threshold resets, refractory steps and gate clamps. Therefore these two histories are indistinguishable to every registered dynamic model under shared frozen constants. M1's zero gate is one additional reason this particular pair is difficult, but it is not needed for the input-equivalence argument.

## 5. A primed N=3 collision separates projection ambiguity from zero-gate erasure

Use slots (0,2,4), with an identical positive-novelty primer followed by two low-novelty events. Shift the low-event values so both low content currents are nonzero:

| History | Ordered events | Content | Novelty | Common queries | Targets |
|---|---|---|---|---|---|
| P0 | (2,0), (.8,1.5), (1,-.5) | (2,2.3,.5) | (.6,0,0) | (.8,1) | (1.5,-.5) |
| P1 | (2,0), (1,1.3), (.8,-.3) | (2,2.3,.5) | (.6,0,0) | (.8,1) | (-.3,1.3) |

This is legal N=3, with all three keys distinct and all queried keys present. Both probes request the two low events, in opposite event-index order across the pair. The common first event is not queried. Full input/control paths remain identical.

For M1, let b=149/150 be the gate decay per Euler update. The gate at the first ambiguous low-event onset (40 ms) is exactly

`A_400=(3/5)*(1-b^100)*b^300 > 0`.

At the second low-event onset (80 ms), it is

`A_800=(3/5)*(1-b^100)*b^700 > 0`.

The gate remains strictly positive at every intervening finite step. The low events have nonzero content currents 2.3 and .5, so their gated currents are also nonzero. This cannot be dismissed as only a first event arriving at A=0. The ambiguity is in assigning keys to two events on the rectifier-flat branch, before any carrier compression. It remains restricted to those low-novelty events; it does not create a collision between two all-positive-novelty histories.

This distinguishes the current diagnostic from the already registered Proposition 6.1. That proposition states that a zero-novelty event beginning with A=0 is erased by M1 and derives its known N=2 population bound. The present primed construction has A>0 and supplies no new original-distribution probability bound.

## 6. Correct lower bound and probability scope

For either L or P pair, the two target vectors differ by 9/5 in maximum norm. Any shared response vector z obeys

`max(||z−target_0||_infinity, ||z−target_1||_infinity) >= 9/10`.

The registered composition loss equals maximum absolute operand error, so this is also a pairwise minimax Ecomp lower bound. A randomized decoder with the same output law on the two indistinguishable inputs has mean Ecomp at least .9 under an explicitly imposed equal two-point mixture. At strict .5 operand tolerance the two joint-success regions are disjoint, giving joint success at most 1/2 under that artificial equal mixture. Abstention cannot improve joint delivery and its fixed fallback remains in Ecomp.

These are constructed-pair or explicitly artificial-mixture statements. They are not a failure probability, average risk or frequency estimate under the original continuous generator. They do not imply the README's general architectural necessity claim, and they do not apply to M0's richer raw-cache input contract.

## 7. Correction and next original-project decision

Earlier phrasing should be narrowed to: “The registered P1B transduction has task-relevant ambiguities when multiple events lie on its rectifier-flat key branch; there are exact same-input, same-query counterexamples even after a common positive-novelty primer activates M1's gate.” Add explicitly: “When all events have positive novelty, the full input map is invertible; more generally, at most one zero-novelty event is sufficient for valid-query value identification by a full-path oracle.”

This still leaves retention, addressing and readout as separate questions. The smallest contract-preserving next analysis is a secondary poststratification of the unchanged frozen P1B replay: count zero-novelty events, first-event novelty and known-query identifiability, retain all denominators, and apply the already frozen M1 model/decoder without fitting. The all-positive and at-most-one-zero strata remove this particular input ambiguity; any remaining poor delivery must be located downstream. Neither subset success nor failure changes the original preregistered labels or permits a new equivalence inference after a baseline failure.

If the task instead requires a guarantee for every allowed history, the current interface is insufficient. A minimal channel-level remedy is to make the second input coordinate distinguish low keys, for example replace the rectified novelty coordinate with raw k while retaining the content sum. This uses two input coordinates but changes the registered meaning/dynamics of the second channel; it does not by itself solve retention or readout. Adding a tiny strictly monotone low-key component would likewise require precision/noise accounting. No such change is made in this audit, and no hidden waveform cache or new decoder access is granted.

The exact certificate script checks the six declared histories and full rational waveform/control equality or inequality, including every delay and repeated role query, without drawing an IID sample or simulating the native carrier. Native-code execution and poststratified replay are separate separate validation steps. Original files/results remain untouched.
