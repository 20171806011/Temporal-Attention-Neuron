# Independent review of registered input-boundary reassessment

Reviewed 2026-11-02. **Pass within the narrowed P1B contract and descriptive replay scope.** The review independently reconstructed the inputs, native M1 recurrence, full boundary call records and diagnostic statistics. No primary analysis plan, model, generator, input ledger, prediction ledger or historical repository result was modified.

Executed verification:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/interface_reassessment/independent_review.py > tan_audit/interface_reassessment/independent_review.log 2>&1
```

The complete checker passed with exit0. It does not import the experimental scripts, repository generator or native model implementation. The checker was first run for the completed stratum outputs, then expanded to include the later-arriving boundary evidence and complete runner-output fields; these were verification passes, not new experimental conditions. Final checks are recorded in INDEPENDENT_REVIEW_CHECKS.json.

## Independently reproduced evidence

The reviewer recreated the exact random-call sequence for 2,000 N2 and 2,000 N3 histories, including 82 and279 rejected key draws respectively. All4,000 serialized histories match, including query/target assignment, event positions, masks and per-trial seed metadata. Seed metadata was not used to reset the continuous RNG.

A separately implemented vectorized hybrid recurrence reconstructed all encoding states, silence delays, independent role probes, deadline candidates/refractory conditions and frozen affine predictions. All12,000 paired episode predictions and all encoding states match the primary ledger exactly in binary64 (maximum discrepancy0). This is stronger than checking aggregate summaries alone. All24,000 operand outputs, all66 reported stratum cells, the six unstratified frozen-summary comparisons and all oracle predictions were independently checked.

The zero-count partitions and ordered novelty patterns retain full-N denominators. The offline oracle recovers the2,534 histories in the sufficient-information subset with maximum target discrepancy4.44e-16. Its values never feed into the native simulation. The oracle is an evaluator with historical event-port access, not an implemented M1 decoder or a free memory baseline. The primary M1 functions have no added state or access to its answer.

For deterministic fixtures, the reviewer independently reconstructed the six rational histories and checked their original-generator admissibility: positive in-range distinct keys, minimum spacing, value bounds, legal slots and queries that reuse actual candidate keys. The three pair classifications agree with the source. All52,500 stored physical Euler calls, 306 NPZ arrays, five physical carriers, all initial conditions, event-offset interventions, delay paths and independent query checkpoints were checked against separately implemented equations. The144 runner episode outputs across all eight registered models and all three delays agree with those records, and their144 complete-call hashes match. The saved call count of210,000 includes the primary runner rechecks; the reviewer independently validated the52,500 unique stored physical calls and hashes for the additional repetitions.

## Correct input-information boundary

The generator and `simulate_encoding(keys,values,mask,...)` presenter still possess the raw arrays. M0 deliberately reads a raw key/value cache at query time. They are not given identical raw inputs in the collision examples. The equality occurs after the registered transduction into content current, rectified novelty and query current, with the same timing and intervention controls. The audit should not be described as proving information loss in the entire Python API or the original window-based TAN model.

For strictly positive novelty, the full active-event pair has the inverse `k=nu+1.4`, `v=I-k`. Thus the ideal positive-novelty map is injective. The both-positive N2 control has equal content sums but different novelty waveforms and is correctly rejected as a full-input collision. This corrects any earlier inference based only on the projected content scalar.

Machine arithmetic is appropriately qualified. Positive-domain subtraction of the binary64 key and binary64 threshold is exact under the Sterbenz condition; the checker also verifies this equality for every positive-key event in the reconstructed history ledger. Rounded content addition can still merge adjacent representable values, so literal binary64 injectivity of the entire map is not claimed. Its at-most2^-50 value-scale ambiguity cannot produce the0.5-scale counterexample at issue.

The stronger query-aware statement is also sound with its assumptions: if at most one active event has zero novelty, a clean valid query either matches a decoded positive key or identifies the unique remaining low event, whose value is content minus q. **The saved oracle is granted the active-event list/mask and the fixed-N or at-most-one-zero promise.** These are explicit information-analysis conditions, not a historical mask buffer passed into M1. M1's `euler_step` has no explicit mask argument; the presenter uses masks for input scheduling and model interventions. An unconstrained cross-N comparison with unknown masks and additional invisible zero-current events would require a different statement. The all-positive query-matching result is direct without such a fallback.

Conversely, two or more zero-novelty events are not automatically ambiguous for every history or query. The fixtures prove existence of task-relevant ambiguity within that part of the domain, not its prevalence or a complete ambiguity classification.

## Primed collision is distinct from the known closed-gate erasure

The N2 low-flat pair has identical complete drive/control paths and common queries, with ideal target separation9/5. The N3 pair prepends a common positive-novelty event and uses two later low-novelty events with nonzero content currents2.3 and0.5. In native M1 the gate is strictly positive at both low-event onsets and throughout those presentations; the exact gate formulas and actual records agree. Consequently the later gated currents are also nonzero. This is not merely the already registered first-event A=0 erasure mechanism.

For both pairs, the same-drive argument holds across all registered dynamic models with shared model-specific initial conditions and parameters, including the constant-gate controls. The full encoding/delay/query trajectories match exactly in the recorded floating run. M0 distinguishes the raw records and is correctly excluded from this bound.

The target separation gives a constructed-pair minimax Ecomp lower bound9/10, or an equal-mixture expected-risk bound under an explicitly imposed two-point mixture. The strict0.5 joint-success regions are disjoint. Neither statement estimates risk under the original continuous task distribution. No biological or distributed-population necessity theorem follows.

Positive-input injectivity does not imply every ablation uses every input. M2b/M2d pass a novelty argument but `clamp_A_const` makes their gate ignore it; the positive-projection pair therefore gives equal states/outputs there despite different full novelty arguments. M2a/M2c intervention/readout behavior and M4 quantization can likewise merge distinct streams. In the delay0 fixture, M1 roleB outputs differ (0.005672490 versus0.000946262), while M2a/b/c/d and the two M4 outputs coincide. These are downstream intervention/state/readout effects, not counterexamples to full positive-input injectivity.

## Interpretation of the native replay

The six unstratified M1 cells reproduce the frozen historical joint accuracies and mean Ecomp. However, the original historical per-trial input bytes were unavailable; the new ledger is a conditional reconstruction using frozen source, seeds and NumPy2.3.5. Exact agreement with aggregate results does not by itself prove byte identity with an absent historical ledger.

At delay0, all-positive novelty histories have joint accuracy16/486 for N2 and7/229 for N3. The broader at-most-one-zero group has47/1525 and31/1009 successes respectively, while the offline observer identifies the queried values. Native failure therefore persists where this particular upstream ambiguity is removed. This does not isolate retention versus query transduction versus frozen readout as the cause. Novelty strata also change key amplitudes and query currents, so their risk contrasts are descriptive rather than causal interventions.

These are retrospective diagnostic strata specified before their new outcomes, with aggregate original results already known. They are not new confirmatory groups, new pass/fail gates or an opportunity to infer membrane redundancy after the original baseline-floor failure. All original labels remain unchanged.

## Source map and manuscript crosscheck

A separate read-only source crosscheck confirms that the source-contract errata have concrete support: Paper2's P1B passage describes different state equations, a different r/readout role, the wrong M2a intervention, and incorrect operand/TOST thresholds, while its historical numerical outcomes refer to the registered run. README necessity language exceeds the closure's explicit model-specific scope. The historical reports' numerical results are not invalidated solely by those description inconsistencies; they require a separately reviewed documentation correction.

The project map correctly distinguishes the original TAN history buffer, P1A's static routing question, the authoritative P1A rev2 report and P1B's closed-state carrier test. It does not claim that this audit reran every historical manuscript experiment or the whole P1A suite. SOURCE_CROSSCHECK.md includes exact source anchors and the additional READMEline30 necessity wording. No publication or repository edit was performed.

 The scientifically justified next decision is to correct the contract description and keep upstream ambiguity separate from the remaining native carrier/query/readout failures, rather than add further enriched interpolation variants.
