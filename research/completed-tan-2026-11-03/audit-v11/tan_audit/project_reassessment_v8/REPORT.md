# Original-project reassessment: stored target versus recoverable response

2026-11-02. Scientific continuation at frozen repository commit `85213f0d417d54e4f48b301ed1c0e4200ef3c421`.

The new check finds an exact, task-valid failure of the original TAN-I Probe1 observation contract. Swapping a low target and a high distractor changes the correct class while retaining both in the ordered window. A native common reset removes the remaining membrane difference; the query then pools the same multiset. Both histories produce the same complete permitted ten-coordinate response. This holds throughout a declared five-dimensional input box, not just at a numerically matched point. All 545 predeclared pairs were also literally identical in the frozen floating implementation.

This is not a collision of complete system state, proof that the neuron has no memory, or an estimate of frequent failure. It distinguishes stored task content from content available to the permitted readout. One intervention—suppressing only the t8 reset—restores an analog difference on these pairs, while their spike response remains identical. No classifier was fitted and no parameters were searched.

## Why this question was selected

[P1B_CAUSAL_ACCOUNT.md](P1B_CAUSAL_ACCOUNT.md) consolidates the previous work into input transformation, carrier evolution, query access and output policy. Partial input ambiguity cannot explain all-positive-novelty failure; limited external prediction does not establish binding; output-support bounds do not uniquely attribute the entire failure to spiking. The current P1B decoder exploration is closed rather than expanded.

The natural-collision study is TAN-I **Phase1**, distinct from later **P1A Rev2** static addressing. Phase1 already has strong common-reset evidence that h alone is not a sufficient state when an explicit window remains. The current paper already distinguishes covariance geometry from intrinsic dimension and content selection. Its Probe1 evidence is limited and seed-dependent, with a strong positional delay-line baseline; Probe2's scalar routing failure is already established. Repeating any of those generic statements or another near-collision ratio would be low value. See [NATURAL_COLLISION_REVIEW.md](NATURAL_COLLISION_REVIEW.md) and [BROAD_MANUSCRIPT_REVIEW.md](BROAD_MANUSCRIPT_REVIEW.md).

P1A was independently checked before selecting the new experiment: all **60,000 saved condition/trial records and 420,000 outputs** were recomputed from separately written formulas. Hard choices/values agree exactly; the largest soft arithmetic discrepancy is 2.58e-14. All 120,000 scalar-metric/vector-QK query choices agree. The 0/6,705 signed-scalar intermediate-key result is confirmed. These are not 60,000 independent underlying histories: C3/C4 share 10,000 histories and scaled noise draws. This is saved-ledger validation, not regenerated historical randomness or a new static-address experiment. The complete evidence and limitations are in [P1A_EVIDENCE_REVIEW.md](P1A_EVIDENCE_REVIEW.md).

The unresolved question selected here is narrower and task-relevant: **can the complete response granted to the original Probe1 decoder fail to distinguish unambiguous target classes even though the ordered state still contains the target?**

## Original task and fixed experiment

The original source is [temporal_distractor.py](../../Temporal-Attention-Neuron/snapshot/code/experiments/temporal_distractor.py), with dynamics imported from [natural_collision.py](../../Temporal-Attention-Neuron/snapshot/code/experiments/natural_collision.py). Target is at t5, distractors at t6–8, query at t9, and all other inputs in the fourteen-step block are zero. Low/high target supports overlap globally, [.55,1.05] and [.95,1.45], but the selected amplitudes lie strictly outside that overlap. Thus the witness does not exploit ambiguous latent labels.

Parameters remain W=5, lambda=.5, threshold=.5, strict `u>threshold` reset, Wq=2, Wk=Wv=1, beta=1 and epsilon=0. The original +1e-9 softmax denominator is retained. The only allowed task response is

`F = [u9,u10,u11,u12,u13,y9,y10,y11,y12,y13]`,

where u is the pre-reset candidate. No pre-query voltage, raw input, attention-vector coordinate or window tap is an allowed decoder feature.

[PREREGISTRATION.md](PREREGISTRATION.md) was saved before fixture execution, SHA256 `d8f0b97300f7fd1d3e3b0c07cd5ca17c3cfb4b40b20efa25792e6ab3dd78df23`. The anchor is:

| History | t5 target | t6 distractor | t7 distractor | t8 distractor | t9 query | Correct class |
|---|---:|---:|---:|---:|---:|---|
| L | .625 | 1.375 | .625 | 1.375 | 1.375 | Low |
| H | 1.375 | .625 | .625 | 1.375 | 1.375 | High |

The fixed family is `(a,b,c,d,q)` versus `(b,a,c,d,q)`, with a,c in [.615,.635] and b,d,q in [1.365,1.385]. Exactly one anchor, all 32 corners and 512 RandomState2026110225 draws yield **545 pairs / 1,090 histories**, with no exclusions. The random cases sample this paired intervention family, not the original unrestricted trial population. [probe1_inputs.jsonl](probe1_inputs.jsonl) was saved and hashed before simulation: `1773c6bd6ee595fe1049b71ff6b8e34a3e7c9007f5de7f509d51d8536117fcf4`.

## Exact mechanism and scope of the proof

At t8 both histories have a window that is a permutation of `(0,a,b,c,d)`, with the same current d. A rational interval proof bounds the original gated attention current strictly above **.528613674287**, hence above the .5 threshold. Earlier voltages are nonnegative, so both native h8 values reset to zero regardless of earlier order effects.

At t9 the ordered windows differ: `(a,b,c,d,q)` and `(b,a,c,d,q)`. Their means, positive surprise, sum of exponentials and weighted sum are equal in ideal arithmetic. The query remains strictly gate-open: exact S9 lies in [.284,.316]. The bound `0<u9<.4432<.5` prevents a query reset. Afterward, the zero continuation with nonnegative windows closes the gate. Consequently both complete responses equal

`[u9, u9/2, u9/4, u9/8, u9/16, 0,0,0,0,0]`.

At the anchor, u9 is approximately **.33538361861604077**. The ordered windows remain different at t9 and t10, then agree at t11 after the two swapped items have left. Thus a target available in full state at query is never exposed by this allowed continuation before it is discarded. The ideal equality includes the stabilizing denominator; it is proved algebraically, not inferred from a floating tolerance.

The interval certificate uses exact `Fraction` arithmetic: degree-40 positive exponential Taylor sums plus a geometric upper bound on the remaining tail. [theory_certificates.json](theory_certificates.json) records the rational endpoints and margins. A notation-only clarification is required in the frozen protocol: the numerator expression bounds `(sum(exp)+delta)*(C8−1)`, not `sum(exp)*(C8−1)`. The explicit expression already includes `−delta`, so the certificate and all parameters are unchanged. See [THEORY_SCOPE_REVIEW.md](THEORY_SCOPE_REVIEW.md).

Permutation-invariant aggregation is standard mathematics, not a new attention theorem; see [Zaheer et al., Deep Sets (2017)](https://arxiv.org/abs/1703.06114). The project-specific contribution is the admissible opposite-label pair, native common reset, complete-response proof, supported input neighborhood and intervention within this actual task. The already-known scalar argmax-order obstruction is a separate result.

## Fixed controls and outcomes

All four native models use the identical histories with their own original evolution. The named B4 intervention changes only the t8 voltage-reset assignment: retain u8 as h8, keep that step's threshold-event indicator, then resume the ordinary rule. It therefore deliberately permits an above-threshold post-step voltage at t8; it is a causal intervention, not an unchanged neuron implementation.

| Arm | Identical complete F pairs /545 | Analog max-coordinate gap across pairs | Pairs with different spike features /545 |
|---|---:|---:|---:|
| Native B1 LIF | 545 | 0 | 0 |
| Native B2 recency delay line | 0 | .048667–.051333 | 0 |
| Native B3 uniform gated context | 0 | .241311–.249725 | 545 |
| Native B4 full TAN | **545** | **0** | **0** |
| B4, suppress t8 reset only | 0 | .014339–.015923 | 0 |

The complete ordered-window target-tap oracle identifies **1,090/1,090** labels on this box; this is privileged access and not a general oracle for the original overlapping-support population. Target and window maximum-coordinate separation is .73–.77. The native B2 query gap is exactly `(b−a)/15` in ideal arithmetic (anchor 1.175 versus 1.125). This positive control uses an already-declared pre-reset voltage feature; its spikes nevertheless agree on every pair.

Native B4 resets at t8 on all 1,090 histories, has positive query surprise on all of them, and has no post-query current. All 545 native F pairs are **literally bit-identical**, not merely within the 1e-13 diagnostic tolerance. Observed query voltages span .318729–.351905; the exact whole-box proof is stronger than this sample range. B3's differing behavior reflects its own earlier state/reset trajectory and does not establish general superiority of removing attention.

The single-reset intervention preserves all window-derived currents and attention arrays but keeps the prior order-dependent membrane difference. Anchor query voltages become .6286703214272766 and .643798459505203. Both spike, so **analog separation is restored while spike-only separation is not**. This is not a demonstration that a classifier recovers labels over the full task, and no classifier or threshold rule was trained on these controls.

## What the collision establishes

Any classifier using only the common ideal F has error at least1/2 under the artificial equal mixture of each opposite-label pair. This concerns the task observation, not the complete state. The ordered window stores the target, and a different future-input or state-access contract could expose information that this particular query/continuation does not. The original task permits only the specified zero continuation. No universal neuronal impossibility or minimal state-count claim follows.

The fixed box gives a structure-preserving neighborhood of the witness, not arbitrary-noise robustness. Under the **ideal continuous** original amplitude generator, conditional on a present-distractor trial, each chosen non-overlapping width-.02 interval has mixture density1. Each orientation has probability mass(.02)^5=3.2e-9; the mirrored union has mass6.4e-9. The swap preserves density and reverses labels, giving a conditional ideal population-error lower bound3.2e-9 for response-only classification. This is a tiny analytical bound, not empirical prevalence from 545 pairs, not a finite-PRNG frequency claim, not the reported balanced-three-class error, and not an explanation of the large historical average errors. The finite dataset's rounded catch/present/distractor counts must not be replaced by a guessed exact one-third multiplier.

More broadly, the original B3/B4 Probe1 response is a deterministic function of u9 for any legal nonnegative trial: if the query spikes, later voltages are zero; otherwise they decay geometrically. Its ten coordinates are not ten independent temporal measurements. This source-level consequence does not invalidate the saved classifier results, but prevents interpreting the longer response vector as automatically providing extra memory observations.

## Evidence, resources and preservation

Completed primary commands returned exit0:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/project_reassessment_v8/theory_certificates.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/project_reassessment_v8/run_probe1_collision.py
```

The fixed run executed 4,360 native trajectories plus 1,090 single-reset intervention trajectories, 76,300 primary updates. A separate primary-side native reconstruction checked 4,360 trajectories /61,040 updates. The JSON field names `primary_euler_steps` and `validation_euler_steps` refer here to discrete TAN-map updates, not integration of a continuous-time ODE. Main runtime was3.029s, peak RSS125,648KiB. Native TAN retains its explicit five-input window plus membrane; the privileged target-tap and single-step reset control are separately counted access/intervention resources. No biological energy, hardware timing or noise budget is inferred from software runtime.

Full evidence: [probe1_results.json](probe1_results.json), [probe1_pair_ledger.jsonl](probe1_pair_ledger.jsonl), [probe1_trajectories.npz](probe1_trajectories.npz), [probe1_execution.log](probe1_execution.log), [EXECUTION_NOTES.md](EXECUTION_NOTES.md), and the exact certificates. The independent theorem and numeric reviews are [THEORY_SCOPE_REVIEW.md](THEORY_SCOPE_REVIEW.md), [REVIEW.md](REVIEW.md) and [INDEPENDENT_REVIEW_CHECKS.json](INDEPENDENT_REVIEW_CHECKS.json). The separate scalar checker completed exit0 and verified all545 pairs,5,450 trajectories/76,300 updates,35 saved arrays,2,725 pair/model ledger rows and all summary fields. Maximum scalar-versus-saved trajectory discrepancy was7.77e-16; summary recomputation matched exactly. Its initial checker-only schema mismatch and subsequent adaptation are disclosed in the review, with the initial log preserved; primary outputs were not rerun or changed. A source-only archive replay also passed with Git provenance explicitly marked unavailable rather than falsely clean.

The historical P1A arithmetic check is separately reproducible from its compressed ledgers and [p1a_ledger_check.py](p1a_ledger_check.py), with [P1A_LEDGER_CHECKS.json](P1A_LEDGER_CHECKS.json) and the supplemental input-field check. Previous evidence and source-cited errata remain preserved. No public edits, new dependencies, spending or historical label changes occurred. The updated research decision is in [RESEARCH_AGENDA.md](RESEARCH_AGENDA.md).
