# Independent review of the original Probe1 response-collision study

2026-11-02. **PASS for the frozen experiment, numerical reconstruction and ideal response-collision theorem, with the scope limits below.** No primary scientific artifact or repository file was changed by this review. The numerical checker imports neither the primary scripts nor the original model/generator modules; it independently reconstructs the stated scalar operations. A separate [THEORY_SCOPE_REVIEW.md](THEORY_SCOPE_REVIEW.md) checks admissibility, exact inequalities and probability interpretation.

Preregistration SHA256: `d8f0b97300f7fd1d3e3b0c07cd5ca17c3cfb4b40b20efa25792e6ab3dd78df23`. Original source hashes remain `8c7b4e75ec8284b671bca6280ee848498ddd9c6109a6dba31254f6d407ec4d7f` for `natural_collision.py` and `6177df497f9735d9fdb434ff20e7751a8dbb4af7a32d836bb3c4898c06ad4f23` for `temporal_distractor.py`. The frozen source was read directly, including the generator, timing, latent labels, threshold/reset, unshifted softmax and response-feature definitions.

## Completed execution and numerical checks

The command

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python tan_audit/project_reassessment_v8/independent_review.py
```

completed with exit **0**, in 0.436 s with recorded peak RSS 51,396 KiB. These are software observations, not biological resource measurements. [independent_review.log](independent_review.log) and [INDEPENDENT_REVIEW_CHECKS.json](INDEPENDENT_REVIEW_CHECKS.json) preserve the result and hashes.

| Independent check | Coverage | Largest numerical discrepancy |
|---|---|---:|
| Frozen inputs and latent classes | Anchor, all 32 corners, all 512 RandomState box draws; 545 pairs / 1,090 histories | Exact input-record equality |
| Scalar original/intervention trajectories | 5,450 trajectories, 76,300 updates; all u, h, current, surprise, spike flags and available attention arrays | 7.78e-16 |
| Saved NPZ fields | All 35 arrays, including windows, labels, parameters, oracle and ten response coordinates | Within the scalar bound above; common inputs/windows exact |
| Pair/model ledger | All 2,725 rows, every saved feature, target, query, margin, label and gap | 2.23e-16 |
| Arm/control summaries | Every summary field and denominator | 0 |
| Exact theorem | Independent degree-40 Fraction exponential bounds; support, branch and mass arithmetic | Exact rational sign checks |

The initial reviewer command stopped because the primary summary had three additional fields beyond the provisional schema: analog-only gap, spike-feature disagreement and query-spike disagreement. The checker was extended to reconstruct these fields explicitly, and then passed. The initial log is preserved as [independent_review_initial_schema.log](independent_review_initial_schema.log). This was a checker-schema correction, not a changed input, model, primary outcome or scientific selection. Primary files stayed unchanged. The primary metadata fields named `primary_euler_steps` and `validation_euler_steps` count discrete TAN-map updates; that naming does not establish a continuous-time Euler implementation of this original model.

## What the native data and controls show

The native B4 output has **545/545 literally equal floating response vectors** across the target/distractor swaps. A separate exact C-order float64 byte comparison also confirms all 545 complete B4 responses are bit-identical; [BIT_PATTERN_CHECKS.json](BIT_PATTERN_CHECKS.json) binds that check to the saved NPZ hash. This is a numerical implementation observation in addition to the separate ideal proof. Every paired full ordered window differs at t9 and t10; the target-tap oracle identifies all 1,090 target classes on these disjoint-support intervals. The common query is exactly the same on both sides. There are no exclusions or fitted classifiers.

| Arm | Pairs with different analog response | Pairs with different spike-only response | Interpretation |
|---|---:|---:|---|
| Native B1 LIF | 0 / 545 | 0 / 545 | Negative control: common query after earlier resets |
| Native B2 delay line | 545 / 545 | 0 / 545 | Existing analog observation preserves order; not a claim of spike-only recovery |
| Native B3 uniform gated | 545 / 545 | 545 / 545 | Diagnostic alternative native history evolution |
| Native B4 full TAN | 0 / 545 | 0 / 545 | Complete allowed response collision |
| B4 with only t8 reset suppressed | 545 / 545 | 0 / 545 | Order-dependent membrane residue becomes visible in the pre-reset analog response |

B2's query gap matches `(b−a)/15` throughout the family and equals 1/20 at the anchor. The reset-suppressed B4 analog gap ranges from 0.0143386 to 0.0159234; its spike-only responses remain identical. Separating an analog feature is not a demonstration of a successful trained classifier or a biological readout. The native B3/B4 comparison uses each model's own history evolution, so it is not a matched intervention on attention at one fixed checkpoint.

The single-step B4 intervention is implemented correctly: u through t8 and h through t7 match native B4, the t8 threshold event flag is preserved, h8 becomes that step's candidate, and every other reset obeys the original rule. Current, surprise and attention remain exactly unchanged at every step, since they depend on the unchanged input window rather than membrane state. This is an explicit change to the reset assignment, producing an otherwise non-native post-reset voltage; it must not be called an unchanged neuronal implementation.

## Ideal theorem and notation clarification

All amplitudes are strictly inside their native class supports and outside the overlap region. The target and first distractor exchange amplitudes and compatible latent classes; only the target class defines the answer. At t8, the two windows have the same multiset, though the preceding membrane states can differ. Exact inequalities guarantee current greater than threshold on both sides, forcing the common reset. At t9, surprise is strictly positive and the query voltage is strictly below threshold, so this is neither zero-novelty erasure nor a threshold-readout coincidence.

With D=`sum(exp)` and the source denominator offset δ=10^-9, the exact identity is **`(D+δ)(C8−1)=N−D−δ`**. The protocol's informal prefactor `D(C8−1)` omits δ; its explicit lower-bound expression already contains the correct `−δ` and remains a valid positive-sign certificate. This harmless notation correction is recorded here without changing the frozen protocol. Independent rational bounds certify the context-sign expression greater than 0.70459354 and `exp(1.114)−3>0.04652013`. Together they prove the required common reset uniformly over the box. The conservative query bound is `0<u9<277/625<1/2`.

The native query aggregation is symmetric in the window multiset. After the common reset the two ideal u9 values therefore coincide. Subsequent zero inputs close the gate, so the full ten-vector is

`[u9,u9/2,u9/4,u9/8,u9/16,0,0,0,0,0]`.

This proves equality of the **entire permitted task response**, while the complete ordered state at query still contains different target values. It is an observation-sufficiency failure, not a full-state collision, a refutation of Markovianity of the complete system, or a theorem about every possible future probe.

More generally, in this original Probe1 task, all B3/B4 response coordinates after t9 are determined by u9 and its threshold event: if u9 spikes, later voltages are zero; otherwise they decay geometrically. Thus five voltage samples are not five independent temporal observations. This is direct source algebra under nonnegative histories and the prescribed zero continuation, not a new population experiment. Different future nonzero probes or access to the window would change the observation contract.

## Probability, robustness and novelty limits

Under the ideal continuous original generator conditional on a present-distractor trial, each selected unambiguous amplitude interval has width .02 and mixture density 1. The orientation-box mass is therefore `(.02)^5=1/312500000=3.2e-9`. The target/distractor swap is a measure-preserving involution between two disjoint equal-density boxes and reverses the label while preserving ideal F. Any response-only randomized classifier incurs at least one-half error conditional on their union, giving the tiny conditional-population lower bound 3.2e-9.

This is **not** the observed occurrence rate in the 512 coupled draws, an exact discrete-PRNG probability, a direct bound on the paper's balanced-three-class metric, or an explanation for the reported large mean task error. The positive-volume family shows validity across admissible structure-preserving changes. It does not establish robustness to independent sensor/process noise, arbitrary precision, changed timing or changed readout. Exact floating equality in these 545 cases is recorded separately from the ideal continuous argument.

Permutation-invariant aggregation, hard-reset information loss and gate-closed decay are standard mathematical facts. The project-specific contribution is the admissible opposite-label construction, its complete native-response identity, the positive query gate, explicit ordered-state distinction and the isolated reset intervention. Existing natural-collision evidence already established h-only insufficiency; this study neither rediscovers that alone nor claims flexible key-addressing capability. P1A static addressing, TAN-I Probe1 target-class recall and P1B compressed dynamic retrieval remain distinct contracts.

## Preservation and archive replay

The checker locates sibling `tan_audit/` and `Temporal-Attention-Neuron/` directories from its own path and writes only review outputs. Mandatory hashes verify both original sources even when an extracted archive omits Git metadata. A separate source-only extraction replay completed with exit 0; its unavailable Git provenance is explicitly `repository_status_verified=false` and `repository_status_unchanged=null`. [ARCHIVE_REPLAY_REVIEW.json](ARCHIVE_REPLAY_REVIEW.json) records this test.

In the actual checkout, Git status was clean before and after, and all primary hashes were unchanged. No primary rerun, parameter change, dependency installation, publication or purchase occurred during review. No blocker remains. This review does not claim to audit the full historical Probe1 classifier suite, the P1A ledger audit or the deferred mutual-information analysis.
