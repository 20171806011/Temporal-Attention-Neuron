# Independent review of the binding-metadata positive control

Reviewed the frozen preregistration, `run_control.py`, its imported coefficient
helper, the frozen model/generator, all inputs/predictions/resource rows and
reported summaries. No primary file or result was rewritten.

## Verdict

No material leakage, root-selection, scoring, branch, or input-generation defect
was found. The experiment supports this narrow claim: on the 256 fresh original
N=2 histories tested, the modified carrier's full analog state can be inverted
with correct externally retained key/slot metadata; this particular inverse
fails substantially when that association is swapped. This is conditional
full-state reconstruction, not original neuron-only or two-probe retrieval.

It does not establish global injectivity over the complete full-value domain,
population-perfect retrieval, robustness to noise, N=3 retrieval, or a universally
minimal amount of binding information. No such extrapolation is needed for the
reported positive control. A domain-wide **no-reset encoding branch** is verified
separately; that branch property is not a global-injectivity proof.

## What was independently checked

`independent_review_check.py` completed with exit0. It preserves its outputs in
`INDEPENDENT_CHECKS.json` and `independent_review.log`.

- All768 input rows regenerate exactly from unmodified `TaskGenerator`, one
  continuous RandomState2026110211, with256 base histories and the two specified
  swap variants. No data-dependent history exclusion or replacement occurs.
- All2,304 prediction records are present:256 histories ×3 variants ×3 arms,
  giving4,608 role outputs. The768 physical resource rows correspond one-to-one
  with history-variants and are shared by correct/wrong decoder arms.
- Protocol, script, input and frozen-model hashes match the result. The imported
  `variable_key_test.py` coefficient helper also matches its previously recorded
  SHA256, separately captured by this review. The helper is a reproducibility
  dependency and must remain in the deliverable.
- An independent exact Fraction100-step block formula, without importing the
  primary coefficient implementation, reconstructs every input's state. Its
  maximum difference from saved scalar encoding is3.33e−16.
- An exact rational, per-step upper-envelope calculation for all ten slot pairs
  verifies maximum candidate voltage0.5970423517197371<1 and gate<5. Legal values
  have nonnegative gate drive≤1.7, content≤5.5, and gate≤envelope; even negative
  content cannot exceed that envelope current. The linear membrane recurrence
  preserves the upper bound. Thus all legal histories in this fixed N=2 domain
  have no reset/clipping during encoding. Equal upper-envelope keys are a
  conservative relaxation, not a sampled legal minimum-spacing history.
- All1,536 correct/wrong inverse records were independently reconstructed using
  a separately expanded quadratic and `numpy.roots`, filtering by domain and
  state consistency without consulting true values. Validity decisions agree;
  the largest prediction difference is1.61e−12. This alternate calculation uses
  the whole-domain branch bound above rather than a truth-based branch choice.
- All score cells were reconstructed directly from saved predictions, including
  unconditional max-operand error, coverage and both tolerances. Direct sum and
  difference composition errors also match the max-operand score. The2,000
  paired-history bootstrap samples reproduce the recorded percentile interval.
- M0 outputs equal the exact buffered nearest-key values. Query swaps exchange
  role outputs/statuses. Value swaps have independently regenerated targets and
  newly encoded states; the review makes no false assumption that state should
  be unchanged by a value swap.

An initial review-only attempt used a deliberately loose block convex envelope
bound and failed to certify it below1. That was a limitation of the coarse bound,
not a primary trajectory failure. Its log is retained as
`independent_review_initial_coarse_bound.log`; the final check uses exact
per-step envelope arithmetic. No primary run was repeated or modified.

## Recomputed original-variant outcomes

| Arm | Coverage | E_comp | Valid joint delivery, strict .5 |
|---|---:|---:|---:|
| Correct key/slot metadata |1|5.0082281e−13|1|
| Swapped key/slot metadata |.75|2.208525699|.0703125|
| M0 exact key–value buffer |1|0|1|

Wrong metadata:192 valid decodes,2 no-real-root and62 no-valid-root cases. Failed
cases retain fixed zero fallback in all unconditional scores. Query-swap cells
match the original metrics exactly. Value-swap wrong-metadata coverage is
193/256, E_comp2.320356486, with1 no-real-root and62 no-valid-root cases. Correct
metadata and M0 succeed for every tested variant. The maximum correct-metadata
operand error over all768 variants is1.562e−12.

The original-history paired wrong-minus-correct mean error is2.208525699, with
fixed2,000-replicate percentile95% interval[2.071585705,2.345021850]. This interval
uses256 independent base histories, never treats the three dependent variants
as768 independent observations, and measures this frozen algorithm contrast.

## Leakage and interface audit

The inverse receives only `state`, `keys`, `positions`, and `queries`. Its
quadratic, domain filtering, branch replay and root selection never consume
true event values, target indices, trial identity or evaluator target values.
True values enter the encoding and later scoring, as expected. M0 is explicitly
given a key–value buffer and therefore has a different memory resource contract.

The harness supplies the two role queries together in one call. However, root
generation/selection is independent of those queries; each query is used only
for final matching to the supplied keys. Thirty-two bounded interface tests
passing only one query reproduce exactly the corresponding paired-call result
and status. Thus the implementation can realize independent role queries under
its already-declared external all-key metadata contract. The paired call is not
evidence that an unreported second cue is needed to solve the state inverse.

Wrong metadata swaps keys between the same chronological positions. This alters
both the assumed encoding map and query-to-slot lookup. Accordingly the contrast
shows reliance on correct externally supplied binding metadata; it does not
isolate a particular intrinsic neuronal binding mechanism or prove that every
possible state-only decoder would suffer this measured population error.

Stable roots, negative-discriminant rejection, both-coordinate bounds,
limited numerical snapping, root merging and candidate replay follow the frozen
rules. No numerical snaps or merges occurred. No target-directed root choice or
post hoc clipping was observed.

## Resource qualifications

Correct and wrong arms share the same physical encoding:1,000 steps/100ms,
three-coordinate carrier state, no physical query probe, direct full-state
snapshot. Both retain two external key scalars plus two slot indices, exact model
constants and external nonlinear computation. Candidate replays are computational
validation work, not extra biological probe trajectories. The result records768
correct-arm and577 wrong-arm candidate replays separately. M0 retains two keys,
two values and position metadata, performs lookup, and does not require a neuron.

Per-history ∑A h, ∑|Isyn|h and ∑u²h are dimensionless physical-model proxies, not
ATP or a complete hardware energy account. CPU time and process high-water memory
include Python/reviewable solver overhead and cannot establish a neural resource
advantage. Global constants, lookup metadata and the external inverse remain
part of the successful interface.

One minor range-label qualification: saved `ranges.u` and `ranges.A` are **post-
step** extrema and omit the initial state at t=0. In particular an A minimum may
be positive when the first slot is active, although the complete trajectory
including initialization has minimum0. Initial state0 is already explicit;
including it changes no score, integrated pre-step resource, peak, bound or
branch decision. Label these as post-step ranges in summaries, or derive a
separate complete-trace minimum by including0 rather than rewriting prior rows.

No remaining blocker for this independent check was identified.
