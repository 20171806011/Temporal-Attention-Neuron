# Frozen Probe1 execution notes

2026-11-02. The primary experiment completed on its first execution; no fitted decoder, retry, box change or parameter search occurred. Independent verification is a separate task and is not claimed merely from this primary run.

Completed commands, both exit 0:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/project_reassessment_v8/theory_certificates.py > tan_audit/project_reassessment_v8/theory_certificates.log 2>&1
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/project_reassessment_v8/run_probe1_collision.py > tan_audit/project_reassessment_v8/probe1_execution.log 2>&1
```

The rational theorem certificate ran before the numerical fixtures. The numerical command took 3.029 s internally and reported 125,648 KiB maximum RSS. These are software observations, not biological time or energy.

## Complete counts and outcomes

The fixed ledger has 545 paired cases: one anchor, all 32 box corners and 512 RandomState2026110225 draws. Each case has two opposite-label histories. There are 4,360 primary native trajectories across B1–B4, 1,090 trajectories for the one reset intervention, and 4,360 additional vector-reconstruction checks of the native implementations. All histories have 14 steps. No case was excluded.

| Arm | Literal-equal full response pairs | Analog response gap range | Pairs with different allowed spike responses |
|---|---:|---:|---:|
| B1 LIF | 545/545 | 0 | 0/545 |
| B2 delay line | 0/545 | .0486667–.0513333 | 0/545 |
| B3 no-attention gate | 0/545 | .2413112–.2497250 | 545/545 |
| B4 native TAN | 545/545 | 0 | 0/545 |
| B4, suppress only t8 reset | 0/545 | .0143386–.0159234 | 0/545 |

The analog gap is the largest absolute difference among the five allowed pre-reset voltage features. B2 and the reset intervention separate the analog observation but do not separate the allowed spikes; neither result is a trained classifier accuracy. B3's distinction is a diagnostic of this selected family, not evidence of general B3 superiority or a population-performance comparison.

For native B4, all 1,090 histories reset at t8, have strictly positive query surprise, and remain below threshold at the query. The observed exact feature equality is stronger numerically than the tolerance fallback, but the ideal equality theorem is proved separately rather than inferred from a finite floating-point sample. The ordered windows differ at t9 and t10 and match from t11; the target-tap oracle identifies all 1,090 labels on the stipulated disjoint supports.

Anchor native B4 responses begin at u9=0.33538361861604077 and follow exact powers of one half with five zero spikes. The anchor t8 candidates are .5865734056224717 and .6168296817783244, both reset to0. Under the one-step reset suppression, the query candidates become .6286703214272766 and .643798459505203; both query spikes are1, and the later candidates are0. This intervention therefore recovers a small analog distinction but no spike distinction.

All native outputs were also reconstructed by a separately instrumented vector recurrence. The largest residual is below1e-14. Intervention scope checks confirm identical pre-t8 dynamics, unchanged t8 threshold event indicators, identical current/surprise/attention throughout, and unchanged reset rules at every other step. The intervention deliberately permits an above-threshold retained h8 and is not the original physical reset policy.

## Exact certificate and small scope of the bound

Fraction-only degree40 exponential intervals certify a uniform t8 current lower bound above .528613674287, hence a margin above threshold exceeding .028613674287. The query has 0<u9<.4432<.5 throughout the box. The complete ideal response identity and the different target labels imply a conditional equal-mixture classification error of at least1/2 on the mirrored box. Each orientation's original present-distractor conditional probability mass is exactly1/312500000. This is a tiny analytical bound, not empirical collision prevalence or the original balanced-three-class error.

Two nomenclature clarifications leave protocol parameters and results unchanged:

- In the protocol's short C8 statement, the explicit bound with `-1e-9` is the numerator of `(sum(exp)+1e-9)*(C8-1)`, not of `sum(exp)*(C8-1)`. The executable rational certificate uses the full denominator correctly. The original preregistration text/hash is preserved.
- The result fields named `primary_euler_steps` and `validation_euler_steps` count 76,300 and 61,040 discrete updates. TAN-I here is the frozen discrete recurrence; these field names do not assert a new continuous-time Euler integration scheme.

## Reproducible files

- `probe1_inputs.jsonl`: all545 paired input records, parameters, labels, latent classes and full input streams. Saved/hash-printed before numerical model execution.
- `probe1_trajectories.npz`: all full14-step arrays and windows, every arm's complete ten-feature responses, and the target-tap oracle.
- `probe1_pair_ledger.jsonl`: all2,725 pair/arm rows, full features, margins, labels, equality checks and target/window differences.
- `probe1_results.json`: all checks, arm summaries, counts, runtime and provenance.
- `theory_certificates.py`, `theory_certificates.json`, `theory_certificates.log`: exact proof and its execution evidence.
- `run_probe1_collision.py`, `probe1_execution.log`: primary numerical implementation and completed output.

Protocol SHA256: `d8f0b97300f7fd1d3e3b0c07cd5ca17c3cfb4b40b20efa25792e6ab3dd78df23`.

Input SHA256: `1773c6bd6ee595fe1049b71ff6b8e34a3e7c9007f5de7f509d51d8536117fcf4`.

Primary script SHA256: `697a6640b9cfc3cfc8959f9841ead9e88701416642614d752a3b7860bc40e725`.

All frozen source hashes remained unchanged.  The original labels and primary artifacts are preserved for independent review.
