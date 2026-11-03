# P1A evidence review: static addressing is already a strong, scoped baseline

2026-11-02. Frozen repository commit `85213f0d417d54e4f48b301ed1c0e4200ef3c421`. Read-only review and independent arithmetic replay of existing saved ledgers; no new histories, parameter choices, model runs through the production pipeline, or repository edits.

## Decision for the research agenda

P1A's static address-identifiability question is already resolved substantially more strongly than a trajectory-separation claim. It has an exact sufficiency proof, a sharp failure theorem for scalar multiplication, explicit query/value interventions, and complete saved input/output ledgers. This continuation independently reconstructs all 420,000 saved predictions. Another scalar-versus-vector accuracy comparison, query-noise sweep, or nonlinear decoder fit is low value.

P1A and the natural-collision study are **different stages**. The natural-collision experiment is TAN-I Phase1, summarized in `snapshot/paper2/sections/s3_state_geometry.tex:18–62`. P1A is the later static candidate-array/address-query audit in section5 and `p1_experiments/`. A successful static lookup with an externally retained array does not establish sufficiency of a compressed recurrent neuron state; natural separation of futures does not establish successful lookup of a requested payload. The original-model question should be selected from the independent TAN-I source review, rather than re-opening this closed P1A comparison.

## Directly inspected authoritative evidence

The authority is `p1_experiments/results/run_p1a_rev2/`, not the superseded Run1 root report. `P1A_FINAL_BASELINE_MANIFEST.md` records the final freeze, hash manifest and corrected middle-query denominator. The currently read source and output hashes match that manifest, including both compressed ledgers.

| Evidence | Actual coverage and meaning |
|---|---|
| Exact metric proposition | `THEORETICAL_PROPOSITION_1D_METRIC.md:18–48`: distinct stored scalar keys, exact query q=k_t, score −(q−k)^2 and hard argmax uniquely select t. This is nearest-neighbor addressing of an available array, not a dynamical storage theorem. |
| Scalar multiplier obstruction | Same file §§3–4; `models.py:171–189`: nonzero scalar multiplication only selects an extreme key; zero multiplier gives a tie, not unique identification. The signed implementation never uses zero multiplier. |
| Stored resources | `models.py:33–67`: every trial presents the full key/value/mask arrays; hard output reads values[winner], soft output sums the retained values, and addition/subtraction use explicit arithmetic. No recurrent compressed-state storage is being tested. |
| Query dependence and payload controls | Amendment1 §4 and `g1_gate_report.json`: 200 tests each of query swap, value swap, zero payload, equal payload and position permutation for both Metric and VectorQK. Their saved certificates report all passing. This review read those certificates and source contracts; it did not rerun the G1 harness. |
| Leakage negative fixture | Same sources: query-blind N3 baseline near1/6, with a deliberately leaky model rejected. Source `models.py:341–373` makes the cheat explicit. |
| Robustness | Six frozen conditions include two query-noise levels and two OOD key ranges. `task_generator.py:109–115` adds query SD2.2σ, clips below .01, and does not perturb stored keys/values. Thus the experiment is not a general state-noise, storage, write-operation or biophysical robustness test. |
| Soft versus hard behavior | `THEORETICAL_PROPOSITION_1D_METRIC.md:79–97` already proves the finite-temperature error upper bound and states the dependence on separation, temperature and payload differences. This is not an unresolved discovery to rediscover with another sweep. |

The manuscript's source qualifications already distinguish address dimension from circuit-state dimension, explicit combiner arithmetic, sufficiency from necessity, and abstract memory from biological realization. Read those qualifications together with the headline. A broad “single-neuron memory works” statement would exceed this evidence.

## Completed independent ledger check

Command completed with exit0:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python3 tan_audit/project_reassessment_v8/p1a_ledger_check.py
```

The script imports neither production `models.py` nor the evaluator. It reads the existing ledgers, independently forms all seven scoring/selection/readout rules in vectorized arithmetic, joins every output by condition/seed/trial, and checks exact model coverage. Files: [p1a_ledger_check.py](p1a_ledger_check.py), [p1a_ledger_check.log](p1a_ledger_check.log), [P1A_LEDGER_CHECKS.json](P1A_LEDGER_CHECKS.json).

- 60,000 saved condition/trial records; exactly one output per record/model; 420,000 output rows. These are not 60,000 independent underlying histories: C3/C4 deliberately share their 10,000 base histories and noise draws, with different scaling.
- Every hard winner, per-channel correctness and joint flag matches independently reconstructed formulas exactly.
- Hard payloads and hard add/subtract errors match exactly.
- Maximum independent soft-value discrepancy2.40e−14; maximum soft arithmetic-error discrepancy2.58e−14. Different vectorized arithmetic accounts for these rounding-scale differences; declared check tolerance is1e−12.
- All120,000 Metric/VectorQK query winners are identical.
- For each target, its nearest-key Voronoi interval is bounded by the midpoints to its adjacent stored keys, with unbounded outer endpoints. Membership of the observed q in this interval predicts all120,000 Metric correctness flags exactly, including noisy conditions. This is an independently checked explanation of existing outcomes, not a new robustness theorem.
- Actual maximum |.4(q−k)| over all active candidates is0.9600123628<π. Cosine decreases with absolute argument throughout this observed domain, so the hard-routing equality has a direct mathematical explanation. It is not evidence that two independent mechanisms separately solve storage, nor a guarantee for arbitrary query/key ranges or unbounded noise tails.
- C2 has6,705 middle-key role queries: signed scalar gets0; Metric and VectorQK each get6,705. C3 both-routed hits are9,876/10,000, C4 hits7,986/10,000, identical between the two addressing models.

A separate supplemental command, `PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/project_reassessment_v8/p1a_input_field_check.py`, completed with exit0. It independently checked all60,000 stored target-key/value fields, active positions, arithmetic targets and middle-key labels. It also verified that the10,000 C3/C4 histories match before query noise and that their observed noise is the same draw scaled by5, with the stated lower clipping; maximum scaling residual1.34e−15. See `P1A_INPUT_FIELD_CHECKS.json` and its source/log. This confirms the6705 denominator directly from stored keys rather than trusting the saved middle flags.

The bounded check took14.026s, peak RSS207,708KiB on NumPy2.3.5. It does not regenerate the old random inputs from seeds, rerun the original production workflow, recertify every G1 fixture, or reproduce the historical Windows environment. It does independently validate the complete saved hard/soft output ledger against the inspected scoring equations. Source/ledger hashes are saved in the check JSON. No historical file was modified.

## What remains open, and what should not be called new

The exact hard result assumes the entire finite candidate set and its payload binding are available. The soft difference (C2 mean add error .4247 versus .9463) does not establish a dimension-dependent precision advantage: near zero, cos(.4d)=1−.08d²+O(d⁴), so the same numeric inverse temperature does not equalize local score contrast. The existing notes already attribute soft differences to curvature/temperature. Equal numeric noise in these different score coordinates would likewise need an explicit resource interpretation.

The nearest-neighbor noise margin is an elementary consequence of the score rule: a query error smaller than half the nearest-key gap preserves the hard address, subject to the declared tie semantics. Checking that fact again would be a useful implementation check only, not new science. Finite precision of stored keys, long-term retention of their binding, and a biologically implementable write/read process remain untested by P1A. However, opening those resources without an original-model hypothesis would repeat the enriched interpolation detour.

### Candidate priority for the study

1. **Preferred direction: a causal original-TAN-I state intervention tied to the actual output contract.** Select it only after checking the existing natural-collision/reset/geometry controls. The decisive gap is whether a claimed history statistic or trajectory geometry is sufficient for the relevant observable, as opposed to being correlated with it. A test must retain/count the explicit window and refractory variables, freeze common future input, and distinguish an exact output discrepancy from large ratios caused by tiny initial voltage separation. The detailed new intervention should be set by the natural-collision source audit; this review has not established novelty for a particular clamp or task and does not authorize an unregistered execution.
2. **Fallback only if original-TAN-I evidence already covers that gap: a fixed finite-precision storage/readout certificate under an explicitly declared representation and noise contract.** This would concern the P1A retained key/value array, not state-count necessity. The novelty search here found no stored-key noise or finite-precision certificate in the inspected P1A protocol/theory/source. Nonetheless, midpoint robustness and cosine aliasing are standard facts, so a pure restatement is not worth a new experiment. Choose this only if it answers a concrete original-project implementation question beyond the existing precision controls.

No higher-capacity decoder or broad architecture/noise/temperature sweep is recommended. P1A's contribution is the project's internal counterexample to an overly narrow geometry claim, with task-specific audited evidence; nearest-neighbor lookup itself is standard mathematics. Literature framing should use that distinction rather than claim a new universal theory of associative memory.

## Limitations and status

This report uses the inspected repository as primary evidence. It does not claim a complete search of external associative-memory literature or every uninspected local note. One initial text-search command referenced a root-level research-audit plan path that does not exist there; all conclusions above instead rely on the present authoritative protocol, theory, sources and ledgers. That missing optional path is not an input blocker.

Completed: source review, separate saved-ledger arithmetic, scope distinction and next-question recommendation. Selection and preregistration of a subsequent experiment were outside this review.
