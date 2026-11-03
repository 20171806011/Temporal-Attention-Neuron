# Correct versus corrupted key/slot metadata: fixed-budget positive control

The preregistered test completed on its first run, exit0. Exactly256 fresh original-generator N=2 histories were evaluated, each with original, query-swap and value-swap variants, under three arms. There are768 history-variants per arm,2,304 paired-prediction rows and4,608 role predictions. Only the256 base histories are independent draws. All generated histories remain in the denominator; no filtering or resampling after generation occurred.

ProtocolSHA256 `0ce7a191e731404f5297b7f4177e776bf775dff325105f3a81e1c847fc5c3ed1` preceded input generation. The full input ledger and hash were saved before encoding and metrics. Seeds: continuous RandomState2026110211 for histories;2026110212 for fixed2,000 paired bootstrap replicates.

## Result

| Arm | Variant | Coverage | Unconditional Ecomp | Valid joint success, strict .5 | Status counts |
|---|---|---:|---:|---:|---|
| Correct ordered keys | Original |1|5.00823e−13|1|256 VALID|
| Correct ordered keys | Query swap |1|5.00823e−13|1|256 VALID|
| Correct ordered keys | Value swap |1|5.03925e−13|1|256 VALID|
| Swapped key/slot association | Original |.75|2.20852570|.0703125|192 VALID;2 NO_REAL_ROOT;62 NO_VALID_ROOT|
| Swapped key/slot association | Query swap |.75|2.20852570|.0703125|192 VALID;2 NO_REAL_ROOT;62 NO_VALID_ROOT|
| Swapped key/slot association | Value swap |.75390625|2.32035649|.03125|193 VALID;1 NO_REAL_ROOT;62 NO_VALID_ROOT|
| Exact M0 buffer | All three, separately |1|0|1|256 VALID in each|

Wrong-minus-correct unconditional Ecomp on original histories is2.20852570; paired percentile95% interval[2.07158570,2.34502185]. This is the fixed primary contrast, not a simultaneous familywise interval. Strict.05 local joint success for wrong metadata is.01171875 on original/query-swap and0 on value-swap; correct/M0 achieve1 on this sample. No boundary snaps, root merges or correct-arm abstentions occurred. Complete RMSE/MAE, conditional errors and null results are in results.json. All zero fallback cases stay in unconditional losses and fail validity-aware delivery.

Both role queries follow the original clean task, values span[-3,3], keys span[.3,2.5] with spacing.05, and active positions are random among five slots. Query swaps exactly exchange each arm's outputs; value swaps generate and encode a different history, with swapped targets. M0 equals all supplied target values exactly using explicit buffered values, not evaluator target fields.

## What was tested

The candidate carrier uses tau_m10/tau_A150 and added content gate drive eta(v+3), eta=.1. Encoding is100ms/1,000 Euler steps from zero, with native threshold/refractory/clipping logic. The external inverse reads the complete final(u,A,r), both stored keys in chronological order, both slot indices and current queryq. It receives no true values or target indices. It reconstructs both values and matches q to its supplied key metadata to select the requested value. The corrupted arm exchanges key-to-slot associations while retaining the same encoded state, actual query and key set. It therefore perturbs both coefficient construction and role selection; it does not isolate those two substeps separately.

Correct metadata yields a unique legal root in all768 evaluated variants. The inverse uses a stable quadratic solver, both-coordinate domain filtering, separately declared numerical tolerances, polynomial residual checks and a full candidate encoding replay before acceptance. No truth-based root choice occurs. All candidate roots, rejection reasons, features supplied to the inverse and replay residuals are retained in predictions.jsonl.

This positive result is **conditional recovery with additional binding metadata and full-state readout**. It is not the original state-and-query-only interface, not the earlier two-membrane-sample protocol, and not native Count/Latency delivery. The negative metadata result is deliberate model/association misspecification, not a fitted q-only decoder benchmark. Together with the prior ideal same-state/same-q counterexamples, it shows why privileged known-key inversion and general addressing must be kept separate.

These256 successes do not prove global injectivity across all possible full-domain keys/values/timings, robustness to noise, or an N=3 result. No unseen data were used for tuning, no parameter sweep was run, and no full-domain universal inversion theorem is inferred from this finite test.

## Branch and implementation controls

For all10 slot pairs an artificial pointwise upper envelope k=2.5,v=3 gives maximum candidate .59704235172<1. Its duplicate maximum keys are an envelope, not a claimed generator-valid trial. Since A>=0 and actual content<=5.5, gated current is bounded above by5.5 times the envelope gate, even when actual content is negative. Positive convex Euler recurrences then bound every legal history's membrane trajectory; all envelope gates stay below5. This justifies the no-reset polynomial for this fixed N=2 candidate on the tested full domain. It does not prove the inverse globally unique.

Independent vector/scalar encoding comparisons cover16 fresh base histories and all10 envelope pairs. Every decoder candidate is replayed through the frozen scalar Euler implementation using supplied metadata and reconstructed values. Old A drives current; returned spikes and pre-reset candidates are separately checked. Original M0 and query-swap controls pass. There are no implementation corrections or failed primary runs to omit.

## Resource accounting

Each correct/corrupted history uses three carrier coordinates, retains **two external real-valued keys plus two slot indices**, reads two informative analog state coordinates(u,A) plus r, and receives currentq. Exact model coefficients, event timing and floating-point nonlinear computation are assumed known. The snapshot is reused for two independent role evaluations; there are **zero physical query-probe steps**, so the test does not validate a10ms physical readout circuit or infer a latency advantage. The metadata itself is additional memory beyond the original neuron contract.

M0 retains two keys and two values; position metadata is logged but the nearest-key decoder needs only the key–value buffer. It has no neuron trajectory and performs four distance calculations for two queries. Diagnostic histories/targets and solver replay traces exist in the evaluator, not as inputs to the inverse. All comparator data access is explicit.

encoding_resources.jsonl contains768 records of integrals sum(A*h),sum(abs(Isyn)*h),sum(u²*h), candidate/u/A/current ranges, finalstate, actual spikes/clips and physical duration. These are dimensionless model proxies, not ATP or full physical energy. External key memory, state sensing, computation, initialization/reset/carryover costs are not represented by those integrals and cannot be declared free. Candidate replays are computational validation, not additional biological probes:768 correct-arm and577 corrupted-arm replays. Main run used about6.996s wall/6.992s CPU with92,748KiB process high-water memory in this executor; these host costs are not neural costs and are environment-specific.

The stored u/A ranges are post-update extrema and omit the initial zero state; a whole-trajectory range must include zero as well. Integral accumulation does use the pre-update state, including its initial zero. This reporting qualification preserves the recorded primary resource ledger rather than changing its values after execution.

An independent reviewer reconstructed all768 input variants from the continuous seed, every prediction row/metric and the bootstrap; independently inverted all1,536 correct/wrong rows using another polynomial expansion and np.roots; and checked32 equivalent single-query calls. Validity classifications agree and inverse predictions differ by at most1.61e−12. An independent exact Fraction recurrence also certifies the full-domain upper-envelope maximum .5970423517197371. No primary source or result correction was required. See INDEPENDENT_REVIEW.md and its supporting script/results.

## Reproduce and inspect

`PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/binding_control/run_control.py`

Read PREREGISTRATION.md, INPUTS_SHA256, inputs.jsonl, predictions.jsonl, encoding_resources.jsonl, results.json and run.log. The source imports the frozen Euler function, unchanged TaskGenerator and the previous audit's fixed recurrence-coefficient helper; all are included in the deliverable. Original repository and historic outputs remain unchanged.
