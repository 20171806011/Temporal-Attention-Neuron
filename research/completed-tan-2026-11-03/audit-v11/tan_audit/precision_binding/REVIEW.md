# Independent review of the fixed persistent-precision binding test

Reviewed 2026-11-02. **Pass within the declared ideal storage/arithmetic contract.** No primary input, script, protocol, result or historical repository file was changed. The reviewer independently reconstructed every input and every prediction without importing `run_precision.py` or the repository generator into the checking script.

The executed command was:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/precision_binding/independent_review.py > tan_audit/precision_binding/independent_review.log 2>&1
```

It completed with **exit 0** in 5.44 seconds on its final execution. The separate reviewer script was expanded once to add a full rational-arithmetic online oracle; both executions passed. This was additional verification, not another experimental run or a changed condition. Evidence is in `INDEPENDENT_REVIEW_CHECKS.json` and `independent_review.log`.

## Coverage and data contract

- Independently reproduced the repository generator's exact random draw order using one continuous `RandomState(2026110215)`, including 21 rejected key draws. All 512 fresh base histories and their query/value variants match exactly.
- Reconstructed all 18 frozen adversarial histories and variants. All keys/values are in the generator's half-open numerical ranges, all key gaps are at least .05, and all event positions and queried targets are valid.
- Verified 1,590 input records, 6,360 prediction rows and 12,720 role outputs. No exclusions, invalid cases, saturation, singular writes or numerical boundary-band cases occur.
- Recalculated all 24 group/variant/arm metric cells and the three fixed 2,000-replicate paired bootstrap comparisons. Only the 512 fresh original histories enter those bootstrap intervals. Query/value variants are dependent controls; the adversarial cases are reused support tests, not prevalence estimates.
- Verified all input, protocol and implementation hashes recorded in `results.json`. The adversaries include repeated key/value conditions with different timings; equal outcomes are expected under perfect retention, not additional independent evidence.

## Online state and arithmetic

Source inspection verifies `Writer.__slots__ = ('arm','phase','words')`. The arm identifier selects fixed shared constants. Matched-arm retained payload consists only of integer codes: two 16-bit words for RAW32/PHASE32 or four 8-bit fields for CACHE32. CACHE32 holds two fields after the first write. The second interpolation write uses the dequantized first codes and current input, then overwrites the two words; it does not consult a retained exact first record.

`query(self,q)` reads only the final words, current query, fixed arm constants and phase. The independent reconstruction uses a separate query function with only those inputs. Evaluator traces retain history for audit, but the writer never consults that trace. ENDPOINT_ONLY explicitly holds two binary64 values between events, so its peak payload is 128 bits and it is not a matched 32-bit competitor.

All 22,260 quantization events were checked against the specified endpoint-inclusive grid and binary64 scaled nearest-even rounding. A second check used exact Fraction arithmetic on the binary64 input/range endpoints throughout every online write. It reproduced the same final integer payload for **all 6,360 rows**. There were also zero individual rounding-code disagreements between binary64 scaled rounding and exact rounding of each logged raw value. The maximum decoded-value floating residual against its exact rational grid level was 5.69e-14. These are numerical checks, not additional usable bits.

All resource records and summaries were independently recalculated. Mean occupied payload bit-ms on fresh originals are RAW32/PHASE32 2581.25, CACHE32 1920.625, and ENDPOINT_ONLY 6545.0; allocated peak payload and controller bits agree with the protocol. Bit-ms is logical occupancy, not physical energy. Python objects themselves consume more than these logical bit budgets. The reported 64-bit arithmetic is an operand width, not a count of transient registers or total computational memory. No finite-time divider, physical controller or energy model is implemented.

## Bounds and mechanism localization

Exact rational bounds reproduce RAW32 **.258753467543**, PHASE32 **.0118264337303**, CACHE32 **.0117647058824**, and ENDPOINT_ONLY **.00976577401389**. All effective denominators, intermediate coefficient bounds and final errors satisfy their certificates. The cache's nearest-key identity is preserved for every row; its all-domain certificate follows from twice the key half-step being strictly below .05.

The chronological pre-final error decomposition is independently verified. Across the complete ledger, first-target pre-final errors reach .2420468068 for RAW32, .0017400330 for PHASE32 and 8.89e-15 for ENDPOINT_ONLY. Second-target pre-final errors remain below 6.22e-15 in every interpolation arm, consistent with the frozen `b=v2-a*k2` update. The largest decomposition residual is 3.06e-14. Thus the large RAW32 tail is directly attributable to retained-prefix quantization, rather than an unlogged query or lookup error.

The smooth-flow observation in `THEORY.md` is correctly scoped: for a fixed second event, the literal exact coefficient overwrite maps first records onto a line and is noninjective, whereas finite unique smooth flows and the declared nonreset TAN branch are injective. This obstructs identifying that exact overwrite with the existing smooth branch. It does not exclude resets, quantization, additional hidden state, alternative output representations, or all two-state binding mechanisms. The quantitative contraction bound is norm-dependent and is not an energy lower bound.

## Interpretation checks

Fresh-original mean Ecomp is RAW32 .00699701, PHASE32 .00319812, CACHE32 .00787244 and ENDPOINT_ONLY .00314729. RAW32's mean is slightly **lower** than the cache's here, while its maximum error .0655800 exceeds the cache's .0117515 and three of 512 histories fail the new local .05 joint criterion. It would be incorrect to summarize this as uniform cache dominance. The reused adversarial RAW32 maximum .242237 exposes its tail; it is not a distributional risk estimate.

All arms pass the original strict .5 operand-delivery criterion in every tested condition, as the analytic bounds predict. The separate .05 local joint diagnostic is not the original full P1B pass label or a replacement for any original composition-risk criterion. Existing labels remain unchanged.

The literature report appropriately identifies two-point affine interpolation as standard mathematics and distinguishes the algorithm from Hopfield dynamics and fast-weight update rules. This reviewer read that report and local baseline definitions; independent retrieval of the external papers was performed by the literature reviewer, not repeated here. The contribution is this task-specific online precision/resource audit, not a new interpolation method or a novelty claim about all associative memory.

PHASE32's gain is attached to explicitly switched coding and high-precision transient arithmetic. CACHE32 quantizes both arriving values, whereas interpolation uses the current second value before final coefficient quantization; this is a disclosed algorithmic difference. Equal persistent peak bits therefore do not establish equal total hardware, time, noise, energy, or neuronal resources. Neither a minimal-bit theorem nor an optimized cache comparison is established. The missing finite-time neural write/read mechanisms remain a scientific gap, not an execution blocker.

No implementation error was found.

The completed `REPORT.md` and `../CURRENT_RESEARCH_REPORT.md` were also read after the ledger checks. Their numerical tables, fresh-original prefix-error summaries, mean-versus-tail distinction, classical-interpolation attribution, smooth-branch obstruction and implementation limits agree with this review. The 32-bit comparison is correctly presented as a persistent-payload comparison, and all original P1B gates remain outside any success claim for the changed encoder/readout.
