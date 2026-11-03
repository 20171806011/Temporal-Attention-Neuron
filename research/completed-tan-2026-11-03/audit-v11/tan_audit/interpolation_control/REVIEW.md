# Independent review of the two-register interpolation control

The completed experiment supports the narrow constructive claim: an explicit online nonlinear writer can replace the post-encoding two-key/two-value cache by two affine-function coefficients and answer either promised clean N=2 query. No material target leakage, hidden query-time key cache, scoring error or violation of the frozen scenario budget was found. This is a different input/write/control/readout mechanism from TAN, not evidence of a biophysical two-state neuron or equal noise/energy resources.

This review did not change the protocol, experimental source, inputs or primary results. `review_check.py` and the separate `range_precision_check.py` are independent checks over the existing data, not new populations, hyperparameter sweeps or fitted decoders.

## Reproduction of records and scores

The protocol and script hashes recorded by the main run match their current files. The protocol hash is `d1a76b593ef6887a4750847f144a8006cb671e18cbb35a0bada94af0e999b9f6`. I regenerated all512 original TaskGenerator histories and their1,024 query/value-swap variants with the declared continuous RandomState, and checked all18 adversarial histories against actual floating key spacing and value bounds. The full input record count is1,554. Original and variant target values match their query keys in the original arrays.

The noise ledger exactly reproduces the declared seed with the implemented batch order: all530 four-component state draws, then530 two-key draws, then530 two-query draws. There is no fitting or noise resampling. Two histories' diagnostic variants are not independent new samples; the adversarial set is separate from the512 IID histories.

All14,166 paired prediction records (28,332 role outputs) are present exactly once. This is530 original/adversarial histories ×19 arm/scenario combinations plus1,024 clean dependent variants ×4 arms. I independently recomputed every field in all46 metric cells, including unconditional MAE/RMSE/Ecomp, coverage, strict validity-aware joint delivery, local success, status counts and raw finite-only loss. Maximum metric discrepancy was zero. All invalid scored outputs are the declared zero fallback; they remain unsuccessful valid-delivery outcomes.

## Query information and controller audit

`OnlineWriter` retains only x,y,phase and a static dtype selector. At the first event x,y hold the raw key/value; the second event overwrites both with a,b. Old values appear only as local combinational operands during that atomic update. The evaluator's transition ledger retains past states for audit, but the query kernel never receives or reads it.

The query kernel has arguments(state,q,dtype), and its source names are confined to these variables, its local cast and `float`. Its result is solely a*q+b. It does not access original keys, event positions, target indices or values. Target arrays and the reference line are read only after predictions for error/bound verification. Thus the constructive query-only result is not a disguised key-cache lookup.

The writer has three ordinary phases(empty,first,ready) plus an error phase for a singular write; all four fit the declared two-bit controller budget. Phase is a real resource. The random event mask and position schedule do not eliminate the need to decide which write law applies. The event-onset strobe, trial reset, external clock, separate raw key/value ports, nonlinear division and atomic overwrite, arithmetic temporaries and ideal analog readout are all disclosed. Native TAN's content/novelty interface does not supply those same separately recoverable raw ports for every key.

The two-register claim refers to persistent real payload after encoding; it is not a count of every physical component, every transient wire or every controller state. Timing and write control occur once at each event onset, unlike100-step native current integration. The query decoder cannot validate whether an arbitrary off-contract query was one of the two original keys, because the coefficient representation does not retain that set. The clean task does promise membership, so this is an important scope limitation rather than a failure on the evaluated task.

## Independent numerical and mechanism checks

For every interpolation record I derived the exact ideal line from Fraction representations of the original binary64 inputs. It satisfies both endpoint identities and support bounds |a|<=120,|b|<=297. Independent rational coefficient/decay calculations agree with stored states to maximum relative residual6.03e−13 across the corresponding non-float32 conditions. This check uses the determinant form for the intercept, rather than importing the experimental writer.

All3,180 raw coefficient/query/write-noise or common/differential-leak bound checks pass. The largest residual in the exact common/differential leak identity is4.22e−14. Bounds are applied to raw affine outputs before device-range rejection, as registered; they do not bound a later zero-fallback error. The perturbation cross term is retained. Write-key noise is measured in key units and requires the declared separation denominator .05−2εk, rather than being equated to coefficient noise.

I independently verified2,084 clean/binary32 precision records with exact rational coefficients, actual rounded stored coefficients and queries, and the final readout-rounding term. For first-record leakage, a separate closed-form decay matches the state before the second write; the designed valid-input key pair yields the intended WRITE_SINGULAR status. Common coefficient leakage scales both large cancelling terms equally; differential leakage does not. First-record leakage changes the division denominator and is not covered by the coefficient-leak bound.

All M0 caches and their raw nearest-key outputs were independently reconstructed from the input/noise ledger. In the M0 float32 condition, cache entries and queries are cast to binary32, while the nearest-distance calculation is performed on float64 arrays. This is a precision-scope detail, not a demonstrated numerical defect: each clean query is exactly the same cast key stored in the cache and keys remain far apart relative to rounding, so selection is unchanged here. Do not call it an end-to-end binary32 implementation of every M0 arithmetic operation.

The existing modified native analog control was checked on34 existing histories using an independent100-step block encoding recurrence plus direct query recurrence. State agreement is within8.28e−15 and output agreement within2.78e−16. Its reused original calibration is explicitly a frozen diagnostic, not an optimized readout for the modified carrier. The other modified-carrier control is explicitly privileged by stored keys/positions and full state; its good performance must not be grouped as another query-only method.

## Essential dynamic-range and precision qualification

The fixed support-tight range guard is an engineering policy. It is not the same question as whether the computed affine output is accurate. The registered raw and fallback scores correctly preserve that distinction, but a report that showed only fallback loss would be misleading at adversarial endpoints.

| Condition | Raw maximum error | Raw mean Ecomp | Guarded mean Ecomp | Rejections |
|---|---:|---:|---:|---:|
| IID interpolation float32 |3.96399e−6|2.93049e−7|2.93049e−7|0/512|
| Adversarial interpolation float32 |4.44089e−16|3.94746e−16|2.66666667|16/18|
| Adversarial M0 state noise |.00281732|.00203196|2.66687945|16/18|
| Adversarial M0 write/query key noise |0|0|1.66666667|10/18|

All16 minimum-spacing float32 interpolation cases exceed the declared slope range slightly. Some also exceed the intercept range. Nonetheless their actual binary32 multiply/add returns ±3 at the tested endpoints, matching targets to4.44e−16. In one high-key example the exact real evaluation of the rounded coefficient line would have about1.14e−5 error, which the final binary32 operation happens to cancel at that endpoint. Therefore neither the raw endpoint coincidence nor the fallback score proves a general float32 robustness/failure claim.

Boundary noise similarly pushes some M0 keys or values beyond the nominal support, although unguarded nearest-key retrieval still works. No posthoc range relaxation is recommended for the completed registered run; additional range headroom would be another stated assumption. The raw versus guarded comparison must remain visible. The same applies to interwrite leakage: raw IID loss is2.96080 with maximum527.679, while the range/fallback policy lowers scored loss to1.69115 and maximum51.220. That is a fallback effect, not restored accuracy.

See RANGE_PRECISION_REVIEW.md and RANGE_PRECISION_CHECKS.json for all endpoint cases, coefficient overshoots, raw/guarded comparisons and leak details.

## Inference limits

The construction is not redundant with M0's representational accounting: it demonstrates that separate per-key records need not remain after encoding for this promised two-query membership task. It is redundant with M0's ideal accuracy ceiling, which was already zero error. That distinction is the useful result.

No claim of superior biological memory, optimal robustness, equal energy or minimal state dimension follows. Interpolation and M0 have different coordinate scales, register counts, arithmetic and controller assumptions. Equal fractions of their fullscale ranges do not imply equal physical noise, signal-to-noise ratio or energy. Correlated casts of the same original keys at write and query time are not independent key-measurement disturbances. First-record and coefficient holding are ideal except where explicitly perturbed; no phase-control noise, event-detection errors or complete physical implementation is simulated. The fixed centered coordinate observation is analytical only, not a silently selected better arm.

All original labels remain unchanged. No extra parameter sweep, source edit, publication or dependency installation occurred. No blocker remains for this review.

Completed command: `PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/interpolation_control/review_check.py`, exit0. Supporting results: REVIEW_CHECKS.json and review_check.log.
