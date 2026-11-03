# TAN concept branch: v4 minimal scalar moments

**Safety/meaning boundary first:** finite-precision guarantees are conditional on a valid sensor error bound, the external minimum separation promise for all future distinct canonical identities, correct count/history provenance and absence of unbudgeted holding drift. This is not general noisy membership. The package is not original TAN/P1-B/v11 execution and does not change any of their conclusions. These are AI-assisted Chat-sandbox experiments, not independent external verification. See PUBLIC_REPRODUCTION.md.

## What was isolated

A single fixed comparison on the exact prior-v3 12 requests x 3 conditions:

1. redundant moment payload, common root/value certificate;
2. minimal moment payload, exactly the same certificate/budget;
3. explicit key/value table, the same noisy observations/address promise.

Both moment methods deliberately remove the extra Hankel spectral gate and high-order consistency check. Both retain the v3 OLD bound with factor L, including the minimal method whose highest value exponent is only L-1. No training, tuning, additional grid or selected successful seed.

The specification and source were hashed before the first v4 execution. Existing v3 outputs are only cited as a saved report, not re-executed. The included v3 source is a byte-preserved reference, not the v4 runtime dependency.

## Main result

| Condition | old v3 moment count (reported only) | redundant/common | minimal/common | table |
|---|---:|---:|---:|---:|
| separated q24 | 9/12 | 9/12 | 9/12 | 9/12 |
| clustered q24 | 4/12 | 4/12 | 4/12 | 9/12 |
| clustered q40 | 6/12 | 6/12 | 6/12 | 9/12 |

The two new moment methods have identical accessed prefixes, budgets, root/value certificates, decisions and read outputs on every event. The principle is proved by induction in THEORY.md. Payload deletion saved mutable storage, not coverage.

Deleting v3's spectral/high-order gates did not improve acceptance on this set. Two clustered-q24 refusal labels changed from spectral failure in the saved v3 report to content-budget failure under v4, but the requests still failed. This does not prove the old gates never matter on other data.

Max accepted latest errors for either moment method were 7.212149771188065e-8, 8.864910013865092e-6, 1.830762345101464e-9. Max other-key retention errors were 9.247833249238679e-8, 1.1282913434162225e-5, 1.830762345101464e-9. Table errors were zero for these exactly representable dyadic values, not a general real-value claim.

Across the new 108 transactions: 65 commits, 43 refusals, every refusal bitwise unchanged including count and budgets. All input observation offsets were nonzero. Distinct requested versus committed targets and missing admissions are retained in full records and refusal reports; returns of stale committed values are not completed latest writes.

## Reproduce

Python 3.13.5, NumPy 2.3.5, SymPy 1.14.0 were used. No random generator or seed: all observations are deterministic rational offsets.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python minimal_moment_v4.py \
  --spec frozen_spec_v4.json --out my_results_v4.json --timing-out my_timing_v4.json
python verify_v4.py --spec frozen_spec_v4.json --results my_results_v4.json \
  --out my_verification_v4.json
```

Reference correctness JSON SHA256:
`8ba2c43b96f038635af8f3cec3d48b6858f9eadfdc20e4b67b7a203203624b06`.

The two same-environment executions produced this identical JSON; wall-clock timing files naturally differ. A different library/platform can change root proposals or recorded environment strings; use the exact-rational checker rather than assuming cross-platform byte equality.

The separate checker imports no learner and recomputes interval operations, transitions/budgets, schedule, truth comparisons and snapshot bytes. It passed 11,360 assertions after fixing a final-report variable-shadowing bug. This is same-agent separate-path checking, not external independent reproduction. See SCIENTIFIC_CORRECTIONS.md for the preserved failure and correction history.

## Files

* `frozen_spec_v4.json`, `SPEC_INITIAL_SHA256.txt`, `SOURCE_PRE_RUN.sha256`: pre-run scope and hashes.
* `minimal_moment_v4.py`: self-contained learner, scorer and exact minimal fixture.
* `results_v4.json`: full transactions, candidate certificates, queries and state bytes.
* `summary_v4.json`, `comparison_v4.csv`, `refusals_v4.json`: extracted results, unchanged-state refusals, unfinished requested targets.
* `verify_v4.py`, `verification_v4.json`: independent-path checker and actual output.
* `timing_v4.json`, `timing_rerun_v4.json`: first and second host measurements, kept out of deterministic results.
* `THEORY.md`, `RESOURCES.md`, `SOURCES.md`: exact proof, conditional safety and honest resource ledger.
* `v3_reference_*`: byte-preserved prior sources/report, not a rerun.
* `preservation_v4.json`: historical preservation checks of 20 prior artifact files, including old archives; not replay dependencies.
* `HISTORICAL_MANIFEST.sha256`: original package hashes; `PUBLIC_SHA256SUMS.txt`: current public file hashes.

Core payload: redundant 11, minimal 6, table 6 real-valued words at L=3. Including count/budgets: 15, 10, 8 signed-int64 words. Candidate state, rollback snapshot, temporary decoded root lists and variable-precision interval workspace are additional. Matching 2L core scalars does not imply equal total bits, reconstruction time or energy.

The construction removes redundant persistent moments while retaining exact intrinsic root membership, including at capacity. It does not remove key information: keys are decoded transiently. On this fixed finite set it did not beat the explicit table.
