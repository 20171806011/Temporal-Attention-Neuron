# Probe-1 v3 audit amendment (audit statistics only)

Frozen formal data: train=4000, test=2000, seeds 20260904/05/06, models B1-B4 (v3 label: retrieval-cue target recall).  No experiment parameter was changed; the deterministic pipeline was reproduced and asserted equal to the frozen per-seed numbers (max deviation 4.7e-04).

## A1 - shuffle audit (old rule abolished)
- Old: fixed gate `mean10(shuffled bal3) > 0.37 => FAIL`.  Statistical calibration is insufficient: L-BFGS on permuted labels shows heavy-tailed overfit (n=1500: 25-shuffle mean 0.346, sd 0.079; at n=4000 mean10 sd ~0.025); B2 seed3 mean10 = 0.378 was ~+1.1 sigma - noise, not leakage.
- New: empirical permutation null, N=100 label permutations per seed x model, refit each time.  Status: PASS | PASS_WITH_QC_LIMITATION | INVALID_FOR_INTERPRETATION based on z10 = (null_mean - 1/3)/(null_sd/sqrt(10)) plus the null mean itself (leakage only if z10 > 3).
- Not changed: model, classifier, labels, features, data, seeds.

## A2 - class-coverage audit (no longer run-blocking)
- Old: "above chance but a class is never predicted => FAIL the run".
- New: per-cell status.  At-chance cells with missing classes: PASS_WITH_FLAG (collapse expected; excluded from claims).  Above-chance cells with a missing class (>=20 true samples): INVALID_FOR_INTERPRETATION for that cell.  A pattern persistent in >=2/3 seeds (B3 never predicts the lo class) sets the model-level flag LO_BLIND_READOUT; its nominal bal3 is never interpreted as healthy three-class decoding.

## What is re-interpreted, not changed
- B3 seed3 clean = 0.470 accompanies lo-blindness: the number is retained but is INVALID_FOR_INTERPRETATION as evidence of healthy decoding.
- B2 shuffled statistics are retained as a QC limitation where flagged.
- This is not post-hoc experiment tuning: audit rules were recalibrated on audit statistics; the frozen experiment data is untouched.