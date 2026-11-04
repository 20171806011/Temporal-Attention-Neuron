# Clean-only response collision: reviewed supplemental evidence

This supplement resolves the evidence gap identified in the first manuscript version. The source is the separately supplied **TAN-Chat-clean-map-v1-reviewed-2026-11-03.zip**, not part of the original 680-file snapshot. Its SHA-256 and the selected unchanged scientific-file hashes are in [provenance.json](provenance.json). All 23 entries in the original inner package manifest passed before selection. Only necessary scientific evidence is published here.

## Verified conclusion

For clean inputs `x5=a`, `x9=b`, all others zero, the permitted response is exactly

`F = (T, R(T)/2, R(T)/4, R(T)/8, R(T)/16, 1[T>1/2], 0, 0, 0, 0)`

where `T=u9` is the **pre-reset** candidate and `R(z)=z` for `z<=1/2`, otherwise zero. The full internal state still contains the ordered W5 window. The cue `b` is not separately supplied to the decoder.

Compare `(a,b)=(3/4,1)` with `a=17/16` and a cue near `0.9406837741`. Both targets lie in exclusive, different class supports. Exact rational endpoint signs and continuity certify a real cue giving the same full response, with no target/query reset on either history. A second certified pair uses `a=5/4` and a cue near `0.9508824513`, crossing the target reset branch. Historical unshifted and native stabilized softmax have **separate** certificates because their epsilon terms differ.

The equality is at the real root defined by the intermediate value theorem. Printed decimals are not exact roots; small floating residuals are not the proof. This rules out a perfect decoder for every allowed clean input under the stated real-valued observation contract. It does not calculate Bayes accuracy, establish a one-third ceiling, or quantify the share of historical errors caused by information loss. The independent cue changes across the collision pair; a decoder given the exact cue has an additional resource.

## New verification in the publication environment

Run from any directory:

```sh
python3 verify_independent.py
```

The standard-library-only checker imports no archived primary or verification code. It uses new scalar recurrence code and exact Fraction exponential bounds with range reduction, a 40-term Taylor bound, and four squarings. It checks:

- pinned original model, simulator and task Git blobs;
- all four broad and saved tight collision brackets and target/query branches;
- both saved target reset-boundary brackets;
- all 64 saved fixtures, 896 full recurrence updates and exact floating tail factorization.

[verification_independent.json](verification_independent.json) records **5,482 checks passed**, with maximum saved-versus-new scalar discrepancy `3.3306690738754696e-16`. This is new computational verification of the prior Chat-sandbox evidence, not independent human replication, a formal proof-assistant certificate, or a new classification experiment. It does not independently regenerate the local derivative-rectangle or global monotonicity certificates. Those are unnecessary for the existence proof: continuity with opposite endpoint signs is sufficient.

## Original evidence and timing

[THEORY.md](THEORY.md), [SOURCE_REVIEW.md](SOURCE_REVIEW.md), [frozen_spec.json](frozen_spec.json), and [results.json](results.json) are unchanged selected scientific sources. Their statements that the clean/mixed training intervention was unexecuted describe the earlier clean-map stage. The later completed local training control is separately published in the 680-file snapshot and discussed in the updated manuscript. No old conclusion is silently relabeled as a new experiment.
