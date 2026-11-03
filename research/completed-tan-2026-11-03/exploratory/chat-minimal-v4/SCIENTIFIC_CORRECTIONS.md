# Scientific correction history

The frozen grid, source learner, common certificates and numerical result file were not changed after execution.

The first separate-path checker reached final report construction and raised an AttributeError because a loop variable shadowed the argument-parser object. No certificate assertion had failed. The parser variable was renamed to `args`; no numerical check or acceptance rule was changed. `VERIFIER_BEFORE_FIX.sha256` records the historical pre-fix verifier hash.

This was a checker/reporting defect, not a successful first verification. The corrected execution is recorded in `verification_v4.json`. The operational traceback is omitted from the public copy; this account preserves the failure and the scope of its correction.
