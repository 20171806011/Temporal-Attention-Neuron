# TAN research experiment archive — 2026-11-02

This public scientific snapshot preserves eight completed, bounded experiments, their scientific source and protocols, reports, JSON results, and CSV target/trace/prediction ledgers. Start with `EXPERIMENT_SERIES_SUMMARY.md`, then each experiment's `REPORT.md` and `PROTOCOL.md` where present.

The principal result is conditional: modified content-dependent gating, a longer gate timescale, a nonzero query probe, two analog samples, and a known-equation nonlinear inverse substantially improve held-out performance on fixed known keys and a narrow target prior. This is not success of the original TAN system, variable-key retrieval, N3 retrieval, biological decoding, or evidence of network necessity. Original fixed readouts have decoder floors and the original carrier has mechanism limitations; see the summary for numerical results and caveats.

## Reproduction and verification

Run from this directory. Python, NumPy and SciPy are required by the experiment series. Scripts resolve dependencies relative to their own locations. Preserve the experiment order shown in the summary. Commands overwrite their result files; use a separate copy when rerunning. The root model/generator/runner and calibration files are frozen evidence/dependencies; this archive does not claim to reproduce the original production benchmark. The original runner is a reference dependency, not an entry point for this eight-stage series.

This is AI-assisted research. The analytic and computational cross-checks called “independent” in historical reports refer to separate calculations or implementations, not independent human review or external replication. This publication packages already completed experiments; it does not rerun them or validate new scientific claims.

`PUBLIC_SHA256SUMS` verifies the public bytes from this directory. Files named `HISTORICAL_*` preserve original recorded checksums with portable path labels. Some report/protocol text was edited for privacy and portability, so historical hashes are evidence of the original run, not a promise that every public text file has those bytes. Numerical results and scientific code are unchanged. The top-level publication manifest records current public hashes.

## Scope and exclusions

Included experiments: oracle-validation, finite-dataset, paired-mechanism, collision-witness, content-gate-rank, two-sample-timing, nonlinear-heldout, timescale-heldout. Selected root scientific protocols and independent review are included for interpretation.

Excluded: assistant/runtime/authentication material, browser/session logs, screenshots, internal working notes, caches and credentials. This archive contains no new experiment or subsequent tuning sweep. Random/unknown keys/timing, larger value sets, noisy encoding, biological decoding, physical implementation and throughput costs remain open research boundaries.
