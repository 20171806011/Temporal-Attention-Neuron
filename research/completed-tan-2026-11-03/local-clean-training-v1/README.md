# Fixed clean-only versus mixed training control

Completed experiment, archived results from 2026-11-03. This publication was prepared without running new experiments or refitting models.

The original stable-softmax W5 TAN response and 33-parameter linear readout are held fixed. For each of three historical seeds, 1,500 clean-only training examples are compared with 1,500 mixed examples, with 500 examples per class in either arm. Both arms evaluate the same clean test subset. See FROZEN_PROTOCOL.md and REPORT.md for the exact protocol and limitations.

## Result and limits

Equal-seed clean balanced accuracy: 38.1199% (clean-only), 32.5881% (mixed), difference +5.5318 percentage points. The saved conditional paired-bootstrap interval is [+3.3391, +7.6080] percentage points. All six optimizers reported convergence. This is a modest historical-task result, not a fresh blind benchmark, a population significance claim, a spike-only result, or proof of architectural superiority. Per-class effects are uneven. Training-composition effects include induced train-only normalization. Established response collisions remain unresolved.

PRIOR_PROTOCOL.md is the historical proposal, retained as written; its statement that the experiment was not yet executed describes that earlier stage. FROZEN_PROTOCOL.md superseded it before the completed run.

## Files and provenance

- results.json, predictions_*.npz and conditional_bootstrap.npy are byte-identical historical scientific evidence, including optimizer diagnostics, selected training indices, coefficients and normalization.
- FROZEN_PROTOCOL.md and its digest are unchanged. FREEZE.sha256 uses a portable relative filename instead of a private absolute path.
- run_control.py has only its dependency path changed to the bundled upstream_probe1 directory. Its numerical procedure is unchanged.
- upstream_probe1 contains the required archived source, synthetic input arrays and saved reference traces. source/LICENSE is retained. run/failures.json preserves earlier source-uniform failure flags; these belong to the upstream experiment, not the six clean-training fits.
- upstream_probe1/run_probe1.py has only an operational reference removed from its module description. The scientific source and data are otherwise unchanged. Its historical main entry point requires the complete upstream archive and is not the reproduction entry point for this package.
- REPORT.md has publication-only wording/path changes. Operational execution/verification logs are omitted; scientific diagnostics remain in results.json.
- HISTORICAL_MANIFEST.sha256 and upstream_probe1/HISTORICAL_MANIFEST.sha256 preserve all original digest records, including omitted files. They are historical records, not current verification manifests. Historical source hashes inside results.json remain unchanged and must not be confused with the current edited source hash.
- MANIFEST.sha256 and upstream_probe1/MANIFEST.sha256 verify current public bytes. PUBLICATION_CHANGES.json maps included original files to current hashes. No claim is made that publication edits occurred before the historical freeze.

## Offline verification

Requires NumPy. From this directory:

    sha256sum -c MANIFEST.sha256
    python verify_saved.py
    python verify_publication.py

The historical saved-score verifier checks 22 conditions. The additional verifier checks saved inputs, indices, fitted logits, scores, bootstrap percentiles and both current manifests without simulation, data generation, training or network access. Validation is saved-evidence consistency checking, not an independent experimental replication.

## Optional reproduction (not run during publication)

Requires Python, NumPy and SciPy; historical versions are recorded in results.json (Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0). In a disposable copy of this directory:

    python run_control.py

This performs the prescribed six fits and overwrites saved results/predictions/bootstrap evidence. It checks the frozen protocol and dependency manifest first; upstream main is not invoked. Rerun outputs invalidate the current public manifest until deliberately regenerated. There is no bit-for-bit portability claim across numerical-library/platform versions.
