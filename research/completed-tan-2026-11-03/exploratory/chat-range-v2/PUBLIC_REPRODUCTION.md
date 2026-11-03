# Public reproduction notes

## Provenance and integrity

These are AI-assisted Chat-sandbox experiments. Same-agent separate-path checks and same-environment reruns are not independent external verification. The generic associative-memory v1-v4 studies do not validate original TAN/P1-B/v11. Mathematical derivations, negative results, failed conditions and full structured numerical evidence are retained.

The public copy removes operational logs and private absolute path labels and edits explanatory prose. Frozen specifications, core experiment numerical algorithms, inputs, result/summary data, and historical numerical measurements are unchanged. Historical source/spec/result digests authenticate the original files, not edited public prose or wrappers. `HISTORICAL_*` manifests are original archive records and can name omitted or edited files; do not use them as current-file checks. Historical prior-artifact names identify provenance records, not dependencies that must exist in this checkout. `PUBLIC_SHA256SUMS.txt` checks the files in this public package.

Packaging validation parsed every Python file with `ast.parse`, loaded every JSON file, checked frozen-spec digests and compared retained numerical evidence with the original bytes. It did not rerun simulations or claim new scientific verification. Reference runs used Python 3.13.5. Follow the package's recorded dependency versions when reproducing; cross-platform bit identity is not promised. Run commands from a disposable copy of this package so historical files stay intact. Output flags below avoid replacing archived results where supported. Do not run design-freeze scripts to reproduce old results: regenerating a freeze receipt does not establish historical preregistration.

## Replay

Dependencies: NumPy 2.3.5 (`requirements.txt`); no network or external archives.

```sh
python range_consistency_audit_v2.py --out rerun_results_v2.json
python verify_results_v2.py --results rerun_results_v2.json --out rerun_verification_v2.json
```

The verifier checks both the frozen spec and main-script hashes. Both files remain byte-identical to the archived versions. `SCIENTIFIC_CORRECTIONS.json` preserves the historical stopped-trajectory reporting-label correction and initial hashes; the scientific grid/equations were unchanged.
