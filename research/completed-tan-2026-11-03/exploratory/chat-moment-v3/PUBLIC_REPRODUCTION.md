# Public reproduction notes

## Provenance and integrity

These are AI-assisted Chat-sandbox experiments. Same-agent separate-path checks and same-environment reruns are not independent external verification. The generic associative-memory v1-v4 studies do not validate original TAN/P1-B/v11. Mathematical derivations, negative results, failed conditions and full structured numerical evidence are retained.

The public copy removes operational logs and private absolute path labels and edits explanatory prose. Frozen specifications, core experiment numerical algorithms, inputs, result/summary data, and historical numerical measurements are unchanged. Historical source/spec/result digests authenticate the original files, not edited public prose or wrappers. `HISTORICAL_*` manifests are original archive records and can name omitted or edited files; do not use them as current-file checks. Historical prior-artifact names identify provenance records, not dependencies that must exist in this checkout. `PUBLIC_SHA256SUMS.txt` checks the files in this public package.

Packaging validation parsed every Python file with `ast.parse`, loaded every JSON file, checked frozen-spec digests and compared retained numerical evidence with the original bytes. It did not rerun simulations or claim new scientific verification. Reference runs used Python 3.13.5. Follow the package's recorded dependency versions when reproducing; cross-platform bit identity is not promised. Run commands from a disposable copy of this package so historical files stay intact. Output flags below avoid replacing archived results where supported. Do not run design-freeze scripts to reproduce old results: regenerating a freeze receipt does not establish historical preregistration.

## Replay

Dependencies: NumPy 2.3.5 and SymPy 1.14.0 (`requirements.txt`); no network or external archives. The verifier uses the standard library and does not import the learner.

```sh
python scalar_moment_v3.py --spec frozen_spec.json --out rerun_results_v3.json
python verify_v3.py --spec frozen_spec.json --results rerun_results_v3.json --out rerun_verification_v3.json
```

The `SPEC_INITIAL_SHA256.txt` path label is now package-relative; the frozen specification and its SHA256 value are unchanged. Historical preservation checks concern original v1/v2 archives and are not rerun prerequisites.
