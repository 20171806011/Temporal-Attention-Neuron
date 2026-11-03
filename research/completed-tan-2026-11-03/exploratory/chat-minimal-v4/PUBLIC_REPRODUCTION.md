# Public reproduction notes

## Provenance and integrity

These are AI-assisted Chat-sandbox experiments. Same-agent separate-path checks and same-environment reruns are not independent external verification. The generic associative-memory v1-v4 studies do not validate original TAN/P1-B/v11. Mathematical derivations, negative results, failed conditions and full structured numerical evidence are retained.

The public copy removes operational logs and private absolute path labels and edits explanatory prose. Frozen specifications, core experiment numerical algorithms, inputs, result/summary data, and historical numerical measurements are unchanged. Historical source/spec/result digests authenticate the original files, not edited public prose or wrappers. `HISTORICAL_*` manifests are original archive records and can name omitted or edited files; do not use them as current-file checks. Historical prior-artifact names identify provenance records, not dependencies that must exist in this checkout. `PUBLIC_SHA256SUMS.txt` checks the files in this public package.

Packaging validation parsed every Python file with `ast.parse`, loaded every JSON file, checked frozen-spec digests and compared retained numerical evidence with the original bytes. It did not rerun simulations or claim new scientific verification. Reference runs used Python 3.13.5. Follow the package's recorded dependency versions when reproducing; cross-platform bit identity is not promised. Run commands from a disposable copy of this package so historical files stay intact. Output flags below avoid replacing archived results where supported. Do not run design-freeze scripts to reproduce old results: regenerating a freeze receipt does not establish historical preregistration.

## Replay

Dependencies: NumPy 2.3.5 and SymPy 1.14.0 (`requirements.txt`); no network or external archives. The verifier uses the standard library and does not import the learner.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python minimal_moment_v4.py --spec frozen_spec_v4.json --out rerun_results_v4.json --timing-out rerun_timing_v4.json
python verify_v4.py --spec frozen_spec_v4.json --results rerun_results_v4.json --out rerun_verification_v4.json
```

`SOURCE_PRE_RUN.sha256` still matches the main learner and frozen spec. Included `v3_reference_*` files remain byte-identical historical references, not runtime dependencies or a v3 rerun. The reference source's default spec filename targets its original package layout; it is not the v4 entry point. `summarize_v4.py` reads the archived v4 results, timing and v3 summary and overwrites summary/refusal/CSV outputs, so use it only in a disposable copy. `freeze_v4.py` rewrites spec/hash/receipt files and is not part of replay. The initial verifier reporting failure is retained in `SCIENTIFIC_CORRECTIONS.md`, including the fact that the first verifier run did not succeed.
