# Public reproduction notes

## Provenance and integrity

These are AI-assisted Chat-sandbox experiments. Same-agent separate-path checks and same-environment reruns are not independent external verification. The generic associative-memory v1-v4 studies do not validate original TAN/P1-B/v11. Mathematical derivations, negative results, failed conditions and full structured numerical evidence are retained.

The public copy removes operational logs and private absolute path labels and edits explanatory prose. Frozen specifications, core experiment numerical algorithms, inputs, result/summary data, and historical numerical measurements are unchanged. Historical source/spec/result digests authenticate the original files, not edited public prose or wrappers. `HISTORICAL_*` manifests are original archive records and can name omitted or edited files; do not use them as current-file checks. Historical prior-artifact names identify provenance records, not dependencies that must exist in this checkout. `PUBLIC_SHA256SUMS.txt` checks the files in this public package.

Packaging validation parsed every Python file with `ast.parse`, loaded every JSON file, checked frozen-spec digests and compared retained numerical evidence with the original bytes. It did not rerun simulations or claim new scientific verification. Reference runs used Python 3.13.5. Follow the package's recorded dependency versions when reproducing; cross-platform bit identity is not promised. Run commands from a disposable copy of this package so historical files stay intact. Output flags below avoid replacing archived results where supported. Do not run design-freeze scripts to reproduce old results: regenerating a freeze receipt does not establish historical preregistration.

## Replay and modified wrappers

Dependency: NumPy 2.3.5, plus the Python standard library. No network, model weights or full upstream checkout is required.

```sh
python run_whq.py --out rerun_results.json --timing rerun_timing.json
python verify_whq.py --results rerun_results.json --out rerun_verification.json
```

The runner imports only `source/reference_m1.py` locally and checks the unchanged frozen-spec digest. It deterministically rewrites all three `inputs_*.json` files, so use a disposable copy even with custom output flags. `source/reference_m1.py` is a labeled scalar transcription of pinned source equations, not a production import or a byte-identical production file. `fetch_sources.py` is an optional network retrieval helper, not a replay dependency; it was not run for public packaging.

Only archive-preservation plumbing changed in `verify_whq.py`: it now accepts optional `--prior-bundles-dir DIRECTORY`, with no private absolute path. Omit it for all scientific checks without original archives. Historical `verification.json` retains its original 305-check count, which includes six historical preservation checks; two unrelated preservation entries are omitted. A public default replay therefore has 299 checks, or 303 with the four retained original archives. These are differences in preservation bookkeeping, not scientific assertion results. No replay was executed during packaging. `freeze.py` no longer scans for prior archives; its scientific spec generator is unchanged and is not needed for replay.

WHQ adds a content state, hardwired keys, altered holding dynamics and a new observer. It is not a cost-free repair of the original neuron. The resource-matched EMA is slightly better in RMSE; the 200 ms condition and wrong-address controls fail. These results do not establish distinctive TAN superiority, general online binding, spike-only computation or biological feasibility.
