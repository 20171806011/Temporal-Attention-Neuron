# Public-copy provenance and verification

## What this copy establishes

The source archive is the completed v11 scientific audit dated 2026-11-02. This public copy retains an explicit scientific allowlist: reports, source/literature reviews, protocols, code, mathematical certificates, inputs, raw predictions, numerical trajectories, statistics, manuscript correction drafts, citations, and scientific provenance. Runtime logs, delivery diagnostics, continuation controls, and transport/execution duplicates are omitted. The original archive was not modified.

All CSV, JSONL, NPY, NPZ, and gzip data files included here retain their original bytes. All numeric, Boolean, and null JSON leaves were compared in order against their original counterparts and retained. NumPy files were inspected with `allow_pickle=False`; no object arrays were loaded. Python source was parsed for syntax without importing or executing the experiments. These are publication-integrity checks, not a new scientific rerun.

## Hashes: historical and current are different contracts

- `CURRENT_PUBLIC_SHA256SUMS.txt` verifies the bytes distributed in this public copy. Run `sha256sum -c CURRENT_PUBLIC_SHA256SUMS.txt` from this directory.
- `CURRENT_PUBLIC_MANIFEST.json` contains the current path, size, and SHA-256 inventory.
- `HISTORICAL_SCIENTIFIC_SHA256SUMS.txt` retains original archive hashes for only the scientific files selected for this copy. It is historical evidence, not a manifest expected to pass on edited public text.
- `PUBLIC_COPY_TRANSFORMS.json` maps each modified scientific path from its original hash to its public-copy hash. Historical hashes embedded in protocols, source manifests, review records and scripts have not been silently refreshed.

Public edits remove operational wording and private identifiers, relocate machine-specific paths, and clarify computational-review scope. No experimental algorithm, parameter, numerical raw evidence, result, or scientific conclusion was intentionally changed. Reference copies of upstream source may consequently differ from their original bytes. Previously public AI authorship and scientific citations are retained.

The original selected 75-file source snapshot was byte-for-byte identical to repository commit `85213f0d417d54e4f48b301ed1c0e4200ef3c421` before public-copy edits. Use that commit to resolve historical source hashes. The source snapshot is deliberately incomplete: some full-repository inventories, Git-status assertions, complete manuscript builds, and historical suites require the full repository and are not reproducible solely from the subset.

## Reproduction limits

The original studies record Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, psutil 7.2.2, and (for the v8 source module) Matplotlib 3.10.8. Package versions are historical observations, not newly installed or tested here. A complete original-source import may require other upstream dependencies; the relevant later audits intentionally extract selected AST definitions to avoid unrelated plotting and output-directory side effects.

Keep sibling directories `tan_audit/` and `Temporal-Attention-Neuron/` in a disposable replay copy. Three previously absolute source-path assignments now resolve the sibling snapshot from `__file__`. Recorded command examples use paths relative to this package root. For newer scripts, `TAN_REPO_ROOT` can select a separate pristine source snapshot. Do not run experiments in the evidence tree: several older scripts write outputs beside themselves. The later primary runners support a fresh `--output-dir`, while their checkers may still read outputs beside their own script.

Some historical hash guards deliberately fail on redacted protocols/scripts/source copies. This is expected and must not be represented as a failed scientific result or as successful reproduction. Read `PUBLIC_REPLAY_PATCH.md` before adapting a disposable copy. No guards were disabled, and no expensive research rerun was performed for this publication. Previously recorded successful replays remain historical records, not a fresh validation of the sanitized copy.

Scientific review reports are AI-assisted, separately implemented computational checks. The word “independent” in historical filenames does not assert independent human review. Early scope limitations concerning unavailable portions of the prior eight-stage archive remain disclosed; later source-text reproductions do not establish byte identity with unseen historical files.
