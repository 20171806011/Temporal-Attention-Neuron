# TAN-width-step-validation-20261108

Stage: **State mechanisms**. Curatorial status: **Completed**.

Fixed-CE width-by-learning-rate comparison: h96 DEND D24 BA changed from 0% to 100% when learning rate decreased from .01 to .0025. This is a bounded optimizer intervention, not a scaling law.

This is an English curatorial description, not a new analysis or a translation of the frozen protocol. Numerical intervals are those reported by this stage; exploratory pointwise intervals and later two-endpoint 97.5% intervals have different inferential scopes.

Source archive identifier: `outputs/TAN-width-step-validation-20261108/` (workspace-relative identifier, not a private machine path).

Study/design ancestry: [TAN-width-validation-20261108](../TAN-width-validation-20261108/README.md)

An ancestry link can denote design development, source reuse, or weight reuse. It does not claim that all linked studies contain independent training blocks. The frozen protocol and source locks distinguish these relationships.

## Included historical evidence

- [AUDIT.json](AUDIT.json)
- [FREEZE.redacted.json](FREEZE.redacted.json)
- [INTERPRETATION.md](INTERPRETATION.md)
- [MANIFEST.redacted.json](MANIFEST.redacted.json)
- [PROTOCOL.md](PROTOCOL.md)
- [REPORT.md](REPORT.md)
- [RESULTS.summary.json](RESULTS.summary.json)
- [RUN_STATE.json](RUN_STATE.json)
- [SCIENCE_SUMMARY.json](SCIENCE_SUMMARY.json)
- [SOURCE_LOCK.redacted.json](SOURCE_LOCK.redacted.json)
- [audit.py](audit.py)
- [base_data.py](base_data.py)
- [common.py](common.py)
- [engine.py](engine.py)
- [independent.py](independent.py)
- [model.py](model.py)
- [precheck.py](precheck.py)
- [run.py](run.py)
- [summarize.py](summarize.py)

## Archive limits

Only allowlisted root-level files are included. Raw model arrays, weights, trial subdirectories, figures, external checkouts, copied third-party scalar code and execution environments are omitted. Historical reports/protocols retain their original language. `RESULTS.summary.json`, when present, explicitly lists retained and omitted keys.

Sanitized files are archival inspection material. They are not byte-identical originals and are not a standalone runnable reproduction. Imports and source references can depend on omitted files. Original manifest digests describe the private source archive; public-file digests are separately recorded in [the archive index](../ARCHIVE_INDEX.json). No historical digest is claimed to authenticate a redacted file.
