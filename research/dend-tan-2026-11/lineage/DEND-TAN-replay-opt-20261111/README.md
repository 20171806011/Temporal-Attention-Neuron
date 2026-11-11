# DEND-TAN-replay-opt-20261111

Stage: **Interference and repair**. Curatorial status: **Completed optimization with latency negatives**.

Caching plus zero-row multiplication skipping reduced replay MACs by 44.15% but increased triggered-batch time by 4.738 ms and single-trigger time by .961 ms. Full-prefix replay was retained.

This is an English curatorial description, not a new analysis or a translation of the frozen protocol. Numerical intervals are those reported by this stage; exploratory pointwise intervals and later two-endpoint 97.5% intervals have different inferential scopes.

Source archive identifier: `outputs/DEND-TAN-replay-opt-20261111/` (workspace-relative identifier, not a private machine path).

Study/design ancestry: [DEND-TAN-event-replay-20261111](../DEND-TAN-event-replay-20261111/README.md)

An ancestry link can denote design development, source reuse, or weight reuse. It does not claim that all linked studies contain independent training blocks. The frozen protocol and source locks distinguish these relationships.

## Included historical evidence

- [AUDIT.json](AUDIT.json)
- [FREEZE.redacted.json](FREEZE.redacted.json)
- [MANIFEST.redacted.json](MANIFEST.redacted.json)
- [PROTOCOL.md](PROTOCOL.md)
- [REPORT.md](REPORT.md)
- [RESULTS.summary.json](RESULTS.summary.json)
- [RUN_STATE.json](RUN_STATE.json)
- [SCIENCE_SUMMARY.json](SCIENCE_SUMMARY.json)
- [SOURCE_LOCK.redacted.json](SOURCE_LOCK.redacted.json)
- [TIMING.json](TIMING.json)
- [WORK_STATUS.json](WORK_STATUS.json)
- [audit.py](audit.py)
- [capacity_data.py](capacity_data.py)
- [capacity_metrics.py](capacity_metrics.py)
- [capacity_model.py](capacity_model.py)
- [event_model.py](event_model.py)
- [finalize.py](finalize.py)
- [independent_capacity.py](independent_capacity.py)
- [independent_capacity_data.py](independent_capacity_data.py)
- [independent_event.py](independent_event.py)
- [independent_opt.py](independent_opt.py)
- [independent_reset.py](independent_reset.py)
- [independent_warm.py](independent_warm.py)
- [learned_model.py](learned_model.py)
- [opt_model.py](opt_model.py)
- [population.py](population.py)
- [preflight.py](preflight.py)
- [reference.py](reference.py)
- [reset_model.py](reset_model.py)
- [run.py](run.py)
- [summarize.py](summarize.py)
- [warm_model.py](warm_model.py)

## Archive limits

Only allowlisted root-level files are included. Raw model arrays, weights, trial subdirectories, figures, external checkouts, copied third-party scalar code and execution environments are omitted. Historical reports/protocols retain their original language. `RESULTS.summary.json`, when present, explicitly lists retained and omitted keys.

Sanitized files are archival inspection material. They are not byte-identical originals and are not a standalone runnable reproduction. Imports and source references can depend on omitted files. Original manifest digests describe the private source archive; public-file digests are separately recorded in [the archive index](../ARCHIVE_INDEX.json). No historical digest is claimed to authenticate a redacted file.
