> Historical archival document. Original language preserved. Personal paths/host metadata and unavailable media links were sanitized; consult README.md and ARCHIVE_INDEX.json.

# Post-run resource accounting correction

Recorded after scientific data generation, before the final report. The frozen protocol, source, raw rows, checkpoints, data, primary criterion and selection decisions are preserved. No experiment was rerun with a changed hypothesis, dynamics or training configuration. This separate log supplements the frozen AMENDMENTS.md without altering its recorded pre-data contents.

Code review found that `branch_event_deliveries` in raw mechanism rows was calculated from all emitted POINT branch events even during the two steps when branch-to-soma communication was blocked. These spikes still occur, but they are not delivered on the cut path. Consequently `consumed_internal_event_deliveries` was also overcounted in those 15 POINT local-path-cut rows (five seeds times three phases). The original native event counts themselves remain correct.

`correct_resource_accounting.py` recomputes these same retained inputs and weights with both existing forward implementations, requires all spikes and saved counts to agree, and counts branch deliveries only outside the blocked two-step window. `RESOURCE_CORRECTION.json` records every old value, corrected value and blocked-event difference. All main-report resource tables use the intact fair-comparison models, whose resource counts are unaffected. The main verifier checks the frozen raw schema; this supplementary check corrects its delivery-accounting interpretation.

This changes no primary effect, accuracy, confidence interval, state trajectory, training result, model selection or frontier eligibility. It is an accounting correction, not a protocol change or a new independent experiment.
