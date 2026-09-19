# Archive Integrity Statement

> Phase 1, Phase 2, Probe-1, and Probe-2 are treated as frozen research
> history.

Built 2026-09-04 (Phase 4).  No experimental file was modified during
Phase 4; only new documentation files were added:
`code/RESEARCH_ARCHIVE_INDEX.md` and the files under `docs/`.

## Frozen scope

- Frozen experiment scripts (not modified since their final frozen
  state):
  - `code/experiments/natural_collision.py` (Phase 1)
  - `code/experiments/effective_dimension.py` (Phase 2)
  - `code/experiments/temporal_distractor.py` (Probe-1 v3)
  - `code/experiments/probe2/identifiability/probe2_identifiability_audit.py`
    (Probe-2; SHA-256
    AABF8EBBB81E911E69117591F84E4F836CF59CF8C1B505DE53A8B7C1A88143C7,
    verified byte-identical across relocation)
- Frozen analysis scripts:
  - `code/analysis/effective_dimension_velocity_ampcontrol.py`
  - `code/analysis/probe1_v3_amended_audit.py`
- Frozen raw outputs: figures fig22-fig27 (results/figures + manuscript
  mirror), tables in results/tables + data/experiment_results, NPZ
  ensembles, logs in results/logs (full path index:
  `code/RESEARCH_ARCHIVE_INDEX.md`).
- Frozen reports/audits:
  - `PHASE1_NATURAL_COLLISION_REPORT.md`
  - `audit_v3_amended/` (AUDIT_AMENDMENT.md, PER_SEED_AUDIT.md,
    POOLED_PROBE1_V3.md, MACHINE_READABLE_AUDIT.json)
  - `code/experiments/probe2/identifiability/` (THEORY_PRECHECK.md,
    AUDIT_REPORT.md, AUDIT_OUTPUT.txt, README.md)
- `CHANGELOG.md` (versions [1.0.0] - [1.3.0]); `README.md`;
  `code/model/tan.py`; `code/model/lif.py`.

## Known limitations (recorded, deliberately NOT fixed)

1. Probe-2 is a terminated negative-control identifiability result
   (STATUS: AUDIT_FAILED).  It must remain as archived; no re-encoding,
   re-tuning or kernel modification is permitted to produce a different
   result.
2. `results/figures/fig27_distractor_task.*` and
   `results/tables/temporal_distractor_{summary,perseed,confusion}.csv`
   contain the single-seed diagnostic run (seed 20260904) that predates
   the frozen three-seed data; the authoritative three-seed Probe-1 v3
   evidence is `results/logs/temporal_distractor.log` plus
   `audit_v3_amended/`.  Regeneration was not performed (no-rerun rule).
3. File hashes were recorded only for the Probe-2 audit script; other
   files are marked NOT RECORDED in docs/REPRODUCIBILITY_LEDGER.md.
4. No git repository is present in this workspace; no commit hashes
   exist.  No archive commit was created.

## Phase 4 additions (documentation only)

- `code/RESEARCH_ARCHIVE_INDEX.md`
- `docs/TAN_RESEARCH_DOSSIER.md`
- `docs/CLAIM_LEDGER.md`
- `docs/REPRODUCIBILITY_LEDGER.md`
- `docs/NEXT_AGENT_BRIEF.md`
- `docs/ARCHIVE_INTEGRITY.md`
- `docs/PHASE4_COMPLETION_REPORT.md`
