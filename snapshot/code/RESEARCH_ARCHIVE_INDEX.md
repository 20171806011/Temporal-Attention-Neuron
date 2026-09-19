# TAN Research Archive Index

Read-only index of the frozen research archive (built 2026-09-04, Phase 4).
All paths exist as recorded (verified by read-only repository audit).

| Component | Location | Status | Frozen? | Purpose |
| --- | --- | --- | --- | --- |
| Phase 1 (natural state collision) | `code/experiments/natural_collision.py` | COMPLETE | YES | State-sufficiency / non-Markovianity test of h_t |
| Phase 1 report | `PHASE1_NATURAL_COLLISION_REPORT.md` | COMPLETE | YES | Lemma 1, protocol, statistics, Phase-2 interface |
| Phase 1 figures | `results/figures/fig22_natural_collision.{png,pdf}`, `fig23_reset_rebifurcation.{png,pdf}` (+ manuscript mirror) | COMPLETE | YES | Collision bifurcation; reset re-bifurcation |
| Phase 1 tables | `results/tables/natural_collision_{summary,pairs,correlations}.csv` (mirrored to `data/experiment_results/`) | COMPLETE | YES | Pair-level + summary statistics |
| Phase 1 raw data | `data/experiment_results/natural_collision_ensembles.npz` | COMPLETE | YES | Per-pair trajectories (Phase-2 interface) |
| Phase 1 log | `results/logs/natural_collision.log` | COMPLETE | YES | Run record |
| Phase 2 (local geometry) | `code/experiments/effective_dimension.py` | COMPLETE | YES | Event-locked effective response geometry |
| Phase 2 analysis | `code/analysis/effective_dimension_velocity_ampcontrol.py` | COMPLETE | YES | Velocity amplitude controls (raw/strat/resid) |
| Phase 2 figures | `results/figures/fig24_eventlocked.{png,pdf}`, `fig25_recovery.{png,pdf}`, `fig26_spectra.{png,pdf}` (+ mirror) | COMPLETE | YES | Event-locked traces; recovery profiles; spectra |
| Phase 2 tables | `results/tables/effective_dimension_{summary,curves}.csv`, `effective_dimension_velocity_{ampcontrol,eventvalues}.csv` | COMPLETE | YES | Geometry curves + summary |
| Phase 2 raw data | `data/experiment_results/effective_dimension_{events,geometry}.npz` | COMPLETE | YES | Event ensembles + geometry curves |
| Phase 2 logs | `results/logs/effective_dimension.log`, `effective_dimension_velocity_control.log` | COMPLETE | YES | Run records |
| Probe-1 v3 (temporal distractor) | `code/experiments/temporal_distractor.py` | COMPLETE (frozen v3 label) | YES | Retrieval-cue target recall task |
| Probe-1 v3 figure | `results/figures/fig27_distractor_task.{png,pdf}` | COMPLETE* | YES | *single-seed diagnostic version - see integrity note below |
| Probe-1 tables | `results/tables/temporal_distractor_{summary,perseed,confusion}.csv` | COMPLETE* | YES | *single-seed diagnostic version - see integrity note below |
| Probe-1 log (3-seed per-seed evidence) | `results/logs/temporal_distractor.log` | COMPLETE | YES | 3-seed per-seed bal3 (run blocked before pooling persistence) |
| Probe-1 amended audit | `audit_v3_amended/{AUDIT_AMENDMENT,PER_SEED_AUDIT,POOLED_PROBE1_V3}.md`, `MACHINE_READABLE_AUDIT.json` | COMPLETE | YES | Amended audit on frozen 3-seed data + pooling |
| Probe-1 amended audit code | `code/analysis/probe1_v3_amended_audit.py` | COMPLETE | YES | Deterministic reproduction + permutation-null audit |
| Probe-2 (identifiability audit) | `code/experiments/probe2/identifiability/probe2_identifiability_audit.py` | TERMINATED / AUDIT_FAILED | YES | Query-conditioned binding identifiability |
| Probe-2 theory precheck | `code/experiments/probe2/identifiability/THEORY_PRECHECK.md` | TERMINATED | YES | Scalar-S_t kernel analysis |
| Probe-2 report | `code/experiments/probe2/identifiability/AUDIT_REPORT.md` | TERMINATED | YES | Full audit results + mechanism analysis |
| Probe-2 raw output | `code/experiments/probe2/identifiability/AUDIT_OUTPUT.txt` | TERMINATED | YES | Raw console output (AUDIT_FAILED) |
| Probe-2 archive readme | `code/experiments/probe2/identifiability/README.md` | TERMINATED | YES | Archive rules and prohibitions |
| Model implementation | `code/model/tan.py` (TANNeuronAblation, LIFNeuron, variants) | COMPLETE | YES | Frozen TAN equations |
| Requirements | `code/requirements.txt` | - | YES | numpy/scipy/matplotlib (also lists pandas/torch/Pillow; unused) |
| Project log | `CHANGELOG.md` ([1.0.0]-[1.3.0]) | COMPLETE | YES | Experiment lifecycle log |
| Project README | `README.md` | - | - | Package description / reproduction entry |

Integrity notes (recorded, not fixed - Phase 4 rule: do not re-run):
- `results/figures/fig27_distractor_task.*` and
  `results/tables/temporal_distractor_{summary,perseed,confusion}.csv`
  were produced by the single-seed diagnostic run (seed 20260904,
  n=2000) that predates the frozen 3-seed data; the 3-seed formal run was
  blocked by pre-registered audit gates *before* persistence.  The
  authoritative frozen 3-seed Probe-1 v3 evidence is
  `results/logs/temporal_distractor.log` (per-seed bal3) and
  `audit_v3_amended/` (frozen-data verification, per-seed audit table,
  pooled results).  Do not cite the single-seed CSVs/figure as the formal
  result.
- No git repository is present; file hashes are recorded only where they
  were saved during the project (Probe-2 script, SHA-256
  `AABF8EBBB81E911E69117591F84E4F836CF59CF8C1B505DE53A8B7C1A88143C7`).
