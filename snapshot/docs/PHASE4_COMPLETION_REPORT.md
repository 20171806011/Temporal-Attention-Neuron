# Phase 4 Completion Report

## 1. What Phase 4 accomplished

Research consolidation and handoff only: a read-only repository audit, a
full experimental dossier, a claim ledger, a reproducibility ledger, a
next-agent brief, an archive-integrity statement, and this completion
report.  No new experiments, no parameter changes, no re-runs, no
re-interpretation of frozen results.

## 2. Files created (documentation only)

- `code/RESEARCH_ARCHIVE_INDEX.md` - component/location/status/frozen
  index with verified paths and recorded integrity notes.
- `docs/TAN_RESEARCH_DOSSIER.md` - research lifecycle dossier
  (executive summary, frozen model specification, experimental lineage,
  Phase 1/2/Probe-1/Probe-2 records with restricted wording, claim
  overview, epistemic boundary, future-hypothesis map).
- `docs/CLAIM_LEDGER.md` - full claim register with evidence and allowed
  wording.
- `docs/REPRODUCIBILITY_LEDGER.md` - environment, seeds, frozen
  parameters, paths, runtimes, sample sizes, audit status, recorded
  hashes (one hash exists; all others NOT RECORDED).
- `docs/NEXT_AGENT_BRIEF.md` - 5-10 minute handoff brief.
- `docs/ARCHIVE_INTEGRITY.md` - frozen scope, known limitations,
  Phase-4 additions.
- `docs/PHASE4_COMPLETION_REPORT.md` - this file.

## 3. Frozen history confirmed

Phase 1 (natural collision), Phase 2 (local geometry pilot), Probe-1 v3
(temporal distractor; amended audit), Probe-2 (identifiability audit;
TERMINATED / AUDIT_FAILED), CHANGELOG [1.0.0]-[1.3.0], all experiment/
analysis scripts, raw outputs, logs, tables, figures, reports and
NPZ ensembles were verified present (see archive index).  No experiment
script or frozen result file was modified during Phase 4.

## 4. Archive integrity issues found (recorded, not fixed)

1. `results/figures/fig27_distractor_task.*` and
   `results/tables/temporal_distractor_{summary,perseed,confusion}.csv`
   contain the single-seed diagnostic (seed 20260904) that predates the
   frozen three-seed data; the authoritative three-seed evidence is in
   `results/logs/temporal_distractor.log` and `audit_v3_amended/`
   (pooled B4-B3 distractor +0.0570, CI [+0.0387, +0.0758]).  Per the
   no-rerun rule this was recorded and not regenerated.
2. Only one file hash was recorded during the project (Probe-2 audit
   script).  Other files are marked NOT RECORDED; no hashes were
   fabricated.
3. No git repository is present; no commit hashes exist.

## 5. Final scientific status of the project

- ESTABLISHED: history dependence of TAN dynamics; exact gate-closed
  contraction; generic event-locked expansion of local response
  geometry; B2 as a strong positional-memory baseline; scalar attention
  as surprise-conditioned amplitude/saliency reweighting.
- NOT SUPPORTED / REJECTED: h_t sufficiency; higher intrinsic dimension
  than LIF; universal memory/retrieval advantage; distractor robustness;
  genuine Q-K routing / query-conditioned binding in the current scalar
  kernel.
- LIMITED SUPPORT: attention-specific effective-rank/scale contrast in
  the matched full frame (Phase 2); B4 retrieval improvement over B3 in
  content-rich context (Probe-1, seed-attenuated).
- Probe-2 remains TERMINATED / AUDIT_FAILED (negative control).

## 6. Where the next researcher starts

1. Read `docs/NEXT_AGENT_BRIEF.md` (5-10 minutes).
2. Consult `docs/TAN_RESEARCH_DOSSIER.md` and `docs/CLAIM_LEDGER.md`
   for details and wording constraints.
3. Use `code/RESEARCH_ARCHIVE_INDEX.md` for paths and
   `docs/REPRODUCIBILITY_LEDGER.md` for reproduction metadata.
4. The only sanctioned open question is a NEW hypothesis branch:
   "what is the minimal additional mechanism required for
   query-conditioned temporal binding?" - not implemented, not tuned,
   and not to be pursued inside the frozen Probe-2 formulation.

---

```
PHASE 4 STATUS: COMPLETE
SCIENTIFIC EXPERIMENTATION: CLOSED
HANDOFF STATUS: READY
```
