# TAN-I Paper Completion Report

**Project:** Temporal Attention Neuron (TAN) — Independent Mechanistic Paper
**Task:** Full manuscript production from the frozen research archive (LaTeX + PDF)
**Status:** **TAN-I PAPER STATUS: COMPLETE**
**Date:** 2026-09-04 (session wrap-up)
**Author record:** Li Zexu, School of Physics and Astronomy, University of Leeds

---

## 1. Scope and constraints honoured

This task produced a submission-style mechanistic paper **from the frozen archive only**.

| Constraint | Outcome |
|---|---|
| New scientific experiments run | **0** (no code under `code/experiments/`, `code/analysis/`, or `audit_v3_amended/` was executed or modified) |
| Parameter changes to frozen models | **0** (all numbers quoted verbatim from frozen outputs) |
| Claim inflation | None; claim language follows the frozen claim ledger (ESTABLISHED / LIMITED SUPPORT / NEGATIVE), see `docs/CLAIM_LEDGER.md` |
| Probe-2 AUDIT_FAILED preserved | Yes — status `TERMINATED / AUDIT_FAILED` retained in Methods, Results, Discussion, Tables 5, and the appendix; described as an identifiability **boundary result**, not an incomplete experiment |
| Negative results preserved | Yes — shared-frame B4 not > B1 (Phase 2), Probe-1 distractor-robustness NEGATIVE, Probe-1 seed-20260906 `INVALID_FOR_INTERPRETATION` cell kept with per-cell status |
| Banned-phrase discipline | QA scan confirms no affirmative use of "intelligent", "superior memory", "universally outperforms", "semantic binding is established", "distractor-robust", or "intrinsic manifold dimension" (the single occurrence of the last is a negation) |
| References | 58 real entries only; no fabricated citations |

---

## 2. Question and headline answer of the manuscript

> What can a **scalar-input, scalar-gate temporal attention neuron** (one dynamical unit
> with a windowed, surprise-gated Boltzmann kernel) actually compute, once architectural
> resemblance to attention is separated from capability?

**Headline answer (as supported by frozen evidence):**
TAN's scalar query collapses the kernel to a saliency re-weighting that **cannot reorder
history by query content**; it accumulates a genuinely non-Markovian state (Phase 1),
produces event-locked response geometries that are not shared-frame coupling (Phase 2,
B4 zF 2.089 vs zC 1.589), is **not** robust to a temporal distractor (Probe 1, pooled
clean/dist 0.328/0.389, B4−B3 +0.057 CI [+0.039,+0.076]), and **fails** the
query-conditioned binding identifiability audit (Probe 2, TERMINATED / AUDIT_FAILED:
hit rate 0.511, contrast R 0.034 ≪ 2, query-conditioned JSD 0.0002, permutation p = 0.92).
Mechanistically, TAN implements surprise-gated temporal memory plus query-independent
saliency weighting — a memory and saliency unit, not a binding or routing unit.

---

## 3. Deliverables

| Deliverable | Path |
|---|---|
| Manuscript (LaTeX master) | `paper/main.tex` |
| Compiled PDF | `paper/main.pdf` (28 pages, 28 pages incl. supplementary, 877 KB) |
| Bibliography | `paper/references.bib` (58 entries; plainnat author-year) |
| Sections | `paper/sections/{introduction,background,model,methods,results,discussion,limitations,conclusion}.tex` |
| Supplementary S1–S8 | `paper/supplementary/S1_equations.tex … S8_reproducibility.tex` |
| Figures (8) | `paper/figures/` (see §4) |
| Tables (6) | `paper/tables/tab_{params,phase1,phase2,probe1,probe2,claims}.tex` |
| Build files | `paper/Makefile`, `paper/README.md` |
| Reproducible generator | `paper/scripts/make_figs_tables.py` (+ `_fix_*` helpers) |
| QA script | `paper/scripts/qa_pdf.py` |
| This report | `docs/TAN_I_PAPER_COMPLETION_REPORT.md` |

---

## 4. Figures (8) and tables (6)

### Figures
1. **fig1_architecture.pdf** — TAN architecture schematic (new artwork from frozen equations).
2. **fig2_state_sufficiency.pdf** — verbatim copy of frozen `fig22` (Phase-1 collision results).
3. **fig3_geometry.pdf** — verbatim copy of frozen `fig24` (Phase-2 event-locked geometry).
4. **fig4a_rank_recovery.pdf / fig4b_spectra.pdf** — verbatim copies of frozen `fig25`/`fig26`
   (rank-recovery and spectra controls).
5. **fig5_probe1.pdf** — chart of frozen Probe-1 v3 three-seed results (pooled and per-seed;
   includes the `INVALID_FOR_INTERPRETATION` status cell).
6. **fig6_probe2.pdf** — chart of frozen Probe-2 audit results (B2/B3 linear shortcuts vs red
   line 0.58; B4 hit 0.511; contrast R 0.034; self-attraction P = 1.0).
7. **fig7_mechanism.pdf** — schematic of the two-factor mechanism (state memory +
   query-independent saliency), with explicit "no binding" callout.
8. **fig8_summary.pdf** — study summary schematic (claims map with ESTABLISHED / NEGATIVE /
   AUDIT_FAILED colour coding).

### Tables
1. **tab_params** — frozen TAN parameter set (Table 1).
2. **tab_phase1** — Phase-1 state-sufficiency collision statistics (Table 2).
3. **tab_phase2** — Phase-2 event-locked geometry statistics (Table 3).
4. **tab_probe1** — Probe-1 v3 amended three-seed audit results, with per-cell statuses
   including `INVALID_FOR_INTERPRETATION` (Table 4).
5. **tab_probe2** — Probe-2 identifiability audit summary, Status: TERMINATED / AUDIT_FAILED
   (Table 5).
6. **tab_claims** — claim ledger summary mapped to evidence and status (Table 6).

Supplementary **S1** equations; **S2** phase-1 collisions; **S3** phase-2 geometry;
**S4** probe-1; **S5** probe-2; **S6** archive provenance; **S7** limitations in full;
**S8** reproducibility details.

---

## 5. Build chain (reproducible)

- Compiler: **MiKTeX 24.1** (`pdflatex`/`bibtex` from `C:\Program Files\MiKTeX\miktex\bin\x64`)
  — `latexmk` is NOT used (requires Perl, absent on this machine); `paper/Makefile` documents
  both paths, the fallback is the command sequence below.
- Working directory: `paper/`
- Commands:
  ```
  pdflatex --enable-installer -interaction=nonstopmode main.tex
  bibtex main
  pdflatex --enable-installer -interaction=nonstopmode main.tex
  pdflatex --enable-installer -interaction=nonstopmode main.tex
  ```
- Packages auto-installed from the MiKTeX CTAN repository (`mpm --set-repository=…` configured)
  during the first build; subsequent builds are offline-capable.
- Style: 11 pt article, a4paper, plainnat author-year (natbib), amsmath/amssymb/bm,
  booktabs, siunitx, hyperref, cleveref, xcolor, geometry, microtype, caption, subcaption.
- Figure/table artifacts regenerable from frozen numbers via
  `python paper/scripts/make_figs_tables.py` (deterministic seeds; does not re-run experiments).

---

## 6. QA status

Textual QA (`python paper/scripts/qa_pdf.py`, pypdf 6.x):

- Pages: **28** (main text + references pp. 1–22; supplementary S1–S8 pp. 23–28).
- All 14 content checks **FOUND** (title, abstract, keywords, key frozen statistics of every
  phase, `AUDIT_FAILED`/`TERMINATED`, flip-rate 0.000, claim-separation box, References,
  S1, S8).
- Banned-phrase scan: **0 affirmative occurrences** (all hits negated) — PASS.
- Undefined references/citations: **0**; literal `??`: **0**; compile errors: **0**.
- Bibliography entries: **58**.

### Residual cosmetic warnings (disclosed, not hidden)
11 `Overfull \hbox` warnings remain after layout tuning (0 errors). Breakdown:

| Size (pt) | Count | Location / cause |
|---|---|---|
| ~13.5 | 2 | glue-only row boxes at table-float boundaries (log content shows no glyphs); ≤ 4.8 mm past the text edge |
| 4.66 | 1 | Table 1 header row (`symbol` column region) |
| 1.08 / 1.49 | 2 | Table 3 header region rows |
| 2.79 | 6 | identical-width row-box warnings in Table 3 data rows (invisible, < 0.1 mm visible at print scale) |

No overfull box exceeds ~4.8 mm, none contains clipped glyphs beyond the paper edge, and all
occur inside tables; this is a copy-editing-level nit (further per-row column tuning or a
final `\resizebox` pass would remove them). Text extraction of the full document is subject to
a pypdf 5000-form-XObject cap on later pages (tool limitation, not a document defect); the
supplementary checks still pass because S1/S8 text is ordinary content stream.

**Limitations of this QA:** text-level only; no visual (render) inspection was available in
this environment. Recommended pre-submission step: a human visual pass over pp. 7–20 (table
pages) after any journal-template port.

---

## 7. Scientific integrity statement

- All numbers, CIs, p-values, statuses and quotes originate from the frozen archive:
  Phase-1/Phase-2 experiment logs and figures, `audit_v3_amended/` (Probe-1 authoritative
  three-seed results), and `code/experiments/probe2/identifiability/` + `docs/` ledgers
  (Probe-2). The formal Probe-1 evidence is the amended audit; the single-seed diagnostic
  (`fig27_distractor_task`, `results/tables/temporal_distractor_*.csv`) is cited nowhere as
  formal evidence.
- No frozen file was modified during the TAN-I task; new files were confined to `paper/` and
  this report. Archive integrity therefore remains PASS (see also `docs/ARCHIVE_INTEGRITY.md`
  and `docs/REPRODUCIBILITY_LEDGER.md`).
- The Probe-2 script's recorded SHA-256
  `AABF8EBB…143C7` (full hash in `docs/ARCHIVE_INTEGRITY.md`) is unchanged.

---

## 8. Status block

```
TAN-I PAPER STATUS: COMPLETE
Manuscript: paper/main.tex
PDF: paper/main.pdf
Supplementary: paper/supplementary/
Completion Report: docs/TAN_I_PAPER_COMPLETION_REPORT.md
Page Count: 28
Figures: 8
Tables: 6
References: 58
Scientific Experiments Added: 0
Parameters Changed: 0
Archive Integrity: PASS
LaTeX Build: PASS
PDF QA: PASS (with minor overfull hbox warnings, see report)
```
