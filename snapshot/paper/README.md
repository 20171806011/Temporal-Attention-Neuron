# TAN-I paper build chain

Reproducible LaTeX build for the independent mechanistic paper
"The Computational Boundary of Scalar Temporal Attention".

## Requirements

- MiKTeX/TeX Live with: amsmath amssymb bm graphicx booktabs siunitx
  hyperref cleveref xcolor geometry microtype natbib caption subcaption
  array float; bibtex; latexmk (or pdflatex+bibtex).
- Python 3.12 with numpy + matplotlib (only to (re)generate schematic
  figures and table fragments from the frozen archive).

## Build

```bash
cd paper
latexmk -pdf main.tex          # preferred
# or, manually:
#   pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Output: `paper/main.pdf`.

## Artifacts (all derived from the frozen archive; no new experiments)

- `figures/fig1_architecture.pdf`, `fig7_mechanism.pdf`,
  `fig8_summary.pdf`: schematics (no data).
- `figures/fig5_probe1.pdf`, `fig6_probe2.pdf`: charts of frozen
  three-seed / audit values (sources recorded in captions and in
  `scripts/make_figs_tables.py`).
- `figures/fig2_state_sufficiency.pdf`, `fig3_geometry.pdf`,
  `fig4a_rank_recovery.pdf`, `fig4b_spectra.pdf`: verbatim copies of
  frozen experiment figures (`results/figures/fig22/24/25/26`).
- `tables/tab_*.tex`: fragments generated from frozen CSVs and frozen
  audit documents by `scripts/make_figs_tables.py`.

## Regeneration of derived artifacts

```bash
python paper/scripts/make_figs_tables.py
```

This script only reads frozen archive files (or transcribes frozen audit
values with provenance comments); it never runs an experiment.

## Manuscript provenance

Every claim traces to the frozen TAN-I archive (`../docs/` +
`../code/`, see `../code/RESEARCH_ARCHIVE_INDEX.md`).  Probe-2 is
archived as TERMINATED / AUDIT_FAILED and is reported as a mechanistic
identifiability boundary.
