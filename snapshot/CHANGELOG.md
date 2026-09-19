# Changelog

## [1.3.0] — 2026-09-04

### Phase 3 Probe-2 Identifiability Audit — TERMINATED / AUDIT_FAILED

The frozen scalar TAN attention mechanism was tested for
query-conditioned historical selection (archive:
`code/experiments/probe2/identifiability/`).

- B2 fixed-buffer shortcut: BA = 0.779
- B3 uniform-integration shortcut: BA = 0.709
- B4 attention: HitRate = 0.511, Contrast = 0.034, JSD = 0.0002,
  permutation p = 0.92
- Same-history query counterfactual: argmax flip rate = 0.000

Mechanistic conclusion: E_i = beta*Wq*Wk*S_t*x_i with scalar S_t cannot
alter the relative ordering of historical attention logits as a function
of query identity; observed attention was amplitude-driven rather than
query-conditioned.  No formal Probe-2 experiment was performed.  No TAN
parameters were tuned.  No Phase 1/2/Probe-1 results were modified.
Next possible research direction: a separate minimal-mechanism extension
investigating the smallest architectural change required for genuine
query-conditioned selection.

### Also in this cycle
- Phase 3 Probe-1 v3 (Temporal Distractor Task) closed with constrained
  results: B2 strong positional-memory baseline (bal3 ~0.84-0.88); B4 >
  B3 in the distractor condition (+0.057, 95% CI [+0.039, +0.076],
  n=6000) but with seed-dependent absolute elevation; B3 lo-blind cells;
  interpretation retired for "distractor robustness" (see
  `audit_v3_amended/` and `results/tables/temporal_distractor_*.csv`).

## [1.2.0] — 2026-09-04

### New: Phase 2-Pilot — event-locked effective response geometry

- Added `code/experiments/effective_dimension.py` (frozen protocol: episode
  generator, frames zU/zC/zF, pooled event-locked PR/logTr/spectra, quiet
  reference, recovery, pre-registered decision gates A-D) and
  `code/analysis/effective_dimension_velocity_ampcontrol.py` (velocity
  amplitude-control fill-in; audit gate reproduces pooled dD exactly).
- Headline results (T=5000, 3 fixed seeds, 374 pooled events; all
  calibration/parity/generator assertions PASS):
  * Event-locked velocity-rank expansion exists for every h-bearing model
    (C0 floor dD 1.05 -> B1 1.67 / B3 1.46-1.48 / B4 1.59 zC, 2.09 zF).
  * Pre-registered Outcome A NOT supported: in the shared zC frame B4
    (1.589) does not exceed B1 (1.674) - robust across amplitude controls
    (raw/strat/resid).  "TAN higher effective dimension than LIF" is
    explicitly dropped.
  * Within the matched zF frame, Full TAN exceeds TAN-noAttn in velocity
    effective rank (2.09 vs 1.48; event CIs fully separated) AND in
    covariance scale (event logTr +1.36 nats); the effect survives and
    slightly strengthens under amplitude stratification/residualisation
    (+0.61/+0.74/+0.78).  Rank peak is delayed to tau ~ 4-6 (context-rich
    burst content), not at the first pulse (tau=0: dD 1.66).
  * Recovery: B4 zF tau_e-fold 5.4 [4.6, 6.3] < B3 zF 8.5 [6.1, 10.6].
- New outputs: fig24_eventlocked / fig25_recovery / fig26_spectra
  (png+pdf, mirrored to manuscript/figures),
  results/tables/effective_dimension_{summary,curves}.csv,
  effective_dimension_velocity_{ampcontrol,eventvalues}.csv,
  data/experiment_results/effective_dimension_{events,geometry}.npz,
  results/logs/effective_dimension*.log.
- Framing (working language): attention acts as context-dependent
  geometric reweighting/expansion of the event-locked response, not as a
  system-wide dimensionality increase; "computational dimensionality
  breathing" remains a working name only.

## [1.1.0] — 2026-09-04

### New: Phase 1 — Natural state-collision experiment (non-Markovianity)

- Added `code/experiments/natural_collision.py`: Monte-Carlo natural-collision
  protocol (160k random histories; subthreshold-band pairs with
  |Δh|<1e-4; identical probe injection; four-model ladder LIF /
  LIF+DelayLine / TAN-noAttn / Full TAN) plus reset-re-bifurcation control.
- Results: LIF is exactly contracting (K ≡ λ = 0.5); every window-carrying
  model refutes the scalar-state Markovian claim (mean K 4.5e3–2.0e4);
  Full TAN dominates uniform-attention and passive-buffer amplification
  (p ≤ 1.3e-32) and its divergence is predicted by attention divergence
  (r = 0.87–0.89, p < 1e-300); gate-closed episodes restore K = λ exactly.
- New outputs: fig22_natural_collision, fig23_reset_rebifurcation,
  natural_collision_{summary,pairs,correlations}.csv,
  natural_collision_ensembles.npz (Phase-2 interface), report
  PHASE1_NATURAL_COLLISION_REPORT.md.

## [1.0.0] — 2026-07-26

### Major Revision

This version reflects a systematic pre-submission major revision addressing:

#### Mathematical Corrections
- Fixed attention/energy sign: $E_{t,i}$ is a compatibility score (higher = better), not a physical energy to minimise
- Fixed Q=0 description: attention weights become uniform (1/W), not "no attention applied"; suppression comes from g(0)=0
- Clarified surprise definition: $S_t = x_t - \mu_t$ is a deviation-from-window-mean signal, not a strict past-only prediction error
- Limited state-space claim: $(h_t, S_t)$ is a projection, not a complete closed state
- Limited attractor/absorbing region claims to constant/stationary input only

#### Statistical Corrections
- Flagged LIF baseline calibration limitation in Task A (threshold too high for stimulus amplitude)
- Marked clean phototaxis (Task C) as deterministic comparison (no statistical test applicable)
- Added paired-statistics caveat in Limitations
- Removed "statistically significant" from deterministic comparisons

#### Overclaim Corrections
- Removed "first model" novelty claim → "early and minimal formulation"
- Removed "isomorphic" equivalence claim → "possible link... not proven"
- Removed "threshold is the actual noise filter" causal claim → "consistent with information loss"
- Removed "energy-efficient" → "spike-sparse" + "net energy advantage cannot be established"
- Removed "biologically equivalent" → "computational analogue"
- Removed "mechanistically reproduces E. coli chemotaxis" → "phenomenologically analogous"
- Changed "clean division of labour" → "consistent with functional specialization"
- Changed RNN/GRU from "independent baselines" to "distillation/imitation baselines"
- Changed GRU failure from "distribution shift proven" → "consistent with distribution shift, not causally established"

#### Reference Corrections
- Fixed [gardner1970] (was conway1970): author = Martin Gardner (not John Conway); correct title and pages
- Fixed [bahri2020]: pages = 501--528 (was 201--226)

### Original Release
- Initial TAN model formulation
- 9 experiments (Stage 1: 4 single-seed + Stage 2: 5 systematic)
- 21 figures, 9 tables, 47 references
