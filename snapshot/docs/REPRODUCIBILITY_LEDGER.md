# TAN Reproducibility Ledger

Recorded from existing information only (no experimental re-runs were
performed to produce this ledger; Phase 4 rule).

## Environment (verified read-only, 2026-09-04)

| item | value |
| --- | --- |
| Python | 3.12.7 |
| numpy | 1.26.4 |
| scipy | 1.13.1 |
| matplotlib | 3.9.2 |
| other packages required by code | none (pandas/torch/Pillow are listed in `code/requirements.txt` but are not used by the experiment scripts) |

## Random seeds (frozen per experiment)

| experiment | seeds / rng seed |
| --- | --- |
| Phase 1 natural collision | `--seed 20260726` (global deterministic stream) |
| Phase 2 effective dimension | per-seed streams 20260904, 20260905, 20260906 (+ 20260907, 20260908 available via `--seeds 5`); boostrap seeds fixed in code (424242, 777, 31337 etc.) |
| Phase 2 velocity amplitude control | deterministic rebuild on the same 3 seeds |
| Probe-1 v3 temporal distractor | seeds 20260904, 20260905, 20260906; bootstrap 4242 |
| Probe-1 amended audit | same 3 seeds; permutation null per (seed, model), rng derived deterministically; N_perm = 100 |
| Probe-2 identifiability audit | SEED = 20260904, N = 1000, W = 5, sigma_noise = 0.05 |

## Frozen parameters

TAN defaults: W=5, lambda=0.5, theta=0.5, Wq=2.0, Wk=Wv=1.0, beta=1.0,
eps=0.0; window includes current input; zero-padded onset; spike when
h > theta then reset to 0.  B2 buffer weights: recency ramp (1..5)/15.
Probe-1 v3: classes lo U(0.55,1.05) / hi U(0.95,1.45), overlap
[0.95,1.05], distractors from the same mixture, catch fraction 1/3,
query code random from mixture, response-only readout [u9..u13,
y9..u13], 3-class linear softmax (L-BFGS, L2 = 1e-4), train 4000 /
test 2000 per seed.  Probe-2: K_amp(A)=1.0, K_amp(B)=2.0, Delta=0.25,
query codes 3.0/4.0.

## Experiment paths, outputs, runtime (recorded at execution time)

| experiment | script | key outputs | recorded runtime |
| --- | --- | --- | --- |
| Phase 1 | `code/experiments/natural_collision.py` | results/figures/fig22, fig23; results/tables/natural_collision_*.csv; data/experiment_results/natural_collision_ensembles.npz; results/logs/natural_collision.log; PHASE1_NATURAL_COLLISION_REPORT.md | ~21-36 s |
| Phase 2 | `code/experiments/effective_dimension.py` | results/figures/fig24-26; results/tables/effective_dimension_{summary,curves}.csv; data/experiment_results/effective_dimension_{events,geometry}.npz; results/logs/effective_dimension.log | 54.4 s |
| Phase 2 amp control | `code/analysis/effective_dimension_velocity_ampcontrol.py` | results/tables/effective_dimension_velocity_{ampcontrol,eventvalues}.csv; results/logs/effective_dimension_velocity_control.log | 22.4 s |
| Probe-1 v3 (3-seed run) | `code/experiments/temporal_distractor.py` | results/logs/temporal_distractor.log (per-seed evidence; run blocked before pooling persistence by pre-registered audit gates) | ~30 s to block |
| Probe-1 amended audit | `code/analysis/probe1_v3_amended_audit.py` | audit_v3_amended/{AUDIT_AMENDMENT,PER_SEED_AUDIT,POOLED_PROBE1_V3}.md, MACHINE_READABLE_AUDIT.json | 126-137 s |
| Probe-2 audit | `code/experiments/probe2/identifiability/probe2_identifiability_audit.py` | prints only; archive copies AUDIT_OUTPUT.txt | 0.6-0.7 s |

## Test counts / sample sizes (frozen)

- Phase 1: N = 160,000 histories (4 x 40,000); 1,500 collision pairs
  analysed per model (of 92,600 / 19,234 / 1,633,844 / 1,427,817
  found); 843 fully resynchronised pairs in the reset-re-bifurcation
  subset.
- Phase 2: T = 5000, burn-in 500, 3 seeds; 374 valid pooled events;
  570 quiet samples pooled.
- Probe-1 v3: 4,000 train / 2,000 test per seed, 3 seeds; pooled test
  n = 6,000 trials.
- Probe-2: N = 1,000 trials; 96-point idealised discrete dataset for the
  separability check; 2,000 bootstrap resamples; 1,000 permutation
  resamples for the JSD test.

## Hashes

Only one hash was recorded during the project (do not treat other files
as hash-verified):

| file | hash |
| --- | --- |
| code/experiments/probe2/identifiability/probe2_identifiability_audit.py | SHA-256 AABF8EBBB81E911E69117591F84E4F836CF59CF8C1B505DE53A8B7C1A88143C7 (verified byte-identical across relocation) |

All other files: `NOT RECORDED` (no hashes were saved during the
project; none are fabricated here).

## Audit status summary

- Phase 1: all calibration/parity assertions PASS (parity vs
  code/model/tan.py; LIF quiet dGq in range; NaN policy recorded).
- Phase 2: all calibration assertions PASS; per-seed centring fix logged
  (estimator-level; decision variables unaffected).
- Probe-1 v3: frozen-data integrity 4.7e-4 (rounding) verified by the
  amended audit; per-cell statuses in audit_v3_amended/PER_SEED_AUDIT.md
  (PASS / PASS_WITH_FLAG / INVALID_FOR_INTERPRETATION); pooled contrast
  B4-B3 = +0.0570 [+0.0387, +0.0758].
- Probe-2: STATUS = AUDIT_FAILED (terminated negative control).

## Archive integrity notes (recorded, not fixed)

1. `results/figures/fig27_distractor_task.*` and
   `results/tables/temporal_distractor_{summary,perseed,confusion}.csv`
   hold the single-seed diagnostic (seed 20260904, n=2000) that predates
   the frozen 3-seed data; the authoritative 3-seed evidence is in
   `results/logs/temporal_distractor.log` and `audit_v3_amended/`.
   Regeneration was deliberately not performed (Phase 4 no-rerun rule).
2. No git repository is present; commit hashes are not applicable.
