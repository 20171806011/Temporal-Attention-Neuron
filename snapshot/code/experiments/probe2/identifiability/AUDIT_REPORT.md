# Probe-2 Identifiability Audit — Report (terminated)

Date: 2026-09-04 · Seed 20260904 · N = 1000 · W = 5 · sigma_noise = 0.05
Script: `probe2_identifiability_audit.py` (frozen, byte-identical)
Raw console output: `AUDIT_OUTPUT.txt`

## Task candidate under test

E_A = (K_A, V_A), E_B = (K_B, V_B); composite scalar code
`x(E) = K_amp + V*0.25` with K_A = 1.0, K_B = 2.0, V in {-1,+1} iid fair;
A/B placed on two distinct random history slots (uniform over 12
arrangements), remaining slots N(0, 0.05^2); query x_t = 3.0 (q=A) or 4.0
(q=B); label y = V_q.

## Generator audit

- position balance: slot counts [506, 476, 492, 526], chi2 p = 0.440 -> ok
- I(q; y) = 3.05e-4, chi2 p = 0.473 -> ok (query-label independence holds)
- single-slot balanced accuracy: best slot 0 at 0.555 [0.487, 0.625]
  (< 0.58 engineering gate -> ok)
- **structural note (recorded, generator is NOT leakage-free by the strict
  criterion):** per-slot conditional bias |E[V_q | x_i bin]| reaches
  0.45-0.60 at B-code values (e.g. x_i = 2.25 => with q=B, y = +1
  deterministically): a single tap carrying a B code carries real
  conditional information about y on the 50% of q=B trials.  The strict
  "E[V_q|x_i] ~ 0" idealisation of the spec does not hold for this
  composite encoding.

## Audit A — B2 fixed buffer + linear readout

- balanced accuracy: **0.7789** [0.7236, 0.8358] -> **FAIL** (red line
  0.58)
- meaning: fixed delay-line storage + linear readout extracts large
  task-relevant signal from this encoding; success cannot be attributed to
  dynamic routing.

## Audit B — B3 uniform integration + linear readout

- balanced accuracy: **0.7094** [0.6435, 0.7745] -> **FAIL**
- meaning: uniform temporal integration without attention also solves the
  task far above chance: storage/integration shortcut exists.

## Audit C — B4 attention

- C1 hit rate (argmax on target history tap): **0.511** [0.481, 0.542]
  -> FAIL (>= 0.50 required at CI).  Split by query: hit(q=A) = 0.000,
  hit(q=B) = 1.000: attention mechanically selects the B event (larger
  amplitude), which is the target only when q=B.  This is amplitude
  argmax, NOT query-conditioned selection.
- C2 target/distractor contrast: R = **0.034** -> FAIL (>= 2.0 required):
  attention systematically favours the (amplitude-larger) distractor.
- C3 query-conditioned routing: JSD(mean alpha | q=A, | q=B) = **0.0002**,
  permutation p = **0.92** -> FAIL: attention distribution has no
  detectable dependence on query identity.
- diagnostic: argmax over the full 5-tap window falls on the current query
  tap in 100% of trials (self-attraction: x_q in {3,4} exceeds all
  historical codes <= 2.25).

## Audit D — counterfactuals

- query permutation: hit 0.511 (unchanged from 0.511)
- value permutation: hit 0.511
- position-label permutation: hit 0.276 (~chance, metric sanity ok)
- same history, q=A vs q=B: mean per-trial JSD = 0.0043, **history-argmax
  flip fraction = 0.000**
- meaning: with identical history, changing the query changes attention
  sharpness but never the attention winner: no query-conditioned routing.

## Mechanism analysis (three layers)

1. Task shortcut layer: asymmetric key amplitudes (K_A=1, K_B=2) make the
   B event the dominant tap; B2/B3 linear shortcuts (0.779 / 0.709) follow
   from "decode the largest tap" being correct on q=B trials.
2. Model layer: with E_i = c*S_t*x_i and scalar S_t, attention ranks taps
   by x_i; query identity enters only as a common scale.  Hence
   argmax(alpha | q=A) = argmax(alpha | q=B) for identical history
   (flip rate 0.000).
3. Architectural layer: the frozen TAN query is not a vector that matches
   keys; Q -> K matching does not occur.  Query amplitudes above all
   history values additionally cause self-attraction to the current tap
   (100% of trials).

## Final status

```
STATUS: AUDIT_FAILED
```

No formal Probe-2 experiment was performed.  No TAN parameters were
tuned.  No Phase 1/2/Probe-1 results were modified.  The audit was a
predetermined stop state with explanatory power: the current scalar TAN
attention provides surprise-conditioned amplitude reweighting, not
genuine Q-K routing.
