# TAN Research Dossier

Research lifecycle record / experimental dossier / scientific handoff
document.  Built 2026-09-04 (Phase 4).  This is NOT a paper; it records
what was done, what was found, what was ruled out, and what must not be
claimed.

---

## 0. Executive Summary

**Research question.** Why study a minimalist temporal-attention spiking
neuron?  Because the question is not "does one more attention-like
component help a benchmark", but whether a *single neuron* can host
temporal attention dynamics whose computational character can be
identified mechanistically: what kind of memory, geometry and selection
does it actually implement, and what are its structural boundaries?

**Model.** TAN is a discrete-time single-neuron system with a windowed
temporal receptive field, surprise-gated attention-like weighting, and
leaky membrane dynamics:

```
X_t = [x_{t-W+1}, ..., x_t],        mu_t = (1/W) sum X_t
S_t = [x_t - mu_t - eps]_+          (surprise, scalar, rectified)
E_{t,i} = beta*Wq*Wk*S_t*x_i,       alpha_{t,i} = exp(E_{t,i}) / (sum_j exp(E_{t,j}) + 1e-9)
C_t = sum_i alpha_{t,i}*Wv*x_i      (context)
A_t = tanh(S_t)*C_t                 (gated current)
h_t = lambda*h_{t-1} + A_t,         spike if h_t > theta, then h_t <- 0
```

Frozen defaults: W=5, lambda=0.5, theta=0.5, Wq=2.0, Wk=Wv=1.0, beta=1.0,
eps=0.0; windows zero-padded at trial onset (see Model Specification).

**Main findings (restrained wording).**
1. The scalar membrane state is not a sufficient Markov state for TAN;
   history/context participates in the transition law (Phase 1).
2. Event-locked response geometry can re-organise locally, and Full TAN
   differs from a no-attention control in the matched full frame in
   effective rank and covariance scale - but this is NOT evidence of
   higher intrinsic dimensionality (Phase 2).
3. Passive memory must be distinguished from adaptive computation: a
   fixed delay-line buffer is a very strong positional-memory baseline and
   can outperform attention variants on fixed-position retrieval
   (Probe-1).
4. Probe-1 gives limited, condition-dependent evidence that Full TAN
   improves target-relevant retrieval in content-rich temporal context
   relative to the no-attention control (partial seed replication).
5. Probe-2 establishes a mechanistic boundary: with the scalar kernel
   E_i = c*S_t*x_i, the current TAN attention is
   surprise-conditioned amplitude reweighting, NOT genuine Q-K
   content-addressable routing.

**One-line mechanistic definition.** TAN is a prediction-error-gated,
history-dependent dynamical system with scalar attention-like saliency
weighting.

---

## 1. Model Specification (frozen, self-contained)

| parameter | value | meaning |
| --- | --- | --- |
| W | 5 | receptive-field length |
| lambda | 0.5 | membrane leak |
| theta | 0.5 | spike threshold |
| Wq, Wk, Wv | 2.0, 1.0, 1.0 | query/key/value linear maps |
| beta | 1.0 | inverse temperature of the softmax |
| eps | 0.0 | noise tolerance (dead zone) in surprise |
| reset rule | spike if h > theta, then h <- 0 | readout/reset |
| attention normalisation | softmax over W taps; denominator sum + 1e-9 | - |
| gate function | g(S) = tanh(S) | - |
| history convention | window includes current x_t; zero-padded at trial onset; h_0 = 0 | - |

Equations (as in Section 0).  Reference implementation:
`code/model/tan.py` (`TANNeuronAblation`, `LIFNeuron`, variant factory);
experiment scripts re-implement the identical equations vectorised
(parity-checked against `code/model/tan.py`; worst deviation 1.47e-10 in
h, spike agreement exact).

Model family used across experiments:
- B1 = LIF: h_t = lambda h_{t-1} + x_t (same readout/reset)
- B2 = LIF + fixed delay-line read of the window (weights recency ramp)
- B3 = TAN without attention: alpha_i = 1/W (uniform), C_t = mean(X_t)
- B4 = Full TAN (above)
- C0 = input-only control (no dynamics)

---

## 2. Experimental Lineage

```
Phase 1 (natural state collision; non-Markovianity)
  -> Phase 2 (event-locked local response geometry pilot)
  -> Probe-1 v1 (temporal distractor; XOR label)
  -> Probe-1 v2 (XOR retained; readout/endpoint revision)
  -> Probe-1 v3 (retrieval-cue target recall; FROZEN)
  -> Probe-1 amended audit (frozen 3-seed data)
  -> Probe-2 identifiability audit (query-conditioned binding)
  -> Phase 4 (this dossier)
```

**Why Probe-1 v1/v2 were abandoned (kept as design-evolution record).**
- v1: response-epoch-only readout collapsed (readouts predicted a single
  majority class); partly an optimizer bug (fixed) and partly an
  information-limited feature set for a feed-forward single neuron.
- v2: the XOR label is linearly inseparable from raw amplitude history
  (oracle-history linear readout ~0.44 < 0.5), so the task could not even
  be calibrated with a linear readout; a mathematical audit showed that
  with binary amplitude codes every two-variable relation (XOR/AND/OR)
  is either linearly inseparable or trivially reducible to query
  detection under the strict linear-readout protocol.
- v3 replaced the label with the single-variable target property
  (retrieval-cue target recall: target lo / target hi / absent), which is
  linearly learnable by an oracle (0.93 balanced 3-class) and keeps the
  query functional as the retrieval cue.  The rejected formulations are
  documented in the logs and CHANGELOG [1.3.0].

---

## 3. Phase 1 - State Sufficiency / Non-Markovianity

**Hypothesis tested.** Is the scalar membrane potential h_t a sufficient
(Markovian) state variable of TAN?

**Formal state.** s_t = (h_t, Xbar_t) with projection pi(h, Xbar) = h.
Lemma 1 (report): for S_t > 0 the map Xbar -> Phi(h, Xbar, x) is
non-constant on generic fibres, so pi is not a lumpable projection; h_t
is not a sufficient statistic and no single-valued vector field exists on
the scalar manifold.  Supporting propositions: LIF is exactly contracting
(Delta h' = lambda Delta h); gate-closed episodes restore K = lambda
exactly; spike reset erases h but not context memory (re-bifurcation).

**Experimental evidence (natural collision experiment; N = 160,000
histories, 4 stimulus families, collision tolerance 1e-4 in the
subthreshold band).** Pairs found per model: B1 92,600 / B2 19,234 /
B3 1,633,844 / B4 1,427,817; 1,500 analysed per model.  One-step
amplification on the continuous branch (mean K with 95% bootstrap CI):
B1 = 0.5000000 (exact; max deviation 1.8e-8); B2 = 4.47e3
[3.26e3, 5.92e3]; B3 = 1.235e4 [7.78e3, 1.83e4]; B4 = 2.047e4
[1.12e4, 3.57e4].  %K>1e3: 0 / 46.7 / 56.7 / 73.7.  Post-reset Markov
violation rate: 0 / 0.6 / 33.1 / 32.2 %.  Gate-closed collisions: median
K = 0.5 exactly for B3/B4.  Attention divergence predicts current
divergence: Pearson r(JSD, dA) = 0.87-0.89 (p < 1e-300); dA vs dH'
r = 1.000.  Cross-model (log10 K): B4 vs B3 p = 1.3e-32 (median ratio
2.15); B4 vs B2 p = 1.3e-63 (3.29); B2 vs B1 p < 1e-300.  Reset
re-bifurcation (843 fully resynchronised pairs): LIF stays synchronised
forever (machine exact); B2 re-diverges on the continuous branch then
returns to zero after the receptive-field flush; B3/B4 retain a gap
beyond the flush in ~25% of pairs.

**Correct interpretation.** History dependence is established.  It is NOT
claimed that attention alone causes non-Markovianity: B2 (linear buffer)
and B3 (no attention) are also history-dependent; attention modulates the
degree and the gate makes the dependence conditional on surprise.

---

## 4. Phase 2 - Local Fluctuation Geometry (pilot)

**Metrics.** Effective rank of local/event-locked response covariance
d_PR = (sum lambda_i)^2 / sum lambda_i^2, together with log Tr(C) and the
eigenvalue spectrum; velocity geometry d_D from adjacent-phase increments.
d_PR is the effective rank of the event-locked response covariance - it is
NEVER called intrinsic/manifold dimension.

**Design.** Sparse Poisson-style episode generator (burst structure,
amplitude diversity), T = 5000, burn-in 500, seeds 20260904/05/06,
3 seeds; coordinate frames zU (input-only), zC = [h, mu, S, A] (B1-B4),
zF = [h, mu, S, A, alpha_1..5, C] (B3/B4); global per-coordinate Z-score
(post-burn-in), per-seed centring before cross-seed pooling; no whitening;
NaN policy for zero-variance windows; pooled event-locked covariance at
fixed lag (never per-window PR averaging); amplitude controls (raw /
stratified / log-A residualised); event bootstrap CIs; recovery times on
isolated events; pre-registered outcome gates.

**Main results (374 pooled events).** Velocity-rank event value dDe:
B1 zC 1.674 (Delta 0.674), B3 zC 1.458, B3 zF 1.484, B4 zC 1.589, B4 zF
2.089 (Delta 1.089; event CI [1.87, 2.30]), C0 zU 1.055.  In the matched
full frame zF, B4 exceeds B3 (CIs fully separated) in effective rank and
in scale (event logTr 3.58 vs 2.48 nats; +1.09 nats); amplitude controls
survive and slightly strengthen the contrast (B4 - B3: +0.61 raw /
+0.74 stratified / +0.78 residualised); velocity-geometry peak is delayed
to tau ~ 4-6 (content-rich windows), not at the first pulse; recovery
tau_e-fold = 5.4 [4.6, 6.3] (B4 zF) < 8.5 [6.1, 10.6] (B3 zF).

**Critical limitation (explicit).** In the shared observable frame zC,
B4 does NOT exceed B1 (1.589 vs 1.674, robust across amplitude
controls): the event-locked expansion is generic to any h-bearing model
(LIF included).  Therefore the pre-registered TAN-specific expansion
claim is NOT supported, and one must not claim that TAN has higher
intrinsic dimensionality than LIF.  All Phase-2 language must use "local
fluctuation geometry / effective rank / effective computational
dimensionality".

---

## 5. Probe-1 - Temporal Distractor Task (v3, frozen)

**Research question.** Does history participate in computational
retrieval (not merely storage)?

**Task (v3).** 5-step trial: target at fixed t=5 with amplitude class cT
(lo U(0.55,1.05) / hi U(0.95,1.45); overlap, both straddle no fixed
relation to theta); 3 distractors at t=6..8 (condition A) or silence
(condition B); query at t=9 (retrieval cue, amplitude random, NOT in the
label); label = cT (lo/hi) or absent (catch), 3 classes.  Readout: linear
3-class softmax on the response trajectory [u9..u13, y9..u13] (t<9 never
visible).  Training clean+distractor 50/50; evaluation per condition.
Primary endpoint Delta_dist = acc(clean) - acc(distractor); B4 vs B3;
catch = negative control.  Seeds 20260904/05/06, train 4000/test 2000.

**Formal results (frozen 3-seed data; see audit_v3_amended/).** Per-seed
balanced 3-class accuracy (clean / distractor): B1 ~ chance
(0.32-0.36); B2 0.83-0.88 (strong, stable, all classes healthy); B3
0.32-0.47 with lo-blind cells (seed-3 clean 0.470 is INVALID_FOR_
INTERPRETATION); B4 0.33-0.43 in the distractor condition with absolute
elevation attenuating across seeds (0.427 / 0.398 / 0.342).  Pooled B4-B3
in the distractor condition: +0.0570 (95% CI [+0.0387, +0.0758],
n = 6000); per-seed differences +0.056 / +0.043 / +0.056.  Amended audit
(permutation-null shuffle audit, per-cell coverage statuses) is in
audit_v3_amended/; frozen-data reproduction deviated by <= 4.7e-4
(rounding only).

**Main conclusion (constrained wording - must be used verbatim).**
"Full TAN shows evidence of improved target-relevant retrieval under
content-rich temporal context relative to the no-attention TAN control;
the absolute effect is partially replicated across seeds, with
substantially weaker evidence in seed 20260906."

Forbidden wording: "TAN is robust to distractors"; "TAN has superior
memory"; "TAN universally outperforms memory baselines".  B2 is an
explicitly strong fixed-position memory baseline; the task measures
fixed-position retrieval, for which a delay line has structural
advantage.

---

## 6. Probe-2 - Query-Conditioned Binding Identifiability Audit

```
STATUS: TERMINATED / AUDIT_FAILED
```

(Not "experiment incomplete", not "implementation bug".)

**Research question.** Does the current query determine which historical
event is selected (Q -> K -> V binding)?

**Theoretical precheck.** With E_i = beta*Wq*Wk*S_t*x_i and scalar S_t,
for S_t > 0: argmax_i alpha_i = argmax_i x_i; the query identity enters
only as a common scale/sharpness factor and cannot change the relative
ordering of historical taps.

**Audit results (N=1000, seed 20260904, W=5).**
- B2 linear shortcut: balanced accuracy 0.779 [0.724, 0.836] -> FAIL;
- B3 linear shortcut: 0.709 [0.644, 0.775] -> FAIL;
- B4 attention: hit rate 0.511 [0.481, 0.542] with hit(q=A) = 0.000 and
  hit(q=B) = 1.000 -> FAIL; target/distractor contrast 0.034 -> FAIL;
  query-conditioned JSD 0.0002, permutation p = 0.92 -> FAIL;
- counterfactual: same history, q=A vs q=B, argmax flip rate = 0.000;
- generator: position balance ok, I(q;y) ~ 3e-4; single-tap conditional
  bias at B-code values 0.45-0.60 (structural note: the encoding is not
  strictly leakage-free, though single-tap BA < 0.58).

**Mechanism diagnosis (three layers).**
- Layer 1 (encoding): the B-key codes carry larger amplitude, so B2/B3
  can exploit amplitude and B4 selects the larger-amplitude B event;
  hit(q=A)=0, hit(q=B)=1 is amplitude argmax, not routing.
- Layer 2 (self-attraction): query amplitudes 3.0/4.0 exceed all history
  codes, so the softmax concentrates on the current tap
  (P(argmax = query tap) = 1.0): self-attraction is a mechanistic
  consequence of the chosen scalar encoding.
- Layer 3 (structural kernel limitation): with E_i = q_scalar * x_i the
  ratio E_i/E_j = x_i/x_j is query-independent for fixed history; query
  changes weighting strength, not selected content.  This layer survives
  any re-encoding while the frozen kernel is unchanged.

**Mechanistic boundary (project-level).**
History dependence != attention routing != semantic binding.

---

## 7. Claim Ledger

See `docs/CLAIM_LEDGER.md` (complete table).

## 8. Epistemic Boundary

**Established.** History dependence of TAN dynamics (Phase 1);
event-locked expansion of response geometry is generic to h-bearing
models and the input floor is low (Phase 2); B2 is a strong positional-
memory baseline (Probe-1); scalar attention ranks history by amplitude
and is not query-conditioned (Probe-2); surprise-conditioned saliency
weighting with exact gate-closed contraction (Phase 1 gate stats).

**Limited evidence.** B4 > B3 effective-rank/scale advantage in the
matched full frame (Phase 2); B4 target retrieval advantage over B3 in
the content-rich distractor condition (Probe-1; seed-attenuated).

**Not established.** Higher intrinsic dimensionality than LIF (rejected
in the shared frame); distractor robustness / universal memory advantage
of TAN; any claim that the Phase-2 geometry translates to a general
computational benefit.

**Structurally excluded (by the frozen equations).** Genuine
query-conditioned Q-K content selection with scalar S_t; single-tap
independence of the composite encoding used in Probe-2 was also not
achieved.

**Unknown.** Whether any behavioural task exists on which the Phase-2
geometry yields an attention-specific advantage that is seed-stable;
what the minimal mechanistic extension is that enables genuine
query-conditioned selection.

---

## 9. Future Research Map (hypotheses only, NOT TESTED)

- A. Metric attention: E_i = -||Q - K_i||^2 in one dimension - is a
  scalar metric sufficient for content matching?
  `FUTURE HYPOTHESIS - NOT TESTED`
- B. Minimal 2-D representation: does a 2-D query/key representation
  suffice for content discrimination (Q perpendicular to K)?
  `FUTURE HYPOTHESIS - NOT TESTED`
- C. Opponent channels: can dual-channel polarity representations break
  the monotone amplitude ordering of scalar attention?
  `FUTURE HYPOTHESIS - NOT TESTED`

None of these were implemented or tested in this research lifecycle.
