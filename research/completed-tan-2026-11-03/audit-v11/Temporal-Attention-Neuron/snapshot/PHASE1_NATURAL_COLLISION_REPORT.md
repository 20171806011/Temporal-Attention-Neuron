# TAN Phase 1 — Natural State-Collision Experiment (Non-Markovianity of h_t)

**Experiment**: `code/experiments/natural_collision.py` · **Log**: `results/logs/natural_collision.log`
**Figures**: `results/figures/fig22_natural_collision.{png,pdf}`, `results/figures/fig23_reset_rebifurcation.{png,pdf}` (mirrored to `manuscript/figures/`)
**Tables**: `results/tables/natural_collision_{summary,pairs,correlations}.csv` (mirrored to `data/experiment_results/`)
**Phase-2 dataset**: `data/experiment_results/natural_collision_ensembles.npz`

Author: Li Zexu (with DeepSeek), University of Leeds. Deterministic seed 20260726; runtime ≈ 25 s.

---

## 1. Question and headline answer

> If the scalar membrane potential `h_t` were a *sufficient* (Markovian) state
> variable, two systems with different histories but
> `|h^A_{T-1} - h^B_{T-1}| < δ` would stay contracted after an identical
> probe: `|h^A_T - h^B_T| ≤ δ`.

Measured one-step amplification `K = Δh̃_T / Δh_{T-1}` (continuous branch,
1500 natural collision pairs per model, δ = 10⁻⁴):

| model | mean K [95% CI] | median K | % K>1 | % K>10⁴·δ⇒K>10 | % K>10³ |
|---|---|---|---|---|---|
| B1 LIF | **0.5000000** (exact, max |K−λ|=1.8e-8) | 0.5 | 0% | 0% | 0% |
| B2 LIF+DelayLine | 4.47e3 [3.3e3, 5.9e3] | 8.96e2 | 99.9% | 99.3% | 46.7% |
| B3 TAN-noAttn | 1.24e4 [7.8e3, 1.8e4] | 1.37e3 | 87.0% | 86.4% | 56.7% |
| B4 Full TAN | 2.05e4 [1.1e4, 3.6e4] | 2.95e3 | 90.3% | 89.9% | 73.7% |

**The Markovian claim is refuted for every history-carrying model, including
the purely linear delay-line LIF** (whose true state is `(h_t, X_t)`, not
`h_t`). Full TAN amplifies beyond the uniform-attention and linear-buffer
variants (median-K ratios 2.15 vs noAttn, p = 1.3e-32; 3.29 vs buffer,
p = 1.3e-63, Mann–Whitney on log10 K), and its bifurcation is mechanistically
tied to attention: Pearson r(JSD(α_T) vs ΔA_T) = 0.87–0.89 (p < 1e-300) and
ΔA_T → Δh̃_T is essentially deterministic (r = 1.000). Divergence is
*conditional*: when the probe is predictable relative to local history
(S_T = 0, gates closed) the current vanishes and K ≡ λ exactly — TAN is a
conditional non-Markovian system, not an unconditional one.

---

## 2. Formal statements

**Lemma 1 (State insufficiency of h_t — TAN).** Let
Φ_TAN(h, x, X̄) = λh + tanh(S(x, X̄))·C(x, X̄) denote the continuous-branch
update of the Full TAN with window content X̄ = (x_{t-W+1},…,x_{t-1}),
S = [x − mean(X̄, x) − ε]₊, C = Σᵢ αᵢ(X̄,x)·xᵢ. For any h, the restriction of
Φ_TAN to the collision fibre F_h = {(h, X̄): X̄ ∈ ℝ^{W-1}} is non-constant
generically: for open sets of window pairs X̄ ≠ X̄′, Φ_TAN(h, x, X̄) ≠
Φ_TAN(h, x, X̄′). Hence the projection π: (h, X̄) ↦ h is **not a lumpable /
consistent reduction** (Kemeny lumpability fails): no single-valued
transition map on ℝ reproduces the dynamics, i.e. h_t is not a sufficient
statistic and the dynamics does not define a single-valued vector field on
the scalar state manifold.

*Proof sketch.* (i) Fix h and a probe x > ε. Pick X̄ = (x̄,…,x̄) and
X̄′ = (x̄+2Δ, x̄−Δ…, x̄−Δ): both have equal means, so S is equal, but the
softmax compatibility scores Q·Kᵢ = WqWk S xᵢ differ ⇒ α(X̄) ≠ α(X̄′) with
positive probability mass on different taps; by continuity this holds on an
open neighbourhood of such a pair. (ii) Since Vᵢ = xᵢ and C = ΣαᵢVᵢ,
C(X̄) − C(X̄′) = Σ(αᵢ−αᵢ′)xᵢ + Σαᵢ′(xᵢ−xᵢ′) ≠ 0 for a generic choice;
with tanh(S) > 0 the current difference ΔA = |Φ−Φ′| is bounded away from
zero on an open set, while the state gap Δh = 0 by construction. (iii)
Therefore the one-step map from the scalar coordinate h is multi-branched
(Δh = 0 ↦ Δh ~ O(ΔA)); the local vector field is not single-valued on ℝ and
the fibre F_h is not contractible in the induced quotient dynamics. Any
valid reduced state must carry at least the W−1 recent inputs (or an
information-equivalent coordinate such as S together with the attention
scores α), giving a minimal Markov order ≥ W−1 = 4 for W = 5.

**Proposition 2 (LIF is exactly contracting).** For B1, u_T = λh_{T-1} + x_T
with a common x_T gives, on every collision pair, Δh̃_T = λ·Δh_{T-1}
identically (measured max deviation 1.8e-8 ≈ float precision), and a common
post-probe spike resets both systems to the *identical* state h_T = 0. h_t
is a sufficient statistic for LIF: equal h and equal future inputs imply
identical futures (verified to machine precision over 843 resynchronised
pairs).

**Proposition 3 (Conditional Markovian restoration under predictive
silence).** Whenever the probe is below the local context, S_T = 0, the gate
tanh(S_T) = 0 kills the current and u_T = λh_{T-1}: the TAN family behaves
exactly like the LIF contraction map on those episodes. Measured: median
K = 0.5000000 for gate-closed collision pairs (both B3 and B4).

**Proposition 4 (Reset erases h, not memory).** After a common double-spike
at T all four models start the continuation from the identical observable
state h_T = 0 with identical inputs. LIF stays synchronised forever;
LIF+DelayLine re-bifurcates immediately on the continuous branch
(⟨|Δu|⟩ = 0.115 at T+1) but re-converges once the differing window content
is flushed at T+4; the TAN family re-bifurcates (⟨|Δu|⟩ = 0.087 / 0.054 at
T+1) and **still carries ⟨|Δu|⟩ ≈ 3e-3 with ~26% of pairs above 1e-3 at
T+8**, i.e. beyond receptive-field flush, via spike/reset hysteresis of the
pre-flush gap. h = 0 is therefore not an absorbing equivalence class for TAN.

---

## 3. Protocol (as executed, 100% paper defaults)

* Model equations and defaults exactly as in the manuscript (W = 5, λ = 0.5,
  θ = 0.5, Wq = 2.0, Wk = Wv = 1.0, β = 1.0, ε = 0; softmax denominator
  Σexp + 1e-9; S = [x − mean(window) − ε]₊ with the window including x;
  spike if u > θ, reset h ← 0 — matching `code/model/tan.py`).
* **Stimulus pool**: N = 160,000 random histories of length T−1 = 9
  (4 × 40,000 from gauss / step / ramp / sine families). No artificial
  pairing anywhere.
* **Collision search (per model)**: unordered pairs with both h₉ in the
  subthreshold band (0.05, θ) — reset-induced h = 0 coincidences are thereby
  excluded — and |h₉ᴬ − h₉ᴮ| < δ = 10⁻⁴. Found: LIF 92,600; LIF+Buffer
  19,234; TAN-noAttn 1,633,844; Full TAN 1,427,817 pairs; 1,500 analysed
  per model (fixed subsample).
* **Probe**: one x* ~ U(0.5, 3.0) per pair, identical for A and B, at
  T = 10; then 8 common continuation inputs U(0.3, 1.8) for the
  reset-re-bifurcation analysis.
* **Divergence metric** — the protocol excludes trivial reset coincidences;
  the threshold+reset is a readout discontinuity shared by all four models.
  The headline one-step divergence is therefore measured on the
  continuous branch (pre-reset membrane potential u₁₀ = λh₉ + A₁₀):
  Δh̃_T = |u₁₀ᴬ − u₁₀ᴮ|. Post-reset divergence, spike-outcome classes and
  reset re-synchronisation are tabulated separately and honestly.
* Onset windows are zero-padded (silent pre-trial); h₀ = 0.

## 4. Main results

**Amplification (Task 3)** — table above. Non-parametric cross-model
comparisons on log10 K: TAN vs noAttn p = 1.3e-32; noAttn vs buffer
p = 2.1e-3; TAN vs buffer p = 1.3e-63; buffer vs LIF p < 1e-300.
Heavy tails: mean ≫ median; geometric means are 1483 (TAN), 562 (noAttn),
896 (buffer).

**Probe outcomes (T = 10) & post-reset Markov violation**

| model | none | one spiked | both | post-reset violation \|h₁₀ diff\|>δ |
|---|---|---|---|---|
| LIF | 0% | 0% | 100% | 0.0% |
| LIF+Buffer | 0.1% | 0.5% | 99.4% | 0.6% |
| TAN-noAttn | 38.9% | 7.1% | 54.1% | 33.1% |
| Full TAN | 33.3% | 8.6% | 58.1% | 32.2% |

Median Δh̃_T (one probe step): LIF 2.5e-5 (i.e. λδ), LIF+Buffer 4.1e-2,
TAN-noAttn 5.9e-2, Full TAN 1.28e-1 — a ~5,000× state-space expansion in one
step for the TAN family against δ.

**Gate-state stratification (B3/B4):** gates both open 74.5%/75.7%, one open
12.7%/14.6%, both closed 12.8%/9.7%; median K by state — both-open
1.6e3/3.5e3, one-open 3.0e3/2.9e3, closed = 0.5000000 for both (Prop. 3).

**Attention mechanism (Task 4, Full TAN ensemble, n = 1500):**
JSD(α₁₀ᴬ‖α₁₀ᴮ): mean 0.028 nats, median 0.009, max 0.376; L1 mean 0.262;
gate-closed origin pile n = 146 (JSD = ΔA = 0). Correlations:

| quantity | Pearson r | p | Spearman ρ | p |
|---|---|---|---|---|
| JSD vs ΔA₁₀ (all) | 0.870 | <1e-300 | 0.813 | <1e-300 |
| JSD vs ΔA₁₀ (both gates open, n=1135) | 0.894 | <1e-300 | 0.778 | 1.4e-230 |
| JSD vs Δh̃₁₀ (all) | 0.870 | <1e-300 | 0.812 | <1e-300 |
| ΔA₁₀ vs Δh̃₁₀ (all) | 1.000 | <1e-300 | 1.000 | <1e-300 |

The synaptic-current gap is the *transducer* of the bifurcation (r = 1.000),
and the attention divergence is its principal predictor; no h-derived
quantity can be, because h is identical by construction on the collision
ensemble.

**Reset re-bifurcation (n = 843 fully resynchronised pairs):** continuous
⟨|Δu|⟩ at T+1 / T+8: LIF 0 / 0 (machine exact); LIF+Buffer 0.115 / 0;
TAN-noAttn 0.054 / 3.2e-3; Full TAN 0.087 / 3.1e-3 (25.5% of pairs still
> 1e-3 at T+8). Post-reset ⟨|Δh|⟩ at T+8: 3.9e-3 (noAttn), 3.7e-3 (TAN).

## 5. Interpretation

1. **The collision test separates the memory architectures.** A scalar
   leaky integrator (LIF) is exactly contracting and reset-synchronised
   states are absorbing: h is truly sufficient. Any architecture whose
   update reads the windowed past — *even the linear delay line* — maps the
   h-collision fibre non-trivially, because the current injected at T
   depends on content that h cannot encode. For TAN the effective state
   dimension is at least 1 + (W−1) (h plus the recent context), consistent
   with Prop. 1.
2. **Attention is an amplifier and an orthogonaliser, not the only source
   of history-dependence.** The natural-collision data show graded — rarely
   disjoint — attention differences (max JSD 0.376 nats; only 2/1500 pairs
   exceed 0.5·ln2). But those differences are tightly coupled to the current
   gap (r ≈ 0.89 within fully gated-open pairs) and attention raises median
   amplification 2.15× over uniform weighting and 3.29× over the passive
   buffer, shifting 74% of collision pairs into the K > 10³ explosive tail.
   The interpretable statement for the manuscript is therefore: *α carries
   context information that h cannot, and this information is causally
   transduced into the membrane through A_T* — not that natural histories
   produce exactly disjoint attention supports (that would require
   adversarial stimuli).
3. **Non-Markovianity is conditional (predictive-coding behaviour).** When
   the probe is not surprising relative to local history the gate shuts and
   K = λ exactly: the neuron *chooses* when to be Markovian. This is the
   dynamical signature of the ε-dead-zone: predictable context ⇒ LIF-like
   contraction; novel context ⇒ attention-mediated bifurcation. The median-K
   ladder over gate-open pairs (buffer 9e2 → noAttn 1.4e3 → TAN 2.9e3)
   quantifies the functional gain of each architectural ingredient.
4. **Reset is not forgetting.** h = 0 after a common double-spike is an
   absorbing state only for LIF; the TAN family re-bifurcates from it
   (Prop. 4, fig23). Any observer that reads only spikes/h and discards the
   window loses information that changes the very next input current.
5. Post-reset divergence (≈1/3 of pairs violate the δ-bound at T) shows the
   claim is not an artefact of the continuous-branch convention: spike/no-
   spike bifurcations (8.6% one-side firing for TAN) produce order-θ
   post-reset gaps from δ-collisions.

## 6. Phase 2 interface (PR / TwoNN intrinsic-dimension measurement)

`natural_collision_ensembles.npz` stores, per model and per analysed pair,
both members' full 18-step trajectories of h, u, A (and S, α for the TAN
family), the collision gaps, the common continuation inputs and the
resync-subset mask. This provides the exact ensembles needed for Phase 2:

* *Collision-anchored local clouds.* Each collision pair gives two points
  with equal h but different context — the fibre F_h. Augmenting the state
  as z = (h, S, α₁…α₅, C) and measuring TwoNN intrinsic dimension on clouds
  grown from the T+1..T+4 continuation windows yields the local dimension
  of the true state manifold; subtracting the trivial scalar direction tests
  how many memory coordinates the fibre actually contributes (expectation
  ≈ W−1 while h-only clouds give 1).
* *Participation ratio.* PR = (Σλᵢ)²/Σλᵢ² of the covariance of
  z-trajectory velocities (or of the Jacobian of the augmented map) quantifies
  the *effective computational dimension*; comparing PR across B1→B4 on the
  same continuation streams isolates the dimensionality contributed by
  attention versus the passive buffer.
* *Consistency check.* The measured K-tail and gate-state stratification
  provide ground-truth "hard episodes" against which Phase-2 dimension
  estimates can be stratified (e.g., dim(fibre | gate open) vs
  dim(fibre | gate closed) — predicted 4 vs 0).

## 7. Reproduction

```bash
cd TAN_Final_Submission
python code/experiments/natural_collision.py            # full run, seed 20260726
# optional:
python code/experiments/natural_collision.py --eps 0.1 --cap 3000 --n-per-gen 20000
```

Dependencies: numpy, scipy, matplotlib only.
