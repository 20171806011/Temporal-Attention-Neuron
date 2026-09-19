# Next-Agent Brief

For a researcher who has never seen this project: read this first
(5-10 minutes), then `code/RESEARCH_ARCHIVE_INDEX.md` for file paths and
`docs/TAN_RESEARCH_DOSSIER.md` for details.

## A. What this project is

TAN (Temporal Attention Neuron) is a single-neuron, discrete-time model
that fuses a windowed, surprise-gated, attention-like weighting with
leaky spiking dynamics:

```
X_t = window of W=5 recent inputs (incl. current)
S_t = [x_t - mean(X_t) - eps]_+            (scalar surprise)
alpha = softmax(beta*Wq*Wk*S_t*X_t)        (saliency weights)
A_t = tanh(S_t) * sum(alpha*Wv*X_t)        (gated current)
h_t = lambda*h_{t-1} + A_t; spike if h > theta, reset h <- 0
```

lambda=0.5, theta=0.5, Wq=2.0, Wk=Wv=1.0, beta=1.0, eps=0.0.
Reference implementation: `code/model/tan.py`.

The project asks mechanistic questions (not benchmark questions): what
kind of memory, response geometry and selection does this minimal system
implement, and what are its structural limits?  Companion models: B1 LIF,
B2 LIF + fixed delay-line buffer, B3 TAN without attention (uniform
weights), B4 full TAN, C0 input-only control.

## B. What has been established (frozen results)

1. **Phase 1 (state sufficiency):** h_t is not a sufficient Markov
   state.  Natural-collision experiment (delta = 1e-4, 1,500 pairs per
   model): one-step continuous-branch amplification K: LIF 0.5 (exact),
   LIF+buffer 4.5e3, TAN-noAttn 1.2e4, Full TAN 2.0e4; gate-closed
   episodes restore K = 0.5 exactly; reset erases h but not context
   memory.  History dependence is established for every window-carrying
   model (not only TAN).
2. **Phase 2 (local response geometry):** event-locked effective rank of
   the response covariance (d_PR, log Tr, spectrum, velocity version) is
   a well-defined observable.  Event-locked expansion exists for every
   h-bearing model and is NOT TAN-specific (in the shared 4-D frame
   B4 ~ B1).  In the matched full 10-D frame B4 > B3 (rank + scale,
   separated CIs, robust to amplitude controls, delayed peak at
   tau ~ 4-6).  PR must never be called intrinsic dimension.
3. **Probe-1 (temporal distractor retrieval):** fixed-position retrieval
   is dominated by a fixed delay-line (B2, balanced 3-class 0.83-0.88);
   B4 shows limited, seed-attenuated evidence of improved retrieval over
   the no-attention control in content-rich (distractor) windows
   (pooled +0.0570, CI [+0.039, +0.076]).  B3 has lo-blind cells; B1 ~
   chance.
4. **Probe-2 (identifiability audit):** with the frozen scalar kernel
   E_i = beta*Wq*Wk*S_t*x_i, the query identity enters only as a common
   scale (S_t), so argmax(alpha) = argmax(x): attention is
   surprise-conditioned amplitude reweighting.  Query-conditioned
   selection was not observed (JSD 0.0002, p = 0.92; same-history
   argmax flip rate 0.000).  Linear shortcuts existed in the audited
   encoding (B2 0.779, B3 0.709).

## C. What has been ruled out (do not revive without new evidence)

- Scalar state sufficiency (rejected empirically, Phase 1).
- "TAN has higher intrinsic dimensionality than LIF" (not supported;
   PR is an effective-rank statistic, and the shared-frame contrast
   fails).
- Universal memory/retrieval advantage of attention (Probe-1: B2 wins
   fixed-position retrieval).
- "TAN is distractor-robust" (not established; condition effects run
   opposite to the naive robustness reading).
- Genuine Q-K content-addressable routing or query-conditioned binding
   in the current scalar kernel (structurally excluded, Probe-2).

## D. What must NOT be done

- Do not tune Probe-2 (no re-encoding until it "passes"; the failure is
  structural, see THEORY_PRECHECK.md).
- Do not redefine old experiments after changing encodings and present
  them as the original result.
- Do not increase classifier complexity to raise accuracy; the linear
  readout is a frozen protocol constraint.
- Do not modify frozen baselines/models/parameters/seeds/data.
- Do not delete negative results (Probe-2 AUDIT_FAILED must remain).
- Do not rewrite limited evidence as universal claims (claim ledger in
  docs/CLAIM_LEDGER.md governs wording).
- Do not present architecture extensions as results of the original TAN.

## E. The legitimate next research question

```
What is the minimal additional mechanism required for
query-conditioned temporal binding?
```

Not implemented in this phase.  Any future experiment must start from a
new, explicitly stated hypothesis and pass the identifiability-style
audits used here (no fixed-position / amplitude / query-leakage /
uniform-integration / classifier / self-attraction shortcuts), including
the counterfactual alpha(X, q=A) != alpha(X, q=B) on identical history.
Candidate hypotheses (NOT TESTED, framing only): metric attention
E = -||Q-K||^2; minimal 2-D query/key representations; opponent/polarity
channels to escape monotone amplitude ordering.

Project lifecycle status: ARCHIVED / HANDOFF READY.
