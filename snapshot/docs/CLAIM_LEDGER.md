# TAN Claim Ledger

Complete claim register with status, evidence and allowed wording
(built 2026-09-04, Phase 4).  Status terms: REJECTED (counter-evidence
exists), NOT SUPPORTED (no adequate evidence), ESTABLISHED (adequate
consistent evidence), LIMITED SUPPORT (partial/seed-dependent),
SUPPORTED AS FRAMING (interpretive), STRUCTURALLY EXCLUDED (excluded by
the frozen equations), NOT ESTABLISHED.

| Claim | Status | Evidence | Allowed wording |
| --- | --- | --- | --- |
| h_t is a sufficient Markov state of TAN | REJECTED | Phase 1 (natural collisions, Lemma 1) | Do not claim |
| TAN is history-dependent (context enters the transition law) | ESTABLISHED | Phase 1 | Allowed |
| Gate-closed episodes restore exact LIF contraction (K = lambda) | ESTABLISHED | Phase 1 gate stats | Allowed |
| Attention (adaptive alpha) further amplifies collision divergence beyond uniform attention | ESTABLISHED (ensemble-level) | Phase 1 (median ratio 2.15 vs B3, p = 1.3e-32) | Allowed with ensemble framing |
| TAN necessarily has higher intrinsic dimension than LIF | NOT SUPPORTED | Phase 2 (shared zC frame: B4 1.589 < B1 1.674) | Do not claim; do not call PR intrinsic dimension |
| Event-locked local response geometry reorganises (effective rank / scale) | SUPPORTED | Phase 2 (zF B4 > B3, CIs separated; amplitude controls) | Allowed with "local fluctuation geometry / effective rank" wording and seed caveat |
| Attention universally improves memory / retrieval | REJECTED | Probe-1 | Do not claim |
| A fixed delay-line buffer can outperform TAN on fixed-position retrieval | ESTABLISHED | Probe-1 (B2 0.83-0.88 across seeds) | Allowed |
| Full TAN improves target-relevant retrieval under content-rich context vs no-attention control | LIMITED SUPPORT | Probe-1 v3 (pooled +0.0570, CI [+0.039, +0.076]; seed-attenuated) | Only the constrained wording in the dossier, verbatim |
| TAN is distractor-robust | NOT ESTABLISHED | Probe-1 (Delta_dist negative/inverted for B3/B4 cells; mechanism differs) | Do not claim |
| TAN performs genuine Q-K routing / query-conditioned selection | REJECTED | Probe-2 (JSD 0.0002, p = 0.92; argmax flip 0.000) | Do not claim |
| Scalar TAN performs surprise-conditioned saliency/amplitude reweighting | ESTABLISHED | Theory (kernel form) + Phase 1/2/Probe-2 | Allowed |
| Current TAN can perform query-conditioned binding | STRUCTURALLY EXCLUDED | Theory precheck + Probe-2 (E_i/E_j = x_i/x_j query-independent) | Do not claim |
| TAN provides a minimal laboratory for temporal-attention mechanism studies | SUPPORTED AS FRAMING | Entire project | Allowed as framing only |
| Higher-dimensional state implies higher computational power (general) | NOT SUPPORTED | Phase 2 vs Probe-1 | Do not claim |

Wording rules: "NOT SUPPORTED" must not be rewritten as "FALSE" unless a
true counter-proof exists (the only structural exclusions are: scalar
state sufficiency (rejected empirically), universal memory/retrieval
advantage, and Q-K routing/binding in the current scalar kernel
(excluded by equations)).
