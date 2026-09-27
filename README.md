# Temporal Attention Neuron (TAN)

An independent undergraduate research project by **Li Zexu** (BSc Physics, University of Leeds).

**Status:** exploratory work. The manuscripts in this repository are **unpublished, have not been peer-reviewed, and are not ready for submission** (see `ERRATA.md`, ERR-16 and ERR-18–ERR-20). All results come from simulations of small, hand-designed models; no biological data are used.

**Use of AI tools:** implementation, internal audits and much of the manuscript drafting were done with extensive help from AI coding agents (OpenAI Codex, Google Antigravity). The author takes responsibility for the content. Where this repository says "audited", it means internally audited by the author with these tools, not independently reviewed.

The earlier repository `20171806011/-TAN-` is archived; this repository replaces it.

---

## Question

Which minimal mechanisms let a single spiking unit, or a small circuit, **select**, **hold** and **combine** continuous values?

TAN is a scalar-input spiking neuron. A rectified surprise signal, $S_t = [x_t - \mu_t - \varepsilon]_+$ (where $\mu_t$ is the mean of a five-step input window), gates a softmax weighting over that window.

## What was found

The results below are sufficiency results within one hand-designed model family. Several of them follow directly from the model's definition and are best read as worked examples, not as new neural mechanisms.

1. **TAN-I** (manuscript 1; `release/TAN_I_Mechanistic_Analysis.pdf`, 28 pages).
   - Across 160,000 simulated histories, distinct histories reached the same membrane potential (within a numerical tolerance of $10^{-4}$) and then diverged; 1,500 such pairs were analysed per model, with a mean expansion factor of about $2.05\times10^4$ for the full model. The membrane potential alone is therefore not a sufficient state. This is expected for a model with a finite input window; the full state (potential plus window) is still Markovian.
   - Probe-2: with a positive scalar attention kernel, the counterfactual winner-flip rate was 0.000 over 1,000 trials (JSD 0.0002, permutation $p = 0.92$). The general statement is Theorem 5.1 of manuscript 2: a positive scalar kernel preserves the order of the scores, so it cannot route by content.
   - Probe-1 (distractor task): limited support only. In the distractor condition, the full model's balanced accuracy (mean over three seeds) was 0.389, against 0.337 for the version without attention; the difference computed on the pooled trials was +0.057 (95% CI +0.039 to +0.076). The full model was close to chance in one seed, and a simple buffer baseline scored 0.843.
2. **TAN-II** (manuscript 2; `release/TAN_II_Minimal_Architectural_Ladder.pdf`, 28 pages).
   - A stepwise set of additions (opponent competition, two-dimensional query–key rotation, key–value binding, winner-take-all readout, and a combiner) is sufficient for routing, binding and composition in this model family.
   - The composition stage uses hand-written arithmetic nodes. Its zero combiner error holds only when both channels are routed correctly (6,608 of 30,000 histories, 22%).
3. **Phase P1-A** (60,000 trials; 7 models; 420,000 evaluations; results in `p1_experiments/results/run_p1a_rev2/`).
   - A one-dimensional distance kernel, $-(q-k)^2$, selected the queried key in every noise-free trial under hard (argmax) selection, including middle keys.
   - It made the same routing decisions as the two-dimensional model at every noise level tested: 98.76% of trials had both channels correct at relative noise $\sigma = 0.01$, and 79.86% at $\sigma = 0.05$ (the query-noise SD is $2.2\sigma$).
   - This corrected the earlier assumption that two dimensions were needed.
   - Signed scalar dot-product kernels never selected a middle key (0 of 6,705 middle-key queries).
4. **Phase P1-B** (internally pre-registered, Protocol V2.6; 96,000 episodes).
   - The pre-registration derived a population bound of 7/12 (about 58%) on two-item joint delivery for the baseline carrier, so it was predicted to miss the 80% floor.
   - It delivered both stored values in 2.75–2.90% of episodes, and both pre-registered floors (80% and 40%) failed.
   - Because the carrier failed, the planned comparison of readouts is floor-limited and **inconclusive**.
   - Single-unit spiking decoders had composition errors of 2.25–2.53 (cell means), a limitation of this readout channel.
   - One hand-built pair of histories (fixture G1.5) reaches the same subthreshold state, showing that the subthreshold map is not one-to-one. It is a single constructed example, not a general erasure result.
   - Final report: `p1_experiments/p1b/P1B_FINAL_SCIENTIFIC_CLOSURE_REPORT.md`.
5. **Stage 1 prototype** (exploratory; `snapshot/manuscript/`).
   - A single TAN neuron drove a one-dimensional phototaxis agent. Over 50 seeds it stopped about 13 units from the light peak, against about 38 for an LIF-driven agent.
   - The motor rule only ever moves the agent forward, so the agent stops near the peak; it cannot hover around it, and the manuscript's *E. coli* analogy does not hold.
   - The baselines were LIF, adaptive LIF, and Elman RNN / GRU networks (hidden size 8) trained once, by imitation, on a seed-0 TAN trajectory. The LIF threshold was set by hand. No other spiking-network or Transformer baseline was run. This is a demonstration, not evidence of an advantage.

## Known issues

- The manuscripts were **not** edited in the 2026-09-27 corrections. Their remaining problems (numbers that do not match the data, an equation that does not match the code, and overstated wording) are listed in `ERRATA.md`: ERR-18 (manuscript 2), ERR-19 (manuscript 1 and the Stage 1 manuscript) and ERR-20 (frozen snapshot documents and the release manifest).
- The manuscripts do not yet contain an AI-use statement.
- Older documents in `snapshot/` and `release/`, and the Stage 1 manuscript, use stronger language ("proves", "certified", "breakthrough", "publication-ready", "fusion of Transformer attention and SNNs") than the evidence supports. Where they disagree with this README or `ERRATA.md`, this README and `ERRATA.md` apply.
- The P1-A documents mixed figures from a superseded first run with the final (rev2) data. This has been corrected (ERR-17). Hash-locked P1-A files were not edited; their remaining small inconsistencies are listed in ERR-17. The files at the top level of `p1_experiments/results/` are first-run outputs; use `p1_experiments/results/run_p1a_rev2/`.
- `manifest_after.json` is a historical record of the 2026-09-19 tree and no longer matches the repository. `scripts/generate_manifest_after.py` compares against the original local review archive, so it cannot be run from a clone.
- Related work is not yet covered properly. Missing areas include the Neural Engineering Framework (Eliasmith and Anderson), synaptic theories of working memory (Mongillo et al., 2008) and spiking Transformer models (for example Spikformer).

## Repository layout

```
release/          compiled TAN-I and TAN-II PDFs (28 pages each) and the release manifest
snapshot/         Stage 1 code and manuscript (code/, manuscript/); TAN-I and TAN-II LaTeX
                  sources (paper/, paper2/); data (data/); figures fig01–fig27 (results/figures/)
p1_experiments/   Phase P1-A code, protocols and reports (top level; final results in
                  results/run_p1a_rev2/) and Phase P1-B (p1b/)
s41cd/ … s43a/    TAN-II sprint reproduction scripts
scripts/          audit, ledger and manifest scripts
CURRENT_RESEARCH_STATE.md, RESEARCH_CHRONOLOGY.md, ERRATA.md, MASTER_CLAIM_LEDGER.csv, FINDINGS.csv
```

## Reproduction

Requirements: Python 3.10+ with `numpy scipy matplotlib pandas pypdf pytest`, plus `torch` for the Stage 1 RNN/GRU baselines. A TeX distribution is needed only to rebuild the PDFs.

```bash
# Stage 1 experiments (writes to snapshot/results/)
bash snapshot/scripts/run_all_experiments.sh      # Windows: snapshot\scripts\run_all_experiments.bat

# Phase P1-B fixture tests (12 tests)
python -m pytest p1_experiments/p1b/test_p1b_g1_fixtures.py

# Manuscript layout and wording checks (each must run from the directory shown)
(cd snapshot && python paper/scripts/qa_pdf.py)
(cd snapshot/paper2 && python scripts/qa_paper2.py)
```

Several files are checked against recorded SHA-256 hashes. `.gitattributes` fixes their line endings so that these checks pass on every platform. On a Windows clone made before 2026-09-27, the P1-B test G1.0 fails for this reason only.

## Licence

MIT; see `LICENSE`.
