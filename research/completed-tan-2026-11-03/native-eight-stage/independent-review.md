# TAN independent read-only scientific review

Date: 2026-11-02. Repository: `20171806011/Temporal-Attention-Neuron`.
Pinned main: `85213f0d417d54e4f48b301ed1c0e4200ef3c421` (2026-09-28).
This AI-assisted review used the pinned public repository. No simulation reruns were performed for this review. Arithmetic below is a separate analytic calculation (piecewise polynomial integration checked with JavaScript), not an empirical experiment or independent external replication.

## Main finding

The next useful experiment is a **decoder-ceiling / causal bottleneck audit**, not immediate expansion to population coding. The current README claims P1-B establishes necessity of multi-channel circuits, whereas `p1_experiments/p1b/P1B_FINAL_SCIENTIFIC_CLOSURE_REPORT.md` §§1,5,6 explicitly disclaims that conclusion: M1 joint delivery is only 2.75–2.90%, all baseline floors fail, and spike readout errors do not isolate discretization from upstream failure.

The specified spiking output alphabets impose an additional unavoidable **population error floor above the registered 0.05 target**, even for an oracle encoder. This does not invalidate the reported observed failures; it narrows what those failures can establish.

## Precise implementation evidence

All paths below relative to repository root and pinned commit above; canonical URLs take form `https://github.com/20171806011/Temporal-Attention-Neuron/blob/85213f0d417d54e4f48b301ed1c0e4200ef3c421/<path>#L<line>`.

- `p1_experiments/task_generator.py:45`: default `val_range=(-3.0,3.0)`.
- Same file :74: `cand_values = rng.uniform(self.val_range[0], self.val_range[1], size=N)`.
- Same file :90: two distinct targets sampled from positions, independent of their values. Key rejection sampling occurs before values are drawn. Thus conditional on N, keys, and positions, target values are independent uniform[-3,3] in the stipulated generative model.
- `p1_experiments/p1b/runner_p1b.py:40–58`: default TaskGenerator, frozen N2/N3 seeds2026091920/2026091930, 2000 histories each. Delays reuse histories.
- `p1_experiments/p1b/models_p1b_v2.py:277–284`: count output map `{1:-2,2:-1,3:0,4:1,5:2}`; silence fallback0, so only5 distinct numeric reconstructions. This remains true even if some counts are physically unreachable.
- Same file :302–303: latency `bin_idx=min(19,max(0,int(t_first/0.5)))`, `v_hat_latency=-2.85+bin_idx*0.30`; silence fallback0 (:295–297). Thus at most21 distinct numeric outputs including silence.
- `runner_p1b.py:134–143`: M4 outputs taken directly from `v_hat_count` / `v_hat_latency`; no affine or other continuous correction. Passed `g_dec,b_dec` only affect analog output inside model, not these M4 values.
- Same file :152–157: `y_hat_add=v_hat_A+v_hat_B`, `y_hat_sub=v_hat_A-v_hat_B`, followed by absolute error for each and `e_comp=(err_add+err_sub)/2.0`. This is per episode, not absolute error after batch averaging. :206 onward averages these per-episode numbers. No rescaling by6 or operand magnitude.
- Same file :271–273 and :404–406: success requires bootstrap upper CI<0.05; limitation lower CI>=0.05. Protocol `P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md:219–223` matches this.

## Algebraic lower bound

Let eA=predA-targetA and eB=predB-targetB. For every real pair:

`(|eA+eB|+|eA-eB|)/2 = max(|eA|,|eB|)`.

For a fixed output alphabet C, define d_C(v)=min_{c in C}|v-c|. Every permitted decoder obeys, pointwise on every episode:

`E_comp >= max(d_C(targetA),d_C(targetB))`.

This is robust to arbitrary correlations, randomized encoding, query use, dynamics, and clever joint coding: neither role can achieve less than its nearest-alphabet distance. Granting access to target values and every alphabet symbol independently attains the lower bound in the relaxed oracle problem. The actual dynamics may be strictly worse. Silence is optimistically treated as a numeric zero despite delivery abstention, valid because unconditional composition error includes its zero fallback.

For a uniform scalar on an interval of length L with at most K reconstruction values, population absolute error is at least L/(4K), by partitioning into Voronoi cells and Jensen/Cauchy-Schwarz on sum(cell_width^2)/(4L). Thus even allowing optimally placed21 symbols yields at least6/84=1/14≈0.07143 per-role error; joint max error cannot be smaller. Count with5 symbols gives the loose bound0.3.

### Exact actual-alphabet oracle population composition errors

For independent target values U,V, let F(t)=P(d_C(U)<=t). Then `E[max(d_C(U),d_C(V))]=integral_0^infinity(1-F(t)^2)dt`.

**Count:** C={-2,-1,0,1,2}.

- 0<=t<=0.5: F(t)=5t/3.
- 0.5<=t<=1: F(t)=(2+t)/3.
- t>=1: F(t)=1.

Integral =25/54=0.462962962963. Per-role MAE=1/3.

**Latency including silence:** C={-2.85+0.30k:k=0,...,19} union{0}.

- 0<=t<=0.075: F(t)=7t.
- 0.075<=t<=0.15: F(t)=(38t+0.30)/6.
- t>=0.15: F(t)=1.

Integral=0.09809375. Per-role MAE=0.073125. Without the extra zero symbol, oracle joint max error=0.10, per-role MAE=0.075.

**Scope:** These exact constants are population expectations under independent uniform targets. They are not deterministic lower bounds on a finite sample average. A finite dataset can have smaller oracle mean. The per-episode inequality is deterministic and can directly certify a finite test dataset after extracting its targets. Pooled delays do not increase the number of independent histories.

### Existing protocol context (do not overstate novelty)

Protocol V2.6:149–151 already states ideal6-bin midpoint composition error1/3 and20-bin0.10 as reference predictions, and says these are not universal lower bounds for the neuronal implementation or finite samples. The new precise refinement is to use the **actual fixed output alphabets** (5 count values,20 latency plus zero fallback), prove the pointwise nearest-alphabet inequality, and compute the resulting exact population floors / empirical oracle certificates. It is not a universal impossibility theorem for spiking neurons or continuous-time latency codes.

## Second implementation observation

`models_p1b_v2.py:94–95`: gate enters membrane equation only through `I_syn=g_A*A*I_content`. Cue loop :252 always passes `I_content=0`, novelty0, `I_query=query_q`. Thus A cannot causally influence cue-time output if u,r are held fixed. During encoding :150–151, content=key+value and novelty=max(0,abs(key)-1.4). A depends on keys but not directly on values. This structurally constrains retrieval, irrespective of whether A retains history. A finite-difference intervention A->other value at matched u,r should be exactly null at readout. This deserves a simple causal fixture, not speculation that all3-variable systems fail.

## Reproducible next experiment proposal

1. Preserve frozen P1-B; create a separate P1-C/audit artifact. Pin commit, generator, seeds, metric, alphabets and hashes. No parameter tuning on final held-out data.
2. First extract original2000 N2+2000 N3 targets using reviewed generator only; compute nearest-alphabet per-role errors and per-history max. Compare empirical oracle means and history-cluster bootstrap CIs against0.05. Include exact algebra fixtures and several fixed synthetic value pairs. This is cheaper and more diagnostic than rerunning96000 ODE episodes.
3. Report a decomposed ladder: continuous oracle retrieval (zero noise should be machine-precision0), oracle plus fixed codebook, original carrier plus original decoder. Differences are descriptive excess losses, not automatic causal attribution across unmatched stages.
4. Verify A intervention at matched u,r through query returns identical outputs; verify carrier-state collision adversaries separately. Keep internal-composition H3 deferred.
5. For a genuine channel-resolution experiment, pre-register a feasible criterion or a precision ladder; uniform midpoint K-bin oracle joint max error=2/K, so strict<0.05 needs K>40 even before dynamic errors. Respect half-open10ms,0.1ms grid, refractoriness, attainable spike timings and noise; more nominal bins do not imply feasible output symbols.
6. Only after a positive-control carrier meets delivery prerequisites compare memory ablations and spike transport. A future multi-channel circuit can be evaluated as constructive sufficiency, not claimed necessity from current failures.

## Research state / provenance

Latest two commits85213f0/8191bdc are README presentation edits; prior18ce89c (2026-09-27) corrected numeric contradictions and explicitly documented unresolved manuscript issues. ERRATA ERR-18–20 and current-state file should govern interpretation over the newer promotional README. P1-A supports1D metric sufficiency for tested static retrieval (not universal capacity or neural dynamics). P1-B is closed with negative results, all baseline comparisons floor-limited, H3 deferred. Existing byte-for-byte replay assertions are repository claims read here, not independently verified in this review.

## Confirmed existing first-event erasure and a targeted rescue

Protocol V2.6:157–165 already proves and anticipates first-event erasure: with A(0)=0 and first key<=1.4, novelty=0, hence A and gated input remain0 throughout the event. Its value never enters state. Reflection symmetry of the key generator about1.4 survives spacing rejection and random permutation, giving erased-first-event probability1/2. With N2 both events queried, independent unobserved value uniform[-3,3] has at most1/6 chance of lying within any fixed prediction's tolerance0.5. Hence population joint success<=7/12<0.8. This is an existing declared design limitation, not a new discovery or implementation bug. Finite sample accuracy is not deterministically capped7/12.

A narrowly targeted new intervention could hold state dimension and timing fixed while replacing effective synaptic gate A with A+a_min for a prespecified positive gate floor. Starting A>0 is another but nonpersistent alternative. Report added current/energy, not just unchanged state count. Existing M2b is a constant-gate arm and must remain a comparator. Rescue does not guarantee multi-item retrieval: superposition, fading and affine accessibility can still fail. Pair original/rescued carriers with (a) frozen affine readout, (b) independently trained and held-out nonlinear diagnostic readout using (u,A,r,q), and (c) oracle-state / key-value retrieval positive controls. Query-blind and value-permutation negatives identify shortcuts. Label nonlinear decoder as external diagnostic machinery, not evidence of implemented biological dynamics. Do not retry memory-equivalence inference until functional baseline floors are met.
