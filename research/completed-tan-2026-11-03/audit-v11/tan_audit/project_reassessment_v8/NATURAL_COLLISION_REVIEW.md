# TAN-I collision evidence and the remaining task-observability question

2026-11-02. Read-only review against repository commit `85213f0d417d54e4f48b301ed1c0e4200ef3c421`. No new model simulation or proposed-fixture execution was performed. Source hashes and inspected saved-array shapes are in `NATURAL_COLLISION_SOURCE_MANIFEST.json`.

## What is already established

The natural-collision programme is **TAN-I Phase 1**, not the later P1-A static key-routing experiment. The original TAN stores an ordered five-input window in addition to membrane h. The complete state is Markovian; h alone need not be. The revised paper and `CURRENT_RESEARCH_STATE.md:17,41,59` already make this distinction. Repeating an h-only collision or adding a window-reset demonstration would not answer a new question.

`snapshot/code/experiments/natural_collision.py:203–252,260–282,289–346` defines four models and searches near-collisions separately in each model, with 0.05<h9<0.5 and |Δh9|<1e-4. The saved programme uses 160,000 nine-step histories and 1,500 analysed pairs per model. The common identical probe is followed by eight identical continuation inputs. Pre-reset u and post-reset h are kept separately.

The stronger existing control is the **843-pair common-reset subset**: all four models spike on both histories at t10, giving exact h10=0, after which windowed models can diverge again. This bypasses the logical limitation of approximate initial h matching. It supports h-only insufficiency under a common future input, not recovery of any designated historical content. Source: `natural_collision.py:429–464,639–668`; `snapshot/paper/supplementary/S2_phase1_collisions.tex:34–45`.

The saved NPZ contains 1,500×2×18 h/u/current arrays for each model, Full-TAN attention arrays, 1,500×2×18 complete input sequences for the TAN-selected ensemble, and 843×9 reset-subset gaps for every model. It does not contain the complete original 160,000-history pool or raw input sequences for each other model's separately selected ensemble. Metadata inspection was performed; this review does not claim a numerical replay of the arrays.

Existing task-relevant evidence is also stronger than trajectory separation alone. Probe1 v3 trains a fixed linear readout on post-query response features and reports actual low/high/absent target classification. The audited paper reports B2 balanced accuracy about .843 in the distractor condition, B4 .389, and B3 .337, with B4 seed attenuation and B3 class-coverage caveats. These restrictions are already explicit in `snapshot/audit_v3_amended/`, `snapshot/paper/sections/results.tex:78–100` and `discussion.tex:1–72`. A new small decoder comparison would mostly revisit that evidence.

Probe2 already holds history fixed while changing the query: the argmax flip fraction is zero. The source audit identifies amplitude shortcuts, current-query self-attraction and scalar-kernel order preservation. This is explicitly a failed audit, not successful binding: `snapshot/code/experiments/probe2/identifiability/AUDIT_REPORT.md`. Later P1-A supplies a scalar distance-kernel positive control, so another proof that positive scalar multiplication cannot select interior keys would be redundant.

## What separation statistics do not establish

The reported K=|Δu10|/|Δh9| is a **projection mismatch statistic**, not a full-state perturbation gain. The numerator satisfies Δu10=λΔh9+ΔA10 while the denominator omits the differing window. Large K can result from an arbitrarily small projected denominator and a finite context difference. It is not itself a Lyapunov exponent, a full-state Lipschitz estimate, a capacity measure or a robustness certificate. Markovianity alone does not imply contraction; K=.5 is the particular LIF identity.

Cross-model K ranks use separately selected near-collision ensembles and different drawn probes (`natural_collision.py:393–428,622–636`). Therefore B4>B3>B2 is not a matched intervention isolating attention's causal effect. Many selected pairs can also share an underlying history; pairwise bootstrap/Mann–Whitney calculations do not automatically have an independent-pair population interpretation. The JSD–current correlation includes changes in both gates and weighted contexts, and Δcurrent–Δu is nearly algebraically fixed by the collision constraint. None of these points negates the exact-reset counterexample, but they limit magnitude and mechanism claims.

Phase2's effective-rank increases are already described as coordinate-dependent covariance statistics rather than intrinsic dimension or computational selectivity. The revised paper explicitly reports the negative shared-coordinate contrast B4<B1. There is no need to rediscover that distinction; preserve it when describing the broader project.

## Recommended narrow next check: target-order information can survive in the window but be absent from the frozen response

The most useful unresolved question in this branch is whether the **actual Probe1 observation contract**, not the full ordered buffer, can identify the designated oldest target. A no-training exact paired-history check would test this directly.

Authoritative implementation: `snapshot/code/experiments/temporal_distractor.py:72–89,103–146`; it imports `natural_collision.py`'s four-model simulator. Present target is fixed at t5, distractors occupy t6–8 in the distractor condition, and query is at t9. Other steps in the 14-step block are zero. Target class is fair lo U(.55,1.05) or hi U(.95,1.45); each distractor and the query independently use the same fair mixture. Labels are target-lo/target-hi/absent; query class is not part of the label. Features are exactly `[u9,u10,u11,u12,u13,y9,y10,y11,y12,y13]`, with pre-reset u.

The proposed histories are zero outside t5–9 and have:

| History | Target t5 | Distractor t6 | t7 | t8 | Query t9 | Target class |
|---|---:|---:|---:|---:|---:|---|
| H | 5/8 | 11/8 | 5/8 | 11/8 | 11/8 | lo |
| H' | 11/8 | 5/8 | 5/8 | 11/8 | 11/8 | hi |

The values .625 and 1.375 lie in the interiors of disjoint lo-only and hi-only support regions. Both histories are admissible under the original generator, share the same query and differ only by swapping the target with the first distractor. At query, the complete ordered windows differ, but their multisets agree. The following exact proof proposal establishes the necessary reset and active-gate conditions; it should be encoded and independently checked only after the study records the experiment protocol.

1. Nonnegative inputs imply nonnegative membrane states by induction. At t8, both windows are permutations of `(0,5/8,11/8,5/8,11/8)`, so S8=23/40. Let E=exp(23/32), F=exp(253/160), δ=1e-9. The frozen unshifted-softmax context is C8=(5E/4+11F/4)/(1+2E+2F+δ).
2. C8>1 because (3/4)(F−E)>(3/4)(1+23/32)(69/80)=11385/10240>1+δ. Also exp(23/20)>1+23/20+(23/20)^2/2+(23/20)^3/6=147107/48000>3, so tanh(23/40)>1/2. Therefore A8>1/2 and both histories reset h8 to zero, regardless of their earlier h7.
3. At t9 both windows contain two values 5/8 and three values 11/8. Hence S9=3/10>0 and the context/current is permutation invariant. With h8=0, both have the same u9. Furthermore 0<u9<(3/10)(11/8)=33/80<1/2, so the queried output is strictly subthreshold; equality does not rely on a query reset or a closed query gate.
4. The actual generator gives x10..x13=0. Their window means are nonnegative, so S=0 and the gated currents vanish. Consequently the full permitted response is `[u9,u9/2,u9/4,u9/8,u9/16,0,0,0,0,0]` on both histories. Any decoder of that frozen response, however powerful, must assign the same output to distinct target classes in the ideal model.
5. Positive controls: the raw ordered target-tap oracle distinguishes the two classes. B2's fixed recency ramp also distinguishes them through its existing pre-reset response: h8 resets on both, but u9 is 47/40 versus 9/8, a difference of 1/20. LIF is an appropriate negative control because preceding pulses reset its scalar membrane and its query response depends only on the common query. B3 should be reported without assuming equality or separation in advance.

This is a specific application of standard permutation invariance and threshold reset, not a new general theorem about attention. The searched existing collision, Probe1, Probe2 and revised-paper notes do not state this exact opposite-label/full-response collision. Their known fixed-position bias and no-routing findings are related but do not supply this proof. A successful check would localize a particular loss of **target position in the frozen query/observation**, while the complete ordered buffer still carries the answer. It would not establish absence of all history memory, impossibility for arbitrary future probes, or a biological state-count lower bound.

A constructed pair has no empirical prevalence interpretation. Equal-prior classification risk on this artificial pair is at least one half; a lower bound for the original continuous task would need a separately declared positive-measure family and the actual generator density. Numerical tiny residuals must be distinguished from the ideal permutation identity. Do not infer that an arbitrarily precise floating decoder cannot discriminate if its implemented outputs are only approximately equal.

## Secondary candidate, not recommended as the next run

A matched common-history attention-on/off intervention could re-evaluate the cross-model K ordering while avoiding separately selected ensembles and recording full-state distances. It would clarify causal attribution of numerical separation, but would still not measure task-content recovery and would mostly formalize a generic projection issue. The opposite-label response collision above is more decisive and better aligned with the project's outstanding sufficiency question. No candidate experiment was executed in this review.
