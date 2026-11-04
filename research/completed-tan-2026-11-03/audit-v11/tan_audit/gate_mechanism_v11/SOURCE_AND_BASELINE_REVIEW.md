# Original Task B and ordinary-adaptation baseline review

2026-11-02. Prospective source/literature review only: no new neuron runs, calibration, test-set evaluation, or metric estimates. Frozen repository `85213f0d417d54e4f48b301ed1c0e4200ef3c421`; original files, labels, and P1A evidence remain unchanged. 

**Recommendation:** a fixed causal mean-subtraction/rectification plus LIF control, calibrated by one declared training-only gain, is a stronger challenge to the Task B explanation than the repository's spike-only AdaptiveLIF alone. A single held-out temporal-correlation change can compare their transfer behavior with the stimulus amplitude and marginal noise variance unchanged. It cannot by itself establish attention-specific causality or discover adaptation as a new mechanism.

## Exact AdaptiveLIF boundary

Authoritative class: [baseline_comparison.py:107](../../Temporal-Attention-Neuron/snapshot/code/experiments/baseline_comparison.py#L107); Task B overrides: [460](../../Temporal-Attention-Neuron/snapshot/code/experiments/baseline_comparison.py#L460). Constructor defaults are leak `.5`, baseline threshold `2`, spike increment `.3`, recovery `.05`. **Task B overrides only leak to `.8`**. Initial membrane is0 and threshold is2. Reset restores both values.

Let `a_t=threshold_t−threshold0` immediately before the t-th update. The exact Task B equations are:

`v_t=.8*h_t+x_t`,

`y_t=1[v_t>2+a_t]`, `h_(t+1)=(1−y_t)*v_t`,

`a_(t+1)=.95*(a_t+.3*y_t)=.95*a_t+.285*y_t`.

The comparison uses the **old threshold**. Increment happens after a spike, followed immediately by recovery. A reimplementation with `a_next=.95*a+.3*y` would not reproduce the source. The adaptation state is driven by emitted spikes, not directly by subthreshold input or surprise. Before the first spike, its trajectory and first-spike decision exactly equal LIF with the same leak/base threshold. Thus onset discrimination may barely test its adaptation mechanism if the relevant histories have no earlier spikes.

The class retains two real scalars, membrane and threshold/adaptation. It uses fixed parameters and constant work per update, with no history buffer, softmax, or stored input record. The class body needs only Python arithmetic; importing its whole enclosing script would unnecessarily trigger plotting/font setup, hard-coded `the original output directory` output creation, logging and PyTorch setup ([57](../../Temporal-Attention-Neuron/snapshot/code/experiments/baseline_comparison.py#L57), [193](../../Temporal-Attention-Neuron/snapshot/code/experiments/baseline_comparison.py#L193)). Extracting the exact class AST avoids those side effects. The convenience copy in [model/baselines.py:7](../../Temporal-Attention-Neuron/snapshot/code/model/baselines.py#L7) has the same class AST but its module imports PyTorch unnecessarily for this class.

No training/validation calibration of AdaptiveLIF was found in the inspected comparison source: all four task runners hand-set its parameters. The training routine applies to RNN/GRU imitators ([baseline_comparison.py:568](../../Temporal-Attention-Neuron/snapshot/code/experiments/baseline_comparison.py#L568)). The saved Task B summary reports AdaptiveLIF late rate `.32` versus TAN `.2645`; this is the previously audited plateau-activity metric, not a calibrated signal-preservation or mechanistic comparison. These numbers are existing-table reads, not new runs. Task A's non-firing calibration limitation must not be generalized into a claim that all adaptive models fail to detect changes.

## What is actually different in original TAN

Original Task B uses [statistical_validation.py:174](../../Temporal-Attention-Neuron/snapshot/code/experiments/statistical_validation.py#L174): W10, leak.8, threshold1, zero membrane/history, strict threshold/reset. It appends the current input **before** subtracting the window mean. With `s_t=x_t−mean(H_t)`, the attention logits use signed `2*s_t*H_t`, while the gate is `tanh(abs(s_t))`. Effective current is that gate times the weighted input-history context. It therefore reacts to input deviations even before any spike. This is different from spike-triggered threshold feedback, but is not alone evidence of a unique new computation.

TAN retains ten input samples plus membrane and computes window statistics, exponentials, normalization, weighted context and a nonlinear gate. A matched-window FIR baseline also pays for its history buffer; an EMA baseline needs only its running mean and membrane but changes the temporal kernel. Do not call either comparison equal-state or equal-energy unless those resources are actually equal and accounted for. The existing rectified ablation class is a different branch, as established in v9; it must not silently replace the original signed-surprise Task B implementation.

The manuscript calls these behaviors spontaneous adaptation without a separate adaptation module ([experiments.tex:3](../../Temporal-Attention-Neuron/snapshot/manuscript/sections/experiments.tex#L3)); its related-work contrast calls Markovian dynamics memory-free ([related_work.tex:9](../../Temporal-Attention-Neuron/snapshot/manuscript/sections/related_work.tex#L9)). Those interpretations are too broad: retained membrane/adaptation variables contain history, and TAN itself is Markovian when its full finite buffer is included. The meaningful comparison concerns the retained state and update mechanism, not the word “Markovian.”

## Primary literature: what is already known

- [Benda and Herz (2003)](https://pubmed.ncbi.nlm.nih.gov/14577853/) derive adaptation-induced high-pass behavior, with limitations to appropriate firing rates and moderate fluctuations. The repository already cites this work. Generic onset transients, slow-background attenuation, or high-pass responses are not TAN-specific discoveries.
- [Benda, Longtin and Maler (2005), original article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6726095/) demonstrate ordinary adaptation separating fast communication transients from slower background oscillations; their model parameters were determined from step responses. That is a direct precedent for the proposed fast-signal/slow-nuisance distinction, not proof that this repository's particular AdaptiveLIF must match their biological model.
- [Platkiewicz and Brette (2010), original article](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000850) derive threshold adaptation driven by membrane potential as well as spike history. The repository baseline lacks the subthreshold voltage dependence. A TAN advantage over it cannot eliminate ordinary adaptive-threshold mechanisms as a class.
- [Franzen, Ramlow and Lindner (2023), original article](https://pmc.ncbi.nlm.nih.gov/articles/PMC9840600/) show that even ordinary theta/QIF neurons need not have one universal firing-rate response to correlation time at fixed noise variance. The proposed empirical rate/detection direction must remain a model-specific prediction, not a theorem about all LIF/adaptive neurons.

Further source details are in [PRIMARY_ADAPTATION_NOTES.md](PRIMARY_ADAPTATION_NOTES.md). None of these references establishes TAN's mechanism, hardware cost, or superiority.

## Strongest bounded comparison and prediction

Prefer one predetermined change from white to positively correlated stationary Gaussian nuisance over a rise-time manipulation. Keep the clean step, onset15, amplitude1.5, duration60 and pre-clipping SD.4 unchanged. Stationary AR(1) initialization preserves the marginal Gaussian law from the first sample; initializing colored noise to zero would inadvertently reduce early variance. With the unchanged clipping, the per-time delivered-input marginal law and expected input-square sum also remain matched. Realized per-history energy will vary and should not be normalized post hoc. Altering rise time while simultaneously fixing amplitude, total duration and step energy generally needs compensation elsewhere and changes more than one feature.

There is an exact source-grounded **filter-level** prediction. For stationary *unclipped* AR(1) noise with correlation `r^|lag|`, the current-inclusive W-sample residual has

`Var(N_t−mean(N_t,...,N_(t−W+1))) = sigma²*((W−1)/W − (2/W²)*sum_(k=1)^(W−1) k*r^k)`.

This decreases with positive r and tends to0 as r approaches1. It is ordinary boxcar high-pass behavior shared by a simple FIR control, not an attention-specific prediction. By contrast, a stationary unreset linear leaky accumulator has variance `sigma²*(1+lambda*r)/((1−lambda²)*(1−lambda*r))`, which increases with positive r. Neither formula predicts the complete spiking models after clipping, reset, threshold adaptation, nonlinear gating or onset transients; no such numerical prediction is claimed here.

The strongest useful question is therefore whether TAN's frozen output tradeoff under this correlation shift is reproduced by the ordinary input-deviation baseline, after the declared training calibration. If both transfer similarly, generic input high-pass behavior is an adequate explanation for this Task B result. A reproducible difference would distinguish the tested implementations, but could reflect context weighting, rectification/saturation, kernel shape or operating point; it would not alone isolate attention as the cause. Native/catch detection and activity must remain jointly visible, so silence is not a successful filter. P1A's separate positive results are unaffected either way.

## Calibration choices and constraints

For the ordinary baseline, freeze either a causal FIR or EMA kernel before training; do not choose between them after test outcomes. Original TAN uses a magnitude gate, so a full-wave residual `abs(x−mean)` is the closest ordinary rectification control. A half-wave `max(0,x−mean)` control is legitimate only if its polarity difference is stated. Use one prespecified analytic gain, for example the ratio of summed TAN effective current to summed baseline unscaled rectified current on the training distribution. This matches **one training current moment**, not test firing rate, complete gain, current variance or physical energy. The teacher-current access used for calibration is a declared training resource, not test-time access. A zero denominator is an explicit calibration failure, not permission to change the rule.

If the study instead uses AdaptiveLIF calibration, its frozen `.3/.05` adaptation parameters can remain unchanged while one declared threshold parameter is selected using training histories only. Do not assume threshold-to-spike-count monotonicity across recurrent resets/adaptation and silently use an unjustified bisection. Any finite candidate set, objective, constraint and tie rule must precede training; no test-set extension or failure-driven extra knob. One matched statistic does not equate response strategy or physical resources. The simpler analytic-gain filter control is the recommended bounded route for the present task; this note does not request an architecture sweep.

Sources and hashes, including the exact AdaptiveLIF node and duplicate-class check, are recorded in `SOURCE_HASHES.json`. No new model/calibration/evaluation was run. 
