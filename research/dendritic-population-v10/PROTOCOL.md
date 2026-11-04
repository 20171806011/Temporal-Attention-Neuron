# V10A: learned feature binding across four fixed network sizes

Protocol date: 2026-11-04. This is a bounded exploratory mechanism/size pilot, not an architecture search or another loss-by-learning-rate sweep. The owner commits this protocol, source, seed registry and freeze hashes before generating scientific data; commits all final weights and training artifacts before generating held-out data; then evaluates that data once. Earlier V1–V9 results remain unchanged. No subsequent experiment or public submission is automatic.

The earlier request to remove an additional pulse-generation module was a misattribution by author Li Zexu, corrected on 2026-11-03: that module does not exist in the original implementation; no code is missing and no existing module was deleted. This closed issue imposes no new requirement. Native neuronal firing remains required.

## Question and interpretation fixed before data

Can the existing DEND and strong POINT branch-cell networks learn a precisely defined paired-feature binding task, under the same fixed learning rule, as hidden-soma count decreases from 6 to 4, 2 and 1? Which tradeoffs appear in held-out balanced accuracy, coverage, state/parameter requirements and model event proxies?

This differs from V1's handwired local-binding versus pooled-input demonstration by learning dense input allocation and recurrent/output weights at four fixed sizes. There is no pooling ablation in V10A, so this experiment does not identify the causal necessity of local nonlinear partition. It does not establish a representation gap, anatomical equivalence, energy efficiency, general structural superiority or impossibility of learning under other recipes. All sizes are reported; there is no best-size selection. Because input drive ends before readout, the task includes brief retention as well as feature binding. Fixed per-weight initialization standard deviations change aggregate drive variance with h, while mean-squared regularization has a size-dependent per-coordinate denominator P. The curves describe the complete fixed model/training setup, not pure representational capacity with all optimization factors isolated.

## Task, input information and splits

Each episode lasts 24 dimensionless steps and has six input currents. The latent pattern is an integer p in 0..15, with b_j=(p>>j)&1 for j=0..3. The target is y=(b_0 AND b_1) OR (b_2 AND b_3). Each of the 16 patterns is equally frequent. There are seven positive patterns and nine negative patterns; all four features high is positive. A cross-pair-only pattern is negative. Three-high patterns are positive. This resolves the blueprint's ambiguity between continuous features and Boolean conjunction, and its contradictory blanket rejection of overlapping cross-pairs.

For each episode draw four high amplitudes independently from U[.8,1.2] and four low amplitudes independently from U[0,.1]; select each feature's amplitude by its latent bit and hold it constant on steps 8–15 inclusive (zero-based). Feature drive is zero at all other steps before corruption. Channel 4 is a common, label-independent readout cue of 1 on steps 16–23. Channel 5 contains independent U[0,.5] nuisance samples on steps 7–9. Add independent N(0,.03) noise to all channels at all 24 steps, then clip inputs below zero to zero, as in the earlier population data pipeline. Full RNG order is frozen in data.py. There is no label-dependent cue, timing, state initialization or output generator.

Each of five independent seed blocks has 128 training episodes (8 per pattern) and 512 untouched held-out episodes (32 per pattern). Training/test RNG namespaces are disjoint. All arms and sizes within a block receive exactly the same data. Their initialization seeds are size-specific and paired between DEND and POINT. No validation split or data-driven selection is needed: loss, learning rate, size set, step budget and final checkpoint are all fixed. Every completed fit is tested, including silent, tied and poor-training fits. Held-out data tests new amplitude/noise realizations of the same 16 Boolean patterns; it does not test unseen Boolean combinations or a changed distribution. Identical split seeds are not evidence of identical features across different tasks; no earlier scientific split is reused.

## Models and learning

For h in {6,4,2,1}, each model has 2h branch states, h hidden somata and two output cells. Both have dense access to all six input channels, the same recurrent/output topology and the same initial parameter vector within each block/size. There is no privileged DEND input mask or hand-assigned conjunction. The h=6 model must match frozen V8 initialization, native forward trajectory and surrogate gradient exactly on deterministic fixtures before scientific data.

The parameter shapes are wi:(6,2h), wr:(h,2h), db:(2h), vb:(h), wo:(h,2), ob:(2). Initialize weights respectively with independent normal SD .45, .25 and .5, and biases .15, .05 and .1, in that order. Parameters number P=2h²+17h+2. Every episode resets branch, hidden, output, previous-hidden-spike and output-count states to zero. Native equations retain V2 coefficients:

```
dc = .85*d + .25*(x@wi + previous_hidden_spikes@wr + db)
DEND: local = max(dc,0); d = dc
POINT: branch_spike = 1[dc>.5]; local = branch_spike; d=dc*(1-branch_spike)
vc = .65*v + .5*sum(the two local values per hidden soma) + vb
hidden_spike = 1[vc>.5]; v=vc*(1-hidden_spike)
oc = .65*o + previous_hidden_spikes@wo + ob
output_spike = 1[oc>.5]; o=oc*(1-output_spike)
previous_hidden_spikes = hidden_spike
```

All forward spikes are hard binary, with strict greater-than threshold and reset. The output observes previous-step hidden spikes. Sum only native output spikes on steps 16–23; a unique largest count wins. Ties, including all-zero counts, abstain and count as errors. There is no analog decoder or label-conditioned forward argument.

Each of the 40 fits uses 200 full-batch updates, RATE loss, Adam learning rate .01, beta1=.9, beta2=.999, epsilon=1e-8, global gradient norm cap1, and parameter clipping to [-2,2]. RATE targets are .75 for the correct output and .125 for the other output, applied to count/8, with mean squared error over both outputs and batch. Add .0001*mean(theta²). The surrogate slope is 1/(1+5*abs(candidate-.5))². It is a proxy learning rule, not the exact derivative of hard spikes. Resets retain the inherited surrogate chain rule. No hyperparameter search, early stopping, restart or outcome-dependent extension is permitted.

## Endpoints and statistics

Primary outcomes are four DEND-minus-POINT held-out balanced-accuracy contrasts, one per h. Balanced accuracy is the mean of both class recalls, each counting abstentions as wrong. Pattern balance implies positive prevalence 7/16, so majority-class ordinary accuracy is 9/16=56.25% and majority-class balanced accuracy is50%. Report ordinary accuracy, both recalls, all16 pattern recalls, confusion matrix including abstention, coverage, silence, non-silent ties, count margins, and every individual seed. A fit passes only if balanced accuracy>=.90 and coverage>=.90. This V10 gate intentionally differs from V9's ordinary-accuracy gate.

Compute seed-paired differences and 20,000 percentile-bootstrap resamples of five independent training/data blocks, with the frozen analysis RNG. Report the four pointwise 95% intervals as exploratory and unadjusted for multiple comparisons. Five blocks do not provide reliable population-tail precision or a power guarantee. Do not treat the 512 episodes per trained model as independent training replications. A zero-crossing interval does not show equivalence; a degenerate interval does not establish population equality or noninferiority. Report all failed/silent runs and Wilson intervals for descriptive failure proportions. Do not call a finite-budget training failure a representational impossibility.

Secondary cost endpoints retain all episodes, including errors. Report branch/hidden/output spikes separately, total spikes, and L1 changes of all continuous states starting from zero. The optional dense event-delivery proxy counts one local delivery per POINT branch spike, 2h+2 targets per hidden spike and one external event per output spike; it does not imply sparse/event-driven hardware was executed. DEND has graded local transfers every step, so its zero branch-spike count is definitional and cannot alone establish reduced total cost. These are model-unit computational proxies, not ATP, joules, physical energy or measured hardware efficiency. State variation depends on coordinates and scales.

For latency, use prefix output counts after each readout step1..8. At each prefix determine its unique leader or abstention. An episode is correctly settled by step k only if the prefix leader equals the true label at every readout prefix from k through8. This is a retrospective curve using future prefixes for scoring, not a realizable online stopping rule. Report its all-episode fraction and mean earliest correctly settled step, assigning9 to every incorrect/abstaining/unsettled episode. Labels enter only this post hoc metric. Final decoding remains the full-window count rule.

## Resource and verification contract

Report per-size parameters, continuous states3h+2, previous hidden spike bits h, two count accumulators, cells, local nonlinearities, thresholds/resets and dense matrix MAC terms. DEND represents h+2 spiking cells plus2h graded branch compartments; POINT represents3h+2 spiking cells. Matrix/state matching at the same h does not mean equal biological cells, scalar nonlinearities, reset derivatives, activation patterns, energy or wall time. Dimension changes do not preserve state/parameter cost across sizes. Do not label the h-axis total neuron count.

Forward matrix MAC terms per episode-step are 12h+2h²+2h=2h²+14h. Backward terms, excluding input gradients, are 2h+2h+12h+2h²+2h²=4h²+16h. Count training forward/backward, nine checkpoint diagnostics and held-out inference separately; scalar arithmetic, loss, Adam, I/O and verification are outside those matrix counts. Record measured CPU wall time separately.

Before training, use deterministic fixtures for exact h=6 V8 parity, batch/episode independence, binary/strict threshold/reset behavior, and central finite differences of every parameter of the locally anchored surrogate graph for both arms and four sizes. At each final model, check every parameter on the first two saved training episodes; replay frozen sampled training gradients and reconstruct all Adam updates. A separately written verifier regenerates all data, initialization, checkpoints, native training/test trajectories, predictions, metrics and seed-level statistics without importing the scientific implementation. Native spike/count/prediction arrays must agree exactly; small state roundoff is reported explicitly. These are internal checks within the same AI-assisted workflow, not external independent replication or peer review.

Save initial/final weights, every clipped gradient and preupdate count, checkpoints at updates0,1,2,5,10,25,50,100,200, checkpoint metrics, all datasets/seeds, full final native train/test trajectories, tables, figures, cost accounting, report and manifest. Freeze hashes before training. Commit gates are enforced by code. Maximum wall budget300s for training+diagnostics,300s for test+artifacts and300s per audit. A timeout/nonfinite interruption is saved and reported; no automatic retries or expansion. CPU NumPy with one BLAS/OpenMP thread; no paid compute.

## Biological motivation, mathematical limits and deferred task

See [DESIGN_REVIEW.md](DESIGN_REVIEW.md) for the limited static-threshold obstruction, ordinary two-subunit POINT construction and verified primary sources. DEND here uses graded ReLU, not NMDA conductance or literal multiplication. Both current arms retain subthreshold analog state and the same .85 branch/.65 somatic retention factors. The native-eight-stage 150ms/10ms result used a different modified model and analog inverse decoder; it cannot supply physical time constants or zero-energy memory claims for this pilot. Candidate B is therefore deferred without running it.

Research direction and native firing constraints were proposed by Li Zexu. Protocol formalization, implementation, execution, numerical verification and report drafting are AI-assisted. The final report must contain separate “What this does not establish” and “Limitations” sections.
