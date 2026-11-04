# V7 frozen protocol: loss × learning-rate factorial in native-spike networks

## Question and scope

Start from completed V6 commit `75a595f3c0c5aa3f50d7700b8c142906aca442b0`. Preserve V1–V6 and main. V6 compared RATE/.01 with CE/.03 and therefore could not separate loss from learning rate. V7 crosses these two factors at their two already specified levels for both architectures, using 20 fresh paired seed/data blocks (160 training runs). This is a bounded exploratory factorial study, not a confirmation, architecture search, or independently tuned comparison. No subsequent experiment or publication is included.

The extra pulse-generation module never existed in the original implementation: the author closed this earlier misattribution. Nothing is deleted or added in its name. Native firing remains mandatory.

## Fixed model, task, initialization and learning

Copy V6 `model.py` and `data.py` byte-for-byte. Both arms have 176 float64 trainable parameters, 20 continuous states, six stored previous hidden spikes and two output-count accumulators. DEND has 12 graded dendritic compartments, six hidden somata and two outputs; POINT replaces the graded compartments with 12 native-spiking branch cells. Hidden/output cells and POINT branches fire strictly when candidate > .5 and reset. Every forward spike is hard binary. There is no forward label argument, label-conditioned timing, separate output pulse generator or analog decoder. Labels enter loss and evaluation only.

The surrogate slope `1/(1+5*abs(candidate-.5))^2`, including reset derivatives, is a proxy learning rule, not the exact derivative of the hard-spike objective. The original initialization distribution is unchanged: input weights SD .45, recurrent weights SD .25, output weights SD .5, and biases .15/.05/.1 for branch/hidden/output. Initial vectors are shared across all eight cells within a seed block.

The exact V6 delayed-XOR generator uses 24 steps and six input channels. Four-step first-bit cue starts at 2–4, second-bit cue at 10–12; amplitudes are uniform [.8,1.2]. Distractor channel5 is active at7–9; common readout cue channel4 at16–23; Gaussian .03 noise is clipped below zero after addition. All four bit pairs are balanced. Informative cues finish by15; output counts use16–23. A unique largest count predicts the class. All ties, including silence, abstain and count as errors. Episode state resets to zero; no batch statistics or interepisode state are used.

The literal registry `SEEDS.json` fixes s=0..19: train2070000000+10*s, test2070010000+10*s, initialization2070020000+s; analysis2070030001. These namespaces must be checked against saved prior seed registries before data generation. Each block shares identical inputs, labels and initialization across all eight cells. There are128 balanced training episodes and512 fresh balanced held-out episodes per block; different train and test RNG streams. Exact independent regeneration and split uniqueness are checked; uniqueness alone does not establish independence.

Eight fixed cells: DEND/POINT × RATE/CE × .01/.03. All160 runs receive200 full-batch Adam updates, beta1=.9, beta2=.999, epsilon1e-8, global gradient norm clip1 and parameter clipping[-2,2]. CE uses .5*counts as logits; RATE uses count/8 against targets .75/.125. Both add .0001 mean squared parameters. Use only final weights, with no tuning, ranking, checkpoint selection, early stopping, seed replacement or post-test repair. Test all final models, including failed or silent training runs.

Loss-factor estimates include the loss's fixed scaling, target/logit conventions, gradient and clipping behavior, and their interaction with Adam and the fixed regularizer. Equal nominal learning rate is not equal effective step size. The factorial separates these specified computational factors; it does not identify a biological mechanism, representational limitation, or universal architecture property. Matrix-work/data/update/parameter budgets match; scalar nonlinearities, reset derivatives, cell counts, activation patterns, wall time and energy do not.

## Outcomes and diagnostics

Save all initial/final weights, every clipped gradient, pre-update loss/raw gradient norm/update norm and output counts for all200 updates. Save weights and training activity/margins at updates0,1,5,10,20,50,100,150,200. Retain full native final-training and test spikes, counts, predictions, states and candidates. These diagnostics do not change training or select a checkpoint.

Per run report unconditional accuracy, coverage, abstention count/rate, class recalls with abstentions incorrect, 2×3 confusion (prediction0/1/abstention), signed count margin count[true]−count[other], its median/mean and histogram−8..8. Distinguish episode silence (both counts0), nonsilent ties, and both-saturated outputs (both counts8). Report hidden/output event rates, POINT branch event rates, and readout candidate-minus-.5 quantiles. DEND's branch event array is a zero placeholder for graded compartments, not evidence of no activity or low energy.

A run fails if accuracy<.90 OR coverage<.90. Fully silent means every episode silent; fully abstaining means coverage0, which can also result from saturation. Include every run and classify all four train-pass/fail to test-pass/fail transitions. Numerical interruptions retain partial evidence and make the complete factorial incomplete; never drop or reseed them.

## Prespecified factorial estimates and uncertainty

For each architecture let R1=RATE/.01, R3=RATE/.03, C1=CE/.01, C3=CE/.03. Six PRIMARY held-out unconditional accuracy contrasts (three per architecture) form one family:

1. Loss main effect: .5*((C1−R1)+(C3−R3)).
2. Learning-rate main effect: .5*((R3−R1)+(C3−C1)).
3. Loss-by-learning-rate interaction: (C3−C1)−(R3−R1).

Main effects average equally over just the two tested levels. They may hide crossover, so report the four secondary simple effects per architecture: C1−R1, C3−R3, R3−R1, C3−C1. Also report four secondary DEND−POINT cell contrasts and three secondary architecture differences in the above factorial effects. These21 contrast vectors and their order are fixed in `CONTRASTS.json`. Apply the same21 contrasts to binary run-failure outcomes as secondary risk differences. Report eight additional mean test-minus-train accuracy differences (50 intervals overall). No secondary contrast becomes primary after viewing results.

The principal statistical unit is the20 independently trained seed/data blocks, paired across all eight cells, not160 runs or10,240 held-out episodes. All contrasts use the same20,000 bootstrap draws of20 whole block indices with the fixed analysis RNG. Report paired values and percentile95% intervals. The six primary accuracy contrasts additionally receive nominal99.1666667% marginal percentile intervals (quantiles .0041666667 and .9958333333), a Bonferroni adjustment targeting approximately95% family coverage IF the bootstrap marginal coverage is adequate. This finite-sample bootstrap coverage is approximate, especially at extreme quantiles with only20 blocks. It is not a confirmatory guarantee; increasing bootstrap draws does not increase biological or training-seed sample size.

Only those six primary accuracy intervals are adjusted. Secondary simple effects, architecture comparisons, failure-risk contrasts and train/test differences remain pointwise exploratory and cannot support a protected family-wide declaration. A5-percentage-point accuracy difference is a practical planning threshold, not evidence that n=20 has sufficient power. Zero-crossing intervals are uncertainty, not equivalence. Degenerate intervals such as[0,0] do not establish equality, noninferiority or zero population failure. Marginal failed/fully-silent/fully-abstaining run proportions use two-sided95% Wilson intervals with z=1.959963984540054,n=20. Episode silence means and run-level silence risks are different quantities. No episode-bootstrap pseudo-replication.

## Commit gates, caps and verification

Before scientific training data: commit protocol, seeds, contrast registry and executable scientific sources with SHA-256 freeze. Deterministic contract and initial gradient fixtures may precede that gate. Independently written verification/reporting tools can be added later without changing frozen scientific sources. After training all160 trajectories, commit every checkpoint plus the independent verifier before any held-out dataset is generated. Generate/evaluate tests in a single guarded invocation; do not automatically retry partial or observed evaluation.

CPU only, existing NumPy implementation, one BLAS/OpenMP thread, no installation or paid compute. Caps:300s training plus saved diagnostics;300s held-out evaluation/artifacts;300s per audit. V6 times suggest this is feasible, not a guarantee. Abort and preserve partial results on cap/numerical error. Record deviations without rewriting frozen sources.

Before training: exact native parity with V6; batch permutation/splitting/reset checks; no-history fixture (erase first cue and confirm identical state before second cue, not a performance experiment); strict threshold/reset/binary checks;704 initial locally anchored proxy finite differences. After training: reconstruct every Adam update from saved clipped gradients and all1,440 saved checkpoint vectors; resample updates0/99/199 for480 loss/gradient/count checks. Check all176 proxy derivatives at160 final checkpoints (28,160 scalar derivatives). Independently regenerate inputs/initialization and replay native equations on all final train/test episodes: spikes/counts/predictions exact, states/candidates within1e-10. Independently recompute count metrics, contrast coefficients, bootstrap and Wilson statistics. All are internal workflow consistency checks, not external independent replication or peer review.

## Deliverables and stopping rule

Deliver complete protocol/code/configurations, all160 trajectories including failures, datasets/seeds, per-seed/cell/contrast tables, diagnostics, figures, resource and compute accounting, verifier outputs, commit provenance, and manifest covering every deliverable except itself. Preserve all older manifest entries and main. Keep work on local branch `research/dendritic-population-v7`.

Assess from actual findings whether one focused independent follow-up is warranted; describe the question and a sample-size or precision approach while acknowledging mixture distributions, exploratory selection and pilot uncertainty. Do not execute a subsequent phase, expand the grid, or publish automatically. Report explicit What this does not establish and Limitations sections. Research direction and native-firing constraint: Li Zexu. Protocol, implementation, execution, numerical checking and report draft: AI-assisted.
