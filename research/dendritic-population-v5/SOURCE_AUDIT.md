# V5 pre-data source audit of V4

This audit reads the preserved V4 sources and existing datasets/results only. It generates no V5 training, validation or test data and runs no new research experiment.

## Actual V4 findings and selection

- The nine scientific files listed in V4 `FREEZE.json` match their recorded SHA-256 values.
- V4 L3 fresh-test DEND accuracies were `[1, 1, 1, .994140625, .96484375]`; POINT accuracies were `[.28125, .462890625, .958984375, .642578125, .7265625]`. Means are .991796875 and .614453125, paired difference .37734375. The recorded hierarchical interval is [.1749609375, .57736328125].
- The shared A200 selection ranked the worst validation accuracy across BOTH arms and five seeds first. RATE/.01 won with minimum .5625 and mean .846875. CE/.01 and CE/.03 had higher overall means (.88203125 and .87421875) but lower worst-case values (.0859375 and 0). Thus the V4 outcome is conditional on a joint robustness-oriented selection rule, not separately optimized training recipes.
- The secondary RATE-error tie-break explicitly favors adherence to RATE targets; it is a declared selection preference, not an independent measure proving a better classifier.
- V4 B used the same selected configuration at every level, without per-level search. L3 replication used new initialization/training seeds and new test histories. Five training seed pairs were the independent replication units; the earlier hierarchical episode resampling does not create additional independent training runs.
- V4 cost records deduplicate nested A100/A200 checkpoints by their shared trajectory. Full V4 cost was 140 training trajectories and 26,000 updates, including 24,000 Adam and 2,000 SPSA updates. A small V5 search with five development seed pairs and six configurations per arm is therefore within the previous order of compute, without paid compute.

No discrepancy between the inspected V4 source, selection rule and reported L3 outcomes was found. This is a source/result audit, not a claim that finite outcomes establish an intrinsic neuron-family property.

## Label and target visibility

The model forward signature is `forward(x, t, arm, anchor=None)`; it receives the input, parameters and architecture name, not labels, pair IDs, data seed, or validation outcomes. Labels enter `loss_counts`, the training objective and evaluation only. The anchored graph is an auxiliary derivative-check path; real training/inference calls use `anchor=None` and hard binary spikes.

The unchanged XOR generator samples a balanced list of four ordered pairs, sets first bit `a=pair//2`, second bit `b=pair%2`, and target `y=a^b`. The label is not a separate input channel. Choosing class-dependent external sensory content is the task definition, not a label-driven output schedule.

First-bit input is on channel 0/1, with onset 2, 3 or 4 and duration four updates. It therefore ends at update 5, 6 or 7. Second-bit input is on channel 2/3, with onset 10, 11 or 12 and duration four; it ends at update 13, 14 or 15. Output counting starts at update 16 and runs through 23. Consequently, no informative sensory pulse remains during the readout window. The common channel-4 readout cue is identical across classes before class-independent additive noise. Amplitude, jitter, distractor and noise are sampled without label-dependent distributions.

The first-to-second gap is at least two updates. At the start of readout, neither raw cue is still supplied. A successful network must retain task-relevant information in its persistent state or previous-spike pathways. This does not identify which particular state or recurrence pathway carries that information, nor prove that long-term or biologically realistic memory is involved.

## Episode reset, batching and splitting

Every call to `forward` initializes all 12 branch states, 6 hidden soma states, 2 output soma states, 6 previous-hidden-spike values and 2 readout counters to zero. Every batch row has its own state. Matrix multiplications mix channels/cells within a row, not episodes; there is no batch-normalization statistic or cross-row forward reduction. Training-gradient sums across rows update shared parameters as intended but do not pass episode labels into inference dynamics.

Training, validation and test generators use distinct recorded RNG seeds. Architectures and candidate recipes intentionally share a dataset/initialization within each seed for paired comparisons. C reuses B data for an intervention; those observations are not independent repetitions. L1/L2 deliberately share nuisance draws and are not independent task datasets.

The audit inspected 30 existing L3 dataset blocks (A train/validation, B L3 train/validation, fresh-replication train and final tests). All have exactly balanced four-pair strata and labels equal to the XOR of the stored pair. There were 5,760 unique complete episode byte hashes and no exact episode duplication between these blocks. This is an exact-overlap check, not proof of arbitrary probabilistic independence or protection against every possible implementation error.

## Proposed no-history diagnostic

For each fresh V5 test input, copy the input and set **all channels at updates 0–7 inclusive to fixed zeros**, leaving updates 8–23 unchanged. Run the same frozen model from its ordinary zero initial state, with the same hard firing/reset, weights and readout. Do not retrain on this condition and do not use it for candidate selection.

This removes every first-bit sensory pulse. The branch/soma/previous-spike state at the start of update 8 is then identical across all episodes for a given model: biases may cause deterministic activity, but that activity contains no first-bit information. The preserved suffix contains the second bit, common cue and class-independent nuisances; the last possible first-bit pulse ended at update 7.

Under the intended balanced XOR distribution, `P(a=0 | suffix)=P(a=1 | suffix)=1/2`, so `P(y=0 | suffix)=P(y=1 | suffix)=1/2`. For any label-blind deterministic predictor of this suffix (including one that abstains), expected **unconditional accuracy equals one half of coverage**, and cannot exceed 50%. A finite ordinary test sample can exceed that expectation by chance; do not force an observed exactly-50% result or interpret a small excursion as proof of leakage. Seed-level uncertainty and per-class recall should accompany the observation.

The ablation removes prefix noise and the update-7 part of the distractor as well as the first cue. It is therefore a transparent removal of *all early observed history*, not an isolated biological memory-mechanism intervention. An accuracy drop diagnoses reliance on information available before update 8; it does not by itself prove which cell type stores it or that recurrence is necessary. If unusually high no-history performance persists across fresh seeds, investigate data/label pathways before any new architectural conclusion.

An optional deterministic audit fixture may present identical erased-prefix inputs paired with opposite labels solely to verify label-blind predictions; identical predictions must give unconditional accuracy exactly coverage/2 over each such twin pair. This is an algebraic code fixture, not a substitute for held-out task performance and not new training data.

## Interpretation of V5 stages

Stage A and stage B use disjoint final initialization/data seed blocks, so their raw gap difference remains descriptive. To isolate the training-recipe contrast, V5 adopts one additional **prespecified paired-configuration control** on the same ten stage-B final seed pairs: train fixed RATE/.01 for 200 updates alongside each arm-selected recipe, using identical data and initialization. Compare the fixed-versus-selected change in the paired DEND-minus-POINT gap within each seed. If an arm selects RATE/.01, reuse its identical trajectory and record the deduplication rather than manufacturing an independent run. This adds at most twenty trajectories; the bounded total is at most 120 trajectories/24,000 updates. The control identifies a recipe-selection effect within this grid and budget, not a universal causal decomposition of architecture and optimization. No new architecture ablation is executed in this turn; any surviving difference motivates a prospective, separately frozen L3 recurrence study.

Independent per-arm selection with equal candidate counts, training examples, updates and validation rule addresses the shared-recipe limitation within the chosen grid. It does not guarantee either architecture received its globally optimal training setup. A seed-level paired interval crossing zero establishes neither equivalence nor a non-inferiority bound. Finite optimization success/failure must stay separate from representational ability.
