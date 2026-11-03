# DEND-POP-2 frozen prospective protocol

This is a bounded new experiment, not a retuning of v1. Native neuronal firing must remain. Old code/results are immutable. The first phase found no dendritic advantage; the new question is whether learned, interacting populations solve a temporal conjunction and whether dendritic compartments help compared with ordinary native-spiking cells.

## Task, data and splits

Delayed binary XOR: first bit on sensory channels 0/1, second on 2/3, with a gap and unrelated distractor channel 5. Channel 4 is a common readout cue, not a label. 24 updates; two four-update presentations begin at 2+j1 and 10+j2 with independent j in {0,1,2}. Amplitudes uniform [.8,1.2], distractor uniform [0,.5] at steps 7–9; independent Gaussian input noise SD .03, clipped below zero. Output readout uses native spike counts at steps 16–23 inclusive; ties abstain and count incorrect. Each of the four ordered pairs is equally represented. This is fresh histories from a known four-binding task, NOT generalization to unseen logical bindings. Labels are used only for training/selection/evaluation.

Three paired initialization/data seeds, indexed s=0,1,2. Training 128 histories with seed 2026110400+10s, validation 128 with seed +1, and a fresh test of 512 with seed +2. The test generator will not be called until all selected checkpoints and their hashes are frozen in SELECTION.json. No v1 data is loaded. All candidate runs, including failures, must be retained. Development/validation inform selection only as specified below; no architecture adjustments after the main run starts.

## Model and fairness

DEND: 6 hidden neurons, each 2 graded dendritic states, plus 2 native output neurons. POINT: 12 ordinary branch neurons, 6 hidden neurons and 2 output neurons. Both: 20 float64 continuous states, 6 previous-hidden-spike bits (stored as 6 float64 in this implementation), and 2 common readout counters. Both use the same trainable arrays: input 6x12, recurrent 6x12, branch bias 12, soma bias 6, output 6x2, output bias 2: 176 parameters, 1,408 parameter bytes. Recurrent connections include cross-cell and self connections; output neurons receive previous hidden spikes. There is no separately scheduled output pulse generator.

Exact recurrence in model.py is frozen with this protocol. The two arms differ in branch nonlinearity/reset: graded ReLU vs strict >.5 native branch firing/reset. Both retain strict >.5 hidden and output native firing/reset. Signed weights, biases and negative states are allowed; no Dale's law, morphology, biological units or biological plasticity is claimed. Output counts alone determine predictions; training also uses these counts. No analog trajectory decoder.

Dense matrix multiply terms per sample/update are identical: 72 input +72 recurrent +12 output =156 multiply-accumulate terms, plus matching branch-to-soma sums. Both allocate identical state arrays and execute identical loop lengths and candidate evaluations. DEND uses 12 branch rectifiers; POINT uses 12 branch comparisons/reset products. Both use 8 soma comparisons/reset products. Thus state, parameter, matrix arithmetic and evaluation budgets are matched; exact wall time, scalar operation count, cell count, energy and biological resources are not equal. Save elapsed time and spike traffic rather than claiming complete compute equivalence.

## Training and search

NumPy only, CPU, one thread. Two-sided simultaneous perturbation stochastic approximation (SPSA), not biological learning or backpropagation. Initialize from identical theta per paired arm: input SD .45, recurrent SD .25, output SD .5; biases .15/.05/.1. Two learning-rate candidates .03 and .10. Each candidate starts from the same initial vector and gets 100 steps on the full 128-example training set. Perturbation c=.15/(step+1)^.101, learning rate a=LR/(step+1)^.602; Rademacher direction shared across paired arms and learning rates; gradient clipped componentwise to [-5,5]; parameters clipped to [-2,2]. Objective: cross entropy from .5 times output spike counts plus .0001 mean squared parameters. Each step uses exactly two objective evaluations. No minibatch or adaptive early stopping; final checkpoint only.

For each arm/seed select the final checkpoint with highest unconditional validation accuracy; ties choose lower validation objective, then smaller learning rate. Save initial and all final parameters, losses, validation scores, update norms, timing and selection. Total 12 candidate runs, 1,200 updates, 2,400 training objective evaluations. No additional hyperparameter search or reruns after test inspection. CPU total cap 300 seconds for training and 120 seconds for evaluation; stop and retain partial logs if exceeded. No installs, GPU or paid compute.

## Endpoints and interaction intervention

Primary: DEND minus POINT test unconditional accuracy, averaged over all three seeds. Save each seed separately. Use 2,000 paired hierarchical bootstrap draws: resample three seed pairs, then within each selected seed resample test examples within each of four input-pair strata. Seed 2026110499. A useful narrow advantage requires mean difference >.05 and bootstrap 95% lower endpoint >0. Three seeds are insufficient for strong population-of-training-runs inference; interval is exploratory.

Predeclared diagnostic: for each selected trained checkpoint, zero cross-hidden-neuron recurrent weights at inference while retaining self connections and all other weights. Compare intact minus cut accuracy, count changes and spikes for each arm. This tests dependence of a fixed trained circuit on cross-cell interactions, not superiority over a separately retrained no-interaction architecture. Also evaluate initial untrained checkpoints. No post-test adjustments, no claims of universal superiority, and no model chosen based on test performance. Report negative or inconclusive results and ties explicitly.

## Verification

Freeze code/protocol hashes before generating training data. Check threshold/reset and quiet-state fixtures, parameter/state budgets and cross-mask behavior before running. After training, lock selected parameter hashes before test generation. Save test inputs, selected/initial checkpoints, predictions, spike/state/candidate trajectories and all candidate logs. Independently implement NumPy scalar equations for selected checkpoints and verify all test trajectories, masks, readouts, data generation, hashes and reported accuracy/intervals. Same workflow implementation checks are not outside replication.
