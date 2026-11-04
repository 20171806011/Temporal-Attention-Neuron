# V9 frozen protocol: L3 cross-cell recurrence controls

## Question and bounded interpretation

Resume research under the user's new architecture-focused blueprint after completed V8 commit `9cddbbad7f1722d8a116c0a8245fc40d3a5e97ca`. Historical V8 pause metadata remains unchanged as a record of its time. Separate archival synthesis of line A and V6–V8 does not alter their original sources. Stop further loss×learning-rate searches; V9 tests one structural intervention on L3 delayed XOR.

Question: under the same native model, task, fixed training recipe and finite budget, does removing cross-cell recurrence before training prevent successful learned solutions, and do intact learned solutions depend on those connections at inference? This is not a proof of physical necessity. Five failed fits cannot establish representational impossibility. A no-cross network still has six hidden cells, branch states, self-feedback, two output cells and feedforward population integration; it is not a single neuron.

Both DEND and POINT receive five fresh paired seed/data blocks. Train intact and retrained-no-cross conditions from scratch (20 fits total). Derive ten inference-cut checkpoints from the ten intact fits without further training. All30 models are evaluated once on the same held-out inputs within each seed block. No data or initialization is replaced after outcomes, no configuration search and no selection among checkpoints.

## Fixed task, native dynamics and learning

`model.py` and `data.py` are byte-identical to V8 and retain the V2 equations. Each24-step delayed-XOR example has six channels. First four-step bit cue begins at2–4, second at10–12; amplitudes uniform[.8,1.2]. Distractor channel5 occupies7–9; common readout cue channel4 occupies16–23. Gaussian.03 noise is added then clipped below zero. All four bit pairs are equally frequent. Informative cues end by15; count output spikes at16–23. Unique largest count predicts class; all ties, including silence, abstain and count wrong. States reset independently per episode and labels never enter forward dynamics or timing.

DEND has12 graded branch states, six spiking somata and two spiking outputs. POINT has12 spiking branch cells in addition to six somata and two outputs. Hidden/output and POINT-branch firing uses strict candidate>.5 and reset. No pulse generator or analog decoder is added. Each intact model has176 parameters,20 continuous states, six previous hidden-spike values and two output counters.

Use RATE loss, learning rate.01,200 full-batch Adam updates on128 training histories, beta1=.9,beta2=.999,epsilon1e-8, global gradient-norm clip1 and parameter bounds[−2,2]. RATE compares count/8 to.75 on target and.125 on other output; regularizer remains.0001*mean(theta²) over176 slots, unchanged. The surrogate slope1/(1+5*abs(candidate−.5))² with reset derivatives is a proxy, not the exact hard-spike derivative. This fixed configuration follows V4's recurrence control and is not newly selected on V9 outcomes.

Seed registry fixes s=0..4: train2090000000+10*s,test2090010000+10*s,initial2090020000+s,analysis2090030001. Each set has128 training and512 fresh held-out histories. Both arms and all conditions share inputs within each seed; intact and retrained start with the same sampled values on all retained coordinates. No validation data or best-epoch selection is used.

## Exact intervention and resource limits

Recurrent weights occupy theta[72:144], reshaped6×12. Weight[source,branch] survives iff source==branch//2. Thus12 within-cell feedback weights remain and60 cross-cell weights are removed. Input weights, biases, output weights, state variables, time constants and native rules remain intact.

For retraining, initialize those60 weights to zero; project raw gradients through the mask before global clipping; keep their Adam moments and parameters exactly zero after every update. This leaves116 effective trainable parameters versus176 intact, although storage and the unmodified regularizer denominator remain176. This capacity change is part of the intervention and prevents interpreting finite-budget failure as a pure proof of pathway necessity. Dense matrix implementation still performs the same156 forward/240 backward MAC terms per episode-step; zero weights are not automatically sparse hardware savings.

For inference cut, save theta_cut=mask*theta_intact, derived only from the completed intact checkpoint. Save and commit these vectors before generating tests. No retraining, threshold adjustment or decoder change follows a cut. Its effect describes dependence of that trained solution on the removed pathway, not nonexistence of alternative solutions.

## Predeclared outcomes and interpretation

A model passes the finite held-out gate when accuracy>=.90 AND coverage>=.90. A condition passes the observed all-seed gate when all five models pass. Report every seed's unconditional accuracy, balanced accuracy, coverage, abstentions, both class recalls,2×3 confusion, silence, nonsilent ties, both-saturated outputs, count margins and activity. All failures remain in all summaries.

- Any retrained-no-cross fit passing the gate supplies an observed task-solving fit without cross-cell recurrence; all-five success is stronger observed consistency. Neither proves exact correctness on all possible inputs, a single-cell solution, nor a particular internal memory mechanism.
- Intact all-five pass with retrained failures supports a conditional learning/reliability difference under this recipe and budget. It does not prove physical necessity or universal impossibility of the masked architecture.
- Mixed results are reported as counts, not forced into a binary breakthrough/failure narrative. If intact also fails, the experiment cannot claim stable intact success.
- Inference cuts can establish a change in a fixed model's behavior, but are not a substitute for retraining.
- On balanced labels, label-independent forced binary guessing has expected accuracy50%; with abstention, expected unconditional accuracy is coverage/2. Zero or50% observed accuracy is not itself an impossibility proof.

Two prespecified primary descriptive contrasts: intact-minus-retrained unconditional accuracy, one per architecture. Secondary contrasts: intact-minus-cut accuracy and the analogous four failed-run risk differences, eight interval records total. Use20,000 percentile resamples of the same five complete paired seed indices, fixed analysis RNG. All intervals are pointwise exploratory with no family-wide multiplicity protection. Five training units remain five regardless of512 episodes or20,000 bootstrap draws. Zero-crossing or degenerate intervals do not establish equivalence. Marginal failure risk gets95% Wilson intervals;0/5 failures still has an upper endpoint about43.45%.

## Computational activity and decision settling

Record native event counts separately for branch/hidden/output and total over24steps. DEND's graded branches contribute no discrete branch events, but their state updates are not free. Record sum of absolute state changes including zero initialization. These are computational activity proxies; neither quantity measures joules, ATP, biological metabolism or hardware efficiency.

Preserve the final8-step count decoder. For descriptive timing, form cumulative output counts after each readout step1..8. A prefix is retrospectively settled if its unique winner equals the final unique winner at every later prefix. Report all-episode correct-and-settled fractions by each deadline and unique-settled fractions. Wrong and abstaining episodes remain in the denominator. This uses future prefixes to describe settling; it is not an online stop rule, a label-triggered spike, or a time in milliseconds. Do not report only successful-episode latency as if failures had no cost.

## Gates, checks and records

Before any scientific data generation, commit protocol, seeds, scientific sources and independent verifier; lock hashes in FREEZE.json. Deterministic fixtures may precede this gate. Before every scientific stage check source hashes. Commit all20 trained weights,10 derived cut weights, data and traces in TRAIN_LOCK before generating test inputs. Evaluate all30 models in one guarded invocation. On numerical failure or a300-second stage cap, preserve partial evidence and report incomplete; no automatic retry or seed substitution. CPU single BLAS/OpenMP thread, existing NumPy, no costly compute or installation.

Pre-data fixtures check intact and masked forward identity against preserved V2, mask shape/60zero/12self, binary spikes, strict thresholds/reset, batch independence and704 initial proxy derivatives. Additional masked finite differences check116 active coordinates for each arm; inactive coordinates are constrained, not free derivatives.

Save every initial/final weight, all200 clipped gradients and pre-update native counts/loss/norms, nine checkpoint weights/metrics (0,1,5,10,20,50,100,150,200), and complete final-training/test native traces. Independently regenerate data and initialization, reconstruct4,000 Adam updates and180 checkpoints, verify inactive weights/gradients/moments remain zero, and verify cuts exactly match masked intact checkpoints. Replay all15,360 test model-episodes and2,560 final-training model-episodes: spikes/counts/predictions exact, state/candidate tolerance1e-10. Recompute metrics/settling curves and all paired statistics independently. Resample60 full-batch training gradients at updates0/99/199; final finite differences on first two saved training episodes cover all176 intact or116 active retrained coordinates,2,920 derivatives total. These internal checks are not external independent replication or peer review.

Deliver full protocol/code/data/weights/gradients, per-seed results, computational resource/activity tables, plots, numerical checks, bounded report, commit provenance and file-hash manifest. Preserve all earlier archives. No journal submission or automatic publication is implied by local archival preparation. V10 has its own independently frozen task/protocol and does not retrofit V9. Research direction/native-firing requirements: Li Zexu; protocol, implementation, execution, verification and reporting: AI-assisted.
