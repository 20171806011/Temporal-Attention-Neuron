# DEND-POP-1: fixed prospective protocol

Frozen before generating datasets or running the models. This is a small computational mechanism experiment, not a neuroscience replication or a demonstration that TAN is generally superior.

## Research question

Does local nonlinear integration in separate dendritic compartments preserve a spatial conjunction that is lost when inputs are pooled before the nonlinearity? Does a multicompartment implementation offer any advantage over an ordinary, state-matched spiking circuit implementing the same conjunction?

The user clarified that native neuronal firing must remain; only an additional pulse-generation module should be removed. The audited original implementations have native threshold/reset firing and no identified independent output pulse generator. Therefore no historical module is deleted. The new prototype has no scheduled/Poisson/output pulse generator: all spikes arise from its own membrane thresholds. Experimental input waveforms are external stimuli, not generated neuronal outputs.

## Architecture and fair comparisons

Four input channels. Three hidden neurons correspond to pair partitions ((0,1),(2,3)), ((0,2),(1,3)), ((0,3),(1,2)). Each hidden neuron has two dendritic state variables d[j,b] and one somatic state h[j]. Three output neurons have one somatic state o[j] each. Total: six cells, twelve binary64 dynamic scalars (96 logical payload bytes), plus three previous hidden spike bits held for a one-step synaptic delay. No raw history buffer, label input, target selector, pretrained model or trainable weights.

Dendritic update: d_new = .8*d_old + .2*sum(two assigned sensory channels).
DEND current: sum_b max(d_new[j,b] - 1.1, 0).
Hidden candidate: u = .5*h_old + current. Native firing y=(u>.5), h_new=0 if y else u.
Output candidate: v = .5*o_old + .6*previous_hidden_spike; native firing and reset at strict >.5. Current hidden spikes are held for delivery at the next step. All state starts at zero, and nothing is reset between stimulus onset and the end of that episode except native threshold resets. There are no lateral/recurrent links in this first prototype; intercellular communication is feedforward.

POOLED ablation: same twelve states and same input/output wiring, but hidden current=max(sum_b d_new[j,b]-1.1,0). This isolates spatial segregation of the nonlinearity. It is deliberately limited by permutation invariance on the selected task, not a universal point-neuron baseline.

POINT_NETWORK strong baseline: six ordinary dendritic-site LIF cells replace the six graded compartments, with the same .8/.2 recurrence and strict threshold 1.1, resetting their own state to zero on crossing. Their native spikes, summed within each pair, drive the same three hidden and three output somata. It uses twelve ordinary cells, twelve continuous state scalars and the same three delayed hidden spike bits. All arms have the same input information, connectivity pattern, duration, precision, output observation and fixed parameter-selection budget (zero trials). The baseline differs in cell count, operations, thresholds and internal spike traffic; matching state is not matching physical area, ATP, joules, or total computation.

State is updated synchronously at the specified stages. New dendritic state drives hidden soma on the same step; output receives only previous hidden spikes. No output spikes are manufactured from ground truth, class labels or a posthoc waveform. Diagnostic internal voltages are saved but are not decoder features.

## Task, splits and noise

Class is the partition containing the active pair. The six possible unordered pairs are used equally often, and each partition has two pairs. Each episode presents two independently sampled positive amplitudes U[.8,1.2] for ten updates on the selected channels. The first begins at index4. The second begins at 4 + floor(jitter_uniform*(J+1)); J is 0,4 or8. Episode length32. Other channels are zero before noise. No time unit in milliseconds is claimed.

Three predeclared regimes: synchronous clean (J=0, sigma=0); primary jitter/noise (J=4, sigma=.03); stress (J=8, sigma=.10). Add independent standard-normal waveform noise scaled by sigma, then clip inputs below zero. Same pair, amplitudes, jitter uniform and noise innovations are paired across the three regimes and all models. This is noisy input, not process or measurement noise. No exclusions.

Independent RNG seeds: development2026110311, validation2026110312, test2026110313. Sizes:120,120,600 (each multiple of6). Each has independent amplitudes/noise and balanced pair counts; no sample-level reuse across splits. No architecture or parameters will be selected from any split. Development and validation are separately reported diagnostics; the primary decision uses test only. No learning is performed in this phase. The wiring encodes the benchmark's pair-partition structure; success is not learned or out-of-domain generalization.

Decoder: counts of the three output neurons over all32 updates; predict the unique largest count only if positive; otherwise abstain(-1). Abstention counts as incorrect in unconditional accuracy. Report coverage, conditional accuracy (undefined if no coverage), per-class confusion including abstention, mean spike counts, first correct-channel output time (conditional on that channel firing), and capped latency (32 for no correct-channel output). No convenient tie breaking or exclusion of silent cases.

## Hypotheses and fixed analysis

H1 primary: in the test jitter/noise regime, DEND minus POOLED unconditional accuracy exceeds .10, and the lower endpoint of a paired stratified 95% percentile bootstrap is positive. This is a controlled demonstration against a known pooling obstruction, not an architecture-wide advantage claim.

H2 secondary descriptive: compare DEND minus POINT_NETWORK accuracy in each regime. Report all signed differences and paired intervals. No superiority hypothesis is assumed for DEND. A negative or null comparison must be retained and limits any distinctive dendritic claim.

Bootstrap:2000 draws, seed2026110314; resample episodes within each of the six fixed pair strata, preserving model/regime pairing. Test episodes are the sampling unit. No multiple-comparison confirmatory claim for the secondary comparisons. These intervals concern the specified synthetic generator with frozen models, not biological or population-wide performance.

Save all inputs, labels, innovations, trajectories, native spikes and output predictions. Record current absolute/squared sums, state peaks, and branch/somatic/output spike traffic as dimensionless proxies. Count 12 state updates per global step and 12 sensory synaptic contributions per step for all arms. Compute comparison is disclosed rather than declared identical: DEND uses six rectifiers, POOLED three; POINT_NETWORK uses six additional threshold/reset events. All use six somatic threshold/reset decisions and three delayed spike transmissions.

## Verification and stop policy

Before the main run: boundary fixtures for strict thresholds, one-step propagation, zero-input silence, no forced spikes, reset state, and state shapes. Freeze protocol/specification/source hashes. One main run only. If verification exposes an implementation error, retain failed output and describe the correction; no silent trial replacement or tuning. If the prototype fails, report the failure and do not search parameters in this round.

Afterward: independently written scalar reconstruction, without importing the primary simulator, verifies every saved trajectory and spike decision, predictions, resource counters, dataset pairing, and bootstrap arithmetic. Repeatability is not third-party replication. A second full experimental campaign, learned synapses, recurrent dynamics, unknown keys, continuous-value memory, physical biophysics and applications are out of scope.

Budget: CPU only, no external data or packages downloaded, no paid services, at most120 seconds per runner/checker and under100 MB of saved experimental artifacts. This is a new local research branch; no public release or main-branch change is required for this phase.
