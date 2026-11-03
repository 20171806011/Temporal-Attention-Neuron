# Existing TAN source and module audit

Baseline: research paper commit007d0b4ebe8208aa95e8a70c1175b9bdc2c8373f; original model commit85213f0d417d54e4f48b301ed1c0e4200ef3c421. Main remained at85213f0 when checked for this phase. Current paper, clean-only certificates, training control, v11 report, native eight-stage report, and the exploratory WHQ summary were reviewed before choosing this task.

The old clean collision and training results are preserved. Static-address success assumes an available key–value array. The native eight-stage rescue changes mechanisms and observations. None establishes universal single-cell superiority; adding neurons/compartments does not inherit those old success rates.

## What generates the existing spikes?

- `snapshot/code/model/tan.py`, `TANNeuronAblation.forward`: the candidate membrane is compared to its threshold, and a native spike resets the membrane. `softmax` computes current weights; it is not a spike generator.
- `snapshot/code/model/lif.py`, `LIFNeuron.forward`: the same native integrate/threshold/reset pattern.
- `snapshot/code/experiments/statistical_validation.py`, `TANNeuron.forward`: native strict threshold and reset in the W10 study. The phrase 'spike generator' in v11 reports denotes this shared soma mechanism, not a separate Poisson or scheduled output source.
- `p1_experiments/p1b/models_p1b_v2.py`, `euler_step`: native threshold crossing, reset and refractory state. Its >= comparison differs from the discrete TAN's > and is not silently unified.
- The same file, `simulate_cue_and_readout`, and `runner_p1b.py`: M4_Count/M4_Latency consume the spikes already emitted by the carrier. They are output decoders, not modules manufacturing spikes. Deleting them would not implement the requested distinction.
- `snapshot/code/experiments/effective_dimension.py`: pulse/episode generation describes externally supplied benchmark stimuli, not an additional neuronal output generator.

A repository text/code inventory did not identify a distinct extra output pulse-generation module. Therefore no deletion is claimed, and no historical module is modified. If a separate uninspected implementation is meant, its path remains the specific unresolved target. Independent work proceeds using native threshold/reset cells and no extrinsic output generator.

## New model relationship

DEND-POP-1 is a deliberately new six-cell feedforward architecture with twelve compartment/somatic state variables and three one-step spike-delay bits. It uses the TAN-like native strict-threshold/reset convention for somata, but has neither TAN's W5 window nor a claim to be the unchanged TAN model. Local graded dendritic integration is coupled to native somatic spikes. All stimuli are provided externally; no clock forces a spike, no label enters the cell update, and no readout regenerates a spike waveform.

The ordinary point-neuron network is preserved as a new explicit comparator with the same continuous-state and output-access budget. Original TAN files and previous research results remain untouched. A future exact removal of a separately identified module should be a separate, audited intervention.
