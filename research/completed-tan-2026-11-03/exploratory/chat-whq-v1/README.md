# TAN bounded write/hold/query local experiment (WHQ-1)

This package returns to the pinned original TAN/P1-B carrier question. It does
NOT extend generic moment encoding or model weight editing. These are AI-assisted
Chat-sandbox CPU experiments, with no large training. Same-agent checks and reruns
are not independent external verification. See PUBLIC_REPRODUCTION.md.

## Main finding
An explicitly larger (four-state) two-fixed-key content circuit with separated
write/hold/query operation has stable analog recall in this small test. It meets
the frozen local targets at0/50ms delay under nonzero bounded noise and per-step
fixed-point state storage. A same-resource simple rate-limited EMA does equally
well and slightly better in RMSE: no distinctive TAN tanh-writing advantage was
established. At200ms delay both fail the local threshold. Wrong-address queries
break the result. This is NOT general online key-value binding or spike-only
computation, and does not change any P1-B or v11 result.

## Reproduce, offline
Python3.13.5 and NumPy2.3.5 were used (NumPy is the only third-party dependency).
No network/model weights are needed:

    python run_whq.py --out rerun_results.json --timing rerun_timing.json
    python verify_whq.py --results rerun_results.json --out rerun_verification.json

run_whq.py checks the frozen spec hash. It also rewrites the three input JSON files
with the same deterministic bytes. Timing is separate and is not expected to be
bitwise reproducible. It does NOT execute fetch_sources.py. The latter was an
unsuccessful optional retrieval attempt, retained for provenance only.

Public verification runs all scientific checks without requiring prior archives.
Optional historical preservation checking requires --prior-bundles-dir DIRECTORY
containing the original archive names and bytes. --skip-prior-bundles also skips
these preservation checks. See PUBLIC_REPRODUCTION.md for historical count caveats.

## Design and exclusions
- Three nonoverlapping history splits:128 calibration (source auxiliary affine
  only),32 validation,256 final test. New WHQ and EMA decoders have no fitted
  parameters. No tuning or parameter selection used validation/test results.
- Keys1.8,2.2 (both positive novelty in source M1), uniformly drawn independent
  values[-.5,.5], randomized arrival/key-query order, fixed event slots0 and4.
- New local criterion: RMSE<=.03 and >=90% of histories have BOTH errors<.05.
  It does not replace P1-B's registered .50 delivery tolerance or .05 composition
  criterion. Zero prediction gives RMSE~.27945 and only3/256 completed histories.
- All histories are attempted, no abstention/rejection or exclusion. Report
  numerical output availability separately from correct-history completion.
- Clean and finite-bounded modes. Finite:20 fractional bits in four int32 state
  codes, process increments uniform +/-2e-5 each .1ms in normalized u/m0/m1,
  observation error uniform +/-1e-3, query jitter +/- .01. These are sampled
  bounded perturbations, not an adversarial worst-case certificate.
- Two sequential10ms queries; two samples per query at1ms and10ms. No state
  cloning or trial buffer available to the neuron. Full evaluator states saved
  for metrics are not decoder input. Waveform history is not an observation.
- A second content gate A1 is added and A0 is repurposed. Hardwired key centers,
  phase timing, slow holding, initial bias and readout operations are all explicit
  costs. EMA receives the identical resources. No learned key dictionary.

## Evidence checks
Source was read via the GitHub connector. Due unavailable raw-network retrieval,
source/reference_m1.py is a labelled scalar transcription of the read equation,
not a byte-identical production file. A separate vector implementation agrees
with that source reference on8 noise-free histories to4.86e-16. Further54 scalar
local episodes match stored two-sample outputs exactly in this environment.
No full P1-B experiment was run. v11 is cited here as a prior reported audit, not rerun by this package.

The first run took about7.92 seconds on the available CPU sandbox. A second run
produced identical results JSON. Same-agent verification, not third-party replay.

## Files
- frozen_spec.json / FREEZE_SHA256.txt: pre-output design freeze
- run_whq.py, source/reference_m1.py: experiment and scalar source recurrence
- inputs_*.json: full small evaluator datasets, with split hashes in results
- results.json: per-history outputs, metrics, diagnostics, paired bootstrap
- summary.json, failures.json: compact reports, including all negative outcomes
- verify_whq.py, verification.json: separate scalar and accounting checks
- THEORY.md: equations, analytic observer and scope
- RESOURCES.md: state/precision/control/measurement/latency accounting
- SOURCES_AND_BOUNDARIES.md: source and prior-evidence separation
- EXECUTION_NOTES.md: retrieval and implementation audit notes
