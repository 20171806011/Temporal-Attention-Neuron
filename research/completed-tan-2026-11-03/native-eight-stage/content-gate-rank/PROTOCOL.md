# Prespecified bounded content-gate rank/observability diagnostic

2026-11-02, recorded before execution. Read reviewer rank-and-content-gate.md first. No fitted parameters or broad search.

Fixed N2 keys(.8,1.8), active slots(0,4), h=.1ms,100ms encoding, no delay; basev=(0,0). Three arms: original novelty, matched-offset novelty+3η, content novelty+η(v+3), η=.1 only during active events. Same u,A,r, clipping/reset rule and original membrane recurrence. New coupling is an intervention, not original TAN.

All9 v∈{-.5,0,.5}² checked for no encoding spikes/clipping. Atbase: exact analytic tangent recurrence vs central differences h1e-4 and1e-5. Report entries, determinant, singular values, raw-unit conditioning; do not equate rank2 with retrieval. Offset/content trajectories must be bitwise identical atbase. Report meanA and integrated absolute synaptic-input proxy (not physical energy).

Original cue q=.8,10ms, fixed affine calibration: verify numerical derivative matches g*.99^100 times first row of endpoint J. At matched u vary A0,.1,1,5 to test A-blind actual readout. Separately execute explicitly added query coupling Iprobe=.1 with two analog samples u0,u1; observationmatrix[[1,0],[.99,.001]], no hidden encoder coordinate but new current/readout-time resource.

Local diagnostic only: actual perturbation δv=(.001,-.001); reconstruct with base-J Moore–Penrose inverse (no fitting). Endpoint(u,A), original scalar raw u_readout, and modified two voltage samples are separate observation models. For each add independent Gaussian coordinate noiseσ∈{0,1e-8,1e-6,1e-4},2000 draws, seed20261102. Same standard draws reused across arms within each measurement type. Report RMSE, bias, predicted linear-noise covariance; raw u/A noise and voltage-sample noise are not physically equivalent. Zero-noise residual includes local-linearization bias/nullspace loss. No clipping estimated values to target range.

Stop after this fixed grid, derivative checks, observability and noise diagnostics; no automatic training, timing tuning or architecture expansion. Save source hash, equations, branch margins, traces/results and reproduction. No installs, public writes or spending.
