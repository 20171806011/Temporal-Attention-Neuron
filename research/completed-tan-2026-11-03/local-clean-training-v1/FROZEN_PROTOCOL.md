# Fixed clean-only versus mixed training intervention
Frozen 2026-11-03 before fitting, following PRIOR_PROTOCOL.md.
Use only archived Probe1-v3 input arrays for seeds20260904/05/06, original stable-softmax source dynamics and original33-parameter linear readout. No new architecture, data generation, tuning, or installs. Exactly6 fits.

For each seed use RandomState(2026110351+seed).permutation(4000) once. Filter that common ordered list by label and condition. Clean-only selects first500clean per class; mixed selects same first250clean plus first250distractor per class. Sort selected indices into original array order before fitting. Stop if insufficient. Both1500examples and500per class. Original train-only normalization, zero initialization, L2=.0001, L-BFGS-B maxiter800 ftol1e-12. Preserve nonconvergence without retuning.

Evaluate on identical archived clean test subset. Primary is equal-seed mean balanced3(clean-only minus mixed). Retain each seed, confusion, per-class recall, optimizer diagnostics, weights, selected indices and predictions. All inputs and source hashes recorded; train/test data unchanged.

Supplementary conditional95% paired percentile interval:2000bootstrap draws, default_rng(2026110409), complete original test group IDs resampled within each seed before clean filtering, same draws for both arms. Average three seed balanced3 equally. This reuses previous task's bootstrap convention, not a new hypothesis test. No refitting or seed-population uncertainty; historical test set, no claim of significance/general superiority.

Check regenerated test features against saved source_softmax_features, with1e-12 tolerance, BEFORE fitting. Stop on mismatch/nonfinite/hash failure. Main dynamics imported from reviewed run_probe1.py; readout from reviewed task_excerpt.py. No scientific script main invoked. No extra fits, shuffled controls or other readouts. Independent arithmetic checks recompute confusion/score from saved predictions, and inspect standardized linear logits. This is a new local execution, not independent implementation of all upstream source.
