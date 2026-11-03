# Fixed training-composition intervention

Executed locally 2026-11-03 after freezing FROZEN_PROTOCOL.md at11:29UTC. Six linear fits only, no architecture change or new data.

| Seed | Clean-only BA | Mixed BA | Difference pp |
|---|---:|---:|---:|
|20260904|38.5541%|32.9776%|+5.5765|
|20260905|38.0594%|32.0142%|+6.0451|
|20260906|37.7462%|32.7725%|+4.9737|
|Equal-seed mean|38.1199%|32.5881%|+5.5318|

All6 optimizers converged. Regenerated test features matched previous archived source_softmax_features exactly. All models preserve original stabilized W5 source dynamics. Source has 48-byte logical neuronal payload; original analog ten-feature readout retained, not spike-only. This is not P1B or v11 noise detection.

Both arms contain exactly1500 training examples,500per class. Clean-only500clean/class; mixed250clean+250distractor/class. The750 mixed clean examples are a subset of clean-only. All selection uses one fixed permutation per seed, without consulting outputs/test values. Each seed tests the same1001 clean examples for both arms. Standardization uses only that arm's training examples; zero weight initialization,33parameters,L2=.0001,L-BFGS-B max800 ftol1e-12 unchanged. No separate validation/tuning. All classes predicted by both arms, but per-class effects are uneven; e.g.seed20260905 clean-only lo recall falls to11.51% while hi/catch improve.

Conditional paired bootstrap interval for the equal-seed mean difference: [+3.3391,+7.6080] percentage points,2000draws. Resampling unit is original code-reuse test group within each fixed seed; both arms share draws, clean subset filtered after grouping. On clean subset each group has at most one record. Interval conditions on fixed training sets/fits and does not capture training/subset/seed-selection uncertainty, nor establish a population significance claim. Historical tests are not a new blind benchmark.

Interpretation: this controlled training-composition change improves clean performance for the specified linear readout, supporting mixed-training composition as a contributor to previous weak clean scores. It does not explain all failure, eliminate established exact response collisions, prove Bayes optimality, or establish TAN architectural superiority.38.12% remains modest. Preprocessing changes are part of the prescribed train-only pipeline; effect is attributed to training composition including induced normalization, not isolated loss-gradient conflict alone.

Reproducibility: python3 local-clean-training-v1/run_control.py from any directory after extracting the archive root. The public runner uses the included upstream_probe1 source/data subset. It validates the current public dependency manifest and the unchanged frozen protocol hash before fitting. The historical runner used a sibling snapshot with a35-entry manifest. The old main pipeline is not invoked. Then python3 local-clean-training-v1/verify_saved.py recomputes confusion/recall and index intersection without fitting;22checks passed. The exact6fits were executed once, not independently reimplemented; saved-score arithmetic was separately checked. Full selected indices, coefficients, normalization, predictions, probabilities and input hashes are retained.

Source commit85213f0d417d54e4f48b301ed1c0e4200ef3c421; model blobd4e0ba873955b6e6b1f10bd36403f79c2666de51. Probe1 task blob440daff8aced676aa66e97e99196cb9bd8e185ca. Source/data derive from the earlier reviewed task archive. Numerical arrays are unchanged; publication-only edits and historical hashes are documented in README.md. Current versions: Python3.12.14, NumPy2.3.5,SciPy1.17.0 (results.json authoritative).
