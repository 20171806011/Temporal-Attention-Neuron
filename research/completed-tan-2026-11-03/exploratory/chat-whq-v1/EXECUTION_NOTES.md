# Execution notes

1. The pinned source equations were inspected before finalizing the model.
   The included reference is a transcription, not a production import; see
   SOURCES_AND_BOUNDARIES.md.
2. Wrote frozen_spec.json and its SHA256 before the first simulated output.
   No parameter/grid/threshold changes were made after seeing validation or test.
3. Implemented one repair and one matched simple control, then executed run_whq.py.
   All simulations completed without runtime or assertion failure.
4. verify_whq.py does not import run_whq.py. It recomputes metrics/observer outputs,
   checks no split overlap, and independently follows scalar recurrences for54
   test episodes. It shares the separately transcribed original one-step function
   as an explicit source reference. This is same-agent verification.
5. Repeated the fixed main run and compared full numerical-result JSON bytes.
   Wall-clock records are kept separately because they are nondeterministic.
6. Failed scientific conditions are retained:200ms holding fails local success;
   wrong-address queries fail; no TAN-specific gain over matched EMA was found.
   Failure of those tests did not trigger parameter changes or dropped histories.

Implementation audit: the final decision expression now explicitly checks the
already-frozen requirement RMSE<zero predictor, rather than relying on the much
stronger .03 target in this dataset. This changes no simulation, parameter,
output or decision here; the rerun numerical JSON remains identical.
