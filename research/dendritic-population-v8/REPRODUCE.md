# Reproducing without overwriting the delivered evidence

Use a separate checkout or complete copy. Verification scripts write their timing/results files; running them in the delivered directory would change hashes even when the numerical checks pass. Never remove a guard from the delivered output tree to retry an observed evaluation.

The model uses NumPy; the independent Student-t quantile check and pre-data planning also use SciPy. Exact installed versions are in ENVIRONMENT.json and requirements.txt. Numerical libraries or platforms can change floating-point threshold behavior; bitwise reproducibility on another stack is not assumed.

To reproduce the study, create an isolated checkout at the pre-data commit `cd5cd4ef1cc65fd098b3a73d36d68e7a49307d17`. That tree contains all required V1–V7 dependencies and V8 scientific sources, but no V8 results. From its repository root, use the existing installed environment and run:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python research/dendritic-population-v8/runner.py
```

The runner has a900-second internal cap, saves all416 trajectories and creates TRAIN_LOCK. If it aborts, retain partial evidence and report incomplete; do not silently replace seeds or resume toward a selected outcome. Confirm successful completion and commit the generated training artifacts before invoking the held-out stage:

```bash
git add research/dendritic-population-v8/results
git commit -m 'Lock reproduced V8 weights before held-out generation'
python research/dendritic-population-v8/evaluate.py
```

The evaluator has a600-second internal cap and a once-only marker. It generates104 fresh held-out sets and evaluates all416 models. The saved Student-t interval, bootstrap sensitivity and all secondary estimates are computed from that single evaluation. Do not choose another checkpoint, configuration or interval method afterward.

The two audits can then run with an external900-second limit each:

```bash
timeout 900 python research/dendritic-population-v8/verify.py
timeout 900 python research/dendritic-population-v8/audit_gradients.py
```

They check stored data and parameters, reconstruct training and compare native trajectories. They are not a fresh experiment or an external replication. The delivered summarize.py is a postprocessing-only script; it can regenerate CSV tables and figures in a separate full-result copy after both audits pass. It is added after the pre-data commit and never changes frozen scientific code.
