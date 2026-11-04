# V9 native-spike cross-cell recurrence controls

This independent research directory preserves the earlier studies and tests intact, retrained no-cross and inference-cut conditions on delayed XOR. It contains 20 training fits and 30 held-out evaluations across five paired blocks, with all failed fits retained.

- [Report](REPORT.md) and [all per-seed tables](TABLES.md).
- [Frozen protocol](PROTOCOL.md), [seeds](SEEDS.json), and [source freeze](FREEZE.json).
- [Resource/accounting details](RESOURCE_COST.md) and [machine-readable costs](RESOURCE_COST.json).
- `results/TRAIN_LOCK.json`: all 20 fitted models, 10 derived cuts and pre-test artifact hashes.
- `results/*_trace.npz`: every clipped gradient, pre-update count and training diagnostic.
- `results/*_checkpoints.npz`: nine parameter checkpoints per fit; matching JSON retains diagnostics.
- `results/train_s*.npz`, `test_s*.npz`: complete fixed datasets and generation seeds.
- `results/*_train_native.npz`, `*_test_native.npz`: native states, candidates, hard spikes, counts and predictions.
- [Held-out summary](results/TEST_SUMMARY.json), [native/optimizer verification](results/verification.json), [gradient audit](results/gradient_audit.json).
- [50 train/test metric rows](results/per_seed_metrics.csv), [six cell summaries](results/cell_metrics.csv), [eight paired intervals](results/paired_contrasts.csv).
- `figures/`: PNG and SVG paired-outcome, interval and retrospective-settling figures.

In a separate copy of this directory, verify the existing scientific artifacts (NumPy required). These scripts rewrite their own audit JSON; leave the archived copy unchanged:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit_gradients.py
```

Reporting only (NumPy and Matplotlib; no training/model calls):

```bash
python summarize.py --protocol-commit ad1d807b1d48be559fd67d4f6d196b8693b4da90 --checkpoint-commit d4fde77eb49e21c00950cb93050819ab72032e90
```

The scientific runner refuses an existing result directory and the evaluator refuses an existing test marker. Do not delete these guards or rerun the same test as a new observation. Reproduction that generates new artifacts belongs in a separate copy/new output location, preserving this frozen evidence.

No-cross still contains multiple cells, internal state, self-feedback and output integration. Passing this finite gate does not prove exact correctness over all inputs; failure does not prove physical necessity. All eight intervals are exploratory pointwise estimates from five training/data blocks. Activity is not energy, and retrospective settling is not online stopping. Verification is internal, not outside replication or peer review.
