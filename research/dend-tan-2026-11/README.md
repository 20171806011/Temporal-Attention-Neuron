# DEND+TAN research release · November 2026

This release records the local studies from native-state interventions to learned key--value routing, causal event replay, and the evicted-key veto. The final experiments use a dendritic spiking population with explicit FIFO memory. The original single-unit Temporal Attention Neuron and the intermediate history-attention pilots have separate model identities in the lineage.

- [Read the new manuscript](../../papers/dend-tan-event-replay-2026/manuscript.pdf) or [edit its LaTeX source](../../papers/dend-tan-event-replay-2026/manuscript.tex).
- [Follow the study lineage](LINEAGE.md), including exploratory results, negative outcomes, engineering failures, and stopped milestones.
- [Final independent confirmation](evidence/RESULTS.json), [independent audit](evidence/AUDIT.json), [training audit](evidence/TRAIN_AUDIT.json), and [phase resources](evidence/RUN_SUMMARY.json).
- [Separate earlier replay confirmation](evidence/replay_confirmation_results.json) and [its audit](evidence/replay_confirmation_audit.json).
- [Download the full study archives and checksum manifest](https://github.com/20171806011/Temporal-Attention-Neuron/releases/tag/dend-tan-2026-11-11).

## Main result

The final veto confirmation uses 12 fresh training blocks and 288 matched evaluation paths. In near-neighbor three-write histories, the veto improved evicted-target balanced accuracy by **0.781 percentage points** over old replay (97.5% paired-block interval **[0.130, 1.563]**). This restores the original circuit's 59.115%. The recent-record difference was zero in all 12 blocks and retains old replay's 96.094%. The result concerns the tested geometry, task and training recipe. The interval is nominal, and the isolated latency evidence comes from a selected CPU sample.

## Verify and reproduce

Use Python 3.12 and the NumPy version in `requirements.txt`. Run from this directory.

```text
python -m pip install -r requirements.txt
python verify.py
python replay_block.py --block 0
```

`verify.py` recalculates published scores, trigger summaries, replay work and both primary intervals from the archived counts. The independent unit is the paired training block. `replay_block.py` uses a fixed released checkpoint and regenerates its held-out input from the published seed. It compares the production and independent paths, and checks counts, events, trigger masks, replay and first-read scores against the compact archive. Select another released block with `--block 0` through `--block 11`. These commands leave the release files unchanged. The released training recipe and independent implementation are in `src/`. Training a fresh copy is unnecessary to verify the published results.

## Archive contents and scope

`evidence/confirmation_counts.npz` stores all 288 paths' native counts and episode-level events, triggers, replay steps and routing summaries. Axis order is block, geometry (`orthogonal`, `near`), history (`four`, `three0`, `three2`, `two`), rule (`original`, `replay`, `guarded`), then episode. `selected`, `y` and `schedule` use block and episode axes. `confirmation_weights.npz` stores the 12 initial and final weight vectors in block order. `evidence/raw_index.json` records original sealed file paths, sizes and hashes for provenance. The full optimizer streams and dense time-step archives are distributed as GitHub Release assets. Use the release manifest for volume membership, sanitized metadata, exclusions and download verification.

The two formal cohorts use generated synthetic episodes. No human-subject data, unpublished correspondence, external datasets or system credentials are required. Source modules and notices are covered by the repository [MIT license](../../LICENSE). Archive excerpts and redactions are documented in the [lineage](LINEAGE.md). The article is an unreviewed research manuscript.
