# Frozen-history target reconstruction and oracle certificate

2026-11-02. Independent local numerical follow-up; no neuron simulation, dependency installation, remote write or publication.

## Result

Both actual-alphabet reconstructed finite-history oracle averages exceed 0.05, for N2 and N3 independently. These are deterministic lower bounds for **these reconstructed histories**. Whether the histories byte-match the original run remains unverified because no original per-history target ledger was saved.

| Dataset | Alphabet | Oracle mean E_comp | History-bootstrap 95% percentile CI |
|---|---|---:|---:|
| N2 (2,000 histories) | M4_Count | 0.4663902416 | [0.4559490333, 0.4768565740] |
| N2 (2,000 histories) | M4_Latency | 0.0995216573 | [0.0979069815, 0.1010960313] |
| N3 (2,000 histories) | M4_Count | 0.4774184421 | [0.4670687486, 0.4880495216] |
| N3 (2,000 histories) | M4_Latency | 0.0969753947 | [0.0953487754, 0.0985701798] |

Bootstrap intervals quantify resampling uncertainty about a population average, not uncertainty in the deterministic average of the reconstructed dataset. All lower endpoints exceed 0.05. The 10,000-replicate history bootstrap here is a diagnostic implementation, not a claim to reproduce the registered bootstrap's exact procedure/seeds. Delays reuse histories and are not extra independent samples.

## Reconstruction and checks

- Pinned source commit: `85213f0d417d54e4f48b301ed1c0e4200ef3c421`.
- Entire downloaded `task_generator.py` reviewed before execution: only NumPy, dataclasses and typing imports; definitions only; no I/O or neuron simulation. Script verifies its reviewed SHA256 before import.
- Source SHA256 `674cf9f49f5cd7f4975b87d29cd413e801fe40bd10fc56b966e663c9e6fc989d` matches the reviewer's frozen closure manifest. Reviewer independently verified source and aggregate JSON against pinned Git blob IDs.
- N2 uses continuous `RandomState(2026091920)` and N3 continuous `RandomState(2026091930)`, 2,000 trials each; W=5, keys uniform[0.3,2.5] with spacing rejection>=0.05, values uniform[-3,3], zero query noise. Per-trial `seed=base+trial_id` is metadata, **not** a reseed.
- RNG order preserved: rejection-key draws; permutation of sorted keys; candidate value draws; distinct position choice; ordered target-role choice. Middle-key flags consume no randomness.
- A separately written faithful implementation matches generator keys, values, positions, target indices and rejection counts for all 4,000 trials. Entire RandomState state agrees after **every** trial, verifying RNG-consumption order in the current runtime.
- Repeat run reproduces JSON and target-ledger SHA256 hashes exactly. This is local reproducibility, not original-run verification.
- Current NumPy version: 2.3.5. Historical requirement was only numpy>=1.24; exact historical NumPy/runtime is unavailable. Source fidelity alone does not prove historical RNG output identity.
- CSVs preserve ordered targets and indices; JSON records initial examples, input/ledger/target-array fingerprints and rejection totals.

## Descriptive mean excess

Subtracting the reconstructed dataset oracle mean from the saved full-precision model mean gives the following values. This arithmetic is conditional on reconstruction matching the original histories and on the reported summary being valid. It neither reproduces the model nor establishes causal attribution of excess error.

| Cell | Model | Reported mean | Reconstructed oracle mean | Mean excess |
|---|---|---:|---:|---:|
| N2_delay0ms | M4_Count | 2.4521386885 | 0.4663902416 | 1.9857484469 |
| N2_delay5ms | M4_Count | 2.4470159139 | 0.4663902416 | 1.9806256723 |
| N2_delay10ms | M4_Count | 2.4453415602 | 0.4663902416 | 1.9789513186 |
| N2_delay0ms | M4_Latency | 2.2893801570 | 0.0995216573 | 2.1898584997 |
| N2_delay5ms | M4_Latency | 2.2870166667 | 0.0995216573 | 2.1874950095 |
| N2_delay10ms | M4_Latency | 2.2872642221 | 0.0995216573 | 2.1877425648 |
| N3_delay0ms | M4_Count | 2.5286101375 | 0.4774184421 | 2.0511916954 |
| N3_delay5ms | M4_Count | 2.5195403075 | 0.4774184421 | 2.0421218654 |
| N3_delay10ms | M4_Count | 2.5188492806 | 0.4774184421 | 2.0414308385 |
| N3_delay0ms | M4_Latency | 2.2558190242 | 0.0969753947 | 2.1588436295 |
| N3_delay5ms | M4_Latency | 2.2544477703 | 0.0969753947 | 2.1574723756 |
| N3_delay10ms | M4_Latency | 2.2547306943 | 0.0969753947 | 2.1577552997 |

No paired excess bootstrap or covariance estimate is possible from aggregate-only model means; no such interval is produced. Excess is about 1.979–2.051 for Count and 2.157–2.190 for Latency. Thus fixed-alphabet quantization already makes 0.05 infeasible on these reconstructed histories, but it is far smaller than the observed error. Upstream retrieval, dynamics, decoder behavior and delivery failures cannot be distinguished by subtracting these means alone.

## Reproduce

```sh
python finite-dataset/reconstruct.py
```

Requires existing NumPy (no install performed) and the hash-checked source/JSON inputs one directory above. Outputs `results.json`, `reconstructed_targets_N2.csv`, `reconstructed_targets_N3.csv`. No ODE code imports. Approximately 1.5 seconds locally. `SHA256SUMS` records all artifacts plus source inputs.
