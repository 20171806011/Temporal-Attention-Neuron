# Executed paired-history mechanism experiment

2026-11-02. **Twelve actual frozen-model executions**, not a proposed experiment. Minimal deterministic diagnostic, not a population success test or new confirmatory claim.

## Main results

1. Original M1 irreversibly erases the first value in this fixture: its complete [u,A,r] trajectories for -0.5 and +0.5 are **bitwise identical**, including encoding, delay and cue, at all three delays. This is exact equality, not a small-value tolerance.
2. Existing constant-gate M2b preserves a nonzero value-dependent state difference. At first-event offset delta-u=0.03923418271, falling to 0.000004627429 at encoding end. During cue and added silence it fades further. This is pairwise distinguishability/local sensitivity, **not** successful key-addressed or robust multi-item retrieval.
3. All 12 runs have **zero spikes** in all phases; max voltage candidate <0.527, well below threshold 1. Thus spike/reset nonlinearities do not confound the analytic subthreshold recurrence.
4. The frozen affine output is ~0.041–0.043 for both target signs: +0.5 passes the strict error<0.5 delivery tolerance but -0.5 fails. Neither arm distinguishes both pair members sufficiently to deliver both. This “both members” statement concerns two counterfactual histories, **not** the repository's two-query joint delivery metric.

| Model | Added delay ms | Delta-u cue deadline | Delta calibrated prediction | Full trajectories bitwise identical |
|---|---:|---:|---:|---|
| M1 | 0 | 0 | 0 | True |
| M1 | 5 | 0 | 0 | True |
| M1 | 10 | 0 | 0 | True |
| M2b | 0 | 1.69378860881e-06 | 2.24648566999e-07 | False |
| M2b | 5 | 1.02475238462e-06 | 1.3591374598e-07 | False |
| M2b | 10 | 6.19981409988e-07 | 8.22286409241e-08 | False |

## Protocol and analytic check

Protocol saved before execution in `PROTOCOL.md`: slots[0,4], keys[0.8,1.8], first values +/-0.5, second value0, single query q0.8, M1 versus existing M2b. Added delays {0,5,10}ms were fixed in advance, not selected after inspection. No training, gain adjustment, gate-floor modification, or timing search. Both models retain frozen code and saved calibration.

M1 begins A=0; first key0.8 produces zero novelty, so first-event current A*(key+value)=0 regardless of value. Subsequent identical inputs produce identical states.

For M2b, in the verified no-spike regime, delta-u after the first100 steps is A0*delta-v*(1-0.99^100). Each later common-input step multiplies it by0.99. At encoding end the exponent is900, at cue deadline1000+10*delay_ms. The affine prediction delta is g_dec times this voltage delta. Observed boundary residuals from this formula are below1.4e-16 absolute; the prespecified tolerance was1e-15. Zero delta-A and delta-r hold exactly. Added5ms multiplies the residual trace by0.99^50 (~0.605006), and added10ms by0.99^100 (~0.366032).

## Source, environment and safety

- Commit85213f0d417d54e4f48b301ed1c0e4200ef3c421.
- Full model and runner source read before execution. Model imports only math, NumPy, dataclasses, typing and scipy.stats.norm, with definitions/constants and no top-level I/O. Runner contains writing/tamper-test routines: it was **read only, never imported or run**.
- Model SHA2566b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f matches frozen source evidence. Script checks source and summary hashes before import.
- Only `simulate_encoding`, `simulate_delay`, `simulate_cue_and_readout` were called. A transparent wrapper around `euler_step` records its returned states without altering recurrence inputs or results.
- Frozen full-precision calibration: {"A_0": 0.06188672587424413, "g_dec": 0.1326308169898006, "b_dec": -0.02681648355488859}. Reviewer's rough analytic note used rounded constants; tiny numerical differences from that note reflect use of full precision, not retuning.
- Python3.12.14, NumPy2.3.5, SciPy1.17.0. Existing libraries only, no installation.
- No external publication, repo modifications/push, or original benchmark artifact overwrite. Runtime ~1.1seconds.

## Artifacts and reproduction

```sh
python paired-mechanism/run_fixture.py
```

`results.json` contains all readouts, calibrated prediction errors, success flags, phase-boundary states, analytic deltas and residuals. Twelve CSVs contain every post-step state, voltage candidate and spike, with phase/time labels. `SHA256SUMS` fingerprints artifacts. Repeat execution reproduced JSON and all trace hashes exactly.

## Scope

The collision confirms a known first-event erasure mechanism for this fixture, not a universal theorem against three-state or single-channel neurons. M2b removes that exact collision on this pair but retains a severely attenuated signal and fails two-sign delivery with the frozen readout. No noisy robustness, learned nonlinear diagnostic, multi-item retrieval performance, or population success is inferred. Further rescue experiments must not equate nonzero state separation with usable memory.
