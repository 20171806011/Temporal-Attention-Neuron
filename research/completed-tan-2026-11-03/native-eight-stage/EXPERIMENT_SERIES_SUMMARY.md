# TAN executed experiment series — consolidated audit index

2026-11-02. This index describes eight completed AI-assisted experiments, originally executed on 2026-11-02 and packaged publicly on 2026-11-03. Frozen model source is commit `85213f0d417d54e4f48b301ed1c0e4200ef3c421`; model SHA256 `6b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f`.

## Bottom line

1. **Original fixed readouts have unavoidable decoder floors.** Count oracle population E_comp=25/54≈0.462963; latency including zero=3139/32000=0.09809375. Neither can meet population 0.05 under stipulated iid uniform[-3,3] targets. This is a decoder-specific statement, not a theorem against spiking neurons.
2. **Original carrier has identifiable mechanism failures.** A below-novelty first event is erased exactly; even novelty-positive, fixed-key subthreshold encoding places multiple independent values into an affine rank-one endpoint dependence. Constructive collision witnesses hold in ideal arithmetic and are numerically verified with rounding caveats.
3. **Changing existing A can remove this local rank obstruction, but original query still cannot observe A.** Content-dependent gating produces rank two at fixed known keys/timing; modified query coupling and two analog samples add an observation route. These are new mechanisms/resources, not original TAN success.
4. **Timing and noise matter.** Wider spacing of the same two analog samples improves sensor/membrane-noise conditioning but worsens gate-process noise under the declared after-update model. Unbounded tangent inverse errors were explicitly treated as sensitivity diagnostics, not retrieval performance.
5. **A bounded nonlinear inverse reveals actual conditional risk.** It is globally unique on the fixed narrow box by a connected-fiber proof, not rank alone. Baseline parameters remain fragile under noise; full-denominator fallback and zero-baseline comparisons prevent selective success claims.
6. **The final fixed timescale comparison yields a substantial conditional rescue.** Increasing only tau_A from15 to150ms, while keeping tau_m10ms, reduces sensor-only unconditional E_comp from0.270803 to0.008093 on4096 fresh held-out histories; coverage/local0.05 joint success99.46%. Under membrane-process or combined noise, E_comp is0.04810/0.04951. Uniform slowing is weaker. This changes retention and charging/current resource costs and is not a general addressing result.

**The successful conditional system is not the original system:** it uses value-dependent A drive, a longer gate time constant, nonzero query content probe, two analog time samples, a known-equation nonlinear inverse, fixed known keys/positions, and targets restricted to[-0.5,0.5]². It does not demonstrate variable-key TAN retrieval, N3 success, a biological decoder, or network necessity.

## Executed artifact index

Each directory has a report, executable script(s), JSON results and SHA256SUMS. Trace/target/prediction ledgers are preserved where applicable. The script commands below require only existing Python, NumPy and SciPy; no production runner is imported.

| Directory | Actual execution and main caveat |
|---|---|
| `oracle-validation/` | Exact rational integration,104786 rational fixtures, fixed-seed200k/1m Monte Carlo. Actual latency's1m approximate normal95% interval narrowly misses exact mean (z1.986), transparently recorded; conservative interval covers. |
| `finite-dataset/` | Reconstructs4000 frozen-seed target histories; independent implementation matches reviewed generator arrays and full RNG states. No original ledger/historical runtime exists to certify historical byte match. |
| `paired-mechanism/` |12 frozen-model runs; M1 paired trajectories bitwise identical, M2b nonzero but tiny fading difference; no spikes, no usable two-sign delivery. |
| `collision-witness/` |4 frozen-model encodings; exact rational compensated collision construction. Float endpoints differ1/9ULP; not falsely called identical. Actual readout outputs coincide. |
| `content-gate-rank/` |3 arms,9-point grid, analytic/finite-difference Jacobians, actual cue blindness, extra probe observer and declared-noise local diagnostics. Baseline equality of offset/content only at v0. |
| `two-sample-timing/` |4096 paired noise paths, fixed m1/m100 and same100-step physics. Exact covariance checks, countervailing gate-noise result retained. |
| `nonlinear-heldout/` |4096 fresh bounded targets,10 schedule/noise cells, exact domain-filtered nonlinear inverse, zero fallback, roots/statuses/per-history predictions. Sensor-only later sample helps Ecomp but worsens first-roleRMSE versus zero. |
| `timescale-heldout/` |Final predeclared3-arm×5-noise comparison on4096 fresh paired targets; exact per-arm maps, full risks, primary paired bootstrap and resource proxies. No subsequent tuning sweep. |

## Reproduction commands

```sh
python oracle-validation/validate_oracle.py --samples 1000000
python finite-dataset/reconstruct.py
python paired-mechanism/run_fixture.py
python collision-witness/run_witness.py
python content-gate-rank/run_rank.py
python two-sample-timing/run_timing.py
python nonlinear-heldout/run_heldout.py
python nonlinear-heldout/summarize_uncertainty.py
python timescale-heldout/run_timescales.py
python timescale-heldout/analyze_noise_geometry.py
```

The200k oracle result is separately preserved; command for it is in its report. All final major results and per-history ledgers were repeated and hashes reproduced. `py_compile` passes for executable scripts. These are targeted tests, not a full production benchmark replay.

## Independent audit priorities

- Check **old A** is used in membrane updates and analytic tangents; source does not use updated A.
- Check pre-reset candidates and returned spikes, not only post-reset u. Model readout deadline semantics and eligible99 versus raw100 spikes remain distinct.
- Verify content-gate modification is active only during events; no extra encoder state is hidden in diagnostic tangent arrays.
- Inspect exact polynomial coefficient recurrence, determinant/fiber uniqueness argument and whole-box monotonic branch proof.
- Verify stable quadratic solver never chooses using true targets; both-coordinate domain filter,1e-10 roundoff/residual tolerances, all abstentions and fallback0 remain fixed.
- Check per-history noise recurrence is genuinely after-drift process noise and is paired across schedules/arms; equal raw noise across changed taus is not equal input-current noise.
- Verify all full-denominator risks, zero baselines and explicitly local0.05 success criteria. Original0.5 is trivial on the narrow prior and is not presented as success.
- Interpret `Sigma^(-1/2) O J Dv`, not observation conditioning alone; singular covariances preserve noiseless constraints instead of arbitrary regularization.
- Separate model-unit input/state proxy integrals from physical energy. Slowing tau changes leak and forcing; no pure decay-causal claim or free intrinsic-memory resource.
- Preserve finite-precision distinctions: ideal rational collision/global-map results versus actual float residuals and empirical inversion accuracy.

## Open research boundary, not executed

Random/unknown keys and timing, N3 and larger independent value sets, noisy encoding, realistic sensor/physical noise, finite-precision implementation, trained/biologically realizable decoding, and reset/carryover/throughput costs remain untested. Advancing there needs a new task/resource specification. This bounded conditional series stopped after the one final timescale comparison rather than expanding into another optimization sweep.
