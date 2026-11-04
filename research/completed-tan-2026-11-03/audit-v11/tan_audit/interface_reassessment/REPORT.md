# Return to the registered TAN questions

Completed 2026-11-02. The previous input-loss claim needs a narrower scope: **the registered P1B current map is invertible when novelty is positive.** Exact task-relevant collisions remain on its flat low-key branch, including after a positive primer has activated the gate. They do not explain M1's failures on all-positive inputs. One unchanged-model diagnostic now verifies that separation.

## What was actually available

The generator and runner retain raw keys, values, masks, target indices and queries. The Python `simulate_encoding` wrapper receives raw arrays and converts them into registered drives. The recurrent M1 boundary receives content `c=k+v`, novelty `nu=max(0,abs(k)-1.4)`, query current and shared constants. Mask/timing controls presentation; some ablations additionally reset state at active-event offsets. Raw arrays and targets are not supplied to the recurrent update or native query decoder. M0 explicitly retains access to the raw candidate cache and is an exception.

All tested pairs use identical masks, times, query values/order, model identity, calibration and registered initial conditions. M1/M2a/M2c/M4 start at u=A=r=0; M2b/M2d use the same frozen A0. Role queries independently probe the same checkpoint. No manual gate precharge, post-query carryover or changed parameters are introduced. The original window-based TAN is a different model and is not covered by a statement about this P1B boundary.

## Correct information result

All registered keys are positive. For nu>0, `k=nu+1.4` and `v=c-k`. Thus no nontrivial ideal same-full-waveform/same-query collision with different targets exists when both N=2 events have positive novelty. A control with equal content pulses but swapped positive novelties was explicitly rejected as a full-input collision. Binary64 content summation can merge values only at ULP scale here, not at the 0.5 operand tolerance.

More generally, within the declared fixed-N/full-presentation contract and the at-most-one-zero-event condition, an offline full-waveform observer can decode positive keys; a valid query that matches none must request the unique low event, with value c-q. Our diagnostic oracle receives the active-event list from the presenter mask. That mask/history is **not** an extra M1 input or memory. The broader statement depends on the stated mask/fixed-N/unique-low promise; it is not an unconditional decoder for an unknown mixture of histories containing arbitrarily many invisible events. All-positive queried keys can be identified directly from the positive channel without this low-event fallback.

The retained collision uses two low-key events. In the strengthened N=3 pair:

| History | Events at slots 0,2,4 | Queries | Targets |
|---|---|---|---|
| P0 | (2,0), (.8,1.5), (1,-.5) | (.8,1) | (1.5,-.5) |
| P1 | (2,0), (1,1.3), (.8,-.3) | (.8,1) | (-.3,1.3) |

Both present content (2,2.3,.5), novelty (.6,0,0), and identical complete timing/control paths. Actual native M1 gate values at the two ambiguous onsets are 0.03933973346 and 0.00270916313. Their gated content currents remain strictly positive throughout both events. This is therefore distinct from the already registered first-event A=0 erasure, while still relying on two events in the flat novelty region.

The target vectors differ by 1.8 in maximum norm. Any common response has worst-pair Ecomp at least **0.9**; an explicitly artificial equal mixture also has mean Ecomp at least 0.9. This is not an original-distribution prevalence or mean-risk estimate. The earlier N=2 low pair remains valid. M0 distinguishes both pairs because it sees the raw records.

Exact rational path checks and instrumented native calls cover six histories, all eight registered models, all three delays and both queries: 144 runner episodes and 210,000 checked Euler calls. Input/control equality covers masks, reset schedules, shared initial states and query drives; state trajectories, candidate voltages and spike outputs are also verified. Positive-novelty controls have different full inputs. A clamped-gate ablation can ignore novelty, or a restricted readout can merge distinct inputs; such downstream equalities do not contradict input-map injectivity.

## One completed contract-preserving diagnostic

We reconstructed the original 4,000 histories under the frozen generator and installed NumPy, kept the saved global calibration, and replayed only unchanged M1 across three delays. All 12,000 paired rows remain included. Ordered novelty patterns and the all-positive/one-zero/multiple-zero partitions were fixed before subgroup outcomes. The six whole-cell joint accuracies and mean Ecomp values match the historical JSON exactly; this is not a claim of historical trial-ledger byte identity or a full 96,000-episode replay.

At zero delay:

| Input subset | N | Histories | Joint strict ±0.5 success | Mean Ecomp |
|---|---:|---:|---:|---:|
| All positive novelty | 2 | 486/2,000 | 16/486 = 3.29% | 2.00134 |
| All positive novelty | 3 | 229/2,000 | 7/229 = 3.06% | 1.99741 |
| At most one zero event | 2 | 1,525/2,000 | 47/1,525 = 3.08% | 1.99350 |
| At most one zero event | 3 | 1,009/2,000 | 31/1,009 = 3.07% | 1.98555 |

All 66 pattern/partition/whole-cell summaries are saved, including every delay. For all-positive N=3, success is 8/229 at 5 and10 ms; N=2 remains16/486. The offline oracle recovers the requested values for all 2,534 unique histories in its prespecified sufficient-information subset, maximum error4.45e-16. Its history access is an evaluation instrument, never provided to M1.

The all-positive result shows that flat-branch input ambiguity is not the sole cause of native failure. It does not isolate carrier retention from query transduction or readout, nor estimate a causal effect of novelty: the strata also differ in key magnitude and dynamics. No subgroup is assigned a new original pass/fail label, and no floor-limited comparison is reinterpreted as memory redundancy.

## Research map and next-step decision

The [research map](PROJECT_MAP.md) separates four stages: window-based saliency/history dependence; the sufficient architectural ladder; P1-A static key selection; and P1-B online carrier/readout delivery. Historical full-suite findings are labeled repository-reported unless actually rerun here. Prior source reproductions and enriched interpolation controls remain preserved with their resource limitations.

Two source issues now deserve attention before another architecture experiment. Paper2's P1B insertion describes different state equations/readout, misidentifies M2a and misstates the operand and equivalence thresholds. README's distributed-circuit necessity wording exceeds the closure, which explicitly establishes only model-specific insufficiency and defers H3. The [source-contract errata crosswalk](MANUSCRIPT_CONTRACT_ERRATA.md) records exact locations and required corrections; no source or publication was changed.

**Decision:** close the interpolation-variant branch and keep the registered M1 result as the baseline. The selected minimal next experiment was the frozen novelty-stratified replay, now completed. Its useful next original-project question is where information is lost after a distinguishable input enters the existing carrier/query/readout chain; the present audit does not yet assign that loss to one stage. Align the manuscript description with the actual registered contract before drawing further architectural conclusions.

For a future guarantee over every currently allowed history, some front-end distinction between low keys is necessary. Replacing the second coordinate with a key-preserving feature, such as raw k while retaining k+v, is one concrete channel-level change; it is not a proven minimum-bit design and changes the novelty/gating contract. No extra state, cache or readout access is added here, and an interface change alone is not claimed sufficient for retrieval.

## Execution and provenance

`boundary_check.py` completed with exit0 in2.94s; `run_stratified_m1.py` completed with exit0 in4.66s. The independent checker also exited0, recreating all4,000 inputs and all12,000 native paired predictions/encoding states exactly in binary64, and verifying52,500 stored physical calls plus144 runner outputs and hashes. Its full scope is in REVIEW.md. Analysis plan, pre-execution amendment, source hashes, exact witnesses, actual-call arrays and complete prediction ledgers are included. The amendment only made both ambiguous N3 content currents nonzero before any fixture execution.

All historical negatives and original labels remain unchanged.   Unseen historical archive stages and unrerun full manuscript/P1-A suites are not represented as independently replicated.
