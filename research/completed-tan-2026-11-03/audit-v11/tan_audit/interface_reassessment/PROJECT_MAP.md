# TAN research map: return to the registered project

2026-11-02; frozen repository commit `85213f0d417d54e4f48b301ed1c0e4200ef3c421`. This is a source/evidence map, not a new replay of every historical experiment. The exact input-interface audit is a separate task in this directory. Prior negative results and labels remain intact.

## 1. The original questions are distinct

| Project stage | Actual question and resources | Evidence status in this workspace |
|---|---|---|
| Original TAN / TAN-I | Can a membrane plus an explicit recent-input window produce habituation, saliency weighting and history dependence? The complete state includes the window H, not membrane voltage alone. | Repository reports sensory/phototaxis and natural-collision experiments. Their full empirical suites were **not rerun by the current independent audit**. The source distinction is directly verified: `snapshot/manuscript/sections/model.tex:18–42`; `CURRENT_RESEARCH_STATE.md:18–20`. |
| TAN-II architectural ladder | Which added operations suffice for routing, binding and external arithmetic composition within tested families? | Repository reports vector-QK, key/value separation, WTA and parallel-combiner experiments. The ladder itself says **sufficient, not necessary** (`snapshot/paper2/sections/s9_ladder.tex:20–25`). Prior static composition success is conditional on delivered operands and uses an explicit algebraic combiner; it is not an online neuronal implementation. |
| P1-A | Does a requested key select the correct stored candidate, rather than merely change a winner under a surprise drive? | Frozen report records 60,000 trials and 420,000 evaluations. Scalar squared-distance routing matches vector-QK hard selection; 2D is not necessary for this static task. This audit has inspected reports/code, **not independently rerun the full 60,000-trial ledger**. `p1_experiments/results/run_p1a_rev2/P1A_SCIENTIFIC_REPORT.md`; `P1A_FINAL_BASELINE_MANIFEST.md:85–90`. |
| P1-B | Can the specified online `(u,A,r)` carrier and frozen analog/spike readouts deliver operands under the registered timeline? | Historical closure reports M1 failing every prerequisite and all comparisons floor-limited; memory redundancy and population-coding necessity remain unestablished. H3 internal composition stays deferred. Current audit independently checks selected dynamics, proofs and sources, but does not claim a full historical 96,000-episode byte replay. `p1_experiments/p1b/P1B_FINAL_SCIENTIFIC_CLOSURE_REPORT.md:16,46,104–111`. |

The original window-based TAN and P1-B's closed three-coordinate carrier must not be used interchangeably. A result about lost P1-B input features or its refractory timer is not a result about every window-based TAN implementation.

## 2. Facts already known before this continuation

- Positive scalar scoring cannot select an interior key by changing a common scalar multiplier; P1-A's scalar metric counterexample already removes a universal 2D-necessity claim.
- P1-B V2.6 already predicts zero-novelty first-event erasure and the N=2 population upper bound 7/12, and anticipates M1 failure of the 0.80 prerequisite. It already includes quantization benchmarks and a subthreshold final-state collision fixture. These are not discoveries of the recent controls (`P1B_PRE_REGISTRATION_PROTOCOL_V2_6.md:147–165,229–247`).
- M0 has an explicit stored candidate-key/value buffer. Dynamic models do not receive that buffer at query. The runner retains targets for scoring; that is not model access (`runner_p1b.py:85–127`; protocol:106,112–133).
- The closure explicitly limits conclusions to the tested carrier and end-to-end decoders. README line37's claim of demonstrated distributed-circuit necessity exceeds that closure. No repository wording was changed.

## 3. What the current independent work actually adds

| Evidence category | Completed local evidence | Correct inference limit |
|---|---|---|
| Original-model replication/checks | Exact metric identity `Ecomp=max(|eA|,|eB|)`; native timing/support and silence-accounting checks; stronger clean Count/Latency support floors in `floors/native_support_recheck/`. | Diagnostic explanations of the frozen failures; original labels remain unchanged. These are not all-neuron capacity bounds. |
| Supplied-source reproduction | All 15 three-timescale and 10 nonlinear-heldout cells rerun and independently reconstructed; source-text provenance preserved. | Known keys/timing, narrow value prior, altered gate, extra analog access/probe. The old unavailable whole ZIP was not audited in full. See `timescale_review/REPORT.md`, `nonlinear_review/REPORT.md`, `FOLLOWUP_REPORT.md`. |
| New binding diagnostics | Exact ideal-state counterexamples for the modified carrier; correct versus mismatched ordered-key metadata controls. | Conditional inverse success is not query-only variable-key addressing. Constructed collision counts are not original-distribution prevalence estimates. |
| Alternative-algorithm controls | Ideal interpolation, fixed persistent-bit accounting and finite residual-write experiment. | Standard interpolation/residual learning with explicit changed ports, state/control, arithmetic, retention and readout. Useful resource controls; **not the original TAN research frontier to keep tuning**. `interpolation_control/`, `precision_binding/`, `finite_write/`. |
| New source-review finding | Paper2's P1-B insertion describes equations, readout and controls inconsistent with the frozen protocol/code. | A manuscript contract error, not a changed experimental outcome. See `MANUSCRIPT_CONTRACT_ERRATA.md`. |

## 4. Decision after the exact-interface check

Do not merge three different questions: **is the complete input waveform injective**, **does the finite carrier retain the relevant information**, and **can the registered query/readout expose it**. In the positive-novelty region of the registered positive-key domain, the event-level features `(c,n)=(k+v,k-1.4)` algebraically recover `k=n+1.4` and `v=c-k`. Thus an input-collision claim there cannot be inferred from a collision of c alone. A final-state collision can still occur after distinct input waveforms and is a different claim. Also distinguish the registered drive/Euler-port boundary from the upstream `simulate_encoding` Python API: that wrapper and the runner possess raw arrays to construct the waveform. There is no information-loss claim about the entire software API or the original Stage1 window-based TAN. The root audit verifies the complete actual path and exact admissibility separately.

Selected single next original-contract diagnostic: **stratify frozen M1 on the original-generator 4,000-history reconstruction (N=2 and3) across all three delays by chronological novelty pattern**, keeping original keys/values, slot timing, initial states and global readout calibration unchanged. Report every pattern, full-denominator totals and subgroup coverage. Include the all-positive subgroup and the broader at-most-one-zero-novelty subgroup identified by the study's waveform-plus-exact-query analysis. Do not fit a decoder or select a model on these subgroups. This asks whether failure persists where the relevant target is identifiable from the input contract, separating known erasure/transduction ambiguity from subsequent carrier/readout limitations. The primary study freezes and executes this bounded test; its results belong in a separate report. Because historical target bytes are absent, call this a reconstruction under the frozen generator/runtime, not a historical byte match. Preserve original confirmatory decisions and do not treat a subgroup analysis as renewed equivalence authorization.

If full-domain interface insufficiency is established, the smallest **information requirement** is to distinguish keys within the rectified region while keeping the content sum, for example a key-preserving feature rather than saturated novelty. This is a changed input contract requiring separate preregistration; it is not a claim that one extra scalar is universally minimal, nor that interface repair alone yields a sufficient memory/readout. No such redesign is authorized by this map as an already completed experiment.

Before new architecture work, use the source-contract errata ledger to align the manuscript's P1-B description with the actual registered model. Retain H3 as deferred. Do not infer that new states or privileged readouts are necessary merely because successive enriched toys can be made accurate.

## Provenance and limits

`PROJECT_MAP_SOURCES.json` records SHA256 for the inspected repository sources and selected local evidence reports. This source-review task ran only file reads/hashes and wrote scientific audit artifacts. It did not rerun historical suites, alter repository files, publish, install or spend. 
