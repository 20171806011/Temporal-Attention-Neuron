# V4 protocol — DRAFT, NOT FROZEN, AWAITING AUTHOR CONFIRMATION

> **Correction of record (2026-11-03).** The extra module previously requested does not exist in the original implementation. The request was a misattribution by the author, Li Zexu, confirmed on 2026-11-03. No code is missing, and no existing module was deleted or altered. This item is closed and places no requirement on V4.

No V4 datasets have been generated, no V4 model trained and no V4 test opened. This is a reviewable proposal resolving the pending items and internal conflicts in Li Zexu's requirements. It is not authorization to freeze or execute. Baseline: V3 commit `630525bff536be5abf640ec71616b016ee421238`; V3 protocol/checkpoint commits `44ca749`/`93a239b`.

## 1. Pending decisions and proposed defaults

| Item | Concrete proposal |
|---|---|
| D1 | Five paired initialization/data seeds for A, B and C. Same seed index pairs DEND and POINT. |
| D2 | A: 100-update primary endpoint; 200-update secondary endpoint. Save step 100 and step 200 from the same prespecified Adam trajectory; never choose between them after test. B/C: fixed 200 updates. |
| D3 | 24-update sequences, zero-based indices. L1 informative pulse at 12–15 inclusive; readout cue/counts at 16–23. L2 informative pulse at 8–11; blank informative channels at 12–15; readout remains 16–23. |
| D4 | Local research branch only. No push, public PR, merge or main-branch edits. |
| D5 | Execute B from L0 regardless of A success. Stop immediately at the first level where both structures fail the all-seed gate; do not continue higher even if A already examined L3. |

Approval must also resolve the methodological amendments below. None is silently treated as an accepted change to the user's specification.

## 2. Invariants

Use V2 native forward equations and initialization distributions, V3's exact surrogate derivative, 176 parameters in the intact model, float64 precision, native strict >.5 threshold/reset, hard binary spikes, common data access and unique-output-count winner. Labels never enter the forward function. Preserve signed weights, leaks, all coefficients and 24 updates. Input class determines external stimulus as in V2/V3; it never schedules output spikes. No analog decoder, architecture search, energy claim or comparison to a different model family.

Perform deterministic parity with preserved V2 code at the start of each executed stage. C masks weights as an explicit intervention, not a change of neuron equations. No old source/checkpoints/frozen protocol will be changed.

## 3. A — distinguish paired optimizer controls from broader configuration selection

### Problem in the supplied causal claim

Comparing fresh Adam runs against historical V2 changes data, initialization seeds, learning-rate search and potentially loss as well as optimizer. RATE and CE are different objectives. Consequently, Adam success does not by itself prove that the historical failure was caused solely by SPSA.

### Proposed minimal repair, requires confirmation

1. Keep the V2 delayed-XOR generator byte-identical; retain all timing/amplitude/noise/readout choices.
2. On each of five fresh paired training/validation sets and identical initial parameters, run a CE-only control with both SPSA and surrogate-BPTT/Adam at learning rates .01 and .03 and 100 updates. Compare methods at each fixed learning rate; do not compare a RATE winner against a CE historical result. SPSA uses its frozen V2 perturbation schedule, coordinate clipping and parameter bounds; Adam uses V3 moments/global clipping/bounds. Thus the factor is the complete specified **training method**, including gradient estimator/update/clipping, not just the Adam moment formula.
3. Run the requested Adam CE/RATE × .01/.03 search to steps 100 and 200. Its CE/100 checkpoints supply the Adam side of the paired control, so there is no duplicate training. Preserve all runs. SPSA's new .01/.03 grid is explicitly not a reproduction of V2's historical .03/.10 search.
4. For each budget select one common Adam (loss, LR) configuration across both arms and all five seeds using exactly V3 ordering: maximize minimum validation accuracy, then mean accuracy, minimize mean validation RATE error, smaller LR, then RATE before CE. A's 200-update selection is the sole B/C configuration, even if A itself fails. B is a new task-difficulty diagnosis, not a test of a failed A checkpoint.
5. Report CE paired effects separately from RATE results and from 100→200 budget effects. Main-arm success supports an effect of the tested training method under these task/settings/seeds; it does not establish that SPSA is incapable of learning, or that all historical failure is solely attributable to it. Secondary-only success remains method/budget-confounded. If both fail, neither task impossibility nor insufficient representational capacity has been established.

Both structures get the same method-specific hyperparameters and search budget. SPSA uses 2 forward evaluations/update; BPTT uses 1 forward +1 backward. With batch 128 and 24 steps, count separately the forward matrix MAC terms (156/sample/time) and backward matrix MAC terms (240/sample/time), plus timings. Never call equal updates equal compute. There are 40 Adam trajectories of 200 updates and 20 SPSA runs of 100 updates: 8,000 BPTT and 2,000 SPSA updates in A. These additional SPSA controls require acceptance of this repair.

## 4. B — prespecified task ladder

Train every level from the same paired initialization distribution, from scratch, with A's common 200-update configuration. Do not warm-start from the preceding level. Five seeds per arm, 128 training/128 validation histories per seed/level. Only development histories are used to identify failures. Failure means ANY seed has validation accuracy <.90 or coverage <.90. Record every seed's accuracy, coverage, abstention and all-silent fraction, not just means.

| Level | Frozen proposed informative input | Cue and decoder |
|---|---|---|
| L0 | V3 generator unchanged: one of channels 0/1, onset randomly 2/3/4, sustained through 23. | 16–23 |
| L1 | Four updates on channel 0/1, exactly 12–15; no blank step before cue. | 16–23 |
| L2 | Same four-update pulse, exactly 8–11; four blank informative steps 12–15. | 16–23 |
| L3 | V2 delayed-XOR generator unchanged, two jittered cues and XOR labels. | 16–23 |

L1/L2 retain balanced labels, uniform amplitude [.8,1.2], distractor channel 5 at 7–9, Gaussian noise SD .03/clipping, and common cue channel 4. Generate matched L1/L2 nuisance draws within each seed to isolate the pulse's position relative to fixed readout. Input cessation refers to informative signal: background noise remains.

### Timing and causal scope amendment, requires confirmation

A four-step pulse immediately before a fixed readout cannot simultaneously retain V3's random early onset. Therefore L0→L1 changes onset/jitter and duration; L2→L3 changes cue count, encoding and target rule. Those are composite task-family transitions, NOT strict single-factor causal comparisons. L1→L2 fixes pulse duration and shifts its position while keeping distractor/readout timing fixed; relative overlap with the distractor also changes. Report these concrete changes. Do not infer a universal monotonic capacity boundary from the first failure. The original claim that every level adds exactly one isolated difficulty cannot hold alongside exact V2/V3 endpoints, fixed readout and these short pulses. The proposed timing accepts this limitation instead of silently changing generators.

## 5. C — two recurrence interventions

Run only if B passes L1 or L2 in at least one intact arm. Choose the highest such level, common to both arms; run both arms at that level, with the same five seeds, data, initialization, loss, LR and 200-update budget.

- Retrained control: zero all recurrent weights from hidden cell h to branches not owned by h, before the first forward pass and after each update; zero corresponding gradients/moments. Retain self connections. Retrain from the same paired initialization with that mask applied.
- Inference cut: evaluate the intact selected checkpoint with those exact cross-cell weights zero, no retraining or reselection.

Each intact circuit has 176 effective trainable parameters. The mask fixes 60 cross-cell weights, leaving 116 effective trainable parameters in each retrained control; storage may remain 176 slots. DEND and POINT stay matched within each condition, but intact and masked controls are NOT equally parameterized. No fictitious unused trainable parameters will be counted. Neuron equations/state size and dense implementation shape stay unchanged.

Even when intact passes and retrained fails, restrict language to an observed dependence within this task, model family, optimizer, seed set and budget. Finite optimization failure cannot establish mathematical necessity or rule out a more successfully trained masked network. An inference cut alone only establishes dependence of the existing solution. The stronger unconditional 'necessary' wording is not supportable and is proposed for replacement.

## 6. Selection locks, testing and unresolved eligibility conflict

All protocol, code and seed schedules must be committed before generating any development set. Configurations, selected checkpoints and the eligibility table must be committed again before generating test histories. No post-test tuning. Each eligible frozen condition is evaluated once.

Proposed ceiling interpretation: if BOTH intact arms have accuracy >=.99 on EVERY development seed at a level, do not generate that level's test set; label it '预期达到天花板，未进行测试'. C at a ceiling level remains development-only; no exception is silently introduced for ablations.

The request simultaneously bans testing failed configurations and demands test confirmation of a passed-versus-failed structural difference. These cannot both be satisfied literally. **Default proposal preserves the ban:** test only eligible passing arms; report the failed arm's development result, but make no test-confirmed between-structure conclusion. If the author wants Q2's three-part confirmation reachable, authorize one narrow exception before freeze: include the already locked failed arm as a prespecified comparator on the same one-shot test, with no reselection. The same issue applies to a failed retrained C control; without that exception, C's negative-control conclusion stays development-only.

For an eligible task that is not ceiling-limited, propose five new initialization/data seed pairs, distinct from development. Train the already selected fixed configuration on fresh 128-example training data for these replicate seeds, without new selection. Commit these replication checkpoints before generating their 512-example test histories. This distinguishes fresh initialization replication from merely changing test-input RNG; all extra training is counted. No validation-based dropping of replication seeds.

Use 2,000 paired hierarchical bootstrap draws, resampling seed pairs and then label/input-pair strata. With only five seed pairs these remain exploratory. Direction consistency, opposite all-seed gates, fresh-seed reproduction and an interval excluding zero are ALL required for the specified structural-difference claim. If unmet, state that the criteria were not met; do not turn missing evidence into proof of equality. Degenerate intervals do not establish population equality or a non-inferiority bound.

## 7. Proposed RNG schedule and budget (not used)

Use explicit nonoverlapping ranges, checked against all v1–v3 logs before freeze:

- A training/validation: base 2026111000, offset 10s +0/+1; initialization 2026111100+s; SPSA directions 2026111200+s.
- B level L: base 2026112000+100L, offset 10s +0/+1; common level initialization 2026112500+s. Paired L1/L2 nuisance draws use the same development RNG but different deterministic timing; never reuse these as tests.
- C: reuse the selected B level's development histories/initializations for the intervention comparison; this is intentional pairing, not independent evidence.
- Replication level L: initialization 2026113000+100L+s; training data 2026114000+100L+10s; test data 2026114002+100L+10s; bootstrap 2026115000+L.

Five seeds: s=0..4. B costs 2,000 updates per executed level, max 8,000; C retraining costs 2,000. A max 10,000 updates including SPSA; total development max 20,000 updates. Fresh-seed confirmation, if eligible, adds 2,000 updates per level, plus masked retraining where separately eligible. Exact ceiling/eligibility is logged before test generation. Proposed wall cap 600 seconds for development training, 600 seconds for eligible fixed replication training, and 300 seconds for numerical verification; no installs, GPU or paid compute. Keep partial logs and stop at a cap without secretly reducing seeds/budget or restarting candidates.

## 8. Verification and outputs

Keep V3's finite-difference checks before training and at selection, clearly local surrogate derivatives rather than exact hard-network derivatives. Audit sampled loss/gradient computations and reconstruct every Adam update from saved clipped gradients. Preserve SPSA probe directions/objectives/clipped updates separately; these are not analytic gradients. Independently regenerate data and replay every test trajectory using separate native equations; spikes/counts/predictions must match exactly. Report residual maxima, failed checks/corrections and verification limits.

Deliver frozen protocol, code/configs, full weights/update traces, actual generated data and seeds, per-level/per-seed tables, resources (including masked effective parameters), verification logs, REPORT.md, scientific figures and SHA-256 manifest. The report must contain separate 'What this does not establish' and 'Limitations' sections and the attribution: research direction/native-firing constraints by Li Zexu; protocol, implementation, execution, numerical verification and first report draft AI-assisted. This is same-workflow verification, not outside replication or peer review. No merge into the TAN paper.

## 9. V3 correction and hash-preservation decision

V3 REPORT.md is absent from experimental FREEZE.json but is covered by MANIFEST.json. Appending changes its checksum even if the original prose is untouched. To honor the literal ban on changing any hash-listed file, this draft leaves V3's original report and manifest byte-identical and supplies:

- ../dendritic-population-v3/REPORT_WITH_APPENDIX.md: exact original bytes followed by the requested correction;
- V3_APPEND_ONLY.patch: an unapplied, append-only patch;
- CORRECTION_PROVENANCE.json: baseline commit, old/new SHA-256 and prefix verification.

Authorize either this preserved-original/new-version delivery, or explicitly authorize applying the report-only append on the V4 branch while preserving the historical commit/manifest (whose old report checksum would then describe the old version). No other V3 bytes will change. The record correction itself is accepted now; the nonexistent module is a closed issue, not a research blocker.
