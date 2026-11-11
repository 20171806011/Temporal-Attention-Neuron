> Historical archival document. Original language preserved. Personal paths/host metadata and unavailable media links were sanitized; consult README.md and ARCHIVE_INDEX.json.

# TAN trained state mechanisms and bounded matched budget comparison

Status: prospective protocol for this local milestone, frozen before new scientific data. Historical V9/V10 results were read before choosing the questions. This is exploratory follow-up, not external preregistration. Date of authorization: 7 November 2026, Europe/London. Source revision: 7dcddee41490d8d917936891989eeb7ddb67e0da.

## Questions and interpretation

The primary question is where an already trained no-cross DEND solution carries information about the first bit of delayed XOR. Reuse every one of the five V9 no-cross DEND final weights, without selecting seeds by this experiment. Five intact DEND weights are a separate robustness cohort; five no-cross POINT weights are a descriptive floor control. These are fifteen existing fits, not fifteen new independent training replications. V9's cross-cell cuts and no-cross training are not repeated.

The principal intervention exchanges one compartment's state at the start of step 9 with its state from a counterfactual history in which only the first latent bit changes. Amplitudes, timing, nuisance, noise and second bit are matched exactly. Future inputs from step 8 onward are identical. Compare branch state, hidden-soma membrane state, output membrane state, previous hidden spikes, branch plus hidden membrane, and all four state groups. A same-pattern donor permutation and a same-time identity copy control for transplant mechanics and nonspecific state disruption. A state transplant changes the internal state of a fixed model; it does not create a new trained architecture or prove a task-level impossibility.

Primary descriptive contrast, in the five no-cross DEND blocks only: for component c, F_c = fraction predicting the flipped XOR label after counterfactual transplant minus that fraction after same-pattern transplant. Report F_branch minus F_soma, its five paired values and a pointwise percentile bootstrap 95% interval (20,000 resamples, seed 2611079001). A mean difference of at least 0.20 with lower interval endpoint above zero will be described as evidence favoring branch-mediated information in these trained solutions. Otherwise call this criterion unmet. This heuristic is not a protected confirmatory test or a claim of exclusive storage. All other contrasts are exploratory, unadjusted, and reported irrespective of direction.

## Task and phase definitions

V9 delayed XOR uses 24 steps, six currents, balanced four bit pairs, first four-step cue starting at 2, 3 or 4, second at 10, 11 or 12, amplitudes U[0.8,1.2], nuisance on channel 5 at steps 7–9, common query current on channel 4 at 16–23, and clipped independent Gaussian noise SD 0.03. The same RNG order as V9 is used. A counterfactual is constructed before adding the same noise realization; only the first cue channel changes. Labels never enter neuronal updates. Intervention assignment uses stimulus metadata solely to construct the causal comparison.

Generate 512 new histories per frozen weight block with seeds 2611070100+10*s, s=0..4. These data have not been used for training or selection. All model families share each block's data. Same-pattern donors use a fixed cyclic shift of the index list for each of the four patterns; donor mappings are saved. Data, counterfactuals and labels are all retained.

Single pre-update state interventions occur at step 5 (write: every first cue is active), step 9 (hold: both informative cues absent), and step 16 (query start). Zero branch d, soma v, output o, previous spike p, or all four. Identity-copy shams occur at exactly these same times. The hold period includes nuisance and noise; it is not a silent, long-delay memory task. Comparisons across phases reflect different causal opportunities, not identical information content.

Path interventions last exactly two updates: write 4–5, hold 8–9, query 16–17. Separately zero local branch-to-soma transmission, only within-cell soma-to-branch feedback, or hidden-to-output transmission. Native spike production and state updates otherwise continue. Same-window multiply-by-one path shams must reproduce baseline exactly. No new cross-cell cut is included. Transplant, reset, and path-cut effect sizes have different interventions and should not be interpreted as equal physical perturbation strengths.

## Native model and decoder

The vendored V10 model gives the generalized V9 equations at h=6. Parameters have shapes (6,2h), (h,2h), (2h,), (h,), (h,2), (2,), totaling 2h²+17h+2. For branch state d, hidden membrane v, output membrane o and previous hidden spike p:

```
dc = rd*d + .25*(x@wi + p@wr + db)
DEND: local = max(dc,0); d = dc
POINT: bs = 1[dc>.5]; local = bs; d = dc*(1-bs)
vc = rs*v + .5*sum(local within each two-branch cell) + vb
hs = 1[vc>.5]; v = vc*(1-hs)
oc = rs*o + p@wo + ob
os = 1[oc>.5]; o = oc*(1-os); p = hs
```

Every episode begins at zero state. Mechanism trials use rd=.85, rs=.65. Count native output spikes at 16–23; only a unique largest count predicts, and ties/silence abstain and count wrong. DEND branches are mathematical graded compartments, not asserted literal biophysical dendrites. POINT already has 2h additional native spiking branch cells and subthreshold memory. No external spike generator, analog decoder or label-dependent neuronal signal is introduced.

## Fair comparison protocol and bounded pilot

Use the V10A feature-binding distribution: all 16 patterns equally represented; y=(b0 AND b1) OR (b2 AND b3); high U[.8,1.2], low U[0,.1], feature window 8–15, query 16–23, nuisance 7–9 and clipped Gaussian SD .03. Held-out data test new amplitude/noise realizations of the same patterns, not compositional extrapolation. This task has brief retention but no isolated hold window.

Run exactly h={2,6}, both DEND and POINT, and four candidates per size and architecture: RATE/.01 with (.85,.65), CE/.01 with (.85,.65), RATE/.01 with (.95,.85), CE/.01 with (.95,.85). Both architectures may independently select either retention pair, including slower retention. This small factorial does not exhaust possible timescales or learning rates; initialization drive coefficients are unchanged when retention changes, so the manipulation affects charging as well as memory.

Tuning uses two paired blocks, 128 training and 256 validation episodes each. Train seeds 2611071000+10*s, validation 2611072000+10*s. Initialization seed 2611073000+100*h+s, paired across architectures/candidates. Select separately for each architecture and h by highest mean validation balanced accuracy, then highest mean coverage, then smallest candidate index. All 32 candidate fits and failures remain recorded. No held-out test is generated at this stage.

After saving a selection lock and all candidate artifacts, train the four selected architecture/size cells on five fresh paired blocks. Train seeds 2611074000+10*s, initialization 2611075000+100*h+s. There are exactly 20 final fits. After locking all final weights, generate untouched test data (512 episodes, seed 2611076000+10*s) and evaluate every fit. Data and initialization namespaces are disjoint from both mechanism data and historical V9/V10. Models receive the same input information and spike-count decoder. Training uses 200 full-batch Adam updates in every fit, beta=(.9,.999), epsilon=1e-8, global gradient clipping at 1, theta in [-2,2], and regularizer .0001*mean(theta²). RATE and CE follow vendored V10 loss definitions. Surrogate slope 1/(1+5*abs(candidate-.5))² trains native hard-spike forwards; it is not a hard-spike derivative. Fixed final checkpoint; no restarts, early stopping, extensions or seed replacement.

Report every final seed's balanced accuracy, ordinary accuracy, class recalls, pattern recalls, coverage, silence/ties, and pass (BA>=.90 and coverage>=.90). Two paired architecture contrasts (one per h) use five final training/data blocks and pointwise bootstrap intervals; tuning seeds and episodes do not inflate n. Seeds share selection on only two tuning blocks, so intervals condition on that selection and omit tuning-selection variability.

Resource comparison targets observed equal-performance frontiers at BA=.90 and .95, coverage>=.90. A sampled cell is provisionally eligible only if at least 4/5 final fits meet both gates; report all eligible sizes, means and uncertainty. With only two h values this is a sampled pilot, not a global frontier. If an architecture has no eligible point, report no demonstrated matched-performance comparison, not dominance or impossibility. A full subsequent frontier would need a new frozen size/candidate grid, equally budgeted independent tuning and fresh test blocks; it is not automatically authorized or executed here.

## Resource accounting

Per episode retain continuous-state updates (3h+2)*24; h previous-spike registers and two decoder counters separately; parameters; DEND h+2 spiking cells plus 2h graded compartments, versus POINT 3h+2 spiking cells. Record native branch, hidden and output events separately. Count actual consumed recurrent event deliveries using the previous spikes and each source's nonzero recurrent fanout, and hidden-to-output deliveries likewise (last-step outgoing hidden spikes are not counted as delivered outside the simulated window). POINT branch events each deliver to one soma. DEND has 2h*24 scheduled graded local transfers, including zero-valued transfers. Dense input MACs are 12h*24.

Forward dense MAC terms are (2h²+14h)*24 per episode; backward terms are (4h²+16h)*24, excluding input gradients. Dense zero multiplication is counted. Report tuning, final training, validation and test budgets separately; scalar nonlinear/reset operations, loss, optimizer, file I/O and audits are outside these MAC counts. Report measured process CPU time, wall time, thread settings and peak working set where available. Events, fanout, state updates, transfers and MACs remain separate coordinates: do not combine them using arbitrary energy weights, or infer joules/ATP/biological time/hardware savings.

## Integrity, stopping rules and verification

Use existing Python/NumPy, one BLAS/OpenMP thread, CPU only. Pre-data deterministic fixtures check exact native parity, strict thresholds, binary/reset behavior, no-op interventions, full-state donor rescue, and finite differences of the anchored surrogate graph for both architectures/retention pairs at h=2,6. No scientific seed is used in these fixtures. Freeze protocol, sources, weights and seed specifications with SHA-256 before data generation. Local hash/timestamp evidence is not a trusted external timestamp or a historical Git commit.

Each scientific stage refuses an existing output directory. Caps: mechanism 300 s, tuning 300 s, final training 300 s, test 300 s, and independent verification 300 s. A nonfinite result, invariant failure or timeout saves partial files and failure details, stops that stage without seed substitution or automatic retry. Numerical or code corrections require an explicit amendment, preserving old freeze/failure evidence; no test-guided tuning. All completed and failed fits are retained. Resource exhaustion is a limitation, not a scientific negative result.

Save data, weights, complete clipped gradients/pre-update counts/losses for all new fits, per-condition final counts, spike and state trajectories for the first 32 mechanism episodes, metrics for all episodes and intervention specifications. Independent implementation must regenerate datasets, replay native updates for every mechanism condition and final test, reconstruct every Adam update, recompute seed-level statistics, and compare counts/predictions exactly and retained continuous states within 1e-10. The checker does not import the experimental forward/training implementation. This is independently coded internal verification by the same AI workflow, not external replication or peer review.

Original repository files and all earlier failure labels, especially P1-B, are preserved. No old runner is invoked in its evidence directory, no GitHub push/publication/third-party contact, no software installation, no safety/credential changes, and no deletion of existing research material. New structural state-constrained retraining is outside this initial milestone; existing V9 no-cross fits and new fair-comparison training must not be conflated.
