# V8 frozen protocol: fresh-cohort test of the POINT loss-by-learning-rate interaction

## Question, provenance and scope

Start from completed V7 commit `0c74816da453003490f313c8ee7939e16840e4f2`. Preserve every V1–V7 artifact and main. V7's POINT loss-by-learning-rate interaction was exploratory and motivated this new question: does its negative direction recur in a fresh, larger paired training/data cohort under exactly the same task and four configurations? This is a prespecified new-data replication within the same implementation and AI-assisted workflow, not external independent replication or peer review. Do not pool V7 observations into V8 or select between reporting V7 and V8 results.

Only POINT is trained in V8. This study cannot establish a DEND-versus-POINT difference or structural superiority. No architecture search, extra loss/LR level, task change, independently tuned comparison or subsequent phase is included. The earlier request concerning an extra pulse generator was an author-confirmed misattribution; no such module existed. No separate pulse module is added or removed. Native firing stays unchanged.

## Fixed count and precision planning before any new data

The V7 twenty paired POINT interaction values have sample SD0.5125263075142555. `plan_sample_size.py` reproduces the calculation from the frozen V7 output; `PLANNING.json` records its source hash. Fix **104 paired training/data blocks and416 training runs**, four conditions per block. This is the smallest integer n for which `t(.975,n−1)*s_pilot/sqrt(n) <= .10`: with df103 and t critical1.9832641447734565, planned half-width is .0996735790, or9.9674 percentage points. The target concerns one paired mean interaction, not individual-cell accuracy or failure risk.

This is a plug-in precision calculation, not a power guarantee or a promise about the realized interval. The pilot has only20 blocks and mixed successful/collapsed fits. Increasing its SD by1.5 yields planned half-width14.9510pp at104; that sensitivity value is not a variance confidence bound. Meeting10pp under that larger SD with the same t calculation would require230 blocks, which is not the present study. V8 is deliberately fixed at104 before any new outcome. Report realized SD, SE and interval half-width. No outcome-dependent extension, optional stopping, seed replacement or re-estimation of sample size.

V7 POINT training took74.6972s for80 runs, projecting approximately388.4s for416 runs on the current single-thread CPU. Existing free storage exceeds25GB; projected artifacts are roughly1.6GB. The fixed study is technically feasible without paid compute. Caps are900s for training and diagnostics,600s for held-out evaluation/artifacts and900s for each audit. These bounds accommodate overhead; they are not outcome-based stopping rules.

## Fixed native model, task and training

Copy V7 `model.py` and `data.py` byte-for-byte. POINT has176 float64 trainable parameters,20 continuous state values, six stored previous hidden spikes and two output-count accumulators. Twelve native-spiking branch cells feed six native-spiking hidden somata and two outputs. Branch/hidden/output firing uses strict candidate>.5 and reset; all forward spikes remain hard binary. There is no forward label argument, label-conditioned timing, separate pulse generator or analog decoder. Labels enter loss and metrics only. Original initialization is unchanged: input/recurrent/output weight SD .45/.25/.5 and branch/hidden/output biases .15/.05/.1.

The exact24-step delayed-XOR generator has six channels. Four-step first-bit cues start at2–4; second-bit cues start at10–12. Independent amplitudes are uniform [.8,1.2]. Distractor channel5 occupies7–9; common readout channel4 occupies16–23. Gaussian noise SD.03 is added and clipped below zero. Each dataset balances all four bit pairs; target is XOR. Informative cues end by15. Count native output spikes at16–23 and require a unique largest count. All ties, including both counts zero or both counts eight, abstain and count as errors. Every episode starts with zero state and has no interepisode/batch statistics.

Four fixed conditions: RATE/.01, RATE/.03, CE/.01 and CE/.03. Each of104 blocks shares exact inputs, labels and sampled initial parameters across all four conditions. Every run gets128 training episodes,200 full-batch Adam updates, beta1=.9, beta2=.999, epsilon1e-8, global gradient-norm clip1 and parameter bounds[−2,2]. Final weights only; no validation selection, ranking, early stopping or checkpoint tuning. Test all416 final models once, including failed training fits, on512 fresh balanced episodes per block.

CE uses .5*counts logits; RATE compares count/8 to .75 for correct and .125 for other output. Both add .0001*mean(theta²). Surrogate slope `1/(1+5*abs(candidate-.5))^2`, with reset derivatives, is the same proxy learning rule, not the exact derivative of the discontinuous hard-spike objective. Loss effects include the specified scaling, target/logit conventions, gradients, clipping and Adam interactions. Equal nominal learning rate does not imply equal effective parameter steps. Failure after200 updates does not establish lack of representational ability.

Literal `SEEDS.json` fixes s=0..103: train2080000000+10*s; test2080010000+10*s; initialization2080020000+s; analysis2080030001. Verify the313 registered train/test/init/analysis integers are distinct and absent from prior population-study text artifacts before data generation. Train and test RNG streams are separate. Independently regenerate exact arrays and check no episode duplication, while recognizing uniqueness alone does not prove independence.

## Single primary estimand and decision rule

For block i, let R1,R3,C1,C3 be unconditional held-out accuracies at RATE/.01,RATE/.03,CE/.01,CE/.03. The **one primary estimand** is the population mean of paired interaction `delta_i=(C3−C1)−(R3−R1)` for independently generated seed/data blocks from this fixed procedure. Compute104 delta values first, preserving all four-cell covariance. Neither416 runs nor53,248 held-out episodes are the independent inferential sample size; n=104 blocks.

Primary two-sided95% interval: `mean(delta) ± 1.9832641447734565 * sample_sd(delta,ddof=1)/sqrt(104)`, df103. This Student-t interval has approximate coverage for the mean of these bounded mixture-distributed block effects; exact normality is not asserted. The five-percentage-point negative interaction is the inherited practical threshold. Prespecify interpretation by this PRIMARY interval only:

| Primary interval | Interpretation on this task and fixed budget |
|---|---|
| Upper endpoint < −.05 | Supports negative direction and a magnitude exceeding5pp under this interval procedure |
| Upper endpoint <0 but >=−.05 | Supports negative direction; magnitude exceeding5pp remains unresolved |
| Interval includes0 | Inconclusive direction, not equivalence or evidence of no interaction |
| Lower endpoint >0 | Evidence in the opposite direction from V7 |

Report the complete interval and estimate even if these labels are unfavorable. A degenerate interval is not proof of population certainty or equivalence. Do not use a narrow task result to make structural, biological, energy or hardware claims.

Prespecified sensitivity:20,000 percentile bootstrap samples of104 whole paired block indices, RNG2080030001, quantiles.025/.975. Compare its direction with the primary interval and report any disagreement. Never switch the primary decision to the bootstrap after results. One primary hypothesis does not need a multi-primary correction; bootstrap is a sensitivity calculation for the same estimand, not another success criterion.

Seven fixed coefficient vectors in `CONTRASTS.json` cover interaction, loss main, LR main, and four simple effects. The other six accuracy contrasts are secondary exploratory descriptions; their intervals are pointwise bootstrap95. Apply all seven vectors to binary run-failure outcomes as secondary risk differences. Also report four test-minus-train accuracy differences. Eighteen interval records total: seven accuracy, seven failure, four train/test, with the primary additionally carrying its Student-t interval/SD/SE. Secondary findings do not get promoted to primary or receive family-wide confirmatory interpretation. No architecture contrasts occur.

## Retained outcomes and diagnostics

Save all initial/final weights and every clipped gradient, pre-update loss, raw gradient norm, update norm and native output counts for all200 updates. Save nine fixed checkpoints after0,1,5,10,20,50,100,150,200 updates with activity/margin summaries. Retain full native final-training and test spikes/counts/predictions/states/candidates. These diagnostics never alter training.

Per model report accuracy, coverage, abstentions, class recalls with abstentions wrong, 2×3 confusion (predicted0/1/abstention), signed count-margin mean/median/histogram−8..8, episode silence, nonsilent ties and both-saturated8/8 fraction. Keep branch/hidden/output activity and readout candidate-minus-.5 quantiles descriptive, never as a new decoder or energy proxy.

Failure means accuracy<.90 OR coverage<.90. Fully silent means every episode silent; fully abstaining means coverage0 and can also be nonsilent saturation. Retain every failure; report all four train/test pass/fail transitions. Marginal failure, fully-silent and fully-abstaining run proportions use two-sided95% Wilson intervals with z1.959963984540054,n104. No observed events does not prove zero population risk. No episode-bootstrap pseudoreplication.

## Gates, verification, interruption and preservation

Commit protocol, planning, seeds, contrasts and executable scientific sources with freeze hashes BEFORE generating V8 training data. Deterministic contract/gradient fixtures may precede this gate and are not scientific splits. After all416 runs complete, save the training lock and commit all weights/checkpoints before any held-out data are generated. Execute the guarded test generator/evaluator only once. No test-based repair or automatic retry of a partial/observed scientific run.

CPU-only existing NumPy environment, one BLAS/OpenMP thread, no installation or paid compute. SciPy already installed is used for planning and independent Student-t quantile verification; model implementation remains NumPy. Abort and preserve partial evidence on numerical failure or runtime cap. A partial study is incomplete; do not silently drop/reseed problematic trajectories or use a different n. Record any departure without rewriting frozen scientific files.

Pre-data checks: exact native parity with V7, batch split/permutation/reset tests, label-free forward signature, strict threshold/reset/binary fixtures, and locally anchored proxy finite differences. Inherited fixture checks include both model branches for704 scalar derivatives; DEND is not scientifically trained or compared in V8. The erased-prefix no-history fixture checks state equality before the second cue, not performance of a restricted-memory classifier.

Independent checker: regenerate inputs/initializations; reconstruct all83,200 Adam updates from saved clipped gradients and all3,744 checkpoint vectors; replay native equations at every saved training checkpoint. Compare full final-training traces for53,248 model-episodes and full held-out traces for212,992 model-episodes: spikes/counts/predictions exact, state/candidate tolerance1e-10. There are66,560 unique train/test histories (13,312 training and53,248 test). Independently rederive contrast coefficients, Student-t statistics, paired bootstrap and Wilson intervals.

Gradient audit: recompute updates0/99/199 for1,248 sampled loss/gradient/count checks; test all176 locally anchored proxy derivatives at416 final checkpoints,73,216 scalar derivatives. These sampled checks reuse the frozen analytic backward. The separate native/data/Adam/statistics checker does not independently derive every proxy gradient or replay native trajectories at every unsampled training update. All verification is internal workflow consistency checking, not external independent replication or peer review.

Deliver complete code/configuration/planning, all data/seeds/weights/gradients/failures, per-seed/cell/contrast tables, resource and execution costs, numerical audit outputs, figures, report with What this does not establish and Limitations, and full-file SHA-256 manifest. Preserve V1–V7 and main; local branch `research/dendritic-population-v8`, no push or merge in this task. After reporting this fixed study, stop; do not run another phase or expand the grid. Research direction/native-firing constraint: Li Zexu. Protocol, implementation, execution, verification and report draft: AI-assisted.
