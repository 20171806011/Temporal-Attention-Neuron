# One history-split external decoder diagnostic on original M1

Frozen2026-11-02 before new histories, features or learned-probe outcomes. This follows the original-project map; there is no enriched carrier, architecture sweep or model selection. Manuscript corrections are a separate scientific artifact and do not change executable sources or historical labels.

## Scientific question and why a learned probe

During native query presentation I_content=0, so A does not affect the u/r trajectory or native output. This exact structural fact alone does not show that ignored A carries useful target information. Compare a single fixed supervised probe family at successive observable stages to test recoverability, while retaining all native dynamics and frozen calibration. A-only endpoint intervention is a structural check, not the main localization result.

An external learner's success shows predictive information accessible with its supervised computation and direct feature access. It is not a neuronal implementation or a proof that only one native stage is at fault. Probe failure is no evidence of universal information absence. No true keys, values, positions, N, delay, history ID or trajectory enter learned features; labels enter training only.

## Frozen histories and split

Original TaskGenerator clean N=2 and N=3 distribution, unchanged minimum spacing, values, slots and query contract. Generate1792histories per N with continuous RandomState2026110220 for N2 and2026110221 for N3; metadata seeds never reset RNG. Within each N, RandomState2026110222 generates one permutation (same stream continues from N2 toN3): first1024train, next256validation, last512test. Thus3584unique histories:2048train,512validation,1024test. All two role queries and all three delays0/5/10ms from one history stay in one split. No reuse of previous analysis inputs.

Save/hash complete inputs and split IDs before native model/learned metrics. Native M1 source, parameters, event timing and initial state stay frozen. Read g_dec,b_dec,A0 unchanged from historical summary_p1b.json; do not refit native calibration. Run encoding once per history with independent probes from each delayed checkpoint. Verify r_cue=0 for every observation; its omission from variable learned features follows the registered final silent interval, not an inferred label. Stop if this invariant fails.

The validation set is reported once after fitting and never used for hyperparameters, feature selection, stopping, threshold choice or refitting. The test set is evaluated once after model parameters and transformations are frozen. No performance-driven retries.

## Fixed feature views and controls

One degree4 total-degree polynomial ridge family, including intercept, with these prespecified views:

1. QUERY_ONLY: current q,5monomials includingintercept.
2. NATIVE_OUTPUT_Q: exact native deadline u_readout and current q,15monomials. Use the returned pre-reset candidate/reset semantic, not post-step u.
3. CUE_U_Q: query-onset u and q,15monomials.
4. CUE_U_A_Q: query-onset u,A and q,35monomials; permitted r is verifiedconstantzero.
5. SHUFFLED_STATE_Q: same35-term family, but u,A are from a donor history at the same delay, q/label from the recipient. Fit this negative-control learner independently on its shuffled training data.

For shuffled state, RandomState2026110223 makes a random cycle independently within each N/split. Each recipient receives the next history's state in that cycle, so there are no self-donors. Reuse the donor assignment for both queries/all delays, match delay, and never move donors across splits. This intentionally destroys history alignment and also changes the q-state joint distribution; deterioration is not uniquely a causal A effect. Save donor IDs.

Nonlearned controls: FROZEN_NATIVE uses the original g_dec*u_readout+b_dec; ZERO predicts0; CORRECT_HISTORY_ORACLE uses existing M0 nearest-key lookup over raw candidate key/value records. The oracle is privileged evaluator memory, not a neural decoder or an added model input. Its outputs must match targets; no oracle answer feeds any learned view.

## Fixed solver and output rule

For each view, use training-only raw-feature means and population standard deviations; zero-variance dimensions use scale1. Construct all monomials of total degree0..4 in deterministic exponent order. Center/scale nonconstant expanded columns using training-only means/population SD (zero SD ->1). Keep intercept1. Solve the deterministic ridge objective mean squared target error +lambda*sum(nonintercept coefficients squared), lambda=.001, using NumPy linear solve. Intercept is unpenalized. No stochastic optimizer, search or validation selection. Save transforms, exponent order, coefficients, matrix/condition diagnostics and source/runtime hashes before validation/test predictions.

Use a prespecified bounded prediction clip[-3,3], reflecting the original value support. Save every raw prediction as well, raw/clipped metrics and clip counts. No truth-dependent clipping or threshold. This is an explicit external readout policy. The views have different effective capacities (5/15/35terms), so comparisons establish recovery within this frozen family rather than optimal-decoder or capacity-matched information theorems.

## Metrics, inference and structural control

Report all train/validation/test results for full distribution and all-positive-novelty stratum, with N-specific and pooled counts, all delays, unconditional Ecomp=maxoperanderror, per-role MAE/RMSE, strict.5joint accuracy and clip counts. Stratum labels are evaluation metadata only; fit one model per view on all training histories, never subgroup models. All-positive subgroup sizes may be modest and results remain descriptive. No new P1B pass labels.

Primary contrast: test CUE_U_A_Q minus FROZEN_NATIVE mean Ecomp. Secondary fixed contrasts: NATIVE_OUTPUT_Q minus FROZEN_NATIVE; CUE_U_Q minus NATIVE_OUTPUT_Q; CUE_U_A_Q minus CUE_U_Q; SHUFFLED_STATE_Q minus CUE_U_A_Q; QUERY_ONLY minus CUE_U_A_Q. Compute each history's mean Ecomp across its three delays before uncertainty analysis. Fixed2000paired percentile bootstrap replicates RandomState2026110224, resampling histories separately within N and combining with observed counts; do this for full test and all-positive test. No significance stopping or multiple-comparison-adjusted discovery claims. Save all contrasts, including null/unfavorable outcomes.

For the first8test histories in ID order per N, everydelayandquery, compare native query runs after replacing only A_cue by0 or5 (withinstatebounds), leaving u,r,q and all parameters fixed. Require identical native output and spike sequences; record96queriedcheckpoints and192interventionwindows. This demonstrates query-stage A invisibility only. It does not show A contains values, nor represent a naturally reachable encoded history. All empirical learner features still use unmodified original states.

Interpretation rules: improvement from native-output+q supports recovery beyond the frozen affine/direct-access policy; improvement from cue-u+q supports recovery inaccessible through that specific native query/deadline observation under this probe; improvement from full endpoint supports A-assisted recovery within this family. None alone uniquely separates a physical causal mechanism from conditioning, feature access or approximation capacity. If full-state recovery does not beat query-only/shuffled controls, report no demonstrated recovery with this probe, not absence of memory. Preserve the distinction between raw-history oracle success and compressed-state availability.

## Verification and stopping

Save perhistoryinput/split/donorledgers, all native endpoints/readouts, exact allowed feature matrices and rowmetadata, fitted model artifacts, all raw/clipped predictions, metrics and paired intervals. Independently verify splitdisjointness, perhistory grouping, train-only preprocessing, absence of forbidden features, negative-control mappings, deterministic solver and native recurrence, all metric denominators and no query mutations. Check a fixed subset against original runner.evaluate_episode. Check native source hashes before/after. Stop after this one comparison and independent review; no installs, purchases, remote edits, publishing or historical label changes.
