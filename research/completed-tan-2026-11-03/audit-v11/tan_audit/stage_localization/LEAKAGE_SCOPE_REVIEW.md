# Independent leakage and interpretation review

Reviewed 2026-11-02. This is a separate source/ledger review, not a rerun of the primary experiment or an independent refit. The numerical reviewer is separately checking fitted coefficients, native recurrences, all metric cells and bootstrap arithmetic.

Read in full: `PREREGISTRATION.md` and `run_probe.py`; inspected the frozen model's encoding, delay and cue/readout implementations, the generator's key support, `DESIGN_REVIEW.md`, completed `results.json`, input/split/donor/native-observation ledgers, and all five saved allowed-feature arrays. Primary files were preserved.

Source identities:

- Preregistration SHA256: `8ffcda523dfc042f9d64473c361ecc39c6f69b1f9d2b2af7994bbd35dea6c1ff`.
- Probe script SHA256: `31021b95d5bc76f9a0798c406e63337f4a2d9056bd9fbac8dd6c8ae20eb6f5f2`.
- Reviewed results SHA256: `f4b43e845a8642d9f5b7b72af4d035bb378d225a3e6dfbe482b51cebf9f7583f`.

## Leakage and contract findings

No forbidden learned feature or train/test leakage was found in the inspected source and saved feature matrices.

`run_probe.py:37–49` assigns splits before expanding histories into observations. The read-only ledger check verified all 3,584 histories and 21,504 query rows: exactly the six distinct delay/role combinations per history, one split per history, and query/target metadata matching the original input record. Each donor differs from its recipient, remains within the same N and split, and the assignment is a permutation. A donor is reused for all queries/delays of its recipient. There is no donor-mediated crossing between training, validation and test.

`run_probe.py:71–88` constructs only q, native u_readout, cue u, cue A, or the declared matched-delay donor u/A. All five saved matrices were reconstructed from the observation/donor records and matched element-for-element. N, delay, keys other than the current q, values, positions, masks, role labels, history IDs and targets are not additional learned features. Evaluator metadata in the same observation file does not enter these feature constructors. All recorded cue r values are zero.

Both raw-feature and polynomial-column standardization use the training mask only (`run_probe.py:20–25`). The ridge solve uses training rows/targets only, with the stated unpenalized intercept and mean-MSE scaling (`83–87`). Validation/test rows are transformed but do not supply fitted statistics or coefficients. Source flow freezes model artifacts before prediction; there is no validation-based selection, refit, or outcome-driven branch. This establishes the implementation path, not an independently certified timestamp history.

The native-output feature is faithful: it reads the returned `u_readout` (`74`), which the frozen model defines as reset voltage if the pre-deadline refractory state is positive, otherwise the deadline pre-reset candidate (`models_p1b_v2.py:258–269`). It does not substitute the post-step membrane voltage. Both role queries start independently from the same unmutated cue checkpoint. The raw-history oracle is evaluator-only and never enters learned features.

The all-positive classification `keys[event_positions] > 1.4` agrees with original novelty `max(0, abs(key)-1.4)` because the generator's key support is positive, [0.3, 2.5]. It conditions the test distribution; it does not isolate a causal novelty intervention.

## Two additional limitations to retain in the final interpretation

1. **The shuffled all-positive subgroup is not novelty-matched.** Donors are matched by N/split/delay, not all-positive status. Of 173 all-positive test recipients, 144 have donors outside that stratum. Thus this subgroup's shuffle contrast changes marginal state/novelty composition as well as history/query alignment. This follows the frozen protocol and is not leakage. Do not describe it as an isolated A or binding intervention, or as shuffling within the all-positive population.

2. **Shuffled losses have donor dependence.** Under the one-cycle permutation, each history supplies its own recipient variables and another recipient's state. Adjacent recipient losses can therefore share an underlying history. Resampling individual recipient loss rows does not preserve this donor-cycle dependence. The saved shuffled-versus-full-state percentile interval is appropriately descriptive, but should not be represented as a rigorously independent-history population confidence interval for that randomized control. This qualification does not affect the primary full-state-versus-native contrast, whose two losses are functions of the same recipient history only. No post hoc bootstrap replacement or new experiment is proposed here.

Other scope restrictions are already stated well in the protocol and should survive condensation: native-output+q changes direct q access as well as the output function; endpoint probes expose extra analog state and external supervised computation; degree-four views have 5/15/35 terms and unequal capacity; cue-u versus deadline differences can reflect conditioning/model approximation in addition to physical query loss; A invariance establishes structural invisibility, not useful content by itself; a failed limited decoder cannot prove absent information. Clipping is a declared bounded-output policy, with raw results retained.

## What the completed results support

The inspected result table reports a modest predictive signal at the original endpoint, recoverable by this fixed external learner. On 1,024 test histories, pooled across delays, full-state Ecomp is 1.890126 versus 1.977919 native, 1.976712 query-only and 1.977327 shuffled-state. The primary paired difference is −0.087793, with its descriptive 95% interval [−0.111071, −0.063862]. Full-state strict-0.5 joint accuracy is 4.13%, while the privileged correct-history oracle is 100%. This is not near-complete retrieval or a reversal of original task failure. Clipping affects only one of 3,072 full-state test pairs; raw Ecomp is 1.890424, so the observed mean improvement is not a clipping artifact.

The all-positive test subset has 173 histories (N2:119; N3:54). Full-state Ecomp is 1.780266 versus 1.974595 native; the primary difference is −0.194330 [−0.270234, −0.119664]. Its joint accuracy remains 5.97%. Report those denominators and conditional scope. The all-positive cue-u versus native-output contrast includes zero in its interval [−0.069727, 0.005952]; avoid claiming every stage comparison is decisive. Neither pooled improvement nor this conditioned result identifies a unique physical bottleneck.

The supported conclusion is: some target-predictive information remains accessible in the original endpoint under a richer, supervised external readout; the registered native query ignores A exactly, and the full-state probe improves modestly over its fixed controls. The result does not establish sufficient memory, an optimal decoder, a minimal state count, or a neuronal implementation of the probe.

## Execution evidence and remaining scope

A read-only Python assertion command completed with exit code 0, checking all history/role/delay groupings, donor split/N/no-self conditions, q/target consistency, constant-zero cue r, and exact reconstruction of the five allowed-feature matrices. Reported check time: 0.4211 s; peak RSS: 85,792 KiB. The completed primary results report 7.7394 s and 213,600 KiB, but those are primary-run records rather than a new run by this reviewer.  
