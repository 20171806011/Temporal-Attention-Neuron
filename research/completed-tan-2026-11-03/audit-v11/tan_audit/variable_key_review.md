# Independent review of the completed variable-key test

Reviewed `variable_key_test.py`, its registered protocol, all 2,560 attempt
records, summary, and first witnesses. The recorded protocol/script hashes match
the actual files. The primary run's 1,882 accepted pairs (988 narrow, 894 full)
are internally consistent. No primary artifact was rewritten during this review.

## Verdict

No material algebra, branch-validation, domain, or target-leakage defect was
found in the primary existence result. The evidence supports a model-specific,
same-query ambiguity of the reconstructed modified carrier despite invertibility
when supplied all keys. It does not estimate the original IID task failure rate.

The recurrence coefficients correctly use OLD gate A for membrane drive. Since
event 1 precedes event 2, the missing D2*s1 term is identically zero. Eliminating
D2 through the gate state, solving s2 through membrane state, and recovering
`k2=(D2-.1*s2+1.1)/.9` is correct on the positive-novelty branch. The full formula
and native Euler outputs are checked separately. Both histories must pass the
pre-reset candidate margin, spike, clipping, refractory, key spacing, and value
domain checks. The fixed target shifts remain above twice the respective stated
tolerances.

The unique-root inverse uses supplied keys, state, and fixed domain only; it does
not inspect true values when choosing roots. Ground truth appears in evaluation
after inversion, which is appropriate. All reported accepted-history inversions
have one retained root and error far below the chosen .05 diagnostic threshold.
The acceptance step does use ground-truth history to construct and certify
counterexamples, as required by an adversarial existence test; it is not a
learned decoder's held-out test set.

## Independent exact-rational check

`independent_variable_key_check.py` does not import the primary implementation.
It derives the 100-step block transition independently using geometric sums:

```
am = 99/100; aa = 1499/1500; bm = 1/100
C = (am^100-aa^100)/(am-aa)
S = (1-am^100)/(1-am)
A' = aa^100*A + (1-aa^100)*D
u' = am^100*u + bm*s*(A*C + D*(S-C))
```

All quantities use exact Python Fraction arithmetic. For the first narrow and
full anchors, the check reconstructs the transformed history from the decimal
anchor and the exact target shift, verifies all task-domain constraints, and
obtains exactly equal rational u and A. Refractory state is zero in both.
Gate is monotonic within each constant-input block; membrane updates are convex
combinations of prior u and A*s. Exact block bounds therefore certify the
nonspiking/unclipped branches without relying on a rounded near-threshold test:

| Domain | Exact target separation | Conservative entire-history u upper bound | Gate upper bound |
| --- | --- | --- | --- |
| Narrow | 1/5 | 0.14851016995906716 | 0.08396464157545566 |
| Full | 6/5 | 0.5334692055018946 | 0.12360066055680154 |

These exact rational witnesses are reconstructed ideal histories, not a claim
that the rounded double-precision paired JSON states are literally equal.
Results are in `independent_variable_key_check.json`.

The independent check also validates all accepted records' same-q condition,
fixed target shift, value/key domain and key spacing, recorded branch criteria,
state/probe residual thresholds, attempt IDs, and totals. It does not re-run all
1,882 pairs' trajectories; the exact two-witness check supplies an independent
validation of the primary existence claim.

## Reporting qualifications

1. The known-key inverse receives complete state, including A, plus all keys,
   event positions, and exact dynamics. It does **not** reconstruct from only
   the two membrane probe samples. Thus its success demonstrates privileged
   full-state inversion, not successful two-sample analog retrieval. The first
   narrow witness even has probe sample100=0 after query-driven spiking. Keep
   this distinction explicit in the main report.
2. The added .1 content probe is combined with the query current q. Report it
   as this specific diagnostic. The test does not reproduce the earlier archive's
   reported noisy probe-readout performance because the archive was unavailable.
3. The native diagnostic calls the repository function, whose default query-phase
   tauA is15; the candidate encoding/probe use150. This does not change any native
   diagnostic membrane/spike observable because native query I_content is zero.
   Also, the raw analog diagnostic uses default g_dec=1,b_dec=0. It establishes
   observational equality, not a calibrated native prediction benchmark.
4. The fixed-key inverse filters quadratic roots by value domain but does not
   re-simulate alternative roots to reject a potential reset/clipping branch.
   That could conservatively add abstentions in other data. Here no accepted
   history abstained, and its unique solution is the validated nonspiking history,
   so this has no demonstrated effect on the reported diagnostics.
5. All anchors use k>1.4 and target the first chronological event. This is a
   legal subset of the original task support, sufficient for an existence
   counterexample. It is not the full original target-selection distribution.
6. Do not convert 1,882/2,560 into a population failure percentage. Two shifts
   share each anchor; acceptances are manufactured pairs and filtered by domain.
   No binomial CI or average-case risk estimate follows from those counts.
7. Exact same-state/same-q histories imply the claimed average absolute error
   bound for their equal-weight two-point distribution, and preclude both
   members passing the stated strict tolerance. This applies to the current
   state-and-query information contract, including any deterministic future
   state-only probing. It does not rule out architectures retaining extra
   history, receiving other keys, or using another state encoding.

No remaining blocker or necessary change to primary outputs was identified.
