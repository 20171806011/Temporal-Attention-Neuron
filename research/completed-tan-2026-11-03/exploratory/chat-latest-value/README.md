# Conditional latest-value associative memory: AI-assisted concept audit

## Scope

This is a newly implemented mathematical/numerical concept audit. It is **not**
TAN/P1-B production code, a neuronal/ODE simulation, or a rerun of v11.
These are AI-assisted Chat-sandbox experiments; same-agent reruns and checking
are not independent external verification.

The tested exact-contract learner stores `C`, `b`, and a distinct-direction count.
It receives only the incoming feature vector and value. The test harness keeps
canonical features/labels to score predictions; those are not learner inputs.
No trained model is used.

## Reproduction

The executed environment was Python 3.13.5, NumPy 2.3.5, SymPy 1.14.0.

```sh
python latest_value_audit.py --seed 20261102 --out results.json
```

Random tests use `numpy.random.RandomState`; the exact-stream test uses seed
20261102, the classification test uses seed+1, and the noisy-dual test uses
seed+2. The same final script was run twice in the same environment and produced
byte-identical result JSON files. This does not promise bitwise identity across
other BLAS/platform/version combinations.

`results.json` contains the complete structured scientific output. Operational
console logs are omitted. `HISTORICAL_SHA256SUMS.txt` records original archive
hashes; `PUBLIC_SHA256SUMS.txt` checks the edited public files. Neither establishes
independent verification of a pre-existing TAN experiment. See `PUBLIC_REPRODUCTION.md`.

## Tested claims and limitations

1. **Exact rational arithmetic:** seven new-key/old-key writes in dimension 3,
   checking interpolation, latest-value invariant, dual identity, direct w/P
   update equivalence, and the full-capacity null projector.
2. **Floating-point exact-contract streams:** 24 streams, d=8, six distinct
   keys, 50 writes each (1,200 total). Candidate feature matrices are explicitly
   restricted to condition number <=10. These are not ill-conditioned tests.
   C pseudoinverse cutoff is `rcond=1e-13`; admission threshold is `1e-9`.
3. **Noisy geometric classification:** 2,000 old and 2,000 new observations,
   R=2, zeta=.001, projector error=.002, new-direction margin=.04, threshold=.02.
   A separately constructed overlapping-observation case violates the margin.
4. **Noisy dual bound:** 1,000 random one-step old-key updates. The script
   compares the analytical vector-error identity to numerical predictions and
   checks the corresponding operator-norm upper bound. These are not
   population-confidence or infinite-horizon guarantees.
5. **Nullspace contamination:** an old noisy write leaves an unobserved b
   component outside range(C); a subsequent new-direction admission makes it
   visible. Projecting the old write into range(C) removes this specific failure.
6. **Near-collinear keys and finite precision:** float64/float32, gaps from .1
   to 1e-8, feature error 1e-6. This test explicitly uses
   `rcond=100*eps(dtype)` and records retained numerical rank. Rank truncation
   changes the algorithm; it is not exact Moore-Penrose interpolation.
7. **Regularization:** fixed two-key gap .001, six lambda settings. Closed-form
   cross-talk predictions are checked against linear solves.
8. **Baselines:** exact distinct-key dual overwrite, additive repeated-sample
   averaging, and normalized plain delta, on the same nonorthogonal features.
9. **Guarded old-write update:** range projection plus leverage normalization
   stabilizes a fixed noisy observation. The test also records that the true-key
   values remain incorrect; stability is not address correctness.

## Principal output

- Exact rational checks: passed (7 writes).
- 1,200 moderate-condition writes: maximum key-prediction absolute error
  8.43769498715119e-15.
- Maximum direct-vs-C/b w-update difference: 4.39648317751562e-14.
- Classification: zero errors among the 4,000 prescribed cases; this is not a
  universal statistical accuracy estimate.
- Noisy-dual identity: maximum residual 6.473009435181481e-14; all 1,000 bound
  checks passed.
- Naive nullspace example: new key intended value 0 instead reads .001;
  projected update reads 0.
- Two-key gap 1e-6 and feature error 1e-6: float64 readouts approximately (0,1)
  instead of the intended (1,0), despite full-span classification as old.
- At gap .001, lambda=1e-6: readouts (.6666662222,.3333331111), illustrating
  regularized loss of exact protection.
- Additive averaging / normalized delta / dual overwrite respectively return
  (.5,0), (1,1), and (1,0) in the designated nonorthogonal overwrite fixture.

## Interpretation

The exact algorithm is conditional on each incoming feature being either an
exact old feature or outside the existing span. It cannot verify this promise.
Neither projected feature novelty nor invertibility solves general membership.
No numerical test here supports claims about original TAN success rates,
universal neuron limitations, biological feasibility, or physical energy.
