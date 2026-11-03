# Independent fixed-alphabet oracle verification

2026-11-02. Input evidence: `../independent-review.md`, pinned TAN commit `85213f0d417d54e4f48b301ed1c0e4200ef3c421`. This check takes the documented decoder alphabets and iid Uniform[-3,3] target model as inputs; it does **not** independently re-fetch source or replay the repository's frozen datasets. Only the newly written standalone stdlib Python script was executed. No dependencies, repo execution, publication or remote changes.

## Findings

The exact arithmetic agrees with all fixed-alphabet constants and CDF expressions in the independent review. No derivation error was found within this scope.

| Alphabet | Exact population E_comp oracle | Per-role MAE | 1m-pair Monte Carlo mean | MC standard error |
|---|---:|---:|---:|---:|
| Count {-2,-1,0,1,2} | 25/54 = 0.462962963 | 1/3 | 0.463184764 | 0.000232857 |
| Latency: 20 values -2.85+0.30k, plus zero | 3139/32000 = 0.09809375 | 0.073125 | 0.098165155 | 0.000035963 |
| Latency without zero (reference only) | 1/10 = 0.1 | 0.075 | 0.100058348 | 0.000035330 |

MC uses seed 20261102, shared target draws across alphabets, and therefore correlated—not independent—codebook comparisons. The original 200,000-pair run is preserved in `results_200k.json`; the 1,000,000-pair extension is `results.json`. The latter extends the same seeded stream, not an independent replication. No seed selection was performed. Actual latency's exact mean falls just outside the approximate 95% normal interval (z=1.986); a 95% interval is not guaranteed to cover every run. All exact means lie within their conservative Hoeffding 95% intervals. Do not present all normal intervals as passing. The exact rational calculation, not MC coverage, establishes these constants. Confidence intervals presume the ideal iid sampling model; PRNG output is a numerical approximation to it.

## Independent integration method

For each alphabet, form clipped Voronoi cells on [-3,3]. Split each cell at its codeword into left/right arms with lengths l_j. Then

`F(t) = sum_j min(t,l_j)/6` for t>=0.

The script derives all breakpoints from arm lengths, rather than hard-coding the review's CDFs. Between consecutive breakpoints, F is linear. Exact `fractions.Fraction` arithmetic integrates `1-F(t)^2` and `2t(1-F(t)^2)` to obtain E[M] and E[M²], where M=max(d(U),d(V)). Scalar MAE integrates `1-F(t)`.

Derived CDFs:

- Count: F=5t/3 on [0,1/2]; F=2/3+t/3 on [1/2,1]; then 1.
- Actual latency: F=7t on [0,3/40]; F=1/20+19t/3 on [3/40,3/20]; then 1.
- No-zero latency reference: F=20t/3 on [0,3/20]; then 1.

## Pointwise proof and fixtures

For every pair of real errors, `(abs(eA+eB)+abs(eA-eB))/2=max(abs(eA),abs(eB))`. If each output lies in C, each absolute error is at least its nearest-codeword distance. Hence **for every episode**, E_comp >= max(d_C(A),d_C(B)). Choosing each nearest codeword attains equality in the relaxed oracle problem. This proof covers arbitrary coupled/randomized decoders with the specified output alphabet; target independence is needed for the population F² calculation, not the pointwise inequality.

The script exhaustively checks all permitted output pairs for 121 representative rational target pairs per codebook: 104,786 exact-rational cases verify the metric identity and inequality, and each target pair admits oracle attainment. Fixtures supplement the proof; a finite grid is not itself a universal proof.

## Population versus finite dataset

The constants above are population expected errors, **not** deterministic lower bounds on a finite dataset mean. At codeword-valued targets a finite oracle dataset has zero loss; under continuous uniform sampling exact codewords have probability zero, but small neighborhoods around them have positive probability, so arbitrarily low finite means are possible. A valid deterministic dataset certificate is the average of its actual per-episode nearest-alphabet max distances. The original frozen histories were not extracted here; no finite-dataset certificate or bootstrap conclusion is claimed. Reusing histories at multiple delays does not add independent histories.

Both actual-alphabet population floors exceed the registered 0.05 threshold. This constrains these fixed numeric decoders, not spiking neurons universally, continuous-time codes, all single-channel designs, or the original dataset's bootstrap CI. Numeric silence=0 is included optimistically, and physiological attainability or delivery criteria may make actual performance worse. Rational decimal values idealize decoder constants; ordinary floating-point rounding is negligible for the reported precision but is not an exact bitwise execution audit.

## Reproduction

```sh
python oracle-validation/validate_oracle.py --samples 200000 --seed 20261102 --output oracle-validation/results_200k.json
python oracle-validation/validate_oracle.py --samples 1000000 --seed 20261102 --output oracle-validation/results.json
```

JSON includes exact fractions, derived CDF segments, second moments, MC sample uncertainty, both interval types and finite-sample counterexamples. Script default is 200,000 pairs; pass 1,000,000 to reproduce the primary table. Runtime here approximately 1.6s and 5.6s respectively.
