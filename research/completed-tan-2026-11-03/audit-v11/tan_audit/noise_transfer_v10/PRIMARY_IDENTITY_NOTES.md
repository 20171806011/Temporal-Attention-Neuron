# Primary identity notes for prospective v10

Source/literature review, 2026-11-02. No simulations, estimators, numerical outcomes, or design selection. These are standard identities and their application assumptions, not novelty claims.

## Exact population identities

Use finite-valued output `Y` (a finite spike word, count, or detection), stimulus condition `S`, and the **complete** input-noise realization `N`. Fix model parameters, initial state, window and readout. For a deterministic model `Y=f(S,N)`, each conditional output law is a point mass, so

`H(Y|S,N)=0`, and `I(N;Y|S)=H(Y|S)−H(Y|S,N)=H(Y|S)`.

Independence of `S` and `N` is not required for this identity. A coarse noise summary does not generally determine `Y`, so replacing `N` by a mean, amplitude, bin, or late segment can invalidate the zero-entropy premise. All unconditioned randomness in parameters, initialization or observation must also be accounted for.

**Primary support:** Shannon (1948), §6, properties 1 and 5, corrected-reprint pp.11–12, establishes zero entropy for a point mass and conditional entropy as the mean entropy of conditional laws. This directly supplies the deterministic step; Shannon's original notation differs from modern conditional MI notation. [Original paper, corrected reprint](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf#page=11). Cover and Thomas (1991), p.22, Eqs.2.60–2.62, gives conditional MI and its chain rule. [Original textbook chapter](https://www.cs.columbia.edu/~vh/courses/LexicalSemantics/Association/Cover%26Thomas-Ch2.pdf#page=11), [publisher record](https://onlinelibrary.wiley.com/doi/10.1002/0471200611.ch2).

**Continuous-noise caveat:** `N` may be a continuous Gaussian vector while `Y` stays finite. Use discrete conditional output entropy averaged over `N`; do not substitute differential entropy or assert that a deterministic continuous output has zero differential conditional entropy. Gray's author-posted corrected first edition, §5.5, p.121, Lemma5.5.6, explicitly gives `I(X;Z)=H(X)−H(X|Z)` for finite `X` and general `Z`; applying it within each `S` condition justifies the mixed-alphabet identity. [Primary textbook](https://ee.stanford.edu/~gray/it.pdf#page=143).

## Independent randomized thinning

Let `Y=g(S,N,U)` and `U` be thinning randomness independent of `(S,N)`. Then

`I(N;Y|S)=H(Y|S)−H(Y|S,N)`;

`H(Y|S,N,U)=0`, hence `I(N;Y|S,U)=H(Y|S,U)`.

These are different estimands. Independence of `U` does **not** imply its effect on `Y` disappears. It also does **not** guarantee strictly positive `H(Y|S,N)`: the output law must actually be nondegenerate on a set of positive probability. For ordinary Bernoulli retention with `0<p<1`, a finite native spike word containing an event has multiple possible thinned words; a word with no events, or retention `p=0` or `p=1`, does not. A compressed readout requires checking whether thinning can change that readout.

The above is a direct application of the cited conditional-entropy definitions, not an observed v10 result. By the conditional-MI chain rule and `I(N;U|S)=0`,

`I(N;Y|S,U)−I(N;Y|S)=I(N;U|S,Y)≥0`.

Thus reporting the quantity conditioned on thinning draws as though it were the ordinary noise-transfer quantity would change the question.

## Count then detection: data processing

For a fixed count `C=c(Y)` and `D=1(C≥1)`,

`I(S;D)≤I(S;C)≤I(S;Y)` and `I(N;D|S)≤I(N;C|S)≤I(N;Y|S)`.

The conditional chain applies separately within each stimulus condition and then averages. It remains valid with randomized upstream outputs because the downstream maps are deterministic. It does not assert strict loss: equality is possible. If the detector receives extra onset timing, labels, or side information, include those resources consistently in the representation and conditioning; do not claim the displayed chain for different inputs.

**Primary support:** Cover and Thomas (1991), §2.8, pp.32–33, Theorem2.8.1 and its function-processing corollary. [Original textbook chapter](https://www.cs.columbia.edu/~vh/courses/LexicalSemantics/Association/Cover%26Thomas-Ch2.pdf#page=21). Gray §2.5, pp.36–37, Lemmas2.5.1–2.5.2, also gives information reduction under deterministic measurements. [Primary textbook](https://ee.stanford.edu/~gray/it.pdf#page=58).

## Finite-sample interpretation

Paninski (2003), §3.3 p.1200 and §4 pp.1202–1204, establishes downward plug-in entropy bias but explicitly rejects a universal positive-bias assertion for plug-in MI: support structure and finite-sample concentration matter, and MI bias can be negative. §9 p.1234 warns that common bias corrections may leave substantial bias and resampling error bars can understate total estimation error. [Original paper, author-institution PDF](https://www.cns.nyu.edu/pub/lcv/paninski-infoEst-2003.pdf#page=10).

Panzeri and Treves (1996), §3.1 pp.91–93, Eqs.6–9, gives support-dependent leading bias terms; §4 p.100 distinguishes fixed from random stimulus counts. Its term “conditional information” is the stimulus-specific quantity defined in Eq.1, not general three-variable conditional MI. [Original paper, coauthor-hosted PDF](https://people.sissa.it/~ale/Pan%2B96a.pdf).

Design implication, not an estimator result: record sampling units, each conditioning stratum's sample count, possible/occupied output support, logarithm base and whether uncertainty includes estimator bias. Full-word, count and binary outputs have very different support sizes. If every continuous `N` path occurs once, treating its exact identity as a categorical sample key makes empirical `H_hat(Y|S,N)=0` mechanically—even when population thinning is random. That empirical degeneracy cannot establish population determinism. Algebraic reduction for a known deterministic model and empirical validation of an estimator are separate claims.

The identities do not establish useful-signal preservation: zero output has zero signal information and zero noise information. Likewise, `I(N;Y|S)` is not automatically equal to unconditional `I(N;Y)`, and conditioning must remain unchanged when comparing processing stages. No conclusion here revises the separate P1-A evidence or validates the historical MI cascade estimator.
