# Joint 95% population certificate for the two native arms

2026-11-02. Independent mathematical validation of the study's selected **N=2048 fresh paired-history design**. No new native trajectories or empirical information estimates were run. This is a proof and implementation specification; study must still freeze the complete experimental protocol.

**Verdict: the proposed certificate is sound.** It gives simultaneous population statements for both fixed native models, under iid Gaussian-history rows, exact original initialization/dynamics, the balanced S prior, count `C in {0,...,5}`, and fixed decoder `D=1[C>=1]`. Pairing/correlation across models and conditions does not invalidate either union bound. N means2048 observations *per model and condition*, not4096 pooled conditions.

## 1. Four count histograms: simultaneous total-variation bounds

For one model-condition histogram, let p be its population pmf and p_hat its empirical pmf. For every nonempty proper subset A of the six-value alphabet, Hoeffding's one-sided inequality gives

`P(p_hat(A)-p(A)>delta) <= exp(-2*N*delta^2)`.

There are `2^6-2=62` such subsets, and

`TV(p_hat,p)=max_A[p_hat(A)-p(A)]`.

Both deviation directions are already covered by including complementary subsets; do not insert an additional factor2. A union bound across all62 subsets and all4 fixed model-condition histograms gives failure probability at most `4*62*exp(-2*N*delta^2)`. Therefore, with

`alpha_TV=.025`,

`delta=sqrt(ln(4*(2^6-2)/alpha_TV)/(2*N))`,

all four TV bounds hold jointly with probability at least.975. Only independence of the N rows within each histogram is used. [Hoeffding (1963), original article](https://www.csee.umbc.edu/~lomonaco/f08/643/hwk643/Hoeffding.pdf).

For six-category pmfs, if their exact TV distance is t, the entropy difference is at most `h2(t)+t*log2(5)`. One elementary proof couples the variables with mismatch probability t: conditional on mismatch and one variable, the other has at most5 possibilities, so each directional conditional entropy is at most `h2(t)+t*log2(5)`. Entropy difference is bounded by the corresponding conditional entropy. This is also the classical special case of the sharp entropy-continuity bound. [Audenaert (2007), theorem1 and classical reduction, equations6 and11–12](https://arxiv.org/html/quant-ph/0610146v1).

The modulus is increasing through t=5/6. Here delta is below5/6, so `f6(delta)=h2(delta)+delta*log2(5)` bounds each entropy error. Averaging the two conditions gives, for both native arms,

`H(C|S) >= H_hat(C|S)-f6(delta)`.

This finite-sample coverage statement does not require the plug-in entropy to be unbiased or the bootstrap to include unseen categories.

## 2. Four binary error cells: exact upper limits

Allocate `alpha_CP=.025` equally across nativeTAN/nativeLIF × miss/false-alarm cells: `a=alpha_CP/4=.00625`. For observed error count k out of N, use the exact one-sided binomial upper confidence limit

`q_U=BetaQuantile(1-a; k+1,N-k)` for `k<N`, and `q_U=1` for `k=N`.

Equivalently, q_U solves `P_{Binomial(N,q_U)}(K<=k)=a`; this binomial-tail inversion gives coverage at least `1-a`. When k=0 it is `q_U=1-a^(1/N)`. These are one-sided limits: do not halve a again. The four-cell union bound gives joint coverage at least.975. No independence between error cells is required. This is the exact binomial construction commonly called Clopper–Pearson; the inversion above is the self-contained justification.

For each native arm define `e_U=(miss_U+FPR_U)/2`. Set `b=h2(e_U)` when `e_U<=.5`; otherwise set `b=1`. Under the balanced prior, actual classification error e is at most e_U. With binary error variable `E=S xor D`, both

`H(S|D)=H(E|D)<=H(E)=h2(e)<=b`,

and

`H(D|S)=H(E|S)<=H(E)=h2(e)<=b`.

The second inequality is necessary for the nuisance-detail result; bounding only `H(S|D)` would leave that proof incomplete.

## 3. Joint implications, with at least95% coverage

Intersect the TV and binomial events. Their total failure probability is at most`.025+.025=.05`, even though they use the same histories. All following statements then hold simultaneously for both native arms:

1. **Signal information in the fixed detector:** `I(S;D)>=1-b`.
2. **Additional signal information discarded when count is compressed to D:** `0<=DeltaI_S=I(S;C)-I(S;D)=H(S|D)-H(S|C)<=b`.
3. **Conditional Gaussian-path count detail discarded by that compression:** `DeltaI_N=I(N;C|S)-I(N;D|S)=H(C|S)-H(D|S)=H(C|D,S)`, hence

   `DeltaI_N >= max(0,H_hat(C|S)-f6(delta)-b)`.

The native deterministic-map identities justify replacing the two conditional mutual informations by entropies. The nonnegative truncation in the lower bound follows from the exact identity `DeltaI_N=H(C|D,S)>=0`; it is not clipping an empirical point estimate. If this lower bound is positive, it certifies population count detail about the nuisance path that the fixed binary readout discards, while the second statement bounds the accompanying task-information loss. It does not label individual spikes as noise caused, establish a native denoising mechanism, or assert that all such detail is harmful for other tasks.

## 4. Fixed-design numerical constants and implementation guards

With N2048, the formulas alone give `delta=.047398916395406346` and `f6(delta)=.3853001400875786 bits`. In the *hypothetical* all-zero-error case, each class upper error limit is`.00247504422918976` and b is`.024996061169898776 bits`. These are pre-execution algebraic checks using no empirical outcomes; they are not predictions that errors will be zero.

Use natural log in the Hoeffding radius and base2 logs in all entropy quantities. Preserve all six count categories. Verify exact k=0/k=N branches and the binomial-tail inversion for interior k. Report all four error counts, all four upper limits, the common delta/f6, both native b values, and all three certificates. A failed or zero lower bound remains in the table. Freeze N, models, alpha allocation, observation, and decoder before data; do not stop early, replace the decoder, or add selected models while reusing this coverage claim.

The certificate applies only to native deterministic TAN/LIF. Do not apply binomial CP directly to fractionally integrated thinning probabilities. All-arm analytic-thinning information quantities and whole-history bootstrap contrasts may still be reported descriptively, with their separate interpretation. No population energy, timing-information, new-information-creation, or perfect-performance claim follows.

Primary-source provenance: the original Hoeffding PDF was retrieved, but its browser text extraction was unavailable; the standard one-sided Bernoulli bound is stated explicitly above. Audenaert's original preprint and its classical statement were opened and inspected. The original1934 Clopper–Pearson publisher page was not retrievable through the available browser; no text from it is relied on. Its required coverage property is established directly by the stated binomial-tail inversion. This provenance clarification changes no formula or experimental choice.
