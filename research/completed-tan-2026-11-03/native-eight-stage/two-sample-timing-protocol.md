# Two-sample timing: fixed physics, different readout separation

Proposed2026-11-02, before new model runs. Context: same content-coupled A encoder and known fixed keys/timing from prior experiment. This is a readout-conditioning test, not proof of general content retrieval or new memory capacity.

## Exact discrete observation matrix

During a subthreshold query, retain original A decay (zero novelty), queryq=.8 and constant added content probeIprobe=.1 throughout the entire100-step/10ms interval. Let a=.99, b=149/150, κ=(h/τm)gA=.01, k=κIprobe=.001. No spikes/reset/clipping may occur on the analytic branch. Then:

`A_m=b^m A0`,
`u_m=a^m u0+d_m A0+q(1−a^m)`,
`d_m=k Σ_{j=0}^{m−1} a^(m−1−j)b^j = k(b^m−a^m)/(b−a)`.

After subtracting the known affine query contribution, two observations(u0,u_m) give O_m=[[1,0],[a^m,d_m]]. detO_m=d_m. Ifa=b, use limitd_m=k*m*a^(m−1), not division by0.

Predeclare onlym=1 versusm=100. m100 is the original analog deadline; it is not an eligible spike-codebin (which excludes the last-step spike), and no spike-code claim is made. KeepIprobe amplitude.1 and duration100steps identical in both arms, even when the second sample ism1. Same100-step path, hence identical drive, physical trajectory and probe-energy history. Sample count remains2. Later information availability/decision latency differs and must be reported (0.1ms vs10ms).

Numerical constants (analytic arithmetic, not a simulation):

| m | a^m | d_m | A sensor-noise SD amplification |
|---|---|---|---|
|1|.99|.001|1407.1602609511|
|100|.3660323412732292|.04387200979431237|24.2725332767|

The last column=sqrt(1+a^(2m))/d_m for independent equal-variance sensor errors at the two samples: ≈58× less amplification at the deadline. This conditions reconstruction of A0; total value reconstruction also contains encoder conditioning, J. Report singular values ofO_m, J andO_mJ separately with declared units, not justdet or a raw condition number.

With the fixed parameters, d_m grows throughm100 (the continuous-m maximum ofd occurs aroundm121.125); the comparison follows the analytic design rather than optimizing on held-out outcomes. Do not add furtherm choices after seeing results.

## Safe branch condition

A0≥0 decays monotonically. If actual endpointu0<1 and actualA0<2, thenq+Iprobe*A_n<1, and the convex Euler recurrenceu_next=.99u+.01*(q+.1A) cannot cross threshold. Record actual margins over all test histories. A≤5 alone is insufficient: it permits total drive1.3. Do not claim global safety from gate clipping. For stochastic paths, record any threshold/clipping branch departures separately; analytic subthreshold formulas do not cover them. Model's existing analog deadline uses pre-reset candidate voltage, but under the asserted no-spike branch candidate=post-stepu.

## Distinguish sensor noise from physical process noise

Let ε0,εm be sensor errors. Let ξu,l and ξA,l be independent additive physical state innovations at each Euler update1..m, after deterministic drift. The intended observation target remains the pre-query endpoint(u0,A0), not the noisy future state. Without reset/clipping:

`u_m observed = a^m u0+d_m A0+knownquery + εm + Σ_{l=1}^m a^(m−l)ξu,l + Σ_{l=1}^m d_(m−l)ξA,l`.

The first sample isu0+ε0. Define reconstructedA=(y_m−a^m y0−knownquery)/d_m. Independent, zero-mean innovations yield:

`Var(Ahat−A0) = [σsensor²(1+a^(2m)) + σu²Σ_{j=0}^{m−1}a^(2j) + σA²Σ_{j=0}^{m−1}d_j²]/d_m²`.

In reconstructed state covariance, Cov(uhat−u0,Ahat−A0)=−a^m σsensor²/d_m and Var(uhat−u0)=σsensor². Initial *physical* state uncertainty is another model, not the same as sensor noise; do not substitute it silently.

| m | Σa^(2j) | Σd_j² | u-process SD amplification to A | A-process SD amplification to A |
|---|---|---|---|---|
|1|1|0|1000|0|
|100|43.51860930362|.101819237469791|about150.36|about7.27|

Thus wider separation strongly helps independent sensor noise and helps this specified membrane-process-noise case, but **does not universally improve every process-noise model**: gate-process noise is invisible at the first step but accumulates for later observations. Noise covariance must be propagated through J's inverse (when defined) for value error; matrix conditioning alone cannot choose the winner.

## Bounded executable comparison

Use exactly the previous basepoint and already-prespecified bounded value grid; do not add a new grid. Keep the content-coupled encoder and all parameters fixed. First validate noiseless O_m predictions against the actual revised query recurrence, including full100-step trajectories and no-spike margins.

For noise, predeclare a small finite set before execution (recommended sensor-onlyσsensor=1e−6; membrane-process-onlyσu=1e−6; their combination; gate-process-onlyσA=1e−6 as explicit countervailing-noise control). Same SD units per Euler update; do not compare these as equal physical energies. Use fixed seed and an explicitly stated number of paired paths; no tuning on results.

Generate each100-step physical innovation path once; observe at1 and100 from the same path. Reuse the sameε0 across designs, with independent equal-SD errors for sample1 versus sample100. Do NOT restart dynamics after sample1, resample separate endpoints, or give one arm a longer probe. Physical noise must affect recurrent states, not be added independently to snapshots and mislabeled process noise. Compare empirical state-error covariance against exact predicted covariance; then propagate through the frozen local diagnostic inverse. Report noise-free nonlinear decoder bias separately from noise-induced variation.

OptionalMonteCarlo is a check on these exact covariance expressions, not a training or performance search. Stop once finite-difference/model consistency and covariance agreement or a diagnosed discrepancy establish whether the fixed separation improves this specified observation model. No general biological/plasticity/network necessity claim follows.
