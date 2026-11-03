# Exact nonlinear fixed-domain decoder: preregistered held-out test

Proposed2026-11-02 before held-out execution. Continues fixed content-A armη=.1, keys(.8,1.8), positions(0,4),100ms encoding, queryq=.8 withIprobe=.1 for100steps, two analog samples(u0,u_m), m∈{1,100}. No new state, no training, no current/duration changes. Known fixed keys/timing and model equations are decoder assumptions, not inferred capabilities. This is conditional two-value decoding, not variable-key relational memory.

## 1. Exact subthreshold endpoint polynomial

A is affine in values and u is quadratic because pre-step A multiplies(k+v). With v=(x,y):

`A_T=d+f1*x+f2*y`,
`u_T=b+c1*x+c2*y+q11*x²+q12*x*y+q22*y²`.

Coefficients from an independent deterministic polynomial recurrence (no data fitting):

- d=.1752500455240791
- f1=.00011849111241245325
- f2=.024984938883834424
- b=.09044539592763301
- c1=.0001392666246935837
- c2=.06310841767423814
- q11=.0000023028240410664316
- q12=.00007250882963196175
- q22=.007146692180152506

Validator should regenerate full-precision coefficients from equations, not truncate this printed table. Initialize A affine coefficients0 and u degree≤2 coefficients0. At each step first updateu=.99u+.01*A*(k+active_value) with oldA; then A=(149/150)A+(1/150)*(novelty+.1*(active_value+3)) for active events, only decay for blanks. This is exact polynomial algebra on the declared branch. Validate against actual encoding code on prior fixed grid plus held-out values, max discrepancies logged; do not fit coefficients from targets.

## 2. Fix prior domain before testing: D=[−.5,.5]²

This is deliberately narrower than original[-3,3]. Both event currents are positive throughoutD: first .3..1.3, second1.3..2.3. A grows monotonically with each value, so nonnegative subthreshold recurrence is coordinatewise monotone. Maximum candidate voltage overD occurs at(.5,.5). Verify this corner's complete trajectory has strict threshold/clipping margin (previous grid already did), establishing branch validity for allD. Do not use the polynomial on[-3,3] without separately proving its branch conditions; larger values can change spikes/resets.

### Global uniqueness proof onD

`detJ = f2*(c1+2q11*x+q12*y) − f1*(c2+q12*x+2q22*y)`.

This is affine in(x,y); its extrema onD occur at the four corners. Values range approximately−4.110453430219e−6 to−3.885983582347e−6, strictly negative. At each fixed A, the intersection of its straight-line fiber with convexD is connected, and u varies strictly monotonically along it since f2>0 and derivative alongfiber=detJ/f2. Hence the mapping(u,A) is globally injective **onD**, not just locally full rank. Any second quadratic root must lie outsideD. Rank/nonzero determinant alone would not generally prove global injectivity; the connected affine-fiber argument supplies the missing condition here.

No global fold is present insideD under this model. This is the result to test, not an assumption to retrofit if noisy inversions fail. Out-of-image noisy measurements are not evidence of an underlying noiseless fold.

## 3. Frozen exact inverse and rejection rules

First use exact timingO_m inverse from the preceding protocol to estimateuhat,Ahat from the two noisy analog samples, subtracting knownq(1−a^m). No target information enters this step.

Letρ=f1/f2, α=(Ahat−d)/f2 and substitute y=α−ρx. Solve:

- B2=q11−ρ*q12+ρ²*q22
- B1=c1−ρ*c2+α*q12−2αρ*q22
- B0=b+c2α+q22α²−uhat
- B2*x²+B1*x+B0=0.

Use numerically stable quadratic roots, with a separately tested linear fallback ifB2=0 exactly; compute discriminant and count real candidate roots. Do not choose a root using true targets, distance to truth, previous trial, or posthoc learned prior. Filter roots against **both** x,y∈D. Exactly one validroot =>decode. Zero validroots =>ABSTAIN_OUT_OF_DOMAIN (also distinguish NO_REAL_ROOT); multiple validroots =>AMBIGUOUS / testflag, not arbitrary tie-break. Record candidates and residuals for debugging.

Predeclare numerical boundary tolerance1e−10 in value units, solely to handle arithmetic roundoff. A root at most this distance outside a boundary may snap to that boundary, with a separately counted NUMERICAL_BOUNDARY_SNAP. Larger departures must abstain; no broad clipping/projecting noisy estimates intoD. Exactboundary test cases are deterministic sanity checks, reported separately from continuous held-out population results.

For any abstention/ambiguity, fixed unconditional fallback(xhat,yhat)=(0,0), chosen before seeing results. This is a declared decoder policy, not evidence of recovered values. Report both its numerical risk and validity-aware success.

## 4. Held-out design, fixed before execution

- Frozen seeds agreed with validation worker before execution:20261103 for targets and20261104 for noise;4096 independent pairs uniformD. Record seed, generator version, targetCSV/hash. These replace the note's earlier unexecuted seedproposal2026110218.
- No training set or fitted parameters. Previous9-point grid only validates implementation; new pairs assess bounded decoder risk.
- Reuse same physical encoding and100-step query trajectory for both sample schedules. All same resource/latency accounting from prior timing protocol.
- Noise cases unchanged from prior protocol: noiseless, independent sensorσs=1e−6 only, membraneprocessσu=1e−6 only, both, gateprocessσA=1e−6 only. Physical noise added at each update after drift, paired paths across schedules; ε0 shared, ε1/ε100 independent. Freeze seed streams and order. No sigma/domain/η tuning after results.
- Record branchdepartures, clipping, saturation. Separate those trials; do not silently apply subthreshold inverse or discard them from denominators. For unconditional reporting treat invalidphysicalbranch as abstention with fixedfallback0.
- Numerical choices fixed before execution: discriminant<0 meansNO_REAL_ROOT with no toleranceclamp; rootforward residual tolerance1e−10 in each endpointstate coordinate; domainboundary tolerance1e−10 valueunits as above. Verify polynomial forward against actual vectorized recurrence on all4096 and independently against scalar recurrence on first8 plus4domaincorners. Residual/branch failures are counted and diagnosed, not silently excluded.

## 5. Metrics and mandatory null baseline

**Critical:** Original delivery tolerance.5 is trivial on priorD: zero predicts every interior target within.5. Do not report that as successful P1-B retrieval. Primary metrics are unconditional per-role RMSE/MAE and unconditionalEcomp=max(|e1|,|e2|), plus abstention rate and decodingcoverage. Also report conditional valid-decode errors, clearly labeled; never substitute them for unconditional results.

If a delivery metric is wanted, predeclare **local tolerance.05**, and joint success requires exactlyone validroot and both errors<.05. Label this new local criterion, not original P1-B threshold. Include zero-predictor baseline on the identical held-out pairs, with no access to true values: population per-roleRMSE=1/sqrt12≈.288675; populationEcomp=1/3; local jointsuccess≈.01. Also include the frozen linearized inverse as an unbounded diagnostic comparator, but do not reinterpret its huge off-domain errors as behavioral predictions.

Report counts for validroot, norealroot, outofdomain, ambiguity, boundarysnap and physicalbranchdeparture; full-denominator metrics and coverage byschedule/noisecase. Every model/decoder must see the same targets and paired noise. Sampling uncertainty can be paired bootstrap over histories if desired, but keep decoder and abstention policy frozen.

## 6. Stopping and inference

Stop after solver/noiseless validation, full fixed4096-pair table and repeatability hashes. A reliable exact inverse insideD would show how strongly bounded nonlinear decoding changes the local-linear noise diagnostic. High abstention or no gain overzero is equally informative. Neither result establishes unknown-key addressing, memory beyond fixedtwo operands, capacity of every neuron, or need for a network. No public/remote changes.
