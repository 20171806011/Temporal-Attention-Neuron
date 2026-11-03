# Finite-tolerance residual writer: candidate-specific resource bounds

Prepared before any new IID execution. The exact-flow obstruction from the preceding report is not a finite-error impossibility theorem. The following analysis instead asks whether a particular division-free, finite-time residual writer can meet the original 0.5 operand tolerance under the registered 10 ms event and 0.1 ms Euler update schedule.

## Candidate and its actual state

Retain three analog coordinates `(K,V,S)`. During the first event, K and V track the current key/value with a declared first-order charging time constant, while S stays zero. After that event K and V are held. During the second event, define combinational signals `d=k2-K`, `e=v2-V` and use

`dS/dt = gamma*d*(e-d*S)`.

After encoding, query only q through `V+S*(q-K)`. No second event record survives the write; the first anchor does survive, so this is a three-coordinate anchored representation, not the preceding two-coordinate overwrite. It does not hide an ideal division in an update helper: inversion emerges only through settling of the differential equation, and its finite settling cost is the question.

This is scalar residual/error-correction learning: the write follows the negative gradient of `(e-d*S)^2/2`. Input-dependent delta-rule updates are established associative-memory mechanisms, but that literature does not establish this circuit's neuronal equivalence or timing. [Schlag, Irie and Schmidhuber, ICML 2021](https://proceedings.mlr.press/v139/schlag21a.html). The present equations and bounds are an explicitly proposed, task-specific specialization, not a claim that this paper implemented the same anchored writer.

Required resources are three persistent analog coordinates, a finite phase/controller, raw separate key/value input ports, external event onset/offset/reset/clock, two held anchor coordinates, two products and three subtractions in the residual path, gain gamma, and query subtraction/product/addition. A physical realization must count any temporary registers used by these arithmetic operations. Calling the residual wires combinational does not prove a zero-state analog multiplier. No ideal division operation is called during the proposed write, but the current input remains externally presented for the whole event. The native TAN r coordinate is a bounded refractory timer, not an available signed slope; merely retaining three names does not match its state range or update law. This candidate is enriched relative to native TAN.

## Favorable ideal-anchor continuous bound

First grant exact K=k1, V=v1, perfect anchor holding and zero slope leak. These assumptions can only remove this candidate's first-write errors. With S(0)=0, true gap d and value difference e,

`S(T)=(e/d)*(1-exp(-gamma*d^2*T))`,

and the first query is exact, while the second-target residual is

`R(T)=e*exp(-gamma*d^2*T)`.

The actual generator has key-gap magnitude in `[delta,D]=[0.05,2.2]` and value-difference magnitude at most 6, interpreted at support closure. Thus a uniform error strictly below 0.5 requires

`gamma > log(12)/(10*0.05^2) = 99.39626599152`

in this declared key-coordinate system, for T=10 ms. The equality yields the non-strict closure bound 0.5. Units are inverse ms times inverse squared key units. This is not comparable to a bare neuronal leak coefficient without specifying input/state scales.

## Registered Euler schedule: an exact minimax certificate

The literal Euler update is

`S_(j+1)=S_j+h*gamma*d*(e-d*S_j)`.

Let alpha=h*gamma. Its residual satisfies exactly

`R_(j+1)=(1-alpha*d^2)*R_j`.

Stability across all legal gaps requires `0<alpha<2/D^2`, or `gamma<4.13223140496` at h=0.1 ms. That cannot satisfy the preceding continuous gain requirement at the same discretization. Increasing continuous gain without changing the numerical/physical update bandwidth is not a valid solution.

More strongly, among every constant positive alpha, the smallest worst-case contraction over the gap interval is obtained by equalizing the two endpoint magnitudes:

`alpha_star=2/(D^2+delta^2)=800/1937`,
`gamma_star=alpha_star/0.1=8000/1937`,
`rho_star=(D^2-delta^2)/(D^2+delta^2)=1935/1937`.

The maximum of the absolute affine function in d^2 occurs at its endpoints, and balancing them proves the minimax statement; no numerical parameter sweep is involved. Every constant-gain Euler writer in this candidate family has worst closure residual at least

`6*(1935/1937)^100 = 5.41110690216`

after the registered 100 updates. This is a finite-tolerance lower bound for this family, even with its favorable exact first anchor. It is not an all-neuron or all-adaptive-update lower bound. Because the bound is far above 0.5, interior legal key/value pairs arbitrarily near the closure also fail; the result does not depend on sampling an excluded upper endpoint.

Exactly 2,406 updates are sufficient and necessary for this constant-gain minimax guarantee at 0.5: exact rational arithmetic verifies `6*rho^2405 > 0.5` and `6*rho^2406 < 0.5`. At h=0.1 ms that is 240.6 ms of second-event writing, exceeding the event by a factor 24.06. Keeping T=10 ms would require at least 2,406 updates, with h<=10/2406≈0.004156 ms and a correspondingly changed gain. This is a bandwidth/time tradeoff, not a free ODE integrator detail.

The optimal large-gap branch alternates residual sign. For the closure pair |d|=2.2, |e|=6, the accumulated absolute slope movement is `5280*(1-rho^N)`. It is a state-variation proxy, not energy. Monotonic convergence at every gap imposes alpha<=1/D^2 and is slower. A scheme that wants to avoid the high-gain oscillatory branch needs its own mechanism and cost accounting.

Globally rescaling key coordinates does not remove the ratio `D^2/delta^2=1936`; compensating the gain leaves the actual rates unchanged. Adaptive normalization proportional to 1/d^2 would change the condition ratio but introduces an explicit inverse-square operation or its own approximation circuit. A different sign/rectifier update would likewise be a different declared mechanism, not a coordinate trick. None is silently granted here.

## First-event finite charging is another counted resource

Let first-event Euler tracking start at zero and use a common charging time constant tau_w. After 100 updates, with `lambda=(1-h/tau_w)^100`, the retained anchor is

`K=(1-lambda)k1`, `V=(1-lambda)v1`.

At native tau_w=10 ms, lambda=0.99^100≈0.3660323413. The effective second gap is `d_eff=k2-(1-lambda)k1`. It can equal zero for valid original keys: choose k1=2 and k2=2(1-lambda). The true separation is about 0.732, well above 0.05. Then the residual slope update is identically zero for every gain and duration, so values near (-3,+3) cannot both be retrieved. This is specific to this uncorrected tracking representation, not a general failure of leaky memory.

If `0<=lambda<0.02`, every effective gap remains nonzero. With infinite second-event settling the first-query error is exactly

`lambda*(k1*v2-k2*v1)/(k2-(1-lambda)k1)`.

Maximizing its magnitude over the support closure gives

`E_first,infinity <= 14.85*lambda/(0.05-2.5*lambda)`.

The maximum occurs at descending nearest keys k1=2.5, k2=2.45, opposite extreme values. For ascending keys the denominator is larger; for a descending pair with separation g, the ratio `3*(2*k1-g)/(g-lambda*k1)` increases with k1 and decreases with g. This proves the stated maximum.

This converged first-query guarantee is below 0.5 only if `lambda<1/644≈0.001552795`. In 100 Euler charging steps this requires `tau_w<1.59668393007 ms`; continuous tracking requires `tau_w<10/log(644)≈1.54614499275 ms`. The first and second target errors still need a joint finite-time analysis; making this bound small alone does not solve the slope-settling bound. The 0.5 allocation is not enough to absorb arbitrary additional rounding/noise/readout errors.

At finite second-write time with stable monotonic residual contraction, the first-query output is a convex interpolation between its uncorrected anchor value and its converged value. Its error is bounded by the larger of `3*lambda` and the converged bound. The exact finite-time second residual is `e_eff*exp(-gamma*d_eff^2*T)` for the ODE, or `e_eff*(1-alpha*d_eff^2)^N` for the recurrence. Oscillatory Euler updates require their actual factor rather than the convexity argument.

## Adding ordinary slope leak

For an ideal anchor and slope leak ell>=0,

`dS/dt=gamma*d*(e-d*S)-ell*S`.

Its second-target residual is

`e*[ell/(gamma*d^2+ell) + gamma*d^2/(gamma*d^2+ell)*exp(-(gamma*d^2+ell)*T)]`.

Leak makes settling faster but leaves bias. Even infinite time requires `gamma*delta^2>11*ell` for the uniform 0.5 closure tolerance. At a 10 ms slope leak (ell=0.1/ms), this necessary condition is gamma>440 in the declared units, before finite-time and numerical stability costs. At the best nonleaky registered-step gamma≈4.13, the same bias constraint requires a slope-retention time greater than about 1,065 ms. These comparisons concern this specific state/gain convention, not equal biological currents or energy.

## Appropriate decisive test and scope

A bounded test can freeze the analytically minimax gain, reproduce the exact-anchor 100-step residual on legal minimum/maximum-gap cases, and verify the native-charge effective-gap collision. This is more informative than choosing a new architecture or tuning gains until some IID examples pass. An exact-anchor arm is deliberately privileged and can establish a negative feasibility bound; it must not be described as a faithful finite-time first write. If an enriched high-bandwidth arm is later requested, its extra update count/gain and imperfect first charging must be declared before execution.

This analysis establishes a specific missing operation/resource: the residual rule needs either substantially more settling bandwidth/time, a declared input-dependent normalization mechanism, or a different write law, alongside accurate anchor capture and holding. It does not extend the earlier exact-flow theorem to approximate computation, does not exclude hybrid mechanisms, and supplies no ATP/area estimate or N=3 claim.

## Selected implementation contract, appended before outcome generation

The root researcher selected two fixed arms that expose the bandwidth tradeoff rather than a gain search. Both use actual finite first-event Euler charging, with h=0.1 ms, tau_c=10 ms and exactly 100 updates:

`K_(j+1)=0.99*K_j+0.01*G*k1`,
`V_(j+1)=0.99*V_j+0.01*G*v1`,
`G=1/(1-0.99^100)=1.5773675300856045`.

K and V start at zero. This shared, fixed input-gain calibration makes the ideal arithmetic endpoint exactly (k1,v1), without an offset assignment, target lookup or event-specific inversion. It is a counted gain and duration/model calibration assumption, not an intrinsic property of a 10 ms passive tracker. The commanded equilibrium magnitudes are up to G*2.5≈3.9434 for key and G*3≈4.7321 for value; the actual finite-episode states approach only the input values. K,V are held by gated zero derivatives after capture; S has zero derivative before its write and after encoding. These holds remove autonomous leakage by assumption and require control/gating resources.

The only primary arms are:

1. **A100:** calibrated capture; h_s=0.1 ms, N_s=100 and gamma_star=8000/1937 for the 10 ms second event. This is the analytically best constant Euler gain over the whole key-gap interval, including its oscillatory large-gap branch.
2. **FAST10:** identical capture and physical 10 ms second event, but gamma_fast=`log(24)/(10*0.05^2)`=127.122153213918, N_fast=`ceil(10*gamma_fast*2.2^2)`=6,153, and h_fast=10/6,153≈0.00162522346823 ms. This enforces monotone Euler contraction for every true legal gap. It deliberately changes update bandwidth and continuous feedback gain.

For FAST10, the slowest continuous rate is gamma_fast*0.05^2≈0.317805383/ms, giving continuous worst residual 6/24=0.25. Its Euler factor at that gap is approximately 0.9994834952 and 6 times that factor to the power 6,153 is about 0.2497948286. The largest rate is gamma_fast*2.2^2≈615.271222/ms, corresponding to a fastest relaxation time about 1.6253 microseconds. The nominal largest-gap Euler factor is positive, about 0.0000467714, so rounding of d at the 1e-12 scale cannot change the contraction sign. The second-event clock uses 61.53 times as many updates as the registered 100, while the continuous gain changes by a factor about 30.78 relative to A100. These are distinct resource changes; a numerical integrator step is not automatically a biological clock, but the registered discrete model has a fixed update schedule and this simulation explicitly changes it.

The candidate itself must execute every residual update from its current states. The closed-form ODE solution and ideal interpolation slope are validation oracles only and must never substitute for the iterative writer. Current external inputs are still needed during the whole 10 ms second event. No key/value event cache or evaluator trace may feed the query decoder.

### State bounds for these two arms

Exact calibrated first capture keeps K in [0,2.5] and V in [-3,3] throughout charging. With an exact anchor, all intermediate slope states in FAST10 satisfy |S|<=|e/d|<=120 because the recursion is monotone toward its equilibrium. For A100, where r=1-alpha*d^2 is nonnegative, the same bound holds. Where r<0, the largest factor |1-r^n| occurs at the first odd step and equals alpha*d^2. Hence |S|<=6*alpha*|d|<=6*alpha*2.2≈5.452 on its oscillatory branch. The global bound is therefore 120 for either arm, inside the fixed ±128 guard.

Capture roundoff must be measured rather than silently corrected. A predeclared |K-k1|,|V-v1|<=1e-12 check permits replacing 6 and 0.05 in the preceding equilibrium bound by 6+1e-12 and 0.05-1e-12; the result stays far below128. Guard comparisons for K/V at half-open support boundaries should report tiny floating excursions separately, with no broad clipping or post-event exact assignment. A failed capture or range certificate must stop interpretation rather than trigger a gain adjustment.

### Calibration sensitivity is a scientific limitation

G is calibrated to the exact duration, update rule and charging time constant. Keeping G fixed but capturing for 99 or 101 steps changes the endpoint gain to G*(1-0.99^n), producing relative anchor errors approximately -0.0058319953 and +0.0057736753. At minimum-spacing high-key pairs, converged first-target error can then exceed2.3 even though the intended input tolerance is0.5. This is an analytic sensitivity example, not an additional empirical timing sweep or evidence of its probability under the original fixed schedule.

For a common multiplicative gain error, the earlier converged first-target certificate permits relative undercharge below1/644 and overcharge below1/643 to guarantee0.5 over the support closure. Those bounds leave no budget for arbitrary other errors. The selected exact-duration test does not establish robustness to uncertain timing, analog calibration drift or intrinsic retention noise.

If independent raw input noise of variance sigma_I^2 is added before the fixed gain on each first-stage update, its endpoint variance is `(0.01*G)^2*sigma_I^2*sum_(j=0)^99 0.99^(2j)`; the standard-deviation multiplier is about0.1040568. Direct per-update state innovations instead have multiplier about6.59686. These different noise conventions must not be equated. They are formulas for disclosure only; no extra noise arm is added to the selected test.

S has an input-dependent contraction gamma*d^2 during writing but **no additional autonomous slope leak** in these arms. The previous autonomous-leak formulas remain an explicit limitation. Neither finite feedback gain, total slope variation, register occupancy nor input-drive amplitude is an ATP or energy estimate. A hardware implementation would additionally need explicit multipliers, temporary arithmetic storage, phase logic and a high-bandwidth feedback path. This is an enriched three-state algorithmic mechanism test, not a faithful reuse of the native TAN (u,A,r) update law.
