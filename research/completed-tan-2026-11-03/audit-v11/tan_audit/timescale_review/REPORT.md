# Supplied timescale study: independent audit and replay

The exact supplied executable completed once with exit code 0 under
`PYTHONDONTWRITEBYTECODE=1`. All 15 cells ran, with 4,096 paired histories per
cell and no parameter search. The executable, its protocol, transmitted text,
and frozen model were not edited. Fresh replay outputs are next to that source
in `../prior_sources/timescale-heldout/`; log: `rerun.log`.

The central supplied results reproduce: sensor-only Ecomp is 0.27080341003844277
for baseline, 0.00809317129694613 for gate-only slowing, and
0.10261778243177685 for uniform slowing. Gate-only sensor coverage and local
joint success are both 4074/4096 = 99.462890625%. Its membrane-process and
combined-noise Ecomp are 0.04810462207358091 and 0.04951443129138196.
This verifies the supplied source's conditional experiment, not all eight prior
stages or a historical archive byte match.

## Results and controls

All 15 cells, including both-role unconditional RMSE/MAE, Ecomp, coverage,
local joint success, and rejection counts, appear in `all15cells.md`. Every
abstention is OUT_OF_DOMAIN; there are no physical branch departures,
numerical boundary snaps, ambiguous roots, negative discriminants, or residual
rejections in this run. The unconditional zero predictor Ecomp is
0.33272635072886175 and local joint success is 1.025390625%.

The sensor-only paired-history bootstrap uses 2,000 shared bootstrap index
draws across the fixed contrasts. Gate-only minus baseline mean Ecomp is
-0.26271023874149657 with percentile 95% CI
[-0.2677712271257672, -0.25726232482317785]. Uniform minus baseline is
-0.1681856276066659 with CI [-0.17306648917465037, -0.16333007808095057].
These are pointwise intervals for the fixed contrasts, not simultaneous
familywise intervals, and apply only to the declared distribution/noise model.

The target CSV hash is
`4944142f2f3f2fb25b488c21a8c7cfc9224a85021703a56b84aeea6d27bc5c30`.
An independent audit verified these targets against RandomState(20261105),
checked all 61,440 prediction rows, recomputed every unconditional metric and
status count, and confirmed that all rejected predictions are exactly (0,0).
See `independent_check.py` and `independent_checks.json`.

## Whole-box and global-injectivity proof

The endpoint map on D=[-.5,.5]^2 is `(u(v1,v2),A(v1,v2))`, with u quadratic
and A affine. I independently reconstructed its coefficients from exact
Fraction geometric-sum block transitions and polynomial interpolation, without
importing the supplied executable. All coefficients agree with its stored
float coefficients. Both gate slopes a1,a2 are strictly positive.

Since the derivative of u is affine and the gradient of A is constant,
`det J = a2*u_v1 - a1*u_v2` is affine over the rectangle. Its maximum and
minimum occur at corners. The independently computed exact rational signs
are strictly negative at every corner in every arm. Approximate ranges are:

| Arm | Minimum det J | Maximum det J |
| --- | ---: | ---: |
| baseline | -4.110453430218917e-6 | -3.885983582347569e-6 |
| gate_only | -2.891480878547956e-5 | -2.499474578308363e-5 |
| uniform | -6.000608875534205e-6 | -5.628691095738086e-6 |

For fixed A, its intersection with the convex box is an empty set, point, or
connected segment. Parameterize a nontrivial segment by v1, giving
`v2=(A-a0-a1*v1)/a2`. Then `du/dv1 = det J/a2 < 0` throughout the segment.
Consequently u cannot repeat on that fiber, and distinct fibers have distinct A.
This proves global injectivity on this entire fixed-key/fixed-timing box, rather
than inferring it from local rank alone.

The physical branch proof also holds on the whole box. With fixed keys (.8,1.8),
content currents are respectively [.3,1.3] and [1.3,2.3], while gate drives are
[.25,.35] and [.65,.75]. They are positive and increase with their value. Since
all recurrence coefficients are positive, A and u increase coordinatewise with
values before any reset/clipping. Thus the (.5,.5) corner dominates the whole
encoding trajectory. An independent exact block bound, using monotonic A
inside each constant-input block and convex-combination membrane updates,
certifies no threshold crossing or gate clipping throughout the box:

| Arm | Conservative encoding u upper bound | A upper bound | Noiseless query u upper bound |
| --- | ---: | ---: | ---: |
| baseline | .8431921595 | .3666052867 | .8187801761 |
| gate_only | .1417475708 | .0616293786 | .8057653432 |
| uniform | .1417475708 | .0616293786 | .8057653432 |

These bounds are conservative, not observed trajectory peaks. For query,
A decreases and membrane drive is `.8+.1*A`; its convex recurrence gives the
listed bound. All are below threshold1/gate ceiling5. Noise is unbounded, so
the whole-box branch argument applies to noiseless evolution; the finite noisy
draws separately pass the code's branch checks. No all-noise guarantee follows.

## Decoder, noise, and information audit

No target-based root selection or fitted decoder was found. The stable
quadratic solver handles negative discriminants by abstaining, permits only
1e-10 boundary-roundoff snaps, checks both coordinates and forward residual,
and uses zero fallback for invalid/ambiguous cases. All failures remain in
unconditional metrics. The noiseless 9-point grid check plus held-out vector
recurrence and frozen scalar checks passed. The exact global proof justifies
unique in-box solutions when the recovered state lies in the image.

The decoder is privileged by known keys, event positions, exact time constants,
exact content-gate law, and domain. It recovers two analog state coordinates
from membrane samples at time0 and10ms of a constant `.1` content probe plus
q=.8 current, then applies the exact inverse. The nonzero observation matrix
determinant makes this possible. There is no variable-key addressing or
role-selective query experiment here. The added probe and two analog samples
are new readout resources relative to original P1B. Original tolerance .5 would
be trivial for the narrow prior; the reported .05 local criterion is distinct.

All three arms use the modified eta=.1 content-driven gate. The arm called
`baseline` means original time constants (10,15), not original M1 and not a
matched offset-only gate control. Gate-only is (10,150), uniform is (100,150).
There is no eta=0 comparison in this particular executable.

Noise exists only during the 100-step query or at its two sensor samples:
there is no encoding noise, initial-state uncertainty, key/timing uncertainty,
or model misspecification. State innovations are added after deterministic
drift; equal raw 1e-6 innovations across time constants are not equal physical
current noise. Sharing target draws and disturbance arrays across arms is
appropriate for the paired comparison. A third generated sensor column is
unused, preserving the declared earlier drawing order without leakage.

## Resource limitations and missing source scope

The resource computation is useful but incomplete relative to the supplied
protocol: encoding endpoint_u/A ranges are present, while query endpoint_u/A
ranges are absent. Integrals are computed per history but saved only as
mean/min/max summaries; prediction ledgers do not retain per-history resource
traces or integrals. Query peak gate/candidate logs are before innovations;
they must not be called total noisy-state peaks. Here zero branch departures
avoid an additional concern about resource logs after invalid trajectories.

The Isyn proxy includes A*content, excluding the .8 query current and gate
forcing. It is not total input, energy, ATP, or an equal-cost normalization.
Changing tau modifies both leak and h/tau forcing gain. Mean encoding gate
integral changes from 7.3683 to 2.0702, absolute synaptic-current integral from
4.0728 to .6986, and squared membrane integral from .4474 to .01399 (gate-only)
or .0005069 (uniform). Same coordinate count does not mean matched retention,
charging, precision, noise physics, or biological cost.

The directly supplied material does not include the referenced reviewer files
`timescale-retention-protocol.md` or `nonlinear-heldout-protocol.md`, nor the
nonlinear-heldout executable. Thus their complete requirements and previous
results cannot be audited. The earlier nonlinear protocol mentions an unbounded
frozen-linear-inverse diagnostic; that diagnostic is absent from this executable.
The timescale protocol does not clearly require repeating it, so this is a
scope omission, not an established violation. Full prior archive bytes remain
outside this source audit.

No installed dependency is missing: stdlib, NumPy2.3.5, SciPy1.17.0, and the
hash-verified frozen model suffice. No installation, publication, security
change, or additional timescale/domain/noise search was performed.

Reproduction finding: the fixed known-key conditional-retrieval improvement is
supported; claims of native/general addressing, equal physical noise/current
cost, or complete resource accounting are not supported by this experiment.

## Addendum — 2026-11-02: reviewer protocols received and endpoint gap filled

The reviewer protocols are now available at
`../prior_sources/tan-research/timescale-retention-protocol.md` and
`../prior_sources/tan-research/nonlinear-heldout-protocol.md`, and were read for
this addendum. The missing-reviewer-source statements above record the earlier
audit state and are superseded here. The nonlinear executables have also been
supplied and assigned to a separate source auditor; their results are not
claimed as reviewed by this timescale note. No complete archive-byte
verification is asserted.

The full timescale reviewer protocol confirms the narrow interpretation and
all fixed choices used by the replay: three specified arms, both leak and
normalized forcing changed together, zero starting states, m100 only, unchanged
eta/keys/positions/query/probe, the five query-noise cases, known keys/timing,
exact inverse, local .05 criterion, and no fitting or extra sweep. Its explicit
monotone-upper-corner and connected-affine-fiber arguments agree with the
independent exact proof above. It does not require a new variable-key task or
a linearized-inverse comparison within this timescale experiment; the latter
is mandatory in the distinct preceding nonlinear protocol. Therefore no
timescale violation is established from its absence here.

Its observation matrix specification matches the executable and independent
checks: rows `[1,0]` and `[a^100,d100]`, known offset `.8*(1-a^100)`, with
d100=.04387200979431237, .06101291203942187, .009208211958237782 for baseline,
gate-only, uniform respectively. Gate retention f1/f2 is
.0047425015911935165 for baseline and .5865418897514582 for either slow-gate
arm. The tabulated Jacobian singular values and condition numbers are at the
declared center linearization, while the independent determinant/fiber proof
establishes injectivity throughout the box. Improved retention and changed
query transduction remain separate mechanisms, as the protocol requires.

The reviewer resource section explicitly permits actual-current/time
**summaries** rather than complete per-history traces. Accordingly, aggregate
integrals are compliant; the lack of per-history resource ledgers is a
reanalysis limitation, not a protocol failure. Query endpoint ranges were
explicitly requested and omitted from the supplied output. This audit now
supplies them separately in `query_endpoint_ranges.json`, generated by
`query_endpoint_supplement.py` from the same 4,096 saved targets, fixed arms,
and frozen disturbance seed/draw order. Only deterministic state propagation
was recomputed: no decoder, fitting, risk metrics, parameter choices, or
completed 15-cell performance table was rerun. Source and primary results
remain unchanged.

The supplement records mean/min/max u and A immediately after the hundredth
query drift and state innovation for all 15 arm/noise cells. Sensor noise is
observation-only and is excluded from these physical endpoint ranges. All
recomputed encoding endpoint summaries match the original result to <1e-14,
and all query cases again have zero physical branch departures. Noiseless
query ranges are:

| Arm | u range | A range |
| --- | --- | --- |
| baseline | [.5365269455, .5607103321] | [.08336875636, .09617725222] |
| gate_only | [.5139404663, .5187934942] | [.04509025712, .05380437864] |
| uniform | [.07989633787, .08401770541] | [.04509025712, .05380437864] |

The earlier original-output endpoint omission is thus closed by a separate
audit artifact; it remains historically true that the original supplied code
did not save those fields. The remaining resource caveats still apply: Isyn is
the gated-content proxy, peaks in the source are pre-innovation, and neither
quantity represents total physical energy. Circuit reset/carryover was not
simulated; its energy and throughput costs are explicitly excluded, as required
by the reviewer protocol. No new missing-input or execution blocker remains
for this timescale-specific audit.
