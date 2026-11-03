# Constructive two-register N=2 binding control

## Outcome and novelty

The preregistered constructive control succeeds. Across512 fresh original-generator N=2 histories, the ideal online writer answers both queries using **only two final real coefficients and the current query**, with mean max-operand error5.08e−16 in binary64. All18 adversarial support histories and the clean query/value-swap checks also succeed. No other event key, event-position list or key–value cache is supplied at query time.

This is classical two-point affine interpolation, not new mathematics. A search of218 inspected repository text sources found no implementation of this specific online carrier; prior M0 already solves the task using a two-key/two-value cache. The new control's purpose is therefore representation and write mechanism, not improved clean accuracy over M0. Different histories can share a line without causing a retrieval contradiction: if the same q is a valid stored key in both, the required value is the same line value. The scheme does not determine whether an arbitrary q was actually stored.

The finding narrows the interpretation of the preceding metadata experiment. That particular modified TAN recurrence needed extra binding information for the demonstrated inverse, but **post-encoding storage of every key as a separate record is not mathematically necessary for this ideal N=2 task**. The constructive writer is a different dynamical/computational mechanism, not original TAN or a demonstrated physical neuron.

## Exact mechanism and required resources

At the first event store(k1,v1). At the second event atomically replace them by

`a=(v2-v1)/(k2-k1)`, `b=v1-a*k1`.

For either stored key, `a*k_i+b=v_i` exactly. The query function receives `(a,b,q)` and performs one multiplication and one addition. The persistent payload is two reals; it is **not the entire resource budget**:

| Resource | Assumption in this control |
|---|---|
| Write input | Separate raw key and value ports; original TAN's summed content/novelty ports are not generally invertible |
| Payload | Two real registers, with phase-dependent meanings(k,v) then(a,b) |
| Controller | Empty/first/ready/error finite states, budgeted as2bits |
| Timing | External event-onset strobe, trial reset and slot/readout clock; random event positions do not supply ordinal phase for free |
| Second write |3subtractions,1division,1multiplication; old-register/current-input combinational operands and atomic overwrite |
| Query | Ideal continuous-valued multiplication/addition; no native spike quantizer or10ms physical circuit demonstrated |
| Retention | Perfect holding except explicitly labeled leak mechanisms |
| Precision/cost | Floating arithmetic and declared state-range assumptions; no ATP, transistor, noise-energy or neuron-equivalence measurement |

The raw first-event key/value fit inside the later coefficient ranges but use them inefficiently. No stage-dependent gain switching was silently added. Python's temporary arithmetic values represent disclosed combinational workspace, not a post-encoding hidden history cache. The controller's error state handles singular writes; it still fits in2bits. The clean task promises exactly two events and queries one of their keys. No N=3, unknown-key rejection or minimal-state lower bound is claimed.

The finite/singular-denominator and dynamic-range monitors also require comparisons/control logic; they are not included in the two-real payload count or claimed biologically free. Operation counts above describe the affine arithmetic, not a transistor-level implementation of all guards/controller logic.

## Tight support bounds and conditioning

The actual generator uses keys[.3,2.5), minimum separation.05, and independent values[-3,3). Let ordered keys x<y, d=y-x. Then

`|a| <= 6/.05 =120`,
`|b| <=3(x+y)/d <=3(2*2.5/.05-1)=297`.

These are tight support-closure suprema; excluded upper endpoints are not claimed to occur in an IID draw. The exact closure pair(2.45,2.5) with opposing values±3 attains them. Float fixtures were adjusted with nextafter and explicitly checked against the actual spacing guard: literal2.5−2.45 rounds below.05 and is not a valid float fixture.

With final coefficient errors bounded by eps_a,eps_b and query error eps_q, the exact error contains the cross term:

`e=q*Δa+Δb+a*Δq+Δa*Δq`,
`|e|<=2.5*eps_a+eps_b+(120+eps_a)*eps_q`.

For independent perturbations of fraction epsilon of the raw coefficient fullscales(120,297), the zero-query-noise bound is597*epsilon. At epsilon=.001 and eps_q=.0001, the global bound is.609012. Measured write-key errors eps_k=.0001 instead give slope bound6/(.05−2eps_k), and queried error at most6(eps_q+eps_k)/(.05−2eps_k)=.02409638555 for exact written values. Every tested raw-output bound passed before any device-range fallback.

The597 factor is coordinate-dependent, not an intrinsic price of two-dimensional memory. A fixed analytic change to centered intercept d=b+1.4a gives |d|<=129 and a corresponding261*epsilon bound for normalized state perturbation. This was derived only; no extra representation arm, tuning or sweep was run. Equal fractional fullscale is an explicit engineering assumption, not equal raw noise, information capacity, precision cost or biological noise energy across interpolation, M0 or TAN coordinates.

## Fixed-budget matched experiment

Protocol SHA256 `d1a76b593ef6887a4750847f144a8006cb671e18cbb35a0bada94af0e999b9f6` was frozen before outcome computation. A resource clarification about separate raw input ports was added before execution, not after outcomes. Input seed2026110213 is one continuous RandomState; bounded-noise seed2026110214. Inputs and noise were saved/hashed before model metrics.

The512 original histories plus18 constructed adversarial histories enter ten interpolation scenarios, seven applicable M0 scenarios and two clean modified-carrier controls. Another1,024 clean query/value-swap variants enter allfour clean controls. This gives14,166 paired prediction records and28,332 role predictions; the swaps and adversarial cases are not additional independent population draws. No training, hyperparameter search, exclusions of generated histories or primary rerun occurred.

| IID condition, n512 | Interpolation Ecomp | Valid joint±.5 | M0 Ecomp where matched |
|---|---:|---:|---:|
| Clean binary64 |5.08e−16|100%|0|
| Final-state noise, .001 fractional fullscale |.192569732|99.21875%|.001985934|
| Query noise±.0001 key units |.000326557|100%|0|
| Final-state + query noise |.192596524|99.21875%|.001985934|
| Write-key + query noise, each±.0001 |.000458569|100%|0|
| Binary32 inputs/write/readout |2.93049e−7|100%|4.93828e−8|
| Perfect holding for additional10ms |5.08e−16|100%|0|
| Uniform post-encoding leak, tau150,10ms |.128899649|100%|Not equated across coordinates|
| Differential post-encoding leak, tau_a150/tau_b100,10ms |.302322598|84.96094%|Not equated across coordinates|
| Leak between event writes, tau150 |1.691154881|34.96094%|Not imposed on a different cache mechanism|

AllIID interpolation scenarios have coverage1 except interwrite leak, with509/512 valid range-constrained states. That condition has maximum output error51.21965; values are not clipped back to the target domain. M0 fractional storage noise uses its own key half-range1.1 about1.4 and value half-range3, whereas interpolation uses120/297. The lower M0 errors under this declared normalization are not an equal-physical-noise experiment.

The existing modified carrier with its frozen original analog decoder gives IID Ecomp2.00124661 and3.71094% joint delivery. Its privileged full-state inverse supplied all keys/positions gives Ecomp5.07e−13 and100%. The frozen analog readout was not trained/optimized for this modified carrier; it is a diagnostic control, not a bound on every possible query-only decoder. The privileged inverse is not an equal-information comparator. Neither native comparison changes original P1B decision labels.

## Leakage mechanisms and adversarial support checks

Uniform post-encoding leak λ(a,b) preserves coefficient cancellation and yields error(λ−1)v, bounded by3(1−exp(−10/150))=.193479045. Differential leak yields

`e=(λa−λb)*a*q+(λb−1)*v`,

bounded by300|λa−λb|+3|1−λb|=9.486357845 for the fixed scenario. Adversarial maximum observed error9.30234 is consistent with that bound. These are exact exponential leak mechanisms with no compensation, not an Euler-TAN parameter comparison.

Leakage of the first stored(k1,v1) before the second write changes the denominator itself. The original separation does not protect `k2−leaked(k1)`. The preregistered interior-domain singular history sets k1=2, k2 equal to200 applications of exp(−.1/150) to2, slots0/1 and values1/−1. The online recurrence produces WRITE_SINGULAR exactly. This demonstrates a constructive binding mechanism's retention requirement, not a failure of the clean interpolation identity. The time-resolved phase1 leak uses the disclosed external step clock; no hidden timestamp real was stored in the carrier.

Adversarial cases are deliberately concentrated at support boundaries and minimum spacing. Their counts are not original-distribution failure rates. All clean binary64 cases return their targets and satisfy the actual key-spacing guard.

## Range guards are separate from arithmetic accuracy

The protocol fixed tight device ranges±120/±297, allowed only a1e−10 relative numerical band and forbade clipping. That guard is an explicit engineering policy with no headroom; it is not a theorem about every possible device. The ledger saves finite raw answers even when the range guard returns invalid/fallback0.

In binary32,16of18 adversarial histories exceed the tight coefficient range after rounding, triggering fallback and aggregate Ecomp2.6666667. **Their raw answers are nevertheless correct to4.44e−16 for these symmetric fixtures**, because written/query keys share the same casting and cancellation. This is range-policy rejection, not demonstrated numerical inability to retrieve the values. On IID binary32 cases the maximum actual target error is3.96399e−6 and no range rejection occurs. Exact Fraction checks for all18 adversarial and first16IID histories separate input/coefficient rounding from final multiply/add rounding and certify the observed errors. Correlated casting is not independent key measurement noise.

Interpolation's binary32 arm casts each write/read arithmetic operation. M0's corresponding arm quantizes its stored keys/values and query to binary32 but computes nearest-key distances in binary64. Here each clean cast query equals its corresponding cast cache key, so that arithmetic distinction cannot change the selected identity; it must nevertheless not be described as a completely binary32 M0 implementation.

The same distinction matters for M0: adversarial .001 storage-noise raw Ecomp.00203196 becomes2.66687945 after16 boundary-range rejections; write-key-noise raw answers have zero error but10 records violate the deliberately tight key-storage range. For interpolation, noisy-write/query raw maximum error.02303969 respects its .02409639 bound even where the coefficient range rejects. No guardband was tuned or added after observing this. Both raw and guarded metrics are reported to avoid treating these policy failures as inherent representational limits.

## Reproducibility and interpretation

`PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/interpolation_control/run_interpolation.py` completed once with exit0 in about9.23host seconds. Every input, noise draw, phase transition, final state, query, prediction, fallback, candidate-control result and metric is saved. The query kernel accepts only two state numbers,q and static arithmetic dtype. The theory review independently confirms exact interpolation, tight bounds, leak identities and input/control resource requirements.

The independent implementation review passed all14,166 rows and46 metric cells, regenerated inputs/noise, checked3,180 raw bounds and2,084 exact precision records, and cross-checked34 modified-native control trajectories with separate block dynamics. It found no material query leakage, scoring or mechanism defect. REVIEW.md, REVIEW_CHECKS.json and RANGE_PRECISION_REVIEW.md retain the qualifications, especially the distinction between raw recovery and tight-range guard rejection. No primary result was rewritten during review.

The supported conclusion is constructive but restricted: **an explicit nonlinear online write can encode N=2 key–value responses in two persistent real payload coordinates without retaining per-key records at query time**. The task alone therefore does not force such records. The construction shifts requirements into separate input access, controller state, nonlinear arithmetic, ideal retention and analog precision/readout. It does not establish an equivalent physical neuron, free robustness, original-TAN success, a minimal state count or an N=3 capability.
