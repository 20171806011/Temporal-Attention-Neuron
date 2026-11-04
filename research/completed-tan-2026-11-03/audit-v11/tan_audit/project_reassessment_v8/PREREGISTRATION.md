# Original Probe1: target-position swap, native common reset and complete-response collision

2026-11-02, saved before fixture/simulator execution. No new learned decoder, model search or performance-driven retuning. This is the single new experiment in this continuation; independent P1A saved-ledger arithmetic is a historical evidence check, not new trials. The broader source review and consolidated P1B diagnosis precede it.

## Question and reason for selection

P1A Rev2 already has strong static-address evidence, including a complete independent recomputation of its 60,000 condition/trial records and 420,000 outputs. Natural state collisions belong to earlier TAN-I Phase1, not static P1A. Existing Phase1 proves h-only insufficiency and already has common-reset controls. Existing Probe1 reports limited target-class recovery but does not establish that its entire permitted response is sufficient for the target.

Can two valid original Probe1 histories have different, unambiguous target classes, retain those distinct target values in the ordered window at query, yet produce exactly the same full allowed response? A positive answer would locate a concrete failure of observable target recovery without confusing it with a collision of full state or with mere trajectory separation. The candidate mechanism is native reset followed by permutation-invariant query aggregation and gate-closed continuation. These are standard operations; the contribution is their verified task-specific consequence, not a new general symmetry theorem.

Sources: `snapshot/code/experiments/temporal_distractor.py` constants/generator/resp_features; `snapshot/code/experiments/natural_collision.py` frozen simulate; current `snapshot/paper/sections/methods.tex` Probe1 contract. Search evidence and prior-scope review are in NATURAL_COLLISION_REVIEW.md and BROAD_MANUSCRIPT_REVIEW.md.

## Original contract

Zero initial h and zero prehistory. Fourteen input steps. Target at t5; three distractors at t6–8; query at t9; all other inputs zero. Low class support [.55,1.05], high [.95,1.45]; present target label is its latent low/high class, not the query. Distractor/query amplitudes use the same fair low/high mixture. No catch class is needed for this constructed two-label witness, and it is not a three-class population-accuracy estimate.

Frozen native parameters W5, lambda .5, threshold .5 with strict `u>theta` reset, Wq2/Wk1/Wv1, beta1, epsilon0. Original softmax denominator includes +1e-9. Original observable F is the complete ten coordinates `[u9,u10,u11,u12,u13,y9,y10,y11,y12,y13]`, using pre-reset u, not post-reset h. Neither pre-query observations nor raw inputs/window are available to the task decoder.

## Frozen fixtures and predictions

Write the active block as `(a,b,c,d,q)` in history L and `(b,a,c,d,q)` in history H. Labels are respectively low and high. Both histories use the same query. Only target and first distractor exchange values and compatible latent classes.

Anchor: a=c=5/8, b=d=q=11/8. These amplitudes are strictly inside non-overlapping portions of the original class supports. Native B4 prediction: both histories reset at t8, have positive surprise at t9, and produce equal ideal complete F despite different ordered windows and target values. The same finite floating implementation may have tiny rounding differences under permutations; log literal equality and maximum residual separately. Numeric closeness is not the proof of ideal equality.

Robust-support family, fixed before evaluation:

- a,c in [123/200,127/200] = [.615,.635].
- b,d,q in [273/200,277/200] = [1.365,1.385].
- Exactly all 32 corners and 512 independent draws from this five-dimensional box using NumPy RandomState2026110225, plus the anchor: 545 paired cases, 1,090 histories. Save/hash the complete input ledger before simulation. Random cases are coupled interventions, not iid draws from the unrestricted task.

Analytic proof obligations: (i) throughout the box, t8 surprise lies within [.557,.593]; a rational exponential-bound certificate proves C8>1 and tanh(S8)>.5, hence drive>.5 and both native h8=0 regardless of earlier nonnegative h7; (ii) at t9, 0<S9<=.32 and C9<=1.385, so 0<u9<.4432<.5; (iii) the two t9 multisets are equal, hence ideal surprise/context/current/u9 are equal; (iv) t10 onward x=0 and nonnegative windows imply gate closed, so F is exactly `[u9,u9/2,u9/4,u9/8,u9/16,0,0,0,0,0]`. Ordered windows differ at t9/t10 and are identical after t11, so the task observation can miss a real retained target before it leaves the window.

Use exact Fraction interval bounds for exp, summing terms through degree40 and bounding the remaining positive tail geometrically. Do not substitute high precision floating numerics for a certified sign. At t8, sufficient lower bound for `(sum exp)*(C8-1)` is `.73*exp(1.114*1.365) - .77*exp(1.186*.635) -1 -1e-9 >0`. For tanh(.557)>.5, prove exp(1.114)>3. At t9 use tanh(S)<S and C<=max window amplitude. Record actual full trajectories and margins separately.

## Prespecified controls and intervention

Run unchanged native B1 LIF, B2 recency delay line, B3 uniform-attention gated model and B4 full TAN on every history. This is 4,360 native trajectories. They use the same input pair and each model's own native history evolution; do not equate them to a pure one-parameter causal comparison. B1 should have equal query responses for the anchor because preceding active pulses all exceed its threshold. B2 is the positive task-observation control: both h8 reset and anchor u9 values are47/40 and9/8, difference1/20. For the full box its ordered weighted current distinguishes the swapped target by `(b-a)/15`. B3 outcomes are diagnostic, with no success/failure requirement chosen after viewing.

An exact full-window target-tap oracle reads x5 at t9. It identifies the two classes on the entire box because their supports are disjoint. This is a privileged state-access control, not the native readout and not a general oracle for the original overlapping-support distribution.

One named B4 intervention suppresses only the voltage reset at t8, retaining the pre-reset candidate as h8 while preserving that step's threshold event indicator and all other frozen equations, windows and later resets. This adds 1,090 trajectories. It deliberately changes the model at one step; states above the usual post-reset range are intervention states. Prediction: removing that reset can preserve pre-query order effects in h and separate the response. Report all failures/equalities and numerical gaps; absence of separation would not invalidate a native collision but would limit the proposed reset explanation. Do not introduce alternate intervention times or parameters.

## Metrics and inference

For every pair/model save both full14-step native/intervention trajectories (u,h,current,S,spikes,alpha where applicable), complete windows, ten response features, labels, source/runtime hashes, reset/threshold margins and numerical equality checks. Report number of literal-equal F, max/min/L2 response difference, positive-novelty query counts, t8 reset counts, and raw target/window separations. All545 pairs remain in every denominator; no exclusions, fitting, bootstraps or significance stopping.

Ideal equality with opposite labels means any classifier of only F has equal-mixture error at least1/2 on the paired construction. This is not a prevalence estimate or the measured original three-class accuracy. The window oracle shows the task target remains available in full ordered state at query; the response collision does not imply a collision of complete system state.

The five-dimensional box supplies a nonzero-support robustness statement under structure-preserving paired changes, not a claim of robustness to arbitrary sensor/process noise. Independently check its mass if reporting it: conditional on a present distractor trial under the original fair class-mixture generator, each unambiguous amplitude interval of width .02 has probability .02, so each orientation has mass(.02)^5=3.2e-9; the swapped union has6.4e-9. If the measure-preserving pairing and equal densities are verified, the resulting ideal classifier risk bound is3.2e-9 in that conditional population. This tiny analytical floor is not an explanation of the observed large average errors or a discovery of frequent collisions. Otherwise omit the mass and retain only the admissible-pair result.

## Validation, stop and preservation

Native outputs must match the original nc.simulate and td.resp_features contracts; validate with an independent scalar implementation and independent theorem/certificate review. Required controls: source hashes unchanged, admissible latent classes/timing, identical common query, full rather than projected observation comparison, correct strict threshold/pre-reset feature semantics, and exact ideal-versus-float distinction. The intervention implementation must differ only at t8's reset assignment. Stop after the fixed fixture/box table and independent validation, including any negative intervention/control result. If the predeclared proof fails, preserve failure and narrow the claim; do not tune the box or amplitudes after execution. No public edits, installs, purchases, rewritten historical labels or new decoder search.
