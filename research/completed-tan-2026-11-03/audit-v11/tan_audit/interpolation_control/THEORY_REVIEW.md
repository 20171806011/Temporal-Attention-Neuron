# Two-point interpolation: independent theory and repository review

This is a useful constructive representation control, not new interpolation mathematics and not a TAN implementation. With an explicit nonlinear writer, a write-phase mechanism and ideal analog arithmetic, two real payload registers can retain the function needed to answer either clean N=2 key query. It is therefore not logically necessary to retain each event key as a separate post-encoding record. No lower bound on state count is proved here, and no N=3 or biological equivalence claim follows.

## Repository novelty check

I searched218 Python, Markdown, TeX and text sources in the inspected repository, then read the actual generator and M0 paths. Search expressions included interpolation, Lagrange, Vandermonde, divided differences, polyfit/polyval, value-difference formulas and Chinese 插值. The8 matches were plotting/regression/interpolation or shell-string handling, not a two-event online affine memory. The exact hit list and source hashes are in theory_checks.json. This is absence from the inspected textual source search, not an exhaustive literature novelty claim or a proof about unsearched binary artifacts.

The existing ScalarMetricAttention uses `-(q-K_i)^2` over all candidate keys (`p1_experiments/models.py:297`); BaseModel subsequently reads the chosen buffered value. P1B's M0 directly loops over all event keys and values (`p1_experiments/p1b/runner_p1b.py:91`). Thus M0 already solves the ideal task but retains a two-key/two-value buffer for N=2. The interpolation construction has the same noiseless task answer while replacing that post-encoding buffer by two function coefficients. Its value is isolating representation and online binding resources, not claiming superior ideal accuracy to M0.

## Exact representation and tight dynamic ranges

For distinct event keys,

`a=(v2-v1)/(k2-k1)`, `b=v1-a*k1`, `f(q)=a*q+b`.

Then f(k1)=v1 and f(k2)=v2 exactly. Swapping event order leaves the final coefficients unchanged. Different histories may yield the same line, but whenever the same q belongs to both histories their demanded value is the same line value. This avoids the earlier carrier's same-state/same-query/different-value counterexamples without reconstructing the original history. The clean task promises q is one of the stored keys; the coefficient state cannot test key membership for an arbitrary off-support query.

The actual TaskGenerator defaults are L=.3, U=2.5, minimum spacing δ=.05, |v|≤V=3, with keys sampled/rejected before independent values. Positions and chronological assignment are random. NumPy's upper endpoints are excluded from random draws, so the following tight bounds are support-closure suprema, not assertions that an IID draw attains the endpoints.

Ordering keys x<y and writing d=y-x≥δ gives

`|a|≤2V/δ=120`,

`|b|=|(y*v_x-x*v_y)/d|≤V*(x+y)/d≤V*(2U/δ-1)=297`.

The closure case x=2.45,y=2.5,v_x=3,v_y=−3 gives a=−120,b=297 and attains both absolute bounds. The first write stores k1∈[.3,2.5],v1∈[−3,3]; the second write changes the register meanings to slope/intercept, with much larger raw ranges. Fixed physical ranges spanning the coefficient extrema leave much of the first-write range unused; stage-specific rescaling would itself require declared gain switching.

Floating fixture caution: literal Python `2.5-2.45` is `.04999999999999982`, which fails the generator's `>=.05` guard. The exact rational extreme is valid for the mathematical closure, but a float adversarial trial should separately pass its actual spacing guard. One valid interior float pair is `[2.4499999999999993,2.4999999999999996]`, with gap `.050000000000000266`. Do not label a rounded invalid fixture as generator-valid.

## Perturbation, precision and delay bounds

For stored coefficient errors Δa,Δb and measured query q+Δq, the error relative to the true queried value v=a*q+b is exactly

`e=q*Δa+Δb+a*Δq+Δa*Δq`.

Under |Δa|≤εa, |Δb|≤εb, |Δq|≤εq,

`|e|≤|q|εa+εb+(|a|+εa)εq≤2.5εa+εb+(120+εa)εq`.

The cross term is required. The script exhibits an exact rational sign-aligned case attaining the independent perturbation bound. This assumes additive perturbations without undeclared clipping; noise may require range headroom beyond legal coefficient extrema.

If final registers are normalized as z1=a/120,z2=b/297, independent |Δz_i|≤ε gives

`|e|≤597ε+(120+120ε)εq`.

These are fractions of declared coefficient ranges, not physical energy normalization. Equal numeric noise in raw a,b, key/value states, TAN voltage/gate, or differently scaled registers is not a matched noise assumption. If a B-bit uniform quantizer covers each normalized interval[−1,1], nearest-grid rounding has ε≤1/(2^B−1); at exact q a16-bit example gives the worst-case coefficient-quantization bound597/65535≈.00910964. This is an analytical bound, not an empirical bit-depth sweep or a guarantee about write arithmetic.

Measurement errors during writing are different. If both written keys have |δk_i|≤εk<δ/2 and values |δv_i|≤εv, the perturbed line slope satisfies

`|a_tilde|≤(6+2εv)/(.05−2εk)`.

At queried true key k_i with query error Δq, its error is exactly `δv_i+a_tilde*(Δq−δk_i)`, hence bounded by `εv+|a_tilde|*(εq+εk)`. If encoder and query share the same key rounding error, those terms can cancel; independent measurement noise must not be inferred from that correlated precision check. M0 with exact stored keys instead retains the correct nearest-key identity under |Δq|<δ/2, illustrating a real robustness difference despite identical clean accuracy.

Uniform post-encoding coefficient leak `(a,b)→λ(a,b)` gives `f_leaked(q)=λv`, so error≤3|1−λ| at exact q. Differential leak obeys

`e=(λa−λb)*a*q+(λb−1)*v`,

and is bounded by `300|λa−λb|+3|1−λb|`. Common leak preserves the potentially large slope/intercept cancellation; independent leak need not. This must be separate from first-write retention. If stored k1 leaks to λk1 before the second event, the denominator can collapse even though original keys are separated. A uniform guarantee needs `δ−(1−λ)U>0`, i.e. λ>.98. With exponential leak over gapT, this requires τ>T/(−log .98). The original event gaps can be20–80ms, so a chosen write-delay test must retain its event timing and units explicitly.

Floating arithmetic is another mechanism. For exact represented inputs and query, round-to-nearest normal arithmetic, no overflow/underflow, and the stated subtract/divide/multiply/add sequence, standard γ3=3u/(1−3u) gives εa≤120γ3. A conservative propagation through intercept construction and query evaluation yields total endpoint error bounds about1.96695e−4 for binary32 and3.66374e−13 for binary64. The exact formulas are in theory_check.py/.json. These do not silently include conversion of higher-precision inputs; input quantization is covered separately by the written-key/value analysis above. Correlated same-key casting should be identified, not counted as independent query noise.

A fixed analytical coordinate improvement is possible without new information: choose the known midpoint c=1.4, store `(a,d=b+c*a)`, query `a*(q-c)+d`. The tight |d| bound is129 and |q-c|≤1.1, so analogous normalized coefficient noise gives261ε instead of597ε. This is an invertible coordinate/gain change, not extra capacity. It adds explicit fixed centering arithmetic and should remain an analytical observation unless frozen as the chosen implementation; it is not a reason to launch a parameter sweep.

## Required online resources

1. **Input interface:** the writer needs separately accessible k and v at each event, or an equivalent invertible input representation. Native TAN receives content `k+v` and novelty `max(0,k−1.4)`; that pair is noninvertible below the novelty threshold. Raw keys/values being available to the Python simulator does not make them free native neural input ports.
2. **Payload storage:** two persistent real registers, first meaning(k1,v1), later meaning(a,b). At the second event the old pair and current input pair coexist as write operands. Division, subtraction and multiplication require circuitry or combinational computational workspace, not an unreported extra persistent history. An in-place implementation is possible: first `x=(v2-y)/(k2-x)`, then `y=v2-x*k2` while the current input remains available. This uses equivalent b=v2-a*k2 and needs no old-event cache after slope assignment. If using the requested b=v1-a*k1 expression instead, old values must be retained transiently until both results are formed. The two arithmetic orderings have the same ideal answer but can differ in floating rounding; freeze the chosen sequence.
3. **Phase control:** fixed N=2 does not make random active-slot times known. The active mask/onset identifies an event but not its ordinal without memory or external control. Under guaranteed exactly two events plus an external onset/slot clock, trial reset and final read phase, one persistent first-write bit suffices to choose the two update laws. A total controller distinguishing empty/one-event/complete needs three statuses (two binary bits). The raw(k,v) and(a,b) state domains overlap, so their values alone do not supply a reliable phase tag in this specified representation. A fixed two-event timing schedule could supply phase externally, but the original generator randomizes positions; an external first/second label or stored schedule is an added resource.
4. **Event sampling:** the native task presents each event for100 Euler steps. The new writer must execute once per event, not100 times. A supplied event-onset strobe, or the known slot clock plus active mask, provides that control and should be declared. An independently implemented edge detector would need its own state.
5. **Readout:** query-only after encoding means the decoder receives q and the two coefficients, not historical keys, values or positions. It still performs ideal analog multiplication/addition and returns a continuous value. This differs from original Count/Latency codebooks and does not validate a biophysical neuron or its10ms physical readout.

Thus the honest description is **two real payload registers plus disclosed phase/input/write/read circuitry**, not two numbers with otherwise identical biological resources. No post-encoding per-key cache is necessary for this ideal N=2 contract, but the construction does not establish that all the replaced operations are biologically cheap, noiseless, continuous, trainable or implementable in one TAN unit.

## Completed verification

`PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/interpolation_control/theory_check.py` completed with exit0. It verifies exact rational interpolation identities and tight extremes, an exact perturbation saturation case, leak identities, a valid floating minimum-gap fixture, precision bounds and the repository search ledger. It does not generate a new experimental population, tune parameters, or replace the study's preregistered experiment. Original repository files remain unchanged.
