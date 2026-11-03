# Fixed-payload precision comparison: independent pre-execution theory

The proposed fixed32-bit payload comparison is useful if it quantizes the **first event while it is retained**, not only the final coefficients. Exact interval arithmetic already shows a substantial distinction between a fixed physical register scale and an explicitly phase-dependent encoding. It also shows that the original±.5 delivery endpoint will be guaranteed in all three budget-matched clean arms under the stated ideal arithmetic assumptions, so their actual errors and finer accuracy should be the informative outcomes.

This is a storage/representation mechanism test, not a new interpolation identity or a total hardware/energy comparison. No IID histories were generated here, no parameters were searched, and prior results were not changed.

## Resource contract

Budget two16-bit coefficient-carrier words, or four8-bit M0 key/value words:32persistent payload bits in either case. Budget the existing empty/first/ready/error controller separately as2bits. Include event-onset strobes, trial reset/read clock, separate current key/value input ports, range decoding/encoding operations and external arithmetic/readout. Exact current(second-event) input and exact queried key are common assumptions; no initial-state drift or query noise is included in this particular precision calculation.

The two-register writer must retain only the quantized first key/value codes between events. Storing an unrounded Python copy and using it in the second write would invalidate the budget. Prefer explicit integer codes as persistent state and decode them only when required. Arithmetic intermediates may be float64 in the proposed simulator, but they are additional computational precision/workspace, not a claim of a fully32-bit circuit or equal energy. The same distinction applies to ideal real-valued query output.

The phase-aware arm changes the semantic conversion of its fixed16-bit words: first-phase key range[.3,2.5] and value range[-3,3], then final slope range[-128,128] and intercept range[-320,320]. The phase controller can select these fixed conversions without another per-history real register, but switching gains/offsets and conversion circuitry are explicit additional operations. This cannot be described as a coordinate-invariant or intrinsically more robust two-state neuron.

The endpoint-only control keeps an unquantized first record and rounds only(a,b) at the end. It therefore violates the matched peak payload budget: in a float64 implementation it retains128payload bits before the second event, and in an ideal-real formulation its precision is unbounded. It is a diagnostic isolating first-write error, not a fair32-bit memory competitor.

M0 stores only two quantized key/value pairs and performs nearest-key lookup. It does not need slot indices at query time. Its4words allocate16data bits to the first event and32after the second; interpolation allocates both16-bit words to the first record before overwriting them as coefficients. Peak payload is matched, not active-memory-time, arithmetic cost or transistor count.

## Uniform quantizer convention

With Bbits, inclusive endpoints[L,U] and2^B uniformly spaced levels, nearest-level rounding has

`epsilon <= (U-L)/(2*(2^B-1))`.

The denominator is2^B−1, not2^B. A deterministic tie rule should be frozen; the exact checks here use nearest-even, while the half-step bound is independent of which nearest endpoint wins a tie. Inputs outside the range need an explicit policy; the bounds below prove that legal inputs and intermediate coefficients are inside the fixed ranges, so a clean experiment should not encounter genuine saturation. Floating implementation tolerances still need to be logged separately.

## First-write error and safe headroom

Let the stored first record be `k1_hat=k1+ek`, `v1_hat=v1+ev`, with |ek|<=epsilon_k and|ev|<=epsilon_v. The current second event is unquantized. The intermediate line is

`a'=(v2-v1_hat)/(k2-k1_hat)`,
`b'=v1_hat-a'*k1_hat=v2-a'*k2`.

The original key separationdelta=.05 gives

`|k2-k1_hat| >= .05-epsilon_k`,
`|a'| <= (6+epsilon_v)/(.05-epsilon_k) = A_bound`,
`|b'| <= 3+2.5*A_bound = B_bound`.

These hold in either chronological key order. The second original target remains exact before terminal coefficient quantization. The first original target has error

`a'*k1+b'-v1 = ev-a'*ek`,

so its absolute error is at most `epsilon_v+A_bound*epsilon_k`.

Final16-bit coefficient rounding on±128/±320 contributes at most

`2.5*(128/65535)+320/65535 =640/65535 =.00976577401388571`

to either true-key query. The max-operand Ecomp bound is therefore the first-record bound plus this terminal contribution. This is a conservative uniform bound, not a claimed tight achieved maximum.

| Exact-certificate quantity | Fixed raw first-register ranges±128/±320 | Phase-aware first ranges[.3,2.5]/[-3,3] |
|---|---:|---:|
| First key half-step |128/65535=.001953154803|1.1/65535=.00001678492409|
| First value half-step |320/65535=.004882887007|3/65535=.00004577706569|
| Minimum effective denominator |.0480468451972|.0499832150759|
| Intermediate slope bound |124.979753871|120.041213194|
| Intermediate intercept bound |315.449384676|303.103032986|
| First-query error before final rounding |.248987693529|.00206065971639|
| Final coefficient rounding contribution |.00976577401389|.00976577401389|
| Total per-query/Ecomp bound |**.258753467543**|**.0118264337303**|

The guardbands128/320 are analytically self-consistent, not padding chosen from trial outcomes. The raw arm's certified margins are3.02024613 for slope and4.55061532 for intercept; phase-aware margins are larger. Thus an observation of a genuine out-of-range intermediate in a legal noiseless test would indicate an implementation/assumption error, not an expected precision failure. No clipping or zero fallback should be needed for these clean legal histories.

The ideal true coefficient bounds120/297 from the previous study are not sufficient as storage headroom after first-record quantization. The larger fixed ranges account for the perturbed denominator and numerator before any new experiments are generated.

## M0 identity and value precision

For8-bit key storage over[.3,2.5], the key half-step is

`epsilon_key_M0=1.1/255=.004313725490196078`.

At exact true queryq=k_i, the correct stored key is at distance at mostepsilon_key_M0. The other stored key is at distance at leastdelta−epsilon_key_M0. Since2*epsilon_key_M0<delta, nearest-key identity is strictly preserved for every legal history; quantized keys cannot collide. The identity margin isdelta−2*epsilon_key_M0=.04137254901960784. This proof assumes the exact clean query, not arbitrary additional query noise.

Only value quantization then matters:8-bit values over[-3,3] give absolute query error and Ecomp at most

`3/255=1/85=.011764705882352941`.

This bound is slightly below the conservative phase-aware interpolation bound, but comparing upper bounds alone does not establish a ranking of their expected errors. Both are below.05, and both budget-matched interpolation schemes are below the original.5 threshold. Actual error distribution, role asymmetry from the first write, and the gap from endpoint-only quantization are the informative finite-sample quantities. A narrower tolerance can be reported as a distinct diagnostic only if frozen before execution; original P1B labels must not be redefined.

## Why this is not just repeating M0

M0 remains the correct ideal task/reference buffer. The new comparison investigates a specific previously unmeasured resource: precision of the first event while it waits for nonlinear binding. The old final-coefficient noise test omitted that period. Equal continuous coordinate counts do not fix the number of representable first records, and equal persistent bits do not fix amplitude scaling or arithmetic cost. The raw-vs-phase-aware contrast isolates the conversion/precision assumption; the endpoint-only diagnostic reveals what is lost by ignoring the prefix. This makes a fixed-budget comparison informative without inventing a new memory task or optimizing against held-out data.

The result should not be called a universal memory lower bound, an optimal quantizer, equal biological noise energy, a latency measurement, or a biophysical TAN implementation. It is one declared finite-state storage design under exact ports, ideal retention and high-precision combinational arithmetic.

## Optional analytic obstruction to literal smooth coefficient writing

For a fixed second event(k2,v2), the ideal overwrite F maps the first record(k1,v1) to(a,b), where `b+a*k2=v2`. Its Jacobian has rank1 wherever k1!=k2:

`a_k1=(v2-v1)/(k2-k1)^2`, `a_v1=-1/(k2-k1)`,
`db=-k2*da`.

For example, fix(k2,v2)=(2.4,1.8). First records(.4,-2.2) and(1.4,-.2) are both legal and both map exactly to(a,b)=(2,-3). This is intentional forgetting of the old key, not a defect in retrieval.

A finite-time flow of a locally Lipschitz ODE with unique solutions under the same fixed input/time schedule is injective on its domain of existence: two distinct trajectories meeting at a finite final time would violate uniqueness when followed backward. Therefore it cannot exactly implement this literal rank1 overwrite on an open first-record domain using only the same two continuously evolving coordinates. The statement persists under fixed invertible coordinate changes and deterministic nonautonomous drive. Non-Lipschitz finite-time convergence, an explicit reset/projection, quantization, a different readout representation or hidden memory breaks the stated assumptions and is not excluded.

The frozen TAN nonspiking, unclipped, r=0 branch illustrates the gap concretely. For a fixed event input its two-state Euler map has linear part

`M=[[1-h/tau_m,(h/tau_m)*I_content],[0,1-h/tau_A]]`.

For the registered and tested positive time constants, detM=(1-h/tau_m)*(1-h/tau_A)>0, so any finite composition of such steps remains injective. Its existing threshold resets/clamps are noninvertible, hence the argument does **not** prove that all hybrid TAN mechanisms or all neurons fail. It only prevents silently treating the instantaneous ideal coefficient overwrite as the already implemented smooth branch. Implementing a suitable input-dependent reset/transformation would be a new mechanism with its own resource/validation contract.

For an approximate collapse of two initial states separated by distanceD, if the vector field is L-Lipschitz in a fixed declared coordinate norm over timeT, Gronwall gives final separation at leastexp(-L*T)*D. If both outputs must be within epsilon of the same ideal target state, then `epsilon >= D*exp(-L*T)/2`. This is a time/contraction/precision relation under a specified norm, not an energy lower bound. Rescaling coordinates changes its constants; it should not be used to rank different physical state units without an explicit normalization.

## Exact verification completed

`PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/precision_binding/theory_check.py` completed with exit0. It uses Fraction arithmetic throughout the quantizer and bounds, checks72 predetermined closure/boundary histories (144queries per arm, not a random population), verifies M0 identity and all error/headroom certificates, and exhibits the exact rank1 collision. THEORY_CHECKS.json preserves the exact fractions, rounded displays, fixtures and limitations. No repository source, earlier results, network state or publication state was changed.
