# Native readout support: independent strict recheck

**Both stronger bounds survive this recheck**, conditional on the frozen clean P1B task and its readout. Original source, historical outputs, and decision labels remain unchanged. The earlier archive was not used or retried.

## Timeline proof and execution

Every valid stored voltage after a non-refractory encoding step is below threshold 1; a threshold crossing resets it to zero. Each slot ends with 100 silent steps, including slot 4 (`models_p1b_v2.py:178–189`). In exact arithmetic, with lambda=99/100, positive non-refractory voltage decays as lambda*u, negative voltage remains negative, and positive refractory state clamps voltage to zero. Refractory state clears within 20 steps. Therefore at encoding completion, and after any registered nonnegative delay, r=0 and u<lambda^100. This argument does not assume u is positive. Thirty exact/float silence checks include negative voltages down to -100 and refractory starts from .001 to 2 ms.

The clean generator's query is an actual candidate key in [.3,2.5]. Cue dynamics uses I_content=0 and I_query=q (`models_p1b_v2.py:249–254`), so A does not change voltage. Before a first spike the upper envelope is

`U_n = 5/2 + ((99/100)^100 - 5/2)*(99/100)^n`.

| Exact envelope or reset trajectory | Voltage |
| --- | --- |
| Query step 35 | 0.9988653692650254 |
| Query step 36 | 1.0138767155723751 |
| 50 integration steps from reset, q=2.5 | 0.9874848321561583 |
| 51 integration steps from reset, q=2.5 | 1.0026099838345968 |

Thus no first spike can occur before step 36. After a step-36 spike, steps 37–56 are the 20 refractory updates; step 56 clears r but still clamps u. Integration resumes at step 57. At least 51 integration steps are needed, so a second spike cannot occur before step 107. Exact rational recurrence and actual repository `euler_step` both produce [36,107] from the upper-envelope starting state. The float r sequence clears after precisely 20 updates. A generous outward error budget of 2^-40 per step, larger than the rounding budget for the task's bounded binary64 arithmetic, still leaves the step-35 and reset-step-50 candidates below threshold; the exclusion is not resting on an ULP-scale threshold margin.

Step 36 is also attained by a **valid interior-domain encoding history**, not merely the relaxed envelope: active slots [0,4], keys [.5,2.499], values [0,2.1566698870632344]. The final event ends at u≈.999 without spiking; query onset is u=0.3656663089319563, r=0. Actual native query q=2.499 spikes at 36. Continuing that same cue beyond its registered deadline yields a diagnostic second spike at 107. This continuation is used only to check the bound; the registered decoder still receives its original window.

## Eligibility, latency bins, and silence

The production readout generates raw steps 1–100, then takes `raw_spikes[:99]` (`models_p1b_v2.py:272–274`). The exact deadline step 100 is excluded. One hundred fixtures construct first spikes at each integer step 1–100, with exact recurrence followed by the actual production readout. Step 99 is eligible and maps to latency bin 19; step 100 yields count zero and abstention. The registered q=1.58 boundary independently gives a sole raw spike at step 100 and no eligible spike. Early-step fixtures necessarily use out-of-task query magnitudes and are explicitly decoder-conformance checks, not evidence of task reachability.

For **all 99 eligible steps**, the actual float expression `int((k*0.1)/0.5)` equals exact integer `k//5`, including every bin boundary. Thus k>=36 restricts native latency to bins 7–19; floating bin rounding does not introduce bin 6 or another missing output.

Count has only {-2,0 fallback}; latency has the superset `{-2.85+.30*j : j=7,...,19} union {0}`. They are output supersets, not assertions that every value is achievable for every history.

Four actual `runner_p1b.evaluate_episode` checks use histories with zero novelty and no spikes. When targets are (0,0), the returned composition error is zero while both deliveries fail through abstention. When targets are (1.2,-.4), the fallback remains zero and E_comp is 1.2. Therefore zero must be included in the composition alphabet, while zero fallback is unsuccessful delivery. Analog deadline readout has different pre-reset candidate semantics and is not included in this spike-only support argument.

## Independent exact floor calculation

A separate agent used a **two-dimensional Voronoi rectangle integral**, independent of the earlier CDF integration. Each nearest-symbol half-cell has distance coordinate x in [0,a]. Pairing half-cells of lengths a,b contributes

`integral_0^a integral_0^b max(x,y) dy dx = min(a,b)*max(a,b)^2/2 + min(a,b)^3/6`.

Summing and dividing by the target square's area 36 gives:

| Output superset | Number of rectangles | Unnormalized integral | Population E_comp floor |
| --- | --- | --- | --- |
| Count {-2,0} | 16 | 52 | **13/9** |
| Latency bins 7–19 plus zero | 784 | 11187/400 | **1243/1600** |

These exact fractions refer to the intended rational codebooks and independent continuous uniform targets on [-3,3]. Using exact rational representations of the Python binary latency reconstruction values instead changes its floor by approximately **−3.33e−18**. Floating metric arithmetic and finite PRNG discretization also prevent treating a decimal idealization as a literal bitwise theorem about every run; none is remotely large enough to affect the .05 threshold comparison. These remain population relaxation bounds, not lower bounds on every finite-sample average.

The conditions matter: original membrane time constant, threshold/reset, final silent duration, clean bounded q, and zero cue content. Nonzero probes, analog samples, other membrane constants, changed output codebooks, or noisy/unbounded queries require new arguments. The results establish restrictions of this frozen end-to-end readout, not a universal neuron impossibility or population-coding necessity.

## Reproduction

`PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/floors/native_support_recheck/check_native_support.py`

`PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/floors/native_support_recheck/math_independent/rectangle_integral.py`

All assertions passed. Main `results.json` preserves full exact/float timelines, source hashes, the interior-domain witness, 30 silence-state cases, 100 first-step fixtures, and four runner scoring checks. `math_independent/results.json` stores exact rectangle sums and the binary-codebook correction. No blocker remains for this completed recheck.
