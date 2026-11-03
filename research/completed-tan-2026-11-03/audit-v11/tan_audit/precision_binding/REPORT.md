# Online binding has a prefix-precision cost

Completed 2026-11-02. One fixed 32-bit persistent-payload comparison tests a missing resource in the ideal two-register interpolation control: the first event must remain available at finite precision until the second event arrives. This intermediate storage matters, and explicitly changing the coding scale between phases controls its error. The study does not implement the registered TAN neuronal dynamics.

All 6,360 paired prediction rows remain in the ledger. Every arm passed the original strict ±0.5 operand tolerance on these clean cases. This is a tolerance check, not a claim to pass the original P1B experimental gates using a different encoder/readout. No original labels changed.

## Why this experiment

Two-point affine interpolation is standard mathematics: the formula and its divided difference are the degree-one specialization of classical interpolation. Repeating perfect real-valued retrieval would only reconfirm that identity. [NIST DLMF §3.3](https://dlmf.nist.gov/3.3).

The repository's M0 already retrieves values from a key/value cache. Hopfield's collective pattern completion and fast-weight key/value updates use different storage and update mechanisms; neither supplies a free interpolation writer. This continuation contributes a generator-specific resource audit, not a new associative-memory formula or a literature priority claim. The [literature review](LITERATURE_REVIEW.md) records the bounded search and precise scope of source inspection. [Hopfield (1982)](https://pmc.ncbi.nlm.nih.gov/articles/PMC346238/), [Schlag, Irie and Schmidhuber (2021)](https://proceedings.mlr.press/v139/schlag21a.html).

The previous final-coefficient perturbation test did not impose a digital budget during first-event retention. The current comparison isolates that omission. A finite-time divider simulation was not selected: silently supplying its controller and working memory would obscure the resource question.

## Frozen comparison

The actual N=2 generator supplies keys in [0.3,2.5), separation at least 0.05, values in [-3,3), and two occupied slots among five. There are 512 fresh histories from one continuous RandomState(2026110215), plus 18 previously fixed adversarial histories reported separately. Each has original, query-swap and re-encoded value-swap versions. The 1,590 inputs yield four-arm comparisons: 6,360 rows and 12,720 operand outputs. Variants are dependent, not additional IID histories.

| Arm | First-event storage | Storage after second event | Peak payload |
|---|---|---|---:|
| RAW32 | 16-bit key on ±128; 16-bit value on ±320 | 16-bit slope on ±128; 16-bit intercept on ±320 | 32 bits |
| PHASE32 | 16-bit key on [0.3,2.5]; 16-bit value on [-3,3] | Same coefficient coding as RAW32 | 32 bits |
| CACHE32 | Two 8-bit key/value fields | Four 8-bit key/value fields; nearest-key query | 32 bits |
| ENDPOINT_ONLY | Two binary64 key/value fields | Same coefficient coding as RAW32 | 128 bits |

All count a separate 2-bit phase/error controller. PHASE32 explicitly changes scale interpretation with phase. ENDPOINT_ONLY deliberately exceeds the first-phase budget and is a diagnostic, not an equal-resource competitor. Matched-arm persistent payloads contain bounded integer codes; returned logs are evaluator records and are not read by the writer or query method.

Quantization uses endpoint-inclusive uniform grids with 2^B levels and binary64 nearest/ties-even scaled rounding. There is no clipping, range fitting or retry. The second coefficient write uses `a=(v2-v1_hat)/(k2-k1_hat)` and `b=v2-a*k2`. Query receives only stored codes, q and shared constants. Inputs, temporary arithmetic and output arithmetic remain binary64. CACHE32 quantizes both incoming records; interpolation uses the current second event at full input precision before final coefficient quantization. This is an explicit algorithmic difference. The budget is logical persistent information, not total circuit size or Python object memory.

Perfect retention is assumed during the 100 ms episode. There is no additive noise, drift, query error or simulated finite write latency. Event-onset/reset/clock inputs, separate raw key/value ports, gain conversion, division, temporary arithmetic and output precision are additional resources. Original TAN input ports and spike readout are not implemented by this algorithm.

## Pre-outcome analytic guarantees

With first-record errors ek and ev, the effective denominator has magnitude at least `0.05-epsilon_k`, and slope magnitude is at most `(6+epsilon_v)/(0.05-epsilon_k)`. At the first true key the pre-final-quantization error is `ev-a*ek`; the second true key is exact in ideal arithmetic. Small key separation therefore amplifies the prefix error.

Final coefficient quantization adds at most `640/65535 = 0.0097657740` per query. Exact rational arithmetic gives these sufficient full-domain bounds:

| Arm | Per-query / Ecomp upper bound |
|---|---:|
| RAW32 | 0.2587534675 |
| PHASE32 | 0.0118264337 |
| CACHE32 | 0.0117647059 |
| ENDPOINT_ONLY | 0.0097657740 |

RAW32 intermediate slope/intercept bounds are 124.979754 and 315.449385, within frozen ±128/±320 ranges. CACHE32 key error is at most 1.1/255, below half the minimum separation, guaranteeing nearest-key identity for an exact valid query. Its value error is at most 3/255. These are conservative guarantees for these quantizers, not tight maxima or optimal-bit lower bounds. The [theory report](THEORY.md) includes 72 predetermined exact-arithmetic boundary histories; they are not population samples.

## Observed results

Fresh original histories only, n=512 per arm:

| Arm | Mean Ecomp | Maximum error | Local valid joint accuracy, δ=0.05 |
|---|---:|---:|---:|
| RAW32 | 0.006997014 | 0.065580048 | 509/512 |
| PHASE32 | 0.003198119 | 0.008822767 | 512/512 |
| CACHE32 | 0.007872440 | 0.011751497 | 512/512 |
| ENDPOINT_ONLY | 0.003147289 | 0.008811351 | 512/512 |

Coverage and strict ±0.5 joint tolerance are 100% in all 24 arm/group/variant cells. No range, singularity or numerical boundary-band failures occurred; no cases were dropped. All 22,260 quantization operations obeyed their half-grid bounds. The local δ=0.05 criterion was frozen separately and does not replace any original P1B label.

The primary paired RAW32-minus-ENDPOINT_ONLY mean Ecomp contrast is +0.003849725, with fixed 2,000-replicate percentile interval [0.003227670, 0.004559723]. PHASE32-minus-RAW32 is -0.003798895 [-0.004497331, -0.003173994]. PHASE32-minus-CACHE32 is -0.004674321 [-0.004957422, -0.004381074]. These intervals resample the 512 original IID histories, not their variants; they describe these fixed algorithms/distribution without multiple-comparison adjustment.

The decomposition localizes the effect. RAW32's chronological first-event mean absolute error before final coefficient rounding is 0.005553483, maximum 0.066621680; second-event error is at most 5.33e-15. PHASE32 reduces the first-event mean to 0.000049636 and maximum to 0.000556340, leaving terminal coefficient rounding as the dominant error. Every decomposition is saved.

Among 18 reused adversarial histories, RAW32 mean Ecomp is 0.128906649, maximum 0.242236973, local δ=0.05 success 6/18. PHASE32 mean is 0.002387401, maximum 0.004867628; CACHE32 mean is 0.000490196, maximum 0.008823529; ENDPOINT_ONLY mean is 0.001433119, maximum 0.004867628. These chosen cases are not a prevalence estimate; grid alignment favors the cache on many endpoint values.

Query swapping exactly permutes predictions and preserves payloads. Value swapping re-encodes every history. On fresh value swaps RAW32 mean Ecomp is 0.006558010 with local accuracy 510/512; other arms retain 512/512. No arm changed after outcomes.

RAW32 has slightly lower average error than CACHE32 on the fresh sample but a substantially larger observed tail. PHASE32 has lower average error than this equal-bit cache under the specified coding and high-precision arithmetic. Neither establishes an intrinsically superior physical representation: scale conversions, incoming precision, division, retention and output requirements are additional assumptions.

## Resource accounting and mechanism boundary

Matched arms allocate 32 payload bits over 100 ms, or 3,200 allocated bit-ms, plus 200 controller bit-ms. Mean occupied payload bit-ms are 2,581.25 for RAW32/PHASE32 and 1,920.625 for CACHE32, which occupies only 16 bits between writes. ENDPOINT_ONLY has peak 128 bits, 12,800 allocated bit-ms and mean occupied bit-ms 6,545. Mean inter-event delay is 41.289 ms. These logical occupancy measures are not energy, area or biological cost.

There are two payload-write events. The interpolation update requires two subtractions for numerator/denominator, one division, one multiplication and one subtraction for the intercept, beyond conversion and control. Query requires decoding plus multiplication/addition. Cache query requires decoding, two distances and comparison. The simulator does not specify arithmetic hardware, temporary-register scheduling, divider latency or ADC/DAC energy. Equal payload bits do not make these costs free or equal.

A separate exact argument sharpens the gap to the smooth TAN branch. For fixed second input (k2,v2), ideal coefficient overwrite maps all first records onto `b+a*k2=v2`, a rank-one, noninjective map. First records (0.4,-2.2) and (1.4,-0.2), followed by (2.4,1.8), both yield (a,b)=(2,-3). A finite-time locally Lipschitz ODE flow under the same drive is injective by solution uniqueness and cannot exactly equal this literal overwrite on an open first-record domain. The nonspiking, unclipped two-coordinate TAN Euler branch also has nonzero determinant per step. Reset/projection, quantization, non-Lipschitz convergence, a different representation or added state can break the assumptions. TAN already has hybrid resets; this is not a universal neuron impossibility result.

Final two-coordinate sufficiency therefore does not settle prefix precision, coding or write-mechanism cost. An explicit finite-precision coding rule can make N=2 retrieval accurate, while a physical neuronal implementation still needs a counted nonlinear write, controller, retention and readout mechanism. This study stops there, with no dynamics sweep or N=3 claim.

## Evidence

- Protocol SHA256: `ae2c6180dd17eb085277153a58574970a03c413210a9db78c49ef8f8ce207db3`.
- Input SHA256: `93fb40c3de844c59c83e93fd8d40f032fe2dc00d3795c6b0422cecace881ca16`.
- Main script SHA256: `b46c81c57f9bafb0d66dfda7ecc1d0bf498d05bc668d04331e1c5e1b82d8119b`.
- `run_precision.py`, `run.log`, `results.json`, `inputs.jsonl` and `predictions.jsonl` preserve the full run. Recorded wall time was 0.743 s, not physical write time.
- `theory_check.py` completed with exit 0. `independent_review.py` also completed with exit 0: it independently regenerated all inputs, all metrics and bootstrap intervals, and reconstructed every online write with exact rational arithmetic. Final integer payloads matched all 6,360 rows, with zero rounding-code disagreements. `REVIEW.md` and `INDEPENDENT_REVIEW_CHECKS.json` preserve the full audit.

Prior results and repository outputs remain preserved.  The old whole-archive transfer limitation remains a provenance boundary for unseen stages and did not block this repository-based test.
