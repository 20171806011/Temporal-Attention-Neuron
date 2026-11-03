# Supplied nonlinear held-out source audit

## Scope and execution

The audit used supplied source text for `nonlinear-heldout/run_heldout.py`, `summarize_uncertainty.py`, and both reviewer protocols by direct text. Their decoded scientific source is preserved under `prior_sources/`; the original transport-only text duplicates are omitted from this public copy. Historical Python execution copies differed only by HTML entity decoding. `SOURCE_PROVENANCE.json` records both hashes. This is a reproduction from supplied source text, not a byte comparison with the unavailable ZIP. Earlier stages outside these supplied files remain unreviewed.

Both supplied scripts completed once with exit0:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/prior_sources/nonlinear-heldout/run_heldout.py
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/prior_sources/nonlinear-heldout/summarize_uncertainty.py
```

The first command's log is `prior_sources/nonlinear-heldout/rerun.log`. All10 cells and40,960 history-level predictions were produced. The frozen model hash matched `6b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f`. Required imports are installed NumPy/SciPy, standard-library modules and this unchanged model; no missing execution dependency or installation was needed. The reviewer protocols previously listed as missing are now present. No original repository or supplied source was edited.

## Independent checks and verdict

`independent_check.py` does not import the supplied executable. It reconstructs the encoding polynomial using exact Fraction arithmetic over100-step geometric-sum blocks, independently of the source's per-step polynomial propagation. All coefficients match. Gate gradient f2 is positive, and the four exact Jacobian determinant signs are negative, with values from−4.110453430218917e−6 to−3.885983582347569e−6. Because fixed-A fibers intersect the square in connected intervals, u is strictly monotone on each fiber; this proves global injectivity on the declared fixed-key square, not merely local rank.

The positive-input coordinatewise envelope at(.5,.5) bounds the entire square. An independent conservative block bound gives u<.843193, A<.366606 during encoding and u<.818781 during the noiseless query. These are conservative bounds, distinct from the script's tighter simulated peak. Thus the polynomial branch and no-reset query observation formula apply to the specified domain. The actual sampled noisy trajectories also had no physical branch departures.

The audit independently reconstructed both sampled observations across all4096 histories and five noise paths, reproducing the saved inferred states within3.79e−14 despite using a different arithmetic ordering. It verified target RNG bytes and SHA256, all40,960 ledger rows, every unconditional and conditional coverage/status outcome used here, all bounded MAE/RMSE/Ecomp values, all unbounded linear diagnostic RMSE/Ecomp values, zero fallback for invalid inversions, and every posthoc descriptive interval. The target CSV hash is `da0bb3feaffc8258b50a81099f877d70cd948152fc5beb25099d4f5ca5f30148`.

No material leakage, root-selection, or metric defect was found. The inverse uses estimated state, fixed known coefficients and the predeclared domain; it never selects a root by target proximity. Its1e−10 boundary allowance is a separately reported roundoff band, not broad posthoc clipping. In this held-out run there were no boundary snaps, ambiguities, residual rejections, or physical departures; all invalid cases were NO_REAL_ROOT or OUT_OF_DOMAIN. Conditional metrics remain distinct from full-denominator metrics.

## All-condition results and their meaning

`all10cells.md` contains the complete bounded and unbounded result table. The zero predictor on this same dataset has Ecomp .3342145856 and local .05 joint success .01049805.

| Noise | Ecomp m1 | Ecomp m100 | Coverage m1 | Coverage m100 |
|---|---:|---:|---:|---:|
| Noiseless |1.18e−10|9.09e−11|1|1|
| Sensor only |.3338278513|.2681650329|.01831055|.69213867|
| Membrane process only |.3345681406|.3340755365|.02368164|.17041016|
| Sensor+membrane |.3346673969|.3319820735|.01440430|.16308594|
| Gate process only |1.18e−10|.1173511948|1|.91113281|

The exact bounded inverse succeeds without noise. Deadline sampling substantially improves sensor-noise recovery, but membrane-process scenarios remain around the zero baseline and retain high abstention. Gate-process-only m1 is identical to noiseless for a precise temporal reason: u1 uses the pre-step gate, while the first gate innovation is injected after the drift and therefore cannot affect u1. It can affect later voltage samples. This is not general gate-noise immunity.

The unbounded local linear diagnostic has Ecomp3.765187793 even without noise. Its large off-domain errors therefore reflect finite-domain nonlinear model bias as well as possible noise sensitivity; they are not valid behavioral predictions of the bounded decoder or a pure measurement of noise degradation. It is correctly labeled diagnostic in the supplied protocol and output.

The source implements both m1 and m100 from one fixed100-step driven path. The earlier sample is available at.1ms, the deadline sample at10ms; comparing those observations does not eliminate the latency distinction. The code computes physical invalidity over the full path and applies it to both schedules. This would be a future-path validity condition for an operational early decoder if later departures occurred; none occurred here, so no reported number is affected. The study is a prescribed common-trajectory diagnostic, not a demonstrated online early-stop controller.

## Resource, provenance and inference limits

- The two keys(.8,1.8), positions(0,4), model parameters and exact timing are known to the inverse. It uses two analog membrane observations plus a .1-content probe and fixed .8 query drive. This is conditional two-value decoding, not general variable-key addressing and not the original spike-only readout.
- Encoding is deterministic and starts from known zero state. Noise is inserted only during the100-step query, after drift. Sensor noise affects the two observations; no initial-state/encoding noise is included. Gate/membrane state innovations are not physical current-noise measurements.
- Original tolerance±.5 is trivial on[-.5,.5]². The supplied local .05 valid-joint metric is distinct and does not alter P1B labels.
- The exact monotone-fiber proof is restricted to this fixed-key box and branch. A noisy inferred state outside its image does not establish a fold of the noiseless map.
- `summarize_uncertainty.py` explicitly labels its intervals post-execution, descriptive, approximate normal and unadjusted for multiple comparisons. The audit confirmed those values; they must not be retrospectively described as preregistered hypothesis tests.
- The reviewer protocols clarify that changing tau in the later study alters leak and normalized forcing together. That later study's baseline is the modified content-A carrier, not native M1. The supplied sources support that restricted interpretation.

No remaining blocker for this supplied-source audit. Full archive recovery and review of other unseen experimental stages are separate from this completed task.
