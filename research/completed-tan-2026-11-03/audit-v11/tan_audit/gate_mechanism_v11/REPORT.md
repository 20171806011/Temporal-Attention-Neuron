# Final bounded round: temporal correlation does not establish a TAN advantage

2026-11-02. v11 continuation at source commit `85213f0d417d54e4f48b301ed1c0e4200ef3c421`. Original results, labels, source files and positive P1A evidence are preserved. This report describes the completed bounded comparison.

## Finding and decision

The prespecified correlation-robustness claim is disfavored in this comparison. TAN has fewer false alarms, but more misses and longer response delays than a calibrated ordinary adaptive filter under stationary AR(.9) noise. Its balanced-accuracy advantage on iid noise reverses. The two primary paired-bootstrap intervals are negative. This is an informative limitation of the fixed TAN implementation, not evidence that all ordinary adaptive neurons dominate it.

TAN's useful iid detection and low false-alarm behavior remain positive evidence. They do not establish a unique adaptation mechanism. Ordinary adaptation already separates rapid transients from slow backgrounds; this is established primary literature, not a new TAN discovery. The present baseline is an engineering implementation of input adaptation, not a claimed reproduction of a particular biological neuron. [Benda, Longtin & Maler (2005)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6726095/)

## Frozen question, comparator and controls

`PREREGISTRATION.md` was saved and hashed before any v11 calibration/test inputs or execution: SHA256 `1d4a98c9c4b26583b3ab7d388659c4144c585756deebb29214137fdab600d362`. Earlier `DESIGN_REVIEW.md` and `MECHANISM_PROPOSAL.md` contain prospective alternatives, not additional executed arms. Only the final protocol was executed.

Original Task B TAN uses its unchanged W10 signed-surprise attention, magnitude gate, leak .8, strict threshold1 and zero initialization. The ordinary comparator retains only running mean m and membrane u:

```
m_new = .8*m + .2*x
contrast = (9/8)*(x - m_new)
current = kappa*abs(contrast)
candidate = .8*u + current
spike = candidate > 1
u_new = 0 if spike else candidate
```

The updated mean is always retained, including through spikes. This full-wave comparator is not the repository's spike-triggered AdaptiveLIF. Source inspection shows that repository baseline uses `a_next=.95*a+.285*spike`, with hand-set parameters; before its first spike it is the corresponding ordinary LIF. See `SOURCE_AND_BASELINE_REVIEW.md` and `SOURCE_HASHES.json` for exact boundaries.

The comparator's .8 pole and 9/8 normalization were selected analytically: its clean step contrast starts at .9 times the step and has infinite signed transient area4.5 times the step, matching TAN's W10 deviation filter. Both stationary, unclipped linear contrasts have white-noise variance .9 times input variance. At AR correlation .9, the factors are .42497807238 for TAN's finite-window contrast and .32142857143 for the comparator. These filter facts favor stronger ordinary attenuation of correlated noise, but do not by themselves predict nonlinear detection after clipping, context, reset and threshold.

One independent 512-history iid calibration, seed2026110250, fixed only the gain `kappa=0.840938900477691`. It matches the pooled absolute effective-current sum over both classes and all60 updates: both models total12494.317078649212, recorded residual0 under `math.fsum`. It uses TAN training currents, not accuracy, latency or test outcomes. No parameter search or recalibration occurred. The gain was saved before test inputs were generated.

Test seed2026110251 supplies1024 independent innovation rows, each paired across iid and stationary AR(.9) noise, step/catch, and both models. AR starts with `.4*Z0`, maintaining variance.16 from the first sample; model states still start at zero. Signal amplitude1.5, onset15, duration45 and60-step observation remain fixed. Both noise regimes have expected squared sum9.6; realized means are9.525008 and9.439123. No path rescaling or exclusions. Changed noise correlation is a new stimulus distribution. Four clean traces establish noiseless step detection and silent catches separately.

The fixed decision is any spike at15..19; the readout therefore knows onset. Latency is an update index, with misses assigned capped delay5. It is not a measurement in milliseconds or unknown-onset localization. Pre-onset alarms use0..14 and are counted once per history, separately from catch-window false alarms.

## All detection outcomes

Each row has1024 step trials and1024 paired catch trials; the independent statistical unit is the innovation history, not either counterfactual separately.

| Noise | Model | Misses | Catch false alarms | Balanced accuracy | Mean capped delay | Pre-onset alarms |
|---|---|---:|---:|---:|---:|---:|
| iid | TAN | 0 | 1 | .999512 | .560547 | 2 |
| iid | ADAPT | 0 | 137 | .933105 | .103516 | 218 |
| AR(.9) | TAN | 62 | 6 | .966797 | .930664 | 10 |
| AR(.9) | ADAPT | 7 | 40 | .977051 | .301758 | 111 |

The primary difference TAN−ADAPT in AR balanced accuracy is **−.010254**, with predeclared97.5% paired percentile interval **[−.020020, −.000488]**. The change in this difference from iid to AR is **−.076660**, interval **[−.090820, −.062988]**. Both upper endpoints are below0; neither required positive lower endpoint is present. The descriptive iid difference is+.066406,95% interval[.056152,.076660].

All2000 bootstrap draws resample whole1024-history groups and preserve both regimes/classes/models, seed2026110252. Calibration stays frozen. The two97.5% intervals implement the prespecified conservative reporting rule; this is not an exact finite-sample simultaneous-coverage theorem. The full212-metric output contains pointwise descriptive95% intervals, delay distributions and conditional-hit latency with coverage. All424000 bootstrap metric values and2048000 draw indices are saved. Zero empirical errors and degenerate percentile intervals do not prove population perfection.

The tradeoff matters: in AR noise, TAN detection is93.9453% versus99.3164%, while its catch FPR is.5859% versus3.9063%. Conditional-hit delay is.668399 versus.269420; the unconditional capped comparison above retains misses. These are different operating behaviors, not equivalence or simple total suppression.

## Activity and resource accounting

Both clean step trajectories have original late rateR=.025; clean catches haveR=0. TAN's noisy stepR is.266577 iid and.173584 AR; ADAPT's is.138428 and.048145. Thus the historical late-rate metric can improve while detection also loses signal. Full noisy-minus-own-clean rates, onset/late/total counts, false alarms and delays are saved separately.

Per delivered waveform, balanced across step/catch:

| Noise | Model | Mean sum abs(current) | Mean sum current² | Mean sum candidate² | Generated spikes |
|---|---|---:|---:|---:|---:|
| iid | TAN | 12.216428 | 9.000724 | 20.376428 | 6.825684 |
| iid | ADAPT | 12.213750 | 5.028183 | 26.762662 | 4.730957 |
| AR(.9) | TAN | 9.565168 | 6.117099 | 17.217751 | 5.003906 |
| AR(.9) | ADAPT | 7.947791 | 2.811919 | 17.474741 | 2.365234 |

Exact pooled current-L1 matching holds only on the calibration ensemble. The fresh iid TAN-minus-ADAPT gap is.002678, descriptive95% interval[−.044095,.047997]; this is not an equivalence test. In AR it is1.617376[1.527268,1.712130]. Classwise currents are also unequal. Matching one pooled current moment does not match gain everywhere, current variance, spikes, computation or physical energy.

TAN retains ten input coordinates plus membrane; ADAPT retains two scalars. TAN computes window statistics, attention exponentials and normalization, context and gate; the comparator computes mean update, subtraction, magnitude and scalar gain. Both use the same spike generator and observation window. Candidate/current integrals and state peaks are dimensionless proxies, not ATP or joules. Neither a biological implementation claim nor resource-normalized superiority follows.

## Validation and reproducibility

Independent reconstruction passed 5,264 checks and 1,444 numerical comparisons. All10,244 model trajectories (614,640 updates), calibration, inputs, ledgers,212 metrics and all2000 paired bootstrap rows were verified without importing primary simulator or metric helpers. Maximum numerical residual was6.66134e-16; maximum full-trajectory residual was0, within the declared1e-12 tolerance. Every spike decision matched exactly. Original sources, earlier artifacts and primary outputs remained unchanged. See REVIEW.md, INDEPENDENT_REVIEW_CHECKS.json and numbered attempt records for the full verification history.

A separate isolated replay with no.git reproduced all17,663,090 numeric array elements,1,364 scientific JSON numbers and scientific CSV bytes exactly. Its copied independent review also passed; source hashes remained mandatory while Git provenance was explicitly unavailable. ARCHIVE_REPLAY_REVIEW.json records exact commands and results. This was portability validation of the same frozen experiment, not another research condition.

Completed independent command:

```sh
PYTHONDONTWRITEBYTECODE=1 python tan_audit/gate_mechanism_v11/independent_review.py
```

Independent script SHA256: `fe35c0da97a61b52d3cd991d096d9542ccc27958e369b257d2a2cb2b675d3db3`.

Completed primary command:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tan_audit/gate_mechanism_v11/run_gate_mechanism.py
```

The actual execution ran22:49:11–22:49:31UTC and exited0. Primary script SHA256: `e5435b9f48875dda21d20712d89b2b48fa8620e5b12c4073e592ab5d0544eb87`; test-input SHA256: `0a8824c5eb2f232722349e021c86aa239e0312b03caced61b3bd84450887d65f`. `PRIMARY_MANIFEST.json` records complete file hashes/schemas. Preserve delivered outputs; regeneration requires a fresh `--output-dir`, and review should run in a copied audit tree. The v9 dependency and source hashes are mandatory. `TAN_REPO_ROOT` can select the sibling source snapshot; absent.git must be reported as unavailable provenance.

## Scope

This test rejects the chosen TAN-superiority direction under one fixed correlation change and calibrated comparator. It does not identify whether a particular missed response is caused by window shape, nonlinear gate, attention context or reset. Those components vary together across models. No gate-isolation or architecture-wide theorem follows, and no posthoc retuning is justified by these results. The strong iid false-alarm difference is preserved rather than erased by the negative primary outcome.

The finite-output result from v10 remains external readout analysis shared with LIF. This v11 comparison adds a concrete model-specific robustness limitation; it does not establish generic high-pass filtering as new science. Positive P1A static-address evidence and original P1B failure labels remain intact.

Remaining scientific issues are described below; they are not reported as additional completed experiments.
