# Prospective mechanism check: TAN gate versus ordinary input adaptation

2026-11-02. Source/theory review only. No new model trajectories, nuisance paths, thresholds, or empirical outcomes were evaluated. This note concerns one frozen comparison and its separate computational validation.

## Recommendation

Use one correlation change, iid versus stationary AR(1) noise with rho=.9, keeping the original step, marginal noise SD.4, duration, clipping and initialization. Compare native TAN against one explicitly specified, clean-calibrated leaky subtractive-adaptation current feeding the same spike generator. Do not presuppose TAN superiority: a simple analytically matched leaky filter predicts stronger attenuation of correlated nuisance than TAN's boxcar surprise filter. Measure that intermediate mechanism and the unchanged step-detection outcome separately.

Correlation robustness alone cannot identify attention as the cause. Ordinary spike-frequency adaptation has already been shown experimentally and analytically to separate fast transient signals from slower background fluctuations; parameters were obtained from step responses. That is prior art, not a TAN novelty claim. [Benda, Longtin and Maler (2005), primary electrophysiology/model study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6726095/). Current adaptation and threshold adaptation are also distinct conventional mechanisms, rather than interchangeable generic baselines. [Benda, Maler and Longtin (2010), primary model comparison](https://journals.physiology.org/doi/full/10.1152/jn.00240.2010).

## Exact original mechanism and a non-unique prediction

The frozen W10 TAN uses the updated, current-inclusive history: `S_t=x_t-(1/10)sum_{j=0}^9 x_{t-j}`, query2S, history keys/values, stable softmax denominator epsilon1e-9, and `A_t=tanh(abs(S_t))*C_t`. Its spike generator has leak.8, threshold1, strict crossing and reset0. This is the signed-surprise branch, despite the README's general emphasis on rectification. [Exact source](https://github.com/20171806011/Temporal-Attention-Neuron/blob/85213f0d417d54e4f48b301ed1c0e4200ef3c421/snapshot/code/experiments/statistical_validation.py#L89-L132).

Before the nonlinear gate/context, surprise is an ordinary finite high-pass filter:

`H_box(z)=1-(1/10)sum_{j=0}^9 z^(-j)`.

For an unclipped stationary Gaussian input of variance sigma² and correlation `Cov(n_t,n_{t-d})=sigma²*rho^|d|`,

`Var(S)/sigma² = 1 + [10+2sum_{d=1}^9(10-d)rho^d]/100 - .2sum_{j=0}^9 rho^j`.

At rho0 this is.9; at rho.9 it is.42497807238. Since tanh(abs(S)) increases with the magnitude of zero-mean Gaussian S, a lower Gaussian surprise variance lowers mean gate opening. This explains a plausible correlation response without invoking attention retrieval. It does not guarantee fewer spikes: context, temporal clustering, membrane integration, reset and clipping intervene.

There is an additional exact symmetry in the unclipped stationary constant-baseline setting. Reflect every window residual about its common mean mu: `n -> -n`. Surprise changes sign, but the query-key products change only by a common offset, so the softmax weights are unchanged; the absolute-surprise gate is unchanged. With normalized softmax, reflected contexts sum to2mu and hence `E[A]=mu*E[gate]`. Attention redistributes current across reflected histories but contributes no extra mean term under this symmetry. For the registered epsilon denominator, let M be the sum of weights: the exact statement is `E[A]=mu*E[gate*M]`, with `0<=1-M<=1e-9`. This is not a proof of spike-rate equivalence to uniform averaging.

These formulas are pre-clipping, stationary filter statements. The original catch is heavily clipped, and initial history/state is zero. The step plateau is not fully in the boxcar window until index24. Apply finite-time covariance calculations to the actual initialized linear filters, or clearly label stationary formulas as explanatory limits. Do not silently substitute these expressions for native clipped trajectories or move the historical late-score window20:60.

## A precise ordinary baseline with no hyperparameter sweep

Prefer an input-driven adaptive subtractor over a spike-triggered threshold baseline for this narrow question; the subtractor can adapt before its first spike. The repository already contains a spike-triggered AdaptiveLIF with threshold increment.3 and recovery.05, but that existing operating point is not a calibration to TAN's gate or effective-current budget. [Existing baseline and update order](https://github.com/20171806011/Temporal-Attention-Neuron/blob/85213f0d417d54e4f48b301ed1c0e4200ef3c421/snapshot/code/experiments/baseline_comparison.py#L107-L136).

Recommended new diagnostic baseline:

`m_t=.8*m_(t-1)+.2*x_t`, with `m_initial=0`;

`d_t=(9/8)*(x_t-m_t)`;

`I_AD,t=kappa*max(d_t,0)`;

`u_candidate=.8*u_previous+I_AD,t`, spike iff candidate>1, reset0.

This has two retained analog states, m and u, a positive-part rectifier and a fixed scalar gain. It is a conventional leaky input-adaptation/front-end comparison, not the original registered LIF, not a conductance-based biological neuron, and not an implementation-equivalent TAN. The low-pass state corresponds to a first-order adaptation equation; compact adaptation models with explicit state and controlled-step calibration are standard. [Brette and Gerstner (2005), primary adaptive integrate-and-fire model](https://pubmed.ncbi.nlm.nih.gov/16014787/).

The constants above are analytic, not selected from test results. For a clean step increment Delta, TAN's surprise sequence is `Delta*(.9,.8,...,.1,0,...)`; the baseline d sequence is `.9*Delta*(1,.8,.8²,...)`. Thus both have the same first transient value `.9Delta` and the same infinite step-transient area `4.5Delta`. Both also have stationary white-noise variance `.9sigma²`. This matches the linear surprise filters' scale and two temporal summaries without claiming identical kernels.

Use exactly one clean-only current-budget calibration:

`kappa = sum_t abs(A_TAN,clean,t) / [6.75*(1-.8^45)]`.

The numerator comes from the frozen clean TAN trajectory on the original60-step,1.5-amplitude input; the denominator is the exact finite clean sum of positive d for the45 post-onset steps. Prefer its already audited v9 trajectory, or reproduce it exactly as a declared calibration step before generating the new noisy test batch. Freeze kappa then. This matches one clean effective-current L1 proxy; it does not match every onset spike, noisy current variance, squared current, peaks, membrane energy or physical ATP. Report those remaining differences. If the clean-calibrated baseline misses the fixed detector, retain that outcome; do not retune its threshold, rectify differently or normalize it per noisy trial.

The positive-part convention is a deliberate and declared upward-event baseline. A signed inhibitory subtraction model is also scientifically legitimate but is a different model and cannot be swapped in after outcomes. Native TAN uses absolute surprise and can react to either direction; this asymmetry must remain in the interpretation. No additional baseline family or parameter sweep is proposed.

## A distinguishing analytic prediction that does not favor TAN

For the leaky subtractor's d before rectification/gain,

`Var(d)/sigma²=.9*(1-rho)/(1-.8rho)`.

At rho0 it matches TAN's.9. At rho.9 it is.32142857143, below the boxcar value.42497807238. Thus the ordinary matched adaptive filter predicts stronger suppression of positively correlated nuisance in this intermediate variable. A TAN advantage in final detection would have to arise downstream or from clipping/transients, rather than being assumed from generic adaptation. A shared improvement, or baseline parity/advantage, is a useful result that narrows the distinctive claim. Conversely, agreement with the filter prediction plus similar final performance would locate much of the apparent robustness in ordinary temporal adaptation.

Use fixed stationary AR paths `n_0=sigma*z_0`, `n_t=rho*n_(t-1)+sigma*sqrt(1-rho²)*z_t`; reuse Gaussian innovations across the two fixed rho settings if declared. This preserves each timepoint's noise distribution and therefore expected input-energy proxies after clipping, but not exact energy in each realization. The deterministic step energy remains45*(1.5²). Do not rescale individual paths to equal realized energy, which changes their distribution. Preserve all histories and pair the statistics by innovation row.

## Why not make a gate-time shuffle the main result

A cyclic gate shuffle preserves the gate histogram but generally changes `sum(g_t*C_t)` and its current energy. Rescaling afterward to match each trial's native current imports whole-trial information and produces a retrospective intervention, not an online neuron. Shifting the full current preserves its norm but trivially misaligns any event detector with a fixed observation window. Such interventions may diagnose alignment, but they would not by themselves establish a TAN-specific mechanism. The proposed clean-calibrated correlation contrast is the smaller defensible comparison for this final bounded round.

Freeze the sample, seeds, calibration artifact/hash, correlation settings, ordinary-baseline update order, fixed detector and complete resource/outcome table before execution. Keep prior labels and P1A evidence intact. Stop after the single comparison and separate computational validation, regardless of which model scores better. No count-information rerun or additional research round is recommended here.
