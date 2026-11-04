# Training interactions in native-spike networks under a fixed surrogate learning rule

## V6–V8 methodological archive

Archive date: 4 November 2026. Source snapshot: `9cddbbad7f1722d8a116c0a8245fc40d3a5e97ca`.

**The retained evidence supports a conditional loss-by-learning-rate interaction within the tested POINT network, not a universal numerical law or an architecture-superiority claim.** V8 used 104 fresh paired training/data blocks and found an interaction of −14.105694 percentage points, with the prespecified approximate Student-t 95% interval [−22.603812, −5.607576]. The same implementation, fixed delayed-XOR task, 200-update budget and native count decoder define the scope. This report closes the V6–V8 diagnostic line as a versioned methodological record. It does not introduce another hyperparameter search or new experiment.

This is a retrospective synthesis of three separately frozen studies. Their original protocols, outputs, failures, reports and execution-time status statements remain unchanged. [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) pins 57 canonical source files and the three complete phase manifests to the source snapshot; [ARCHIVE_SUMMARY.json](ARCHIVE_SUMMARY.json) extracts the historical tables and primary intervals without calling a model or fitting anything. Full raw evidence remains in the original V6–V8 directories rather than being duplicated into this archive.

## 1. The three studies must remain distinct

| Study | Architectures | Fresh paired seed/data blocks | Fixed configurations per architecture | Training runs | Main purpose |
|---|---|---:|---|---:|---|
| V6 | DEND and POINT | 20 | RATE/.01 and CE/.03 | 80 | Diagnose stability of two bundled recipes; loss and LR remain confounded |
| V7 | DEND and POINT | 20 | RATE/CE × .01/.03 | 160 | Exploratory factorial separation of the two tested factors |
| V8 | POINT only | 104 | RATE/CE × .01/.03 | 416 | Fresh-cohort test of one prespecified POINT interaction |

Each study gives every run 128 training episodes and 200 full-batch updates, followed by one evaluation on 512 fresh held-out episodes per block. Inputs and initialization are paired across all conditions within a block. Different studies use fresh seed namespaces. There are 656 newly trained trajectories in these three studies, but no analysis pools them into a single sample or treats all models as independent. The V6 post hoc replay of already saved V5 trajectories is not another set of newly trained V6 models.

The correct inferential units are 20, 20 and 104 paired blocks in V6, V7 and V8 respectively. In particular, V6 and V7 were not 104-block POINT-only experiments. V8's 416 runs are four dependent conditions within each of 104 blocks, and its many held-out episodes do not replace independent training blocks.

Sources: [V6 protocol](../../dendritic-population-v6/PROTOCOL.md), [V7 protocol](../../dendritic-population-v7/PROTOCOL.md), [V8 protocol](../../dendritic-population-v8/PROTOCOL.md).

## 2. Native dynamics and learning contract

The model and delayed-XOR generator were unchanged across these studies. DEND has twelve graded dendritic compartments, six spiking somata and two spiking outputs. POINT uses twelve native-spiking branch cells, the same six somata and two outputs. Both retain 176 trainable float64 parameters, twenty continuous state values, six stored previous hidden spikes and two output-count accumulators. The comparison is not matched in represented cell count, scalar nonlinearities, reset-derivative paths, activation patterns, wall time, energy or biological resources.

Every emitted forward spike is hard binary. Hidden/output cells and POINT branch cells fire strictly when the candidate exceeds .5 and reset. Labels enter losses and metrics only. The decoder counts native output spikes at steps 16–23 of the 24-step sequence and requires a unique largest count. Silence and all other ties abstain and count as errors. No analog decoder, label-conditioned output timing or separate pulse-generation module is introduced.

All runs use full-batch Adam with beta1=.9, beta2=.999, epsilon1e-8, global gradient-norm clipping at 1, parameter clipping to [−2,2] and the final checkpoint. The surrogate slope is `1/(1+5*abs(candidate-.5))^2`, including reset paths. This is a specified proxy learning rule, not the exact derivative of the discontinuous hard-spike objective and not a demonstrated biological learning rule.

CE uses .5 times output counts as logits. RATE compares count/8 against .75 for the correct output and .125 for the other. Both add .0001 times mean squared parameters. A loss-factor contrast therefore includes the specified scaling, targets or logits, gradients, clipping and Adam interactions, including the relative influence of regularization. Equal nominal learning rates do not imply equal effective parameter steps. The studies distinguish specified computational interventions; they do not isolate a cellular mechanism responsible for training collapse.

The task is one balanced synthetic delayed-XOR distribution, with two informative four-step cues, onset jitter, amplitude variation, background noise and a common readout cue. Each episode starts from zero state. All conclusions below retain that task, initialization/data distribution, readout and finite training budget.

## 3. V6: bundled-recipe instability, with unresolved primary direction

V6 first inspected the already observed V5 DEND CE/.03 failures. The large failures were already present on training examples. Some output-silent runs retained hidden firing and small but nonzero proxy gradients. Those post hoc descriptions did not prove an absorbing zero-gradient state, establish exact collapse onset or identify a cause. The five V5 development seeds that selected CE/.03 had succeeded, while some independently retrained seeds failed; that is different from a failure appearing only on held-out examples.

In the new V6 cohort, the primary DEND CE/.03-minus-RATE/.01 accuracy contrast was −13.564453 pp, with pointwise seed-pair bootstrap 95% interval [−28.750000, +0.136719]. Its direction remained statistically unresolved under the declared interval procedure. Changing both loss and LR together did not establish a CE-only or LR-only effect. All 80 models had matching train/test pass classifications, but this historical observation was not assumed for later cohorts.

| V6 cell | Mean held-out accuracy | Failed /20 | Fully silent /20 | Fully abstaining /20 |
|---|---:|---:|---:|---:|
| DEND RATE/.01 | 99.75% | 0 | 0 | 0 |
| DEND CE/.03 | 86.18% | 3 | 0 | 2 |
| POINT RATE/.01 | 59.56% | 17 | 0 | 0 |
| POINT CE/.03 | 74.24% | 9 | 2 | 2 |

A failure means accuracy below 90% or coverage below 90%. The two fully abstaining DEND CE/.03 fits produced 8/8 saturated ties, not silence. Thus even the output form of failure differed from some V5 cases. V6's fixed-recipe architecture comparisons were secondary; they did not replace V5's separate equal-budget configuration-search comparison. POINT CE/.03 was not the POINT CE/.003 configuration selected by that V5 search.

Sources: [V6 report](../../dendritic-population-v6/REPORT.md), [V5 post hoc record](../../dendritic-population-v6/POSTHOC_V5.md).

## 4. V7: exploratory factorial evidence

V7 crossed RATE versus CE with .01 versus .03 in each architecture using twenty fresh paired blocks. Three accuracy effects per architecture formed one six-contrast primary family: the equal-weight loss main effect, LR main effect and loss-by-LR interaction. It reported pointwise bootstrap95 intervals plus nominal 99.1667% marginal percentile intervals for a Bonferroni adjustment. Approximate family coverage depends on the bootstrap marginal coverage, which is uncertain with only twenty mixture-distributed block outcomes. The adjustment did not convert the study into confirmation or protect secondary contrasts.

| V7 cell | Mean held-out accuracy | Failed /20 | Fully silent /20 | Fully abstaining /20 |
|---|---:|---:|---:|---:|
| DEND RATE/.01 | 99.94% | 0 | 0 | 0 |
| DEND RATE/.03 | 99.96% | 0 | 0 | 0 |
| DEND CE/.01 | 96.19% | 3 | 0 | 0 |
| DEND CE/.03 | 82.49% | 4 | 2 | 3 |
| POINT RATE/.01 | 71.80% | 15 | 0 | 0 |
| POINT RATE/.03 | 75.00% | 11 | 0 | 0 |
| POINT CE/.01 | 87.73% | 7 | 0 | 0 |
| POINT CE/.03 | 57.10% | 13 | 5 | 5 |

Three of six primary adjusted intervals excluded zero: DEND's loss main effect, POINT's LR main effect and POINT's interaction. POINT's interaction estimate was −33.837891 pp, with pointwise interval [−56.328125, −12.431152] and nominal adjusted interval [−64.482666, −5.843058]. Its marginal loss estimate was near zero, but the two simple-effect point estimates had opposite signs. A near-zero average did not imply interchangeable losses.

All 160 runs were included, with 56 failed final-training fits and 53 failed held-out fits. Five fits changed their pass/fail classification across training and held-out examples. The negative POINT interaction motivated the single new-cohort V8 question; it did not justify selecting a favorable secondary architecture comparison as a new primary result.

Source: [V7 report](../../dendritic-population-v7/REPORT.md).

## 5. V8: one prespecified interaction on a larger fresh cohort

V8 retained only the four POINT configurations. Define each block's held-out accuracy contrast as

`delta = (CE/.03 − CE/.01) − (RATE/.03 − RATE/.01)`.

The fixed count of 104 blocks was chosen before any V8 scientific data from V7's sample interaction SD, 0.5125263. It was the smallest integer yielding a plug-in Student-t planned interval half-width at most 10 pp. The planned half-width was 9.9674 pp; inflating the pilot SD by 1.5 would instead give 14.9510 pp at the same count. This was precision planning, not a guarantee about power, true variance or realized precision. No optional stopping or outcome-based expansion occurred.

The sole primary interval was the two-sided paired-mean Student-t95 interval with df103. A 20,000-resample paired-block percentile bootstrap95 interval was prespecified sensitivity only. The primary method could not be replaced after seeing results. Neither method assumes that the finite mixture of seed outcomes is exactly normal; the primary mean interval has approximate coverage.

| V8 primary quantity | Recorded value |
|---|---:|
| Independent paired blocks | 104 |
| Training runs | 416 |
| Mean interaction | −14.105694 pp |
| Sample SD across block interactions | 43.697729 pp |
| Standard error | 4.284915 pp |
| Primary Student-t95 interval | [−22.603812, −5.607576] pp |
| Realized primary half-width | 8.498118 pp |
| Bootstrap95 sensitivity interval | [−22.562491, −5.842426] pp |

Under the frozen decision rule, the primary interval supports a negative interaction and a magnitude exceeding 5 pp on this exact task and budget. Its upper endpoint is only 0.607576 pp beyond the −5 pp threshold. This narrow margin and approximate coverage must remain visible; the threshold is not a boundary of scientific certainty or a broad robustness guarantee.

| V8 POINT cell | Mean held-out accuracy | Failed /104 | Fully silent /104 | Fully abstaining /104 |
|---|---:|---:|---:|---:|
| RATE/.01 | 66.06% | 87 | 0 | 1 |
| RATE/.03 | 68.13% | 79 | 0 | 1 |
| CE/.01 | 85.16% | 38 | 2 | 2 |
| CE/.03 | 73.13% | 47 | 5 | 7 |

V8 retained all 416 runs, including 250 failed final-training fits and 251 failed held-out fits. Six fits moved from train-fail to test-pass and seven in the opposite direction. Some fully abstaining RATE fits produced intermediate equal counts, rather than silence or 8/8 saturation. These output modes cannot be collapsed into one explanation of failure.

The CE-minus-RATE simple-effect point estimate was +19.101187 pp at .01 and +4.995493 pp at .03. Increasing LR gave +2.071439 pp under RATE and −12.034255 pp under CE. Their difference produces the negative interaction. The secondary CE-minus-RATE interval at .03 crossed zero. V7's opposite-sign simple-effect point estimates therefore did not recur, even though the negative interaction direction did. An interaction does not require a crossover in average ranking.

V8's interaction estimate was less negative than V7's −33.837891 pp. No formal between-cohort comparison was prespecified, and no V7 observations were pooled into V8. The result supports direction under the primary interval rule, not an unchanged numerical magnitude across cohorts. The secondary failed-run interaction was +16.35 pp with an interval crossing zero; the thresholded failure-risk result does not replace the unconditional-accuracy primary.

Sources: [V8 report](../../dendritic-population-v8/REPORT.md), [sample-size planning](../../dendritic-population-v8/PLANNING.json), [complete V8 outcomes](../../dendritic-population-v8/TABLES.md).

## 6. What this archive can answer about training controls

The archive documents fixed protocols, paired data and initialization, final-checkpoint use, separate train/test RNG streams, one-shot held-out evaluation, explicit failed-run denominators, full gradient/checkpoint preservation and uncertainty at the independent training-block level. It supports the methodological statement that the tested loss and learning-rate effects cannot be read independently of their other setting.

It does not certify that every later architecture comparison is fairly tuned. The grid contains only two learning rates and two fixed loss definitions. V8 is POINT-only; it cannot serve as a new DEND comparison, and it cannot be paired retrospectively with old DEND cohorts to create one. The earlier V5 independently selected comparison remains a separate result whose main direction was uncertain. POINT CE/.003, selected in V5, is absent from V6–V8's fixed grid. Neither a protected primary conclusion about V8 nor extensive numerical checks remove these design limitations.

This report can supply a precise account of what was controlled and what was not in response to methodological questions. It is not a blanket rebuttal of hyperparameter-choice objections, an exhaustive search, or evidence that all earlier apparent structure differences were caused by optimization. No result is promoted into a universal numerical law.

## 7. Verification and reproducibility record

| Study | Reconstructed Adam updates | Full held-out model-episodes replayed | Sampled full-batch gradient updates | Final local proxy derivatives | Maximum local proxy residual |
|---|---:|---:|---:|---:|---:|
| V6 | 16,000 | 40,960 | 240 | 14,080 | 5.8394e−10 |
| V7 | 32,000 | 81,920 | 480 | 28,160 | 3.7175e−10 |
| V8 | 83,200 | 212,992 | 1,248 | 73,216 | 7.0411e−10 |

Separately written native/data/optimizer/statistics checkers passed 107,001, 202,055 and 555,877 internal checks respectively. Full saved-trajectory comparisons matched hard spikes, counts and predictions exactly. Each study reconstructed all saved-gradient Adam updates and fixed checkpoint vectors, independently regenerated datasets and initialization, and recomputed its reported paired statistics. The sampled full-batch gradient comparisons had zero residual.

Final finite differences used all 176 parameters at each final checkpoint on the first two saved training episodes. They validate a locally anchored differentiable proxy graph, not the exact derivative of the hard-spike objective, and they do not cover every training episode. Sampled gradient replay reuses the frozen analytic backward implementation. The separate native checker does not independently derive every proxy gradient or replay native trajectories at every unsampled training update.

These are internal checks in the same AI-assisted workflow. They are not external independent replication or peer review and do not remove statistical or causal uncertainty. File hashes establish identity and preservation, not validity. This archival synthesis only read source artifacts and generated documentation; it did not rerun experiments or numerical audits.

| Study | Protocol/source gate | All weights before test gate | Completed study |
|---|---|---|---|
| V6 | `26d85e9d6176f7b0d335d55527cd0c8049f55a75` | `0cb7d16aad5fe897406110fc5c79fd483347aab3` | `75a595f3c0c5aa3f50d7700b8c142906aca442b0` |
| V7 | `8ad0b69e65074563929dff905acc0067837e3778` | `041693b7297b888ed31ac78f8d5677408669f1fe` | `0c74816da453003490f313c8ee7939e16840e4f2` |
| V8 | `cd5cd4ef1cc65fd098b3a73d36d68e7a49307d17` | `3ec9ce117ad3ce1a20d6e5ef4f2395916f120821` | `9cddbbad7f1722d8a116c0a8245fc40d3a5e97ca` |

The three complete phase manifests remain the entry points to all raw datasets, native traces, weights, clipped gradients and candidate diagnostics. Selected sources in this archive are verified byte-for-byte against the pinned completed-V8 git snapshot. Reproduction or audit scripts may overwrite their logs; inspect saved outputs first and use a separate checkout for any explicitly authorized rerun.

## 8. Closure, limits and attribution

The V6–V8 loss/LR diagnostic line is now closed as a separate methodological archive. No more loss/LR sweep is implied or launched by this archive. The user's subsequent authorization returns the research focus to dendritic and population organization under separate future protocols. That authorization does not rewrite V8's historically accurate pause record or any frozen study file.

Open architectural questions remain open: these studies do not establish cross-cell recurrence necessity, a single-cell impossibility theorem, a distinctive dendritic representational separation, biological learning, hardware efficiency or metabolic savings. Silence, saturated ties and intermediate ties describe outcomes, not a causal decomposition of their origin. Native output counts and state-change measures cannot be converted to energy without an explicit validated physical cost model.

Li Zexu proposed the research direction and native-firing constraint. Protocol development, implementation, execution, numerical verification and reporting were AI-assisted. This document is an archival synthesis with no new experiments and no external submission or publication. Scientific claims remain bounded by the original protocols and evidence rather than by the act of archiving them.
