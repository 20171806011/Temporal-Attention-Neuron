# Temporal Attention Neuron (TAN)

**History sensitivity, recoverable memory, and computation in native-spike populations**

This repository contains two related research lines: the original Temporal Attention Neuron and its response/readout limits, and separately specified dendritic and point-neuron populations studied under native hard-spike dynamics. It preserves analytical results, controlled experiments, failed runs, complete numerical artifacts and manuscript sources.

**Author:** Li Zexu. Research direction and native-firing constraints were proposed by the author; later protocol formalization, implementation, execution, numerical checks and writing include AI assistance.

**Research edition — 4 November 2026:** the collection now extends through population V10A. The new V6–V10 paper brings together training-configuration interactions, recurrence interventions and network-size comparisons. These are research preprints, not peer-reviewed publications.

**中文导读：** V8 在 104 个全新区组中重现了 POINT 的负损失×学习率交互；V9 找到了不使用跨细胞循环也能通过 delayed XOR 门槛的 DEND 群体解；V10A 观察到固定训练条件下的规模差异，但最小 POINT 网络仍有通过实例。结果不支持一般性的结构排序、不可跨越的表征鸿沟或物理能耗优势。

## Start here

**November 2026 update.** The [DEND+TAN research release](research/dend-tan-2026-11/) records the state-intervention, learned key--value memory, event replay, and evicted-key studies, including negative and stopped results. The [new manuscript](papers/dend-tan-event-replay-2026/manuscript.pdf) reports two fresh confirmation cohorts and provides [editable source and build instructions](papers/dend-tan-event-replay-2026/).

| Read | Purpose |
|:--|:--|
| **[V6–V10 population paper](papers/native-spike-populations-2026/README.md)** · [PDF](papers/native-spike-populations-2026/manuscript.pdf) · [source](papers/native-spike-populations-2026/manuscript.md) | *Training Interactions, Recurrence Dependence, and Size Tradeoffs in Native-Spike Dendritic and Point Networks* |
| **[Original TAN evidence paper](papers/tan-evidence-2026/README.md)** · [PDF](papers/tan-evidence-2026/manuscript.pdf) · [source](papers/tan-evidence-2026/manuscript.md) | *Temporal Attention Neurons: Separating History Sensitivity, Addressing, and Recoverable Memory* |
| **[Complete research index: V1–V10 and earlier work](research/README.md)** | Protocols, code, datasets, weights, all candidates, gradients, per-seed results and manifests |
| **[中文 V9/V10 结果摘要](research/closures-2026-11-04/RESULTS_SUMMARY.md)** · [two-line archive](research/closures-2026-11-04/README.md) | Latest bounded architecture studies and separate preservation of the mainline and training-diagnostic evidence |
| **[Clean-only collision supplement](papers/tan-evidence-2026/clean-map/)** · [680-file scientific snapshot](research/completed-tan-2026-11-03/README.md) | Mainline proof details, retained experimental sources and archive reconstruction |

The population manuscript synthesizes **716 new training runs across V6–V10**, plus ten derived V9 inference-cut models. These are five separately frozen studies, not 716 independent observations of one pooled experiment. No new training is performed to prepare the publication.

## What V6–V10 found

Effect sizes below use percentage points (pp).

Both population architectures retain hard binary forward spikes, strict threshold/reset rules and a native output-count decoder. Ties, including silence, abstain and count as errors. Proxy gradients are used for learning; they are not exact derivatives of hard spikes. The population circuits are distinct from the original single-unit TAN model.

| Study | Design | Result and boundary |
|:--|:--|:--|
| [V6](research/dendritic-population-v6/REPORT.md) | 80 fits; 20 paired seed/data blocks; two recipes in each architecture | DEND CE/.03 minus RATE/.01 accuracy was −13.56 pp, pointwise 95% interval [−28.75, +0.14]. The primary direction remained unresolved, and loss and learning rate changed together. |
| [V7](research/dendritic-population-v7/REPORT.md) | 160 fits; 20 fresh blocks; loss × learning rate factorial in both architectures | Identified an exploratory POINT interaction and motivated a new-data test. Fixed-recipe architecture contrasts do not replace an independently tuned comparison. |
| [V8](research/dendritic-population-v8/REPORT.md) | 416 fits; 104 fresh blocks; four POINT configurations | The prespecified interaction was **−14.11 pp**, primary approximate Student-t 95% interval **[−22.60, −5.61]**, on delayed XOR with 200 updates. The negative interaction recurred; V7's crossover of simple-effect point estimates did not. There was no DEND arm. |
| [V9](research/dendritic-population-v9/REPORT.md) | 20 fits plus 10 inference cuts; 5 fresh paired blocks | DEND passed in **4/5 intact, 5/5 retrained no-cross, and 0/5 inference-cut** models. Mean accuracies were **90.35%, 98.87%, 55.90%**. Trained pathway dependence did not imply necessity for passing this task gate. The no-cross model remains multicellular. |
| [V10A](research/dendritic-population-v10/REPORT.md) | 40 fits; 5 fresh paired blocks; hidden-soma counts 6, 4, 2, 1 | DEND had higher mean balanced accuracy at all four sizes under RATE/.01 and 200 updates. Four paired intervals were pointwise exploratory and unadjusted. A minimum-size POINT model still passed, precluding a claim that this size cannot meet the task gate. |

V9 POINT models passed 0/5 in every condition; failure of the intact models also prevents a stable-intact-success argument for recurrence necessity. Retraining removes 60 effective parameters, leaving 116 versus 176 intact, so it changes capacity as well as connectivity.

V10A uses all 16 latent patterns for `(b0 AND b1) OR (b2 AND b3)`, with 7 positive and 9 negative patterns. Its main metric is balanced accuracy, and its held-out examples are new noisy amplitudes of the same patterns.

| Hidden somata h | Parameters per model | DEND mean balanced accuracy | POINT mean balanced accuracy | Passing DEND / POINT models, each out of 5 |
|---:|---:|---:|---:|---:|
| 6 | 176 | 99.57% | 89.76% | 5 / 3 |
| 4 | 102 | 97.04% | 84.24% | 5 / 1 |
| 2 | 44 | 88.50% | 68.92% | 3 / 0 |
| 1 | 21 | 88.34% | 71.60% | 2 / 1 |

The pass gate requires balanced accuracy and coverage both at least 90%. Here h counts hidden somata, **not total neurons**: at h=1, POINT has five spiking cells; DEND has three spiking cells and two graded compartments. The size comparisons do not isolate representational capacity, a unique dendritic mechanism or energy efficiency. [All seeds and patterns](research/dendritic-population-v10/TABLES.md) · [resource and activity accounting](research/dendritic-population-v10/RESOURCE_COST.md).

The earlier [V5 independently selected-config comparison](research/dendritic-population-v5/REPORT.md) remains relevant: its primary DEND-minus-POINT estimate was +19.20 pp with 95% seed-pair interval [−11.39, +49.41] across ten pairs, and its stability criterion failed. V5 selected POINT CE/.003; V6–V8 did not test that configuration. Neither a zero-crossing interval nor a shared optimizer recipe establishes equivalence or equal optimization quality.

## Original TAN: retained findings and limits

The single-unit mainline separates sensitivity to history, addressing an available array, retaining associations and recovering information through a specified observation. Later population results do not change those original model/readout contracts.

| Finding | Retained evidence and scope |
|:--|:--|
| Static metric addressing | In noiseless P1A Rev2, metric and vector-QK methods each identify 6,705/6,705 intermediate-key queries; signed scalar-product identifies 0/6,705. This is routing over an available array, not dynamic writing or compressed storage. [Review](research/completed-tan-2026-11-03/audit-v11/tan_audit/project_reassessment_v8/P1A_EVIDENCE_REVIEW.md) |
| Clean-only response collisions | Exact-real cross-label collisions exist under the stipulated response contract, including a pair without target/query reset. The checker records 5,482 checks. This obstructs correctness on every allowed input, not a universal average-accuracy ceiling. [Proof and checks](papers/tan-evidence-2026/clean-map/) |
| Training composition matters | Three fixed historical seeds, six fits: balanced accuracy changes from 32.5881% to 38.1199%, a +5.5318 pp effect with conditional paired 95% interval [3.3391, 7.6080]. This fixed-readout improvement does not repair exact collisions. [Control](research/completed-tan-2026-11-03/local-clean-training-v1/REPORT.md) |
| Permitted responses can hide stored content | A constructed distractor-swap audit has 545/545 identical permitted responses despite different targets and ordered windows. The complete internal states need not be identical; this family does not estimate population failure frequency. [Audit](research/completed-tan-2026-11-03/audit-v11/tan_audit/project_reassessment_v8/REPORT.md) |
| Noise changes the detection tradeoff | TAN/adaptive balanced accuracy is 99.9512%/93.3105% under independent noise and 96.6797%/97.7051% under AR(0.9). The comparator is a specified engineered adaptation baseline, not proof of biological equivalence or equal energy. [Comparison](research/completed-tan-2026-11-03/audit-v11/tan_audit/gate_mechanism_v11/REPORT.md) |
| Modified models can solve restricted tasks | The native eight-stage study's improved composition recovery uses changed dynamics, known keys/timing, two analog samples and a known-equation inverse. It is not success of the original carrier or the native-count population decoder. [Study](research/completed-tan-2026-11-03/native-eight-stage/EXPERIMENT_SERIES_SUMMARY.md) |

Read the [mainline claim audit](research/closures-2026-11-04/line-a/CLAIM_AUDIT.md) and [boundary supplement](research/closures-2026-11-04/line-a/BOUNDARY_SUPPLEMENT.md) with the preserved manuscript. In particular, the fixed-readout floors 25/54 and 3139/32000 are bounds on specified expected composition error, not general classification-error rates. Native attention's higher-moment sensitivity does not itself establish a task advantage.

## Earlier work and repository map

| Location | Contents |
|:--|:--|
| [Original package](snapshot/README.md), `snapshot/code/`, `snapshot/data/`, `snapshot/results/` | Initial TAN implementations and behavioural studies |
| [TAN-I PDF](release/TAN_I_Mechanistic_Analysis.pdf), [source](snapshot/paper/) | State sufficiency, response geometry and routing limits |
| [TAN-II PDF](release/TAN_II_Minimal_Architectural_Ladder.pdf), [source](snapshot/paper2/) | Constructive architectural mechanisms with explicitly added resources |
| [P1A/P1B](p1_experiments/), `s41cd/` through `s43a/` | Frozen protocols, carrier/readout boundaries, sprint studies and checks |
| [Screened 680-file snapshot](research/completed-tan-2026-11-03/README.md) | Earlier scientific archive, inventory and complete ZIP parts |
| [Population V1–V10 index](research/README.md) | Every phase's protocol, source, data, weights, failures and integrity manifest |
| [Mainline archive](research/closures-2026-11-04/line-a/README.md), [V6–V8 diagnostic archive](research/closures-2026-11-04/line-b/README.md) | Separate versioned evidence lines with explicit claim boundaries |
| [V6–V10 publication inventory](research/PUBLICATION_V6_V10.json) | Source revisions, retained artifact identities and publication checks |

The [research chronology](RESEARCH_CHRONOLOGY.md), [earlier state record](CURRENT_RESEARCH_STATE.md) and [errata](ERRATA.md) are preserved historical documents. Read the current research index for later status. Original reports may say “local only”, “not pushed” or “paused”: those describe their execution-time state, and are not silently rewritten during publication.

## Obtain and inspect the evidence

```bash
git clone https://github.com/20171806011/Temporal-Attention-Neuron.git
cd Temporal-Attention-Neuron
```

Read the [population paper guide](papers/native-spike-populations-2026/README.md) and [research index](research/README.md) before running anything. The repository includes complete data and numerical traces and therefore requires several gigabytes. Frozen manifests identify exact files; matching hashes establish identity, not scientific validity.

The [680-file archive](research/completed-tan-2026-11-03/archive/) contains three ordered parts of the original 189,164,458-byte ZIP, SHA-256 `ca8a2ae51b4e8f18ab341fa483ca9970a7a3077c5d4e4013e5df13a9811b6167`. Its uncompressed scientific contents total 383,445,961 bytes. The later clean-map supplement is separate from those 680 files.

Publication scripts inspect retained evidence and build manuscripts; they do not rerun the research programme. Some original experiment and audit scripts write beside their inputs, so reproduce them in a separate copy under the phase's own dependency and commit-gate instructions. The root [requirements.txt](requirements.txt) is not a universal lockfile. See each phase's protocol, environment record and manifest.

## Attribution and scientific status

The original TAN model and research direction are credited to **Li Zexu**. AI assistance includes later protocol development, implementation, execution, numerical verification and prose. The separately written checkers are part of the same workflow; they are not external independent replication or peer review.

No universal task superiority, biological realization, physical energy savings or broad impossibility theorem is established. Graded state updates are not free, and spike counts/state changes are computational proxies rather than joules or ATP. No DOI, journal submission or acceptance is asserted.

Code is distributed under the [MIT License](LICENSE); retain source notices. Cite the relevant manuscript version and evidence commit, and distinguish original results from later checks and synthesis.
