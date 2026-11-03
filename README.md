# Temporal Attention Neuron (TAN)

**History sensitivity, static addressing, and the limits of recoverable memory**

TAN studies what a dynamical unit can compute when leaky membrane dynamics receive a surprise-gated, attention-weighted summary of recent inputs. The project now includes model implementations, analytical results, controlled experiments, negative results, and a traceable scientific archive.

**Author:** Li Zexu. Original model and research programme by the author; subsequent analysis, implementation, computational checks, and manuscript preparation include AI assistance.

**Latest update — 3 November 2026:** the evidence synthesis now includes a verified **clean-only response collision**, alongside static-addressing positives, a controlled training-composition improvement, and a comparison with an ordinary adaptive baseline. These results distinguish information present in internal state from information recoverable through the task's permitted output.

**最新成果：** 无干扰输入也可出现跨类别响应碰撞，其中一组不需要 reset；改变训练组成可改善固定读出的 clean 表现，但不能消除这种精确碰撞。静态寻址、动态存储和任务解码是不同能力，不能互相替代。

## Start here

- **[Latest research paper — PDF][paper-pdf]**: *Temporal Attention Neurons: Separating History Sensitivity, Addressing, and Recoverable Memory* (10-page research draft).
- **[Paper source, build instructions, and evidence index][paper-home]**: Markdown, generated LaTeX, and claim-to-source links.
- **[Clean-only collision supplement][clean-map]**: theory, saved results, new verification code, and scope.
- **[Completed research snapshot][snapshot]**: all 680 screened scientific files, with an inventory and checksums.

The latest paper and archive are published on research branches. The links above pin exact commits, so they remain usable from this README without assuming those files have been merged into `main`. The manuscripts are research drafts; repository publication and computational checking are not peer review.

## What the current evidence establishes

| Result | Evidence | What it means—and its boundary |
|:--|:--|:--|
| **Static metric addressing works** | In the noiseless three-key P1A Rev2 condition, metric and vector-QK methods each identify **6,705/6,705** intermediate-key queries; the signed scalar-product method identifies **0/6,705**. | A scalar distance score can address an available key–value array. It does not implement dynamic writing or compressed storage. Noisy conditions have lower success. [P1A review][p1a-review] |
| **Clean-only response collisions exist** | Four exact-real collision brackets are verified separately for historical and stabilized softmax. A cross-label pair has no target/query reset. The new checker passes **5,482 checks**, including 64 fixed trajectories / 896 updates. | No decoder of the stipulated response can classify **every** allowed clean input correctly. This is not a one-third average-accuracy ceiling or a claim about arbitrary neuronal architectures. [Proof and verification][clean-map] |
| **Training composition contributes to weak clean scores** | Three fixed historical seeds, six fits: mean balanced accuracy rises from **32.5881% to 38.1199%**, a **5.5318 percentage-point** gain. Conditional paired 95% interval: **[3.3391, 7.6080] points**. | Improvement concerns a fixed linear readout, including train-only normalization. It remains modest, is uneven across classes, and does not repair exact response collisions. [Training control][training] |
| **Stored content can remain inaccessible** | In the separate distractor-swap audit, **545/545** native TAN pairs have identical complete permitted responses despite different target labels and ordered windows. | The collision concerns the output contract, not equality of complete internal state. The constructed family is not an estimate of population failure frequency. [Native response audit][response-audit] |
| **The adaptive-baseline comparison reveals a tradeoff** | Under independent noise, TAN/adaptive balanced accuracy is **99.9512% / 93.3105%**; under AR(0.9), **96.6797% / 97.7051%**. | TAN has fewer false alarms but more misses and longer delay under correlated noise. Calibration matches one pooled current statistic, not total resources or energy. [v11 comparison][v11] |
| **Modified dynamics can rescue a restricted task** | In the eight-stage series, a gate-timescale change reduces sensor-only composition error from **0.270803 to 0.008093** on 4,096 held-out histories. | The successful system also uses modified gate drive, known keys/timing, two analog samples, a bounded target domain, and a known-equation inverse. It is not original P1B success or general retrieval. [Eight-stage study][eight-stage] |

### Why the clean-only result matters

For the fixed W5 task, the only nonzero inputs are target `x5=a` and independent cue `x9=b`. The decoder observes five **pre-reset** candidate voltages and five spike indicators, not the raw cue or window. Its entire response factors through `T=u9`:

```text
F(a,b) = (T, R(T)/2, R(T)/4, R(T)/8, R(T)/16, 1[T>1/2], 0, 0, 0, 0)
R(z)   = z if z <= 1/2, otherwise 0
```

A low-class target `a=0.75, b=1` and a high-class target `a=1.0625` with a cue near `0.9406837741` can produce the same complete response, without either target/query reset. The targets lie outside the classes' overlapping amplitude range. Equality is proved at an **IVT-defined real root**, not inferred from rounding two floating-point outputs. The cue varies across the pair, as the task permits; supplying the exact cue separately adds an observation resource.

Native exponential attention is sensitive to higher moments of the window distribution, but pooling can still lose target position. Higher-moment sensitivity is an analytical property, not by itself evidence of better task performance. Likewise, membrane alone can be an insufficient state description while the full `(membrane, ordered window)` system remains Markovian.

## Research lineage and earlier papers

| Collection | Role | Entry point |
|:--|:--|:--|
| Initial TAN model and behavioural studies | Surprise gating, temporal weighting, habituation and simulated navigation; inspect baseline limitations before interpreting advantages. | [Original package](snapshot/README.md), [model code](snapshot/code/model/) |
| TAN-I | State sufficiency, response geometry, distractor retrieval and scalar-kernel routing limits. | [PDF](release/TAN_I_Mechanistic_Analysis.pdf), [LaTeX](snapshot/paper/) |
| TAN-II and sprint studies | Constructive routing/binding/composition mechanisms with explicitly added resources. Sufficiency does not establish necessity. | [PDF](release/TAN_II_Minimal_Architectural_Ladder.pdf), [LaTeX](snapshot/paper2/) |
| P1A Rev2 | Authoritative static-address experiment; use Rev2 rather than the superseded first run. | [Results](p1_experiments/results/run_p1a_rev2/), [baseline manifest](p1_experiments/P1A_FINAL_BASELINE_MANIFEST.md) |
| P1B | Frozen carrier/readout study. All registered carrier cells are `FLOOR_LIMITED_COMPARISON`; memory redundancy is not established. | [Closure report](p1_experiments/p1b/P1B_FINAL_SCIENTIFIC_CLOSURE_REPORT.md) |
| November research archive | Native audits, modified-carrier controls, exploratory constructions, and clean-training results. | [680-file snapshot][snapshot] |
| Current synthesis and clean-map supplement | Integrates the results while separating model variants, tasks, and observation contracts. | [Latest paper and evidence][paper-home] |

Earlier documents retain their historical wording and conclusions. The current synthesis explains later qualifications; it does not turn repeated reporting of the same experiment into additional evidence. The root [research-state record](CURRENT_RESEARCH_STATE.md), [chronology](RESEARCH_CHRONOLOGY.md), and [errata](ERRATA.md) remain useful historical records, not automatically the latest status of every study.

## Evidence, verification, and reproducibility

The screened snapshot contains **680 scientific files** totaling **383,445,961 uncompressed bytes**. Its exact ZIP is **189,164,458 bytes**, stored as three ordered parts with reconstruction instructions in the [archive directory][archive].

```text
ZIP SHA-256:
ca8a2ae51b4e8f18ab341fa483ca9970a7a3077c5d4e4013e5df13a9811b6167
```

The later clean-map supplement is a **separate archive**, not one of those 680 files. Its provenance and selected source hashes are published with the supplement. A matching hash establishes file identity; it does not validate the experiment or prove a theorem.

Evidence levels are stated explicitly:

- **Analytical results** apply to specified equations, supports, resets, precision conventions, and readouts.
- **Executed local studies** retain inputs, outputs, protocols, and source provenance. Separately written checkers strengthen computational confidence, but are not independent human replication.
- **Exploratory Chat-sandbox studies** concern changed conceptual architectures. Same-agent reruns are not external replication or proof of original TAN performance.
- **Publication checks** verify source identity, selected arithmetic, and document builds. They do not rerun the entire research programme.

### Inspect and check the current publication

The following selects the exact paper revision used by this README:

```bash
git clone https://github.com/20171806011/Temporal-Attention-Neuron.git
cd Temporal-Attention-Neuron
git fetch origin research/tan-evidence-paper-2026-11-03
git switch --detach 007d0b4ebe8208aa95e8a70c1175b9bdc2c8373f

python3 papers/tan-evidence-2026/clean-map/verify_independent.py
python3 papers/tan-evidence-2026/validate_evidence.py
```

These two checkers use the Python standard library. The first recomputes exact collision brackets and fixed scalar trajectories; the second verifies the snapshot, source hashes, and selected saved-result arithmetic. At the linked revision they pass **5,482** and **774** checks respectively. They do not fit a classifier or estimate new task accuracy.

To rebuild the paper, install Pandoc and a LaTeX distribution, then run:

```bash
bash papers/tan-evidence-2026/build.sh
```

For historical experiments, consult each package's protocol and dependency record. The root [requirements.txt](requirements.txt) is a broad environment description, not a lockfile for every archived study. Different studies record different Python/NumPy/SciPy versions. Historical and current public-file manifests have different purposes; publication edits and some historical hash guards require the documented replay procedure. Run experiments in a disposable copy because some scripts overwrite evidence beside their source. See the [publication and reproduction notes][publication-notes].

## Repository map

| Path | Contents |
|:--|:--|
| `snapshot/code/`, `snapshot/data/`, `snapshot/results/` | Earlier implementations, data, figures, and results |
| `snapshot/manuscript/`, `snapshot/paper/`, `snapshot/paper2/` | Historical manuscript sources |
| `p1_experiments/` | P1A/P1B protocols, implementations, fixtures, and saved results |
| `s41cd/` through `s43a/` | Sprint studies and computational checks |
| `release/` | Earlier compiled manuscript releases |
| `scripts/` | Repository audit and verification utilities |
| [`research/completed-tan-2026-11-03/`][snapshot] | Screened 680-file publication on the research branch |
| [`papers/tan-evidence-2026/`][paper-home] | Current synthesis, source ledger, validation, and clean-map supplement on the research branch |

## Attribution and scope

The original TAN model and research programme are credited to **Li Zexu**. AI-assisted work includes later analysis, experiment implementation, computational checking, and writing; it should not be described as independent human peer review. The current results do not establish universal task superiority, biological implementation, physical energy savings, or a general impossibility theorem for single neurons. Proposed architectures remain proposals unless their implementation and results are supplied.

Code is distributed under the [MIT License](LICENSE); retain applicable source notices. For scientific attribution, cite the relevant manuscript revision and exact evidence commit, and distinguish original results from later checks. No DOI or journal acceptance is asserted here.

[paper-pdf]: https://github.com/20171806011/Temporal-Attention-Neuron/blob/007d0b4ebe8208aa95e8a70c1175b9bdc2c8373f/papers/tan-evidence-2026/manuscript.pdf
[paper-home]: https://github.com/20171806011/Temporal-Attention-Neuron/tree/007d0b4ebe8208aa95e8a70c1175b9bdc2c8373f/papers/tan-evidence-2026
[clean-map]: https://github.com/20171806011/Temporal-Attention-Neuron/tree/007d0b4ebe8208aa95e8a70c1175b9bdc2c8373f/papers/tan-evidence-2026/clean-map
[snapshot]: https://github.com/20171806011/Temporal-Attention-Neuron/tree/222caa0201feff1e37a06049aa44c0e59e1cbaba/research/completed-tan-2026-11-03
[archive]: https://github.com/20171806011/Temporal-Attention-Neuron/tree/222caa0201feff1e37a06049aa44c0e59e1cbaba/research/completed-tan-2026-11-03/archive
[p1a-review]: https://github.com/20171806011/Temporal-Attention-Neuron/blob/222caa0201feff1e37a06049aa44c0e59e1cbaba/research/completed-tan-2026-11-03/audit-v11/tan_audit/project_reassessment_v8/P1A_EVIDENCE_REVIEW.md
[training]: https://github.com/20171806011/Temporal-Attention-Neuron/blob/222caa0201feff1e37a06049aa44c0e59e1cbaba/research/completed-tan-2026-11-03/local-clean-training-v1/REPORT.md
[response-audit]: https://github.com/20171806011/Temporal-Attention-Neuron/blob/222caa0201feff1e37a06049aa44c0e59e1cbaba/research/completed-tan-2026-11-03/audit-v11/tan_audit/project_reassessment_v8/REPORT.md
[v11]: https://github.com/20171806011/Temporal-Attention-Neuron/blob/222caa0201feff1e37a06049aa44c0e59e1cbaba/research/completed-tan-2026-11-03/audit-v11/tan_audit/gate_mechanism_v11/REPORT.md
[eight-stage]: https://github.com/20171806011/Temporal-Attention-Neuron/blob/222caa0201feff1e37a06049aa44c0e59e1cbaba/research/completed-tan-2026-11-03/native-eight-stage/EXPERIMENT_SERIES_SUMMARY.md
[publication-notes]: https://github.com/20171806011/Temporal-Attention-Neuron/blob/222caa0201feff1e37a06049aa44c0e59e1cbaba/research/completed-tan-2026-11-03/audit-v11/PUBLICATION_NOTES.md
