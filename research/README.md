# TAN complete research index — 4 November 2026

This edition collects the original TAN programme, the screened 680-file snapshot, the mainline evidence manuscript and clean-map supplement, and **dendritic-population V1–V10**. The new [V6–V10 population manuscript](../papers/native-spike-populations-2026/README.md) synthesizes separately frozen studies without combining their cohorts into one statistical sample.

**中文入口：** [新论文 PDF](../papers/native-spike-populations-2026/manuscript.pdf) · [V9/V10 中文结果摘要](closures-2026-11-04/RESULTS_SUMMARY.md) · [原有主线与诊断线封存](closures-2026-11-04/README.md)。阶段原文、代码、完整数据及哈希清单保留；本次更新入口和发表说明。

## Manuscripts and preserved collections

| Collection | Entry points | Preserved source revision |
|---|---|---|
| Original TAN, TAN-I, TAN-II, P1A/P1B and sprint studies | [Root guide](../README.md), [historical chronology](../RESEARCH_CHRONOLOGY.md), [earlier papers](../release/) | `85213f0d417d54e4f48b301ed1c0e4200ef3c421` |
| Screened 680-file snapshot | [Guide](completed-tan-2026-11-03/README.md), [inventory](completed-tan-2026-11-03/SNAPSHOT_MANIFEST.json), [complete ZIP parts](completed-tan-2026-11-03/archive/) | `222caa0201feff1e37a06049aa44c0e59e1cbaba` |
| Mainline evidence paper and clean-map supplement | [PDF](../papers/tan-evidence-2026/manuscript.pdf), [source](../papers/tan-evidence-2026/manuscript.md), [evidence index](../papers/tan-evidence-2026/EVIDENCE.md), [clean-map proof](../papers/tan-evidence-2026/clean-map/) | `007d0b4ebe8208aa95e8a70c1175b9bdc2c8373f` |
| Mainline claim-boundary archive and V6–V8 diagnostic archive | [Entry](closures-2026-11-04/README.md), [mainline supplement](closures-2026-11-04/line-a/BOUNDARY_SUPPLEMENT.md), [training technical report](closures-2026-11-04/line-b/TECHNICAL_REPORT.md) | `c75814eac5202e6a265ca42e78805f906d302249` |
| V6–V10 population synthesis | [Paper guide](../papers/native-spike-populations-2026/README.md), [PDF](../papers/native-spike-populations-2026/manuscript.pdf), [source](../papers/native-spike-populations-2026/manuscript.md), [evidence ledger](../papers/native-spike-populations-2026/EVIDENCE.md) | New synthesis of the frozen phase revisions below; [publication inventory](PUBLICATION_V6_V10.json) |

## Population studies: protocols, full artifacts and reports

Every phase directory contains its implementation and retained numerical evidence, not merely a report. Later phases retain initial/final weights, every clipped gradient, checkpoints, datasets, native trajectories and audit outputs as specified by their protocols. The manifest is the authoritative inventory; artifact formats and coverage differ between phases.

| Phase | Question / result | Report and protocol | Files | Original completed revision |
|---|---|---|---|---|
| V1 | Fixed population circuits; the ordinary point control remained competitive and was more robust in tested noise | [Report](dendritic-population-v1/REPORT.md), [protocol](dendritic-population-v1/PROTOCOL.md) | [Directory](dendritic-population-v1/), [manifest](dendritic-population-v1/MANIFEST.json) | `c47db8b16f159f1810c53da53a61287cfe273450` |
| V2 | Learned recurrence with SPSA did not reliably solve delayed XOR | [Report](dendritic-population-v2/REPORT.md), [protocol](dendritic-population-v2/PROTOCOL.md) | [Directory](dendritic-population-v2/), [manifest](dendritic-population-v2/MANIFEST.json) | `ac2625730cba8863b85e52272daf657ba51f893a` |
| V3 | Both native-spike structures learned sustained channel identity | [Report with correction appendix](dendritic-population-v3/REPORT_WITH_APPENDIX.md), [original report](dendritic-population-v3/REPORT.md), [protocol](dendritic-population-v3/PROTOCOL.md) | [Directory](dendritic-population-v3/), [manifest](dendritic-population-v3/MANIFEST.json) | `630525bff536be5abf640ec71616b016ee421238` |
| V4 | Difficulty ladder and recurrence controls; shared-recipe delayed-XOR difference | [Report](dendritic-population-v4/REPORT.md), [protocol](dendritic-population-v4/PROTOCOL.md) | [Directory](dendritic-population-v4/), [manifest](dendritic-population-v4/MANIFEST.json) | `bd985d64aff2a5872870abe11d86e2945bf4a79a` |
| V5 | Fixed-recipe replication and equal configuration search; primary tuned comparison remained uncertain | [Report](dendritic-population-v5/REPORT.md), [tables](dendritic-population-v5/TABLES.md), [protocol](dendritic-population-v5/PROTOCOL.md) | [Directory](dendritic-population-v5/), [manifest](dendritic-population-v5/MANIFEST.json) | `73c1bff6eebf0cb98502f19f8b08d15e2a261182` |
| V6 | Two fixed training recipes; post hoc V5 diagnosis and 80 new fits | [Report](dendritic-population-v6/REPORT.md), [tables](dendritic-population-v6/TABLES.md), [protocol](dendritic-population-v6/PROTOCOL.md) | [Directory](dendritic-population-v6/), [manifest](dendritic-population-v6/MANIFEST.json) | `75a595f3c0c5aa3f50d7700b8c142906aca442b0` |
| V7 | Exploratory loss × learning rate factorial; 160 new fits | [Report](dendritic-population-v7/REPORT.md), [tables](dendritic-population-v7/TABLES.md), [protocol](dendritic-population-v7/PROTOCOL.md) | [Directory](dendritic-population-v7/), [manifest](dendritic-population-v7/MANIFEST.json) | `0c74816da453003490f313c8ee7939e16840e4f2` |
| V8 | One prespecified POINT interaction in 104 fresh blocks / 416 fits | [Report](dendritic-population-v8/REPORT.md), [tables](dendritic-population-v8/TABLES.md), [protocol](dendritic-population-v8/PROTOCOL.md) | [Directory](dendritic-population-v8/), [manifest](dendritic-population-v8/MANIFEST.json) | `9cddbbad7f1722d8a116c0a8245fc40d3a5e97ca` |
| V9 | Delayed-XOR recurrence intervention; 20 fits plus 10 derived inference cuts | [Report](dendritic-population-v9/REPORT.md), [tables](dendritic-population-v9/TABLES.md), [protocol](dendritic-population-v9/PROTOCOL.md) | [Directory](dendritic-population-v9/), [manifest](dendritic-population-v9/MANIFEST.json) | `c75814eac5202e6a265ca42e78805f906d302249` |
| V10A | Feature binding with brief retention at four fixed sizes; 40 fits | [Report](dendritic-population-v10/REPORT.md), [tables](dendritic-population-v10/TABLES.md), [protocol](dendritic-population-v10/PROTOCOL.md) | [Directory](dendritic-population-v10/), [manifest](dendritic-population-v10/MANIFEST.json) | `c75814eac5202e6a265ca42e78805f906d302249` |

V6–V10 total 716 new fits. Their independent training/data-block counts are 20, 20, 104, 5 and 5, respectively, with paired conditions within each block. These cohorts and tasks are not pooled. V9's inference cuts do not add trained models or independent blocks.

## How to interpret the latest results

Effect sizes below use percentage points (pp).

- **V8:** on delayed XOR with 200 updates, the POINT interaction `(CE/.03−CE/.01)−(RATE/.03−RATE/.01)` was −14.11 pp; the prespecified approximate Student-t 95% interval was [−22.60, −5.61]. The direction recurred on fresh blocks; V7's crossover of simple-effect point estimates did not. No DEND arm was included.
- **V9:** DEND intact/retrained-no-cross/inference-cut pass counts were 4/5, 5/5 and 0/5. Retrained no-cross solutions reached the task gate without cross-cell recurrence, while retaining several cells, self-feedback and state. Cutting a learned pathway and proving task-level necessity are different questions. POINT passed 0/5 in all conditions.
- **V10A:** at hidden-soma counts 6/4/2/1, DEND balanced accuracy averaged 99.57/97.04/88.50/88.34%; POINT averaged 89.76/84.24/68.92/71.60%. These are fixed RATE/.01, 200-update, five-block comparisons with unadjusted pointwise intervals. A minimum-size POINT model passed; the data do not establish a representational impossibility.

V5's independently selected primary comparison remains +19.20 pp, 95% seed-pair interval [−11.39, +49.41] across ten pairs; its stability criterion failed. The selected POINT CE/.003 configuration is outside the V6–V8 fixed grid. Subsequent fixed-recipe results do not erase this selection/transfer limitation. Crossing zero is uncertainty, not equivalence.

Activity and settling records are computational proxies, not measured energy or biological time. Native forward dynamics are distinct from surrogate learning. All numerical verification remains within the same AI-assisted workflow, not external independent replication or peer review.

## Publication and historical status

[PUBLICATION_V6_V10.json](PUBLICATION_V6_V10.json) records this edition's retained source revisions and publication checks. The earlier [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) remains the historical V1–V5 publication inventory. The 680-file snapshot retains its own [manifest](completed-tan-2026-11-03/SNAPSHOT_MANIFEST.json) and [ZIP reconstruction instructions](completed-tan-2026-11-03/archive/).

The original protocols, phase reports, pause states and closure records describe their execution-time status, including “local only” or “not pushed”. Those files are intentionally unchanged. This updated index and publication inventory describe the later publication edition; GitHub repository publication is not a journal submission or an external scientific replication. The new manuscript reuses the saved evidence without running additional research experiments.

## Obtain and reproduce

```bash
git clone https://github.com/20171806011/Temporal-Attention-Neuron.git
cd Temporal-Attention-Neuron
```

Use the default-branch edition for the full reading path, or check out a specific recorded commit for an exact historical phase. Complete data and traces require several gigabytes. Check the per-phase manifest for the actual file inventory and the protocol/environment notes for dependencies.

Inspect saved reports and audit outputs first. Some original checkers overwrite logs beside the evidence, and some execution scripts enforce protocol-before-data and checkpoint-before-test commits. Reproduce in a separate working copy rather than overwriting the preserved evidence. The [new paper guide](../papers/native-spike-populations-2026/README.md) describes publication-only validation and rebuilding.
