# TAN complete research index — 3 November 2026

This cumulative publication branch contains the existing TAN programme, the screened 680-file archive, the current evidence manuscript with the clean-map supplement, and dendritic-population V1–V5. The original research commits and all frozen files are preserved. This branch is `research/all-results-2026-11-03`; publishing it does not merge it into `main`.

## Read the results

| Collection | Entry points | Original completed revision |
|---|---|---|
| Original TAN, TAN-I, TAN-II, P1A/P1B and sprint studies | [Root guide](../README.md), [historical chronology](../RESEARCH_CHRONOLOGY.md), [earlier papers](../release/) | `85213f0d417d54e4f48b301ed1c0e4200ef3c421` |
| Screened 680-file snapshot | [Guide](completed-tan-2026-11-03/README.md), [inventory](completed-tan-2026-11-03/SNAPSHOT_MANIFEST.json), [complete ZIP parts](completed-tan-2026-11-03/archive/) | `222caa0201feff1e37a06049aa44c0e59e1cbaba` |
| Evidence manuscript and clean-map supplement | [PDF](../papers/tan-evidence-2026/manuscript.pdf), [source](../papers/tan-evidence-2026/manuscript.md), [evidence index](../papers/tan-evidence-2026/EVIDENCE.md), [clean-map proof and checks](../papers/tan-evidence-2026/clean-map/) | `007d0b4ebe8208aa95e8a70c1175b9bdc2c8373f` |
| V1: initial population circuit | [Report](dendritic-population-v1/REPORT.md), [protocol](dendritic-population-v1/PROTOCOL.md), [integrity manifest](dendritic-population-v1/MANIFEST.json) | `c47db8b16f159f1810c53da53a61287cfe273450` |
| V2: learned recurrence and SPSA failures | [Report](dendritic-population-v2/REPORT.md), [protocol](dendritic-population-v2/PROTOCOL.md), [integrity manifest](dendritic-population-v2/MANIFEST.json) | `ac2625730cba8863b85e52272daf657ba51f893a` |
| V3: native-spike training/readout controls | [Report with correction appendix](dendritic-population-v3/REPORT_WITH_APPENDIX.md), [original report](dendritic-population-v3/REPORT.md), [protocol](dendritic-population-v3/PROTOCOL.md), [integrity manifest](dendritic-population-v3/MANIFEST.json) | `630525bff536be5abf640ec71616b016ee421238` |
| V4: difficulty ladder and recurrence controls | [Report](dendritic-population-v4/REPORT.md), [protocol](dendritic-population-v4/PROTOCOL.md), [integrity manifest](dendritic-population-v4/MANIFEST.json) | `bd985d64aff2a5872870abe11d86e2945bf4a79a` |
| V5: replication and equal configuration search | [中文概览](dendritic-population-v5/README.md), [report](dendritic-population-v5/REPORT.md), [all seeds](dendritic-population-v5/TABLES.md), [protocol](dendritic-population-v5/PROTOCOL.md), [integrity manifest](dendritic-population-v5/MANIFEST.json) | `73c1bff6eebf0cb98502f19f8b08d15e2a261182` |

Every V1–V5 directory includes its code and saved numerical evidence, not merely a report. The population studies preserve weights, datasets, configurations and validation outputs; the later studies retain every candidate and clipped-gradient trajectory as specified in their protocols. Read each manifest for the actual file inventory.

## Current interpretation

The mainline separates history sensitivity, static addressing and recoverable output information. The clean-map result concerns specified response contracts and does not imply an average-accuracy ceiling or a universal neuron limitation. The population line asks bounded questions about native-spike trainability and finite resource-matched comparisons.

V5's primary comparison after independent configuration selection is +19.20 percentage points for DEND minus POINT, with a 95% trained-seed-pair interval of [−11.39,+49.41] across 10 pairs, 512 test histories per pair and 200 training updates. The prespecified stability criterion fails. Fixed-recipe differences repeated, but configuration transfer was unreliable and both selected-condition means fell relative to the same-cohort fixed controls. Do not replace this uncertain primary result with a favorable secondary result or infer equivalence from the interval crossing zero.

No new experiments were run for this publication. Internal numerical checks are not external independent replication or peer review. No energy, hardware, biological-realization or universal task-superiority claim is established.

## Integrity, reconstruction and historical status

[PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json) records the component counts, bytes, source commits and manifests. At publication preparation, all 1,363 V1–V5 manifest entries matched, all 680 snapshot file hashes matched, and the three archive parts reconstructed the exact 189,164,458-byte ZIP with SHA-256 `ca8a2ae51b4e8f18ab341fa483ca9970a7a3077c5d4e4013e5df13a9811b6167`.

The original phase reports and provenance records describe their execution-time status, including “local only” or “not pushed”. Those records are intentionally unchanged; this index records the later publication. The private clean-map source archive was previously screened separately: the published supplement contains the authorized scientific findings and necessary evidence, not excluded operational material.

To obtain all research files, select this branch:

```bash
git clone --branch research/all-results-2026-11-03 https://github.com/20171806011/Temporal-Attention-Neuron.git
```

The aggregate source tree exceeds 1 GB because it includes data and complete evidence archives. Checkers may overwrite their result logs; inspect saved results first and run reproductions in a separate working copy. Each study's frozen environment, seeds, commit gates and limits remain authoritative.
