# TAN: evidence synthesis manuscript

**Temporal Attention Neurons: Separating History Sensitivity, Addressing, and Recoverable Memory**
Li Zexu · 3 November 2026 · AI-assisted research manuscript draft; not peer reviewed.

- [Read the PDF](manuscript.pdf)
- [Read/edit the canonical Markdown](manuscript.md)
- [Generated LaTeX source](manuscript.tex)
- [Evidence guide and exact source links](EVIDENCE.md)
- [Machine-readable evidence ledger](evidence_ledger.json)
- [Prior manuscript/report inventory](prior_artifact_inventory.json)
- [Publication validation](validation.json)

This paper brings the existing TAN manuscripts and later audits together. It separates static addressing, native higher-moment sensitivity, the verified distractor-swap response collision, the clean-training composition control, and matched-calibration adaptive baselines. The main numerical evidence is the previously published 680-file snapshot. A separate clean-map supplement adds previously archived certificates and a new computational check of fixed trajectories; no classifier was fitted and no task-performance experiment was rerun.

The separate [clean-map supplement](clean-map/README.md) resolves the initial missing-certificate caveat. A newly implemented checker verifies four exact-real collision brackets in both softmax conventions and all 64 fixed trajectories: 5,482 checks passed. This establishes a scoped response collision, not a one-third average accuracy ceiling or independent human replication. Native higher-moment sensitivity remains an analytical statement rather than a claimed measured task advantage.

The original model/project is credited to Li Zexu. AI-assisted audits, analysis, checks, and drafting are disclosed in the manuscript. This draft needs human review before journal submission; it claims neither independent replication nor a breakthrough.

## Build and validate

From this directory:

```sh
python3 clean-map/verify_independent.py
python3 validate_evidence.py
bash build.sh
```

The validator uses only the Python standard library and reads the sibling repository snapshot. It checks file identity and selected saved-result arithmetic without fitting or simulation; the separate clean-map checker also recomputes fixed scalar trajectories. The PDF build requires Pandoc and pdfLaTeX with the packages used by Pandoc's standard LaTeX template. `manuscript.md` is canonical; `manuscript.tex` is generated. Build logs and TeX auxiliary files are ignored.

`prepare_evidence.py` rebuilds the ledger from the specifically selected sources. Do not regenerate the ledger to conceal an unexpected source change: first establish why the archived evidence changed.

## 中文说明

这是一篇基于已归档证据的综合研究论文，保留正结果、负结果和方法限制。全文区分原始项目贡献、AI 辅助研究、本地执行记录及尚未在这里独立复现的 Chat 概念实验。旧论文和 680 个科研文件不作改动；main 分支不作改动。

## Concurrent TAN-III manuscript

During final checks, main advanced to `65b0d2e9310a5790e653b77114bf4b6c7aa93e9c` with [TAN-III](https://github.com/20171806011/Temporal-Attention-Neuron/blob/65b0d2e9310a5790e653b77114bf4b6c7aa93e9c/snapshot/paper3/main.tex). It was inspected and left unchanged. This draft complements that carrier/decoder-focused paper with observable-response and training-composition analysis, and does not endorse broad biological or universal claims without additional evidence. Shared results are the same experiments.
