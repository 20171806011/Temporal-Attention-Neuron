# Native-spike population studies: V6–V10

**Training Interactions, Recurrence Dependence, and Size Tradeoffs in Native-Spike Dendritic and Point Networks**

Li Zexu · Research preprint · 4 November 2026

This manuscript brings together five separately frozen studies of native-spike learning and computation. It distinguishes training-configuration interaction, dependence of a learned solution on recurrence, and fixed-recipe network-size tradeoffs. It is separate from the [original TAN evidence paper](../tan-evidence-2026/README.md), and does not turn population results into claims about the original single-unit TAN model.

## Read the paper and evidence

- [Compiled manuscript (PDF)](manuscript.pdf)
- [Manuscript source (Markdown)](manuscript.md)
- [Generated LaTeX source](manuscript.tex)
- [Claim-to-source evidence index](EVIDENCE.md) and [machine-readable ledger](evidence.json)
- [Publication validation](validation.json) and [complete paper file hashes](MANIFEST.json)
- [Publication build script](build.sh) and [citation metadata](CITATION.cff)
- [Complete V1–V10 research index](../../research/README.md)
- [中文 V9/V10 结果摘要](../../research/closures-2026-11-04/RESULTS_SUMMARY.md)

## Three findings, with their limits

**Training configuration.** V6's bundled recipe comparison remained uncertain and changed loss and learning rate together. V7's factorial study motivated V8's fresh-cohort POINT test. In V8's 104 paired blocks, the prespecified loss-by-learning-rate interaction was −14.11 percentage points, primary approximate Student-t 95% interval [−22.60, −5.61], at 200 updates on delayed XOR. The negative direction recurred; the earlier crossover of simple-effect point estimates did not. V8 had no DEND arm.

**Recurrence.** Under fixed RATE/.01 and 200 updates, V9 DEND passed in 4/5 intact, 5/5 retrained-no-cross and 0/5 inference-cut models. The retrained solutions provide observed alternatives without cross-cell recurrence for this finite task gate; they retain multiple cells, state and self-feedback. Removing 60 effective parameters also changes capacity. Failure of five retrained models would not have proved representational impossibility.

**Size.** V10A compares both architectures at four fixed hidden-soma counts on all-pattern feature binding with brief retention. DEND has higher mean balanced accuracy in this five-block cohort, but the intervals are exploratory and not multiplicity-adjusted. A minimum-size POINT model passes, and that model still has five spiking cells. Event counts, state variation and retrospective settling are computational proxies, not physical energy or an online stopping policy.

## Studies and reproducibility records

| Study | New fits | Independent paired blocks | Frozen protocol | Results and artifacts |
|---|---:|---:|---|---|
| V6 | 80 | 20 | [Protocol](../../research/dendritic-population-v6/PROTOCOL.md) | [Report](../../research/dendritic-population-v6/REPORT.md), [manifest](../../research/dendritic-population-v6/MANIFEST.json) |
| V7 | 160 | 20 | [Protocol](../../research/dendritic-population-v7/PROTOCOL.md) | [Report](../../research/dendritic-population-v7/REPORT.md), [manifest](../../research/dendritic-population-v7/MANIFEST.json) |
| V8 | 416 | 104 | [Protocol](../../research/dendritic-population-v8/PROTOCOL.md) | [Report](../../research/dendritic-population-v8/REPORT.md), [manifest](../../research/dendritic-population-v8/MANIFEST.json) |
| V9 | 20 | 5 | [Protocol](../../research/dendritic-population-v9/PROTOCOL.md) | [Report](../../research/dendritic-population-v9/REPORT.md), [manifest](../../research/dendritic-population-v9/MANIFEST.json) |
| V10A | 40 | 5 | [Protocol](../../research/dendritic-population-v10/PROTOCOL.md) | [Report](../../research/dendritic-population-v10/REPORT.md), [manifest](../../research/dendritic-population-v10/MANIFEST.json) |

The 716 new fits are not pooled independent observations. Ten V9 inference-cut vectors are derived without additional training. Within each study, conditions share data and initialization by block. All failed, silent and tied fits remain included. Preserved code, datasets, initial/final weights, clipped-gradient histories, native trajectories and internal checks are linked through each phase's manifest.

The V5 independently tuned comparison remains contextual evidence with an unresolved primary interval; its selected POINT CE/.003 recipe is not in the V6–V8 fixed grid. This paper does not replace that comparison with favorable fixed-recipe results.

## Build and inspect

The build uses Pandoc and XeLaTeX with DejaVu fonts. Read the evidence validator for its Python requirements. From the repository root:

```bash
python3 papers/native-spike-populations-2026/validate_evidence.py
bash papers/native-spike-populations-2026/build.sh
```

Read the scripts and [evidence index](EVIDENCE.md) for their scope and requirements. The publication workflow checks retained source identities and arithmetic and rebuilds the document; it does not train new models or create new scientific test data. Rebuilding writes publication outputs. Original experiment/audit scripts can overwrite their own evidence logs, so use a separate working copy for research reproduction under the corresponding phase's instructions.

[The publication inventory](../../research/PUBLICATION_V6_V10.json) records retained source revisions and publication checks. Original phase statements such as “local only” or “paused” are historical execution records, preserved without alteration; the later publication does not change the original results.

## Authorship and status

Research direction and native firing constraints were proposed by Li Zexu. Protocol formalization, implementation, execution, numerical verification and manuscript drafting were AI-assisted. All verification is internal to that workflow, not external independent replication or peer review. No journal submission, acceptance, DOI, biological realization, measured energy benefit or general architecture superiority is asserted.
