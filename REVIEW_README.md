# P0 Review Workspace: Temporal Attention Neuron (TAN)

**Review Directory:** `C:\Users\李则徐\Downloads\TAN_Review\P0_20260917_181757`  
**Review Timestamp:** 2026-09-17  
**Original Source Path:** `C:\Users\李则徐\Downloads\TAN_Final_Submission\TAN_Final_Submission` (368 files, 100% frozen)  
**Execution Environment:** Python 3.12.7, NumPy 1.26.4, SciPy 1.13.1, Matplotlib 3.9.2, MiKTeX 24.1.0  

---

## 1. Directory Structure and Deliverables

```
P0_20260917_181757/
├── REVIEW_README.md              # [This file] Workspace guide, layout, and instructions
├── P0_REWORK_REPORT.md           # Point-by-point resolution report for user audit critique
├── CURRENT_RESEARCH_STATE.md     # Comprehensive state audit, artifact inventory, and boundary matrix
├── RESEARCH_CHRONOLOGY.md        # Reconstructed 25-round timeline, Sept 4 vs Sept 8-9 analysis, amendments
├── ERRATA.md                     # Complete errata for issues A01–A07 with original/revised text & rationales
├── MASTER_CLAIM_LEDGER.csv       # Unified master claim ledger (10 claims, evidence, counterexamples, wording)
├── FINDINGS.csv                  # Itemized findings registry (ID, evidence, severity, handling, status)
├── NUMERICAL_CHECKS.json         # Machine-readable JSON of all independent numerical verifications
├── VALIDATION_REPORT.md          # Multi-dimensional validation (numerical, replay, literature, LaTeX, QA)
├── P0_HANDOFF.md                 # Scientific claim disposition (retained/downgraded/withdrawn) & P1 GO/NO-GO
├── manifest_before.json          # Pre-review SHA-256 hash manifest of all 368 source files (with 9 authentic historical hashes)
├── manifest_after.json           # Post-review bidirectional manifest verifying source integrity & derived artifacts
├── audit_replay_summary.json     # Replay execution summary across the 9 audit scripts
├── replay_diff_report.json       # Byte-level diff report comparing 11 replay JSON files bit-for-bit against originals
├── scripts/                      # Independent check & replay scripts (all include error fixtures)
│   ├── check_a01_query_rotation.py     # Independent numerical check of A01 (double omega vs code)
│   ├── check_a02_phase2_branch.py      # Independent gate evaluation of A02 (MIN_LOGT_DIFF = 0.2, decision tree gap)
│   ├── check_a03_statistics.py         # Independent verification of A03 (means, medians, JSDs, Probe-1 counts, quadrature vs MC)
│   ├── check_a05_softmax.py            # Numerical check of A05 (max-stabilized vs denominator offsets)
│   ├── check_composition_carrier.py   # AST/data-flow analysis of Sprint 4.3-A computation carrier (AST call visitor)
│   ├── run_audit_replay.py             # Replay runner for 8 TAN-II scripts + theory audit (exit 1) + 11-file JSON diff
│   └── generate_manifest_after.py      # Bidirectional manifest verification & generator
├── logs/                         # Detailed execution logs from the replay suite
│   ├── s41cd.log                       # Opponent dynamics execution log (exit 0)
│   ├── s42a.log                        # Vector QK geometry log (exit 0)
│   ├── s42_shortcuts.log               # Shortcut audit log (exit 0)
│   ├── s42b.log                        # Vector QK canonical audit log (exit 0)
│   ├── s42b_probe.log                  # Routing probe log (exit 0)
│   ├── s42c.log                        # Key-value binding probe log (exit 0)
│   ├── s42d.log                        # Hard binding WTA probe log (exit 0)
│   ├── s43a.log                        # Composition probe log (exit 0)
│   └── s42_theory.log                  # Theory audit log (exit 1, E4(iii) failure confirmed)
└── snapshot/                     # Isolated working copy of source archive
    ├── paper/                          # Revised Paper 1 LaTeX files and recompiled PDF (28 pages)
    └── paper2/                         # Revised Paper 2 LaTeX files and recompiled PDF (26 pages)
```

---

## 2. Key Scientific Findings & Resolutions

1. **A01 (Double \omega in Paper 2 Text):** Resolved. Code uses $\theta = \omega S$ (margins $+0.2587 \to -0.1559$, flipping at $Q=4$); text had $\omega^2 S$ (margin $+0.8426$, no flip). Corrected in `paper2/sections/s2_framework.tex` copy.
2. **A02 (Phase 2 Outcome D Classification Gap):** Resolved. `stable = True` ($dD_{\text{delta}} = 1.0889 \ge 0.5$, $CI_{\text{lo}} = 1.8739 > 0$), `MIN_LOGT_DIFF = 0.2`. Outcome D was caused by falling through conditions A, B, C. Classified as `UNCLASSIFIED_MIXED_CASE`.
3. **A03 (Statistical Metrics):** Resolved. Corrected $K$ mean mislabeled as median; separated 2-candidate toy JSD ($0.0751/0.0647$) from 5-slot canonical JSD ($0.009462/0.008727$); distinguished single-seed max error ($0.009267$) from seed-averaged max error ($0.004667$); disclosed Probe-1 total ($6,000$) vs distractor ($2,997$) counts; reported 4.3-A both-routed coverage ($6,608 / 30,000 = 22.03\%$), distinguishing quadrature benchmarks ($1.1161 / 2.0010$) from 30k MC empirical means ($1.1077 / 1.9759$).
4. **A04 (Scope of Claims):** Resolved. Clarified Markovian non-contraction, membrane potential $h$ state non-sufficiency, finite softmax full support, scalar sorting edge cases, and state space growth across the ladder.
5. **A05 (Softmax Conventions):** Resolved. Sprint 4.2-B code uses standard max-subtracted softmax without offset; base `tan.py` uses denominator offset $10^{-9}$. Text corrected.
6. **A06 (Timeline):** Resolved. Sept 4 dates in TAN-II files explained by template copying from TAN-I; actual execution occurred Sept 8–9. Unsourced times marked `UNKNOWN`. Detailed AM1–AM6 technical amendments.
7. **A07 (Submission Readiness):** Resolved. Marked `PARTIALLY_RESOLVED (VISUAL_QA_PENDING / NOT_READY_FOR_SUBMISSION)` due to lack of rendered page image inspection; placeholders documented; ruled **NOT READY FOR SUBMISSION**.
8. **Computation Carrier:** Carrier analysis via AST proves that Sprint 4.3-A composition is a constructive operator demonstration ($y = C_1 \pm C_2$) operating on statically retrieved values, bypassing membrane potential updates ($h$) and spike integration.

---

## 3. How to Re-Verify All Results

From PowerShell:
```powershell
Set-Location "C:\Users\李则徐\Downloads\TAN_Review\P0_20260917_181757"

# 1. Run all independent checks (including error fixtures)
python -X utf8 -B scripts/check_a01_query_rotation.py
python -X utf8 -B scripts/check_a02_phase2_branch.py
python -X utf8 -B scripts/check_a03_statistics.py
python -X utf8 -B scripts/check_a05_softmax.py
python -X utf8 -B scripts/check_composition_carrier.py

# 2. Replay the audit suite and verify 11/11 JSON bit-for-bit identity
python -X utf8 -B scripts/run_audit_replay.py

# 3. Verify bidirectional file integrity and update manifest
python -X utf8 -B scripts/generate_manifest_after.py
```

