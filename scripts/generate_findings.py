import csv
import os

rows = [
    {
        'id': 'A01',
        'evidence': 'paper2/sections/s2_framework.tex:211-212 (fixed), docs/TANII_SPRINT4_2B_PREREGISTRATION.md:38 (frozen verbatim) vs :63, code/experiments/sprint4_2/sprint4_2_vector_qk.py:97, :123',
        'severity': 'P0 BLOCKER',
        'handling': 'Independent numerical verification of both definitions. Text corrected in paper2 (u(theta)=(cos theta, sin theta), Q=cS*u(omega S)); errata documented; frozen code verified.',
        'status': 'RESOLVED'
    },
    {
        'id': 'A02',
        'evidence': 'code/experiments/effective_dimension.py:112, :967, :984, :1000, results/tables/effective_dimension_summary.csv, results/logs/effective_dimension.log',
        'severity': 'P0 BLOCKER',
        'handling': 'Evaluated all decision gates from frozen CSV. Corrected MIN_LOGT_DIFF threshold to 0.2 (effective_dimension.py:112). Identified stable=True with outcome=D as UNCLASSIFIED_MIXED_CASE branch gap; separated log-trace diff (1.094463) vs diff-in-diff (1.354674); added error fixtures.',
        'status': 'RESOLVED'
    },
    {
        'id': 'A03-1',
        'evidence': 'paper2/sections/s3_state_geometry.tex:29, paper2/sections/s10_ledger.tex:Table 1, results/tables/natural_collision_summary.csv',
        'severity': 'P0 BLOCKER',
        'handling': 'Corrected paper2 text mislabeling K_mean as median. Reported both mean (with 95% CI) and median (B1: 0.5, B2: 895.5, B3: 1369.1, B4: 2947.7) in s3 and Table 1.',
        'status': 'RESOLVED'
    },
    {
        'id': 'A03-2',
        'evidence': 'paper2/sections/s5_routing_geometry.tex:185, results/sprint4_2/sprint4_2b_summary.json:186, :277',
        'severity': 'P0 BLOCKER',
        'handling': 'Separated 2-candidate toy geometric example (JSD=0.0751/0.0647) from full-window canonical model results (M0=0.009462, M2=0.008727).',
        'status': 'RESOLVED'
    },
    {
        'id': 'A03-3',
        'evidence': 'paper2/sections/s5_routing_geometry.tex:118, paper2/sections/s10_ledger.tex:Table 1, results/sprint4_2/sprint4_2_summary.json',
        'severity': 'P0 BLOCKER',
        'handling': 'Clarified maximum deviation metric: 0.009267 (single-seed max) vs 0.004667 (seed-averaged max). Explicitly distinguished 132 isotropic conditions from total 140 checks (including 8 controls). Noted theta_deg is fraction of pi (theta/pi).',
        'status': 'RESOLVED'
    },
    {
        'id': 'A03-4',
        'evidence': 'code/experiments/temporal_distractor.py:74-135, code/analysis/probe1_v3_amended_audit.py:187, audit_v3_amended/POOLED_PROBE1_V3.md',
        'severity': 'P0 BLOCKER',
        'handling': 'Derived sample counts generatively from temporal_distractor.py logic (N_TEST=2000, catch=667, present=1333, dist=666, catch_dist=333 -> 999/seed, total 2997 distractor trials out of 6000 total test trials). Validated bootstrap conditioning scope.',
        'status': 'RESOLVED'
    },
    {
        'id': 'A03-5',
        'evidence': 'paper2/sections/s8_composition.tex:67-73, paper2/sections/s10_ledger.tex:Table 1, results/sprint4_3/monte_carlo_results.json',
        'severity': 'P0 BLOCKER',
        'handling': 'Reported exact both-routed coverage (6,608 / 30,000 = 22.03%; benchmark 22.07%) and distinguished quadrature benchmarks (E_plus=1.116099621, E_minus=2.000999500) from 30k MC pooled empirical means (E_plus=1.107671595, E_minus=1.975892767); conditional error is strictly 0.0.',
        'status': 'RESOLVED'
    },
    {
        'id': 'A04-1',
        'evidence': 'paper/sections/methods.tex:37, paper2/sections/s2_framework.tex:52-58, paper2/sections/s3_state_geometry.tex:18',
        'severity': 'P0 BLOCKER',
        'handling': 'Narrowed theoretical scope in s2_framework.tex: scalar membrane potential h_t alone is insufficient state, but does not imply complete state (h_t, X_t) violates Markovianity. Clarified that Markovianity does not imply state-space contraction.',
        'status': 'RESOLVED'
    },
    {
        'id': 'A04-2',
        'evidence': 'paper2/sections/s2_framework.tex:59-67, paper2/sections/s5_routing_geometry.tex',
        'severity': 'P0 BLOCKER',
        'handling': 'Updated Saliency definition in s2_framework.tex: standard continuous softmax has full positive support, so support invariance is trivial; saliency is defined by ranking invariance across all S > 0. Cited Martins & Astudillo (2016) regarding sparse activations; clarified kernel factorizability as sufficient condition.',
        'status': 'RESOLVED'
    },
    {
        'id': 'A04-3',
        'evidence': 'paper2/sections/s2_framework.tex:66, paper2/sections/s5_routing_geometry.tex:253, docs/TANII_SPRINT4_2_THEORY.md',
        'severity': 'P0 BLOCKER',
        'handling': 'Narrowed scalar sorting theorem: handled S=0 and ties; order invariance does not universally imply rank-1 kernel factorizability without additional regularity assumptions.',
        'status': 'RESOLVED'
    },
    {
        'id': 'A04-4',
        'evidence': 'paper2/sections/s6_binding.tex:74-86, results/sprint4_2/sprint4_2c_summary.json',
        'severity': 'P0 BLOCKER',
        'handling': 'Updated Proposition 6.1 (Softmax Mixing Barrier) in s6_binding.tex: explicitly specified distinct values V_A != V_B and event-normalized readout C_ev. Added Remark 6.2 on background dilution cancellation artifacts (AM5 at gamma approx 1.198).',
        'status': 'RESOLVED'
    },
    {
        'id': 'A04-5',
        'evidence': 'paper2/sections/s7_wta.tex:89-122, paper2/sections/s8_composition.tex',
        'severity': 'P0 BLOCKER',
        'handling': 'Updated Proposition 7.2 in s7_wta.tex: clarified that a single head can compute symmetric pooling functions (e.g. 2 * mean) but cannot deliver two distinct bound operands to downstream algebra. Clarified gamma* = 5.098015 as the E-I net drive direction reversal crossing root (T_drive = 0).',
        'status': 'RESOLVED'
    },
    {
        'id': 'A04-6',
        'evidence': 'paper2/sections/s9_ladder.tex:141',
        'severity': 'P0 BLOCKER',
        'handling': 'Corrected claim: fixed temporal window W=5 does NOT imply state space or representation resources did not increase (E-I units, vector keys/queries, dual channels added degrees of freedom).',
        'status': 'RESOLVED'
    },
    {
        'id': 'A05',
        'evidence': 'paper2/sections/s2_framework.tex:25, code/experiments/sprint4_2/sprint4_2_vector_qk.py:85, code/model/tan.py:42',
        'severity': 'P0 BLOCKER',
        'handling': 'Corrected paper2 claim: Sprint 4.2-B uses standard max-subtracted softmax without delta offset; tan.py adds 1e-9 after max-subtraction. Fully tabulated and verified in check_a05_softmax.py.',
        'status': 'RESOLVED'
    },
    {
        'id': 'A06',
        'evidence': 'docs/TANII_SPRINT4_*.md, session transcripts, filesystem timestamps',
        'severity': 'P0 BLOCKER',
        'handling': 'Reconstructed chronology with unsourced clock times marked UNKNOWN; treated template copying as plausible hypothesis rather than asserted fact; accurately categorized AM1-AM6 (AM4: M5-E drive transfer pre-run correction, AM5: D_full non-monotonicity and cancellation artifact, AM6: margin closed-form runtime correction).',
        'status': 'RESOLVED'
    },
    {
        'id': 'A07',
        'evidence': 'paper/main.tex, paper2/main.tex, paper/main.pdf (28 pp), paper2/main.pdf (28 pp)',
        'severity': 'P0 BLOCKER',
        'handling': 'Compiled both manuscripts without error (exit code 0). Formally flagged VISUAL_QA_NOT_COMPLETED due to lack of rendered visual image inspection in headless environment; ruled NOT READY FOR SUBMISSION pending human visual sign-off.',
        'status': 'PARTIALLY_RESOLVED (VISUAL_QA_PENDING / NOT_READY_FOR_SUBMISSION)'
    }
]

headers = ['id', 'evidence', 'severity', 'handling', 'status']
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path = os.path.join(base_dir, 'FINDINGS.csv')
with open(out_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)

print(f'FINDINGS.csv generated successfully with {len(rows)} entries at {out_path}')
