import csv
import os

rows = [
    {
        'claim_id': 'CLM-01',
        'exact_claim': 'Instantaneous membrane potential h alone is not a sufficient state representation for TAN; natural collision pairs with identical h diverge under identical future inputs.',
        'model_family': 'B1 (LIF), B2 (LIF+Buffer), B3 (TAN-noAttn), B4 (Full TAN)',
        'input_domain': 'Discrete time scalar input sequences x_t in [0.0, 3.0] (stimulus generators gen_gauss, gen_step, gen_ramp, gen_sine clipped to 3.0; probe U(0.5, 3.0), continuation U(0.3, 1.8)), window W=5, 160,000 histories',
        'assumptions': 'Identical membrane reset threshold theta=0.5, noise-free input replay, fixed leak lambda=0.5 (natural_collision.py:95)',
        'evidence_files_and_fields': 'results/tables/natural_collision_summary.csv, results/logs/natural_collision.log, code/experiments/natural_collision.py',
        'original_status': 'CONFIRMED (results/tables/natural_collision_summary.csv, docs/CLAIM_LEDGER.md:14)',
        'review_status': 'RETAINED (QUALIFIED)',
        'allowed_wording': 'Instantaneous membrane potential h alone is insufficient state for TAN; the temporal window provides history dependence. Does not imply complete Markovian state (h_t, X_t) violates Markovianity, nor that Markovianity implies contraction.',
        'counterexample_or_shortcut': 'Markov systems like h_{next} = 2h diverge under identical inputs; non-contraction is not equivalent to non-Markovianity of the complete state.',
        'uncompleted_checks': 'Clean-environment full Monte Carlo rerun (160k histories) preserved in frozen archive; existing summary and logs verified.',
        'stop_condition': 'If natural collision pairs do not diverge under identical future inputs, revoke state non-sufficiency claim.'
    },
    {
        'claim_id': 'CLM-02',
        'exact_claim': 'Surprise-gated attention reorganizes event-locked local response geometry relative to baselines.',
        'model_family': 'B1, B3, B4, C0',
        'input_domain': 'Event-locked transient inputs with 374 detected events across 3 seeds',
        'assumptions': 'Covariance estimation on event-locked vs quiet windows; Participation ratio / Roy dimension metric; MIN_LOGT_DIFF=0.2 (effective_dimension.py:112)',
        'evidence_files_and_fields': 'results/tables/effective_dimension_summary.csv, results/logs/effective_dimension.log, code/experiments/effective_dimension.py',
        'original_status': 'OUTCOME: D (results/logs/effective_dimension.log:984; log note: No stable event-locked expansion)',
        'review_status': 'DOWNGRADED (QUALIFIED)',
        'allowed_wording': 'Local response covariance geometry is altered in specific coordinates; does not support higher intrinsic dimension than LIF. Outcome D was due to decision tree classification gap (UNCLASSIFIED_MIXED_CASE: stable=True but not A/B/C). Note: effective rank / participation ratio of local covariance is not equivalent to coordinate-invariant intrinsic dimension.',
        'counterexample_or_shortcut': 'B4-zC (1.589) < B1-zC (1.674), failing Condition A; B4-zF > B3-zF, failing Condition B; |B4-zC - C0| = 0.534 > 0.5, failing Condition C.',
        'uncompleted_checks': 'Decision gate re-run under expanded multi-category classifier; current claim restricted to raw covariance metrics.',
        'stop_condition': 'If event-locked local response covariance structure shows no statistically significant divergence from quiet baseline or uncoupled controls under cross-validated covariance estimation, retract covariance reorganization claim.'
    },
    {
        'claim_id': 'CLM-03',
        'exact_claim': 'Full TAN improves target-relevant retrieval under content-rich context vs no-attention control.',
        'model_family': 'B1, B2, B3, B4',
        'input_domain': 'Synthetic temporal retrieval sequence with distractor pulses; 6,000 total test trials (2,997 distractor-present/catch across 3 seeds, derived from temporal_distractor.py: N_test=2000, 1/3 catch -> 667 catch, 1333 present; 666 distractor-present, 333 distractor-catch -> 999/seed * 3 = 2997)',
        'assumptions': 'Threshold readout at target retrieval step; cue amplitude matching',
        'evidence_files_and_fields': 'docs/CLAIM_LEDGER.md:20, paper/tables/tab_claims.tex:10, code/experiments/temporal_distractor.py:49,466, results/tables/temporal_distractor_summary.csv',
        'original_status': 'LIMITED SUPPORT (docs/CLAIM_LEDGER.md:20; temporal_distractor.py:49,466 defined Outcome B as: generic dynamical advantage (LIF within CI of B4))',
        'review_status': 'DOWNGRADED (LIMITED CONDITIONAL)',
        'allowed_wording': 'Conditional retrieval advantage over compromised B3 baseline under specific retrieval cues; B2 (LIF+delay line) provides a strong positional baseline (~0.843); exhibits seed decay and class coverage restrictions; does not support universal interference robustness.',
        'counterexample_or_shortcut': 'B2 achieves 0.843 accuracy on distractor condition without attention mechanism; B4 advantage is only relative to B3 (0.389).',
        'uncompleted_checks': 'Bootstrap analysis on independent unpooled seeds.',
        'stop_condition': 'If B2 baseline outperforms B4 under calibrated delay tuning across general distractor distributions, retract retrieval advantage claim.'
    },
    {
        'claim_id': 'CLM-04',
        'exact_claim': 'Positive scalar attention kernel enables dynamic query-conditioned content routing across multiple memorized candidates.',
        'model_family': 'Probe-2 Identifiability Suite (M0, M1, M2)',
        'input_domain': 'Multi-candidate stimulus sequences with variable query amplitudes',
        'assumptions': 'Positive scalar kernel E_{t,i} = beta * W_q * W_k * S_t * x_i with S_t >= 0',
        'evidence_files_and_fields': 'code/experiments/probe2/identifiability/probe2_identifiability_audit.py, code/experiments/probe2/identifiability/README.md:11, docs/REPRODUCIBILITY_LEDGER.md:71',
        'original_status': 'TERMINATED / AUDIT_FAILED (code/experiments/probe2/identifiability/README.md:11, docs/REPRODUCIBILITY_LEDGER.md:71)',
        'review_status': 'REFUTED / TERMINATED',
        'allowed_wording': 'Positive scalar attention kernel cannot perform content routing; candidate argmax order is strictly invariant across all positive query amplitudes (Theorem 5.1). Probe-2 terminated with 0 argmax flips.',
        'counterexample_or_shortcut': 'Theorem 5.1 proof: E_A - E_B = beta * W_q * W_k * S * (x_A - x_B). For S > 0, sign(E_A - E_B) = sign(x_A - x_B) is constant; highest amplitude candidate always wins.',
        'uncompleted_checks': 'None (structural impossibility theorem proved and empirical failure replicated).',
        'stop_condition': 'Claim refuted; permanently closed for scalar kernels.'
    },
    {
        'claim_id': 'CLM-05',
        'exact_claim': 'Excitatory-inhibitory opponent topology produces non-monotonic selective tuning peaks; static opponent competition achieves query-conditioned content routing.',
        'model_family': 'Opponent Network Layouts C1-C6 (Sprint 4.1-C/D)',
        'input_domain': 'Stimulus input sequences across 6 opponent network layouts * 2 variants',
        'assumptions': 'Static linear threshold inhibition; E-I interaction weights; positive scalar inputs',
        'evidence_files_and_fields': 'results/sprint4_1cd/sprint4_1cd_summary.json, code/experiments/sprint4_1cd_ei_dynamics.py, docs/TANII_SPRINT4_1CD_RESULTS.md:186, paper2/sections/s10_ledger.tex:43-47',
        'original_status': 'ESTABLISHED (Tuning peak in C1, x*=1.00, h_E=0.200) / NEGATIVE (Routing FlipRate = 0.000 across 6 layouts) (docs/TANII_SPRINT4_1CD_RESULTS.md:186, paper2/sections/s10_ledger.tex:43-47)',
        'review_status': 'DOWNGRADED (TUNING ESTABLISHED, ROUTING REFUTED)',
        'allowed_wording': 'Opponent structure produces nonmonotonic selective tuning peaks via E-I interaction (Theorem 4.1, layout C1 peak x*=1.00); static opponent competition does NOT achieve content routing (flip rate = 0.000 across 6 layouts * 2 variants; query affects response magnitude only, not winner selection).',
        'counterexample_or_shortcut': 'Layout C1 achieves tuning peak without attention (zero attention necessity counterexample); all 6 layouts fail to produce a single winner flip.',
        'uncompleted_checks': 'Dynamic recurrent E-I coupling with delayed feedback.',
        'stop_condition': 'If opponent units produce argmax winner flips under positive scalar queries, re-evaluate routing capacity.'
    },
    {
        'claim_id': 'CLM-06',
        'exact_claim': 'The theoretical flip probability for isotropic 2D key pairs under uniform query angles follows the linear law P(flip) = theta / pi on S^1.',
        'model_family': 'Geometric Analysis Suite (Sprint 4.2-A, Theorem 5.2)',
        'input_domain': 'Isotropic unit vector pairs separated by angle theta in [0, pi] on S^{d-1} across d=2, 3, 4, 8',
        'assumptions': 'Uniform distribution of query vectors on S^1; d=2 minimal dimension for continuous geometry (Charikar STOC 2002)',
        'evidence_files_and_fields': 'results/sprint4_2/sprint4_2_summary.json, code/experiments/sprint4_2/sprint4_2_geometry.py, paper2/sections/s10_ledger.tex:49-53',
        'original_status': 'ESTABLISHED (Theorem 5.1 / flip law theta/pi; 132/132 isotropic conditions passed, 140 checks total; results/sprint4_2/sprint4_2_summary.json)',
        'review_status': 'RETAINED (REDUCED SCOPE)',
        'allowed_wording': 'Linear flip law theta/pi holds for isotropic 2D pairs (132/132 isotropic conditions passed, 140 checks total; max single-seed diff 0.0093, mean diff 0.0047); higher-dimensional projection maintains F_d = 0.5 (negative control); does not imply general multi-item permutation routing.',
        'counterexample_or_shortcut': 'Non-isotropic key distributions break linearity; F_d=0.5 in higher dimensions reflects subspace projection, not full permutation routing.',
        'uncompleted_checks': 'Non-uniform sphere integration.',
        'stop_condition': 'If analytical theta/pi deviates from numerical simulation beyond Monte Carlo sampling error, revise theoretical formulation.'
    },
    {
        'claim_id': 'CLM-07',
        'exact_claim': 'Dynamic TAN neuron implementation routes between candidates A and B across canonical queries Q=3.0 and Q=4.0.',
        'model_family': 'Full Vector TAN Neuron (Sprint 4.2-B Canonical Test)',
        'input_domain': 'Canonical input history [1, 0, 2, 0, Q] with Q in {3.0, 4.0}, omega=0.4',
        'assumptions': 'Query rotation theta = omega * S with omega=0.4; code implementation uses theta = omega * S (text double-omega formula was erratum ERR-01)',
        'evidence_files_and_fields': 'results/sprint4_2/sprint4_2b_summary.json, code/experiments/sprint4_2/sprint4_2_vector_qk.py, paper2/sections/s10_ledger.tex:54-57',
        'original_status': 'ESTABLISHED (routing margin flip +0.2587 -> -0.1559; ensemble flip rate 0.4425 vs analytic 0.4445; results/sprint4_2/sprint4_2b_summary.json)',
        'review_status': 'RETAINED (TYPO RESOLVED)',
        'allowed_wording': 'Neuron achieves canonical margin flip (+0.2587 at Q=3 -> Winner A; -0.1559 at Q=4 -> Winner B); requires code definition theta = omega * S; literal paper text formula theta = omega^2 * S would fail to flip (+0.7055 and +0.8426, both Winner A). Winner flip rate must not be conflated with semantic address correctness.',
        'counterexample_or_shortcut': 'Literal text formula theta = omega^2 * S prevents flip at Q=4; text erratum corrected in review copy.',
        'uncompleted_checks': 'None (verified numerically and text corrected).',
        'stop_condition': 'If code is altered to match the double-omega typo, routing fails immediately.'
    },
    {
        'claim_id': 'CLM-08',
        'exact_claim': 'Key-value representation decoupling allows TAN to achieve pairing-sensitive value transmission without amplitude leakage.',
        'model_family': 'M2 (Key-Value Decoupled TAN, Sprint 4.2-C)',
        'input_domain': 'Key-value binding sequences with canonical values V_A=5.0, V_B=9.0',
        'assumptions': 'Independent key and value channels; query slots carry zero value',
        'evidence_files_and_fields': 'results/sprint4_2/sprint4_2c_summary.json, code/experiments/sprint4_2/sprint4_2_binding_probe.py, paper2/sections/s10_ledger.tex:58-61',
        'original_status': 'ESTABLISHED (pairing-sensitive soft binding; T_swap = -T to 10^-12; D_ev > 0; results/sprint4_2/sprint4_2c_summary.json)',
        'review_status': 'RETAINED (BOUNDED)',
        'allowed_wording': 'Demonstrates pairing-sensitive value transmission with exact value-swap antisymmetry (T_swap = -T = +0.4129 to 10^-12); however, transmission is soft (D=1.74/1.84); continuous softmax cannot achieve exact symbolic delivery at finite temperature.',
        'counterexample_or_shortcut': 'Proposition 6.1 (Softmax Mixing Barrier): C_ev is in relative interior of convex hull for V_A != V_B; D > 0 strictly for all finite gamma.',
        'uncompleted_checks': 'Evaluation under non-zero query slot values.',
        'stop_condition': 'If value-swap symmetry is broken, binding decoupling fails.'
    },
    {
        'claim_id': 'CLM-09',
        'exact_claim': 'WTA hard selection (gamma -> infinity) achieves exact zero-error value binding; net E-I drive transfer restores value direction at gamma* = 5.098015.',
        'model_family': 'M2-WTA (Sprint 4.2-D)',
        'input_domain': 'Key-value pairs with one-hot argmax readout; sharpening parameter gamma in [1, inf)',
        'assumptions': 'Clean winner separation (margin > 0); winner-take-all argmax selection; bisection root gamma* = 5.098015 where T_drive = 0 (reversal crossing root)',
        'evidence_files_and_fields': 'results/sprint4_2/sprint4_2d_summary.json, code/experiments/sprint4_2/sprint4_2_hard_binding.py, paper2/sections/s10_ledger.tex:62-65',
        'original_status': 'ESTABLISHED (one-hot WTA limit achieves D=0; E-I direction reversal crossing root gamma* = 5.098015; results/sprint4_2/sprint4_2d_summary.json)',
        'review_status': 'RETAINED (CONSTRUCTIVE)',
        'allowed_wording': 'One-hot argmax selection delivers exact bound values conditioned on correct routing; does not fix upstream routing errors; E-I net drive transfer restores positive direction at gamma* = 5.098015 (bisection root where T_drive = 0); limited to tested mechanism family.',
        'counterexample_or_shortcut': 'If upstream routing selects the wrong winner, hard selection delivers the wrong value with zero blending.',
        'uncompleted_checks': 'Biological plausibility of infinite gamma limit in spiking units.',
        'stop_condition': 'If claimed to represent biological spiking WTA without dynamical proof, reject.'
    },
    {
        'claim_id': 'CLM-10',
        'exact_claim': 'Two-channel parallel binding combined with an explicit algebraic node achieves exact compositional arithmetic (y = f(V_A, V_B)).',
        'model_family': 'M2 Dual-Channel Composition (Sprint 4.3-A)',
        'input_domain': 'Dual-query sequences [A, 0, B, Q1, Q2]; addition (+) and subtraction (-) operations; 30,000 Monte Carlo histories',
        'assumptions': 'Constructive static retriever pipeline; exact algebraic combiner node; both-routed coverage 22.03% (benchmark 22.07%)',
        'evidence_files_and_fields': 'results/sprint4_3/audit_summary.json, results/sprint4_3/monte_carlo_results.json, code/experiments/sprint4_3/sprint4_3a_composition_probe.py, paper2/sections/s10_ledger.tex:69-73',
        'original_status': 'ESTABLISHED (constructive operator-level sufficiency under controlled 4.3-A protocol; E_comp|both = 0; 8 counterfactuals PASS; results/sprint4_3/audit_summary.json)',
        'review_status': 'DOWNGRADED TO CONSTRUCTIVE OPERATOR SUFFICIENCY',
        'allowed_wording': 'Constructive proof of operator sufficiency: dual parallel retrieval channels + explicit algebraic node can evaluate relations on the both-routed subset (coverage 22.03%, E_both = 0.0); unconditional error is non-zero (E_plus = 1.1077 MC / 1.1161 bench; E_minus = 1.9759 MC / 2.0010 bench); does NOT prove online continuous neural membrane dynamics or learned symbolic generalization.',
        'counterexample_or_shortcut': 'Sprint 4.3-A implementation uses direct arithmetic node (C1 + C2 if f==+ else C1 - C2), not recurrent neural dynamics; shortcut dict contains 5 structural True flags and 1 direct check(..., True).',
        'uncompleted_checks': 'Fully learned neural composition without explicit hardcoded algebraic operator.',
        'stop_condition': 'If claimed as proof of emergent or learned symbolic reasoning in recurrent spiking networks, reject immediately.'
    },
    {
        'claim_id': 'CLM-11',
        'exact_claim': '1D scalar metric distance attention is sufficient for content addressing under hard argmax; 2D vector rotation is not necessary for addressability; scalar dot-product kernels cannot address intermediate keys (0/6,705 hits).',
        'model_family': 'P1-A Suite (7 models: QueryBlindControl, PositiveScalarKernel, SignedScalarKernel, HistoricalSprint4_2_KernelStaticControl, FrozenSRotationHarmonicReference, ScalarMetricAttention, VectorQKAddressAttention)',
        'input_domain': 'Synthetic multi-slot key-value memory retrieval (6 conditions C1-C6, N=2 and N=3, noise sigma in {0.0, 0.01, 0.05}, OOD intervals [0.05, 0.3] and [2.5, 3.5], 10 seeds 2026091700-2026091709, 60,000 trials, 420,000 evaluations)',
        'assumptions': 'Static multi-slot memory buffer with distinct non-zero keys and values; distinct query cues; hard argmax winner selection vs soft readout',
        'evidence_files_and_fields': 'p1_experiments/results/run_p1a_rev2/summary_p1a.json, condition_summary.csv, post_write_recomputation_audit.json, g1_gate_report.json, P1A_FINAL_BASELINE_MANIFEST.md',
        'original_status': 'PROPOSED (RESEARCH_AUDIT_AND_PLAN_2026-09-17.md: Section 5 / P1-A)',
        'review_status': 'ESTABLISHED & FORMALLY CLOSED (Codex Sign-off 2026-09-19)',
        'allowed_wording': '1D metric distance compatibility (score = -(q - k)^2) is sufficient for single-neuron content addressing under hard argmax routing (100.00% both-routed accuracy on N=2 and N=3); 2D vector rotation on S^1 is not necessary for addressability; scalar dot-product kernels mathematically cannot address intermediate keys (0/6,705 hits on N=3). Results established under static multi-slot memory setting.',
        'counterexample_or_shortcut': 'ScalarMetricAttention achieves 100.00% routing accuracy identical to 2D Vector-QK across all 60,000 trials, refuting the necessity of 2D rotation for addressability; SignedScalarKernel achieves exactly 0/6,705 hits on intermediate keys; Historical 4.2-B control achieves only 13.68% accuracy on N=3.',
        'uncompleted_checks': 'None for static addressability (420,000 evaluations completed; lossless post-write recomputation audit passed with max diff < 1e-14; 5 failure injection fixtures passed; Codex formal closure sign-off obtained). Dynamic carrier modeling deferred to P1-B.',
        'stop_condition': 'Formally closed upon verified full replay, lossless audit, and multi-agent peer sign-off.'
    }
]

headers = [
    'claim_id', 'exact_claim', 'model_family', 'input_domain', 'assumptions',
    'evidence_files_and_fields', 'original_status', 'review_status',
    'allowed_wording', 'counterexample_or_shortcut', 'uncompleted_checks', 'stop_condition'
]

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path = os.path.join(base_dir, 'MASTER_CLAIM_LEDGER.csv')
with open(out_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)

print(f'MASTER_CLAIM_LEDGER.csv generated successfully with {len(rows)} claims at {out_path}')
