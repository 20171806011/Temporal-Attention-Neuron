"""Independent verification of Issue A03: Statistical unit and metric conflation.

Verifies:
1. K mean vs median from natural_collision_summary.csv
2. JSD full-window vs 2-candidate geometric example from sprint4_2b_summary.json
3. E3 maximum single-seed deviation vs averaged deviation across seeds, and theta_deg units from sprint4_2_summary.json
4. Probe-1 total samples vs distractor condition sample size (6000 total vs 2997 distractor)
5. 4.3-A composition: conditional zero error on both-routed subset (6608 / 30000) vs unconditional errors
"""
import os
import csv
import json
import numpy as np
from collections import defaultdict

def run_a03(base_dir):
    results = {}
    
    # 1. K mean vs median
    p1 = os.path.join(base_dir, 'results', 'tables', 'natural_collision_summary.csv')
    k_stats = {}
    with open(p1, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            m = r['model']
            k_stats[m] = {
                'K_mean': float(r['K_mean']),
                'K_median': float(r['K_median']),
                'K_ci_lo': float(r['K_mean_ci_lo']) if r['K_mean_ci_lo'] else None,
                'K_ci_hi': float(r['K_mean_ci_hi']) if r['K_mean_ci_hi'] else None
            }
    results['K_stats'] = k_stats
    
    # 2. JSD full window vs 2-candidate
    p2 = os.path.join(base_dir, 'results', 'sprint4_2', 'sprint4_2b_summary.json')
    with open(p2, 'r', encoding='utf-8') as f:
        data_s42b = json.load(f)
    jsd_m0 = data_s42b['results']['M0']['d2']['jsd']
    jsd_m2 = data_s42b['results']['M2']['d2']['jsd']
    results['JSD_full_window'] = {
        'M0': jsd_m0,
        'M2': jsd_m2
    }
    
    # 3. E3 deviations and theta_deg
    p3 = os.path.join(base_dir, 'results', 'sprint4_2', 'sprint4_2_summary.json')
    with open(p3, 'r', encoding='utf-8') as f:
        data_s42 = json.load(f)
    e3_checks = [c for c in data_s42['checks'] if c['experiment'] == 'E3']
    iso_checks = [c for c in e3_checks if c.get('case') == 'iso']
    control_checks = [c for c in e3_checks if c.get('case') != 'iso']
    single_seed_diffs = [abs(c['estimate'] - c['analytic']) for c in e3_checks]
    max_single_seed_diff = max(single_seed_diffs)
    
    grouped = defaultdict(list)
    for c in e3_checks:
        key = (c['d'], c['case'], c['pair'], c['theta_deg'])
        grouped[key].append(c['estimate'] - c['analytic'])
    mean_across_seeds_diffs = [abs(np.mean(vals)) for vals in grouped.values()]
    max_mean_across_seeds_diff = max(mean_across_seeds_diffs)
    sample_thetas = sorted(list(set([c['theta_deg'] for c in e3_checks if c['theta_deg'] is not None])))
    results['E3_stats'] = {
        'total_checks_count': len(e3_checks),
        'isotropic_conditions_count': len(iso_checks),
        'control_conditions_count': len(control_checks),
        'max_single_seed_diff': max_single_seed_diff,
        'max_mean_across_seeds_diff': max_mean_across_seeds_diff,
        'sample_thetas_theta_over_pi': sample_thetas[:6]
    }
    
def verify_k_stats(k_stats):
    assert abs(k_stats['lif']['K_mean'] - 0.5) < 1e-6 and abs(k_stats['lif']['K_median'] - 0.5) < 1e-6, "LIF K stats mismatch"
    assert abs(k_stats['buf']['K_mean'] - 4469.536489) < 1e-4, "B2 K mean mismatch"
    assert abs(k_stats['buf']['K_median'] - 895.529570) < 1e-4, "B2 K median mismatch"
    assert abs(k_stats['noattn']['K_mean'] - 12347.886007) < 1e-4, "B3 K mean mismatch"
    assert abs(k_stats['noattn']['K_median'] - 1369.079508) < 1e-4, "B3 K median mismatch"
    assert abs(k_stats['tan']['K_mean'] - 20465.855578) < 1e-4, "B4 K mean mismatch"
    assert abs(k_stats['tan']['K_median'] - 2947.724430) < 1e-4, "B4 K median mismatch"

def derive_probe1_counts(N_TEST=2000, N_CATCH_FRAC=1.0/3.0, num_seeds=3):
    n_catch = int(round(N_CATCH_FRAC * N_TEST))         # 667
    n_present = N_TEST - n_catch                         # 1333
    n_dist = n_present // 2                              # 666
    n_clean = n_present - n_dist                         # 667
    n_catch_dist = n_catch // 2                          # 333
    distractor_per_seed = n_dist + n_catch_dist          # 999
    total_test = num_seeds * N_TEST                      # 6000
    total_distractor = num_seeds * distractor_per_seed    # 2997
    return {
        'trials_per_seed': N_TEST,
        'num_seeds': num_seeds,
        'total_test_trials': total_test,
        'derived_distractor_per_seed': distractor_per_seed,
        'derived_distractor_total': total_distractor
    }

def verify_probe1_counts(derived):
    assert derived['total_test_trials'] == 6000, f"Expected 6000 total test, got {derived['total_test_trials']}"
    assert derived['derived_distractor_total'] == 2997, f"Expected 2997 distractor, got {derived['derived_distractor_total']}"
    assert derived['derived_distractor_per_seed'] == 999, f"Expected 999/seed, got {derived['derived_distractor_per_seed']}"

def verify_e3_counts(e3_checks, iso_checks, control_checks, max_single_seed_diff, max_mean_across_seeds_diff):
    assert len(e3_checks) == 140, f"Expected 140 E3 checks, got {len(e3_checks)}"
    assert len(iso_checks) == 132, f"Expected 132 isotropic E3 checks, got {len(iso_checks)}"
    assert len(control_checks) == 8, f"Expected 8 control E3 checks, got {len(control_checks)}"
    assert abs(max_single_seed_diff - 0.009266667) < 1e-7, f"E3 max single seed diff mismatch: {max_single_seed_diff}"
    assert abs(max_mean_across_seeds_diff - 0.004666667) < 1e-7, f"E3 max mean across seeds diff mismatch: {max_mean_across_seeds_diff}"

def verify_composition_results(benchmarks, mc, total_both_routed):
    assert total_both_routed == 6608, f"Expected 6608 both-routed histories, got {total_both_routed}"
    for s, v in mc.items():
        assert v['E_plus_both'] == 0.0, f"Non-zero E_plus_both in seed {s}"
        assert v['E_minus_both'] == 0.0, f"Non-zero E_minus_both in seed {s}"
    assert abs(benchmarks['E_comp_plus'] - 1.116099621) < 1e-7, f"Quadrature E_plus mismatch: {benchmarks['E_comp_plus']}"
    assert abs(benchmarks['E_comp_minus'] - 2.000999500) < 1e-7, f"Quadrature E_minus mismatch: {benchmarks['E_comp_minus']}"
    e_plus_vals = [v['E_plus'] for v in mc.values()]
    e_minus_vals = [v['E_minus'] for v in mc.values()]
    mc_pooled_mean_plus = float(np.mean(e_plus_vals))
    mc_pooled_mean_minus = float(np.mean(e_minus_vals))
    assert abs(mc_pooled_mean_plus - 1.107671595) < 1e-6, f"MC pooled E_plus mismatch: {mc_pooled_mean_plus}"
    assert abs(mc_pooled_mean_minus - 1.975892767) < 1e-6, f"MC pooled E_minus mismatch: {mc_pooled_mean_minus}"

def run_a03(base_dir):
    results = {}
    
    # 1. K mean vs median
    p1 = os.path.join(base_dir, 'results', 'tables', 'natural_collision_summary.csv')
    k_stats = {}
    with open(p1, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            m = r['model']
            k_stats[m] = {
                'K_mean': float(r['K_mean']),
                'K_median': float(r['K_median']),
                'K_ci_lo': float(r['K_mean_ci_lo']) if r['K_mean_ci_lo'] else None,
                'K_ci_hi': float(r['K_mean_ci_hi']) if r['K_mean_ci_hi'] else None
            }
    results['K_stats'] = k_stats
    verify_k_stats(k_stats)
    results['K_statistics_verified'] = True
    
    # 2. JSD full window vs 2-candidate
    p2 = os.path.join(base_dir, 'results', 'sprint4_2', 'sprint4_2b_summary.json')
    with open(p2, 'r', encoding='utf-8') as f:
        data_s42b = json.load(f)
    jsd_m0 = data_s42b['results']['M0']['d2']['jsd']
    jsd_m2 = data_s42b['results']['M2']['d2']['jsd']
    assert abs(jsd_m0 - 0.009462259) < 1e-7
    assert abs(jsd_m2 - 0.008726529) < 1e-7
    results['JSD_full_window'] = {
        'M0': jsd_m0,
        'M2': jsd_m2
    }
    
    # 3. E3 deviations and theta_deg
    p3 = os.path.join(base_dir, 'results', 'sprint4_2', 'sprint4_2_summary.json')
    with open(p3, 'r', encoding='utf-8') as f:
        data_s42 = json.load(f)
    e3_checks = [c for c in data_s42['checks'] if c['experiment'] == 'E3']
    iso_checks = [c for c in e3_checks if c.get('case') == 'iso']
    control_checks = [c for c in e3_checks if c.get('case') != 'iso']
    single_seed_diffs = [abs(c['estimate'] - c['analytic']) for c in e3_checks]
    max_single_seed_diff = max(single_seed_diffs)
    
    grouped = defaultdict(list)
    for c in e3_checks:
        key = (c['d'], c['case'], c['pair'], c['theta_deg'])
        grouped[key].append(c['estimate'] - c['analytic'])
    mean_across_seeds_diffs = [abs(np.mean(vals)) for vals in grouped.values()]
    max_mean_across_seeds_diff = max(mean_across_seeds_diffs)
    sample_thetas = sorted(list(set([c['theta_deg'] for c in e3_checks if c['theta_deg'] is not None])))
    
    verify_e3_counts(e3_checks, iso_checks, control_checks, max_single_seed_diff, max_mean_across_seeds_diff)
    results['E3_stats'] = {
        'total_checks_count': len(e3_checks),
        'isotropic_conditions_count': len(iso_checks),
        'control_conditions_count': len(control_checks),
        'max_single_seed_diff': max_single_seed_diff,
        'max_mean_across_seeds_diff': max_mean_across_seeds_diff,
        'sample_thetas_theta_over_pi': sample_thetas[:6]
    }
    
    # 4. Probe-1 sample size: Generative derivation from temporal_distractor.py
    derived_p1 = derive_probe1_counts(N_TEST=2000, N_CATCH_FRAC=1.0/3.0, num_seeds=3)
    verify_probe1_counts(derived_p1)
    results['Probe1_samples'] = derived_p1
    
    # 5. 4.3-A composition: Benchmark vs Monte Carlo Empirical Mean
    p5 = os.path.join(base_dir, 'results', 'sprint4_3', 'monte_carlo_results.json')
    with open(p5, 'r', encoding='utf-8') as f:
        data_s43 = json.load(f)
    mc = data_s43['mc']
    total_both_routed = sum([int(round(v['P_both'] * 10000)) for v in mc.values()])
    verify_composition_results(data_s43['benchmarks'], mc, total_both_routed)
    
    e_plus_vals = [v['E_plus'] for v in mc.values()]
    e_minus_vals = [v['E_minus'] for v in mc.values()]
    mc_pooled_mean_plus = float(np.mean(e_plus_vals))
    mc_pooled_mean_minus = float(np.mean(e_minus_vals))
    
    results['Composition_43A'] = {
        'total_histories': 30000,
        'both_routed_total': total_both_routed,
        'both_routed_coverage_fraction': total_both_routed / 30000.0,
        'both_routed_per_seed': {s: int(round(v['P_both'] * 10000)) for s, v in mc.items()},
        'error_plus_both': [v['E_plus_both'] for v in mc.values()],
        'error_minus_both': [v['E_minus_both'] for v in mc.values()],
        'mc_empirical_unconditional_plus_per_seed': e_plus_vals,
        'mc_empirical_unconditional_minus_per_seed': e_minus_vals,
        'mc_pooled_mean_plus': mc_pooled_mean_plus,
        'mc_pooled_mean_minus': mc_pooled_mean_minus,
        'quadrature_benchmark_unconditional_plus': data_s43['benchmarks']['E_comp_plus'],
        'quadrature_benchmark_unconditional_minus': data_s43['benchmarks']['E_comp_minus'],
        'quadrature_benchmark_P_both': data_s43['benchmarks']['P_both']
    }
    
    print("A03 Numerical Check Results:")
    print("  1. K stats (Mean vs Median):")
    for m, s in k_stats.items():
        print(f"     {m:8s}: Mean={s['K_mean']:12.4f}, Median={s['K_median']:10.4f}")
    print(f"  2. Full-window JSD: M0={jsd_m0:.9f}, M2={jsd_m2:.9f} (vs 2-candidate geometric 0.0751 / 0.0647)")
    print(f"  3. E3 check counts: total={len(e3_checks)}, isotropic={len(iso_checks)}, controls={len(control_checks)}")
    print(f"     E3 max single-seed diff: {max_single_seed_diff:.9f}")
    print(f"     E3 max mean-across-seeds diff: {max_mean_across_seeds_diff:.9f}")
    print(f"     theta_deg values (fraction of pi): {sample_thetas[:5]}")
    print(f"  4. Probe-1 (generative derivation): total test={derived_p1['total_test_trials']}, distractor={derived_p1['derived_distractor_total']} ({derived_p1['derived_distractor_per_seed']}/seed)")
    print(f"  5. 4.3-A Composition:")
    print(f"     Both-routed subset: {total_both_routed}/30000 ({total_both_routed/30000*100:.2f}%), E_both = 0.0")
    print(f"     Quadrature benchmarks: E_plus = {data_s43['benchmarks']['E_comp_plus']:.9f}, E_minus = {data_s43['benchmarks']['E_comp_minus']:.9f}")
    print(f"     30k MC pooled means:   E_plus = {mc_pooled_mean_plus:.9f}, E_minus = {mc_pooled_mean_minus:.9f}")
    print("ALL A03 ASSERTIONS PASSED!")
    return results

def test_a03_error_fixture(base_dir):
    """Negative testing: mutates real data inputs and verifies that the actual checker functions REJECT them."""
    print("Running A03 Error Fixtures (testing that real checker functions reject perturbed inputs)...")
    
    # 1. Perturb K mean vs median equality -> verify_k_stats MUST reject
    p1 = os.path.join(base_dir, 'results', 'tables', 'natural_collision_summary.csv')
    k_stats_corrupt = {}
    with open(p1, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            m = r['model']
            k_stats_corrupt[m] = {
                'K_mean': float(r['K_mean']),
                'K_median': float(r['K_median'])
            }
    k_stats_corrupt['buf']['K_median'] = k_stats_corrupt['buf']['K_mean']  # conflate median with mean
    try:
        verify_k_stats(k_stats_corrupt)
        raise RuntimeError("verify_k_stats failed to reject conflated mean/median!")
    except AssertionError:
        print("  [PASS] Conflated K mean/median correctly rejected by verify_k_stats")
        
    # 2. Perturb Probe-1 trials -> verify_probe1_counts MUST reject
    derived_corrupt = derive_probe1_counts(N_TEST=2500)
    try:
        verify_probe1_counts(derived_corrupt)
        raise RuntimeError("verify_probe1_counts failed to reject corrupted trial count!")
    except AssertionError:
        print("  [PASS] Corrupted Probe-1 trial count correctly rejected by verify_probe1_counts")
        
    # 3. Perturb E3 count -> verify_e3_counts MUST reject
    fake_e3 = [1] * 132  # missing 8 control checks
    fake_iso = [1] * 132
    fake_ctl = []
    try:
        verify_e3_counts(fake_e3, fake_iso, fake_ctl, 0.009266667, 0.004666667)
        raise RuntimeError("verify_e3_counts failed to reject truncated E3 checks list!")
    except AssertionError:
        print("  [PASS] Truncated E3 checks count correctly rejected by verify_e3_counts")
        
    # 4. Perturb composition benchmarks -> verify_composition_results MUST reject
    p5 = os.path.join(base_dir, 'results', 'sprint4_3', 'monte_carlo_results.json')
    with open(p5, 'r', encoding='utf-8') as f:
        data_s43 = json.load(f)
    corrupted_benchmarks = dict(data_s43['benchmarks'])
    corrupted_benchmarks['E_comp_plus'] = 1.107671595  # Erroneously replace benchmark with MC mean
    try:
        verify_composition_results(corrupted_benchmarks, data_s43['mc'], 6608)
        raise RuntimeError("verify_composition_results failed to reject benchmark/MC conflation!")
    except AssertionError:
        print("  [PASS] Benchmark vs MC mean conflation correctly rejected by verify_composition_results")
        
    print("ALL A03 ERROR FIXTURES PASSED (Real checker functions verified to reject corrupted data)!")

if __name__ == '__main__':
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'snapshot')
    run_a03(base)
    test_a03_error_fixture(base)
