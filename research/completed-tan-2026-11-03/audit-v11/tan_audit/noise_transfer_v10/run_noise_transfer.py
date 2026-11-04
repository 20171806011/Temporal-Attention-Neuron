#!/usr/bin/env python3
"""Frozen v10 finite-output audit. Use --output-dir for a separate replay.

TAN_REPO_ROOT overrides the sibling repository path. Missing .git is recorded
as unavailable; exact source and prior-artifact hashes remain mandatory.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import numpy as np
import scipy
from scipy.stats import beta, binom

BASE = Path(__file__).resolve().parent
V9 = BASE.parent / 'noise_preservation_v9'
PROTOCOL = '14ef73c1418b4dcea2a5a55de20391e3a8c82925bef70741ee3033ebcbfca179'
COMMIT = '85213f0d417d54e4f48b301ed1c0e4200ef3c421'
KEEP = 529 / 1013
N = 2048
TOL = 1e-12
ARMS = ('TAN', 'LIF', 'LIF_THINNED', 'ZERO_OUTPUT')
CONDITIONS = ('step', 'catch')
METRICS = ('C_H', 'C_H_given_S', 'C_I_signal', 'C_U_deletion', 'C_I_path',
           'D_H', 'D_H_given_S', 'D_I_signal', 'D_U_deletion', 'D_I_path',
           'Delta_signal', 'Delta_path', 'Delta_deletion')
RESOURCE_FIELDS = ('sum_abs_current', 'sum_current_squared', 'sum_candidate_squared',
                   'peak_abs_current', 'peak_candidate', 'peak_abs_candidate',
                   'peak_membrane', 'peak_abs_membrane')
LEDGER_FIELDS = ('history_id', 'condition', 'S', 'arm', 'underlying_model',
                 'underlying_onset_count', 'count_type',
                 'P_C0', 'P_C1', 'P_C2', 'P_C3', 'P_C4', 'P_C5', 'P_D1',
                 'observed_total_count', 'observed_onset_count', 'observed_late_count',
                 'R', 'generated_total_count', 'generated_onset_count', 'generated_late_count') + RESOURCE_FIELDS
V9_FIXED = {
    'run_noise_preservation.py': 'bd08b8aeef0ebdef4f69bba1e4efb8d9d4cb916800c0660f60c2bfd59c0e017f',
    'original_results.json': 'cc15aadf6a75264e4627d442e87b99f034e5e074121a5a462fc5db6e36f76ee8',
    'clean_results.json': 'd9fca23d47bb25085710b08d4735a65af278b4df100e339fadbb62f5ed662998',
    'PRIMARY_MANIFEST.json': '2eccf54e95bf4fc4b27dc8e5bda506edafb3df0543185f6f2c738c1118f67e93',
    'INDEPENDENT_REVIEW_CHECKS.json': '0ee9cdb8c4ab89852b56074a36c8825741340e104c61857bf90a7c93843828ee',
    'taskb_original_extracted.py': 'c014426b5d8321fd7bde4186660fb4132fe2f5f90782232b10e404bd330e0d9c',
    'PREREGISTRATION.md': '70a9fcebb77a223e321e34fa11eb18f1228431c3d3ef8ac40cfc1a4706eb87e1',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def csv_write(path, fields, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def status(repo):
    if not (repo / '.git').exists():
        return {'available': False, 'commit': None, 'status': None}
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    state = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain=v1'], text=True)
    assert commit == COMMIT and state == '', (commit, state)
    return {'available': True, 'commit': commit, 'status': state}


def source_guard(repo, out):
    assert sha(BASE / 'PREREGISTRATION.md') == PROTOCOL
    for name, digest in V9_FIXED.items():
        assert sha(V9 / name) == digest, ('v9 frozen artifact', name)
    prior = json.loads((V9 / 'PRIMARY_MANIFEST.json').read_text())
    assert prior['status'] == 'COMPLETED'
    prior_files = {}
    for name, record in prior['files'].items():
        actual = sha(V9 / name)
        assert actual == record['sha256'], ('v9 declared artifact', name)
        prior_files[name] = actual
    prior_review = json.loads((V9 / 'INDEPENDENT_REVIEW_CHECKS.json').read_text())
    assert prior_review['status'] == 'PASS' and prior_review['primary_artifacts_unchanged']
    original = json.loads((V9 / 'original_results.json').read_text())
    assert original['phase_A_completion']['exact_raw_record_matches'] == 100
    manifest = json.loads((V9 / 'NOISE_SOURCE_MANIFEST.json').read_text())
    source_hashes = {}
    for relative, record in manifest['source_files'].items():
        source_hashes[relative] = sha(repo / relative)
        assert source_hashes[relative] == record['sha256'], relative
    raw = (repo / 'snapshot/code/experiments/statistical_validation.py').read_text()
    extracted = (V9 / 'taskb_original_extracted.py').read_text()
    nodes = {node.name: node for node in ast.parse(raw).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    copied = {node.name: node for node in ast.parse(extracted).body
              if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    for record in manifest['selected_source_nodes']:
        name = record['name']
        segment = ast.get_source_segment(raw, nodes[name])
        assert segment == ast.get_source_segment(extracted, copied[name])
        assert hashlib.sha256(segment.encode()).hexdigest() == record['exact_segment_sha256']
        assert ast.dump(nodes[name], include_attributes=False) == ast.dump(copied[name], include_attributes=False)
    namespace = {'np': np}
    tree = ast.Module(body=[copied[name] for name in ('LIFNeuron', 'TANNeuron')], type_ignores=[])
    exec(compile(tree, '<verified-original-native-classes>', 'exec'), namespace)
    spec = importlib.util.spec_from_file_location('audited_v9_primary', V9 / 'run_noise_preservation.py')
    helper = importlib.util.module_from_spec(spec)
    previous_bytecode_policy = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(helper)
    finally:
        sys.dont_write_bytecode = previous_bytecode_policy
    guard = {'time_utc': utc(), 'v10_protocol_sha256': PROTOCOL,
             'v9_fixed_artifact_sha256': V9_FIXED, 'v9_declared_files_sha256': prior_files,
             'source_files_sha256': source_hashes,
             'v9_source_manifest_sha256': sha(V9 / 'NOISE_SOURCE_MANIFEST.json'),
             'original_100_records_previously_reproduced': True,
             'original_100_records_rerun_in_v10': False,
             'all_guards_passed': True,
             'model_instrumentation': 'Audited v9 simulate_native; exact source classes checked every update.',
             'repository_provenance': status(repo)}
    json_write(out / 'SOURCE_GUARD.json', guard)
    return namespace, helper, guard


def entropy(p):
    p = np.asarray(p, dtype=float)
    assert np.min(p) >= 0 and np.isfinite(p).all()
    logs = np.zeros_like(p)
    np.log2(p, out=logs, where=p > 0)
    return -(p * logs).sum(axis=-1)


def h2(p):
    return float(entropy(np.array([1-p, p])))


def kernel(keep):
    return np.array([[math.comb(k, j)*keep**j*(1-keep)**(k-j) if j <= k else 0.
                      for j in range(6)] for k in range(6)], dtype=float)


def counts_from_traces(traces):
    return {arm: traces[arm + '__spikes'][:, :, 15:20].sum(axis=2).astype(np.int64)
            for arm in ('TAN', 'LIF')}


def laws_from_counts(counts, B):
    law = {}
    for arm in ARMS:
        if arm == 'ZERO_OUTPUT':
            underlying = np.full_like(counts['LIF'], -1)
            cpmf = np.zeros(underlying.shape + (6,))
            cpmf[..., 0] = 1.
        else:
            native = 'LIF' if arm == 'LIF_THINNED' else arm
            underlying = counts[native]
            cpmf = B[underlying] if arm == 'LIF_THINNED' else np.eye(6)[underlying]
        # D=1[C>=1] is a deterministic coarsening of the reported count law.
        dpmf = np.stack((cpmf[..., 0], cpmf[..., 1:].sum(axis=-1)), axis=-1)
        law.update({arm + '__underlying_count': underlying,
                    arm + '__count_pmf': cpmf, arm + '__decision_pmf': dpmf})
    return law


def histograms(counts, indices=None):
    result = {}
    for arm in ('TAN', 'LIF'):
        selected = counts[arm] if indices is None else counts[arm][indices]
        result[arm] = np.stack([np.bincount(selected[:, c], minlength=6)/len(selected)
                                for c in range(2)])
    return result


def calculate_information(hist, B):
    pmfs, values = {}, {}
    q = 1 - (1-KEEP)**np.arange(6)
    kernel_C_entropy = entropy(B)
    kernel_D_entropy = entropy(np.column_stack((1-q, q)))
    for arm in ARMS:
        if arm == 'LIF_THINNED':
            C = hist['LIF'] @ B
            uc = float((hist['LIF'] @ kernel_C_entropy).mean())
            ud = float((hist['LIF'] @ kernel_D_entropy).mean())
        elif arm == 'ZERO_OUTPUT':
            C = np.zeros((2, 6)); C[:, 0] = 1.
            uc = ud = 0.
        else:
            C = hist[arm]
            uc = ud = 0.
        D = np.column_stack((C[:, 0], C[:, 1:].sum(axis=1)))
        cmarg, dmarg = C.mean(axis=0), D.mean(axis=0)
        ch, dh = float(entropy(cmarg)), float(entropy(dmarg))
        chs, dhs = float(entropy(C).mean()), float(entropy(D).mean())
        cs, ds, cn, dn = ch-chs, dh-dhs, chs-uc, dhs-ud
        vector = np.array([ch, chs, cs, uc, cn, dh, dhs, ds, ud, dn,
                           cs-ds, cn-dn, uc-ud])
        assert np.isfinite(vector).all()
        pmfs.update({arm+'__C_conditional': C, arm+'__C_marginal': cmarg,
                     arm+'__D_conditional': D, arm+'__D_marginal': dmarg})
        values[arm] = vector
    return pmfs, values


def invariants(pmfs, values):
    residuals = {}
    for arm in ARMS:
        v = dict(zip(METRICS, values[arm]))
        for obs, size in (('C', 6), ('D', 2)):
            conditional = pmfs[arm+'__'+obs+'_conditional']
            marginal = pmfs[arm+'__'+obs+'_marginal']
            residuals[arm+'_'+obs+'_pmf_normalization'] = float(np.max(abs(conditional.sum(axis=1)-1)))
            residuals[arm+'_'+obs+'_chain_closure'] = v[obs+'_H']-(v[obs+'_I_signal']+v[obs+'_I_path']+v[obs+'_U_deletion'])
            assert residuals[arm+'_'+obs+'_pmf_normalization'] <= TOL
            assert abs(residuals[arm+'_'+obs+'_chain_closure']) <= TOL
            assert abs(float(marginal.sum())-1) <= TOL
            assert -TOL <= v[obs+'_H'] <= math.log2(size)+TOL
            assert -TOL <= v[obs+'_I_signal'] <= 1+TOL
            assert v[obs+'_I_path'] >= -TOL and v[obs+'_U_deletion'] >= -TOL
        for name in ('Delta_signal', 'Delta_path', 'Delta_deletion'):
            assert v[name] >= -TOL, (arm, name, v[name])
        residuals[arm+'_count_detail_chain_closure'] = (
            v['Delta_signal']+v['Delta_path']+v['Delta_deletion']-(v['C_H']-v['D_H']))
        assert abs(residuals[arm+'_count_detail_chain_closure']) <= TOL
        if arm in ('TAN', 'LIF', 'ZERO_OUTPUT'):
            assert v['C_U_deletion'] == 0 and v['D_U_deletion'] == 0
        D = pmfs[arm+'__D_conditional']
        tpr, fpr = float(D[0, 1]), float(D[1, 1])
        formula = h2(0.5*(tpr+fpr)) - 0.5*(h2(tpr)+h2(fpr))
        residuals[arm+'_binary_MI_formula'] = v['D_I_signal']-formula
        assert abs(residuals[arm+'_binary_MI_formula']) <= TOL
    assert np.max(abs(values['ZERO_OUTPUT'])) == 0.
    # Valid chain: native LIF count -> thinned count -> thinned decision.
    for index in (2, 4):
        gap = float(values['LIF'][index]-values['LIF_THINNED'][index])
        residuals['LIF_count_minus_thinned_'+METRICS[index]] = gap
        assert gap >= -TOL
    return residuals


def formula_checks(B):
    assert np.max(abs(B.sum(axis=1)-1)) <= TOL
    zero_kernel = kernel(0.)
    expected_zero = np.zeros((6, 6)); expected_zero[:, 0] = 1.
    assert np.array_equal(zero_kernel, expected_zero)
    assert np.array_equal(kernel(1.), np.eye(6))
    assert np.array_equal(B[0], expected_zero[0])
    oracle = np.zeros((2, 6)); oracle[0, 1] = 1.; oracle[1, 0] = 1.
    O = np.column_stack((oracle[:, 0], oracle[:, 1:].sum(axis=1)))
    assert entropy(oracle.mean(axis=0)) == 1 and entropy(oracle).mean() == 0
    assert entropy(O.mean(axis=0)) == 1 and entropy(O).mean() == 0
    q = 1-(1-KEEP)**np.arange(6)
    assert np.max(abs(B[:, 1:].sum(axis=1)-q)) <= TOL
    return {'kernel_normalization_raw_residuals': (B.sum(axis=1)-1).tolist(),
             'kernel_D_probability_raw_residuals': (B[:, 1:].sum(axis=1)-q).tolist(),
             'p0_kernel_exact': True, 'p1_kernel_exact': True,
             'label_oracle_formula_only_signal_bits': 1.,
             'label_oracle_formula_only_conditional_entropy': 0.,
             'zero_count_kernel_exact': True}


def ledger_rows(traces, laws, conditions, ids):
    rows = []
    for i, identity in enumerate(ids):
        for c, condition in enumerate(conditions):
            for arm in ARMS:
                row = dict.fromkeys(LEDGER_FIELDS)
                cp = laws[arm+'__count_pmf'][i, c]
                row.update(history_id=int(identity), condition=condition, S=int(condition == 'step'),
                           arm=arm, count_type='expectation' if arm == 'LIF_THINNED' else 'actual',
                           underlying_model=None if arm == 'ZERO_OUTPUT' else ('LIF' if arm == 'LIF_THINNED' else arm),
                           underlying_onset_count=int(laws[arm+'__underlying_count'][i, c]),
                           P_D1=float(laws[arm+'__decision_pmf'][i, c, 1]))
                row.update({'P_C'+str(j): float(cp[j]) for j in range(6)})
                if arm == 'ZERO_OUTPUT':
                    row.update(observed_total_count=0., observed_onset_count=0., observed_late_count=0., R=0.)
                else:
                    native = row['underlying_model']
                    spikes = traces[native+'__spikes'][i, c]
                    gain = KEEP if arm == 'LIF_THINNED' else 1.
                    total, onset, late = int(spikes.sum()), int(spikes[15:20].sum()), int(spikes[20:60].sum())
                    current, candidate, membrane = (traces[native+'__'+f][i, c]
                                                    for f in ('current', 'candidate', 'membrane'))
                    row.update(observed_total_count=gain*total, observed_onset_count=gain*onset,
                               observed_late_count=gain*late, R=gain*late/40,
                               generated_total_count=total, generated_onset_count=onset, generated_late_count=late,
                               sum_abs_current=float(abs(current).sum()),
                               sum_current_squared=float(np.square(current).sum()),
                               sum_candidate_squared=float(np.square(candidate).sum()),
                               peak_abs_current=float(abs(current).max()), peak_candidate=float(candidate.max()),
                               peak_abs_candidate=float(abs(candidate).max()), peak_membrane=float(membrane.max()),
                               peak_abs_membrane=float(abs(membrane).max()))
                assert abs(float(cp @ np.arange(6))-row['observed_onset_count']) <= TOL
                rows.append(row)
    return rows


def describe(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()), 'min': float(values.min()), 'max': float(values.max()),
             'sample_sd': float(values.std(ddof=1)) if len(values)>1 else None}


def activity_summary(rows, clean_rows):
    result = {}
    for arm in ARMS:
        cells = {}
        for condition in CONDITIONS:
            selected = [r for r in rows if r['arm'] == arm and r['condition'] == condition]
            fields = ('observed_total_count', 'observed_onset_count', 'observed_late_count', 'R')
            cells[condition] = {field: describe([r[field] for r in selected]) for field in fields}
            cells[condition]['generated_total_count'] = (None if arm == 'ZERO_OUTPUT' else
                describe([r['generated_total_count'] for r in selected]))
            cells[condition]['resource_proxies'] = (None if arm == 'ZERO_OUTPUT' else
                {field: describe([r[field] for r in selected]) for field in RESOURCE_FIELDS})
        clean_R = next(r['R'] for r in clean_rows if r['arm']==arm and r['condition']=='step')
        result[arm] = {'conditions': cells, 'clean_step_R': clean_R,
                       'noisy_minus_clean_step_R': cells['step']['R']['mean']-clean_R}
    return result


def certificate(counts, information):
    alpha_tv, alpha_cp = .025, .025
    a = alpha_cp/4
    delta = math.sqrt(math.log(4*(2**6-2)/alpha_tv)/(2*N))
    f6 = h2(delta)+delta*math.log2(5)
    assert delta <= 5/6
    result = {'n_per_native_condition': N, 'alpha_TV': alpha_tv, 'alpha_CP': alpha_cp,
               'alpha_CP_cell': a, 'joint_confidence_at_least': .95,
               'TV_radius': delta, 'F6_entropy_modulus_bits': f6,
               'subsets_per_distribution': 62, 'native_conditional_distributions': 4,
               'native_arms': {}}
    for arm in ('TAN', 'LIF'):
        errors = {'miss': int((counts[arm][:, 0] == 0).sum()),
                  'FPR': int((counts[arm][:, 1] >= 1).sum())}
        cells = {}
        for name, k in errors.items():
            if k == 0:
                upper = -math.expm1(math.log(a)/N)
                method = 'exact_zero_error_expm1'
            elif k == N:
                upper = 1.
                method = 'all_errors_boundary'
            else:
                upper = float(beta.ppf(1-a, k+1, N-k))
                method = 'one_sided_beta_quantile'
            cdf = float(binom.cdf(k, N, upper))
            residual = cdf-a if k < N else None
            if k < N:
                assert abs(residual) <= TOL, (arm, name, residual)
            cells[name] = {'errors': k, 'n': N, 'upper_probability': upper,
                            'method': method, 'binomial_cdf_at_upper': cdf,
                            'cdf_minus_alpha_cell': residual}
        eu = (cells['miss']['upper_probability']+cells['FPR']['upper_probability'])/2
        b = h2(eu) if eu <= .5 else 1.
        chs = float(information[arm][1])
        raw_lower = chs-f6-b
        result['native_arms'][arm] = {
            'error_cells': cells, 'error_upper_balanced': eu, 'b_bits': b,
            'Hhat_C_given_S': chs, 'I_signal_D_lower_bound': 1-b,
            'Delta_signal_lower_bound': 0., 'Delta_signal_upper_bound': b,
            'Delta_path_raw_lower_bound': raw_lower,
            'Delta_path_lower_bound': max(0., raw_lower)}
    return result


def bootstrap(counts, B, point, out):
    draws = np.random.RandomState(2026110241).randint(0, N, size=(2000, N))
    np.save(out / 'bootstrap_draws.npy', draws)
    metrics = {arm: np.empty((2000, len(METRICS))) for arm in ARMS}
    laws = {arm+'__'+obs+'_conditional': np.empty((2000, 2, size))
             for arm in ARMS for obs, size in (('C', 6), ('D', 2))}
    largest_closure = 0.
    for r, indices in enumerate(draws):
        pmfs, values = calculate_information(histograms(counts, indices), B)
        residuals = invariants(pmfs, values)
        largest_closure = max(largest_closure, max(abs(value) for key, value in residuals.items()
                                                   if 'closure' in key or 'formula' in key))
        for arm in ARMS:
            metrics[arm][r] = values[arm]
            for obs in ('C', 'D'):
                key = arm+'__'+obs+'_conditional'
                laws[key][r] = pmfs[key]
    np.savez_compressed(out / 'bootstrap_metrics.npz', metric_names=np.asarray(METRICS), **metrics)
    np.savez_compressed(out / 'bootstrap_pmfs.npz', **laws)
    rows = [{'replicate': r, 'arm': arm,
              **{name: float(metrics[arm][r, j]) for j, name in enumerate(METRICS)}}
             for r in range(2000) for arm in ARMS]
    csv_write(out / 'bootstrap_metrics.csv', ('replicate', 'arm')+METRICS, rows)
    summary = {arm: {name: {'estimate': float(point[arm][j]),
                            'bootstrap_mean': float(metrics[arm][:, j].mean()),
                            'bootstrap_bias_diagnostic': float(metrics[arm][:, j].mean()-point[arm][j]),
                            'percentile_95_ci': np.quantile(metrics[arm][:, j], [.025, .975]).tolist()}
                      for j, name in enumerate(METRICS)} for arm in ARMS}
    return summary, largest_closure


def run(namespace, helper, out):
    noise = np.random.RandomState(2026110240).normal(0, .4, size=(N, 60))
    base = np.stack((np.r_[np.zeros(15), np.full(45, 1.5)], np.zeros(60)))
    inputs = {'noise': noise, 'base': base, 'waveforms': np.maximum(0., noise[:, None, :]+base[None, :, :]),
               'history_ids': np.arange(N), 'conditions': np.asarray(CONDITIONS), 'S': np.array([1, 0])}
    clean = {'noise': np.zeros((1, 60)), 'base': base, 'waveforms': base[None, :, :].copy(),
              'history_ids': np.arange(1), 'conditions': np.asarray(CONDITIONS), 'S': np.array([1, 0])}
    np.savez_compressed(out / 'fresh_inputs.npz', **inputs)
    np.savez_compressed(out / 'clean_inputs.npz', **clean)
    frozen = {'time_utc': utc(), 'native_models_evaluated_yet': False,
               'noise_seed': 2026110240, 'n_histories': N,
               'fresh_inputs_sha256': sha(out / 'fresh_inputs.npz'),
               'clean_inputs_sha256': sha(out / 'clean_inputs.npz'),
               'fresh_array_manifest': helper.array_manifest(inputs),
               'clean_array_manifest': helper.array_manifest(clean)}
    json_write(out / 'INPUTS_FROZEN.json', frozen)
    print('INPUTS FROZEN', frozen['fresh_inputs_sha256'], flush=True)
    B = kernel(KEEP)
    formula = formula_checks(B)
    np.savez_compressed(out / 'thinning_kernel.npz', count_kernel=B,
                         decision_success=1-(1-KEEP)**np.arange(6),
                         count_kernel_entropy=entropy(B),
                         decision_kernel_entropy=entropy(np.column_stack(((1-KEEP)**np.arange(6),
                                                                         1-(1-KEEP)**np.arange(6)))))
    clean_traces, clean_checks = helper.simulate_native(clean['waveforms'], CONDITIONS, [0], namespace)
    np.savez_compressed(out / 'clean_trajectories.npz', **clean_traces)
    clean_max_error = 0.
    with np.load(V9 / 'clean_trajectories.npz', allow_pickle=False) as old:
        assert set(old.files) == set(clean_traces)
        for name, value in clean_traces.items():
            if value.dtype.kind in 'biuf':
                error = float(np.max(abs(value.astype(float)-old[name].astype(float))))
                clean_max_error = max(clean_max_error, error)
                assert error <= TOL
                if 'spike' in name or value.dtype.kind in 'biu':
                    assert np.array_equal(value, old[name])
            else:
                assert np.array_equal(value, old[name])
    assert np.array_equal(np.flatnonzero(clean_traces['LIF__spikes'][0, 0]), np.arange(16, 60, 2))
    clean_counts = counts_from_traces(clean_traces)
    clean_laws = laws_from_counts(clean_counts, B)
    np.savez_compressed(out / 'clean_laws.npz', **clean_laws)
    clean_rows = ledger_rows(clean_traces, clean_laws, CONDITIONS, [0])
    csv_write(out / 'clean_ledger.csv', LEDGER_FIELDS, clean_rows)
    clean_pmfs, clean_information = calculate_information(histograms(clean_counts), B)
    clean_invariants = invariants(clean_pmfs, clean_information)
    np.savez_compressed(out / 'clean_pmfs.npz', **clean_pmfs)
    json_write(out / 'clean_results.json', {
        'native_trajectories': 4, 'no_replicate_interpretation': True,
        'comparison_with_v9_max_abs_error': clean_max_error,
        'native_trace_checks': clean_checks, 'invariants': clean_invariants,
        'information': {arm: dict(zip(METRICS, map(float, clean_information[arm]))) for arm in ARMS}})
    traces, trace_checks = helper.simulate_native(inputs['waveforms'], CONDITIONS, range(N), namespace)
    np.savez_compressed(out / 'fresh_trajectories.npz', **traces)
    counts = counts_from_traces(traces)
    laws = laws_from_counts(counts, B)
    np.savez_compressed(out / 'fresh_laws.npz', **laws)
    rows = ledger_rows(traces, laws, CONDITIONS, range(N))
    assert len(rows) == 16384
    csv_write(out / 'fresh_ledger.csv', LEDGER_FIELDS, rows)
    hist = histograms(counts)
    pmfs, information = calculate_information(hist, B)
    checks = invariants(pmfs, information)
    max_law_average_error = 0.
    for arm in ARMS:
        for obs, suffix in (('C', 'count_pmf'), ('D', 'decision_pmf')):
            error = float(np.max(abs(laws[arm+'__'+suffix].mean(axis=0)-pmfs[arm+'__'+obs+'_conditional'])))
            max_law_average_error = max(max_law_average_error, error)
            assert error <= TOL
    np.savez_compressed(out / 'point_pmfs.npz', **pmfs)
    point_rows = [{'arm': arm, **dict(zip(METRICS, map(float, information[arm])))} for arm in ARMS]
    csv_write(out / 'point_information.csv', ('arm',)+METRICS, point_rows)
    certificate_result = certificate(counts, information)
    json_write(out / 'population_certificate.json', certificate_result)
    boot_summary, boot_residual = bootstrap(counts, B, information, out)
    json_write(out / 'bootstrap_summary.json', boot_summary)
    diagnostics = {}
    for arm in ('TAN', 'LIF'):
        diagnostics[arm] = {}
        for c, condition in enumerate(CONDITIONS):
            frequencies = np.bincount(counts[arm][:, c], minlength=6)
            diagnostics[arm][condition] = {
                'frequencies': frequencies.tolist(), 'alphabet': list(range(6)),
                'empty_bins': np.flatnonzero(frequencies == 0).tolist(),
                'singleton_bins': np.flatnonzero(frequencies == 1).tolist()}
    activity = activity_summary(rows, clean_rows)
    detection = {arm: {'TPR': float(pmfs[arm+'__D_conditional'][0, 1]),
                       'FPR': float(pmfs[arm+'__D_conditional'][1, 1]),
                       'BA': float(.5*(pmfs[arm+'__D_conditional'][0, 1]+pmfs[arm+'__D_conditional'][1, 0]))}
                 for arm in ARMS}
    results = {'time_utc': utc(), 'protocol_sha256': PROTOCOL, 'n_histories': N,
        'n_native_fresh_trajectories': 8192, 'n_native_clean_trajectories': 4,
        'n_ledger_rows': len(rows), 'no_exclusions': True,
        'noise_seed': 2026110240, 'bootstrap_seed': 2026110241, 'bootstrap_replicates': 2000,
        'keep_probability': KEEP, 'new_deletion_masks_drawn': False,
        'condition_order': list(CONDITIONS), 'condition_S_values': [1, 0],
        'balanced_S_prior': [.5, .5], 'onset_time': 15, 'onset_time_entropy': 0.,
        'count_alphabet': list(range(6)), 'metric_order': list(METRICS),
        'information': {arm: dict(zip(METRICS, map(float, information[arm]))) for arm in ARMS},
        'detection_context': detection, 'native_count_frequencies': diagnostics,
        'activity_and_resource_context': activity,
        'fresh_late_budget_mismatch_TAN_minus_thinned_LIF': activity['TAN']['conditions']['step']['R']['mean']-activity['LIF_THINNED']['conditions']['step']['R']['mean'],
        'native_trace_checks': trace_checks, 'clean_trace_checks': clean_checks,
        'clean_v9_max_abs_error': clean_max_error, 'raw_invariant_residuals': checks,
        'max_law_average_vs_histogram_map_error': max_law_average_error,
        'bootstrap_max_chain_or_binary_formula_residual': boot_residual,
        'formula_checks': formula, 'population_certificate': certificate_result,
        'bootstrap_summary_file': 'bootstrap_summary.json', 'point_pmfs_file': 'point_pmfs.npz',
        'interpretation_limits': [
            'Ordered conditional information; not unique causal allocation of individual spikes.',
            'Balanced step/catch ensemble extends original always-step Task B.',
            'External known-time count-to-decision coarsening; no timing-performance claim.',
            'Fixed count alphabet omits spike timing; no complete-pattern or channel-capacity claim.',
            'Thinning integrated analytically; reported observed counts are expectations.',
            'No termwise bias correction or clipping of raw roundoff; bootstrap intervals are descriptive.',
            'Joint population certificate applies to both native arms only.',
            'Resource proxies are dimensionless; no matched physical energy or bits-per-energy claim.']}
    json_write(out / 'primary_results.json', results)
    for arm in ARMS:
        print(arm, json.dumps(results['information'][arm]), flush=True)
    print('NATIVE POPULATION CERTIFICATES', json.dumps(certificate_result['native_arms']), flush=True)
    return results


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file
    def write(self, value):
        self.stream.write(value); self.file.write(value); self.file.flush()
    def flush(self):
        self.stream.flush(); self.file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=BASE)
    out = parser.parse_args().output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'INPUTS_FROZEN.json').exists() or (out/'PRIMARY_MANIFEST.json').exists():
        raise RuntimeError('Preserving earlier results: use a new --output-dir to replay.')
    repo = Path(os.environ.get('TAN_REPO_ROOT', BASE.parents[1]/'Temporal-Attention-Neuron')).resolve()
    started = utc()
    oldout, olderr = sys.stdout, sys.stderr
    with (out/'run.log').open('w') as logfile:
        sys.stdout, sys.stderr = Tee(oldout, logfile), Tee(olderr, logfile)
        try:
            before = status(repo)
            json_write(out/'repo_status_before.json', before)
            print('START', started, 'command', sys.argv, 'repository', before, flush=True)
            namespace, helper, guard = source_guard(repo, out)
            run(namespace, helper, out)
            after = status(repo)
            json_write(out/'repo_status_after.json', after)
            assert before == after
            for name, digest in V9_FIXED.items():
                assert sha(V9/name) == digest
            for relative, digest in guard['source_files_sha256'].items():
                assert sha(repo/relative) == digest
            completed = utc()
            print('COMPLETED', completed, 'repository', after, flush=True)
            names = ['SOURCE_GUARD.json', 'fresh_inputs.npz', 'clean_inputs.npz', 'INPUTS_FROZEN.json',
                     'thinning_kernel.npz', 'clean_trajectories.npz', 'clean_laws.npz', 'clean_ledger.csv',
                     'clean_pmfs.npz', 'clean_results.json', 'fresh_trajectories.npz', 'fresh_laws.npz',
                     'fresh_ledger.csv', 'point_pmfs.npz', 'point_information.csv', 'population_certificate.json',
                     'bootstrap_draws.npy', 'bootstrap_metrics.npz', 'bootstrap_pmfs.npz',
                     'bootstrap_metrics.csv', 'bootstrap_summary.json', 'primary_results.json',
                     'repo_status_before.json', 'repo_status_after.json', 'run.log']
            manifest = {'status': 'COMPLETED', 'started_utc': started, 'completed_utc': completed,
                'execution_command': sys.argv, 'python': platform.python_version(),
                'numpy': np.__version__, 'scipy': scipy.__version__,
                'protocol_sha256': PROTOCOL, 'primary_script_sha256': sha(__file__),
                'audited_v9_helper_sha256': V9_FIXED['run_noise_preservation.py'],
                'repository': str(repo), 'repository_provenance': after,
                'repository_status_unchanged': True if after['available'] else None,
                'conditions': list(CONDITIONS), 'S': [1, 0], 'arms': list(ARMS),
                'metric_order': list(METRICS), 'ledger_fields': list(LEDGER_FIELDS),
                'trajectory_axes': 'history, condition, time[, history_slot]',
                'law_axes': 'history, condition, alphabet_value',
                'underlying_count_zero_output_sentinel': -1,
                'float_tolerance': TOL, 'files': {}, 'npz_schema': {}}
            for name in names:
                path = out/name
                manifest['files'][name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
                if name.endswith('.npz'):
                    with np.load(path, allow_pickle=False) as arrays:
                        manifest['npz_schema'][name] = {key: {'shape': list(arrays[key].shape),
                            'dtype': str(arrays[key].dtype)} for key in arrays.files}
            json_write(out/'PRIMARY_MANIFEST.json', manifest)
        except Exception as exc:
            json_write(out/'PRIMARY_RUN_FAILURE.json', {'time_utc': utc(),
                'exception': repr(exc), 'traceback': traceback.format_exc(),
                'inputs_frozen': (out/'INPUTS_FROZEN.json').exists(),
                'repository_after': status(repo)})
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = oldout, olderr


if __name__ == '__main__':
    main()
