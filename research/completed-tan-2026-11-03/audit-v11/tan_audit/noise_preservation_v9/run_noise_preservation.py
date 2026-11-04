#!/usr/bin/env python3
"""Execute the frozen Task B preservation protocol; no source-repository writes.

Use --output-dir for a new reproduction directory. TAN_REPO_ROOT can point to
the unchanged source repository. The default writes beside this script.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback

import numpy as np

BASE = Path(__file__).resolve().parent
PROTOCOL_SHA256 = '70a9fcebb77a223e321e34fa11eb18f1228431c3d3ef8ac40cfc1a4706eb87e1'
COMMIT = '85213f0d417d54e4f48b301ed1c0e4200ef3c421'
P_KEEP = 529 / 1013
NATIVE = ('LIF', 'TAN')
ARMS = ('TAN', 'LIF', 'LIF_THINNED', 'ZERO_OUTPUT')
CONDITIONS = ('step', 'catch')
CONTRASTS = (
    'TAN_BA_minus_ZERO_BA',
    'TAN_BA_minus_LIF_BA',
    'TAN_BA_minus_LIF_THINNED_BA',
    'TAN_step_R_minus_LIF_step_R',
    'TAN_step_R_minus_LIF_THINNED_step_R',
    'TAN_noise_induced_R_minus_LIF_noise_induced_R',
)
LEDGER_FIELDS = (
    'history_id', 'condition', 'arm', 'step_present', 'predicted_step',
    'observed_total_spikes', 'observed_onset_spikes', 'observed_late_spikes',
    'R', 'generated_total_spikes', 'generated_onset_spikes',
    'generated_late_spikes', 'underlying_model', 'sum_abs_current',
    'sum_current_squared', 'sum_candidate_squared', 'peak_abs_current',
    'peak_candidate', 'peak_abs_candidate', 'peak_membrane',
    'peak_abs_membrane',
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, fields, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def repo_state(repo):
    if not (repo / '.git').exists():
        return None
    return subprocess.check_output(
        ['git', '-C', str(repo), 'status', '--porcelain=v1'], text=True
    )


def array_manifest(arrays):
    return {
        key: {'shape': list(value.shape), 'dtype': str(value.dtype),
              'raw_c_order_sha256': hashlib.sha256(value.tobytes(order='C')).hexdigest()}
        for key, value in arrays.items()
    }


def describe(values):
    values = np.asarray(values)
    return {'mean': float(values.mean()), 'min': float(values.min()),
            'max': float(values.max()),
            'sample_sd': float(values.std(ddof=1)) if values.size > 1 else None}


def check_sources(repo, out):
    protocol = BASE / 'PREREGISTRATION.md'
    assert digest(protocol) == PROTOCOL_SHA256, 'Frozen protocol hash mismatch'
    commit = None
    if (repo / '.git').exists():
        commit = subprocess.check_output(
            ['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        assert commit == COMMIT, (commit, COMMIT)
    source_manifest = json.loads((BASE / 'NOISE_SOURCE_MANIFEST.json').read_text())
    hashes = {}
    for rel, expected in source_manifest['source_files'].items():
        actual = digest(repo / rel)
        assert actual == expected['sha256'], (rel, actual)
        hashes[rel] = actual
    source = (repo / 'snapshot/code/experiments/statistical_validation.py').read_text()
    tree = ast.parse(source)
    selected = {node.name: node for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    extracted = (BASE / 'taskb_original_extracted.py').read_text()
    assert digest(BASE / 'taskb_original_extracted.py') == source_manifest['extracted_sha256']
    extracted_tree = ast.parse(extracted)
    extracted_nodes = {node.name: node for node in extracted_tree.body
                       if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    for info in source_manifest['selected_source_nodes']:
        name = info['name']
        raw = ast.get_source_segment(source, selected[name])
        copied = ast.get_source_segment(extracted, extracted_nodes[name])
        assert raw == copied, name
        assert hashlib.sha256(raw.encode()).hexdigest() == info['exact_segment_sha256']
        dump = ast.dump(selected[name], include_attributes=False)
        assert hashlib.sha256(dump.encode()).hexdigest() == info['ast_sha256']
    namespace = {'np': np}
    # Only the three verified node bodies execute, never original module top level.
    safe_tree = ast.Module(body=[extracted_nodes[name] for name in
                                ('LIFNeuron', 'TANNeuron', 'run_experiment_B')],
                           type_ignores=[])
    exec(compile(safe_tree, '<verified-original-taskb-nodes>', 'exec'), namespace)
    evidence = {'time_utc': utc(), 'commit': commit,
                'expected_source_commit': COMMIT,
                'git_provenance_available': commit is not None,
                'protocol_sha256': PROTOCOL_SHA256,
                'source_manifest_sha256': digest(BASE / 'NOISE_SOURCE_MANIFEST.json'),
                'selected_extracted_sha256': digest(BASE / 'taskb_original_extracted.py'),
                'source_files_sha256': hashes, 'all_guards_passed': True,
                'execution': 'Only exact verified AST class/function bodies; NumPy supplied.'}
    write_json(out / 'SOURCE_GUARD.json', evidence)
    return namespace, evidence


def exact_instance(namespace, model):
    if model == 'LIF':
        return namespace['LIFNeuron'](lambda_leak=0.8, threshold=2.0)
    return namespace['TANNeuron'](
        window_size=10, lambda_leak=0.8, threshold=1.0,
        noise_tolerance=0.0, use_gating=True, use_asym_rect=False, beta=1.0)


def simulate_native(waveforms, conditions, ids, namespace):
    """Separate scalar instrumentation checked against exact source every update."""
    count, nc, steps = waveforms.shape
    assert steps == 60
    result = {'conditions': np.asarray(conditions), 'history_ids': np.asarray(ids)}
    checks = {}
    scalar_fields = ('current', 'candidate', 'membrane', 'exact_class_membrane')
    for model in NATIVE:
        prefix = model + '__'
        for field in scalar_fields:
            result[prefix + field] = np.zeros((count, nc, 60), dtype=np.float64)
        for field in ('spikes', 'exact_class_spikes'):
            result[prefix + field] = np.zeros((count, nc, 60), dtype=np.int8)
        if model == 'TAN':
            for field in ('history', 'exact_class_history', 'attention'):
                result[prefix + field] = np.zeros((count, nc, 60, 10))
            for field in ('surprise', 'gate', 'context', 'query'):
                result[prefix + field] = np.zeros((count, nc, 60))
        max_membrane_error = 0.0
        max_history_error = 0.0
        for i in range(count):
            for c in range(nc):
                exact = exact_instance(namespace, model)
                membrane = 0.0
                history = np.zeros(10)
                for t, stimulus in enumerate(waveforms[i, c]):
                    if model == 'TAN':
                        history = np.concatenate((history[1:], [stimulus]))
                        surprise = stimulus - np.mean(history)
                        query = 2.0 * surprise
                        logits = query * history
                        weights = np.exp(logits - np.max(logits))
                        attention = weights / (weights.sum(axis=0) + 1e-9)
                        context = np.sum(attention * history)
                        gate = np.tanh(abs(surprise))
                        current = gate * context
                        threshold = 1.0
                        for field, value in (
                            ('history', history), ('attention', attention),
                            ('surprise', surprise), ('gate', gate),
                            ('context', context), ('query', query)):
                            result[prefix + field][i, c, t] = value
                    else:
                        current = stimulus
                        threshold = 2.0
                    candidate = 0.8 * membrane + current
                    spike = int(candidate > threshold)
                    membrane = 0.0 if spike else candidate
                    exact_spike = exact.forward(stimulus)
                    difference = abs(membrane - exact.h)
                    max_membrane_error = max(max_membrane_error, float(difference))
                    assert spike == exact_spike, (model, i, c, t, 'spike')
                    assert difference <= 1e-12, (model, i, c, t, difference)
                    for field, value in (
                        ('current', current), ('candidate', candidate),
                        ('spikes', spike), ('membrane', membrane),
                        ('exact_class_spikes', exact_spike),
                        ('exact_class_membrane', exact.h)):
                        result[prefix + field][i, c, t] = value
                    if model == 'TAN':
                        hdiff = float(np.max(abs(history - exact.history)))
                        max_history_error = max(max_history_error, hdiff)
                        assert hdiff <= 1e-12
                        result[prefix + 'exact_class_history'][i, c, t] = exact.history
        checks[model] = {'all_spikes_exact': True,
                         'max_membrane_abs_error': max_membrane_error,
                         'max_history_abs_error': max_history_error,
                         'n_trajectories': count * nc, 'n_updates': count * nc * 60}
    return result, checks


def observed_spikes(traces, uniforms=None):
    observed = {arm: traces[arm + '__spikes'].copy() for arm in NATIVE}
    if uniforms is not None:
        observed['LIF_THINNED'] = (
            observed['LIF'] * (uniforms[:, None, :] < P_KEEP)).astype(np.int8)
        observed['ZERO_OUTPUT'] = np.zeros_like(observed['LIF'])
    return observed


def ledger_rows(traces, observed, conditions, ids):
    rows = []
    for i, history_id in enumerate(ids):
        for c, condition in enumerate(conditions):
            for arm, spikes in observed.items():
                train = spikes[i, c]
                underlying = 'LIF' if arm == 'LIF_THINNED' else arm
                row = dict.fromkeys(LEDGER_FIELDS)
                row.update(history_id=int(history_id), condition=condition, arm=arm,
                           step_present=int(condition == 'step'),
                           predicted_step=int(train[15:20].sum() >= 1),
                           observed_total_spikes=int(train.sum()),
                           observed_onset_spikes=int(train[15:20].sum()),
                           observed_late_spikes=int(train[20:60].sum()),
                           R=float(train[20:60].mean()),
                           underlying_model=None if arm == 'ZERO_OUTPUT' else underlying)
                if arm != 'ZERO_OUTPUT':
                    native_spikes = traces[underlying + '__spikes'][i, c]
                    current = traces[underlying + '__current'][i, c]
                    candidate = traces[underlying + '__candidate'][i, c]
                    membrane = traces[underlying + '__membrane'][i, c]
                    row.update(
                        generated_total_spikes=int(native_spikes.sum()),
                        generated_onset_spikes=int(native_spikes[15:20].sum()),
                        generated_late_spikes=int(native_spikes[20:60].sum()),
                        sum_abs_current=float(np.abs(current).sum()),
                        sum_current_squared=float(np.square(current).sum()),
                        sum_candidate_squared=float(np.square(candidate).sum()),
                        peak_abs_current=float(np.abs(current).max()),
                        peak_candidate=float(candidate.max()),
                        peak_abs_candidate=float(np.abs(candidate).max()),
                        peak_membrane=float(membrane.max()),
                        peak_abs_membrane=float(np.abs(membrane).max()))
                rows.append(row)
    return rows


def run_original(namespace, repo, out):
    frozen_rows = list(csv.DictReader(
        (repo / 'snapshot/data/experiment_results/stats_validation_metrics.csv').open()))
    original = {(row['model'], int(row['seed'])): row
                for row in frozen_rows if row['exp'] == 'B'}
    assert len(original) == 100
    records = []
    for seed in range(50):
        for model in NATIVE:
            actual = namespace['run_experiment_B'](model, seed)
            expected = original[(model, seed)]
            match = (actual['false_spike_rate'] == float(expected['false_spike_rate'])
                     and actual['total_spikes'] == int(expected['total_spikes']))
            records.append({'seed': seed, 'model': model, **actual,
                            'expected_false_spike_rate': float(expected['false_spike_rate']),
                            'expected_total_spikes': int(expected['total_spikes']),
                            'match': match})
            assert match, records[-1]
    write_csv(out / 'original_call_results.csv', tuple(records[0]), records)
    noise = np.stack([np.random.RandomState(seed).normal(0, 0.4, 60)
                      for seed in range(50)])
    base = np.concatenate((np.zeros(15), np.full(45, 1.5)))[None, :]
    inputs = {'noise': noise, 'base': base,
              'waveforms': np.clip(noise[:, None, :] + base[None, :, :], 0, None),
              'history_ids': np.arange(50), 'conditions': np.asarray(['step'])}
    np.savez_compressed(out / 'original_inputs.npz', **inputs)
    traces, checks = simulate_native(inputs['waveforms'], ['step'], range(50), namespace)
    np.savez_compressed(out / 'original_trajectories.npz', **traces)
    means = {}
    summary = next(row for row in csv.DictReader(
        (repo / 'snapshot/data/experiment_results/stats_validation_summary.csv').open())
                   if row['exp'] == 'B')
    for model in NATIVE:
        rates = np.array([r['false_spike_rate'] for r in records if r['model'] == model])
        means[model] = {'false_spike_rate_mean': float(rates.mean()),
                        'false_spike_rate_sample_sd': float(rates.std(ddof=1))}
        assert abs(rates.mean() - float(summary[model + '_mean'])) <= 1e-15
        assert abs(rates.std(ddof=1) - float(summary[model + '_std'])) <= 1e-15
        assert np.array_equal(traces[model + '__spikes'][:, 0, 20:60].mean(axis=1), rates)
        totals = [r['total_spikes'] for r in records if r['model'] == model]
        assert np.array_equal(traces[model + '__spikes'][:, 0].sum(axis=1), totals)
    completed = {'time_utc': utc(), 'phase': 'A original reproduction completed',
                 'records': 100, 'exact_raw_record_matches': 100,
                 'means_and_sample_sd': means, 'native_trace_checks': checks,
                 'original_calls_sha256': digest(out / 'original_call_results.csv'),
                 'original_inputs_sha256': digest(out / 'original_inputs.npz'),
                 'original_trajectories_sha256': digest(out / 'original_trajectories.npz'),
                 'supplementary_metrics_computed_yet': False,
                 'fresh_inputs_generated_yet': False}
    write_json(out / 'PHASE_A_COMPLETE.json', completed)
    print('PHASE A COMPLETE: exact 100/100 records and all native trajectory checks.', flush=True)
    print(json.dumps(means), flush=True)
    # All supplementary quantities begin only after the completion artifact exists.
    draws = np.random.RandomState(2026110232).randint(0, 50, size=(2000, 50))
    np.save(out / 'original_bootstrap_draws.npy', draws)
    difference = (traces['TAN__spikes'][:, 0, 20:60].mean(axis=1)
                  - traces['LIF__spikes'][:, 0, 20:60].mean(axis=1))
    replicate = difference[draws].mean(axis=1)
    write_csv(out / 'original_bootstrap_contrasts.csv',
              ('replicate', 'TAN_step_R_minus_LIF_step_R'),
              [{'replicate': i, 'TAN_step_R_minus_LIF_step_R': float(value)}
               for i, value in enumerate(replicate)])
    write_csv(out / 'original_ledger.csv', LEDGER_FIELDS,
              ledger_rows(traces, observed_spikes(traces), ['step'], range(50)))
    results = {'historical_labels_and_tests_unchanged': True,
               'phase_A_completion': completed,
               'new_labeled_paired_bootstrap': {
                   'contrast': 'TAN_step_R_minus_LIF_step_R',
                   'mean': float(difference.mean()),
                   'percentile_95_ci': np.quantile(replicate, [0.025, 0.975]).tolist(),
                   'n_histories': 50, 'replicates': 2000, 'seed': 2026110232,
                   'draws_sha256': digest(out / 'original_bootstrap_draws.npy')}}
    write_json(out / 'original_results.json', results)
    return results, checks


def clean_results(traces):
    lif_step = traces['LIF__spikes'][0, 0]
    assert np.array_equal(np.flatnonzero(lif_step), np.arange(16, 60, 2))
    assert lif_step.sum() == 22 and lif_step[20:60].sum() == 20
    results = {'native_trajectories': 4, 'deterministic_not_replicated': True,
               'LIF_certificate': {'spike_indices': np.flatnonzero(lif_step).tolist(),
                                   'total': 22, 'late': 20, 'R': 0.5}, 'arms': {}}
    for arm in ARMS:
        arm_result = {'conditions': {}}
        for c, condition in enumerate(CONDITIONS):
            if arm == 'ZERO_OUTPUT':
                total = onset = late = probability = 0.0
            else:
                native = 'LIF' if arm == 'LIF_THINNED' else arm
                spikes = traces[native + '__spikes'][0, c]
                total, onset, late = (float(spikes.sum()), float(spikes[15:20].sum()),
                                      float(spikes[20:60].sum()))
                if arm == 'LIF_THINNED':
                    probability = 1 - (1 - P_KEEP) ** onset
                    total, onset, late = P_KEEP * total, P_KEEP * onset, P_KEEP * late
                else:
                    probability = float(onset >= 1)
            arm_result['conditions'][condition] = {
                'total_spikes': total, 'onset_spikes': onset, 'late_spikes': late,
                'R': late / 40, 'detector_positive_probability': probability,
                'count_type': 'exact_expectation' if arm == 'LIF_THINNED' else 'exact'}
        tpr = arm_result['conditions']['step']['detector_positive_probability']
        fpr = arm_result['conditions']['catch']['detector_positive_probability']
        arm_result.update(TPR=tpr, FPR=fpr, specificity=1-fpr,
                          BA=0.5*(tpr+1-fpr), J=tpr-fpr)
        results['arms'][arm] = arm_result
    return results


def summarize_fresh(observed, rows, clean):
    arms = {}
    samples = {}
    resource_fields = LEDGER_FIELDS[13:]
    for arm in ARMS:
        spikes = observed[arm]
        positives = (spikes[:, :, 15:20].sum(axis=2) >= 1).astype(float)
        rates = spikes[:, :, 20:60].mean(axis=2)
        ba = 0.5 * (positives[:, 0] + 1 - positives[:, 1])
        samples.update({arm + '__step_R': rates[:, 0], arm + '__catch_R': rates[:, 1],
                        arm + '__TPR_sample': positives[:, 0],
                        arm + '__FPR_sample': positives[:, 1], arm + '__BA_sample': ba,
                        arm + '__J_sample': positives[:, 0] - positives[:, 1]})
        tpr, fpr = positives.mean(axis=0)
        data = {'TPR': float(tpr), 'FPR': float(fpr), 'specificity': float(1-fpr),
                'BA': float(ba.mean()), 'J': float(tpr-fpr), 'conditions': {},
                'noisy_minus_clean_step_R': float(rates[:, 0].mean()
                    - clean['arms'][arm]['conditions']['step']['R'])}
        for c, condition in enumerate(CONDITIONS):
            selected = [row for row in rows if row['arm'] == arm and row['condition'] == condition]
            data['conditions'][condition] = {
                'R': describe(rates[:, c]),
                'observed_total_spikes': describe(spikes[:, c].sum(axis=1)),
                'observed_onset_spikes': describe(spikes[:, c, 15:20].sum(axis=1)),
                'observed_late_spikes': describe(spikes[:, c, 20:60].sum(axis=1)),
                'generated_total_spikes': (describe([r['generated_total_spikes'] for r in selected])
                                           if arm != 'ZERO_OUTPUT' else None),
                'resource_proxies': ({field: describe([r[field] for r in selected])
                                     for field in resource_fields} if arm != 'ZERO_OUTPUT' else None)}
        arms[arm] = data
    differences = {
        CONTRASTS[0]: samples['TAN__BA_sample'] - samples['ZERO_OUTPUT__BA_sample'],
        CONTRASTS[1]: samples['TAN__BA_sample'] - samples['LIF__BA_sample'],
        CONTRASTS[2]: samples['TAN__BA_sample'] - samples['LIF_THINNED__BA_sample'],
        CONTRASTS[3]: samples['TAN__step_R'] - samples['LIF__step_R'],
        CONTRASTS[4]: samples['TAN__step_R'] - samples['LIF_THINNED__step_R'],
        CONTRASTS[5]: ((samples['TAN__step_R'] - clean['arms']['TAN']['conditions']['step']['R'])
                       - (samples['LIF__step_R'] - clean['arms']['LIF']['conditions']['step']['R'])),
    }
    for name, difference in differences.items():
        samples[name] = difference
    return arms, samples, differences


def run_fresh(namespace, out):
    assert (out / 'PHASE_A_COMPLETE.json').is_file()
    noise = np.random.RandomState(2026110226).normal(0, 0.4, size=(512, 60))
    uniforms = np.random.RandomState(2026110230).uniform(size=(512, 60))
    base = np.stack((np.concatenate((np.zeros(15), np.full(45, 1.5))), np.zeros(60)))
    fresh = {'noise': noise, 'base': base,
             'waveforms': np.clip(base[None, :, :] + noise[:, None, :], 0, None),
             'thinning_uniforms': uniforms, 'history_ids': np.arange(512),
             'conditions': np.asarray(CONDITIONS)}
    clean_inputs = {'noise': np.zeros((1, 60)), 'base': base,
                    'waveforms': base[None, :, :].copy(), 'history_ids': np.arange(1),
                    'conditions': np.asarray(CONDITIONS)}
    np.savez_compressed(out / 'fresh_inputs.npz', **fresh)
    np.savez_compressed(out / 'clean_inputs.npz', **clean_inputs)
    frozen = {'time_utc': utc(), 'models_evaluated_yet': False,
              'phase_A_completion_sha256': digest(out / 'PHASE_A_COMPLETE.json'),
              'fresh_inputs_sha256': digest(out / 'fresh_inputs.npz'),
              'clean_inputs_sha256': digest(out / 'clean_inputs.npz'),
              'fresh_array_manifest': array_manifest(fresh),
              'clean_array_manifest': array_manifest(clean_inputs)}
    write_json(out / 'FRESH_INPUTS_FROZEN.json', frozen)
    print('FRESH INPUTS FROZEN:', frozen['fresh_inputs_sha256'], flush=True)
    clean_traces, clean_checks = simulate_native(base[None, :, :], CONDITIONS, [0], namespace)
    np.savez_compressed(out / 'clean_trajectories.npz', **clean_traces)
    clean = clean_results(clean_traces)
    write_json(out / 'clean_results.json', clean)
    write_csv(out / 'clean_native_ledger.csv', LEDGER_FIELDS,
              ledger_rows(clean_traces, observed_spikes(clean_traces), CONDITIONS, [0]))
    traces, checks = simulate_native(fresh['waveforms'], CONDITIONS, range(512), namespace)
    observed = observed_spikes(traces, uniforms)
    for arm, value in observed.items():
        traces[arm + '__observed_spikes'] = value
    np.savez_compressed(out / 'fresh_trajectories.npz', **traces)
    rows = ledger_rows(traces, observed, CONDITIONS, range(512))
    assert len(rows) == 4096
    write_csv(out / 'fresh_ledger.csv', LEDGER_FIELDS, rows)
    arms, samples, differences = summarize_fresh(observed, rows, clean)
    np.savez_compressed(out / 'fresh_samples.npz', **samples)
    draws = np.random.RandomState(2026110231).randint(0, 512, size=(2000, 512))
    np.save(out / 'fresh_bootstrap_draws.npy', draws)
    replicates = {name: differences[name][draws].mean(axis=1) for name in CONTRASTS}
    write_csv(out / 'fresh_bootstrap_contrasts.csv', ('replicate',) + CONTRASTS,
              [{'replicate': i, **{name: float(replicates[name][i]) for name in CONTRASTS}}
               for i in range(2000)])
    contrasts = {name: {'mean': float(differences[name].mean()),
                        'percentile_95_ci': np.quantile(replicates[name], [0.025, 0.975]).tolist()}
                 for name in CONTRASTS}
    result = {'time_utc': utc(), 'protocol_sha256': PROTOCOL_SHA256,
              'n_histories': 512, 'n_conditions': 2, 'n_arms': 4,
              'n_ledger_rows': len(rows), 'no_exclusions': True,
              'noise_seed': 2026110226, 'thinning_seed': 2026110230,
              'thinning_keep_probability': P_KEEP,
              'bootstrap_seed': 2026110231, 'bootstrap_replicates': 2000,
              'bootstrap_unit': 'Whole paired history, all arms and both conditions',
              'fresh_inputs_sha256': digest(out / 'fresh_inputs.npz'),
              'bootstrap_draws_sha256': digest(out / 'fresh_bootstrap_draws.npy'),
              'arms': arms, 'paired_contrasts': contrasts,
              'clean_control_results_file': 'clean_results.json',
              'native_trace_checks': checks, 'clean_trace_checks': clean_checks,
              'fresh_observed_late_budget_mismatch': {
                  'TAN_step_R_minus_LIF_THINNED_step_R': contrasts[CONTRASTS[4]]['mean'],
                  'scope': 'Observed events only; not generated spikes, input gain or physical energy.'},
              'limitations': [
                  'Fixed known-onset external readout; no amplitude reconstruction claim.',
                  'Percentile intervals are descriptive; no multiplicity-adjusted claim.',
                  'Degenerate empirical bootstrap intervals do not establish zero population error.',
                  'Resource proxies use discrete dimensionless units, not physical time or ATP.',
                  'LIF_THINNED leaves native LIF computation, resets and generated spikes unchanged.',
                  'ZERO_OUTPUT has no assigned physical implementation or energy.',
                  'Historical labels/tests and independently supported P1A results are unchanged.']}
    write_json(out / 'primary_results.json', result)
    for arm in ARMS:
        entry = arms[arm]
        print(arm, 'BA', entry['BA'], 'TPR', entry['TPR'], 'FPR', entry['FPR'],
              'step_R', entry['conditions']['step']['R']['mean'],
              'noisy_minus_clean_R', entry['noisy_minus_clean_step_R'], flush=True)
    for name, value in contrasts.items():
        print(name, json.dumps(value), flush=True)
    return result


class Tee:
    def __init__(self, stream, logfile):
        self.stream, self.logfile = stream, logfile

    def write(self, value):
        self.stream.write(value)
        self.logfile.write(value)
        self.logfile.flush()

    def flush(self):
        self.stream.flush()
        self.logfile.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=BASE)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'PHASE_A_COMPLETE.json').exists() or (out / 'PRIMARY_MANIFEST.json').exists():
        raise RuntimeError('Preserving previous outputs; supply a fresh --output-dir for a rerun.')
    repo = Path(os.environ.get('TAN_REPO_ROOT', BASE.parents[1] / 'Temporal-Attention-Neuron')).resolve()
    started = utc()
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with (out / 'run.log').open('w') as log:
        sys.stdout, sys.stderr = Tee(original_stdout, log), Tee(original_stderr, log)
        before = repo_state(repo)
        unavailable = 'UNAVAILABLE: source copy has no .git; source hashes remain mandatory.\n'
        (out / 'repo_status_before.txt').write_text(before if before is not None else unavailable)
        print('START', started, 'command:', ' '.join(sys.argv), flush=True)
        print('REPOSITORY', repo, 'status_before', repr(before), flush=True)
        try:
            assert before in (None, ''), 'Full source checkout must be clean before execution'
            namespace, guards = check_sources(repo, out)
            original, _ = run_original(namespace, repo, out)
            fresh = run_fresh(namespace, out)
            after = repo_state(repo)
            (out / 'repo_status_after.txt').write_text(after if after is not None else unavailable)
            assert before == after, 'Repository status changed'
            finished = utc()
            print('COMPLETED', finished, 'Repository status:', repr(after),
                  'status_unchanged:', True if before is not None else None, flush=True)
            manifest = {'started_utc': started, 'completed_utc': finished,
                        'status': 'COMPLETED', 'execution_command': sys.argv,
                        'python': platform.python_version(), 'numpy': np.__version__,
                        'repository': str(repo),
                        'repository_status_unchanged': True if before is not None else None,
                        'git_provenance_available': before is not None,
                        'protocol_sha256': PROTOCOL_SHA256,
                        'primary_script_sha256': digest(Path(__file__)),
                        'ledger_fields': list(LEDGER_FIELDS),
                        'contrast_order': list(CONTRASTS),
                        'phase_A_records': 100, 'fresh_ledger_rows': 4096,
                        'trajectory_axis_order': 'history, condition, timestep[, history_slot]',
                        'float_validation_absolute_tolerance': 1e-12,
                        'original_results': 'original_results.json',
                        'fresh_results': 'primary_results.json',
                        'clean_results': 'clean_results.json', 'files': {}, 'npz_schema': {}}
            names = [
                'SOURCE_GUARD.json', 'original_call_results.csv', 'original_inputs.npz',
                'original_trajectories.npz', 'PHASE_A_COMPLETE.json',
                'original_bootstrap_draws.npy', 'original_bootstrap_contrasts.csv',
                'original_ledger.csv', 'original_results.json', 'fresh_inputs.npz',
                'clean_inputs.npz', 'FRESH_INPUTS_FROZEN.json', 'clean_trajectories.npz',
                'clean_results.json', 'clean_native_ledger.csv', 'fresh_trajectories.npz',
                'fresh_ledger.csv', 'fresh_samples.npz', 'fresh_bootstrap_draws.npy',
                'fresh_bootstrap_contrasts.csv', 'primary_results.json', 'run.log',
                'repo_status_before.txt', 'repo_status_after.txt']
            for name in names:
                path = out / name
                manifest['files'][name] = {'sha256': digest(path), 'bytes': path.stat().st_size}
                if name.endswith('.npz'):
                    with np.load(path) as values:
                        manifest['npz_schema'][name] = {
                            key: {'shape': list(values[key].shape), 'dtype': str(values[key].dtype)}
                            for key in values.files}
            write_json(out / 'PRIMARY_MANIFEST.json', manifest)
        except Exception as exc:
            after = repo_state(repo)
            (out / 'repo_status_after.txt').write_text(after if after is not None else unavailable)
            write_json(out / 'PRIMARY_RUN_FAILURE.json', {
                'time_utc': utc(), 'exception': repr(exc),
                'traceback': traceback.format_exc(), 'repo_status_before': before,
                'repo_status_after': after, 'phase_A_complete': (out / 'PHASE_A_COMPLETE.json').exists()})
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr


if __name__ == '__main__':
    main()
