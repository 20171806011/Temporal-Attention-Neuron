#!/usr/bin/env python3
"""Independent reconstruction of the frozen v9 Task B experiment.

This file imports neither the primary simulator nor a primary metric helper.
Only its completed data products are read.  Every model update, input stream,
surrogate, diagnostic, and bootstrap is rebuilt here.
"""
from __future__ import annotations

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

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('TAN_REPO_ROOT', str(HERE.parents[1] / 'Temporal-Attention-Neuron')))
PROTOCOL_HASH = '70a9fcebb77a223e321e34fa11eb18f1228431c3d3ef8ac40cfc1a4706eb87e1'
COMMIT = '85213f0d417d54e4f48b301ed1c0e4200ef3c421'
KEEP = 529 / 1013
TOL = 1e-12
CHECKS = []
COMPARISONS = []
CONTRAST_NAMES = (
    'TAN_BA_minus_ZERO_BA', 'TAN_BA_minus_LIF_BA',
    'TAN_BA_minus_LIF_THINNED_BA', 'TAN_step_R_minus_LIF_step_R',
    'TAN_step_R_minus_LIF_THINNED_step_R',
    'TAN_noise_induced_R_minus_LIF_noise_induced_R',
)
LEDGER_FIELDS = (
    'history_id', 'condition', 'arm', 'step_present', 'predicted_step',
    'observed_total_spikes', 'observed_onset_spikes', 'observed_late_spikes',
    'R', 'generated_total_spikes', 'generated_onset_spikes',
    'generated_late_spikes', 'underlying_model', 'sum_abs_current',
    'sum_current_squared', 'sum_candidate_squared', 'peak_abs_current',
    'peak_candidate', 'peak_abs_candidate', 'peak_membrane', 'peak_abs_membrane',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(name, condition, detail=None):
    record = {'check': name, 'passed': bool(condition)}
    if detail is not None:
        record['detail'] = detail
    CHECKS.append(record)
    if not condition:
        raise AssertionError(f'{name}: {detail}')


def compare(name, actual, expected, exact=False):
    actual, expected = np.asarray(actual), np.asarray(expected)
    check(name + ': shape', actual.shape == expected.shape,
          {'actual': list(actual.shape), 'expected': list(expected.shape)})
    if actual.dtype.kind in 'biufc' and expected.dtype.kind in 'biufc':
        check(name + ': finite', bool(np.isfinite(actual).all() and np.isfinite(expected).all()))
        max_error = float(np.max(np.abs(actual.astype(float) - expected.astype(float)))) if actual.size else 0.
        equal = np.array_equal(actual, expected) if exact else max_error <= TOL
    else:
        max_error = None
        equal = np.array_equal(actual, expected)
    COMPARISONS.append({'name': name, 'elements': int(actual.size),
                        'exact': bool(exact), 'max_abs_error': max_error})
    check(name + ': values', equal, {'max_abs_error': max_error, 'exact': exact})


def source_validation():
    check('protocol frozen hash', sha(HERE / 'PREREGISTRATION.md') == PROTOCOL_HASH)
    manifest = json.loads((HERE / 'NOISE_SOURCE_MANIFEST.json').read_text())
    for relative, spec in manifest['source_files'].items():
        p = REPO / relative
        check('source bytes ' + relative, p.stat().st_size == spec['bytes'])
        check('source sha256 ' + relative, sha(p) == spec['sha256'])
    original_text = (REPO / 'snapshot/code/experiments/statistical_validation.py').read_text()
    original_nodes = {node.name: node for node in ast.parse(original_text).body
                      if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    extracted_path = HERE / 'taskb_original_extracted.py'
    check('selected source file hash', sha(extracted_path) == manifest['extracted_sha256'])
    extracted_nodes = {node.name: node for node in ast.parse(extracted_path.read_text()).body
                       if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    for selected in manifest['selected_source_nodes']:
        name = selected['name']
        segment = ast.get_source_segment(original_text, original_nodes[name])
        check('exact original node segment ' + name,
              hashlib.sha256(segment.encode()).hexdigest() == selected['exact_segment_sha256'])
        check('extracted AST equals original ' + name,
              ast.dump(original_nodes[name], include_attributes=False) ==
              ast.dump(extracted_nodes[name], include_attributes=False))
    return manifest


def git_status():
    if not (REPO / '.git').exists():
        return {'available': False, 'commit': None, 'status': None,
                'note': 'Source subset has no .git; source hashes remain mandatory.'}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True)
    check('live repository exact commit', commit == COMMIT, commit)
    check('live repository status clean', status == '', status)
    return {'available': True, 'commit': commit, 'status': status}


def independent_native(delivered, model):
    """One full waveform; scalar membrane and explicit finite history.

    Appending a sample, computing a new mean, and then the softmax are explicit;
    no AST source model or primary implementation is invoked.
    """
    delivered = np.asarray(delivered, dtype=float)
    membrane = 0.
    past = np.zeros(10)
    result = {name: [] for name in ('spike', 'candidate', 'membrane', 'current')}
    if model == 'TAN':
        result.update({name: [] for name in ('surprise', 'gate', 'context', 'history', 'attention')})
    for value in delivered:
        if model == 'LIF':
            effective = value
            threshold = 2.
        else:
            past = np.concatenate((past[1:], [value]))
            deviation = value - np.mean(past)
            similarity = (2. * deviation) * past
            unnormalized = np.exp(similarity - np.max(similarity))
            weights = unnormalized / (np.sum(unnormalized) + 1e-9)
            context = np.sum(weights * past)
            gate = np.tanh(abs(deviation))
            effective = gate * context
            threshold = 1.
            result['surprise'].append(deviation)
            result['gate'].append(gate)
            result['context'].append(context)
            result['history'].append(past.copy())
            result['attention'].append(weights.copy())
        candidate = .8 * membrane + effective
        fired = int(candidate > threshold)
        membrane = 0. if fired else candidate
        result['spike'].append(fired)
        result['candidate'].append(candidate)
        result['membrane'].append(membrane)
        result['current'].append(effective)
    return {name: np.array(values, dtype=np.uint8 if name == 'spike' else float)
            for name, values in result.items()}


def batch_native(waveforms, model):
    traces = [independent_native(w, model) for w in waveforms]
    return {key: np.stack([t[key] for t in traces]) for key in traces[0]}


def generated_design():
    base = np.zeros(60)
    base[15:] = 1.5
    noise = np.random.RandomState(2026110226).normal(0., .4, size=(512, 60))
    thinning = np.random.RandomState(2026110230).uniform(size=(512, 60))
    step = np.maximum(noise + base, 0.)
    catch = np.maximum(noise, 0.)
    original_noise = np.stack([np.random.RandomState(seed).normal(0., .4, 60) for seed in range(50)])
    return {'base': base, 'noise': noise, 'thinning_uniforms': thinning,
            'step': step, 'catch': catch,
            'original_noise': original_noise,
            'original_step': np.maximum(original_noise + base, 0.),
            'clean_step': base, 'clean_catch': np.zeros(60)}


def independent_outputs(design):
    output = {}
    for model in ('LIF', 'TAN'):
        for condition in ('step', 'catch'):
            output[('fresh', model, condition)] = batch_native(design[condition], model)
            output[('clean', model, condition)] = batch_native(design['clean_' + condition][None, :], model)
        output[('original', model, 'step')] = batch_native(design['original_step'], model)
    for condition in ('step', 'catch'):
        lif = output[('fresh', 'LIF', condition)]
        output[('fresh', 'LIF_THINNED', condition)] = {
            'spike': lif['spike'] * (design['thinning_uniforms'] < KEEP)}
        output[('fresh', 'ZERO_OUTPUT', condition)] = {'spike': np.zeros((512, 60), dtype=np.uint8)}
    certificate = np.zeros(60, dtype=np.uint8)
    certificate[16:59:2] = 1
    compare('clean LIF algebraic exact spike certificate',
            output[('clean', 'LIF', 'step')]['spike'][0], certificate, exact=True)
    check('clean LIF algebraic total22 and late20',
          certificate.sum() == 22 and certificate[20:].sum() == 20)
    for model in ('LIF', 'TAN'):
        check('clean catch silent ' + model, not output[('clean', model, 'catch')]['spike'].any())
    tan = output[('clean', 'TAN', 'step')]
    check('TAN clean surprise zero after window fill', (tan['surprise'][0, 24:] == 0.).all())
    check('TAN clean gate zero after window fill', (tan['gate'][0, 24:] == 0.).all())
    check('TAN clean no spikes after window fill', not tan['spike'][0, 24:].any())
    return output


def spike_metrics(spikes):
    spikes = np.asarray(spikes)
    onset = spikes[:, 15:20].sum(axis=1)
    late = spikes[:, 20:60].sum(axis=1)
    return {'total_count': spikes.sum(axis=1), 'onset_count': onset,
            'late_count': late, 'R': late / 40., 'detected': (onset >= 1).astype(np.uint8)}


def expected_statistics(output):
    metrics = {key: spike_metrics(value['spike']) for key, value in output.items()}
    summaries = {}
    clean_late = {model: metrics[('clean', model, 'step')]['R'][0] for model in ('TAN', 'LIF')}
    clean_late['LIF_THINNED'] = KEEP * clean_late['LIF']
    clean_late['ZERO_OUTPUT'] = 0.
    for arm in ('TAN', 'LIF', 'LIF_THINNED', 'ZERO_OUTPUT'):
        step = metrics[('fresh', arm, 'step')]
        catch = metrics[('fresh', arm, 'catch')]
        tpr = float(step['detected'].mean())
        fpr = float(catch['detected'].mean())
        summaries[arm] = {'TPR': tpr, 'FPR': fpr, 'specificity': 1-fpr,
                          'BA': .5*(tpr+1-fpr), 'J': tpr-fpr,
                          'clean_step_R': float(clean_late[arm]),
                          'noisy_minus_clean_step_R': float(step['R'].mean() - clean_late[arm]),
                          'step': {key: float(value.mean()) for key, value in step.items()},
                          'catch': {key: float(value.mean()) for key, value in catch.items()}}
    per_history_ba = {arm: .5*(metrics[('fresh', arm, 'step')]['detected'].astype(float) + 1. -
                                 metrics[('fresh', arm, 'catch')]['detected']) for arm in summaries}
    step_R = {arm: metrics[('fresh', arm, 'step')]['R'] for arm in summaries}
    contrasts = np.column_stack([
        per_history_ba['TAN'] - per_history_ba['ZERO_OUTPUT'],
        per_history_ba['TAN'] - per_history_ba['LIF'],
        per_history_ba['TAN'] - per_history_ba['LIF_THINNED'],
        step_R['TAN'] - step_R['LIF'],
        step_R['TAN'] - step_R['LIF_THINNED'],
        (step_R['TAN']-clean_late['TAN']) - (step_R['LIF']-clean_late['LIF']),
    ])
    draws = np.random.RandomState(2026110231).randint(0, 512, size=(2000, 512))
    bootstrap = contrasts[draws].mean(axis=1)
    original_differences = metrics[('original', 'TAN', 'step')]['R'] - metrics[('original', 'LIF', 'step')]['R']
    original_draws = np.random.RandomState(2026110232).randint(0, 50, size=(2000, 50))
    original_bootstrap = original_differences[original_draws].mean(axis=1)
    return {'metrics': metrics, 'summaries': summaries, 'contrast_values': contrasts,
            'bootstrap_draws': draws, 'bootstrap': bootstrap,
            'bootstrap_point': contrasts.mean(axis=0),
            'bootstrap_ci': np.quantile(bootstrap, [.025, .975], axis=0),
            'original_draws': original_draws, 'original_bootstrap': original_bootstrap,
            'original_point': float(original_differences.mean()),
            'original_ci': np.quantile(original_bootstrap, [.025, .975])}


def validate_original_table(output, stats):
    rows = list(csv.DictReader((REPO / 'snapshot/data/experiment_results/stats_validation_metrics.csv').open()))
    selected = [row for row in rows if row['exp'] == 'B']
    check('historical table has exactly100 TaskB records', len(selected) == 100)
    keys = {(r['model'], int(r['seed'])) for r in selected}
    check('historical table complete model-seed grid', keys == {(m, s) for m in ('LIF', 'TAN') for s in range(50)})
    for row in selected:
        model, seed = row['model'], int(row['seed'])
        values = stats['metrics'][('original', model, 'step')]
        compare(f'historical record {model} {seed} R', float(row['false_spike_rate']), values['R'][seed], exact=True)
        check(f'historical record {model} {seed} total', int(row['total_spikes']) == values['total_count'][seed])
    summary = next(r for r in csv.DictReader((REPO / 'snapshot/data/experiment_results/stats_validation_summary.csv').open()) if r['exp'] == 'B')
    for model in ('LIF', 'TAN'):
        values = stats['metrics'][('original', model, 'step')]['R']
        compare('historical summary mean ' + model, float(summary[model+'_mean']), values.mean())
        compare('historical summary sampleSD ' + model, float(summary[model+'_std']), values.std(ddof=1))


def resource_vectors(trace):
    current, candidate = trace['current'], trace['candidate']
    return {'sum_abs_current': abs(current).sum(axis=1),
            'sum_current_squared': (current*current).sum(axis=1),
            'sum_candidate_squared': (candidate*candidate).sum(axis=1),
            'peak_abs_current': abs(current).max(axis=1),
            'peak_candidate': candidate.max(axis=1),
            'peak_abs_candidate': abs(candidate).max(axis=1),
            'peak_membrane': trace['membrane'].max(axis=1),
            'peak_abs_membrane': abs(trace['membrane']).max(axis=1),
            'generated_spikes': trace['spike'].sum(axis=1)}


# Schema adapters for primary artifacts are completed after the producer supplies
# its file contract.  They perform comparisons only, never model computation.


def validate_inputs(design):
    expected = {
        'original': {'noise': design['original_noise'], 'base': design['base'][None, :],
                     'waveforms': design['original_step'][:, None, :],
                     'history_ids': np.arange(50), 'conditions': np.array(['step'])},
        'fresh': {'noise': design['noise'], 'base': np.stack([design['base'], np.zeros(60)]),
                  'waveforms': np.stack([design['step'], design['catch']], axis=1),
                  'thinning_uniforms': design['thinning_uniforms'],
                  'history_ids': np.arange(512), 'conditions': np.array(['step', 'catch'])},
        'clean': {'noise': np.zeros((1, 60)), 'base': np.stack([design['base'], np.zeros(60)]),
                  'waveforms': np.stack([design['clean_step'], design['clean_catch']])[None, ...],
                  'history_ids': np.arange(1), 'conditions': np.array(['step', 'catch'])},
    }
    for stage, fields in expected.items():
        with np.load(HERE / f'{stage}_inputs.npz', allow_pickle=False) as saved:
            for key, value in fields.items():
                compare(f'{stage} input {key}', saved[key], value, exact=True)
            check(stage + ' no unexpected input fields', set(saved.files) == set(fields), saved.files)
    check('fresh pairing before clipping',
          np.array_equal(design['step'][:, :15], design['catch'][:, :15]))
    # Shared uniforms are one array, not independently redrawn by condition.
    check('thinning frozen historical ratio to float precision', abs(KEEP - .2645 / .5065) <= np.finfo(float).eps)
    return expected


def validate_trajectories(output):
    native_map = {'spikes': 'spike', 'current': 'current', 'candidate': 'candidate',
                  'membrane': 'membrane', 'exact_class_spikes': 'spike',
                  'exact_class_membrane': 'membrane'}
    tan_map = {'history': 'history', 'exact_class_history': 'history',
               'attention': 'attention', 'surprise': 'surprise', 'gate': 'gate',
               'context': 'context'}
    for stage in ('original', 'fresh', 'clean'):
        conditions = ['step'] if stage == 'original' else ['step', 'catch']
        expected_ids = np.arange({'original': 50, 'fresh': 512, 'clean': 1}[stage])
        expected_fields = {'conditions', 'history_ids'}
        with np.load(HERE / f'{stage}_trajectories.npz', allow_pickle=False) as saved:
            compare(stage + ' trajectory conditions', saved['conditions'], conditions, exact=True)
            compare(stage + ' trajectory IDs', saved['history_ids'], expected_ids, exact=True)
            for model in ('LIF', 'TAN'):
                lookup = dict(native_map)
                if model == 'TAN':
                    lookup.update(tan_map)
                for field, internal in lookup.items():
                    key = model + '__' + field
                    expected_fields.add(key)
                    expected = np.stack([output[(stage, model, condition)][internal]
                                         for condition in conditions], axis=1)
                    compare(stage + ' native ' + key, saved[key], expected,
                            exact=internal in ('spike', 'history'))
                if model == 'TAN':
                    key = 'TAN__query'
                    expected_fields.add(key)
                    expected = 2. * np.stack([output[(stage, model, c)]['surprise'] for c in conditions], axis=1)
                    compare(stage + ' native ' + key, saved[key], expected)
            if stage == 'fresh':
                for arm in ('LIF', 'TAN', 'LIF_THINNED', 'ZERO_OUTPUT'):
                    key = arm + '__observed_spikes'
                    expected_fields.add(key)
                    spikes = np.stack([output[(stage, arm, c)]['spike'] for c in conditions], axis=1)
                    compare(stage + ' observed ' + arm, saved[key], spikes, exact=True)
            check(stage + ' no unexpected trajectory fields', set(saved.files) == expected_fields,
                  {'saved': sorted(saved.files), 'expected': sorted(expected_fields)})
    for condition in ('step', 'catch'):
        lif = output[('fresh', 'LIF', condition)]['spike']
        observed = output[('fresh', 'LIF_THINNED', condition)]['spike']
        check('thinning cannot introduce spike ' + condition, (observed <= lif).all())
        check('thinning shared label-blind mask ' + condition,
              np.array_equal(observed, lif * (generated_design()['thinning_uniforms'] < KEEP)))


def validate_bootstrap_draws(stats):
    compare('fresh all2000 group bootstrap index rows',
            np.load(HERE / 'fresh_bootstrap_draws.npy', allow_pickle=False),
            stats['bootstrap_draws'], exact=True)
    compare('original all2000 group bootstrap index rows',
            np.load(HERE / 'original_bootstrap_draws.npy', allow_pickle=False),
            stats['original_draws'], exact=True)


def independent_invariants(design, output, stats):
    for model in ('LIF', 'TAN'):
        step, catch = output[('fresh', model, 'step')], output[('fresh', model, 'catch')]
        for field in step:
            compare('identical pre-onset paired states ' + model + ' ' + field,
                    step[field][:, :15], catch[field][:, :15], exact=True)
    for (stage, model, condition), trace in output.items():
        if model not in ('LIF', 'TAN'):
            continue
        label = f'{stage}/{model}/{condition}'
        threshold = 2. if model == 'LIF' else 1.
        spikes = trace['spike']
        check(label + ' strict threshold decision', np.array_equal(spikes, trace['candidate'] > threshold))
        check(label + ' reset membrane exactly zero', (trace['membrane'][spikes.astype(bool)] == 0.).all())
        compare(label + ' all nonfiring states equal candidates',
                trace['membrane'][~spikes.astype(bool)], trace['candidate'][~spikes.astype(bool)], exact=True)
        check(label + ' postreset membrane is within native threshold',
              ((trace['membrane'] >= 0.) & (trace['membrane'] <= threshold)).all())
        previous = np.pad(trace['membrane'][:, :-1], ((0, 0), (1, 0)))
        compare(label + ' old-state recurrence identity', trace['candidate'], .8*previous + trace['current'], exact=True)
        if model == 'TAN':
            compare(label + ' gate is absolute signed-surprise function', trace['gate'], np.tanh(abs(trace['surprise'])), exact=True)
            compare(label + ' current is gate times context', trace['current'], trace['gate']*trace['context'], exact=True)
            check(label + ' attention positive', (trace['attention'] > 0).all())
            compare(label + ' history latest equals delivered sample', trace['history'][:, :, -1],
                    design['original_step'] if stage == 'original' else
                    design[condition] if stage == 'fresh' else design['clean_'+condition][None, :], exact=True)
    compare('zero-output balanced accuracy exact chance', stats['summaries']['ZERO_OUTPUT']['BA'], .5, exact=True)
    for condition in ('step', 'catch'):
        zeros = output[('fresh', 'ZERO_OUTPUT', condition)]['spike']
        check('zero-output exact silence '+condition, (zeros == 0).all())


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def independent_ledger(stage, output, stats):
    count = {'original': 50, 'fresh': 512, 'clean': 1}[stage]
    conditions = ('step',) if stage == 'original' else ('step', 'catch')
    arms = ('TAN', 'LIF', 'LIF_THINNED', 'ZERO_OUTPUT') if stage == 'fresh' else ('TAN', 'LIF')
    resources = {(model, c): resource_vectors(output[(stage, model, c)])
                 for model in ('LIF', 'TAN') for c in conditions}
    expected = {}
    for arm in arms:
        underlying = 'LIF' if arm == 'LIF_THINNED' else arm
        for condition in conditions:
            met = stats['metrics'][(stage, arm, condition)]
            for i in range(count):
                row = {field: None for field in LEDGER_FIELDS}
                row.update(history_id=i, condition=condition, arm=arm,
                           step_present=int(condition == 'step'), predicted_step=int(met['detected'][i]),
                           observed_total_spikes=int(met['total_count'][i]),
                           observed_onset_spikes=int(met['onset_count'][i]),
                           observed_late_spikes=int(met['late_count'][i]), R=float(met['R'][i]),
                           underlying_model=None if arm == 'ZERO_OUTPUT' else underlying)
                if arm != 'ZERO_OUTPUT':
                    generated = stats['metrics'][(stage, underlying, condition)]
                    row.update(generated_total_spikes=int(generated['total_count'][i]),
                               generated_onset_spikes=int(generated['onset_count'][i]),
                               generated_late_spikes=int(generated['late_count'][i]))
                    row.update({field: float(resources[(underlying, condition)][field][i])
                                for field in LEDGER_FIELDS[13:]})
                expected[(i, condition, arm)] = row
    return expected


def validate_ledgers(output, stats):
    expected_all = {}
    for stage, filename in [('original', 'original_ledger.csv'), ('fresh', 'fresh_ledger.csv'),
                             ('clean', 'clean_native_ledger.csv')]:
        expected = independent_ledger(stage, output, stats)
        expected_all[stage] = expected
        with (HERE / filename).open(newline='') as handle:
            reader = csv.DictReader(handle)
            check(stage + ' ledger exact columns', reader.fieldnames == list(LEDGER_FIELDS), reader.fieldnames)
            rows = list(reader)
        check(stage + ' ledger exact row count', len(rows) == len(expected), len(rows))
        actual = {(int(row['history_id']), row['condition'], row['arm']): row for row in rows}
        check(stage + ' ledger has no duplicate history-arm-condition', len(actual) == len(rows))
        check(stage + ' ledger complete denominator and pairing', set(actual) == set(expected))
        ordered_keys = sorted(expected)
        for field in LEDGER_FIELDS:
            target = [expected[k][field] for k in ordered_keys]
            observed = [actual[k][field] for k in ordered_keys]
            nonnull = [i for i, value in enumerate(target) if value is not None]
            check(stage + ' ledger null physical resource fields ' + field,
                  all(observed[i] == '' for i, value in enumerate(target) if value is None))
            if field in ('arm', 'condition', 'underlying_model'):
                compare(stage + ' ledger ' + field, [observed[i] for i in nonnull],
                        [target[i] for i in nonnull], exact=True)
            else:
                exact = field in ('history_id', 'step_present', 'predicted_step') or field.endswith('_spikes')
                compare(stage + ' ledger ' + field, [float(observed[i]) for i in nonnull],
                        [target[i] for i in nonnull], exact=exact)
    return expected_all


def describe(values):
    values = np.asarray(values, dtype=float)
    return {'mean': float(values.mean()), 'min': float(values.min()), 'max': float(values.max()),
            'sample_sd': float(values.std(ddof=1)) if values.size > 1 else None}


def compare_json_tree(label, actual, expected):
    if isinstance(expected, dict):
        check(label + ': expected fields present', isinstance(actual, dict) and set(expected) <= set(actual))
        for key, value in expected.items():
            compare_json_tree(label+'/'+key, actual[key], value)
    elif expected is None:
        check(label, actual is None)
    elif isinstance(expected, (str, bool)):
        check(label, actual == expected, {'actual': actual, 'expected': expected})
    else:
        compare(label, actual, expected)


def validate_bootstrap_contrasts(stats):
    with (HERE / 'fresh_bootstrap_contrasts.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        check('fresh bootstrap contrast column order', reader.fieldnames == ['replicate', *CONTRAST_NAMES])
        rows = list(reader)
    compare('fresh bootstrap replicate IDs', [int(r['replicate']) for r in rows], np.arange(2000), exact=True)
    for j, name in enumerate(CONTRAST_NAMES):
        compare('fresh bootstrap all2000 replicates '+name,
                [float(r[name]) for r in rows], stats['bootstrap'][:, j])
    with (HERE / 'original_bootstrap_contrasts.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        check('original bootstrap contrast columns', reader.fieldnames == ['replicate', CONTRAST_NAMES[3]])
        rows = list(reader)
    compare('original bootstrap replicate IDs', [int(r['replicate']) for r in rows], np.arange(2000), exact=True)
    compare('original bootstrap all2000 contrast replicates',
            [float(r[CONTRAST_NAMES[3]]) for r in rows], stats['original_bootstrap'])


def validate_fresh_samples(stats):
    expected = {}
    for arm in ('TAN', 'LIF', 'LIF_THINNED', 'ZERO_OUTPUT'):
        step = stats['metrics'][('fresh', arm, 'step')]
        catch = stats['metrics'][('fresh', arm, 'catch')]
        tpr, fpr = step['detected'].astype(float), catch['detected'].astype(float)
        values = {'step_R': step['R'], 'catch_R': catch['R'], 'TPR_sample': tpr,
                  'FPR_sample': fpr, 'BA_sample': .5*(tpr+1-fpr), 'J_sample': tpr-fpr}
        expected.update({arm+'__'+key: value for key, value in values.items()})
    expected.update({name: stats['contrast_values'][:, i] for i, name in enumerate(CONTRAST_NAMES)})
    with np.load(HERE / 'fresh_samples.npz', allow_pickle=False) as saved:
        check('fresh per-history sample keys', set(saved.files) == set(expected))
        for key, value in expected.items():
            compare('all512 per-history samples '+key, saved[key], value)


def validate_clean_json(output):
    expected = {'native_trajectories': 4, 'deterministic_not_replicated': True,
                'LIF_certificate': {'spike_indices': list(range(16, 60, 2)), 'total': 22, 'late': 20, 'R': .5},
                'arms': {}}
    for arm in ('TAN', 'LIF', 'LIF_THINNED', 'ZERO_OUTPUT'):
        conditions = {}
        for condition in ('step', 'catch'):
            if arm == 'ZERO_OUTPUT':
                total = onset = late = probability = 0.
            else:
                native = 'LIF' if arm == 'LIF_THINNED' else arm
                spikes = output[('clean', native, condition)]['spike'][0]
                total, onset, late = float(spikes.sum()), float(spikes[15:20].sum()), float(spikes[20:].sum())
                probability = 1. - (1.-KEEP)**onset if arm == 'LIF_THINNED' else float(onset >= 1.)
                if arm == 'LIF_THINNED':
                    total, onset, late = KEEP*total, KEEP*onset, KEEP*late
            conditions[condition] = {'total_spikes': total, 'onset_spikes': onset, 'late_spikes': late,
                                      'R': late/40., 'detector_positive_probability': probability,
                                      'count_type': 'exact_expectation' if arm == 'LIF_THINNED' else 'exact'}
        tpr, fpr = [conditions[c]['detector_positive_probability'] for c in ('step', 'catch')]
        expected['arms'][arm] = {'conditions': conditions, 'TPR': tpr, 'FPR': fpr,
                                 'specificity': 1.-fpr, 'BA': .5*(tpr+1.-fpr), 'J': tpr-fpr}
    actual = json.loads((HERE / 'clean_results.json').read_text())
    compare_json_tree('clean exact expectations', actual, expected)
    return expected


def validate_original_calls_and_json(stats):
    rows = list(csv.DictReader((HERE / 'original_call_results.csv').open()))
    check('original source-call100 records', len(rows) == 100)
    keyed = {(r['model'], int(r['seed'])): r for r in rows}
    check('original source-call no duplicates and complete grid',
          set(keyed) == {(m, s) for m in ('LIF', 'TAN') for s in range(50)})
    means = {}
    for model in ('LIF', 'TAN'):
        met = stats['metrics'][('original', model, 'step')]
        compare('original exact source-call rates '+model,
                [float(keyed[(model, seed)]['false_spike_rate']) for seed in range(50)], met['R'], exact=True)
        compare('original exact source-call totals '+model,
                [int(keyed[(model, seed)]['total_spikes']) for seed in range(50)], met['total_count'], exact=True)
        means[model] = {'false_spike_rate_mean': float(met['R'].mean()),
                        'false_spike_rate_sample_sd': float(met['R'].std(ddof=1))}
    phase = json.loads((HERE / 'PHASE_A_COMPLETE.json').read_text())
    compare_json_tree('phaseA source metrics', phase, {'records': 100, 'exact_raw_record_matches': 100,
                       'means_and_sample_sd': means, 'supplementary_metrics_computed_yet': False,
                       'fresh_inputs_generated_yet': False})
    for field, path in [('original_calls_sha256', 'original_call_results.csv'),
                         ('original_inputs_sha256', 'original_inputs.npz'),
                         ('original_trajectories_sha256', 'original_trajectories.npz')]:
        check('phaseA artifact hash '+path, phase[field] == sha(HERE / path))
    actual = json.loads((HERE / 'original_results.json').read_text())
    compare_json_tree('original result summary', actual, {'historical_labels_and_tests_unchanged': True,
                      'phase_A_completion': phase,
                      'new_labeled_paired_bootstrap': {'contrast': CONTRAST_NAMES[3],
                          'mean': stats['original_point'], 'percentile_95_ci': stats['original_ci'].tolist(),
                          'n_histories': 50, 'replicates': 2000, 'seed': 2026110232,
                          'draws_sha256': sha(HERE / 'original_bootstrap_draws.npy')}})


def validate_primary_summary(stats, ledger):
    expected = {'protocol_sha256': PROTOCOL_HASH, 'n_histories': 512, 'n_conditions': 2,
                'n_arms': 4, 'n_ledger_rows': 4096, 'no_exclusions': True,
                'noise_seed': 2026110226, 'thinning_seed': 2026110230,
                'thinning_keep_probability': KEEP, 'bootstrap_seed': 2026110231,
                'bootstrap_replicates': 2000, 'fresh_inputs_sha256': sha(HERE / 'fresh_inputs.npz'),
                'bootstrap_draws_sha256': sha(HERE / 'fresh_bootstrap_draws.npy'),
                'arms': {}, 'paired_contrasts': {}}
    for arm, source in stats['summaries'].items():
        arm_expected = {key: source[key] for key in ('TPR', 'FPR', 'specificity', 'BA', 'J', 'noisy_minus_clean_step_R')}
        arm_expected['conditions'] = {}
        for condition in ('step', 'catch'):
            met = stats['metrics'][('fresh', arm, condition)]
            condition_data = {field: describe(met[key]) for field, key in
                              [('R', 'R'), ('observed_total_spikes', 'total_count'),
                               ('observed_onset_spikes', 'onset_count'), ('observed_late_spikes', 'late_count')]}
            rows = [ledger['fresh'][(i, condition, arm)] for i in range(512)]
            condition_data['generated_total_spikes'] = None if arm == 'ZERO_OUTPUT' else describe([r['generated_total_spikes'] for r in rows])
            condition_data['resource_proxies'] = None if arm == 'ZERO_OUTPUT' else {
                field: describe([r[field] for r in rows]) for field in LEDGER_FIELDS[13:]}
            arm_expected['conditions'][condition] = condition_data
        expected['arms'][arm] = arm_expected
    for i, name in enumerate(CONTRAST_NAMES):
        expected['paired_contrasts'][name] = {'mean': float(stats['bootstrap_point'][i]),
                                             'percentile_95_ci': stats['bootstrap_ci'][:, i].tolist()}
    expected['fresh_observed_late_budget_mismatch'] = {CONTRAST_NAMES[4]: float(stats['bootstrap_point'][4])}
    actual = json.loads((HERE / 'primary_results.json').read_text())
    compare_json_tree('primary summary', actual, expected)
    return expected


def validate_primary_provenance():
    manifest = json.loads((HERE / 'PRIMARY_MANIFEST.json').read_text())
    check('primary completed before independent comparison', manifest['status'] == 'COMPLETED')
    check('primary frozen protocol hash', manifest['protocol_sha256'] == PROTOCOL_HASH)
    check('primary executing script hash matches present source',
          manifest['primary_script_sha256'] == sha(HERE / 'run_noise_preservation.py'))
    check('primary contrast names and order', manifest['contrast_order'] == list(CONTRAST_NAMES))
    check('primary ledger columns', manifest['ledger_fields'] == list(LEDGER_FIELDS))
    for filename, spec in manifest['files'].items():
        check('primary declared artifact SHA256 '+filename, sha(HERE / filename) == spec['sha256'])
        check('primary declared artifact size '+filename, (HERE / filename).stat().st_size == spec['bytes'])
    for filename, fields in manifest['npz_schema'].items():
        with np.load(HERE / filename, allow_pickle=False) as saved:
            check('declared NPZ all fields '+filename, set(saved.files) == set(fields))
            for field, spec in fields.items():
                check('declared NPZ shape '+filename+'/'+field, list(saved[field].shape) == spec['shape'])
                check('declared NPZ dtype '+filename+'/'+field, str(saved[field].dtype) == spec['dtype'])
    frozen = json.loads((HERE / 'FRESH_INPUTS_FROZEN.json').read_text())
    check('producer input freeze states pre-evaluation', frozen['models_evaluated_yet'] is False)
    check('producer input freeze points to prior phaseA',
          frozen['phase_A_completion_sha256'] == sha(HERE / 'PHASE_A_COMPLETE.json'))
    phase = json.loads((HERE / 'PHASE_A_COMPLETE.json').read_text())
    check('phaseA marker precedes fresh input freeze', phase['time_utc'] < frozen['time_utc'])
    check('fresh input freeze precedes primary completion', frozen['time_utc'] < manifest['completed_utc'])
    for stage in ('fresh', 'clean'):
        filename = stage+'_inputs.npz'
        check('frozen input archive SHA256 '+stage, frozen[stage+'_inputs_sha256'] == sha(HERE / filename))
        with np.load(HERE / filename, allow_pickle=False) as saved:
            declarations = frozen[stage+'_array_manifest']
            check('frozen input raw array field coverage '+stage, set(saved.files) == set(declarations))
            for field in saved.files:
                arr = saved[field]
                spec = declarations[field]
                check('frozen raw input SHA256 '+stage+'/'+field,
                      hashlib.sha256(arr.tobytes(order='C')).hexdigest() == spec['raw_c_order_sha256'])
                check('frozen raw input shape/dtype '+stage+'/'+field,
                      list(arr.shape) == spec['shape'] and str(arr.dtype) == spec['dtype'])
    source_guard = json.loads((HERE / 'SOURCE_GUARD.json').read_text())
    check('producer source guard protocol SHA256', source_guard['protocol_sha256'] == PROTOCOL_HASH)
    for relative, declared in source_guard['source_files_sha256'].items():
        check('producer source guard SHA256 '+relative, sha(REPO / relative) == declared)
    names = sorted(set(manifest['files']) | {'PRIMARY_MANIFEST.json', 'run_noise_preservation.py',
                    'PREREGISTRATION.md', 'NOISE_SOURCE_MANIFEST.json', 'taskb_original_extracted.py'})
    return {name: sha(HERE / name) for name in names}


def make_readable_review(results):
    summaries = results['independent_summaries']
    lines = [
        '# Independent runtime review: original Task B signal preservation', '',
        'Status: **PASS**. This review regenerated the frozen inputs and recurrences without importing the primary simulator, its metric helpers, or the extracted source classes. Source ASTs were parsed for provenance only.', '',
        f"Frozen protocol SHA256: `{PROTOCOL_HASH}`.",
        f"Independent reviewer SHA256: `{results['independent_script_sha256']}`.", '',
        'The independent run checked all 100 historical records, 2,152 full native trajectories (129,120 native update steps), all 4,096 fresh native/surrogate observed trajectories, all 4,096 fresh ledger rows plus100original and4clean rows, and all six fresh plus the original paired bootstrap contrasts. All spike decisions match exactly. Floating state, attention, current, resource, and metric comparisons use the preregistered absolute tolerance 1e-12.', '',
        f"Maximum absolute residual across all numerical comparisons: `{results['max_abs_float_residual']:.17g}`.", '',
        '| Arm | Step hit rate | Catch false-alarm rate | Balanced accuracy | Step late rate |',
        '|---|---:|---:|---:|---:|',
    ]
    for arm in ('TAN', 'LIF', 'LIF_THINNED', 'ZERO_OUTPUT'):
        s = summaries[arm]
        lines.append(f"| {arm} | {s['TPR']:.9f} | {s['FPR']:.9f} | {s['BA']:.9f} | {s['step']['R']:.9f} |")
    lines += ['',
        'The exact clean LIF certificate is spikes at indices 16,18,...,58, total22 and20/40=.5 in the historical late window. Thus clean stimulus-driven spikes contribute to that endpoint. Native clean TAN has zero surprise, gate, and subsequent spikes from index24 after its window fills.', '',
        'A zero observed miss or false-alarm count can produce a degenerate empirical percentile-bootstrap interval. Such an interval is not evidence of perfect population sensitivity or specificity; it cannot represent unobserved error patterns. All confidence intervals here are descriptive empirical whole-history resampling intervals, with no multiplicity-adjusted claim.', '',
        'The fixed detector sees only whether a spike occurs in the known onset window15:20. This is a narrow diagnostic, not general denoising or an optimal decoder. Thinning uses a label-blind fixed probability529/1013 and changes observed communication, while the LIF internal trajectory remains unchanged; it does not equalize physical energy. Existing P1A findings and historical Task B labels are untouched.', '',
        'Both bootstrap streams were regenerated from their frozen seeds; all2,000fresh draws and all2,000original draws match exactly. Each draw resamples independent history groups and retains every model, condition, and thinning counterpart.', '',
        f"Primary artifacts unchanged during review: `{results['primary_artifacts_unchanged']}`.",
        f"Repository status unchanged: `{results['repository_status_unchanged']}`. Repository provenance: `{results['repository_after']}`.", '',
        'Completed command:', '',
        '`PYTHONDONTWRITEBYTECODE=1 python tan_audit/noise_preservation_v9/independent_review.py`', '',
        'Validation attempts and any harness fixes are preserved in numbered reviewer attempt JSON/log/script records; see INDEPENDENT_REVIEW_CHECKS.json. No protocol or primary outcome file was modified by this review.', '',
        'No remaining blocker.',
    ]
    return '\n'.join(lines) + '\n'


def main():
    started = datetime.now(timezone.utc).isoformat()
    attempt = 1
    while (HERE / f'review_attempt_{attempt:02d}.json').exists():
        attempt += 1
    prefix = HERE / f'review_attempt_{attempt:02d}'
    script_hash = sha(__file__)
    prefix.with_suffix('.script.txt').write_text(Path(__file__).read_text())
    result = {'status': 'RUNNING', 'started_utc': started, 'attempt': attempt,
              'independent_script_sha256': script_hash, 'protocol_sha256': PROTOCOL_HASH,
              'python': platform.python_version(), 'numpy': np.__version__,
              'execution_command': f'PYTHONDONTWRITEBYTECODE=1 python {Path(__file__).resolve()}',
              'implementation_independent': True,
              'note': 'No primary model/metric or extracted source code is imported or executed. Source/schema parsing and data comparison only.',
              'preexecution_harness_notes': [
                  'Scalar-only precheck found529/1013 and.2645/.5065 differ by1ULP; exact rational retained and only equality-to-decimal sanity check relaxed to machine epsilon before any trajectory execution.',
                  'Default repository path made relocatable with TAN_REPO_ROOT before execution.',
              ]}
    try:
        print('Independent review started', started, flush=True)
        source_validation()
        result['repository_before'] = git_status()
        result['primary_sha256'] = validate_primary_provenance()
        print('Source/protocol/primary provenance verified; reconstructing all trajectories.', flush=True)
        design = generated_design()
        validate_inputs(design)
        output = independent_outputs(design)
        stats = expected_statistics(output)
        validate_original_table(output, stats)
        validate_trajectories(output)
        independent_invariants(design, output, stats)
        ledger = validate_ledgers(output, stats)
        validate_bootstrap_draws(stats)
        validate_bootstrap_contrasts(stats)
        validate_fresh_samples(stats)
        clean_expected = validate_clean_json(output)
        validate_original_calls_and_json(stats)
        primary_expected = validate_primary_summary(stats, ledger)
        result['repository_after'] = git_status()
        result['repository_status_unchanged'] = (result['repository_before'] == result['repository_after']
                                                if result['repository_before']['available'] else None)
        source_validation()
        current_primary_hashes = {name: sha(HERE / name) for name in result['primary_sha256']}
        result['primary_artifacts_unchanged'] = current_primary_hashes == result['primary_sha256']
        check('all primary files unchanged throughout review', result['primary_artifacts_unchanged'])
        check('review source unchanged throughout run', script_hash == sha(__file__))
        result['status'] = 'PASS'
        result['independent_summaries'] = stats['summaries']
        result['independent_clean_expected'] = clean_expected
        result['independent_paired_contrasts'] = primary_expected['paired_contrasts']
        result['checked_counts'] = {
            'mandatory_source_files': len(json.loads((HERE/'NOISE_SOURCE_MANIFEST.json').read_text())['source_files']),
            'historical_records': 100, 'original_native_trajectories': 100,
            'fresh_noise_paths': 512, 'fresh_thinning_uniform_paths': 512,
            'fresh_native_trajectories': 2048, 'clean_native_trajectories': 4,
            'all_native_trajectories': 2152, 'all_native_update_steps': 129120,
            'TAN_history_vectors': 64560, 'TAN_history_entries': 645600,
            'fresh_observed_trajectories': 4096, 'fresh_observed_spike_decisions': 245760,
            'fresh_ledger_rows': 4096, 'original_ledger_rows': 100, 'clean_ledger_rows': 4,
            'fresh_bootstrap_replicates': 2000, 'fresh_bootstrap_index_entries': 1024000,
            'original_bootstrap_replicates': 2000, 'original_bootstrap_index_entries': 100000,
            'fresh_contrasts': 6, 'fresh_contrast_replicates': 12000,
            'original_contrast_replicates': 2000,
        }
        result['inference_limit'] = ('Empirical zero misses/false alarms may yield degenerate percentile-bootstrap intervals; '
                                     'these do not establish population perfection. Fixed known-onset detection only; '
                                     'observed-output thinning does not equalize energy.')
        print('PASS:100 historical records;2152 native full trajectories;4096 fresh ledger rows;all2000x6 fresh and2000 original contrasts.', flush=True)
    except Exception as exc:
        result['status'] = 'FAIL'
        result['exception'] = repr(exc)
        result['traceback'] = traceback.format_exc()
        print(result['traceback'], flush=True)
    finally:
        result['completed_utc'] = datetime.now(timezone.utc).isoformat()
        result['checks'] = CHECKS
        result['comparisons'] = COMPARISONS
        result['check_count'] = len(CHECKS)
        result['comparison_count'] = len(COMPARISONS)
        result['max_abs_float_residual'] = max((r['max_abs_error'] or 0. for r in COMPARISONS), default=0.)
        result['prior_attempt_files'] = [p.name for p in sorted(HERE.glob('review_attempt_*.json'))]
        json_write(prefix.with_suffix('.json'), result)
        json_write(HERE / 'INDEPENDENT_REVIEW_CHECKS.json', result)
        log_lines = [f"status={result['status']}", f'attempt={attempt}', f'started={started}',
                     f"completed={result['completed_utc']}", f"checks={len(CHECKS)}",
                     f"script_sha256={script_hash}"]
        if 'traceback' in result:
            log_lines.append(result['traceback'])
        prefix.with_suffix('.log').write_text('\n'.join(log_lines)+'\n')
        if result['status'] == 'PASS':
            (HERE / 'REVIEW.md').write_text(make_readable_review(result))
            print('Maximum absolute residual', result['max_abs_float_residual'], flush=True)
        else:
            (HERE / 'REVIEW.md').write_text('# Independent runtime review pending\n\n'
                f"Attempt{attempt} failed: `{result.get('exception')}`. The full error and exact reviewer script are preserved in `{prefix.name}.json`, `.log`, and `.script.txt`. Primary artifacts were not edited.\n")
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
