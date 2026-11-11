"""Independently replay and audit the sealed-design frozen-weight probe batch."""

from datetime import datetime
from pathlib import Path
import hashlib
import itertools
import json
import os
import time

import numpy as np

import independent_capture as capture
import independent_data as data
import independent_probe as probe
from reference import scalar_oracle


ROOT = Path(__file__).resolve().parent
ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2')
LOSSES = ('CE', 'COUNT', 'CECOM')
CONDITIONS = ('nominal', 'tag_removed', 'query_only')
TIMES = (9, 10, 14, 15, 19, 23)
DIMENSIONS = dict(branch=12, soma=6, previous_spikes=6, output=2, joint=26,
                  history=60, augmented=86, counts=2)
SOURCES = {'model.py', 'population.py', 'reference.py', 'original_tan.py',
           'task_data.py', 'independent_data.py'}
META_KEYS = ('id', 'seed', 'arm', 'loss')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    with path.open('x', encoding='utf-8') as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write('\n')


def load_arrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def digest(path):
    with path.open('rb') as stream:
        if hasattr(hashlib, 'file_digest'):
            return hashlib.file_digest(stream, 'sha256').hexdigest()
        return hashlib.sha256(stream.read()).hexdigest()


class Audit:
    def __init__(self):
        self.start = time.perf_counter()
        self.cpu_start = time.process_time()
        self.stage = 'locks'
        self.checks = 0
        self.captures = 0
        self.sample_steps = 0
        self.configs = 0
        self.probe_rows = 0
        self.native_rows = 0
        self.negative_probe_rows = 0
        self.negative_native_rows = 0
        self.pre_content_rows = 0
        self.scalar_models = 0
        self.max_numeric_error = 0.0
        self.max_coefficient_error = 0.0
        self.max_normal_residual = 0.0

    def check(self, condition, name):
        self.checks += 1
        if time.perf_counter() - self.start > 300:
            raise TimeoutError(f'300 s audit cap during {self.stage}')
        if not condition:
            raise AssertionError(name)

    def exact(self, got, expected, name):
        self.check(np.array_equal(got, expected), name)

    def near(self, got, expected, name, *, atol=1e-9, rtol=1e-9):
        got, expected = np.asarray(got), np.asarray(expected)
        self.check(got.shape == expected.shape, f'{name} shape')
        self.check(np.isfinite(got).all() and np.isfinite(expected).all(), f'{name} finite')
        error = float(np.max(np.abs(got - expected), initial=0.0))
        self.max_numeric_error = max(self.max_numeric_error, error)
        self.check(np.allclose(got, expected, atol=atol, rtol=rtol), f'{name}: maximum absolute error {error:.6g}')
        return error

    def finish(self):
        self.check((self.captures, self.sample_steps, self.configs, self.probe_rows, self.native_rows)
                   == (144, 1548288, 1332, 3996, 108), 'audit science counts')
        self.check((self.negative_probe_rows, self.negative_native_rows, self.pre_content_rows, self.scalar_models)
                   == (2664, 72, 216, 18), 'audit control counts')
        return dict(
            status='PASS', checks=self.checks, models=36, neural_training_updates=0,
            neural_capture_files=self.captures, forward_sample_steps=self.sample_steps,
            probe_configs=self.configs, probe_targets=self.configs * 21,
            probe_evaluations=self.probe_rows, native_evaluations=self.native_rows,
            negative_probe_rows=self.negative_probe_rows, negative_native_rows=self.negative_native_rows,
            pre_content_probe_rows=self.pre_content_rows, source_scalar_attention_models=self.scalar_models,
            source_scalar_episodes=2 * self.scalar_models, fit_label_permutations=20, test_label_permutations=20,
            max_numeric_error=self.max_numeric_error, max_coefficient_error=self.max_coefficient_error,
            max_normal_equation_residual=self.max_normal_residual,
            wall_seconds=time.perf_counter() - self.start, cpu_seconds=time.process_time() - self.cpu_start,
            cap_seconds=300,
            scope=('Independent stimulus and permutation regeneration, complete tap/native replay, '
                   'all 21 ridge right-hand sides per configuration via independent augmented least squares, '
                   'all fit/test scores and reported metrics, fit-before-test locks, exact input controls, '
                   'and original scalar TAN checks on two episodes per attention model.'),
        )


def index_rows(audit, rows, key_fields, expected):
    indexed = {tuple(row[key] for key in key_fields): row for row in rows}
    audit.check(len(rows) == len(indexed) and set(indexed) == expected, f'complete unique {key_fields} rows')
    return indexed


def model_configurations(arm):
    groups = ('branch', 'soma', 'previous_spikes', 'output', 'joint')
    if arm in ('FULL', 'NOATT'):
        groups += ('history', 'augmented')
    return list(itertools.product(groups, TIMES)) + [('counts', 23)]


def verify_locks(audit):
    freeze = read(ROOT / 'FREEZE.json')
    source = read(ROOT / 'SOURCE_LOCK.json')
    models = read(ROOT / 'MODELS.json')
    fit_lock = read(ROOT / 'FIT_LOCK.json')
    prior = ROOT.parent / 'DEND-TAN-loss-control-20261108'
    audit.check(freeze['scientific_data_generated'] is False and freeze['neural_weights_fixed'] is True,
                'pre-science freeze assertions')
    audit.check(fit_lock['test_generated'] is False, 'fit lock before test assertion')
    required = SOURCES | {'audit.py', 'capture.py', 'core.py', 'independent_capture.py', 'independent_probe.py',
                          'probe.py', 'run.py', 'PROTOCOL.md', 'PRECHECK.json', 'TIMING.json', 'SOURCE_LOCK.json',
                          'MODELS.json', 'models.npz'}
    audit.check(required <= set(freeze['files']), 'all mandatory freeze files')
    for name, expected in freeze['files'].items():
        audit.check(Path(name).name == name, f'local freeze filename {name}')
        audit.check(digest(ROOT / name) == expected, f'frozen hash {name}')
    for name in ('PRECHECK.json', 'TIMING.json'):
        audit.check(read(ROOT / name)['status'] == 'PASS', f'passed frozen {name}')
    old_manifest_path = prior / 'MANIFEST.json'
    old_hash = digest(old_manifest_path)
    old_manifest = read(old_manifest_path)['files']
    for lock in (models, source):
        audit.check(lock['previous_directory'] == f'../{prior.name}' and lock['previous_manifest_sha256'] == old_hash,
                    'source archive manifest identity')
    audit.check(set(source['files']) == SOURCES, 'source lock coverage')
    for name, expected in source['files'].items():
        audit.check(digest(ROOT / name) == digest(prior / name) == expected == old_manifest[name]['sha256'],
                    f'byte-identical archived source {name}')
        audit.check((ROOT / name).stat().st_size == (prior / name).stat().st_size == old_manifest[name]['bytes'],
                    f'archived source bytes {name}')
    weights = load_arrays(ROOT / 'models.npz')
    audit.check(set(weights) == {'theta'} and weights['theta'].shape == (36, 176), 'frozen weights shape')
    theta = weights['theta']
    prior_rows = read(prior / 'train/ROWS.json')
    model_rows = models['rows']
    audit.check(len(model_rows) == len(prior_rows) == 36, '36 archived model rows')
    expected_design = set(itertools.product(range(3), ARMS, LOSSES))
    index_rows(audit, model_rows, ('seed', 'arm', 'loss'), expected_design)
    expected_fit_files = {'fits/ROWS.json'} | {f'data/fit_s{s}.npz' for s in range(3)}
    for i, (model, old_row) in enumerate(zip(model_rows, prior_rows)):
        audit.check(model['index'] == i and all(model[key] == old_row[key] for key in (*META_KEYS, 'lr')),
                    f'archived model metadata {i}')
        ident = model['id']
        audit.check(ident == f"s{model['seed']}_{model['arm']}_{model['loss']}", f'canonical model ID {ident}')
        filename = f'train/{ident}.npz'
        audit.check(model['source_file'] == filename, f'archived weight path {ident}')
        audit.check(digest(prior / filename) == model['source_sha256'] == old_manifest[filename]['sha256'],
                    f'archived weight hash {ident}')
        audit.check((prior / filename).stat().st_size == old_manifest[filename]['bytes'], f'archived weight bytes {ident}')
        with np.load(prior / filename, allow_pickle=False) as archived:
            audit.exact(theta[i], archived['theta'], f'unchanged theta {ident}')
        expected_fit_files.add(f'snapshots/{ident}_fit_nominal.npz')
        expected_fit_files.update(f'fits/{ident}_{group}_t{tap}.npz' for group, tap in model_configurations(model['arm']))
    audit.check(set(fit_lock['files']) == expected_fit_files, 'complete fit lock coverage')
    freeze_time = (ROOT / 'FREEZE.json').stat().st_mtime_ns
    fit_lock_time = (ROOT / 'FIT_LOCK.json').stat().st_mtime_ns
    audit.check(datetime.fromisoformat(freeze['utc']) < datetime.fromisoformat(fit_lock['utc']), 'lock UTC order')
    for name, expected in fit_lock['files'].items():
        path = ROOT / name
        audit.check(freeze_time <= path.stat().st_mtime_ns <= fit_lock_time, f'fit creation before test lock {name}')
        audit.check(digest(path) == expected, f'fit lock hash {name}')
    for seed in range(3):
        path = ROOT / f'data/test_s{seed}.npz'
        audit.check(path.stat().st_mtime_ns >= fit_lock_time, f'test generated after fit lock s{seed}')
    return model_rows, theta


def opposite_pairs(base, shared_field):
    """Pair opposite labels within a quartet while holding one logical factor fixed."""
    order = np.lexsort((base['y'], base[shared_field], base['quartet']))
    return order[::2], order[1::2]


def base_and_inputs(audit, block, split):
    n, offset, shuffle_offset = (256, 4001010000, 4001030000) if split == 'fit' else (512, 4001020000, 4001040000)
    base = load_arrays(ROOT / f'data/{split}_s{block}.npz')
    expected = data.make_base(n, offset + 100 * block)
    audit.check(set(base) == set(expected) | {'permuted_y'}, f'{split} s{block} data keys')
    for name, value in expected.items():
        audit.exact(base[name], value, f'{split} s{block} independently generated {name}')
    quartets = np.argsort(base['quartet'], kind='stable').reshape(-1, 4)
    permuted = base['permuted_y']
    audit.check(permuted.shape == (n, 20), f'{split} s{block} permutation shape')
    audit.exact(permuted[quartets].sum(axis=1), np.full((n // 4, 20), 2), f'{split} s{block} permutation balance')
    rng = np.random.default_rng(shuffle_offset + 100 * block)
    for column in range(20):
        for quartet in quartets:
            audit.exact(permuted[quartet, column], rng.permutation(base['y'][quartet]), f'{split} s{block} permutation replay')
    prefixes = opposite_pairs(base, 'selected')
    absent_tag = opposite_pairs(base, 'first_bit')
    for left, right in (prefixes, absent_tag):
        audit.exact(base['quartet'][left], base['quartet'][right], f'{split} s{block} matched nuisance')
        audit.exact(base['y'][left], 1 - base['y'][right], f'{split} s{block} opposite labels')
    inputs = {}
    for condition in CONDITIONS if split == 'test' else ('nominal',):
        x, q = data.assemble(base, condition)
        data.check(base, x, condition)
        audit.check(q == 16 and x.shape == (n, 24, 6), f'{split} s{block} {condition} timing')
        inputs[condition] = x
    left, right = prefixes
    audit.exact(inputs['nominal'][left, :10], inputs['nominal'][right, :10], f'{split} s{block} t9 opposite-label input prefixes')
    return base, inputs, prefixes, absent_tag


def check_capture(audit, model, theta, split, condition, x, prefixes, absent_tag):
    name = f"{model['id']}_{split}_{condition}"
    actual = load_arrays(ROOT / f'snapshots/{name}.npz')
    expected = capture.capture(x, theta, model['arm'], taps=TIMES)
    audit.check(set(actual) == set(expected), f'{name} snapshot fields')
    for key, value in expected.items():
        if key in ('p', 'counts', 'tap_counts', 'resources'):
            audit.exact(actual[key], value, f'{name} {key} independent replay')
        else:
            audit.near(actual[key], value, f'{name} {key} independent replay')
    audit.exact(actual['tap_counts'][:, -1], actual['counts'], f'{name} native full query counts')
    def identical(key, left, right, description):
        if key in ('d', 'v', 'o', 'H'):
            audit.near(left, right, description, atol=1e-12, rtol=0.)
        else:
            audit.exact(left, right, description)
    if condition == 'nominal':
        left, right = prefixes
        for key in set(actual) - {'counts', 'resources'}:
            identical(key, actual[key][left, 0], actual[key][right, 0], f'{name} t9 same prefix {key}')
    elif condition == 'tag_removed':
        left, right = absent_tag
        for key, value in actual.items():
            identical(key, value[left], value[right], f'{name} identical opposite-label {key}')
    else:
        for key, value in actual.items():
            identical(key, value, np.broadcast_to(value[0], value.shape), f'{name} uniform input {key}')
    if split == 'test' and condition == 'nominal' and model['arm'] in ('FULL', 'NOATT'):
        scalar = scalar_oracle(x[:2], theta, model['arm'], 16)
        states = np.concatenate([actual[key][:2] for key in ('d', 'v', 'o')], axis=2)
        audit.near(states, scalar['states'][:, TIMES], f'{name} original scalar TAN state')
        audit.near(actual['H'][:2], scalar['histories'][:, TIMES], f'{name} original scalar TAN history')
        audit.exact(actual['p'][:2], scalar['spikes'][:, TIMES, :6], f'{name} original scalar TAN hidden spikes')
        audit.exact(actual['counts'][:2], scalar['counts'], f'{name} original scalar TAN query counts')
        audit.scalar_models += 1
    audit.captures += 1
    audit.sample_steps += len(x) * 24
    return actual


def native_metrics(counts, labels):
    higher_one = counts[:, 1] > counts[:, 0]
    higher_zero = counts[:, 0] > counts[:, 1]
    tie = ~(higher_one | higher_zero)
    silence = (counts == 0).all(axis=1)
    saturation = (counts == 8).all(axis=1)
    strict = (higher_zero[labels == 0].mean() + higher_one[labels == 1].mean()) / 2
    forced = ((~higher_one[labels == 0]).mean() + higher_one[labels == 1].mean()) / 2
    return dict(strict_ba=strict, forced_ba=forced, coverage=(~tie).mean(), silence=silence.mean(),
                saturated_tie=saturation.mean(), intermediate_tie=(tie & ~silence & ~saturation).mean())


def audit_native(audit, actual, record, labels, condition):
    expected = native_metrics(actual['counts'], labels)
    for key, value in expected.items():
        audit.exact(record[key], value, f"{record['id']} {condition} native {key}")
    audit.exact(record['mean_resources'], actual['resources'].mean(axis=0), f"{record['id']} {condition} resource ledger")
    if condition != 'nominal':
        audit.exact(record['forced_ba'], .5, f'{condition} forced native null')
        audit.exact(record['strict_ba'], .5 * record['coverage'], f'{condition} strict native null')
        audit.negative_native_rows += 1
    audit.native_rows += 1


def verify_fit(audit, record, features, targets, labels):
    ident = record['probe_id']
    saved = load_arrays(ROOT / f'fits/{ident}.npz')
    audit.check(set(saved) == {'mean', 'scale', 'active', 'coef', 'fit_scores'}, f'{ident} fit artifact fields')
    audit.check(features.shape == (256, DIMENSIONS[record['group']]) and record['dimension'] == features.shape[1],
                f'{ident} feature dimension')
    expected = probe.fit(features, targets, alpha=.1)
    audit.near(saved['mean'], expected['mean'], f'{ident} fit-only mean')
    active = ~expected['constant']
    audit.exact(saved['active'], active, f'{ident} fit-only constant mask')
    audit.check(record['active_dimension'] == int(active.sum()), f'{ident} active dimensions')
    audit.near(saved['scale'], np.where(active, expected['scale'], 1e-8), f'{ident} fit-only scale', atol=1e-14)
    coefficients = np.vstack((expected['coef'], expected['intercept']))
    error = audit.near(saved['coef'], coefficients, f'{ident} 21 ridge coefficient columns')
    audit.max_coefficient_error = max(audit.max_coefficient_error, error)
    audit.near(saved['fit_scores'], probe.scores(features, expected), f'{ident} fit scores')
    stored = dict(mean=saved['mean'], scale=saved['scale'], constant=~saved['active'],
                  coef=saved['coef'][:-1], intercept=saved['coef'][-1])
    audit.near(saved['fit_scores'], probe.scores(features, stored), f'{ident} fit scores from archived coefficients')
    z = (features - stored['mean']) / stored['scale']
    z[:, ~active] = 0
    residual = saved['fit_scores'] - targets
    normal = z.T @ residual / len(features) + .1 * stored['coef']
    error = max(float(np.max(np.abs(normal))), float(np.max(np.abs(residual.mean(axis=0)))))
    audit.max_normal_residual = max(audit.max_normal_residual, error)
    audit.check(error < 1e-8, f'{ident} normal-equation residual {error:.6g}')
    metrics = probe.metrics(saved['fit_scores'][:, 0], labels)
    audit.exact(record['fit_ba'], metrics['balanced_accuracy'], f'{ident} fit BA')
    audit.near(record['fit_mse'], metrics['mse'], f'{ident} fit MSE')
    audit.configs += 1
    return expected, stored


def verify_probe_tests(audit, fit_record, captured, labels, permuted, expected, stored, test_index, prefixes, absent_tag):
    ident = fit_record['probe_id']
    archive = load_arrays(ROOT / f'evaluate/{ident}.npz')
    audit.check(set(archive) == {'test_scores'} and archive['test_scores'].shape == (3, 512, 21), f'{ident} test score shape')
    for condition, scores in zip(CONDITIONS, archive['test_scores']):
        record = test_index[(ident, condition)]
        audit.check(all(record[key] == value for key, value in fit_record.items()), f'{ident} {condition} metadata')
        features = capture.features(captured[condition], fit_record['group'], fit_record['time'])
        audit.near(scores, probe.scores(features, expected), f'{ident} {condition} independent test scores', atol=1e-8)
        audit.near(scores, probe.scores(features, stored), f'{ident} {condition} archived fit test scores', atol=1e-8)
        metric = probe.metrics(scores[:, 0], labels)
        audit.exact(record['balanced_accuracy'], metric['balanced_accuracy'], f'{ident} {condition} test BA')
        for name in ('mse', 'score_rms'):
            audit.near(record[name], metric[name], f'{ident} {condition} test {name}')
        if condition != 'nominal' or fit_record['time'] == 9:
            if condition == 'query_only':
                left, right = scores, np.broadcast_to(scores[0], scores.shape)
            else:
                first, second = absent_tag if condition == 'tag_removed' else prefixes
                left, right = scores[first], scores[second]
            audit.near(left, right, f'{ident} {condition} duplicate scores', atol=1e-7, rtol=0.)
            audit.exact(probe.classify(left), probe.classify(right), f'{ident} {condition} duplicate decisions')
        if condition == 'nominal':
            fit_null = [probe.metrics(scores[:, j + 1], labels) for j in range(20)]
            test_null = [probe.metrics(scores[:, 0], permuted[:, j]) for j in range(20)]
            for prefix, rows in (('fit_shuffle_', fit_null), ('test_shuffle_', test_null)):
                audit.exact(record[prefix + 'ba'], [r['balanced_accuracy'] for r in rows], f'{ident} {prefix} BA')
                audit.near(record[prefix + 'mse'], [r['mse'] for r in rows], f'{ident} {prefix} MSE')
            audit.near(record['fit_shuffle_score_rms'], [r['score_rms'] for r in fit_null], f'{ident} fit shuffle RMS')
            if fit_record['time'] == 9:
                audit.exact(record['balanced_accuracy'], .5, f'{ident} t9 before content BA')
                audit.pre_content_rows += 1
        else:
            audit.check(not any('shuffle' in key for key in record), f'{ident} {condition} nominal-only null summaries')
            audit.exact(record['balanced_accuracy'], .5, f'{ident} {condition} missing information BA')
            audit.negative_probe_rows += 1
        audit.probe_rows += 1


def verify_results(audit, models, theta):
    expected_fits = {(f"{model['id']}_{g}_t{t}",) for model in models for g, t in model_configurations(model['arm'])}
    fit_index = index_rows(audit, read(ROOT / 'fits/ROWS.json'), ('probe_id',), expected_fits)
    expected_tests = {(probe_id, c) for (probe_id,) in expected_fits for c in CONDITIONS}
    test_index = index_rows(audit, read(ROOT / 'evaluate/ROWS.json'), ('probe_id', 'condition'), expected_tests)
    expected_native = {(m['id'], c) for m in models for c in CONDITIONS}
    native_index = index_rows(audit, read(ROOT / 'evaluate/NATIVE_ROWS.json'), ('id', 'condition'), expected_native)
    audit.check({p.stem for p in (ROOT / 'evaluate').glob('*.npz')} == {k[0] for k in expected_fits},
                'complete archived test score files')
    expected_snapshots = {f"{m['id']}_fit_nominal" for m in models} | {f"{ident}_test_{c}" for ident, c in expected_native}
    audit.check({p.stem for p in (ROOT / 'snapshots').glob('*.npz')} == expected_snapshots, 'complete archived snapshots')
    nuisance_ids = set()
    for block in range(3):
        audit.stage = f'data block {block}'
        fit, fit_inputs, fit_prefixes, fit_absent = base_and_inputs(audit, block, 'fit')
        test, test_inputs, test_prefixes, test_absent = base_and_inputs(audit, block, 'test')
        for base in (fit, test):
            values = {(float(a), float(b)) for a, b in zip(base['amplitude'], base['tag_amplitude'])}
            audit.check(len(values) == len(base['amplitude']) and not values & nuisance_ids, 'separate nuisance quartets across all splits')
            nuisance_ids.update(values)
        targets = 2 * np.column_stack((fit['y'], fit['permuted_y'])) - 1
        for model in (m for m in models if m['seed'] == block):
            ident = model['id']
            audit.stage = ident
            weights = theta[model['index']]
            fit_snap = check_capture(audit, model, weights, 'fit', 'nominal', fit_inputs['nominal'], fit_prefixes, fit_absent)
            test_snap = {}
            for condition in CONDITIONS:
                test_snap[condition] = check_capture(audit, model, weights, 'test', condition, test_inputs[condition],
                                                     test_prefixes, test_absent)
                record = native_index[(ident, condition)]
                audit.check(all(model[k] == record[k] for k in META_KEYS), f'{ident} {condition} native metadata')
                audit_native(audit, test_snap[condition], record, test['y'], condition)
            for group, tap in model_configurations(model['arm']):
                record = fit_index[(f'{ident}_{group}_t{tap}',)]
                audit.check(all(model[k] == record[k] for k in META_KEYS) and (record['group'], record['time']) == (group, tap),
                            f'{ident} {group} t{tap} fit metadata')
                features = capture.features(fit_snap, group, tap)
                expected, stored = verify_fit(audit, record, features, targets, fit['y'])
                verify_probe_tests(audit, record, test_snap, test['y'], test['permuted_y'], expected, stored,
                                   test_index, test_prefixes, test_absent)
        print(f'Independent audit block {block + 1}/3 complete', flush=True)


def verify_science_summary(audit):
    state = read(ROOT / 'RUN_STATE.json')
    summary = read(ROOT / 'SCIENCE_SUMMARY.json')
    environment = read(ROOT / 'ENVIRONMENT.json')
    audit.check(state['status'] == 'complete' and summary['status'] == 'COMPLETE', 'one-shot completed science')
    audit.check(not (ROOT / 'SCIENCE_FAILURE.json').exists(), 'no science failure marker')
    expected = dict(models=36, neural_training_updates=0, probe_configs=audit.configs, probe_targets=audit.configs * 21,
                    probe_evaluations=audit.probe_rows, native_evaluations=audit.native_rows,
                    neural_capture_files=audit.captures, forward_sample_steps=audit.sample_steps,
                    dense_forward_macs=audit.sample_steps * 156, cap_seconds=840)
    for name, value in expected.items():
        audit.check(summary[name] == value, f'science summary {name}')
    audit.check(0 < summary['wall_seconds'] <= 840 and summary['peak_working_set_bytes'] <= 1024**3,
                'science wall-clock and memory budgets')
    audit.check(0 < summary['artifact_bytes_at_science_end'] < int(1.5 * 1024**3), 'science disk budget')
    audit.check(all(environment['threads'][k] == '1' and os.environ.get(k) == '1'
                    for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')), 'single-thread audit and science')


def main():
    audit = Audit()
    try:
        audit.check(not (ROOT / 'AUDIT.json').exists(), 'audit already sealed')
        models, theta = verify_locks(audit)
        verify_results(audit, models, theta)
        verify_science_summary(audit)
        result = audit.finish()
        save(ROOT / 'AUDIT.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as error:
        failure_path = ROOT / 'AUDIT_FAILURE.json'
        if failure_path.exists():
            failure_path = ROOT / f'AUDIT_FAILURE_{time.time_ns()}.json'
        save(failure_path, dict(status='FAIL', stage=audit.stage, checks=audit.checks,
                                wall_seconds=time.perf_counter() - audit.start,
                                exception=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    main()
