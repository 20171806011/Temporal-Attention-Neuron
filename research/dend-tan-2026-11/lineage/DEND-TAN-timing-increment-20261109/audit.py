"""Independent audit of the frozen count-matched timing experiment."""

from datetime import datetime
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import time

import numpy as np

import independent_capture as capture
import independent_data as data
import independent_stats as stats
from reference import scalar_oracle


ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'DEND-TAN-readout-path-20261109'
ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2')
LOSSES = ('CE', 'COUNT', 'CECOM')
META = ('id', 'seed', 'arm', 'loss')
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
PAIR_FIELDS = ('low', 'high', 'quartet', 'weight')
EXPECTED = dict(models=36, neural_training_updates=0, rank_fits=612, class_fits=72,
                baseline_captures=72, forward_sample_steps=1327104, dense_forward_macs=207028224)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, content):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Audit:
    def __init__(self):
        self.start = time.perf_counter()
        self.cpu = time.process_time()
        self.stage = 'source and phase locks'
        self.checks = self.baselines = self.rank_fits = self.class_fits = self.steps = 0
        self.shuffles = self.scalar_models = self.matched_test_models = 0
        self.max_numeric_error = self.max_coefficient_error = self.max_normal_residual = 0.0

    def check(self, condition, context):
        self.checks += 1
        if time.perf_counter() - self.start > 300:
            raise TimeoutError(f'300 s audit cap during {self.stage}')
        if not condition:
            raise AssertionError(context)

    def exact(self, actual, expected, context):
        self.check(np.array_equal(actual, expected), context)

    def near(self, actual, expected, context, *, atol=1e-9, rtol=1e-9):
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.check(actual.shape == expected.shape, context + ' shape')
        self.check(np.isfinite(actual).all() and np.isfinite(expected).all(), context + ' finite')
        error = float(np.max(np.abs(actual - expected), initial=0.0))
        self.max_numeric_error = max(error, self.max_numeric_error)
        self.check(np.allclose(actual, expected, atol=atol, rtol=rtol), f'{context}: max error {error:.6g}')
        return error

    def coefficient(self, actual, expected, context):
        self.max_coefficient_error = max(self.max_coefficient_error, self.near(actual, expected, context))

    def normal_residual(self, residual, context):
        error = float(np.max(np.abs(residual), initial=0.0))
        self.max_normal_residual = max(error, self.max_normal_residual)
        self.check(error < 1e-9, f'{context}: normal residual {error:.6g}')

    def metrics(self, actual, expected, context):
        for key, value in expected.items():
            got = actual[key]
            if value is None:
                self.check(got is None, f'{context} undefined {key}')
            elif isinstance(value, (list, tuple)) and any(x is None for x in value):
                self.check(got == list(value), f'{context} undefined {key} vector')
            elif isinstance(value, (float, np.floating, np.ndarray)) or (
                isinstance(value, (list, tuple)) and value and isinstance(value[0], (float, np.floating))
            ):
                self.near(got, value, f'{context} {key}', atol=1e-12, rtol=1e-10)
            else:
                self.exact(got, value, f'{context} {key}')


def source_locks(audit):
    freeze, source, fit = (read(ROOT / name) for name in ('FREEZE.json', 'SOURCE_LOCK.json', 'FIT_LOCK.json'))
    audit.check(freeze['scientific_data_generated'] is False and freeze['neural_weights_fixed'] is True,
                'pre-science freeze and fixed weights')
    audit.check(fit['test_generated'] is False, 'no test before fit lock assertion')
    old_manifest_file = PRIOR / 'MANIFEST.json'
    manifest = read(old_manifest_file)['files']
    audit.check(source['previous_directory'] == f'../{PRIOR.name}' and
                source['previous_manifest_sha256'] == sha(old_manifest_file), 'parent manifest identity')
    audit.check(source['prior_raster_source_prefix'] == 'fit/', 'old raster probe directory mapping')
    audit.check(read(PRIOR / 'AUDIT.json')['status'] == 'PASS' and
                sha(PRIOR / 'AUDIT.json') == manifest['AUDIT.json']['sha256'], 'audited parent batch')
    old_fit_lock_path = PRIOR / 'FIT_LOCK.json'
    audit.check(sha(old_fit_lock_path) == manifest['FIT_LOCK.json']['sha256'], 'parent fit lock identity')
    old_fit_lock = read(old_fit_lock_path)['files']

    rows = read(ROOT / 'MODELS.json')['rows']
    audit.check(len(rows) == 36 and {(r['seed'], r['arm'], r['loss']) for r in rows} == set(product(range(3), ARMS, LOSSES)),
                '36 complete archived models')
    for i, row in enumerate(rows):
        audit.check(row['index'] == i and row['id'] == f"s{row['seed']}_{row['arm']}_{row['loss']}",
                    f'archive model row {i}')
    theta = load(ROOT / 'models.npz')
    audit.check(set(theta) == {'theta'} and theta['theta'].shape == (36, 176) and
                np.isfinite(theta['theta']).all(), 'finite archived weights')

    probes = {f"prior_raster/{row['id']}_raster.npz" for row in rows}
    audit.check({name for name in source['files'] if name.startswith('prior_raster/')} == probes,
                '36 prior raster probes covered')
    audit.check({'model.py', 'population.py', 'reference.py', 'original_tan.py', 'task_data.py',
                 'independent_data.py', 'independent_capture.py', 'continuation.py', 'models.npz',
                 'MODELS.json'} <= set(source['files']), 'complete reused model/data sources')
    for name, digest in source['files'].items():
        relative = Path(name)
        audit.check(not relative.is_absolute() and '..' not in relative.parts, f'local source path {name}')
        previous_name = 'fit/' + relative.name if name in probes else name
        if name in probes:
            audit.check(digest == old_fit_lock[previous_name], f'{name} parent fit-lock membership')
        entry = manifest[previous_name]
        audit.check(digest == entry['sha256'] == sha(ROOT / relative) == sha(PRIOR / previous_name),
                    f'{name} immutable bytes')
        audit.check((ROOT / relative).stat().st_size == entry['bytes'], f'{name} archive size')

    required = set(source['files']) | {'audit.py', 'independent_stats.py', 'run.py', 'PROTOCOL.md',
                                      'SOURCE_LOCK.json', 'PRECHECK.json', 'TIMING.json'}
    audit.check(required <= set(freeze['files']), 'freeze covers source, settings and verification')
    for name, digest in freeze['files'].items():
        relative = Path(name)
        audit.check(not relative.is_absolute() and '..' not in relative.parts, f'local freeze path {name}')
        audit.check(sha(ROOT / relative) == digest, f'frozen file {name}')
    for name in ('PRECHECK.json', 'TIMING.json'):
        audit.check(read(ROOT / name)['status'] == 'PASS', f'accepted frozen {name}')

    fit_files = {f"fit/{row['id']}_{kind}.npz" for row in rows for kind in ('native', 'fit')}
    fit_data_files = {f'data/fit{tag}_s{block}.npz' for tag in ('', '_perm') for block in range(3)}
    audit.check(set(fit['files']) == fit_files | fit_data_files | {'fit/ROWS.json'}, 'complete calibration lock')
    freeze_time = (ROOT / 'FREEZE.json').stat().st_mtime_ns
    fit_time = (ROOT / 'FIT_LOCK.json').stat().st_mtime_ns
    audit.check(datetime.fromisoformat(freeze['utc']) < datetime.fromisoformat(fit['utc']), 'lock UTC chronology')
    for name, digest in fit['files'].items():
        path = ROOT / name
        audit.check(freeze_time <= path.stat().st_mtime_ns <= fit_time and sha(path) == digest,
                    f'calibration artifact before test {name}')
    test_files = [* (ROOT / 'test').glob('*'), * (ROOT / 'data').glob('test*.npz')]
    audit.check(len(test_files) == 79 and all(path.stat().st_mtime_ns >= fit_time for path in test_files),
                'test datasets, controls and scores generated after fit lock')
    return rows, theta['theta']


def read_data(audit, split, block, nuisance_keys):
    n_rows, data_seed, permutation_seed = (512, 4021010000, 4021030000) if split == 'fit' else (1024, 4021020000, 4021040000)
    name = f'{split} s{block}'
    actual = load(ROOT / f'data/{split}_s{block}.npz')
    expected = data.make_base(n_rows, data_seed + 100 * block)
    audit.check(set(actual) == set(expected), f'{name} data fields')
    for key in expected:
        audit.exact(actual[key], expected[key], f'{name} independently generated {key}')
    x, q = data.assemble(actual, 'nominal')
    audit.check(q == 16, 'query starts at t16')
    data.check(actual, x, 'nominal')
    keys = set(zip(actual['amplitude'], actual['tag_amplitude']))
    audit.check(len(keys) == n_rows // 4 and not keys.intersection(nuisance_keys), 'distinct matched quartets across splits')
    nuisance_keys.update(keys)
    archive = load(ROOT / f'data/{split}_perm_s{block}.npz')
    audit.check(set(archive) == {'permutations'}, f'{name} permutation archive fields')
    permutation = archive['permutations']
    audit.check(permutation.dtype == np.uint8, f'{name} uint8 permutation storage')
    audit.exact(permutation, stats.permutation_maps(n_rows, permutation_seed + 100 * block),
                f'{name} independently regenerated joint time orders')
    return actual, x, permutation


def verify_native(audit, split, model, x, theta):
    name, arm = model['id'], model['arm']
    saved = load(ROOT / f'{split}/{name}_native.npz')
    expected = capture.capture(x, theta, arm, taps=())
    audit.check(set(saved) == set(expected), f'{name} {split} native fields')
    for key, value in expected.items():
        audit.exact(saved[key], value, f'{name} {split} native {key}')
    audit.check(saved['raster'].shape == (len(x), 8, 2) and np.all((saved['raster'] == 0) | (saved['raster'] == 1)),
                f'{name} {split} eight binary query steps')
    audit.exact(saved['counts'], saved['raster'].sum(axis=1), f'{name} {split} count sums')
    audit.baselines += 1
    audit.steps += len(x) * 24
    if split == 'test' and arm in capture.ATTENTION_ARMS:
        scalar = scalar_oracle(x[:2], theta, arm, 16)
        audit.exact(saved['raster'][:2], scalar['spikes'][:, 16:, 6:], f'{name} original scalar TAN raster')
        audit.exact(saved['counts'][:2], scalar['counts'], f'{name} original scalar TAN counts')
        audit.scalar_models += 1
    return saved


def verify_pairs(audit, archive, base, counts, context):
    pairs = stats.paired_indices(counts, base['y'], base['quartet'])
    for key, value in zip(PAIR_FIELDS, pairs):
        compare = audit.near if key == 'weight' else audit.exact
        compare(archive[key], value, f'{context} {key}')
    low, high, quartets, weight = pairs
    audit.exact(counts[low], counts[high], f'{context} identical two-channel counts')
    if len(low):
        audit.near(weight.sum(), 1.0, f'{context} total pair weight')
        _, inverse = np.unique(quartets, return_inverse=True)
        per_group = np.bincount(inverse, weights=weight)
        audit.near(per_group, np.full(len(per_group), 1.0 / len(per_group)), f'{context} equal quartet weight')
    return pairs


def decisions(counts):
    tied = counts[:, 0] == counts[:, 1]
    forced = (counts[:, 1] > counts[:, 0]).astype(int)
    return np.where(tied, -1, forced), forced


def features_for_controls(audit, raster, maps, context):
    codes = np.sort(2 * raster[:, :, 0] + raster[:, :, 1], axis=1)
    counts = raster.sum(axis=1)
    native_decisions = decisions(counts)
    true = stats.helmert_features(raster)
    audit.near(np.sum(true**2, axis=1), raster.sum(axis=(1, 2)), f'{context} orthogonal squared norms')
    audit.near(true[:, :2], counts / np.sqrt(8.0), f'{context} constant Helmert component')
    yield true
    for i, permutation in enumerate(maps):
        shuffled = stats.shuffled_raster(raster, permutation)
        shuffled_counts = shuffled.sum(axis=1)
        audit.exact(shuffled_counts, counts, f'{context} shuffle {i} exact counts')
        audit.exact(np.sort(2 * shuffled[:, :, 0] + shuffled[:, :, 1], axis=1), codes,
                    f'{context} shuffle {i} joint-bin multiset including synchrony')
        audit.exact(decisions(shuffled_counts), native_decisions, f'{context} shuffle {i} native decisions')
        audit.shuffles += 1
        yield stats.helmert_features(shuffled)


def verify_fit(audit, archive, record, base, native, maps, context):
    keys = {'pair_coef', 'count_mean', 'count_coef', 'raster_mean', 'raster_coef', 'intercept', 'pair_scores', *PAIR_FIELDS}
    audit.check(set(archive) == keys and archive['pair_coef'].shape == (17, 14), f'{context} fit artifact fields')
    low, high, groups, weights = verify_pairs(audit, archive, base, native['counts'], context)
    audit.check(archive['pair_scores'].shape == (17, len(low)), f'{context} fit pair score shape')
    coefficients = []
    statuses = []
    class_fits = {}
    for k, features in enumerate(features_for_controls(audit, native['raster'], maps, context)):
        differences = features[high, 2:] - features[low, 2:]
        coef = stats.pair_fit(differences, weights)
        coefficients.append(coef)
        status = 'NO_PAIRS' if not len(low) else 'ZERO_DIFFERENCES' if not np.any(differences) else 'FITTED'
        statuses.append(status)
        audit.coefficient(archive['pair_coef'][k], coef, f'{context} condition {k} weighted SVD')
        fit_scores = differences @ coef
        audit.near(archive['pair_scores'][k], fit_scores, f'{context} condition {k} fit differences')
        saved_coef = archive['pair_coef'][k]
        saved_residual = differences @ saved_coef - 1.0
        normal = differences.T @ (weights * saved_residual) + stats.ALPHA * saved_coef
        audit.normal_residual(normal, f'{context} condition {k}')
        audit.rank_fits += 1

        if k == 0:
            for kind, matrix in (('count', features[:, :2]), ('raster', features)):
                mean, weights_class, intercept = stats.class_fit(matrix, base['y'])
                class_fits[kind] = (mean, weights_class, intercept)
                audit.near(archive[f'{kind}_mean'], mean, f'{context} {kind} calibration mean')
                audit.coefficient(archive[f'{kind}_coef'], weights_class, f'{context} {kind} augmented SVD')
                audit.near(archive['intercept'], intercept, f'{context} {kind} unpenalized intercept')
                center = matrix - archive[f'{kind}_mean']
                saved_coef_class = archive[f'{kind}_coef']
                residual = center @ saved_coef_class + archive['intercept'] - (2.0 * base['y'] - 1)
                normal_class = center.T @ residual / len(matrix) + stats.ALPHA * saved_coef_class
                audit.normal_residual(np.r_[normal_class, residual.mean()], f'{context} {kind}')
                audit.class_fits += 1

    audit.metrics(record, dict(fit_pairs=len(low), fit_matched_quartets=len(np.unique(groups)), pair_fit_status=statuses), context)
    return coefficients, class_fits


def native_metrics(counts, labels):
    strict, forced = decisions(counts)
    tied = strict == -1
    quiet = np.all(counts == 0, axis=1)
    saturated = np.all(counts == 8, axis=1)
    return dict(native_strict_ba=stats.balanced_accuracy(strict, labels),
                native_forced_ba=stats.balanced_accuracy(forced, labels),
                native_coverage=float(np.mean(~tied)), native_silence=float(np.mean(quiet)),
                native_saturated_tie=float(np.mean(saturated)),
                native_intermediate_tie=float(np.mean(tied & ~quiet & ~saturated)))


def verify_test(audit, archive, record, base, native, maps, fits, prior, context):
    keys = {'pair_scores', 'count_scores', 'raster_scores', 'prior_raster_scores', *PAIR_FIELDS}
    audit.check(set(archive) == keys, f'{context} test artifact fields')
    low, high, groups, _ = verify_pairs(audit, archive, base, native['counts'], context)
    audit.check(archive['pair_scores'].shape == (17, len(low)), f'{context} test pair score shape')
    pair_is_4_4 = np.all(native['counts'][low] == 4, axis=1)
    coefficients, class_fits = fits
    ranks, ranks_4_4 = [], []
    expected = native_metrics(native['counts'], base['y'])
    for k, features in enumerate(features_for_controls(audit, native['raster'], maps, context)):
        differences = features[high, 2:] - features[low, 2:]
        scores = differences @ coefficients[k]
        audit.near(archive['pair_scores'][k], scores, f'{context} condition {k} test differences')
        saved_sign = np.where(np.abs(archive['pair_scores'][k]) <= stats.SCORE_TOLERANCE, 0, np.sign(archive['pair_scores'][k]))
        independent_sign = np.where(np.abs(scores) <= stats.SCORE_TOLERANCE, 0, np.sign(scores))
        audit.exact(saved_sign, independent_sign, f'{context} condition {k} exact ranking decisions')
        ranks.append(stats.rank_rate(scores, groups))
        ranks_4_4.append(stats.rank_rate(scores[pair_is_4_4], groups[pair_is_4_4]))

        if k == 0:
            for kind, matrix in (('count', features[:, :2]), ('raster', features)):
                mean, coef, intercept = class_fits[kind]
                predicted = (matrix - mean) @ coef + intercept
                audit.near(archive[f'{kind}_scores'], predicted, f'{context} {kind} independent test scores')
                ba, mse = stats.classification_metrics(predicted, base['y'])
                expected[f'{kind}_ba'], expected[f'{kind}_mse'] = ba, mse
                if kind == 'count':
                    audit.exact(predicted[low], predicted[high], f'{context} count-only pair scores tie exactly')

    fixed = np.all(native['counts'] == 4, axis=1)
    labels_fixed = base['y'][fixed]
    prior_scores = stats.prior_raster_scores(native['raster'], prior)
    audit.near(archive['prior_raster_scores'], prior_scores, f'{context} prior locked raster scores')
    rank_shuffle_mean = None if not len(low) else float(np.mean(ranks[1:]))
    n_groups = len(np.unique(groups))
    expected.update(
        test_pairs=len(low), test_matched_quartets=n_groups, test_matched_fraction=n_groups / len(np.unique(base['quartet'])),
        pair_equal_raster_fraction=float(np.mean(np.all(native['raster'][low] == native['raster'][high], axis=(1, 2)))) if len(low) else None,
        matched_4_4_pairs=int(pair_is_4_4.sum()), matched_4_4_quartets=len(np.unique(groups[pair_is_4_4])),
        pair_rank=ranks[0], pair_rank_shuffled=ranks[1:], pair_rank_shuffle_mean=rank_shuffle_mean,
        pair_rank_delta=None if not len(low) else ranks[0] - rank_shuffle_mean,
        count_pair_rank=.5 if len(low) else None, pair_rank_4_4=ranks_4_4[0],
        pair_rank_shuffled_4_4_mean=float(np.mean(ranks_4_4[1:])) if np.any(pair_is_4_4) else None,
        fixed_4_4_n=int(fixed.sum()), fixed_4_4_label_counts=np.bincount(labels_fixed, minlength=2).tolist(),
        prior_4_4_raster_ba=stats.balanced_accuracy(prior_scores[fixed] > stats.SCORE_TOLERANCE, labels_fixed),
        mean_resources=native['resources'].mean(axis=0),
    )
    audit.metrics(record, expected, context)
    audit.matched_test_models += bool(len(low))


def indexed_rows(audit, split, expected):
    rows = read(ROOT / split / 'ROWS.json')
    indexed = {row['id']: row for row in rows}
    audit.check(len(rows) == len(indexed) and set(indexed) == expected, f'complete unique {split} records')
    second_kind = 'fit' if split == 'fit' else 'scores'
    audit.check({path.stem for path in (ROOT / split).glob('*.npz')} ==
                {f'{name}_{kind}' for name in expected for kind in ('native', second_kind)}, f'complete {split} archives')
    return indexed


def verify_results(audit, models, theta):
    names = {model['id'] for model in models}
    fit_rows = indexed_rows(audit, 'fit', names)
    test_rows = indexed_rows(audit, 'test', names)
    nuisance_keys = set()
    for block in range(3):
        audit.stage = f'data block {block}'
        fit_base, fit_x, fit_maps = read_data(audit, 'fit', block, nuisance_keys)
        test_base, test_x, test_maps = read_data(audit, 'test', block, nuisance_keys)
        for model in (row for row in models if row['seed'] == block):
            ident = model['id']
            audit.stage = ident
            fit_row, test_row = fit_rows[ident], test_rows[ident]
            audit.check(all(model[k] == fit_row[k] == test_row[k] for k in META), f'{ident} row metadata')
            audit.check(all(fit_row[k] == test_row[k] for k in ('fit_pairs', 'fit_matched_quartets', 'pair_fit_status')),
                        f'{ident} archived calibration summaries')
            weights = theta[model['index']]
            native_fit = verify_native(audit, 'fit', model, fit_x, weights)
            native_test = verify_native(audit, 'test', model, test_x, weights)
            archive_fit = load(ROOT / f'fit/{ident}_fit.npz')
            archive_test = load(ROOT / f'test/{ident}_scores.npz')
            fits = verify_fit(audit, archive_fit, fit_row, fit_base, native_fit, fit_maps, f'{ident} fit')
            prior = load(ROOT / f'prior_raster/{ident}_raster.npz')
            verify_test(audit, archive_test, test_row, test_base, native_test, test_maps, fits, prior, f'{ident} test')
        print(f'Independent timing-increment audit block {block + 1}/3 complete', flush=True)


def verify_summary(audit):
    summary = read(ROOT / 'SCIENCE_SUMMARY.json')
    environment = read(ROOT / 'ENVIRONMENT.json')
    audit.check(read(ROOT / 'RUN_STATE.json')['status'] == 'complete' and summary['status'] == 'COMPLETE' and
                not (ROOT / 'SCIENCE_FAILURE.json').exists(), 'single completed scientific batch')
    audit.check((audit.baselines, audit.rank_fits, audit.class_fits, audit.steps, audit.shuffles, audit.scalar_models)
                == (72, 612, 72, 1327104, 1152, 18), 'audited planned scope')
    for key, value in dict(**EXPECTED, cap_seconds=840).items():
        audit.check(summary[key] == value, f'science summary {key}')
    audit.check(0 < summary['wall_seconds'] <= 840 and summary['peak_working_set_bytes'] <= 1024**3,
                'science wall-clock and memory budgets')
    audit.check(0 < summary['artifact_bytes_at_science_end'] <= int(1.5 * 1024**3), 'science disk budget')
    audit.check(all(environment['threads'][key] == os.environ.get(key) == '1' for key in THREADS),
                'single-thread limits')


def main():
    if (ROOT / 'AUDIT.json').exists():
        raise SystemExit('audit already recorded')
    audit = Audit()
    try:
        models, theta = source_locks(audit)
        verify_results(audit, models, theta)
        verify_summary(audit)
        result = dict(
            status='PASS', checks=audit.checks, **EXPECTED, shuffle_invariance_checks=audit.shuffles,
            matched_test_models=audit.matched_test_models, source_scalar_attention_models=audit.scalar_models,
            source_scalar_episodes=2 * audit.scalar_models, max_numeric_error=audit.max_numeric_error,
            max_coefficient_error=audit.max_coefficient_error, max_normal_equation_residual=audit.max_normal_residual,
            wall_seconds=time.perf_counter() - audit.start, cpu_seconds=time.process_time() - audit.cpu, cap_seconds=300,
            scope=('Independent fresh data and time orders, 72 exact native pulse and resource replays, '
                   'complete within-quartet count matching, 612 weighted and 72 intercept-augmented SVD fits, '
                   'all test scores and metrics including prior locked [4,4] probe, count/synchrony/native invariance, '
                   'frozen source and pre-test fit locks, 36 original scalar TAN episode checks.'),
        )
        save(ROOT / 'AUDIT.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as error:
        path = ROOT / 'AUDIT_FAILURE.json'
        if path.exists():
            path = ROOT / f'AUDIT_FAILURE_{time.time_ns()}.json'
        save(path, dict(status='FAIL', stage=audit.stage, checks=audit.checks, exception=type(error).__name__,
                        message=str(error), wall_seconds=time.perf_counter() - audit.start))
        raise


if __name__ == '__main__':
    main()
