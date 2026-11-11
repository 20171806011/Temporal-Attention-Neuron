"""Independently replay the frozen network and audit the streaming readouts."""

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
PRIOR = ROOT.parent / 'DEND-TAN-timing-increment-20261109'
ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2')
LOSSES = ('CE', 'COUNT', 'CECOM')
META = ('id', 'seed', 'arm', 'loss')
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
RAMP = np.arange(-7, 8, 2, dtype=np.int32)
COUNT_SCALE = np.array([4.0, 4.0])
MOMENT_SCALE = np.array([4.0, np.sqrt(336.0)])
PAIR_FIELDS = ('low', 'high', 'quartet', 'weight')
SOURCE_FILES = {'model.py', 'population.py', 'reference.py', 'original_tan.py', 'task_data.py',
                'independent_data.py', 'continuation.py', 'independent_capture.py', 'timing_stats.py',
                'independent_stats.py', 'MODELS.json', 'models.npz'}
EXPECTED = dict(models=36, neural_training_updates=0, baseline_captures=72, probe_fits=396,
                forward_sample_steps=1327104, dense_forward_macs=207028224, benchmark_batches=6912)


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
        self.checks = self.baselines = self.fits = self.steps = self.benchmarks = 0
        self.shuffles = self.scalar_models = self.matched_models = 0
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
            elif isinstance(value, (float, np.floating, np.ndarray)) or (
                isinstance(value, (list, tuple)) and value and isinstance(value[0], (float, np.floating))
            ):
                self.near(got, value, f'{context} {key}', atol=1e-12, rtol=1e-10)
            else:
                self.exact(got, value, f'{context} {key}')


def archive_paths(models, split):
    kinds = ('native', 'fit' if split == 'fit' else 'scores')
    return ({f"{split}/{row['id']}_{kind}.npz" for row in models for kind in kinds}
            | {f'data/{split}{tag}_s{block}.npz' for tag in ('', '_perm') for block in range(3)}
            | {f'{split}/ROWS.json'})


def resource_contract(audit):
    contract = read(ROOT / 'RESOURCE_CONTRACT.json')
    ramp_bound = int(np.abs(RAMP).sum())
    int_bytes, float_bytes = np.dtype(np.int32).itemsize, np.dtype(np.float64).itemsize
    audit.metrics(contract, dict(query_steps=8, extra_simulated_steps=0, benchmark_batch_episodes=1024,
                  benchmark_warmups_per_method=3, benchmark_repetitions_per_method=64,
                  benchmark_method_order=['native', 'count2', 'moment2'], moment_weights=RAMP,
                  moment_integer_range=[-ramp_bound, ramp_bound], integer_counter_bytes=int_bytes,
                  learned_coefficient_bytes=float_bytes), 'resource contract')
    method_operations = dict(
        native=(2, 0, 2, 0, 0, 0, 0, 0, 0),
        count2=(2, 3, 2, 0, 0, 1, 1, 2, 2),
        moment2=(3, 3, 3, 1, 1, 0, 1, 2, 2),
    )
    audit.check(set(contract['methods']) == set(method_operations), 'resource contract methods')
    for name, values in method_operations.items():
        counters, learned, step_add, step_sub, step_mul, final_add, final_sub, score_mul, score_add = values
        audit.metrics(contract['methods'][name], dict(
            persistent_counter_bytes=counters * int_bytes, learned_scalars=learned,
            learned_bytes=learned * float_bytes, raster_buffer_bytes=0, step_integer_adds=step_add,
            step_integer_subtracts=step_sub, step_integer_multiplies=step_mul,
            terminal_integer_adds=final_add, terminal_integer_subtracts=final_sub,
            terminal_float_multiplies=score_mul, terminal_float_adds=score_add, terminal_comparisons=1,
        ), f'{name} arithmetic and storage contract')
    audit.metrics(contract['offline16'], dict(features=16, learned_scalars=17, learned_bytes=17 * float_bytes,
                  archived_uint8_raster_bytes=16 * np.dtype(np.uint8).itemsize), 'offline storage contract')

    # Coefficients on the 16 input pulse positions, expressed in raw raster units.
    sign = np.array([-1.0, 1.0])
    directions = np.stack((np.tile(sign, 8) / 4.0, np.ones(16) / 4.0,
                           (RAMP[:, None] * sign).ravel() / np.sqrt(336.0)))
    audit.near(directions @ directions.T, np.eye(3), 'unit orthogonal count and timing directions')


def source_locks(audit):
    freeze, source, fit = (read(ROOT / name) for name in ('FREEZE.json', 'SOURCE_LOCK.json', 'FIT_LOCK.json'))
    audit.check(freeze['scientific_data_generated'] is False and freeze['neural_weights_fixed'] is True,
                'pre-science freeze and fixed neural weights')
    audit.check(fit['test_generated'] is False, 'no test before fit lock assertion')
    manifest_path = PRIOR / 'MANIFEST.json'
    manifest = read(manifest_path)['files']
    audit.check(source['previous_directory'] == f'../{PRIOR.name}' and
                source['previous_manifest_sha256'] == sha(manifest_path), 'parent manifest identity')
    audit.check(read(PRIOR / 'AUDIT.json')['status'] == 'PASS' and
                sha(PRIOR / 'AUDIT.json') == manifest['AUDIT.json']['sha256'], 'audited parent batch')
    audit.check(set(source['files']) == SOURCE_FILES, 'complete original sources and archived weights')
    for name, digest in source['files'].items():
        entry = manifest[name]
        audit.check(digest == entry['sha256'] == sha(ROOT / name) == sha(PRIOR / name), f'{name} immutable bytes')
        audit.check((ROOT / name).stat().st_size == entry['bytes'], f'{name} archived byte length')

    required = SOURCE_FILES | {'audit.py', 'run.py', 'readout.py', 'core.py', 'PROTOCOL.md',
                               'SOURCE_LOCK.json', 'RESOURCE_CONTRACT.json', 'PRECHECK.json', 'TIMING.json'}
    audit.check(required <= set(freeze['files']), 'freeze covers source, protocol and resource contract')
    for name, digest in freeze['files'].items():
        relative = Path(name)
        audit.check(not relative.is_absolute() and '..' not in relative.parts, f'local freeze path {name}')
        audit.check(sha(ROOT / relative) == digest, f'frozen file {name}')
    for name in ('PRECHECK.json', 'TIMING.json'):
        audit.check(read(ROOT / name)['status'] == 'PASS', f'accepted frozen {name}')

    models = read(ROOT / 'MODELS.json')['rows']
    audit.check(len(models) == 36 and {(r['seed'], r['arm'], r['loss']) for r in models} ==
                set(product(range(3), ARMS, LOSSES)), '36 archived models and three complete blocks')
    for index, model in enumerate(models):
        audit.check(model['index'] == index and model['id'] == f"s{model['seed']}_{model['arm']}_{model['loss']}",
                    f'archive model row {index}')
    theta = load(ROOT / 'models.npz')
    audit.check(set(theta) == {'theta'} and theta['theta'].shape == (36, 176) and
                np.isfinite(theta['theta']).all(), 'finite archived weights')

    fit_files = archive_paths(models, 'fit')
    test_files = archive_paths(models, 'test')
    audit.check(set(fit['files']) == fit_files, 'calibration lock covers all 79 artifacts')
    freeze_time = (ROOT / 'FREEZE.json').stat().st_mtime_ns
    fit_time = (ROOT / 'FIT_LOCK.json').stat().st_mtime_ns
    audit.check(datetime.fromisoformat(freeze['utc']) < datetime.fromisoformat(fit['utc']), 'UTC lock chronology')
    for name, digest in fit['files'].items():
        path = ROOT / name
        audit.check(freeze_time <= path.stat().st_mtime_ns <= fit_time and sha(path) == digest,
                    f'calibration artifact unchanged and generated before test {name}')
    actual_test = {p.relative_to(ROOT).as_posix() for p in (ROOT / 'test').iterdir()}
    actual_test.update(p.relative_to(ROOT).as_posix() for p in (ROOT / 'data').glob('test*.npz'))
    audit.check(actual_test == test_files and all((ROOT / name).stat().st_mtime_ns >= fit_time for name in test_files),
                'all 79 test artifacts were generated after fit lock')
    return models, theta['theta']


def read_data(audit, split, block, nuisance_keys):
    if split == 'fit':
        n_rows, data_seed, permutation_seed = 512, 4031010000, 4031030000
    else:
        n_rows, data_seed, permutation_seed = 1024, 4031020000, 4031040000
    name = f'{split} s{block}'
    actual = load(ROOT / f'data/{split}_s{block}.npz')
    expected = data.make_base(n_rows, data_seed + 100 * block)
    audit.check(set(actual) == set(expected), f'{name} data fields')
    for key, value in expected.items():
        audit.exact(actual[key], value, f'{name} independently generated {key}')
    x, q = data.assemble(actual, 'nominal')
    audit.check(q == 16, f'{name} query starts at t16')
    data.check(actual, x, 'nominal')
    keys = set(zip(actual['amplitude'], actual['tag_amplitude']))
    audit.check(len(keys) == n_rows // 4 and not keys.intersection(nuisance_keys), 'distinct quartets across splits')
    nuisance_keys.update(keys)

    archive = load(ROOT / f'data/{split}_perm_s{block}.npz')
    rng = np.random.default_rng(permutation_seed + 100 * block)
    expected_maps = np.array([[rng.permutation(8) for _ in range(n_rows)] for _ in range(8)], dtype=np.uint8)
    audit.check(set(archive) == {'permutations'} and archive['permutations'].dtype == np.uint8,
                f'{name} uint8 permutation archive')
    audit.exact(archive['permutations'], expected_maps, f'{name} independently regenerated time orders')
    return actual, x, expected_maps


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
    return expected


def decisions(counts):
    tied = counts[:, 0] == counts[:, 1]
    forced = (counts[:, 1] > counts[:, 0]).astype(int)
    return np.where(tied, -1, forced), forced


def raw_features(raster):
    # Independent closed form over the raster, rather than the candidate's step loop.
    pulse = raster.astype(np.int32)
    counts = pulse.sum(axis=1)
    difference = counts[:, 1] - counts[:, 0]
    total = counts[:, 0] + counts[:, 1]
    moment = (pulse[:, :, 1] - pulse[:, :, 0]) @ RAMP
    return np.column_stack((difference, total)), np.column_stack((difference, moment))


def shuffled_controls(audit, raster, maps, context):
    counts = raster.sum(axis=1)
    codes = np.sort(2 * raster[:, :, 0] + raster[:, :, 1], axis=1)
    for index, order in enumerate(maps):
        shuffled = np.take_along_axis(raster, order[:, :, None], axis=1)
        shuffled_counts = shuffled.sum(axis=1)
        audit.exact(shuffled_counts, counts, f'{context} shuffle {index} exact per-channel counts')
        audit.exact(np.sort(2 * shuffled[:, :, 0] + shuffled[:, :, 1], axis=1), codes,
                    f'{context} shuffle {index} joint time-bin multiset including synchrony')
        audit.exact(decisions(shuffled_counts), decisions(counts), f'{context} shuffle {index} native decisions')
        audit.shuffles += 1
        yield shuffled


def verify_classifier(audit, archive, kind, features, labels, context, *, index=None, scale=None):
    def saved(field):
        values = archive[f'{kind}_{field}']
        return values if index is None else values[index]

    mean, coef, intercept = stats.class_fit(features, labels)
    audit.near(saved('mean'), mean, f'{context} training mean')
    audit.coefficient(saved('coef'), coef, f'{context} augmented SVD coefficients')
    audit.near(saved('intercept'), intercept, f'{context} unpenalized centered intercept')
    centered = features - saved('mean')
    residual = centered @ saved('coef') + saved('intercept') - (2.0 * labels - 1.0)
    normal = centered.T @ residual / len(features) + stats.ALPHA * saved('coef')
    audit.normal_residual(np.r_[normal, residual.mean()], context)
    if scale is not None:
        audit.coefficient(saved('raw'), np.r_[coef / scale, intercept - mean @ coef], f'{context} folded raw coefficients')
    audit.fits += 1
    return mean, coef, intercept


def verify_fit(audit, archive, base, native, maps, context):
    shapes = dict(count_mean=(2,), count_coef=(2,), count_intercept=(), count_raw=(3,),
                  moment_mean=(9, 2), moment_coef=(9, 2), moment_intercept=(9,), moment_raw=(9, 3),
                  raster_mean=(16,), raster_coef=(16,), raster_intercept=())
    audit.check(set(archive) == set(shapes), f'{context} all fit fields')
    for key, shape in shapes.items():
        audit.check(archive[key].shape == shape and archive[key].dtype == np.float64,
                    f'{context} {key} shape and float64')
    raster = native['raster']
    count, moment = raw_features(raster)
    labels = base['y']
    audit.check(np.max(np.abs(moment[:, 1])) <= 32, f'{context} signed moment bound')
    count_fit = verify_classifier(audit, archive, 'count', count / COUNT_SCALE, labels,
                                  f'{context} COUNT2', scale=COUNT_SCALE)
    raster_fit = verify_classifier(audit, archive, 'raster', stats.helmert_features(raster), labels,
                                   f'{context} offline16')
    moment_fits = [verify_classifier(audit, archive, 'moment', moment / MOMENT_SCALE, labels,
                                     f'{context} MOMENT2', index=0, scale=MOMENT_SCALE)]
    for index, shuffled in enumerate(shuffled_controls(audit, raster, maps, context), 1):
        _, moment = raw_features(shuffled)
        moment_fits.append(verify_classifier(audit, archive, 'moment', moment / MOMENT_SCALE, labels,
                                             f'{context} MOMENT2 shuffle {index}', index=index, scale=MOMENT_SCALE))
    return count_fit, moment_fits, raster_fit


def native_metrics(counts, labels):
    strict, forced = decisions(counts)
    tied = strict == -1
    silent = np.all(counts == 0, axis=1)
    saturated = np.all(counts == 8, axis=1)
    return dict(native_strict_ba=stats.balanced_accuracy(strict, labels),
                native_forced_ba=stats.balanced_accuracy(forced, labels), native_coverage=float(np.mean(~tied)),
                native_silence=float(np.mean(silent)), native_saturated_tie=float(np.mean(saturated)),
                native_intermediate_tie=float(np.mean(tied & ~silent & ~saturated)))


def verify_scores(audit, archived, expected, context):
    audit.near(archived, expected, context)
    audit.exact(archived > stats.SCORE_TOLERANCE, expected > stats.SCORE_TOLERANCE, f'{context} exact decisions')


def stream_scores(audit, archived_scores, raw, scale, fitted, raw_coef, context):
    mean, coef, intercept = fitted
    independent = (raw / scale - mean) @ coef + intercept
    folded = raw @ raw_coef[:2] + raw_coef[2]
    verify_scores(audit, archived_scores, independent, context + ' independent SVD scores')
    verify_scores(audit, archived_scores, folded, context + ' folded raw affine scores')
    return independent


def verify_test(audit, archive, record, base, native, maps, fit_archive, fits, context):
    audit.check(set(archive) == {'count_scores', 'moment_scores', 'raster_scores', 'timing_ns', *PAIR_FIELDS},
                f'{context} all test artifact fields')
    raster, counts, labels = native['raster'], native['counts'], base['y']
    audit.check(archive['moment_scores'].shape == (9, len(labels)), f'{context} nine moment conditions')
    count_fit, moment_fits, raster_fit = fits
    count, moment = raw_features(raster)
    audit.check(np.max(np.abs(moment[:, 1])) <= 32, f'{context} signed moment bound')
    count_scores = stream_scores(audit, archive['count_scores'], count, COUNT_SCALE, count_fit,
                                 fit_archive['count_raw'], f'{context} COUNT2')
    moment_scores = [stream_scores(audit, archive['moment_scores'][0], moment, MOMENT_SCALE, moment_fits[0],
                                   fit_archive['moment_raw'][0], f'{context} MOMENT2')]
    for index, shuffled in enumerate(shuffled_controls(audit, raster, maps, context), 1):
        _, moment = raw_features(shuffled)
        moment_scores.append(stream_scores(audit, archive['moment_scores'][index], moment, MOMENT_SCALE,
                                           moment_fits[index], fit_archive['moment_raw'][index],
                                           f'{context} MOMENT2 shuffle {index}'))
    mean, coef, intercept = raster_fit
    raster_scores = (stats.helmert_features(raster) - mean) @ coef + intercept
    verify_scores(audit, archive['raster_scores'], raster_scores, f'{context} offline16 scores')

    expected = native_metrics(counts, labels)
    expected['count2_ba'], expected['count2_mse'] = stats.classification_metrics(count_scores, labels)
    expected['moment2_ba'], expected['moment2_mse'] = stats.classification_metrics(moment_scores[0], labels)
    expected['offline16_ba'], expected['offline16_mse'] = stats.classification_metrics(raster_scores, labels)
    shuffled_metrics = [stats.classification_metrics(scores, labels) for scores in moment_scores[1:]]
    expected['shuffled_ba'], expected['shuffled_mse'] = map(list, zip(*shuffled_metrics))
    expected['shuffle_ba_mean'] = float(np.mean(expected['shuffled_ba']))
    for comparison, baseline in (('count', 'count2_ba'), ('shuffle', 'shuffle_ba_mean'), ('native', 'native_forced_ba')):
        expected[f'moment_minus_{comparison}'] = expected['moment2_ba'] - expected[baseline]

    matched = stats.paired_indices(counts, labels, base['quartet'])
    for key, value in zip(PAIR_FIELDS, matched):
        compare = audit.near if key == 'weight' else audit.exact
        compare(archive[key], value, f'{context} matched {key}')
    low, high, quartets, _ = matched
    audit.exact(counts[low], counts[high], f'{context} exact count matches')
    audit.exact(count_scores[low], count_scores[high], f'{context} COUNT2 scores tie within matched pairs')
    ranks = [stats.rank_rate(score[high] - score[low], quartets) for score in moment_scores]
    fixed = np.all(counts == 4, axis=1)
    expected.update(matched_pairs=len(low), matched_quartets=len(np.unique(quartets)),
                    matched_count_rank=.5 if len(low) else None, matched_moment_rank=ranks[0],
                    matched_shuffle_rank_mean=float(np.mean(ranks[1:])) if len(low) else None,
                    matched_offline_rank=stats.rank_rate(raster_scores[high] - raster_scores[low], quartets),
                    fixed_4_4_n=int(fixed.sum()), fixed_4_4_label_counts=np.bincount(labels[fixed], minlength=2).tolist(),
                    fixed_4_4_moment_ba=stats.balanced_accuracy(moment_scores[0][fixed] > stats.SCORE_TOLERANCE, labels[fixed]),
                    fixed_4_4_offline_ba=stats.balanced_accuracy(raster_scores[fixed] > stats.SCORE_TOLERANCE, labels[fixed]),
                    mean_resources=native['resources'].mean(axis=0))
    timing = archive['timing_ns']
    audit.check(timing.shape == (3, 64) and timing.dtype == np.int64 and np.all(timing > 0),
                f'{context} complete positive benchmark observations')
    expected['bench_median_ns'] = np.median(timing, axis=1)
    audit.benchmarks += timing.size
    audit.matched_models += bool(len(low))
    audit.metrics(record, expected, context)


def verify_results(audit, models, theta):
    names = {model['id'] for model in models}
    indexed = {}
    for split in ('fit', 'test'):
        rows = read(ROOT / split / 'ROWS.json')
        indexed[split] = {row['id']: row for row in rows}
        audit.check(len(rows) == len(indexed[split]) == 36 and set(indexed[split]) == names,
                    f'complete unique {split} records')
    nuisance_keys = set()
    for block in range(3):
        audit.stage = f'data block {block}'
        fit_base, fit_x, fit_maps = read_data(audit, 'fit', block, nuisance_keys)
        test_base, test_x, test_maps = read_data(audit, 'test', block, nuisance_keys)
        for model in (row for row in models if row['seed'] == block):
            ident = model['id']
            audit.stage = ident
            fit_row, test_row = (indexed[split][ident] for split in ('fit', 'test'))
            audit.check(all(model[k] == fit_row[k] == test_row[k] for k in META), f'{ident} model metadata')
            for name in ('fit_seconds', 'fit_collect_seconds'):
                audit.check(0 < fit_row[name] <= 840 and fit_row[name] == test_row[name], f'{ident} {name} provenance')
            audit.check(0 < test_row['test_collect_seconds'] <= 840, f'{ident} test collection time')
            weights = theta[model['index']]
            fit_native = verify_native(audit, 'fit', model, fit_x, weights)
            test_native = verify_native(audit, 'test', model, test_x, weights)
            fit_archive = load(ROOT / f'fit/{ident}_fit.npz')
            test_archive = load(ROOT / f'test/{ident}_scores.npz')
            fits = verify_fit(audit, fit_archive, fit_base, fit_native, fit_maps, f'{ident} fit')
            verify_test(audit, test_archive, test_row, test_base, test_native, test_maps, fit_archive, fits, f'{ident} test')
        print(f'Independent streaming-readout audit block {block + 1}/3 complete', flush=True)


def verify_summary(audit):
    summary = read(ROOT / 'SCIENCE_SUMMARY.json')
    environment = read(ROOT / 'ENVIRONMENT.json')
    audit.check(read(ROOT / 'RUN_STATE.json')['status'] == 'complete' and summary['status'] == 'COMPLETE' and
                not (ROOT / 'SCIENCE_FAILURE.json').exists(), 'single completed scientific batch')
    audit.check((audit.baselines, audit.fits, audit.steps, audit.benchmarks, audit.shuffles, audit.scalar_models)
                == (72, 396, 1327104, 6912, 576, 18), 'audited planned scope')
    for key, value in dict(**EXPECTED, cap_seconds=840).items():
        audit.check(summary[key] == value, f'science summary {key}')
    rows = read(ROOT / 'test/ROWS.json')
    fit_time = sum(row['fit_seconds'] for row in rows)
    native_time = sum(row['fit_collect_seconds'] + row['test_collect_seconds'] for row in rows)
    audit.near(summary['offline_fit_seconds'], fit_time, 'offline fit timing sum')
    audit.near(summary['native_collect_seconds'], native_time, 'native collection timing sum')
    audit.check(0 < fit_time + native_time <= summary['wall_seconds'], 'non-overlapping science timing phases')
    audit.check(0 < summary['wall_seconds'] <= 840 and 0 < summary['peak_working_set_bytes'] <= 1024**3,
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
        resource_contract(audit)
        verify_results(audit, models, theta)
        verify_summary(audit)
        result = dict(
            status='PASS', checks=audit.checks, **EXPECTED, shuffle_invariance_checks=audit.shuffles,
            matched_test_models=audit.matched_models, source_scalar_attention_models=audit.scalar_models,
            source_scalar_episodes=2 * audit.scalar_models, max_numeric_error=audit.max_numeric_error,
            max_coefficient_error=audit.max_coefficient_error, max_normal_equation_residual=audit.max_normal_residual,
            wall_seconds=time.perf_counter() - audit.start, cpu_seconds=time.process_time() - audit.cpu, cap_seconds=300,
            scope=('Independent data and time orders, exact native query raster/counts/resources, augmented SVD fits, '
                   'direct and folded affine scores, count/synchrony/native invariance, equal-count ranks, '
                   'fixed-count BA, positive benchmark samples and medians, resource and timing ledgers, pre-test fit and source locks, '
                   '36 original scalar TAN episode checks.'),
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
