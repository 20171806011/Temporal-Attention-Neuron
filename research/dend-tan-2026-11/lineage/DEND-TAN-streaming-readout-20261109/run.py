"""Calibrate fixed streaming rules, then test them on new native pulse streams."""

import os
import platform
import sys
import time

import numpy as np

import continuation
import readout
import task_data
import timing_stats as stats
from core import Budget, R, dump, read, sha, utc, verify_lock


def load(path):
    with np.load(path) as archive:
        return dict(archive)


def make_permutations(n, seed):
    rng = np.random.default_rng(seed)
    maps = np.empty((8, n, 8), dtype=np.uint8)
    for r in range(8):
        for i in range(n):
            maps[r, i] = rng.permutation(8)
    return maps


def fit_model(raster, base, permutations):
    y = base['y']
    count_mean, count_coef, count_intercept, count_raw = readout.fit_rule(raster, y, 'count2')
    moment = [readout.fit_rule(raster, y, 'moment2')]
    for mapping in permutations:
        moment.append(readout.fit_rule(stats.shuffle(raster, mapping), y, 'moment2'))
    moment_mean, moment_coef, moment_intercept, moment_raw = [np.stack(column) for column in zip(*moment)]
    raster_mean, raster_coef, raster_intercept = stats.fit_class(stats.features(raster), y)
    return dict(count_mean=count_mean, count_coef=count_coef, count_intercept=np.array(count_intercept),
                count_raw=count_raw, moment_mean=moment_mean, moment_coef=moment_coef,
                moment_intercept=moment_intercept, moment_raw=moment_raw,
                raster_mean=raster_mean, raster_coef=raster_coef, raster_intercept=np.array(raster_intercept))


def evaluate_model(native, base, permutations, fit):
    raster, counts, y = native['raster'], native['counts'], base['y']
    assert np.array_equal(readout.predict(raster, 'native'), counts[:, 1] > counts[:, 0])
    count_scores = readout.stream_scores(raster, 'count2', fit['count_raw'])
    moment_scores = [readout.stream_scores(raster, 'moment2', fit['moment_raw'][0])]
    for index, mapping in enumerate(permutations, 1):
        shuffled = stats.shuffle(raster, mapping)
        assert np.array_equal(shuffled.sum(1), counts)
        moment_scores.append(readout.stream_scores(shuffled, 'moment2', fit['moment_raw'][index]))
    moment_scores = np.stack(moment_scores)
    raster_scores = stats.class_scores(stats.features(raster), fit['raster_mean'], fit['raster_coef'], fit['raster_intercept'])
    timing_ns = readout.benchmark(raster, fit['count_raw'], fit['moment_raw'][0])

    def ba(scores):
        return stats.balanced_accuracy(stats.classify(scores), y)

    def mse(scores):
        return float(np.mean((scores-(2*y-1))**2))

    matched = stats.pairs(counts, base)
    lo, hi, weight = (matched[key] for key in ('low', 'high', 'weight'))
    assert np.array_equal(count_scores[lo], count_scores[hi])

    def rank(scores):
        return stats.rank_rate(scores[hi]-scores[lo], weight)

    fixed = np.all(counts == 4, axis=1)
    label_counts = np.bincount(y[fixed], minlength=2).tolist()

    def fixed_ba(scores):
        return stats.balanced_accuracy(stats.classify(scores[fixed]), y[fixed]) if min(label_counts) > 0 else None

    native_values = stats.native_metrics(counts, y)
    moment_ba = ba(moment_scores[0])
    count_ba = ba(count_scores)
    shuffled_ba = [ba(scores) for scores in moment_scores[1:]]
    shuffle_mean = float(np.mean(shuffled_ba))
    row = dict(**native_values,
               count2_ba=count_ba, count2_mse=mse(count_scores), moment2_ba=moment_ba, moment2_mse=mse(moment_scores[0]),
               shuffled_ba=shuffled_ba, shuffled_mse=[mse(scores) for scores in moment_scores[1:]],
               shuffle_ba_mean=shuffle_mean, offline16_ba=ba(raster_scores), offline16_mse=mse(raster_scores),
               moment_minus_count=moment_ba-count_ba, moment_minus_shuffle=moment_ba-shuffle_mean,
               moment_minus_native=moment_ba-native_values['native_forced_ba'],
               matched_pairs=len(lo), matched_quartets=len(np.unique(matched['quartet'])),
               matched_count_rank=.5 if len(lo) else None, matched_moment_rank=rank(moment_scores[0]),
               matched_shuffle_rank_mean=float(np.mean([rank(scores) for scores in moment_scores[1:]])) if len(lo) else None,
               matched_offline_rank=rank(raster_scores), fixed_4_4_n=int(fixed.sum()), fixed_4_4_label_counts=label_counts,
               fixed_4_4_moment_ba=fixed_ba(moment_scores[0]), fixed_4_4_offline_ba=fixed_ba(raster_scores),
               bench_median_ns=np.median(timing_ns, axis=1).tolist(), mean_resources=native['resources'].mean(0).tolist())
    return dict(**matched, count_scores=count_scores, moment_scores=moment_scores,
                raster_scores=raster_scores, timing_ns=timing_ns), row


def inputs(split, seed):
    n, offset, shuffle_offset = (512, 4031010000, 4031030000) if split == 'fit' else (1024, 4031020000, 4031040000)
    base = task_data.make_base(n, offset+100*seed)
    permutations = make_permutations(n, shuffle_offset+100*seed)
    np.savez_compressed(R / 'data' / f'{split}_s{seed}.npz', **base)
    np.savez_compressed(R / 'data' / f'{split}_perm_s{seed}.npz', permutations=permutations)
    x, _ = task_data.assemble(base)
    return base, permutations, x


def collect_timed(budget, x, w, arm):
    start = time.perf_counter()
    native = continuation.collect(x, w, arm, taps=())
    elapsed = time.perf_counter()-start
    budget.record_baseline(len(x))
    return native, elapsed


def calibrate(budget, models, theta):
    budget.stage = 'fit'
    budget.mark('running')
    (R / 'data').mkdir()
    folder = R / 'fit'
    folder.mkdir()
    rows = []
    for seed in range(3):
        base, permutations, x = inputs('fit', seed)
        for model in [row for row in models if row['seed'] == seed]:
            budget.guard(disk=True)
            ident = model['id']
            native, collect_seconds = collect_timed(budget, x, theta[model['index']], model['arm'])
            np.savez_compressed(folder / f'{ident}_native.npz', **native)
            start = time.perf_counter()
            fit = fit_model(native['raster'], base, permutations)
            fit_seconds = time.perf_counter()-start
            np.savez_compressed(folder / f'{ident}_fit.npz', **fit)
            budget.probe_fits += 11
            rows.append(dict(**{key: model[key] for key in ('id', 'seed', 'arm', 'loss')},
                             fit_seconds=fit_seconds, fit_collect_seconds=collect_seconds))
            dump(folder / 'ROWS.json', rows)
    files = sorted([*(R / 'data').glob('fit*.npz'), *folder.iterdir()])
    dump(R / 'FIT_LOCK.json', dict(utc=utc(), test_generated=False,
         files={path.relative_to(R).as_posix(): sha(path) for path in files}))
    print('FIT_LOCK: 396 affine readouts and controls locked before test generation', flush=True)
    return {row['id']: row for row in rows}


def test(budget, models, theta, fit_rows):
    verify_lock('FIT_LOCK.json')
    budget.stage = 'test'
    budget.mark('running')
    folder = R / 'test'
    folder.mkdir()
    rows = []
    for seed in range(3):
        base, permutations, x = inputs('test', seed)
        for model in [row for row in models if row['seed'] == seed]:
            budget.guard(disk=True)
            ident = model['id']
            native, collect_seconds = collect_timed(budget, x, theta[model['index']], model['arm'])
            np.savez_compressed(folder / f'{ident}_native.npz', **native)
            fit = load(R / 'fit' / f'{ident}_fit.npz')
            arrays, row = evaluate_model(native, base, permutations, fit)
            np.savez_compressed(folder / f'{ident}_scores.npz', **arrays)
            budget.benchmark_batches += arrays['timing_ns'].size
            rows.append(dict(**fit_rows[ident], test_collect_seconds=collect_seconds, **row))
            dump(folder / 'ROWS.json', rows)
            print('evaluated', ident, flush=True)
    return rows


def freeze():
    assert not (R / 'FREEZE.json').exists() and not (R / 'RUN_STATE.json').exists()
    assert read(R / 'PRECHECK.json')['status'] == read(R / 'TIMING.json')['status'] == 'PASS'
    assert (R / 'audit.py').exists() and (R / 'PROTOCOL.md').exists()
    verify_lock('SOURCE_LOCK.json')
    files = [path for path in R.iterdir() if path.is_file() and path.suffix in ('.py', '.md', '.json', '.npz')
             and path.name not in ('summarize.py', 'report.py')]
    dump(R / 'FREEZE.json', dict(utc=utc(), scientific_data_generated=False, neural_weights_fixed=True,
         files={path.relative_to(R).as_posix(): sha(path) for path in sorted(files)}))
    print('FROZEN', flush=True)


def science():
    budget = Budget()
    dump(R / 'ENVIRONMENT.json', dict(machine=platform.node(), python=sys.version, numpy=np.__version__,
         threads={key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}))
    try:
        models = read(R / 'MODELS.json')['rows']
        theta = load(R / 'models.npz')['theta']
        fitted = calibrate(budget, models, theta)
        rows = test(budget, models, theta, fitted)
        budget.finish(sum(row['fit_seconds'] for row in rows),
                      sum(row['fit_collect_seconds']+row['test_collect_seconds'] for row in rows))
    except Exception as error:
        budget.mark('stopped', reason=str(error))
        dump(R / 'SCIENCE_FAILURE.json', dict(stage=budget.stage, reason=str(error), exception=type(error).__name__,
             baseline_captures=budget.baselines, wall_seconds=time.perf_counter()-budget.start))
        raise


if __name__ == '__main__':
    {'freeze': freeze, 'science': science}[sys.argv[1]]()
