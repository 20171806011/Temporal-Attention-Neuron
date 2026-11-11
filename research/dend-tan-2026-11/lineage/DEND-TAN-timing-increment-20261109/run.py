"""Calibrate frozen-output decoders, lock them, then score independent test pairs."""

import os
import platform
import sys
import time

import numpy as np

import continuation
import probe
import task_data
import timing_stats as stats
from core import Budget, R, dump, read, sha, utc, verify_lock


def load(path):
    with np.load(path) as archive:
        return dict(archive)


def fit_model(raster, base, permutations):
    counts = raster.sum(1)
    matched = stats.pairs(counts, base)
    native_features = stats.features(raster)
    pair_coefs, pair_scores, statuses = [], [], []
    for mapping in [None, *permutations]:
        x = native_features if mapping is None else stats.features(stats.shuffle(raster, mapping))
        delta = stats.timing_differences(x, matched)
        coef, status = stats.fit_rank(delta, matched['weight'])
        pair_coefs.append(coef)
        pair_scores.append(delta @ coef)
        statuses.append(status)
    count_mean, count_coef, intercept = stats.fit_class(native_features[:, :2], base['y'])
    raster_mean, raster_coef, raster_intercept = stats.fit_class(native_features, base['y'])
    assert intercept == raster_intercept
    arrays = dict(**matched, pair_coef=np.stack(pair_coefs), pair_scores=np.stack(pair_scores),
                  count_mean=count_mean, count_coef=count_coef, raster_mean=raster_mean,
                  raster_coef=raster_coef, intercept=np.array(intercept))
    row = dict(fit_pairs=len(matched['low']), fit_matched_quartets=len(np.unique(matched['quartet'])),
               pair_fit_status=statuses)
    return arrays, row


def evaluate_model(native, base, permutations, fit, prior_raster):
    raster, counts, y = native['raster'], native['counts'], base['y']
    matched = stats.pairs(counts, base)
    native_features = stats.features(raster)
    scores = []
    for index, mapping in enumerate([None, *permutations]):
        if mapping is None:
            x = native_features
        else:
            shuffled = stats.shuffle(raster, mapping)
            assert np.array_equal(shuffled.sum(1), counts)
            x = stats.features(shuffled)
        scores.append(stats.timing_differences(x, matched) @ fit['pair_coef'][index])
    scores = np.stack(scores)
    ranks = [stats.rank_rate(value, matched['weight']) for value in scores]
    shuffle_mean = None if ranks[0] is None else float(np.mean(ranks[1:]))
    pair_4_4 = np.all(counts[matched['low']] == 4, axis=1)
    quartets_4_4 = matched['quartet'][pair_4_4]
    weights_4_4 = stats.pair_weights(quartets_4_4)
    ranks_4_4 = [stats.rank_rate(value[pair_4_4], weights_4_4) for value in scores]
    equal_raster = np.all(raster[matched['low']] == raster[matched['high']], axis=(1, 2))
    count_score = stats.class_scores(native_features[:, :2], fit['count_mean'], fit['count_coef'], fit['intercept'])
    raster_score = stats.class_scores(native_features, fit['raster_mean'], fit['raster_coef'], fit['intercept'])
    prior_score = probe.predict(prior_raster, raster.reshape(len(raster), 16))[:, 0]
    fixed_4_4 = np.all(counts == 4, axis=1)
    fixed_labels = np.bincount(y[fixed_4_4], minlength=2).tolist()
    n_quartets = len(np.unique(matched['quartet']))
    target = 2*y-1
    row = dict(**stats.native_metrics(counts, y), test_pairs=len(matched['low']),
               test_matched_quartets=n_quartets, test_matched_fraction=n_quartets/(len(y)//4),
               pair_equal_raster_fraction=float(equal_raster.mean()) if len(equal_raster) else None,
               matched_4_4_pairs=int(pair_4_4.sum()), matched_4_4_quartets=len(np.unique(quartets_4_4)),
               pair_rank=ranks[0], pair_rank_shuffled=ranks[1:], pair_rank_shuffle_mean=shuffle_mean,
               pair_rank_delta=None if shuffle_mean is None else ranks[0]-shuffle_mean,
               count_pair_rank=.5 if len(equal_raster) else None,
               pair_rank_4_4=ranks_4_4[0],
               pair_rank_shuffled_4_4_mean=None if ranks_4_4[0] is None else float(np.mean(ranks_4_4[1:])),
               count_ba=stats.balanced_accuracy(stats.classify(count_score), y),
               count_mse=float(np.mean((count_score-target)**2)),
               raster_ba=stats.balanced_accuracy(stats.classify(raster_score), y),
               raster_mse=float(np.mean((raster_score-target)**2)),
               fixed_4_4_n=int(fixed_4_4.sum()), fixed_4_4_label_counts=fixed_labels,
               prior_4_4_raster_ba=(stats.balanced_accuracy(stats.classify(prior_score[fixed_4_4]), y[fixed_4_4])
                                     if min(fixed_labels) > 0 else None),
               mean_resources=native['resources'].mean(0).tolist())
    arrays = dict(**matched, pair_scores=scores, count_scores=count_score,
                  raster_scores=raster_score, prior_raster_scores=prior_score)
    return arrays, row


def inputs(split, seed):
    n, offset, shuffle_offset = (512, 4021010000, 4021030000) if split == 'fit' else (1024, 4021020000, 4021040000)
    base = task_data.make_base(n, offset+100*seed)
    permutations = stats.make_permutations(n, shuffle_offset+100*seed)
    np.savez_compressed(R / 'data' / f'{split}_s{seed}.npz', **base)
    np.savez_compressed(R / 'data' / f'{split}_perm_s{seed}.npz', permutations=permutations)
    x, _ = task_data.assemble(base)
    return base, permutations, x


def calibrate(budget, models, theta):
    budget.stage = 'fit'
    budget.mark('running')
    (R / 'data').mkdir()
    output = R / 'fit'
    output.mkdir()
    rows = []
    for seed in range(3):
        base, permutations, x = inputs('fit', seed)
        for model in [row for row in models if row['seed'] == seed]:
            budget.guard(disk=True)
            ident = model['id']
            native = continuation.collect(x, theta[model['index']], model['arm'], taps=())
            np.savez_compressed(output / f'{ident}_native.npz', **native)
            budget.record_baseline(len(x))
            arrays, row = fit_model(native['raster'], base, permutations)
            np.savez_compressed(output / f'{ident}_fit.npz', **arrays)
            budget.rank_fits += 17
            budget.class_fits += 2
            rows.append(dict(**{key: model[key] for key in ('id', 'seed', 'arm', 'loss')}, **row))
            dump(output / 'ROWS.json', rows)
    files = sorted([*(R / 'data').glob('fit*.npz'), *output.iterdir()])
    dump(R / 'FIT_LOCK.json', dict(utc=utc(), test_generated=False,
         files={path.relative_to(R).as_posix(): sha(path) for path in files}))
    print('FIT_LOCK: 612 ranking configurations and 72 classifiers locked', flush=True)


def test(budget, models, theta):
    verify_lock('FIT_LOCK.json')
    budget.stage = 'test'
    budget.mark('running')
    output = R / 'test'
    output.mkdir()
    fit_rows = {row['id']: row for row in read(R / 'fit/ROWS.json')}
    rows = []
    for seed in range(3):
        base, permutations, x = inputs('test', seed)
        for model in [row for row in models if row['seed'] == seed]:
            budget.guard(disk=True)
            ident = model['id']
            native = continuation.collect(x, theta[model['index']], model['arm'], taps=())
            np.savez_compressed(output / f'{ident}_native.npz', **native)
            budget.record_baseline(len(x))
            fitted = load(R / 'fit' / f'{ident}_fit.npz')
            prior = load(R / 'prior_raster' / f'{ident}_raster.npz')
            arrays, row = evaluate_model(native, base, permutations, fitted, prior)
            np.savez_compressed(output / f'{ident}_scores.npz', **arrays)
            rows.append(dict(**fit_rows[ident], **row))
            dump(output / 'ROWS.json', rows)
            print('evaluated', ident, 'matched quartets', row['test_matched_quartets'], flush=True)


def freeze():
    assert not (R / 'FREEZE.json').exists() and not (R / 'RUN_STATE.json').exists()
    assert read(R / 'PRECHECK.json')['status'] == read(R / 'TIMING.json')['status'] == 'PASS'
    assert (R / 'audit.py').exists() and (R / 'PROTOCOL.md').exists()
    verify_lock('SOURCE_LOCK.json')
    files = [path for path in R.iterdir() if path.is_file() and path.suffix in ('.py', '.md', '.json', '.npz')
             and path.name not in ('summarize.py', 'report.py')]
    files.extend((R / 'prior_raster').glob('*.npz'))
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
        calibrate(budget, models, theta)
        test(budget, models, theta)
        budget.finish()
    except Exception as error:
        budget.mark('stopped', reason=str(error))
        dump(R / 'SCIENCE_FAILURE.json', dict(stage=budget.stage, reason=str(error), exception=type(error).__name__,
             baseline_captures=budget.baselines, wall_seconds=time.perf_counter()-budget.start))
        raise


if __name__ == '__main__':
    {'freeze': freeze, 'science': science}[sys.argv[1]]()
