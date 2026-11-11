"""One-shot frozen-network calibration and causal splice experiment."""

import os
import platform
import sys
import time

import numpy as np

import continuation
import metrics
import probe
import task_data
from core import Budget, R, TAPS, check_lock, configurations, donors, dump, read, sha, utc


def meta(row):
    return {key: row[key] for key in ('id', 'seed', 'arm', 'loss')}


def load(path):
    with np.load(path) as archive:
        return dict(archive)


def fit_readouts(budget, models, theta):
    budget.stage = 'fit'
    budget.mark('running')
    (R / 'data').mkdir()
    folder = R / 'fit'
    folder.mkdir()
    rows = []
    for seed in range(3):
        base = task_data.make_base(128, 4011010000 + 100*seed)
        np.savez_compressed(R / 'data' / f'fit_s{seed}.npz', **base)
        x, _ = task_data.assemble(base)
        y = base['y']
        for row in [r for r in models if r['seed'] == seed]:
            budget.guard()
            ident = row['id']
            captured = continuation.collect(x, theta[row['index']], row['arm'], taps=())
            np.savez_compressed(folder / f'{ident}_baseline.npz', **captured)
            budget.record_baseline(len(x))
            fitted_scores = {}
            for name, features in metrics.readout_features(captured['raster']).items():
                fitted = probe.fit(features, y[:, None])
                np.savez_compressed(folder / f'{ident}_{name}.npz', **fitted)
                fitted_scores[name] = fitted['fit_scores'][:, 0]
                budget.fits += 1
            rows.append(dict(**meta(row), **metrics.probe_metrics(fitted_scores, y)))
            dump(folder / 'ROWS.json', rows)
    paths = list((R / 'data').glob('fit_*.npz')) + list(folder.iterdir())
    dump(R / 'FIT_LOCK.json', dict(utc=utc(), test_generated=False,
         files={path.relative_to(R).as_posix(): sha(path) for path in sorted(paths)}))
    print('FIT_LOCK: all 72 offline readouts fitted before test generation', flush=True)


def evaluate(budget, models, theta):
    check_lock(read(R / 'FIT_LOCK.json'))
    budget.stage = 'test'
    budget.mark('running')
    folder = R / 'test'
    folder.mkdir()
    baseline_rows = []
    patch_rows = []
    for seed in range(3):
        base = task_data.make_base(256, 4011020000 + 100*seed)
        np.savez_compressed(R / 'data' / f'test_s{seed}.npz', **base)
        x, _ = task_data.assemble(base)
        y = base['y']
        donor_sets = donors(base)
        early = donor_sets['content_flip']
        assert np.array_equal(x[:, :10], x[early, :10]) and np.array_equal(y, 1-y[early])
        assert np.array_equal(x[:, 16:], x[donor_sets['cue_flip'], 16:])
        assert np.array_equal(x[:, 16:], x[early, 16:])
        assert np.array_equal(x[:, 16:], x[donor_sets['both_flip'], 16:])
        for row in [r for r in models if r['seed'] == seed]:
            budget.guard(disk=True)
            ident, arm = row['id'], row['arm']
            w = theta[row['index']]
            native = continuation.collect(x, w, arm)
            np.savez_compressed(folder / f'{ident}_base.npz', **native)
            budget.record_baseline(len(x))
            fitted = {name: load(R / 'fit' / f'{ident}_{name}.npz') for name in ('counts', 'raster')}
            previous_probes = {t: load(R / 'state_probes' / f'{ident}_joint_t{t}.npz') for t in TAPS}
            baseline_state = {}
            for tap in TAPS:
                state = continuation.splice(native, tap, 'full', donor_sets['self'])
                baseline_state[tap] = probe.predict(previous_probes[tap], continuation.joint_features(state))[:, 0]
            raster = native['raster']
            original_scores = metrics.readout_scores(raster, fitted)
            baseline_rows.append(dict(
                **meta(row), **metrics.native_metrics(native['counts'], y), **metrics.probe_metrics(original_scores, y),
                **{f'joint{t}_ba': probe.metrics(scores, y)['balanced_accuracy'] for t, scores in baseline_state.items()},
                mean_resources=native['resources'].mean(0).tolist(), mean_query_counts=native['counts'].mean(0).tolist()))
            dump(folder / 'BASE_ROWS.json', baseline_rows)
            for tap, group, donor in configurations(arm):
                budget.guard()
                donor_index = donor_sets[donor]
                state = continuation.splice(native, tap, group, donor_index)
                state_score = probe.predict(previous_probes[tap], continuation.joint_features(state))[:, 0]
                resumed = continuation.resume(x, w, arm, tap, state)
                patch_id = f'{ident}_t{tap}_{group}_{donor}'
                np.savez_compressed(folder / f'{patch_id}.npz', donor_index=donor_index,
                                    state_score=state_score, **resumed)
                budget.record_patch(len(x), tap)
                suffix = resumed['suffix_raster']
                if group == 'full':
                    source = donor_sets['self'] if tap == 9 else donor_index
                    assert np.array_equal(suffix, raster[source, max(0, tap-15):]), patch_id
                patch_metrics = metrics.intervention(raster, donor_index, suffix, tap, y, baseline_state[tap],
                                                     state_score, original_scores, fitted)
                patch_rows.append(dict(**meta(row), time=tap, group=group, donor=donor, patch_id=patch_id,
                                       **patch_metrics, mean_suffix_resources=resumed['suffix_resources'].mean(0).tolist()))
            dump(folder / 'ROWS.json', patch_rows)
            print('evaluated', ident, len(patch_rows), 'patch conditions', flush=True)


def freeze():
    assert not (R / 'FREEZE.json').exists() and not (R / 'RUN_STATE.json').exists()
    assert read(R / 'PRECHECK.json')['status'] == read(R / 'TIMING.json')['status'] == 'PASS'
    assert (R / 'audit.py').is_file() and (R / 'PROTOCOL.md').is_file()
    check_lock(read(R / 'SOURCE_LOCK.json'))
    paths = [path for path in R.iterdir() if path.is_file() and path.suffix in ('.py', '.md', '.json', '.npz')
             and path.name not in ('summarize.py', 'report.py')]
    paths.extend((R / 'state_probes').glob('*.npz'))
    dump(R / 'FREEZE.json', dict(utc=utc(), scientific_data_generated=False, neural_weights_fixed=True,
         files={path.relative_to(R).as_posix(): sha(path) for path in sorted(paths)}))
    print('FROZEN', flush=True)


def science():
    budget = Budget()
    dump(R / 'ENVIRONMENT.json', dict(python=sys.version, numpy=np.__version__, machine=platform.node(),
         threads={key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}))
    try:
        rows = read(R / 'MODELS.json')['rows']
        weights = load(R / 'models.npz')['theta']
        fit_readouts(budget, rows, weights)
        evaluate(budget, rows, weights)
        budget.finish()
    except Exception as error:
        budget.mark('stopped', reason=str(error))
        dump(R / 'SCIENCE_FAILURE.json', dict(stage=budget.stage, reason=str(error), exception=type(error).__name__,
             patch_conditions_completed=budget.patches, wall_seconds=time.perf_counter()-budget.start))
        raise


if __name__ == '__main__':
    {'freeze': freeze, 'science': science}[sys.argv[1]]()
