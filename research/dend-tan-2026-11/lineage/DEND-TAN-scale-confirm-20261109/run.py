"""One-shot matched-D16 replication with native count readout."""

import os
import platform
import sys
import time

import numpy as np

import metrics
import scale_model as model
import population
import task_data
from core import (ARMS, ATTENTION_ARMS, TRACE_ARMS, CONTROLS, TRAIN_DELAYS, TEST_DELAYS,
                  Budget, R, dump, read, sha, structural_cost, utc, verify_lock)


RATE = .0025


def fit_one(budget, x, y, q, initial, arm, updates, destination):
    w = initial.copy()
    m, v = np.zeros_like(w), np.zeros_like(w)
    gradient_rows, count_rows, history = [], [], []

    def save(path):
        np.savez_compressed(path, initial=initial, theta=w, gradients=gradient_rows,
                            counts=count_rows, history=history)

    try:
        for k in range(updates):
            budget.guard()
            loss, gradient, counts = model.loss_grad(x, y, w, arm, q)
            budget.record_train(len(x), x.shape[1])
            norm = float(np.linalg.norm(gradient))
            regularizer = .0002*w/176
            assert np.isfinite(loss) and np.isfinite(gradient).all() and np.isfinite(norm)
            history.append([loss, norm, np.linalg.norm(gradient-regularizer), np.linalg.norm(regularizer)])
            gradient /= max(1, norm)
            gradient_rows.append(gradient.copy())
            count_rows.append(counts.astype(np.uint8))
            m = .9*m + .1*gradient
            v = .999*v + .001*gradient*gradient
            corrected_m = m/(1-.9**(k+1))
            corrected_v = v/(1-.999**(k+1))
            w = np.clip(w-RATE*corrected_m/(np.sqrt(corrected_v)+1e-8), -2, 2)
    except Exception:
        save(destination.with_name(destination.stem+'_PARTIAL.npz'))
        raise
    save(destination)


def train(budget, blocks, updates):
    budget.stage = 'train'
    budget.mark('running')
    folder = R / 'train'
    folder.mkdir()
    rows = []
    for seed in range(blocks):
        base = task_data.make_base(64, 4071010000+100*seed)
        np.savez_compressed(folder / f'base_s{seed}.npz', **base)
        initial = population.initialize(4071030000+seed, 6)
        for arm in ARMS:
            for delay in TRAIN_DELAYS:
                budget.guard(disk=True)
                x, q = task_data.assemble(base, delay)
                steps = x.shape[1]
                ident = f's{seed}_{arm}_d{delay}'
                start = time.perf_counter()
                fit_one(budget, x, base['y'], q, initial, arm, updates, folder / (ident+'.npz'))
                att = arm in ATTENTION_ARMS
                rows.append(dict(id=ident, seed=seed, arm=arm, train_delay=delay, h=6, lr=RATE,
                                 loss='CE', updates=updates, seconds=time.perf_counter()-start, T=steps, q=q,
                                 dense_forward_macs=updates*64*steps*156,
                                 dense_backward_macs=updates*64*steps*240,
                                 attention_backward_unit_steps=updates*64*steps*12 if att else 0,
                                 history_gradient_accumulations=updates*64*(5*steps-10)*12 if att else 0,
                                 memory_backward_unit_steps=updates*64*steps*12 if arm in TRACE_ARMS else 0,
                                 cost_per_episode=structural_cost(arm, steps)))
                dump(folder / 'ROWS.json', rows)
                print('trained', ident, f'{len(rows)}/{3*blocks}', flush=True)
    dump(folder / 'TRAIN_LOCK.json', dict(utc=utc(), test_generated=False,
         files={path.name: sha(path) for path in sorted(folder.iterdir()) if path.is_file()}))
    return rows


def evaluate(budget, blocks, training_rows):
    for name, digest in read(R / 'train/TRAIN_LOCK.json')['files'].items():
        assert sha(R / 'train' / name) == digest, name
    budget.stage = 'evaluate'
    budget.mark('running')
    folder = R / 'evaluate'
    folder.mkdir()
    rows = []
    for seed in range(blocks):
        base = task_data.make_base(256, 4071020000+100*seed)
        np.savez_compressed(folder / f'test_s{seed}.npz', **base)
        for train_row in [row for row in training_rows if row['seed'] == seed]:
            ident, arm, delay = (train_row[key] for key in ('id', 'arm', 'train_delay'))
            with np.load(R / 'train' / (ident+'.npz')) as archive:
                w = archive['theta']
            cases = [(d, 'nominal') for d in TEST_DELAYS] + [(delay, condition) for condition in CONTROLS]
            for test_delay, condition in cases:
                budget.guard()
                x, q = task_data.assemble(base, test_delay, condition)
                steps = x.shape[1]
                native = model.evaluate(x, w, arm, q)
                budget.record_eval(len(x), steps)
                np.savez_compressed(folder / f'{ident}_d{test_delay}_{condition}.npz', **native)
                scores = metrics.score(native['counts'], base)
                metrics.check_control(condition, scores)
                meta = {key: train_row[key] for key in ('id', 'seed', 'arm', 'train_delay', 'h', 'lr', 'loss', 'updates')}
                rows.append(dict(**meta, test_delay=test_delay, condition=condition, T=steps, q=q,
                                 **scores, mean_resources=native['resources'][:, :4].mean(0).tolist(),
                                 cost_per_episode=structural_cost(arm, steps)))
            dump(folder / 'ROWS.json', rows)
            budget.guard(disk=True)
            print('evaluated', ident, len(rows), 'cases', flush=True)


def freeze():
    assert not (R / 'FREEZE.json').exists() and not (R / 'RUN_STATE.json').exists()
    timing = read(R / 'TIMING.json')
    assert read(R / 'PRECHECK.json')['status'] == timing['status'] == 'PASS'
    assert timing['updates'] == 600 and timing['blocks'] in (8, 12)
    forecasts = timing['candidate_forecast_seconds']
    selected = next(n for n in (12, 8) if forecasts[str(n)] < 840)
    assert timing['blocks'] == selected and timing['forecast_seconds'] == forecasts[str(selected)]
    assert (R / 'audit.py').is_file() and (R / 'PROTOCOL.md').is_file()
    verify_lock('SOURCE_LOCK.json')
    paths = [path for path in R.iterdir() if path.is_file() and path.suffix in ('.py', '.md', '.json')
             and path.name not in ('summarize.py', 'report.py')]
    dump(R / 'FREEZE.json', dict(utc=utc(), scientific_data_generated=False, blocks=timing['blocks'], updates_per_fit=timing['updates'],
         files={path.name: sha(path) for path in sorted(paths)}))
    print('FROZEN', timing['updates'], 'updates per fit', flush=True)


def science():
    frozen = read(R / 'FREEZE.json')
    blocks, updates = frozen['blocks'], frozen['updates_per_fit']
    assert blocks in (8, 12) and updates == 600
    keys = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
    assert all(os.environ.get(key) == '1' for key in keys), 'single-thread CPU contract'
    budget = Budget()
    dump(R / 'ENVIRONMENT.json', dict(machine=platform.node(), python=sys.version, numpy=np.__version__,
         threads={key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}))
    try:
        rows = train(budget, blocks, updates)
        evaluate(budget, blocks, rows)
        budget.finish(blocks, updates)
    except Exception as error:
        budget.mark('stopped', reason=str(error))
        dump(R / 'SCIENCE_FAILURE.json', dict(stage=budget.stage, reason=str(error), exception=type(error).__name__,
             updates=budget.updates, evaluations=budget.evaluations, wall_seconds=time.perf_counter()-budget.start))
        raise


if __name__ == '__main__':
    {'freeze': freeze, 'science': science}[sys.argv[1]]()
