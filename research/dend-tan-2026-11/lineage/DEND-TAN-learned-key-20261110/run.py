"""One-shot matched-D16 learned key-query binding with native count readout."""

import os
import platform
import sys
import time

import numpy as np

import metrics
import learned_model
import population
import learned_data
from core import (ARMS, SWAPPABLE, CONDITIONS,
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
            loss, gradient, counts = learned_model.loss_grad(x, y, w, arm, q)
            budget.record_train(len(x), x.shape[1])
            norm = float(np.linalg.norm(gradient))
            regularizer = .0002*w/180
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
        base = learned_data.make_base(128, 4077010000+100*seed)
        np.savez_compressed(folder / f'base_s{seed}.npz', **base)
        x, q = learned_data.assemble(base)
        steps = x.shape[1]
        assert steps == 36 and q == 28
        initial = np.r_[population.initialize(4077030000+seed, 6), np.zeros(4)]
        for arm in ARMS:
            budget.guard(disk=True)
            ident = f's{seed}_{arm}'
            start = time.perf_counter()
            fit_one(budget, x, base['y'], q, initial, arm, updates, folder / (ident+'.npz'))
            sample_steps = updates*len(x)*steps
            rows.append(dict(id=ident, seed=seed, arm=arm, train_delay=16, h=6, lr=RATE,
                             loss='CE', updates=updates, seconds=time.perf_counter()-start, T=steps, q=q,
                             dense_forward_macs=sample_steps*156, dense_backward_macs=sample_steps*240,
                             memory_backward_unit_steps=sample_steps*12,
                             matcher_backward_unit_steps=sample_steps*2 if arm == 'LEARN' else 0,
                             cost_per_episode=structural_cost(arm, steps)))
            dump(folder / 'ROWS.json', rows)
            print('trained', ident, f'{len(rows)}/{3*blocks}', flush=True)
    dump(folder / 'TRAIN_LOCK.json', dict(utc=utc(), test_generated=False,
         files={path.name: sha(path) for path in sorted(folder.iterdir()) if path.is_file()}))
    return rows


def check_control(condition, scores):
    if condition in ('no_cue', 'no_tags'):
        assert scores['disagree_ba'] == .5*scores['disagree_coverage']
        assert scores['balanced_accuracy'] <= .75
    if condition == 'no_cue':
        assert scores['disagree_joint_correct'] == 0
    if condition == 'no_contents':
        assert scores['balanced_accuracy'] == .5*scores['coverage']


def scores_with_binding(counts, base):
    primary = metrics.score(counts, base)
    counterfactual = metrics.score(counts[:, ::-1], base)
    correct = counts[np.arange(len(counts)), base['y']] > counts[np.arange(len(counts)), 1-base['y']]
    disagree = base['first_bit'] != base['second_bit']
    return dict(**primary, counterfactual_joint=counterfactual['disagree_joint_correct'],
                counterfactual_disagree_ba=counterfactual['disagree_ba'],
                order0_disagree_ba=float(correct[disagree & (base['order'] == 0)].mean()),
                order1_disagree_ba=float(correct[disagree & (base['order'] == 1)].mean()))


def scores_with_routing(native, base):
    scores = scores_with_binding(native['counts'], base)
    rows = np.arange(len(base['y']))
    target = (base['selected'] == base['order']).astype(int)
    attention = native['cue_weights'][:, 0]
    correct, other = attention[rows, target], attention[rows, 1-target]
    return dict(**scores, routing_accuracy=float((correct > other).mean()),
                routing_tie_rate=float((correct == other).mean()), routing_weight_mean=float(correct.mean()))


def evaluate(budget, blocks, training_rows):
    for name, digest in read(R / 'train/TRAIN_LOCK.json')['files'].items():
        assert sha(R / 'train' / name) == digest, name
    budget.stage = 'evaluate'
    budget.mark('running')
    folder = R / 'evaluate'
    folder.mkdir()
    rows = []
    for seed in range(blocks):
        base = learned_data.make_base(256, 4077020000+100*seed)
        np.savez_compressed(folder / f'test_s{seed}.npz', **base)
        inputs = {condition: learned_data.assemble(base, condition) for condition in CONDITIONS}
        for train_row in [row for row in training_rows if row['seed'] == seed]:
            ident, arm = train_row['id'], train_row['arm']
            with np.load(R / 'train' / (ident+'.npz')) as archive:
                w = archive['theta']
            cases = CONDITIONS + ('swap_read',) if arm in SWAPPABLE else CONDITIONS
            for condition in cases:
                budget.guard()
                swap = condition == 'swap_read'
                x, q = inputs['nominal' if swap else condition]
                steps = x.shape[1]
                native = learned_model.evaluate(x, w, arm, q, swap_read=swap)
                assert all(np.isfinite(array).all() for array in native.values()), (ident, condition)
                budget.record_eval(len(x), steps)
                np.savez_compressed(folder / f'{ident}_{condition}.npz', **native)
                scores = scores_with_routing(native, base)
                check_control(condition, scores)
                meta = {key: train_row[key] for key in ('id', 'seed', 'arm', 'train_delay', 'h', 'lr', 'loss', 'updates')}
                rows.append(dict(**meta, test_delay=16, condition=condition, T=steps, q=q,
                                 **scores, mean_resources=native['resources'][:, :4].mean(0).tolist(),
                                 cost_per_episode=structural_cost(arm, steps)))
            dump(folder / 'ROWS.json', rows)
            budget.guard(disk=True)
            print('evaluated', ident, f'{len(rows)}/{14*blocks}', flush=True)


def freeze():
    assert not (R / 'FREEZE.json').exists() and not (R / 'RUN_STATE.json').exists()
    timing = read(R / 'TIMING.json')
    assert read(R / 'PRECHECK.json')['status'] == timing['status'] == 'PASS'
    assert timing['updates'] == 600 and timing['blocks'] in (8, 12)
    forecasts = timing['candidate_forecast_seconds']
    audit_forecasts = timing['candidate_audit_forecast_seconds']
    selected = next(n for n in (12, 8) if forecasts[str(n)] < 840 and audit_forecasts[str(n)] < 300)
    assert timing['blocks'] == selected and timing['forecast_seconds'] == forecasts[str(selected)]
    assert timing['audit_forecast_seconds'] == audit_forecasts[str(selected)]
    for name in ('audit.py', 'independent_learned.py', 'independent_learned_data.py', 'PROTOCOL.md', 'REPRODUCE.md', 'summarize.py', 'report.py'):
        assert (R / name).is_file(), name
    verify_lock('SOURCE_LOCK.json')
    paths = [path for path in R.iterdir() if path.is_file() and path.suffix in ('.py', '.md', '.json')]
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
