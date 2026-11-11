"""Test the local surrogate and size the paired training batch before freeze."""

import time

import numpy as np

from core import ARMS, EVAL_RULES, R, dump, memory
import independent_data
import independent_scale
import independent_training
import population
import scale_model
import task_data
import train_model


DATA_SEED = 4099901801
WEIGHT_SEED = 4099901802
DIRECTION_SEED = 4099901803
TEST_SEED = 4099901804
FIXTURE_UPDATES = 12
UPDATES = 600
EPSILON = 1e-5
FD_TOL = 2e-7


def adam(w, grad, m, v, step):
    m = .9*m+.1*grad
    v = .999*v+.001*grad*grad
    updated = np.clip(w-.0025*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8), -2, 2)
    return updated, m, v


def verify_archives(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if key in ('counts', 'spikes', 'resources'):
            assert np.array_equal(actual[key], expected[key]), key
        else:
            np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=5e-12, err_msg=key)


def run_fixture(start):
    base = task_data.make_base(64, DATA_SEED)
    ibase = independent_data.make_base(64, DATA_SEED)
    assert all(np.array_equal(base[name], ibase[name]) for name in base)
    x, q = task_data.assemble(base, 16)
    ix, iq = independent_data.assemble(ibase, 16)
    assert q == iq == 28 and np.array_equal(x, ix) and not x[:, 9:25].any()
    y = base['y']
    test = task_data.make_base(256, TEST_SEED)
    xt, qt = task_data.assemble(test, 16)
    ixt, iqt = independent_data.assemble(independent_data.make_base(256, TEST_SEED), 16)
    assert qt == iqt == 28 and np.array_equal(xt, ixt)
    initial = population.initialize(WEIGHT_SEED, 6)
    rng = np.random.default_rng(DIRECTION_SEED)
    directions = rng.normal(size=(12, initial.size))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    rows = []
    audit = dict(ledger_seconds=0., evaluation_seconds=0., training_replay_seconds=0., direction_seconds=0.)
    first = {}
    fd_checks, replay_checks, objective_checks = 0, 0, 0
    max_error = 0.
    for arm in ARMS:
        assert time.perf_counter()-start < 120
        w = initial.copy()
        m, v = np.zeros_like(w), np.zeros_like(w)
        times, archive = [], []
        for step in range(1, FIXTURE_UPDATES+1):
            tick = time.perf_counter()
            loss, grad, counts = train_model.loss_grad(x, y, w, arm, q)
            assert np.isfinite(loss) and np.isfinite(grad).all()
            norm = np.linalg.norm(grad)
            if step == 1:
                first[arm] = (loss, grad.copy(), counts.copy())
            clipped = grad/max(1, norm)
            archive.append((w.copy(), loss, clipped, norm, counts))
            w, m, v = adam(w, clipped, m, v, step)
            times.append(time.perf_counter()-tick)
        rows.append(dict(arm=arm, fixture_rows=64, fixture_updates=FIXTURE_UPDATES, seconds=times,
                         last_six_median_seconds=float(np.median(times[6:]))))

        beta_arm = train_model.FORWARD_ARMS[arm]
        tick = time.perf_counter()
        reconstructed = initial.copy()
        m, v = np.zeros_like(reconstructed), np.zeros_like(reconstructed)
        for step, (prior, loss, clipped, norm, counts) in enumerate(archive, start=1):
            assert np.array_equal(prior, reconstructed)
            assert abs(independent_training.count_objective(counts, y, reconstructed)-loss) <= 5e-12
            assert abs(np.linalg.norm(clipped)-min(1, norm)) <= 5e-12
            reconstructed, m, v = adam(reconstructed, clipped, m, v, step)
            objective_checks += 1
        assert np.array_equal(reconstructed, w)
        audit['ledger_seconds'] += time.perf_counter()-tick

        tick = time.perf_counter()
        for prior, _, _, _, counts in (archive[0], archive[-1]):
            ref = independent_scale.replay(ix, prior, beta_arm, iq)
            assert np.array_equal(counts, ref['counts'])
            replay_checks += 1
        audit['training_replay_seconds'] += time.perf_counter()-tick

        for eval_arm in EVAL_RULES.values():
            tick = time.perf_counter()
            expected = independent_scale.replay(ixt, w, eval_arm, iqt)
            audit['evaluation_seconds'] += time.perf_counter()-tick
            verify_archives(scale_model.evaluate(xt, w, eval_arm, qt), expected)
            replay_checks += 1

        loss, grad, counts = first[arm]
        tick = time.perf_counter()
        ref_counts, anchor = independent_training.rollout(ix, initial, arm, iq, None)
        assert np.array_equal(counts, ref_counts)
        assert abs(independent_training.objective(ix, y, initial, arm, iq, anchor)-loss) <= 5e-12
        for i, direction in enumerate(directions):
            high = independent_training.objective(ix, y, initial+EPSILON*direction, arm, iq, anchor)
            low = independent_training.objective(ix, y, initial-EPSILON*direction, arm, iq, anchor)
            numeric = (high-low)/(2*EPSILON)
            err = abs(numeric-float(grad @ direction))
            assert err <= FD_TOL, (arm, i, err)
            max_error = max(max_error, err)
            fd_checks += 1
            if i == 1:
                audit['direction_seconds'] += time.perf_counter()-tick
    assert first['DYN'][0] == first['DETACH'][0]
    assert np.array_equal(first['DYN'][2], first['DETACH'][2])
    diff = float(np.linalg.norm(first['DYN'][1]-first['DETACH'][1]))
    assert diff > 1e-8, 'fixture must exercise the removed gradient'
    assert memory()['peak_working_set_bytes'] < 1024**3 and time.perf_counter()-start < 120
    return rows, audit, dict(gradient_directions=fd_checks, max_fd_error=max_error, replay_checks=replay_checks,
                             stored_objective_checks=objective_checks, initial_dynamic_detach_gradient_l2=diff)


def main():
    assert not any((R / name).exists() for name in ('PRECHECK.json', 'TIMING.json', 'FREEZE.json',
                                                    'RUN_STATE.json', 'train', 'evaluate'))
    start = time.perf_counter()
    try:
        rows, audit, checks = run_fixture(start)
    except Exception as error:
        attempt = len(list(R.glob('PRECHECK_FAILURE_*.json')))+1
        dump(R / f'PRECHECK_FAILURE_{attempt}.json', dict(status='FAIL', reason=str(error),
             exception=type(error).__name__, scientific_data_generated=False))
        raise
    unit = sum(row['last_six_median_seconds'] for row in rows)
    audit_unit = 50*audit['ledger_seconds']+audit['evaluation_seconds']+audit['training_replay_seconds']+audit['direction_seconds']
    forecasts = {str(n): 2*n*UPDATES*unit+100 for n in (12, 8)}
    audit_forecasts = {str(n): 3*n*audit_unit+45 for n in (12, 8)}
    chosen = next((n for n in (12, 8) if forecasts[str(n)] < 840 and audit_forecasts[str(n)] < 300), None)
    dump(R / 'PRECHECK.json', dict(status='PASS', scientific_data_generated=False, scientific_seeds_used=False,
         fixture_data_seed=DATA_SEED, fixture_weight_seed=WEIGHT_SEED, fixture_direction_seed=DIRECTION_SEED,
         fixture_test_seed=TEST_SEED, fd_epsilon=EPSILON, fd_tolerance=FD_TOL, **checks,
         wall_seconds=time.perf_counter()-start, **memory()))
    dump(R / 'TIMING.json', dict(status='PASS' if chosen else 'INFEASIBLE', blocks=chosen, updates=UPDATES,
         forecast_seconds=forecasts[str(chosen)] if chosen else None,
         audit_forecast_seconds=audit_forecasts[str(chosen)] if chosen else None,
         candidate_forecast_seconds=forecasts, candidate_audit_forecast_seconds=audit_forecasts,
         rows=rows, audit_components=audit, scientific_seeds_used=False, scientific_data_generated=False))
    if chosen is None:
        raise SystemExit('No allowed block count fits both budgets')
    print(f'PRECHECK PASS: {checks}; {chosen} blocks; science {forecasts[str(chosen)]:.2f}s, audit {audit_forecasts[str(chosen)]:.2f}s',
          flush=True)


if __name__ == '__main__':
    main()
