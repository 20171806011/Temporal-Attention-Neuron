"""Verify task information, key routing, native gradients, and budgets."""

import time

import numpy as np

from core import ARMS, CONDITIONS, KEYED, R, dump, memory
import binding_data
import independent_binding_data
import independent_key
import key_model
import population
from run import check_control, scores_with_binding


DATA_SEED = 4099901901
WEIGHT_SEED = 4099901902
DIRECTION_SEED = 4099901903
TEST_SEED = 4099901904
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


def count_objective(counts, y, w):
    z = .5*(counts[:, 1].astype(float)-counts[:, 0].astype(float))
    return float(np.mean(np.logaddexp(0., (1-2*y)*z))+.0001*np.mean(w*w))


def check_routing(base, recorded, arm):
    nominal = recorded['nominal']
    # Independent closed form for one write to each slot, in either order.
    m = nominal['memory']
    expected = np.zeros_like(m)
    for row, order in enumerate(base['order'][:8]):
        for tick, address in ((4, order), (8, 1-order)):
            exponents = np.arange(36-tick)
            expected[row, tick:, :, address] = (.97**exponents)[:, None]*nominal['drive'][row, tick]
    np.testing.assert_allclose(m, expected, rtol=0, atol=5e-12)
    assert np.array_equal(recorded['no_tags']['memory'], np.zeros_like(m))
    assert np.all(nominal['read_gate'][:, :25] == 0) and np.all(nominal['read_gate'][:, 28:] == 0)
    assert np.all(nominal['injections'][:, 9:25] == 0)
    cases = ('nominal', 'swap_read') if arm == 'KEY' else ('nominal',)
    for condition in cases:
        values = recorded[condition]
        if arm == 'KEY':
            selected = base['selected'][:8] if condition == 'nominal' else 1-base['selected'][:8]
            weights = np.repeat(np.eye(2)[selected, None, :], 3, axis=1)
        else:
            weights = np.full((8, 3, 2), .5)
        assert np.array_equal(values['selector'][:, 25:28], weights)
        mixing = np.sum(values['memory'][:, 24:27]*weights[:, :, None, :], axis=-1)
        np.testing.assert_allclose(values['injections'][:, 25:28],
                                   2*values['read_gate'][:, 25:28]*mixing, rtol=0, atol=5e-12)
        for field in ('states', 'spikes', 'injections', 'memory', 'drive'):
            assert np.array_equal(values[field][:, :25], nominal[field][:, :25]), field
    return 4+len(cases)*7


def run_fixture(start):
    base = binding_data.make_base(128, DATA_SEED)
    ibase = independent_binding_data.make_base(128, DATA_SEED)
    assert all(np.array_equal(base[name], ibase[name]) for name in base)
    x, q = binding_data.assemble(base)
    ix, iq = independent_binding_data.assemble(ibase)
    assert q == iq == 28 and np.array_equal(x, ix) and not x[:, 9:25].any()
    y = base['y']
    test = binding_data.make_base(256, TEST_SEED)
    itest = independent_binding_data.make_base(256, TEST_SEED)
    assert all(np.array_equal(test[name], itest[name]) for name in test)
    inputs = {}
    ceilings = {}
    for condition in CONDITIONS:
        xt, qt = binding_data.assemble(test, condition)
        ixt, iqt = independent_binding_data.assemble(itest, condition=condition)
        assert qt == iqt == 28 and np.array_equal(xt, ixt)
        checked = independent_binding_data.check(itest, ixt, condition=condition)
        inputs[condition] = xt, qt
        ceilings[condition] = checked['ceiling']
    initial = population.initialize(WEIGHT_SEED, 6)
    rng = np.random.default_rng(DIRECTION_SEED)
    directions = rng.normal(size=(12, initial.size))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    rows = []
    audit = dict(ledger_seconds=0., evaluation_seconds=0., training_replay_seconds=0.)
    first = {}
    fd_checks, replay_checks, objective_checks = 0, 0, 0
    max_error = 0.
    routing_checks = 0
    for arm in ARMS:
        assert time.perf_counter()-start < 120
        w = initial.copy()
        m, v = np.zeros_like(w), np.zeros_like(w)
        times, archive = [], []
        for step in range(1, FIXTURE_UPDATES+1):
            tick = time.perf_counter()
            loss, grad, counts = key_model.loss_grad(x, y, w, arm, q)
            assert np.isfinite(loss) and np.isfinite(grad).all()
            norm = np.linalg.norm(grad)
            if step == 1:
                first[arm] = (loss, grad.copy(), counts.copy())
            clipped = grad/max(1, norm)
            archive.append((w.copy(), loss, clipped, norm, counts))
            w, m, v = adam(w, clipped, m, v, step)
            times.append(time.perf_counter()-tick)
        rows.append(dict(arm=arm, fixture_rows=128, fixture_updates=FIXTURE_UPDATES, seconds=times,
                         last_six_median_seconds=float(np.median(times[6:]))))

        tick = time.perf_counter()
        reconstructed = initial.copy()
        m, v = np.zeros_like(reconstructed), np.zeros_like(reconstructed)
        for step, (prior, loss, clipped, norm, counts) in enumerate(archive, start=1):
            assert np.array_equal(prior, reconstructed)
            assert abs(count_objective(counts, y, reconstructed)-loss) <= 5e-12
            assert abs(np.linalg.norm(clipped)-min(1, norm)) <= 5e-12
            reconstructed, m, v = adam(reconstructed, clipped, m, v, step)
            objective_checks += 1
        assert np.array_equal(reconstructed, w)
        audit['ledger_seconds'] += time.perf_counter()-tick

        tick = time.perf_counter()
        for prior, _, _, _, counts in (archive[0], archive[-1]):
            ref = independent_key.replay(ix, prior, arm, iq)
            assert np.array_equal(counts, ref['counts'])
            replay_checks += 1
        audit['training_replay_seconds'] += time.perf_counter()-tick

        eval_cases = CONDITIONS + ('swap_read',) if arm == 'KEY' else CONDITIONS
        recorded = {}
        for condition in eval_cases:
            swap = condition == 'swap_read'
            xt, qt = inputs['nominal' if swap else condition]
            tick = time.perf_counter()
            expected = independent_key.replay(xt, w, arm, qt, swap_read=swap)
            audit['evaluation_seconds'] += time.perf_counter()-tick
            actual = key_model.evaluate(xt, w, arm, qt, swap_read=swap)
            verify_archives(actual, expected)
            check_control(condition, scores_with_binding(actual['counts'], test))
            recorded[condition] = actual
            replay_checks += 1
        if arm in KEYED:
            routing_checks += check_routing(test, recorded, arm)

        loss, grad, counts = first[arm]
        anchor = independent_key.anchor(ix, initial, arm, iq)
        assert np.array_equal(counts, np.sum([step['os'] for step in anchor[iq:]], axis=0))
        assert abs(independent_key.objective(ix, y, initial, arm, iq, anchor)-loss) <= 5e-12
        for i, direction in enumerate(directions):
            high = independent_key.objective(ix, y, initial+EPSILON*direction, arm, iq, anchor)
            low = independent_key.objective(ix, y, initial-EPSILON*direction, arm, iq, anchor)
            numeric = (high-low)/(2*EPSILON)
            err = abs(numeric-float(grad @ direction))
            assert err <= FD_TOL, (arm, i, err)
            max_error = max(max_error, err)
            fd_checks += 1
    assert memory()['peak_working_set_bytes'] < 1024**3 and time.perf_counter()-start < 120
    return rows, audit, dict(gradient_directions=fd_checks, max_fd_error=max_error, replay_checks=replay_checks,
                             stored_objective_checks=objective_checks, routing_checks=routing_checks, information_ceilings=ceilings)


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
    audit_unit = 50*audit['ledger_seconds']+audit['evaluation_seconds']+audit['training_replay_seconds']
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
