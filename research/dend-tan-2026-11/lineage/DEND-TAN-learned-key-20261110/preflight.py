"""Verify rotated-key information, FIFO, matching gradients, and budgets."""

import time

import numpy as np

from core import ARMS, CONDITIONS, SWAPPABLE, R, dump, memory
import learned_data
import independent_learned_data
import independent_learned
import learned_model
import population
from run import check_control, scores_with_routing


DATA_SEED = 4099902101
WEIGHT_SEED = 4099902102
DIRECTION_SEED = 4099902103
TEST_SEED = 4099902104
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
    k, values = nominal['key_bank'], nominal['value_bank']
    expect_keys, expect_values = np.zeros_like(k), np.zeros_like(values)
    for row, order in enumerate(base['order'][:8]):
        first = base['key_vectors'][base['group'][row], order]
        second = base['key_vectors'][base['group'][row], 1-order]
        expect_keys[row, 4:8, 0] = first
        expect_keys[row, 8:, 0] = second
        expect_keys[row, 8:, 1] = first
        u4, u8 = (nominal['drive'][row, t] for t in (4, 8))
        expect_values[row, 4:8, :, 0] = (.97**np.arange(4))[:, None]*u4
        expect_values[row, 8:, :, 0] = (.97**np.arange(28))[:, None]*u8
        expect_values[row, 8:, :, 1] = (.97**np.arange(4, 32))[:, None]*u4
    assert np.array_equal(k, expect_keys)
    np.testing.assert_allclose(values, expect_values, rtol=0, atol=5e-12)
    assert not recorded['no_tags']['key_bank'].any()
    assert np.all(recorded['no_tags']['cue_weights'] == .5)
    assert np.all(recorded['no_cue']['cue_weights'] == .5)
    assert np.all(nominal['read_gate'][:, :25] == 0) and np.all(nominal['read_gate'][:, 28:] == 0)
    np.testing.assert_allclose(nominal['weights'].sum(axis=-1), 1., rtol=0, atol=5e-15)
    read_values = np.sum(values[:, 24:27]*nominal['weights'][:, 25:28, None, :], axis=-1)
    np.testing.assert_allclose(nominal['injections'][:, 25:28],
                               2*nominal['read_gate'][:, 25:28]*read_values, rtol=0, atol=5e-12)
    if arm == 'MEAN':
        assert np.all(nominal['cue_weights'] == .5)
    if arm == 'ORACLE':
        assert scores_with_routing(nominal, base)['routing_accuracy'] == 1.
        assert scores_with_routing(recorded['swap_read'], base)['routing_accuracy'] == 0.
    if arm in SWAPPABLE:
        swapped = recorded['swap_read']
        for key in ('key_bank', 'value_bank', 'scores', 'write_gate', 'read_gate'):
            assert np.array_equal(swapped[key], nominal[key]), key
        for key in ('states', 'spikes', 'injections', 'drive'):
            assert np.array_equal(swapped[key][:, :25], nominal[key][:, :25]), key
        assert np.array_equal(swapped['cue_weights'], nominal['cue_weights'][..., ::-1])
    return 8+(10 if arm in SWAPPABLE else 1)+(2 if arm == 'ORACLE' else 0)


def run_fixture(start):
    base = learned_data.make_base(128, DATA_SEED)
    ibase = independent_learned_data.make_base(128, DATA_SEED)
    assert all(np.array_equal(base[name], ibase[name]) for name in base)
    x, q = learned_data.assemble(base)
    ix, iq = independent_learned_data.assemble(ibase)
    assert q == iq == 28 and np.array_equal(x, ix) and not x[:, 9:25].any()
    y = base['y']
    test = learned_data.make_base(256, TEST_SEED)
    itest = independent_learned_data.make_base(256, TEST_SEED)
    assert all(np.array_equal(test[name], itest[name]) for name in test)
    inputs = {}
    ceilings = {}
    for condition in CONDITIONS:
        xt, qt = learned_data.assemble(test, condition)
        ixt, iqt = independent_learned_data.assemble(itest, condition=condition)
        assert qt == iqt == 28 and np.array_equal(xt, ixt)
        checked = independent_learned_data.check(itest, ixt, condition=condition)
        inputs[condition] = xt, qt
        ceilings[condition] = checked['ceiling']
    initial = np.r_[population.initialize(WEIGHT_SEED, 6), np.zeros(4)]
    rng = np.random.default_rng(DIRECTION_SEED)
    directions = rng.normal(size=(12, initial.size))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    directions = np.vstack((directions, np.eye(180)[176:]))

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
            loss, grad, counts = learned_model.loss_grad(x, y, w, arm, q)
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
            ref = independent_learned.replay(ix, prior, arm, iq)
            assert np.array_equal(counts, ref['counts'])
            replay_checks += 1
        audit['training_replay_seconds'] += time.perf_counter()-tick

        eval_cases = CONDITIONS + ('swap_read',) if arm in SWAPPABLE else CONDITIONS
        recorded = {}
        for condition in eval_cases:
            swap = condition == 'swap_read'
            xt, qt = inputs['nominal' if swap else condition]
            tick = time.perf_counter()
            expected = independent_learned.replay(xt, w, arm, qt, swap_read=swap)
            audit['evaluation_seconds'] += time.perf_counter()-tick
            actual = learned_model.evaluate(xt, w, arm, qt, swap_read=swap)
            verify_archives(actual, expected)
            check_control(condition, scores_with_routing(actual, test))
            recorded[condition] = actual
            replay_checks += 1
        routing_checks += check_routing(test, recorded, arm)
        if arm != 'LEARN':
            assert np.array_equal(w[176:], np.zeros(4))

        loss, grad, counts = first[arm]
        anchor = independent_learned.anchor(ix, initial, arm, iq)
        assert np.array_equal(counts, np.sum([step['os'] for step in anchor[iq:]], axis=0))
        assert abs(independent_learned.objective(ix, y, initial, arm, iq, anchor)-loss) <= 5e-12
        for i, direction in enumerate(directions):
            high = independent_learned.objective(ix, y, initial+EPSILON*direction, arm, iq, anchor)
            low = independent_learned.objective(ix, y, initial-EPSILON*direction, arm, iq, anchor)
            numeric = (high-low)/(2*EPSILON)
            err = abs(numeric-float(grad @ direction))
            assert err <= FD_TOL, (arm, i, err)
            max_error = max(max_error, err)
            fd_checks += 1
    assert np.array_equal(first['LEARN'][2], first['MEAN'][2]) and first['LEARN'][0] == first['MEAN'][0]
    tail_norm = float(np.linalg.norm(first['LEARN'][1][176:]))
    assert tail_norm > 1e-8, 'fixture must exercise the learned matcher gradient'
    minimum_oracle_margin = float(np.min(independent_learned_data.oracle_margins(itest)))
    assert minimum_oracle_margin >= independent_learned_data.ORACLE_MARGIN_BOUND-1e-13
    assert memory()['peak_working_set_bytes'] < 1024**3 and time.perf_counter()-start < 120
    return rows, audit, dict(gradient_directions=fd_checks, max_fd_error=max_error, replay_checks=replay_checks,
                             stored_objective_checks=objective_checks, routing_checks=routing_checks, information_ceilings=ceilings,
                             minimum_oracle_margin=minimum_oracle_margin, initial_matcher_gradient_l2=tail_norm)


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
