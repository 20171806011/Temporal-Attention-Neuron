"""Verify source parity, four-slot FIFO, and read interventions."""

import time

import numpy as np

import capacity_data
import capacity_model
import capacity_metrics
import independent_capacity
import independent_capacity_data
import learned_data
import learned_model
import population
from run import ARMS, BLOCKS, CAP, MATCHED, ROOT, TOLERANCE, dump, peak_memory, verify_variants


DATA_SEED = 4099902201
WEIGHT_SEED = 4099902202


def check_zero_direction():
    direction = np.zeros((4, 12))
    amplitude = np.zeros_like(direction)
    direction[0, :2] = [3., 4.]
    amplitude[0, 2] = 12.
    amplitude[1, 1] = -4.
    direction[2, 1] = -3.
    expected = np.zeros_like(direction)
    expected[0, :2] = [7.2, 9.6]
    expected[1, 0] = 4.
    expected_flags = [False, True, False, False]
    for operation in (capacity_model.match_norm, independent_capacity.matched_injection):
        observed, flags = operation(direction, amplitude)
        np.testing.assert_allclose(observed, expected, rtol=0, atol=TOLERANCE)
        assert np.array_equal(flags, expected_flags)
        np.testing.assert_allclose(np.linalg.norm(observed, axis=1), np.linalg.norm(amplitude, axis=1),
                                   rtol=0, atol=TOLERANCE)


def check_ties():
    scores = np.array([[2., 2., 0., -1.], [0., 0., 0., 0.], [0., 1., 2., 3.]])
    expected = np.array([[.5, .5, 0., 0.], [.25, .25, .25, .25], [0., 0., 0., 1.]])
    actual, soft = capacity_model.read_weights(scores, True)
    assert np.array_equal(actual, expected)
    assert np.array_equal(independent_capacity.hard_weights(scores), expected)
    np.testing.assert_allclose(soft.sum(axis=1), 1., rtol=0, atol=5e-15)


def check_fifo(x, raw, slots):
    expected_keys = np.zeros_like(raw['key_bank'])
    expected_values = np.zeros_like(raw['value_bank'])
    for row in range(8):
        writes = np.flatnonzero(x[row, :, 5] == 1.)
        for tick in range(x.shape[1]):
            for slot, source in enumerate(writes[writes <= tick][-slots:][::-1]):
                expected_keys[row, tick, slot] = x[row, source, 2:4]
                expected_values[row, tick, :, slot] = .97**(tick-source)*raw['drive'][row, source]
    assert np.array_equal(raw['key_bank'], expected_keys)
    np.testing.assert_allclose(raw['value_bank'], expected_values, rtol=0, atol=5e-12)


def verify_fixture(n, start):
    base = capacity_data.make_base(n, DATA_SEED)
    ibase = independent_capacity_data.make_base(n, DATA_SEED)
    assert base.keys() == ibase.keys()
    assert all(np.array_equal(array, ibase[name]) for name, array in base.items())
    inputs = {}
    minimum_margin = None
    for condition in ('base2', 'full4'):
        x, q = capacity_data.assemble(base, condition)
        ix, iq = independent_capacity_data.assemble(ibase, condition)
        assert q == iq == 28 and np.array_equal(x, ix)
        checked = independent_capacity_data.check(ibase, ix, condition)
        assert checked['ceiling'] == (.75 if condition == 'base2' else 1.)
        minimum_margin = checked['oracle_margin_min']
        inputs[condition] = x, q
    old_x, old_q = learned_data.assemble(learned_data.make_base(64, DATA_SEED))
    w = np.r_[population.initialize(WEIGHT_SEED, 6), np.array([.03, -.4, .45, -.02])]
    science, audit, checks, maximum, max_norm_residual = 0., 0., 0, 0., 0.

    def exact(a, b, context):
        nonlocal checks
        assert np.array_equal(a, b), context
        checks += 1

    def close(a, b, context):
        nonlocal checks, maximum
        assert a.shape == b.shape, context
        error = float(np.max(np.abs(a-b)))
        assert error <= 5e-12, (context, error)
        maximum = max(maximum, error)
        checks += 1

    pairs = independent_capacity_data.pairs(base)
    for arm in ARMS:
        source = learned_model.evaluate(old_x, w, arm, old_q)
        general = capacity_model.evaluate(old_x, w, arm, old_q, 'full2')
        for key, value in source.items():
            exact(value, general[key], (arm, key, 'sealed two-slot source'))
        outcomes = {}
        for condition in capacity_model.conditions(arm):
            assert time.perf_counter()-start < 120
            x, q = inputs['base2' if condition == 'base2' else 'full4']
            tick = time.perf_counter()
            result = capacity_model.evaluate(x, w, arm, q, condition)
            science += time.perf_counter()-tick
            tick = time.perf_counter()
            ref = independent_capacity.replay(x, w, arm, q, condition)
            audit += time.perf_counter()-tick
            assert result.keys() == ref.keys()
            for name, value in result.items():
                if name in ('counts', 'spikes', 'resources', 'cue_fallback'):
                    exact(value, ref[name], (arm, condition, name))
                else:
                    close(value, ref[name], (arm, condition, name))
            slots = result['cue_weights'].shape[-1]
            check_fifo(x, result, slots)
            exact(result['cue_read_vectors'][:8], result['injections'][:, 25:28], (condition, 'cue injections'))
            exact(result['cue_weights'][:8], result['weights'][:, 25:28], (condition, 'cue weights'))
            scores = capacity_metrics.score(result, base)
            if condition == 'base2':
                exact(result['counts'][pairs[:, 0]], result['counts'][pairs[:, 1]], (arm, 'duplicate base2'))
                assert scores['older_ba'] == .5*scores['older_coverage']
            if arm == 'ORACLE':
                assert scores['recent_routing_accuracy'] == 1.
                if slots == 4:
                    assert scores['routing_accuracy'] == 1.
            outcomes[condition] = result
        verify_variants(outcomes)
        if MATCHED in outcomes:
            norm_a = np.linalg.norm(outcomes[MATCHED]['cue_read_vectors'], axis=-1)
            norm_b = np.linalg.norm(outcomes['full4']['cue_read_vectors'], axis=-1)
            max_norm_residual = float(np.max(np.abs(norm_a-norm_b)))
    assert time.perf_counter()-start < 120 and peak_memory() < 1024**3
    return dict(examples=n, science_seconds=science, audit_seconds=audit,
                science_forecast_seconds=3*BLOCKS*science+30, audit_forecast_seconds=3*BLOCKS*audit+30,
                checks=checks, max_numeric_error=maximum, max_matched_norm_residual=max_norm_residual,
                base2_full_ceiling=.75, base2_older_ceiling=.5, full4_ceiling=1.,
                oracle_margin_min=minimum_margin)


def main():
    assert not any((ROOT / name).exists() for name in ('PRECHECK.json', 'TIMING.json', 'FREEZE.json',
                                                       'RUN_STATE.json', 'evaluate'))
    start = time.perf_counter()
    candidates = []
    chosen = None
    try:
        check_zero_direction()
        check_ties()
        for n in (512, 256):
            result = verify_fixture(n, start)
            candidates.append(result)
            if result['science_forecast_seconds'] < CAP and result['audit_forecast_seconds'] < 300:
                chosen = n
                break
    except Exception as error:
        attempt = 1+len(list(ROOT.glob('PRECHECK_FAILURE_*.json')))
        dump(f'PRECHECK_FAILURE_{attempt}.json', dict(status='FAIL', reason=str(error),
             exception=type(error).__name__, scientific_data_generated=False))
        raise
    dump('PRECHECK.json', dict(status='PASS', scientific_data_generated=False, zero_direction_fixture='PASS', tie_fixture='PASS',
         data_seed=DATA_SEED, weight_seed=WEIGHT_SEED, candidates=candidates,
         peak_working_set_bytes=peak_memory(), wall_seconds=time.perf_counter()-start))
    dump('TIMING.json', dict(status='PASS' if chosen else 'INFEASIBLE', blocks=BLOCKS,
         examples_per_block=chosen, candidates=candidates, scientific_data_generated=False))
    if chosen is None:
        raise SystemExit('No allowed example count fits both time budgets')
    print(f'PRECHECK PASS: {candidates[-1]}', flush=True)


if __name__ == '__main__':
    main()
