"""Check evicted-key competition, independent replay, and full block timing."""

import time

import numpy as np

import capacity_data
import guard_model
import independent_capacity_data
import independent_event
import independent_guard
import independent_warm
import population
from run import BLOCKS, CAP, ROOT, assemble_histories, dump, evaluate_cases, measure_single, peak_memory, rule_order


DATA_SEED, WEIGHT_SEED = 4099903001, 4099903002
GEOMETRIES, HISTORIES, RULES = independent_guard.GEOMETRIES, independent_guard.HISTORIES, independent_guard.RULES
TOLERANCE = 5e-12
EXACT = ('counts', 'spikes', 'resources', 'read_onset', 'reset_steps', 'eviction_seen',
         'logged_write_values', 'logged_write_ticks', 'logged_nonwrite_values', 'logged_nonwrite_ticks',
         'logged_nonwrite_count', 'first_read_tick', 'replay_steps', 'replay_resources',
         'evicted_keys', 'evicted_comparisons')


def compare(actual, expected, context):
    assert actual.keys() == expected.keys(), context
    maximum = 0.
    checks = 0
    for name, value in actual.items():
        wanted = expected[name]
        assert value.shape == wanted.shape, (context, name)
        if name in EXACT:
            np.testing.assert_array_equal(value, wanted, err_msg=f'{context} {name}')
        else:
            error = float(np.max(np.abs(value-wanted)))
            assert error <= TOLERANCE, (context, name, error)
            maximum = max(maximum, error)
        if name in ('cue_native_before', 'cue_native_after', 'replay_reference'):
            np.testing.assert_array_equal(value[..., 20:], wanted[..., 20:], err_msg=f'{context} {name} pulses')
            checks += 1
        checks += 1
    return checks, maximum


def check_guard_edges(w):
    x = np.zeros((8, 36, 6))
    e1, e2, zero = (1., 0.), (0., 1.), (0., 0.)
    negative = (-1., 0.)
    histories = ((e1, e2, e2), (negative, e2, e1, e1), (e1, zero, e2, e2),
                 (e1, e1, e1), (e1, e2, negative, e1, e2), (), ((1., 1.), e2, e2),
                 (e1, e2, e2))
    for row, keys in enumerate(histories):
        for tick, key in zip((0, 2, 4, 8, 9), keys):
            x[row, tick] = [1., 0., *key, 0., 1.]
    x[0, 6, 2:4] = 1.
    x[1, 9:11, 2:4] = e1
    x[1, 11] = [1., 0., *e2, 0., 1.]
    x[1, 12, 2:4] = e1
    x[2, 9, 2:4] = e2
    x[3, 6, 2:4] = negative
    x[3, 10, 2:4] = e1
    x[4, 10, 2:4] = e2
    x[5, 7, 2:4] = e1
    x[6, 6, 2:4] = 1.
    x[7, 1, 2:4] = e1
    x[7, 10, 2:4] = e2
    identity = w.copy()
    identity[176:] = [1., 0., 0., 1.]
    raw, log, timing = guard_model.evaluate(x, identity, 28, 'guarded')
    old, _, _ = guard_model.evaluate(x, identity, 28, 'replay')
    checks, maximum = compare({**raw, **log}, independent_guard.replay(x, identity, 28), 'edge fixture')
    expected_keys = np.array([(zero, e1), (negative, e2), (zero, e1), (zero, e1),
                              (e2, negative), (zero, zero), (zero, (1., 1.)), (zero, zero)])
    np.testing.assert_array_equal(log['evicted_keys'], expected_keys)
    np.testing.assert_array_equal(log['first_read_tick'], [6, 9, 9, 6, 10, 7, 6, 1])
    assert [np.flatnonzero(row).tolist() for row in old['reset_steps']] == [[6], [9], [], [], [10], [], [6], []]
    assert [np.flatnonzero(row).tolist() for row in raw['reset_steps']] == [[], [9], [], [], [], [], [], []]
    assert raw['read_onset'][1, 12] and raw['read_onset'][3, 10] and raw['read_onset'][7, 10]
    assert raw['match_confidence'][0, 6] == log['evicted_cosine'][0, 6] == 1./np.sqrt(2.)
    assert raw['match_confidence'][4, 10] == log['evicted_cosine'][4, 10] == 1.
    assert np.all(log['evicted_cosine'][~old['reset_steps']] == -2.)
    np.testing.assert_array_equal(log['evicted_comparisons'].sum(axis=1), [1, 2, 0, 0, 2, 0, 1, 0])
    assert timing['guard_batches'] == 3 and timing['replay_batches'] == 1

    projected = np.array([e1, zero, e1, e1])
    keys = np.array([(zero, zero), (e1, e2), (zero, e2), (e1, negative)])
    cosine, counts = guard_model.evicted_confidence(projected, keys)
    np.testing.assert_array_equal(cosine, [-2., -2., 0., 1.])
    np.testing.assert_array_equal(counts, [0, 0, 1, 2])
    return dict(checks=checks, max_numeric_error=maximum, expected_guard_batches=3,
                expected_replay_batches=1, tie_cases=2)


def check_block(n, w):
    tick = time.perf_counter()
    base = capacity_data.make_base(n, DATA_SEED)
    inputs, q = assemble_histories(base)
    order = rule_order(0)
    cases = evaluate_cases(base, inputs, q, w, order)
    isolated, isolated_steps, isolated_macs, isolated_compares = measure_single(0, inputs, q, w, base, order, cases, tick)
    science_seconds = time.perf_counter()-tick

    tick = time.perf_counter()
    ibase = independent_capacity_data.make_base(n, DATA_SEED)
    assert base.keys() == ibase.keys()
    assert all(np.array_equal(values, ibase[name]) for name, values in base.items())
    assert set(cases) == {f'{geometry}_{history}_{rule}' for geometry in GEOMETRIES for history in HISTORIES for rule in RULES}
    checks, maximum = 0, 0.
    triggered, vetoed, compared = 0, 0, 0
    for geometry in GEOMETRIES:
        expected_base = independent_guard.geometry(ibase, geometry)
        for history in HISTORIES:
            prefix = f'{geometry}_{history}'
            x, iq = independent_guard.assemble(expected_base, history)
            assert q == iq == 28 and np.array_equal(inputs[prefix], x), prefix
            original = independent_warm.replay(x, w, 'LEARN', q, 'original')
            replay = independent_event.replay(x, w, q)
            guarded = independent_guard.apply_guard(x, w, replay, original)
            for rule, expected in (('original', original), ('replay', replay), ('guarded', guarded)):
                raw, diag, _ = cases[f'{prefix}_{rule}']
                count, residual = compare({**raw, **diag}, expected, f'{prefix}_{rule}')
                checks += count
                maximum = max(maximum, residual)
            original, _, _ = cases[f'{prefix}_original']
            replay, replay_diag, replay_timing = cases[f'{prefix}_replay']
            guarded, guard_diag, guard_timing = cases[f'{prefix}_guarded']
            old_mask, new_mask = replay['reset_steps'], guarded['reset_steps']
            assert not np.any(new_mask & ~old_mask)
            assert np.array_equal(guarded['match_confidence'], replay['match_confidence'])
            assert np.all(guard_diag['evicted_cosine'][~old_mask] == -2.)
            assert np.all(guard_diag['evicted_comparisons'][~old_mask] == 0)
            assert np.array_equal(new_mask, old_mask & (replay['match_confidence'] > guard_diag['evicted_cosine']))
            assert guard_timing['guard_batches'] == replay_timing['replay_batches']
            rejected = (old_mask & ~new_mask).any(axis=1)
            unchanged = ~rejected
            for field in ('counts', 'resources', 'spikes'):
                np.testing.assert_array_equal(guarded[field][rejected[:len(guarded[field])]],
                                              original[field][rejected[:len(guarded[field])]])
                np.testing.assert_array_equal(guarded[field][unchanged[:len(guarded[field])]],
                                              replay[field][unchanged[:len(guarded[field])]])
            for field in ('replay_steps', 'replay_resources'):
                np.testing.assert_array_equal(guard_diag[field][unchanged], replay_diag[field][unchanged])
                assert not guard_diag[field][rejected].any()
            if history == 'two':
                assert not guard_diag['evicted_keys'].any()
                for field in original:
                    np.testing.assert_array_equal(guarded[field], original[field], err_msg=f'{prefix} {field}')
                    np.testing.assert_array_equal(replay[field], original[field], err_msg=f'{prefix} {field}')
            triggered += int(old_mask.sum())
            vetoed += int(rejected.sum())
            compared += int(guard_diag['evicted_comparisons'].sum())

    audit_seconds = time.perf_counter()-tick
    assert peak_memory() < 1024**3
    return dict(examples=n, science_seconds=science_seconds, audit_seconds=audit_seconds,
                science_forecast_seconds=3*BLOCKS*science_seconds+30, audit_forecast_seconds=3*BLOCKS*audit_seconds+30,
                checks=checks, max_numeric_error=maximum, paths_verified=len(cases), old_triggers=triggered,
                vetoes=vetoed, evicted_comparisons=compared, isolated_records=len(isolated),
                isolated_replay_steps=isolated_steps, isolated_replay_macs=isolated_macs,
                isolated_evicted_comparisons=isolated_compares)


def main():
    assert not any((ROOT / name).exists() for name in ('PRECHECK.json', 'TIMING.json', 'FREEZE.json',
                                                       'RUN_STATE.json', 'evaluate'))
    started = time.perf_counter()
    candidates = []
    chosen = None
    try:
        w = np.r_[population.initialize(WEIGHT_SEED, 6), [.03, -.4, .45, -.02]]
        edge = check_guard_edges(w)
        for n in (512, 256):
            result = check_block(n, w)
            candidates.append(result)
            if result['science_forecast_seconds'] < CAP and result['audit_forecast_seconds'] < 300:
                chosen = n
                break
    except Exception as error:
        attempt = 1+len(list(ROOT.glob('PRECHECK_FAILURE_*.json')))
        dump(f'PRECHECK_FAILURE_{attempt}.json', dict(status='FAIL', reason=str(error),
             exception=type(error).__name__, scientific_data_generated=False))
        raise
    dump('PRECHECK.json', dict(status='PASS' if chosen else 'INFEASIBLE', scientific_data_generated=False,
         edge_fixture=edge, data_seed=DATA_SEED, weight_seed=WEIGHT_SEED, candidates=candidates,
         peak_working_set_bytes=peak_memory(), wall_seconds=time.perf_counter()-started))
    dump('TIMING.json', dict(status='PASS' if chosen else 'INFEASIBLE', blocks=BLOCKS,
         examples_per_block=chosen, candidates=candidates, scientific_data_generated=False))
    if chosen is None:
        raise SystemExit('No allowed evicted-key fixture fits both budgets')
    print(f'PRECHECK PASS: {candidates[-1]}', flush=True)


if __name__ == '__main__':
    main()
