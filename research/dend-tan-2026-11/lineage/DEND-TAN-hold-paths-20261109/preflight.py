"""Check the fixed intervention design on independent, small fixture data."""

from pathlib import Path
import time

import numpy as np

import core
import dynamics
import independent_data
import independent_replay
import model
import population
import reference
import task_data


ROOT = Path(__file__).resolve().parent
ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2')
DELAYS = (0, 16)
ATTENTION_ARMS = ('FULL', 'NOATT')
FIELDS = ('d', 'v', 'p', 'o')
CONTINUOUS_FIELDS = {'d', 'v', 'o', 'H', 'recurrence'}
STATE_GROUPS = ('branch', 'soma', 'previous', 'soma_p', 'output', 'other', 'full')
DATA_SEED = 4099901001
WEIGHT_SEED = 4099901002
ROWS = 32
ATOL = 2e-10
CUE_ATOL = 1e-12


def fixture_input(n, seed, delay):
    base = task_data.make_base(n, seed)
    expected = independent_data.make_base(n, seed)
    assert base.keys() == expected.keys()
    for key, value in base.items():
        np.testing.assert_array_equal(value, expected[key], err_msg=key)
    values, q = task_data.assemble(base, delay)
    assert q == 12 + delay and values.shape == (n, q + 8, 6)
    independent_data.check(base, values, delay)
    return base, values, q


def fixture_donors(base):
    table = independent_data.lookup(base)
    original = list(zip(base['octet'], base['selected'], base['first_bit'], base['second_bit']))
    assert len(table) == len(original)
    donors = [np.array([table[(octet, s ^ ds, b0 ^ d0, b1 ^ d1)]
                       for octet, s, b0, b1 in original])
              for ds, d0, d1 in ((0, 1, 0), (0, 0, 1), (1, 0, 0))]
    for bit, mapping in enumerate(donors[:2]):
        np.testing.assert_array_equal(base['y'][mapping] != base['y'], base['selected'] == bit)
        np.testing.assert_array_equal(mapping[donors[2]], donors[2][mapping])
    for mapping in donors:
        np.testing.assert_array_equal(mapping[mapping], np.arange(len(original)))
    return donors


def compare(actual, expected, continuous=CONTINUOUS_FIELDS):
    assert actual.keys() == expected.keys(), (actual.keys(), expected.keys())
    max_error = 0.
    for name, array in actual.items():
        if name in continuous:
            np.testing.assert_allclose(array, expected[name], atol=ATOL, rtol=0, err_msg=name)
            max_error = max(max_error, float(np.max(np.abs(array - expected[name]))))
        else:
            np.testing.assert_array_equal(array, expected[name], err_msg=name)
    return max_error


def check_cue_match(value, cue, discrete=False):
    if discrete:
        np.testing.assert_array_equal(value, value[cue])
    else:
        np.testing.assert_allclose(value, value[cue], atol=CUE_ATOL, rtol=0)


def check_native(snapshot, values, theta, arm, q, taps):
    native = model.evaluate(values, theta, arm, q)
    np.testing.assert_array_equal(snapshot['counts'], native['counts'])
    np.testing.assert_array_equal(snapshot['resources'][:, :4], native['resources'][:, :4])
    np.testing.assert_array_equal(snapshot['resources'][:, 4], np.zeros(len(values)))
    np.testing.assert_array_equal(snapshot['raster'][:8], native['spikes'][:, q:, 6:])
    np.testing.assert_array_equal(snapshot['counts'], snapshot['raster'].sum(axis=1))
    states = np.concatenate([snapshot[name][:8] for name in ('d', 'v', 'o')], axis=2)
    np.testing.assert_allclose(states, native['states'][:, taps], atol=ATOL, rtol=0)
    np.testing.assert_array_equal(snapshot['p'][:8], native['spikes'][:, taps, :6])
    assert np.all((snapshot['raster'] == 0) | (snapshot['raster'] == 1))


def check_scalar(snapshot, values, theta, arm, q, taps):
    scalar = reference.scalar_oracle(values[:2], theta, arm, q)
    states = np.concatenate([snapshot[name][:2] for name in ('d', 'v', 'o')], axis=2)
    np.testing.assert_allclose(states, scalar['states'][:, taps], atol=ATOL, rtol=0)
    np.testing.assert_allclose(snapshot['H'][:2], scalar['histories'][:, taps], atol=ATOL, rtol=0)
    np.testing.assert_array_equal(snapshot['p'][:2], scalar['spikes'][:, taps, :6])
    np.testing.assert_array_equal(snapshot['raster'][:2], scalar['spikes'][:, q:, 6:])


def independent_mean(recurrence, base):
    """Group using direct masks over the fixture rows, including same cue."""
    means = np.empty_like(recurrence)
    for octet in np.unique(base['octet']):
        for selected in (0, 1):
            mask = (base['octet'] == octet) & (base['selected'] == selected)
            assert mask.sum() == 4
            means[mask] = recurrence[mask].mean(axis=0)
    return means


def validate_states(values, theta, arm, q, native, independent, taps, state_taps, donors):
    first, second, cue = donors
    groups = STATE_GROUPS + (('history', 'branch_history') if arm in ATTENTION_ARMS else ())
    count = self_checks = full_checks = 0
    max_error = 0.
    state_fields = FIELDS + (('H',) if arm in ATTENTION_ARMS else ())
    for tap in state_taps:
        state = dynamics.state_at(native, taps, tap)
        for key in state_fields:
            check_cue_match(state[key], cue, discrete=key == 'p')
        expected_state = independent_replay.state_at(independent, taps, tap)
        max_error = max(max_error, compare(state, expected_state))
        resumed = dynamics.resume(values, theta, arm, q, tap, state)
        expected = independent_replay.resume(values, theta, arm, q, tap, expected_state)
        compare(resumed, expected)
        np.testing.assert_array_equal(resumed['raster'], native['raster'])
        np.testing.assert_array_equal(resumed['counts'], native['counts'])
        np.testing.assert_array_equal(resumed['resources'] + native['prefix_resources'][:, taps.index(tap)],
                                      native['resources'])
        count += 1
        self_checks += 1
        for donor in (first, second):
            np.testing.assert_array_equal(values[:, tap+1:], values[donor, tap+1:])
            for group in groups:
                mixed, distance = dynamics.splice(native, taps, tap, group, donor)
                expected_mixed, expected_distance = independent_replay.splice(independent, taps, tap, group, donor)
                max_error = max(max_error, compare(mixed, expected_mixed))
                check_cue_match(distance, cue)
                np.testing.assert_allclose(distance, expected_distance, atol=ATOL, rtol=0)
                for name, array in mixed.items():
                    check_cue_match(array, cue, discrete=name == 'p')
                resumed = dynamics.resume(values, theta, arm, q, tap, mixed)
                expected = independent_replay.resume(values, theta, arm, q, tap, expected_mixed)
                compare(resumed, expected)
                np.testing.assert_array_equal(resumed['counts'], resumed['raster'].sum(axis=1))
                np.testing.assert_array_equal(resumed['resources'][:, 4], np.zeros(len(values)))
                if group == 'full':
                    np.testing.assert_array_equal(resumed['raster'], native['raster'][donor])
                    np.testing.assert_array_equal(resumed['counts'], native['counts'][donor])
                    full_checks += 1
                count += 1
    return count, self_checks, full_checks, max_error


def validate_routes(values, theta, arm, q, native, independent, taps, donors, base):
    first, second, cue = donors
    recurrence = native['recurrence']
    expected_recurrence = independent['recurrence']
    mean = dynamics.mean_recurrence(recurrence, base)
    expected_mean = independent_mean(expected_recurrence, base)
    np.testing.assert_allclose(mean, expected_mean, atol=ATOL, rtol=0)
    for archive in (recurrence, mean):
        check_cue_match(archive[:, 9:25], cue)
    windows = ((recurrence, expected_recurrence, 9, 25),
               (mean, expected_mean, 9, 25),
               (recurrence[first], expected_recurrence[first], 9, 25),
               (recurrence[second], expected_recurrence[second], 9, 25),
               (mean, expected_mean, 9, 17),
               (mean, expected_mean, 17, 25),
               (mean, expected_mean, q, q+8))
    for index, (current, expected_current, start, stop) in enumerate(windows):
        tap = start - 1
        state = dynamics.state_at(native, taps, tap)
        expected_state = independent_replay.state_at(independent, taps, tap)
        resumed = dynamics.resume(values, theta, arm, q, tap, state, current, (start, stop))
        expected = independent_replay.resume(values, theta, arm, q, tap, expected_state,
                                              expected_current, (start, stop))
        compare(resumed, expected)
        np.testing.assert_array_equal(resumed['counts'], resumed['raster'].sum(axis=1))
        np.testing.assert_array_equal(resumed['resources'][:, 4], np.full(len(values), 12*(stop-start)))
        if index == 0:
            np.testing.assert_array_equal(resumed['raster'], native['raster'])
            np.testing.assert_array_equal(resumed['counts'], native['counts'])
    return len(windows)


def run_checks():
    lock = core.read(ROOT / 'SOURCE_LOCK.json')
    for name, digest in lock['files'].items():
        assert core.sha(ROOT / name) == digest, name
    native_checks = state_checks = recurrence_checks = self_checks = full_checks = scalar_episodes = 0
    max_error = 0.
    theta = population.initialize(WEIGHT_SEED)
    for delay in DELAYS:
        base, values, q = fixture_input(ROWS, DATA_SEED, delay)
        donors = fixture_donors(base)
        for actual, expected in zip(dynamics.donors(base), donors):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(values[:, :q-3], values[donors[2], :q-3])
        state_taps = (8,) if delay == 0 else (8, 16, 24)
        taps = state_taps if delay == 0 else (*state_taps, 27)
        for arm in ARMS:
            native = dynamics.capture(values, theta, arm, q, taps, save_recurrence=bool(delay))
            independent = independent_replay.capture(values, theta, arm, q, taps, save_recurrence=bool(delay))
            check_native(native, values, theta, arm, q, taps)
            max_error = max(max_error, compare(native, independent))
            native_checks += 1
            if arm in ATTENTION_ARMS:
                check_scalar(native, values, theta, arm, q, taps)
                scalar_episodes += 2
            cases, self_count, full_count, error = validate_states(
                values, theta, arm, q, native, independent, taps, state_taps, donors)
            state_checks += cases
            self_checks += self_count
            full_checks += full_count
            max_error = max(max_error, error)
            if delay:
                recurrence_checks += validate_routes(values, theta, arm, q, native, independent, taps, donors, base)
    assert (native_checks, state_checks, recurrence_checks, self_checks, full_checks, scalar_episodes) == (8, 272, 28, 16, 32, 8)
    return dict(fixture_rows=ROWS, fixture_data_seed=DATA_SEED, fixture_weight_seed=WEIGHT_SEED,
                arms=list(ARMS), delays=list(DELAYS), native_captures=native_checks,
                state_cases=state_checks, recurrence_cases=recurrence_checks,
                state_self_checks=self_checks, complete_donor_checks=full_checks, recurrence_self_checks=4,
                scalar_tan_episodes=scalar_episodes, max_continuous_error=max_error,
                cue_matching_atol=CUE_ATOL, scientific_seeds_used=False)


def main():
    assert not (ROOT / 'FREEZE.json').exists(), 'frozen artifacts are read-only'
    assert not (ROOT / 'PRECHECK.json').exists(), 'retain previous preflight before another invocation'
    start = time.perf_counter()
    try:
        checks = run_checks()
    except Exception as error:
        core.dump(ROOT / 'PRECHECK.json', dict(status='FAIL', reason=str(error), exception=type(error).__name__,
                                              wall_seconds=time.perf_counter()-start))
        raise
    core.dump(ROOT / 'PRECHECK.json', dict(status='PASS', wall_seconds=time.perf_counter()-start, **checks))
    print('PRECHECK PASS', checks, flush=True)


if __name__ == '__main__':
    main()
