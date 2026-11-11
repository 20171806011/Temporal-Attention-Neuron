"""Verify matched-norm mechanics against source and independent replay."""

import time

import numpy as np

import amplitude_model
import binding_data
import independent_amplitude
import independent_binding_data
import key_model
import population
from run import ARMS, BLOCKS, MATCHED, ROOT, TOLERANCE, dump, peak_memory, verify_key_conditions


DATA_SEED = 4099902001
WEIGHT_SEED = 4099902002


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
    for operation in (amplitude_model._rescale, independent_amplitude.match_norm):
        observed, flags = operation(direction, amplitude)
        np.testing.assert_allclose(observed, expected, rtol=0, atol=TOLERANCE)
        assert np.array_equal(flags, expected_flags)
        np.testing.assert_allclose(np.linalg.norm(observed, axis=1), np.linalg.norm(amplitude, axis=1),
                                   rtol=0, atol=TOLERANCE)


def verify_fixture(n, start):
    base = binding_data.make_base(n, DATA_SEED)
    ibase = independent_binding_data.make_base(n, DATA_SEED)
    assert base.keys() == ibase.keys()
    assert all(np.array_equal(array, ibase[name]) for name, array in base.items())
    x, q = binding_data.assemble(base)
    ix, iq = independent_binding_data.assemble(ibase)
    assert q == iq == 28 and np.array_equal(x, ix) and not x[:, 9:25].any()
    assert independent_binding_data.check(ibase, ix)['ceiling'] == 1.
    w = population.initialize(WEIGHT_SEED, 6)
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

    for arm in ARMS:
        outcomes = {}
        for condition in amplitude_model.conditions(arm):
            assert time.perf_counter()-start < 120
            tick = time.perf_counter()
            result = amplitude_model.evaluate(x, w, arm, q, condition)
            science += time.perf_counter()-tick
            tick = time.perf_counter()
            ref = independent_amplitude.replay(ix, w, arm, iq, condition)
            audit += time.perf_counter()-tick
            assert result.keys() == ref.keys()
            for name, values in result.items():
                if name in ('counts', 'spikes', 'resources', 'direction_fallback'):
                    exact(values, ref[name], (arm, condition, name))
                else:
                    close(values, ref[name], (arm, condition, name))
            exact(result['read_vectors'][:8], result['injections'][:, 25:28], (condition, 'cue injections'))
            if 'cue_memory' in result:
                exact(result['cue_memory'][:8], result['memory'][:, 24:27], (condition, 'old slots'))
            if condition in ('nominal', 'swap_read', 'mean_read'):
                source_arm = 'KEY_MEAN' if condition == 'mean_read' else arm
                source = key_model.evaluate(x, w, source_arm, q, swap_read=condition == 'swap_read')
                for name, values in source.items():
                    exact(values, result[name], (arm, condition, name, 'source'))
            outcomes[condition] = result
        if arm == 'KEY':
            verify_key_conditions(outcomes)
            direction_source = {'swap_direction': 'swap_read', 'swap_magnitude': 'nominal', 'mean_matched': 'mean_read'}
            for condition, target in MATCHED.items():
                values = outcomes[condition]['read_vectors']
                target_values = outcomes[target]['read_vectors']
                direction = outcomes[direction_source[condition]]['read_vectors']
                wanted, flags = independent_amplitude.match_norm(direction.reshape(-1, 12),
                                                                  target_values.reshape(-1, 12))
                close(values.reshape(-1, 12), wanted, (condition, 'norm-matched injection'))
                exact(outcomes[condition]['direction_fallback'].ravel(), flags, (condition, 'fallback'))
                residual = np.abs(np.linalg.norm(values, axis=-1)-np.linalg.norm(target_values, axis=-1))
                max_norm_residual = max(max_norm_residual, float(residual.max()))
    assert time.perf_counter()-start < 120 and peak_memory() < 1024**3
    return dict(examples=n, science_seconds=science, audit_seconds=audit,
                science_forecast_seconds=3*BLOCKS*science+30, audit_forecast_seconds=3*BLOCKS*audit+30,
                checks=checks, max_numeric_error=maximum, max_matched_norm_residual=max_norm_residual)


def main():
    assert not any((ROOT / name).exists() for name in ('PRECHECK.json', 'TIMING.json', 'FREEZE.json',
                                                       'RUN_STATE.json', 'evaluate'))
    start = time.perf_counter()
    candidates = []
    chosen = None
    try:
        check_zero_direction()
        for n in (256, 128):
            result = verify_fixture(n, start)
            candidates.append(result)
            if result['science_forecast_seconds'] < 840 and result['audit_forecast_seconds'] < 300:
                chosen = n
                break
    except Exception as error:
        attempt = 1+len(list(ROOT.glob('PRECHECK_FAILURE_*.json')))
        dump(f'PRECHECK_FAILURE_{attempt}.json', dict(status='FAIL', reason=str(error),
             exception=type(error).__name__, scientific_data_generated=False))
        raise
    dump('PRECHECK.json', dict(status='PASS', scientific_data_generated=False, zero_direction_fixture='PASS',
         data_seed=DATA_SEED, weight_seed=WEIGHT_SEED, candidates=candidates,
         peak_working_set_bytes=peak_memory(), wall_seconds=time.perf_counter()-start))
    dump('TIMING.json', dict(status='PASS' if chosen else 'INFEASIBLE', blocks=BLOCKS,
         examples_per_block=chosen, candidates=candidates, scientific_data_generated=False))
    if chosen is None:
        raise SystemExit('No allowed example count fits both time budgets')
    print(f'PRECHECK PASS: {candidates[-1]}', flush=True)


if __name__ == '__main__':
    main()
