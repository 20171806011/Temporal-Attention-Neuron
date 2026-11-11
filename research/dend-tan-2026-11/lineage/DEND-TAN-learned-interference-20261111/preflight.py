"""Verify source parity, crossed states, and continuation timing."""

import time

import numpy as np

import capacity_data
import capacity_model
import capacity_metrics
import independent_capacity_data
import independent_interference
import interference_model
import learned_model
import population
from run import ARMS, BLOCKS, CAP, CONDITIONS, ROOT, dump, peak_memory, verify_crossing


DATA_SEED = 4099902301
WEIGHT_SEED = 4099902302


def verify_fixture(n, start):
    base = capacity_data.make_base(n, DATA_SEED)
    ibase = independent_capacity_data.make_base(n, DATA_SEED)
    assert base.keys() == ibase.keys()
    assert all(np.array_equal(array, ibase[name]) for name, array in base.items())
    inputs = {}
    for condition in ('full4', 'base2'):
        x, q = capacity_data.assemble(base, condition)
        ix, iq = independent_capacity_data.assemble(ibase, condition)
        assert q == iq == 28 and np.array_equal(x, ix)
        info = independent_capacity_data.check(ibase, ix, condition)
        assert info['ceiling'] == (.75 if condition == 'base2' else 1.)
        inputs[condition] = x
    full_x, late_x = inputs['full4'], inputs['base2']
    assert np.array_equal(full_x[:, 3:], late_x[:, 3:])
    w = np.r_[population.initialize(WEIGHT_SEED, 6), np.array([.03, -.4, .45, -.02])]
    science, audit, checks, maximum = 0., 0., 0, 0.

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

    pairs = independent_capacity_data.pairs(ibase)
    for arm in ARMS:
        assert time.perf_counter()-start < 120
        tick = time.perf_counter()
        outcomes = interference_model.evaluate(full_x, late_x, w, arm, 28)
        science += time.perf_counter()-tick
        assert set(outcomes) == set(CONDITIONS)
        verify_crossing(outcomes)
        for condition, actual in outcomes.items():
            tick = time.perf_counter()
            expected = independent_interference.replay(full_x, late_x, w, arm, 28, condition)
            audit += time.perf_counter()-tick
            assert actual.keys() == expected.keys()
            for key, array in actual.items():
                if key in ('counts', 'spikes', 'resources'):
                    exact(array, expected[key], (arm, condition, key))
                else:
                    close(array, expected[key], (arm, condition, key))
            exact(actual['entry_native'][:, 20:], expected['entry_native'][:, 20:], (condition, 'entry spikes'))
            exact(actual['cue_read_vectors'][:8], actual['injections'][:, 25:28], (condition, 'cue injection'))
            exact(actual['cue_weights'][:8], actual['weights'][:, 25:28], (condition, 'cue weights'))
        for case, x, reference_case in (('full_full', full_x, 'full2'), ('late_late', late_x, 'base2')):
            source = learned_model.evaluate(x, w, arm, 28)
            for key, array in source.items():
                exact(outcomes[case][key], array, (arm, case, key, 'sealed source'))
            archived = capacity_model.evaluate(x, w, arm, 28, reference_case)
            for key in ('cue_scores', 'cue_read_vectors'):
                exact(outcomes[case][key], archived[key], (arm, case, key, 'sealed archive'))
            exact(outcomes[case]['entry_native'][:8, :20], source['states'][:, 24], (case, 'entry state'))
            exact(outcomes[case]['entry_native'][:8, 20:], source['spikes'][:, 24, :6], (case, 'entry pulse'))
            exact(outcomes[case]['entry_keys'][:8], source['key_bank'][:, 24], (case, 'entry keys'))
            exact(outcomes[case]['entry_values'][:8], source['value_bank'][:, 24], (case, 'entry values'))
        late = outcomes['late_late']
        exact(late['counts'][pairs[:, 0]], late['counts'][pairs[:, 1]], (arm, 'deleted labels'))
        scores = capacity_metrics.score(late, base)
        assert scores['older_ba'] == .5*scores['older_coverage']
    assert time.perf_counter()-start < 120 and peak_memory() < 1024**3
    return dict(examples=n, science_seconds=science, audit_seconds=audit,
                science_forecast_seconds=3*BLOCKS*science+30, audit_forecast_seconds=3*BLOCKS*audit+30,
                checks=checks, max_numeric_error=maximum, late_old_ceiling=.5,
                full_ceiling=1., natural_cases_verified=6, hybrid_cases_verified=6)


def main():
    assert not any((ROOT / name).exists() for name in ('PRECHECK.json', 'TIMING.json', 'FREEZE.json',
                                                       'RUN_STATE.json', 'evaluate'))
    start = time.perf_counter()
    candidates = []
    chosen = None
    try:
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
    dump('PRECHECK.json', dict(status='PASS', scientific_data_generated=False, state_source_fixture='PASS',
         data_seed=DATA_SEED, weight_seed=WEIGHT_SEED, candidates=candidates,
         peak_working_set_bytes=peak_memory(), wall_seconds=time.perf_counter()-start))
    dump('TIMING.json', dict(status='PASS' if chosen else 'INFEASIBLE', blocks=BLOCKS,
         examples_per_block=chosen, candidates=candidates, scientific_data_generated=False))
    if chosen is None:
        raise SystemExit('No allowed example count fits both time budgets')
    print(f'PRECHECK PASS: {candidates[-1]}', flush=True)


if __name__ == '__main__':
    main()
