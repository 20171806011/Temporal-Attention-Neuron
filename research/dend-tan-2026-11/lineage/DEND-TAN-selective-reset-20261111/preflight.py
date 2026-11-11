"""Verify read-onset reset rules, source parity, and workload before freezing."""

import time

import numpy as np

import capacity_data
import capacity_metrics
import independent_capacity_data
import independent_reset
import learned_model
import population
import reset_model
from run import ARMS, BLOCKS, CAP, ROOT, dump, peak_memory, verify_rules


DATA_SEED = 4099902401
WEIGHT_SEED = 4099902402
DISCRETE = ('counts', 'spikes', 'resources', 'read_onset', 'reset_steps')


def check_onsets(w):
    x = np.zeros((3, 36, 6))
    x[0, 1, 2] = x[0, 1, 5] = 1.
    x[0, 7:9, 2:4] = 1.
    x[0, 11:13, 2] = -1.
    x[1, 5:7, 2] = 1.
    x[2, 1, 3] = x[2, 1, 5] = 1.
    x[2, 25:28, 3] = 1.
    x[:, 28:, 4] = 1.
    identity = w.copy()
    identity[176:] = [1., 0., 0., 1.]
    expected_onsets = [[7, 11], [5], [25]]
    expected_resets = [[7], [], [25]]
    expected_confidence = np.full((3, 36), -2.)
    expected_confidence[0, 7] = 1./np.sqrt(2.)
    expected_confidence[0, 11] = -1.
    expected_confidence[2, 25] = 1.

    for rule in reset_model.RULES:
        actual = reset_model.evaluate(x, identity, 'LEARN', 28, rule)
        expected = independent_reset.replay(x, identity, 'LEARN', 28, rule)
        assert actual.keys() == expected.keys()
        for name, value in actual.items():
            if name in DISCRETE:
                np.testing.assert_array_equal(value, expected[name], err_msg=name)
            else:
                np.testing.assert_allclose(value, expected[name], rtol=0., atol=5e-12, err_msg=name)
        assert [np.flatnonzero(row).tolist() for row in actual['read_onset']] == expected_onsets
        triggers = {'original': [[], [], []], 'always': expected_onsets, 'selective': expected_resets}[rule]
        assert [np.flatnonzero(row).tolist() for row in actual['reset_steps']] == triggers
        confidence = expected_confidence if rule == 'selective' else np.full((3, 36), -2.)
        np.testing.assert_array_equal(actual['match_confidence'], confidence)
        if rule != 'original':
            np.testing.assert_array_equal(actual['cue_native_after'][2, 0], np.zeros(26))

    projected = np.array([[1., 1.], [0., 0.], [1., 0.]])
    keys = np.array([[[1., 0.], [0., 0.]], [[1., 0.], [0., 1.]], [[0., 0.], [0., 0.]]])
    target = np.array([1./np.sqrt(2.), -2., -2.])
    np.testing.assert_array_equal(reset_model.cosine_confidence(projected, keys), target)
    np.testing.assert_array_equal(independent_reset.matched_cosine(projected, keys), target)


def verify_fixture(n, start):
    base = capacity_data.make_base(n, DATA_SEED)
    ibase = independent_capacity_data.make_base(n, DATA_SEED)
    assert base.keys() == ibase.keys()
    assert all(np.array_equal(array, ibase[name]) for name, array in base.items())
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
    for history in ('full4', 'base2'):
        x, q = capacity_data.assemble(base, history)
        ix, iq = independent_capacity_data.assemble(ibase, history)
        assert q == iq == 28 and np.array_equal(x, ix)
        info = independent_capacity_data.check(ibase, ix, history)
        assert info['ceiling'] == (.75 if history == 'base2' else 1.)

        for arm in ARMS:
            cases = {}
            for rule in reset_model.RULES:
                assert time.perf_counter()-start < 120
                tick = time.perf_counter()
                actual = reset_model.evaluate(x, w, arm, q, rule)
                science += time.perf_counter()-tick
                tick = time.perf_counter()
                expected = independent_reset.replay(ix, w, arm, iq, rule)
                audit += time.perf_counter()-tick
                assert actual.keys() == expected.keys()
                for name, values in actual.items():
                    compare = exact if name in DISCRETE else close
                    compare(values, expected[name], (history, arm, rule, name))
                for name in ('cue_native_before', 'cue_native_after'):
                    exact(actual[name][..., 20:], expected[name][..., 20:], (history, arm, rule, name, 'p'))
                for name, field in (('cue_scores', 'scores'), ('cue_weights', 'weights'),
                                    ('cue_read_vectors', 'injections')):
                    exact(actual[name][:8], actual[field][:, 25:28], (history, arm, rule, name, 'cue archive'))
                if history == 'base2':
                    exact(actual['counts'][pairs[:, 0]], actual['counts'][pairs[:, 1]],
                          (arm, rule, 'deleted labels'))
                    scores = capacity_metrics.score(actual, base)
                    assert scores['older_ba'] == .5*scores['older_coverage']
                cases[rule] = actual
            verify_rules(cases, arm, base)
            for name, values in learned_model.evaluate(x, w, arm, q).items():
                exact(cases['original'][name], values, (history, arm, name, 'sealed source'))
    assert time.perf_counter()-start < 120 and peak_memory() < 1024**3
    return dict(examples=n, science_seconds=science, audit_seconds=audit,
                science_forecast_seconds=3*BLOCKS*science+30, audit_forecast_seconds=3*BLOCKS*audit+30,
                checks=checks, max_numeric_error=maximum, late_old_ceiling=.5, full_ceiling=1.,
                native_cases_verified=6, intervention_cases_verified=12)


def main():
    assert not any((ROOT / name).exists() for name in ('PRECHECK.json', 'TIMING.json', 'FREEZE.json',
                                                       'RUN_STATE.json', 'evaluate'))
    start = time.perf_counter()
    candidates = []
    chosen = None
    try:
        w = np.r_[population.initialize(WEIGHT_SEED, 6), np.zeros(4)]
        check_onsets(w)
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
    dump('PRECHECK.json', dict(status='PASS', scientific_data_generated=False, onset_fixture='PASS',
         data_seed=DATA_SEED, weight_seed=WEIGHT_SEED, candidates=candidates,
         peak_working_set_bytes=peak_memory(), wall_seconds=time.perf_counter()-start))
    dump('TIMING.json', dict(status='PASS' if chosen else 'INFEASIBLE', blocks=BLOCKS,
         examples_per_block=chosen, candidates=candidates, scientific_data_generated=False))
    if chosen is None:
        raise SystemExit('No allowed example count fits both time budgets')
    print(f'PRECHECK PASS: {candidates[-1]}', flush=True)


if __name__ == '__main__':
    main()
