"""Verify eviction, reference replacement, source parity, and workload."""

import time

import numpy as np

import capacity_data
import capacity_metrics
import independent_capacity_data
import independent_warm
import learned_model
import population
import reset_model
import warm_model
from run import ARMS, BLOCKS, CAP, ROOT, dump, peak_memory, verify_rules


DATA_SEED = 4099902501
WEIGHT_SEED = 4099902502
STANDARD_RULES = ('original', 'match_zero', 'eviction_zero', 'eviction_warm')
DISCRETE = ('counts', 'spikes', 'resources', 'read_onset', 'reset_steps', 'eviction_seen')


def check_onsets(w):
    x = np.zeros((4, 36, 6))
    x[0, 1:4, 2] = x[0, 1:4, 5] = 1.
    x[0, 7:9, 2:4] = 1.
    x[0, 11:13, 2] = -1.
    x[1, 1:3, 2] = x[1, 1:3, 5] = 1.
    x[1, 5:7, 2] = 1.
    x[2, 5:7, 2] = 1.
    x[2, 7:10, 5] = 1.
    x[2, 8:10, 2] = 1.
    x[2, 15:17, 2] = 1.
    x[3, 1:4, 3] = x[3, 1:4, 5] = 1.
    x[3, 25:28, 3] = 1.
    x[:, 28:, 4] = 1.
    identity = w.copy()
    identity[176:] = [1., 0., 0., 1.]
    warm = warm_model.warm_reference(identity)
    paired = np.repeat(warm[None, :], len(x), axis=0)
    paired[:, 0] += np.arange(len(x)) * .25
    expected_onsets = [[7, 11], [5], [5, 15], [25]]
    expected_eviction = np.zeros((len(x), 36), dtype=bool)
    expected_eviction[[0, 3], 4:] = True
    expected_evicted_resets = [[7], [], [], [25]]
    expected_match_resets = [[7], [5], [15], [25]]
    threshold = 1./np.sqrt(2.)
    for rule in warm_model.RULES:
        reference = {'eviction_warm': warm, 'eviction_late_native': paired}.get(rule)
        actual = warm_model.evaluate(x, identity, 'LEARN', 28, rule, reference)
        expected = independent_warm.replay(x, identity, 'LEARN', 28, rule, reference)
        assert actual.keys() == expected.keys()
        for name, values in actual.items():
            if name in DISCRETE:
                np.testing.assert_array_equal(values, expected[name], err_msg=name)
            else:
                np.testing.assert_allclose(values, expected[name], rtol=0., atol=5e-12, err_msg=name)
        assert [np.flatnonzero(row).tolist() for row in actual['read_onset']] == expected_onsets
        np.testing.assert_array_equal(actual['eviction_seen'], expected_eviction)
        triggers = {'original': [[], [], [], []], 'match_zero': expected_match_resets}.get(
            rule, expected_evicted_resets)
        assert [np.flatnonzero(row).tolist() for row in actual['reset_steps']] == triggers
        confidence = np.full((len(x), 36), -2.)
        if rule != 'original':
            confidence[0, 7], confidence[0, 11], confidence[3, 25] = threshold, -1., 1.
        if rule == 'match_zero':
            confidence[1, 5] = confidence[2, 15] = 1.
        np.testing.assert_array_equal(actual['match_confidence'], confidence)
        if rule != 'original':
            replacement = paired[3] if rule == 'eviction_late_native' else (
                warm if rule == 'eviction_warm' else np.zeros(26))
            np.testing.assert_array_equal(actual['cue_native_after'][3, 0], replacement)

    projected = np.array([[1., 1.], [0., 0.], [1., 0.]])
    keys = np.array([[[1., 0.], [0., 0.]], [[1., 0.], [0., 1.]], [[0., 0.], [0., 0.]]])
    target = np.array([threshold, -2., -2.])
    np.testing.assert_array_equal(warm_model.cosine_confidence(projected, keys), target)
    np.testing.assert_array_equal(independent_warm.matched_cosine(projected, keys), target)


def verify_fixture(n, start):
    base = capacity_data.make_base(n, DATA_SEED)
    ibase = independent_capacity_data.make_base(n, DATA_SEED)
    assert base.keys() == ibase.keys()
    assert all(np.array_equal(array, ibase[name]) for name, array in base.items())
    inputs = {}
    for history, condition in (('full', 'full4'), ('late', 'base2')):
        x, q = capacity_data.assemble(base, condition)
        ix, iq = independent_capacity_data.assemble(ibase, condition)
        assert q == iq == 28 and np.array_equal(x, ix)
        info = independent_capacity_data.check(ibase, ix, condition)
        assert info['ceiling'] == (.75 if history == 'late' else 1.)
        inputs[history] = x, ix
    assert np.array_equal(inputs['full'][0][:, 3:], inputs['late'][0][:, 3:])
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
        tick = time.perf_counter()
        warm = warm_model.warm_reference(w)
        science += time.perf_counter()-tick
        tick = time.perf_counter()
        iwarm = independent_warm.warm_reference(w)
        audit += time.perf_counter()-tick
        close(warm, iwarm, (arm, 'warm state'))
        exact(warm[20:], iwarm[20:], (arm, 'warm pulses'))
        _, blank = learned_model.forward(np.zeros((1, 25, 6)), w, arm, 28)
        exact(warm, np.concatenate((blank[-1]['states'][0], blank[-1]['hs'][0])),
              (arm, 'sealed zero input'))
        cases, independent_cases = {}, {}
        for history in ('late', 'full'):
            x, ix = inputs[history]
            rules = STANDARD_RULES
            if history == 'full' and arm == 'LEARN':
                rules += ('eviction_late_native',)
            for rule in rules:
                assert time.perf_counter()-start < 120
                reference = {'eviction_warm': warm}.get(rule)
                ireference = {'eviction_warm': iwarm}.get(rule)
                if rule == 'eviction_late_native':
                    reference = cases['late_original']['cue_native_before'][:, 0]
                    ireference = independent_cases['late_original']['cue_native_before'][:, 0]
                tick = time.perf_counter()
                actual = warm_model.evaluate(x, w, arm, 28, rule, reference)
                science += time.perf_counter()-tick
                tick = time.perf_counter()
                expected = independent_warm.replay(ix, w, arm, 28, rule, ireference)
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
                if history == 'late':
                    exact(actual['counts'][pairs[:, 0]], actual['counts'][pairs[:, 1]],
                          (arm, rule, 'deleted labels'))
                    scores = capacity_metrics.score(actual, base)
                    assert scores['older_ba'] == .5*scores['older_coverage']
                cases[history+'_'+rule] = actual
                independent_cases[history+'_'+rule] = expected

            for rule, source_rule in (('original', 'original'), ('match_zero', 'selective')):
                source = reset_model.evaluate(x, w, arm, 28, source_rule)
                for name, values in source.items():
                    exact(cases[history+'_'+rule][name], values, (history, arm, name, 'sealed reset source'))
        verify_rules(cases, arm, base, warm)
    assert time.perf_counter()-start < 120 and peak_memory() < 1024**3
    return dict(examples=n, science_seconds=science, audit_seconds=audit,
                science_forecast_seconds=3*BLOCKS*science+30, audit_forecast_seconds=3*BLOCKS*audit+30,
                checks=checks, max_numeric_error=maximum, late_old_ceiling=.5, full_ceiling=1.,
                native_cases_verified=4, intervention_cases_verified=13, warm_states_verified=2)


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
