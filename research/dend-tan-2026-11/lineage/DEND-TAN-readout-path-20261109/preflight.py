"""Validate the intervention on independent, small fixture data."""

import time
from pathlib import Path

import numpy as np

import continuation
import core
import independent_capture
import independent_data
import independent_probe
import model
import population
import probe
import reference
import task_data
from core import dump


ROOT = Path(__file__).resolve().parent
ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2')
TAPS = (9, 15, 19)
FIT_SEED = 4099900301
TEST_SEED = 4099900302
WEIGHT_SEED = 4099900303
ATOL = 2e-10
DUPLICATE_ATOL = 1e-12


def configurations(arm):
    """Explicit fixture enumeration of the protocol, independent of the run loop."""
    early_groups = ('full', 'hidden', 'branch', 'soma_p', 'output')
    if arm in ('FULL', 'NOATT'):
        early_groups += ('history',)
    yield 9, 'full', 'content_flip'
    for tap, groups in ((15, early_groups), (19, ('full', 'hidden', 'output'))):
        yield tap, 'full', 'self'
        for group in groups:
            for donor in ('cue_flip', 'content_flip', 'both_flip'):
                yield tap, group, donor


def donors(base):
    quartet = base['quartet']
    selected = base['selected']
    first_bit = base['first_bit']
    lookup = {key: index for index, key in enumerate(zip(quartet, selected, first_bit))}
    assert len(lookup) == len(quartet)
    result = {'self': np.arange(len(quartet))}
    for name, ds, db in (('cue_flip', 1, 0), ('content_flip', 0, 1), ('both_flip', 1, 1)):
        index = np.array([lookup[k, s ^ ds, b ^ db] for k, s, b in zip(quartet, selected, first_bit)])
        np.testing.assert_array_equal(quartet[index], quartet)
        np.testing.assert_array_equal(index[index], result['self'])
        np.testing.assert_array_equal(base['y'][index], base['y'] ^ (ds ^ db))
        assert np.all(index != result['self'])
        result[name] = index
    return result


def fixture_input(size, seed):
    base = task_data.make_base(size, seed)
    expected = independent_data.make_base(size, seed)
    assert base.keys() == expected.keys()
    for name in base:
        np.testing.assert_array_equal(base[name], expected[name], err_msg=name)
    values, query = task_data.assemble(base, 'nominal')
    assert query == 16
    independent_data.check(base, values, 'nominal')
    return base, values


def compare_capture(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        if name in ('d', 'v', 'o', 'H'):
            np.testing.assert_allclose(actual[name], expected[name], atol=ATOL, rtol=0, err_msg=name)
        else:
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)


def compare_native(actual, values, weights, arm):
    native = model.evaluate(values, weights, arm, 16)
    np.testing.assert_array_equal(actual['counts'], native['counts'])
    np.testing.assert_array_equal(actual['resources'], native['resources'][:, :4])
    np.testing.assert_array_equal(actual['raster'][:8], native['spikes'][:, 16:, 6:])
    np.testing.assert_array_equal(actual['counts'], actual['raster'].sum(axis=1))
    assert np.all((actual['raster'] == 0) | (actual['raster'] == 1))


def compare_scalar(actual, values, weights, arm):
    scalar = reference.scalar_oracle(values[:2], weights, arm, 16)
    states = np.concatenate([actual[name][:2] for name in ('d', 'v', 'o')], axis=2)
    np.testing.assert_allclose(states, scalar['states'][:, TAPS], atol=ATOL, rtol=0)
    np.testing.assert_allclose(actual['H'][:2], scalar['histories'][:, TAPS], atol=ATOL, rtol=0)
    np.testing.assert_array_equal(actual['p'][:2], scalar['spikes'][:, TAPS, :6])
    np.testing.assert_array_equal(actual['raster'][:2], scalar['spikes'][:, 16:, 6:])


def output_features(capture, kind):
    if kind == 'counts':
        return capture['counts']
    assert kind == 'raster'
    return capture['raster'].reshape(len(capture['raster']), 16)


def compare_probe(fit_capture, test_capture, labels, kind):
    values = output_features(fit_capture, kind)
    target = 2 * labels[:, None] - 1
    actual = probe.fit(values, labels[:, None])
    expected = independent_probe.fit(values, target, alpha=0.1)
    expected_coef = np.concatenate((expected['coef'], expected['intercept'][None, :]))
    np.testing.assert_allclose(actual['coef'], expected_coef, atol=ATOL, rtol=0)
    np.testing.assert_array_equal(actual['active'], ~expected['constant'])
    np.testing.assert_allclose(actual['fit_scores'], independent_probe.scores(values, expected), atol=ATOL, rtol=0)
    test = output_features(test_capture, kind)
    np.testing.assert_allclose(probe.predict(actual, test), independent_probe.scores(test, expected), atol=ATOL, rtol=0)
    return float(np.max(np.abs(actual['coef'] - expected_coef)))


def run_checks():
    fit_base, fit_values = fixture_input(64, FIT_SEED)
    test_base, test_values = fixture_input(128, TEST_SEED)
    donor_indices = donors(test_base)
    run_donors = core.donors(test_base)
    assert run_donors.keys() == donor_indices.keys()
    for name in donor_indices:
        np.testing.assert_array_equal(run_donors[name], donor_indices[name], err_msg=name)
    np.testing.assert_array_equal(test_values[:, :10], test_values[donor_indices['content_flip'], :10])
    for tap in (15, 19):
        for index in donor_indices.values():
            np.testing.assert_array_equal(test_values[:, tap + 1:], test_values[index, tap + 1:])

    configuration_count = 0
    complete_suffix_controls = 0
    self_controls = 0
    early_controls = 0
    scalar_episodes = 0
    max_coefficient_error = 0.0
    for arm in ARMS:
        assert list(core.configurations(arm)) == list(configurations(arm))
        weights = population.initialize(WEIGHT_SEED)
        fit_capture = continuation.collect(fit_values, weights, arm, taps=())
        test_capture = continuation.collect(test_values, weights, arm)
        compare_native(fit_capture, fit_values, weights, arm)
        compare_native(test_capture, test_values, weights, arm)
        independent_fit = independent_capture.capture(fit_values, weights, arm, taps=())
        independent_test = independent_capture.capture(test_values, weights, arm)
        compare_capture(fit_capture, independent_fit)
        compare_capture(test_capture, independent_test)
        if arm in ('FULL', 'NOATT'):
            compare_scalar(test_capture, test_values, weights, arm)
            scalar_episodes += 2
        for kind in ('counts', 'raster'):
            error = compare_probe(fit_capture, test_capture, fit_base['y'], kind)
            max_coefficient_error = max(max_coefficient_error, error)

        for tap, group, donor_name in configurations(arm):
            index = donor_indices[donor_name]
            state = continuation.splice(test_capture, tap, group, index)
            independent_state = independent_capture.transplant(independent_test, tap, group, index)
            compare_capture(state, independent_state)
            resumed = continuation.resume(test_values, weights, arm, tap, state)
            expected_raster, expected_resources = independent_capture.continue_capture(
                test_values, weights, arm, tap, independent_state
            )
            expected = dict(suffix_raster=expected_raster, suffix_resources=expected_resources)
            compare_capture(resumed, expected)
            prefix_length = max(0, tap - 15)
            suffix = resumed['suffix_raster']
            assert suffix.shape == (len(test_values), 8 - prefix_length, 2)
            assert np.all((suffix == 0) | (suffix == 1))
            assert resumed['suffix_resources'].shape == (len(test_values), 4)
            reconstructed = np.concatenate((test_capture['raster'][:, :prefix_length], suffix), axis=1)
            assert reconstructed.shape == test_capture['raster'].shape
            if tap == 9:
                for name, value in state.items():
                    original = test_capture[name][:, 0]
                    if name == 'p':
                        np.testing.assert_array_equal(value, original)
                    else:
                        np.testing.assert_allclose(value, original, atol=DUPLICATE_ATOL, rtol=0)
                np.testing.assert_array_equal(reconstructed, test_capture['raster'])
                early_controls += 1
            elif donor_name == 'self':
                np.testing.assert_array_equal(reconstructed, test_capture['raster'])
                self_controls += 1
            elif group == 'full':
                np.testing.assert_array_equal(suffix, test_capture['raster'][index, prefix_length:])
                expected_counts = (
                    test_capture['raster'][:, :prefix_length].sum(axis=1)
                    + test_capture['raster'][index, prefix_length:].sum(axis=1)
                )
                np.testing.assert_array_equal(reconstructed.sum(axis=1), expected_counts)
                complete_suffix_controls += 1
            configuration_count += 1

    assert (configuration_count, complete_suffix_controls, self_controls, early_controls) == (114, 24, 8, 4)
    return dict(fixture_fit_rows=64, fixture_test_rows=128, fixture_seed_fit=FIT_SEED,
                fixture_seed_test=TEST_SEED, fixture_seed_weights=WEIGHT_SEED, arms=list(ARMS),
                configurations=configuration_count, complete_suffix_controls=complete_suffix_controls,
                self_controls=self_controls, early_controls=early_controls, scalar_tan_episodes=scalar_episodes,
                max_coefficient_error=max_coefficient_error, duplicate_continuous_atol=DUPLICATE_ATOL)


def main():
    assert not (ROOT / 'FREEZE.json').exists(), 'frozen artifacts are read-only'
    assert not (ROOT / 'PRECHECK.json').exists(), 'retain the previous PRECHECK result before rerunning'
    start = time.perf_counter()
    try:
        checks = run_checks()
    except Exception as error:
        dump(ROOT / 'PRECHECK.json', dict(status='FAIL', exception=type(error).__name__,
                                         reason=str(error), wall_seconds=time.perf_counter() - start))
        raise
    dump(ROOT / 'PRECHECK.json', dict(status='PASS', wall_seconds=time.perf_counter() - start, **checks))
    print('PRECHECK PASS', checks, flush=True)


if __name__ == '__main__':
    main()
