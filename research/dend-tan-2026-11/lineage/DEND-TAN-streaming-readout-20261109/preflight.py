"""Verify streaming, orthogonal features and shuffled controls before freezing."""

from pathlib import Path
import time

import numpy as np

import continuation
import independent_capture
import independent_data
import independent_stats
import population
import readout
import reference
import task_data
import timing_stats
from core import dump
from run import make_permutations


ROOT = Path(__file__).resolve().parent
ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2')
FIT_SEED, TEST_SEED = 4099900701, 4099900702
WEIGHT_SEED = 4099900703
FIT_PERM_SEED, TEST_PERM_SEED = 4099900704, 4099900705
ATOL = 2e-10


def fixture_input(n, seed):
    base = task_data.make_base(n, seed)
    other = independent_data.make_base(n, seed)
    assert base.keys() == other.keys()
    for key in base:
        np.testing.assert_array_equal(base[key], other[key], err_msg=key)
    x, query = task_data.assemble(base, 'nominal')
    assert query == 16
    independent_data.check(base, x, 'nominal')
    return base, x


def check_capture(x, weights, arm):
    native = continuation.collect(x, weights, arm, taps=())
    other = independent_capture.capture(x, weights, arm, taps=())
    assert native.keys() == other.keys() == {'raster', 'counts', 'resources'}
    for key in native:
        np.testing.assert_array_equal(native[key], other[key], err_msg=key)
    raster = native['raster']
    assert raster.shape == (len(x), 8, 2) and raster.dtype == np.uint8
    assert np.all(raster <= 1)
    return native


def direction_matrix(kind):
    # Independent vector-space specification of the two normalized features.
    difference = np.tile([-1., 1.], 8)
    if kind == 'count2':
        second = np.ones(16)
    else:
        offsets = 2 * (np.arange(8, dtype=float) - 3.5)
        second = np.repeat(offsets, 2) * difference
    directions = np.column_stack((difference, second))
    return directions / np.linalg.norm(directions, axis=0)


def check_integer_states(raster):
    counts, moment = readout.accumulate(raster, moment=True)
    plain_counts, absent = readout.accumulate(raster)
    exact_counts = raster.sum(axis=1)
    differences = raster[:, :, 1].astype(np.int64) - raster[:, :, 0].astype(np.int64)
    exact_moment = differences @ (2 * np.arange(8) - 7)
    assert counts.dtype == moment.dtype == plain_counts.dtype == np.int32
    assert counts.nbytes == len(raster) * 8 and moment.nbytes == len(raster) * 4
    assert absent is None and np.all(np.abs(moment) <= 32)
    np.testing.assert_array_equal(counts, exact_counts)
    np.testing.assert_array_equal(plain_counts, exact_counts)
    np.testing.assert_array_equal(moment, exact_moment)
    np.testing.assert_array_equal(readout.predict(raster, 'native'), exact_counts[:, 1] > exact_counts[:, 0])
    return int(moment.min()), int(moment.max())


def check_folded(fit_raster, test_raster, y, kind):
    directions = direction_matrix(kind)
    np.testing.assert_allclose(directions.T @ directions, np.eye(2), atol=ATOL, rtol=0)
    fit_x = fit_raster.reshape(len(fit_raster), 16) @ directions
    test_x = test_raster.reshape(len(test_raster), 16) @ directions
    np.testing.assert_allclose(readout.raw_features(fit_raster, kind) / readout.SCALES[kind], fit_x, atol=ATOL, rtol=0)
    np.testing.assert_allclose(readout.raw_features(test_raster, kind) / readout.SCALES[kind], test_x, atol=ATOL, rtol=0)
    mean, coef, intercept, folded = readout.fit_rule(fit_raster, y, kind)
    ref_mean, ref_coef, ref_intercept = independent_stats.class_fit(fit_x, y)
    np.testing.assert_allclose(mean, ref_mean, atol=ATOL, rtol=0)
    np.testing.assert_allclose(coef, ref_coef, atol=ATOL, rtol=0)
    np.testing.assert_allclose(intercept, ref_intercept, atol=ATOL, rtol=0)
    expected = (test_x - ref_mean) @ ref_coef + ref_intercept
    score = readout.stream_scores(test_raster, kind, folded)
    np.testing.assert_allclose(score, expected, atol=ATOL, rtol=0)
    np.testing.assert_array_equal(readout.predict(test_raster, kind, folded), expected > 1e-10)
    return score, float(np.max(np.abs(coef - ref_coef))), float(np.max(np.abs(score - expected)))


def check_offline(fit_raster, test_raster, y):
    fit_x = timing_stats.features(fit_raster)
    test_x = timing_stats.features(test_raster)
    ref_fit = independent_stats.helmert_features(fit_raster)
    ref_test = independent_stats.helmert_features(test_raster)
    np.testing.assert_allclose(fit_x, ref_fit, atol=ATOL, rtol=0)
    np.testing.assert_allclose(test_x, ref_test, atol=ATOL, rtol=0)
    mean, coef, intercept = timing_stats.fit_class(fit_x, y)
    ref_mean, ref_coef, ref_intercept = independent_stats.class_fit(ref_fit, y)
    np.testing.assert_allclose(coef, ref_coef, atol=ATOL, rtol=0)
    actual = timing_stats.class_scores(test_x, mean, coef, intercept)
    expected = (ref_test - ref_mean) @ ref_coef + ref_intercept
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)
    return float(np.max(np.abs(coef - ref_coef))), float(np.max(np.abs(actual - expected)))


def check_shuffle(raster, order):
    np.testing.assert_array_equal(np.sort(order, axis=1), np.broadcast_to(np.arange(8), order.shape))
    shuffled = timing_stats.shuffle(raster, order)
    np.testing.assert_array_equal(shuffled, independent_stats.shuffled_raster(raster, order))
    np.testing.assert_array_equal(raster.sum(axis=1), shuffled.sum(axis=1))
    # Compare the multiset of binary channel pairs rather than only marginal counts.
    encode = lambda r: np.sort(2*r[:, :, 0] + r[:, :, 1], axis=1)
    np.testing.assert_array_equal(encode(raster), encode(shuffled))
    np.testing.assert_array_equal(readout.predict(raster, 'native'), readout.predict(shuffled, 'native'))
    return shuffled


def check_boundary_cases():
    words = np.arange(1 << 16, dtype=np.uint32)[:, None]
    bits = ((words >> np.arange(16, dtype=np.uint32)) & 1).astype(np.uint8)
    raster = bits.reshape(-1, 8, 2)
    extremes = check_integer_states(raster)
    assert extremes == (-32, 32)

    # Identical per-channel counts, distinct relative timing across the labels.
    y = np.array([0, 1, 0, 1], dtype=np.uint8)
    separated = np.zeros((4, 8, 2), dtype=np.uint8)
    separated[y == 0, :4, 1] = 1
    separated[y == 0, 4:, 0] = 1
    separated[y == 1, :4, 0] = 1
    separated[y == 1, 4:, 1] = 1
    count = readout.fit_rule(separated, y, 'count2')[-1]
    moment = readout.fit_rule(separated, y, 'moment2')[-1]
    np.testing.assert_array_equal(readout.predict(separated, 'count2', count), np.zeros(4, dtype=np.uint8))
    np.testing.assert_array_equal(readout.predict(separated, 'moment2', moment), y)
    return dict(binary_rasters_checked=len(raster), minimum_moment=extremes[0], maximum_moment=extremes[1],
                equal_count_synthetic_count_ba=0.5, equal_count_synthetic_moment_ba=1.0)


def run_checks():
    boundary = check_boundary_cases()
    fit_base, fit_x = fixture_input(64, FIT_SEED)
    _, test_x = fixture_input(128, TEST_SEED)
    fit_maps = make_permutations(len(fit_x), FIT_PERM_SEED)
    test_maps = make_permutations(len(test_x), TEST_PERM_SEED)
    for maps, x, seed in ((fit_maps, fit_x, FIT_PERM_SEED), (test_maps, test_x, TEST_PERM_SEED)):
        assert maps.shape == (8, len(x), 8) and maps.dtype == np.uint8
        np.testing.assert_array_equal(maps, independent_stats.permutation_maps(len(x), seed)[:8])

    coef_errors, score_errors = [], []
    scalar_episodes = 0
    weights = population.initialize(WEIGHT_SEED)
    for arm in ARMS:
        fit = check_capture(fit_x, weights, arm)
        test = check_capture(test_x, weights, arm)
        fit_raster, test_raster = fit['raster'], test['raster']
        y = fit_base['y']
        for native in (fit, test):
            check_integer_states(native['raster'])
            np.testing.assert_array_equal(native['counts'], native['raster'].sum(axis=1))
        if arm in ('FULL', 'NOATT'):
            scalar = reference.scalar_oracle(test_x[:2], weights, arm, 16)
            np.testing.assert_array_equal(test_raster[:2], scalar['spikes'][:, 16:, 6:])
            scalar_episodes += 2

        _, count_error, count_score_error = check_folded(fit_raster, test_raster, y, 'count2')
        moment_score, moment_error, moment_score_error = check_folded(fit_raster, test_raster, y, 'moment2')
        offline_error, offline_score_error = check_offline(fit_raster, test_raster, y)
        coef_errors.extend((count_error, moment_error, offline_error))
        score_errors.extend((count_score_error, moment_score_error, offline_score_error))
        reversed_score, _, _ = check_folded(fit_raster[:, ::-1], test_raster[:, ::-1], y, 'moment2')
        np.testing.assert_allclose(moment_score, reversed_score, atol=ATOL, rtol=0)
        for fit_order, test_order in zip(fit_maps, test_maps):
            fit_scrambled = check_shuffle(fit_raster, fit_order)
            test_scrambled = check_shuffle(test_raster, test_order)
            _, coef_error, score_error = check_folded(fit_scrambled, test_scrambled, y, 'moment2')
            coef_errors.append(coef_error)
            score_errors.append(score_error)

    return dict(fixture_fit_rows=64, fixture_test_rows=128, fixture_seed_fit=FIT_SEED,
                fixture_seed_test=TEST_SEED, fixture_seed_weights=WEIGHT_SEED,
                fixture_seed_fit_permutations=FIT_PERM_SEED, fixture_seed_test_permutations=TEST_PERM_SEED,
                arms=list(ARMS), native_captures=8, classification_fits=44, reversal_controls=4,
                preserved_shuffle_archives=64, scalar_tan_episodes=scalar_episodes,
                max_coefficient_error=max(coef_errors), max_score_error=max(score_errors), **boundary)


def main():
    assert not (ROOT / 'FREEZE.json').exists(), 'frozen artifacts are read-only'
    assert not (ROOT / 'PRECHECK.json').exists(), 'retain the previous PRECHECK result before rerunning'
    start = time.perf_counter()
    try:
        checks = run_checks()
    except Exception as error:
        dump(ROOT / 'PRECHECK.json', dict(status='FAIL', reason=str(error), exception=type(error).__name__,
                                         wall_seconds=time.perf_counter() - start))
        raise
    dump(ROOT / 'PRECHECK.json', dict(status='PASS', wall_seconds=time.perf_counter() - start, **checks))
    print('PRECHECK PASS', checks, flush=True)


if __name__ == '__main__':
    main()
