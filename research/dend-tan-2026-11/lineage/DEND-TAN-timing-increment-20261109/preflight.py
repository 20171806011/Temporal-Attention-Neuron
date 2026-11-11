"""Independent fixture checks before generating any scientific split."""

from pathlib import Path
import time

import numpy as np

import continuation
import independent_capture
import independent_data
import independent_stats
import population
import reference
import task_data
import timing_stats as stats
from core import dump


ROOT = Path(__file__).resolve().parent
ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2')
FIT_SEED = 4099900501
TEST_SEED = 4099900502
WEIGHT_SEED = 4099900503
FIT_PERM_SEED = 4099900504
TEST_PERM_SEED = 4099900505
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
    assert native['raster'].shape == (len(x), 8, 2)
    assert np.all((native['raster'] == 0) | (native['raster'] == 1))
    np.testing.assert_array_equal(native['raster'].sum(axis=1), native['counts'])
    return native


def check_basis():
    basis = stats.helmert()
    assert basis.shape == (8, 8)
    np.testing.assert_allclose(basis[:, 0], np.full(8, 1 / np.sqrt(8)), atol=ATOL, rtol=0)
    np.testing.assert_allclose(basis[:, 1:].sum(axis=0), np.zeros(7), atol=ATOL, rtol=0)
    np.testing.assert_allclose(basis.T @ basis, np.eye(8), atol=ATOL, rtol=0)
    basis_fixture = np.zeros((8, 8, 2), dtype=np.uint8)
    basis_fixture[np.arange(8), np.arange(8), 0] = 1
    np.testing.assert_allclose(stats.features(basis_fixture)[:, ::2], basis, atol=ATOL, rtol=0)


def check_features(raster):
    actual = stats.features(raster)
    expected = independent_stats.helmert_features(raster)
    assert actual.shape == (len(raster), 16)
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=0)
    np.testing.assert_allclose(actual[:, :2], raster.sum(axis=1) / np.sqrt(8), atol=ATOL, rtol=0)
    np.testing.assert_allclose(np.sum(actual**2, axis=1), raster.sum(axis=(1, 2)), atol=ATOL, rtol=0)
    return actual, expected


def check_pairs(counts, base):
    actual = stats.pairs(counts, base)
    labels, quartets = base['y'], base['quartet']
    expected = independent_stats.paired_indices(counts, labels, quartets)
    for key, value in zip(('low', 'high', 'quartet', 'weight'), expected):
        if key == 'weight':
            np.testing.assert_allclose(actual[key], value, atol=ATOL, rtol=0)
        else:
            np.testing.assert_array_equal(actual[key], value)
    low, high, group, weight = (actual[key] for key in ('low', 'high', 'quartet', 'weight'))
    np.testing.assert_array_equal(labels[low], np.zeros(len(low), dtype=int))
    np.testing.assert_array_equal(labels[high], np.ones(len(high), dtype=int))
    np.testing.assert_array_equal(quartets[low], group)
    np.testing.assert_array_equal(quartets[high], group)
    np.testing.assert_array_equal(counts[low], counts[high])
    if len(low):
        groups, inverse, group_sizes = np.unique(group, return_inverse=True, return_counts=True)
        np.testing.assert_allclose(weight, 1 / (len(groups) * group_sizes[inverse]), atol=ATOL, rtol=0)
        np.testing.assert_allclose(np.bincount(inverse, weights=weight),
                                   np.full(len(groups), 1 / len(groups)), atol=ATOL, rtol=0)
        np.testing.assert_allclose(stats.rank_rate(np.zeros(len(low)), weight), 0.5, atol=ATOL, rtol=0)
    else:
        assert stats.rank_rate(np.empty(0), weight) is None
    return actual


def check_shuffle(raster, order):
    n = len(raster)
    np.testing.assert_array_equal(np.sort(order, axis=1), np.tile(np.arange(8), (n, 1)))
    result = stats.shuffle(raster, order)
    np.testing.assert_array_equal(result, independent_stats.shuffled_raster(raster, order))
    counts = raster.sum(axis=1)
    new_counts = result.sum(axis=1)
    np.testing.assert_array_equal(counts, new_counts)
    for c in (0, 1):
        for d in (0, 1):
            old_pattern = ((raster[:, :, 0] == c) & (raster[:, :, 1] == d)).sum(axis=1)
            new_pattern = ((result[:, :, 0] == c) & (result[:, :, 1] == d)).sum(axis=1)
            np.testing.assert_array_equal(old_pattern, new_pattern)
    # Both strict and tie-to-zero native decisions are functions of these signs.
    np.testing.assert_array_equal(np.sign(counts[:, 1].astype(int) - counts[:, 0].astype(int)),
                                  np.sign(new_counts[:, 1].astype(int) - new_counts[:, 0].astype(int)))
    return result


def check_rank(fit_raster, test_raster, fit_pairs, test_pairs):
    fit_features, expected_fit = check_features(fit_raster)
    test_features, expected_test = check_features(test_raster)
    fit_diff = stats.timing_differences(fit_features, fit_pairs)
    test_diff = stats.timing_differences(test_features, test_pairs)
    independent_fit = expected_fit[fit_pairs['high'], 2:] - expected_fit[fit_pairs['low'], 2:]
    independent_test = expected_test[test_pairs['high'], 2:] - expected_test[test_pairs['low'], 2:]
    np.testing.assert_allclose(fit_diff, independent_fit, atol=ATOL, rtol=0)
    np.testing.assert_allclose(test_diff, independent_test, atol=ATOL, rtol=0)
    coef, status = stats.fit_rank(fit_diff, fit_pairs['weight'])
    other = independent_stats.pair_fit(independent_fit, fit_pairs['weight'])
    assert coef.shape == (14,) and np.isfinite(coef).all()
    np.testing.assert_allclose(coef, other, atol=ATOL, rtol=0)
    score = test_diff @ coef
    expected_score = independent_test @ other
    np.testing.assert_allclose(score, expected_score, atol=ATOL, rtol=0)
    rank = stats.rank_rate(score, test_pairs['weight'])
    expected_rank = independent_stats.rank_rate(expected_score, test_pairs['quartet'])
    if expected_rank is None:
        assert rank is None
    else:
        np.testing.assert_allclose(rank, expected_rank, atol=ATOL, rtol=0)
    return score, rank, status, float(np.max(np.abs(coef - other)))


def check_class(fit_raster, test_raster, labels):
    fit_features, other_fit = check_features(fit_raster)
    test_features, other_test = check_features(test_raster)
    errors = []
    for width in (2, 16):
        mean, coef, intercept = stats.fit_class(fit_features[:, :width], labels)
        ref_mean, ref_coef, ref_intercept = independent_stats.class_fit(other_fit[:, :width], labels)
        np.testing.assert_allclose(mean, ref_mean, atol=ATOL, rtol=0)
        np.testing.assert_allclose(coef, ref_coef, atol=ATOL, rtol=0)
        np.testing.assert_allclose(intercept, ref_intercept, atol=ATOL, rtol=0)
        expected_score = (other_test[:, :width] - ref_mean) @ ref_coef + ref_intercept
        score = stats.class_scores(test_features[:, :width], mean, coef, intercept)
        np.testing.assert_allclose(score, expected_score, atol=ATOL, rtol=0)
        errors.append(float(np.max(np.abs(coef - ref_coef))))
    return max(errors)


def synthetic_checks():
    # Two quartets. One has one match, the other four matches.
    labels = np.tile([0, 0, 1, 1], 2)
    base = dict(y=labels, quartet=np.repeat(np.arange(2), 4))
    counts = np.array([[1, 1], [2, 2], [1, 1], [3, 3]] + [[4, 4]] * 4)
    uneven = check_pairs(counts, base)
    np.testing.assert_array_equal(np.bincount(uneven['quartet']), [1, 4])
    np.testing.assert_allclose(uneven['weight'], [.5, .125, .125, .125, .125], atol=ATOL, rtol=0)

    fit = np.zeros((8, 8, 2), dtype=np.uint8)
    test = np.zeros_like(fit)
    for index, label in enumerate(labels):
        fit[index, slice(0, 4) if label == 0 else slice(4, 8)] = 1
        test[index, [0, 1, 2, 4] if label == 0 else [3, 5, 6, 7]] = 1
    fit_pairs = check_pairs(fit.sum(axis=1), base)
    test_pairs = check_pairs(test.sum(axis=1), base)
    assert len(fit_pairs['low']) == len(test_pairs['low']) == 8
    score, rate, informative_status, _ = check_rank(fit, test, fit_pairs, test_pairs)
    assert np.all(score > 1e-10)
    np.testing.assert_allclose(rate, 1, atol=ATOL, rtol=0)
    reverse, _, _, _ = check_rank(fit[:, ::-1], test[:, ::-1], fit_pairs, test_pairs)
    np.testing.assert_allclose(reverse, score, atol=ATOL, rtol=0)

    identical = np.broadcast_to(fit[0], fit.shape).copy()
    score, rate, identical_status, _ = check_rank(identical, identical, fit_pairs, fit_pairs)
    np.testing.assert_array_equal(score, np.zeros(len(score)))
    assert rate == 0.5

    unmatched = fit.copy()
    unmatched[labels == 1] = 0
    empty = check_pairs(unmatched.sum(axis=1), base)
    score, rate, missing_status, _ = check_rank(unmatched, unmatched, empty, empty)
    assert score.size == 0 and rate is None
    assert len({informative_status, identical_status, missing_status}) == 3
    return dict(unequal_quartet_weights=True, informative_equal_count_rate=1.0,
                identical_raster_rate=0.5, missing_pairs_rate=None,
                fitted_statuses=[informative_status, identical_status, missing_status])


def run_checks():
    check_basis()
    synthetic = synthetic_checks()
    fit_base, fit_x = fixture_input(64, FIT_SEED)
    test_base, test_x = fixture_input(128, TEST_SEED)
    fit_maps = stats.make_permutations(len(fit_x), FIT_PERM_SEED)
    test_maps = stats.make_permutations(len(test_x), TEST_PERM_SEED)
    for maps, x, seed in ((fit_maps, fit_x, FIT_PERM_SEED), (test_maps, test_x, TEST_PERM_SEED)):
        assert maps.shape == (16, len(x), 8) and maps.dtype == np.uint8
        np.testing.assert_array_equal(maps, independent_stats.permutation_maps(len(x), seed))

    errors = []
    scalar_episodes = 0
    for arm in ARMS:
        weights = population.initialize(WEIGHT_SEED)
        fit = check_capture(fit_x, weights, arm)
        test = check_capture(test_x, weights, arm)
        if arm in ('FULL', 'NOATT'):
            scalar = reference.scalar_oracle(test_x[:2], weights, arm, 16)
            np.testing.assert_array_equal(test['raster'][:2], scalar['spikes'][:, 16:, 6:])
            scalar_episodes += 2
        fit_pairs = check_pairs(fit['counts'], fit_base)
        test_pairs = check_pairs(test['counts'], test_base)
        fit_raster, test_raster = fit['raster'], test['raster']
        errors.append(check_class(fit_raster, test_raster, fit_base['y']))
        real_score, _, _, error = check_rank(fit_raster, test_raster, fit_pairs, test_pairs)
        errors.append(error)
        reversed_score, _, _, error = check_rank(fit_raster[:, ::-1], test_raster[:, ::-1], fit_pairs, test_pairs)
        errors.append(error)
        np.testing.assert_allclose(real_score, reversed_score, atol=ATOL, rtol=0)
        for fit_map, test_map in zip(fit_maps, test_maps):
            fit_scrambled = check_shuffle(fit_raster, fit_map)
            test_scrambled = check_shuffle(test_raster, test_map)
            _, _, _, error = check_rank(fit_scrambled, test_scrambled, fit_pairs, test_pairs)
            errors.append(error)

    return dict(fixture_fit_rows=64, fixture_test_rows=128, fixture_seed_fit=FIT_SEED,
                fixture_seed_test=TEST_SEED, fixture_seed_weights=WEIGHT_SEED,
                fixture_seed_fit_permutations=FIT_PERM_SEED, fixture_seed_test_permutations=TEST_PERM_SEED,
                arms=list(ARMS), native_captures=8, rank_fits=68, classification_fits=8,
                reversal_controls=4, preserved_shuffle_archives=128, scalar_tan_episodes=scalar_episodes,
                max_coefficient_error=max(errors), synthetic=synthetic)


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
