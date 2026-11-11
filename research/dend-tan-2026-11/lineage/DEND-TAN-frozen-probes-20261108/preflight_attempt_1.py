"""Exercise the probe pipeline on independent fixture data before freezing."""

import time

import numpy as np

import capture
import independent_capture
import independent_data
import independent_probe
import population
import probe
import reference
import task_data
from core import CONDITIONS, R, dump, native


ARMS = ("DEND", "FULL", "NOATT", "GAIN2")
FIT_SEED = 4099900101
TEST_SEED = 4099900102
PERMUTATION_SEED = 4099900103
WEIGHT_SEED = 4099900104
ATOL = 2e-10


def fixture_base(size, seed):
    actual = task_data.make_base(size, seed)
    expected = independent_data.make_base(size, seed)
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    return actual


def verified_input(base, condition):
    values, query = task_data.assemble(base, condition)
    independent_data.check(base, values, condition)
    assert query == 16
    return values


def prefix_pairs(base):
    """Equal cue position and nuisance values, opposite first bit and target."""
    pairs = np.empty((len(base["y"]) // 2, 2), dtype=int)
    for quartet in range(len(base["amplitude"])):
        for selected in (0, 1):
            indices = np.flatnonzero(
                (base["quartet"] == quartet) & (base["selected"] == selected)
            )
            assert len(indices) == 2
            assert base["y"][indices].sum() == 1
            pairs[2 * quartet + selected] = indices
    return pairs.T


def compare_taps(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if key in ("p", "tap_counts", "counts", "resources"):
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        else:
            np.testing.assert_allclose(actual[key], expected[key], atol=ATOL, rtol=0, err_msg=key)


def compare_scalar(values, weights, arm, taps):
    scalar = reference.scalar_oracle(values[:2], weights, arm, 16)
    states = np.concatenate([taps[key][:2] for key in ("d", "v", "o")], axis=2)
    np.testing.assert_allclose(states, scalar["states"][:, capture.TAPS], atol=ATOL, rtol=0)
    np.testing.assert_allclose(taps["H"][:2], scalar["histories"][:, capture.TAPS], atol=ATOL, rtol=0)
    np.testing.assert_array_equal(taps["p"][:2], scalar["spikes"][:, capture.TAPS, :6])
    np.testing.assert_array_equal(taps["counts"][:2], scalar["counts"])


def compare_fit(values, labels):
    actual = probe.fit(values, labels)
    targets = 2 * labels - 1
    expected = independent_probe.fit(values, targets, alpha=0.1)
    expected_coef = np.concatenate((expected["coef"], expected["intercept"][None, :]))
    np.testing.assert_allclose(actual["mean"], expected["mean"], atol=ATOL, rtol=0)
    np.testing.assert_array_equal(actual["active"], ~expected["constant"])
    np.testing.assert_allclose(actual["scale"][actual["active"]], expected["scale"][actual["active"]],
                               atol=ATOL, rtol=0)
    np.testing.assert_array_equal(actual["scale"][~actual["active"]],
                                  np.full(np.sum(~actual["active"]), 1e-8))
    np.testing.assert_allclose(actual["coef"], expected_coef, atol=ATOL, rtol=0)
    np.testing.assert_allclose(actual["fit_scores"], independent_probe.scores(values, expected),
                               atol=ATOL, rtol=0)

    # The intercept column has zero regularization, even for unbalanced fixture labels.
    transformed = (values - actual["mean"]) / actual["scale"]
    transformed[:, ~actual["active"]] = 0
    design = np.column_stack((transformed, np.ones(len(values))))
    penalty = actual["coef"].copy() * 0.1
    penalty[-1] = 0
    residual = design.T @ (design @ actual["coef"] - targets) / len(values) + penalty
    maximum_residual = float(np.max(np.abs(residual)))
    assert maximum_residual < ATOL
    coefficient_error = float(np.max(np.abs(actual["coef"] - expected_coef)))
    return actual, expected, maximum_residual, coefficient_error


def check_constant_columns():
    ramp = np.linspace(-1, 1, 16)
    values = np.column_stack((ramp, np.full(16, 2.0), ramp * 1e-11))
    labels = (np.arange(16) >= 5).astype(int)[:, None]
    actual, expected, residual, error = compare_fit(values, labels)
    np.testing.assert_array_equal(actual["active"], [True, False, False])
    np.testing.assert_allclose(actual["coef"][-1, 0], (2 * labels - 1).mean(), atol=ATOL, rtol=0)
    shifted = values.copy()
    shifted[:, 1:] += 200
    np.testing.assert_array_equal(probe.predict(actual, values), probe.predict(actual, shifted))
    np.testing.assert_allclose(probe.predict(actual, shifted), independent_probe.scores(shifted, expected),
                               atol=ATOL, rtol=0)
    tie_scores = np.array([0.0, -1e-10, 1e-10, 2e-10])
    assert probe.metrics(tie_scores, np.array([0, 0, 0, 1]))["balanced_accuracy"] == 1.0
    return residual, error


def run_checks():
    fit_base = fixture_base(64, FIT_SEED)
    test_base = fixture_base(128, TEST_SEED)
    fit_input = verified_input(fit_base, "nominal")
    test_inputs = {condition: verified_input(test_base, condition) for condition in CONDITIONS}
    shuffled = probe.permutations(fit_base, PERMUTATION_SEED)
    assert shuffled.shape == (64, 20)
    for quartet in range(16):
        assert np.all(shuffled[fit_base["quartet"] == quartet].sum(axis=0) == 2)
    assert not np.all(shuffled == fit_base["y"][:, None])
    labels = np.column_stack((fit_base["y"], shuffled))
    target = test_base["y"]
    early_left, early_right = prefix_pairs(test_base)
    np.testing.assert_array_equal(test_inputs["nominal"][early_left, :10],
                                  test_inputs["nominal"][early_right, :10])
    removed_left, removed_right = np.array(independent_data.pairs(test_base)).T
    max_residual, max_coefficient_error = check_constant_columns()
    configurations = early_controls = information_controls = scalar_episodes = 0

    for arm in ARMS:
        weights = population.initialize(WEIGHT_SEED)
        fit_taps = capture.capture(fit_input, weights, arm)
        independent_fit = independent_capture.capture(fit_input, weights, arm)
        compare_taps(fit_taps, independent_fit)
        test_taps = {}
        independent_tests = {}
        for condition, values in test_inputs.items():
            actual_taps = capture.capture(values, weights, arm)
            expected_taps = independent_capture.capture(values, weights, arm)
            compare_taps(actual_taps, expected_taps)
            test_taps[condition] = actual_taps
            independent_tests[condition] = expected_taps
            if condition != "nominal":
                metrics = native(actual_taps["counts"], target)
                assert metrics["forced_ba"] == 0.5
                assert metrics["strict_ba"] == 0.5 * metrics["coverage"]
        if arm in ("FULL", "NOATT"):
            compare_scalar(test_inputs["nominal"], weights, arm, test_taps["nominal"])
            scalar_episodes += 2
        for group, tap in capture.configurations(arm):
            fit_features = capture.features(fit_taps, group, tap)
            independent_features = independent_capture.features(independent_fit, group, tap)
            np.testing.assert_allclose(fit_features, independent_features, atol=ATOL, rtol=0)
            actual, expected, residual, error = compare_fit(fit_features, labels)
            max_residual = max(max_residual, residual)
            max_coefficient_error = max(max_coefficient_error, error)
            for condition in CONDITIONS:
                features = capture.features(test_taps[condition], group, tap)
                independent_features = independent_capture.features(independent_tests[condition], group, tap)
                np.testing.assert_allclose(features, independent_features, atol=ATOL, rtol=0)
                scores = probe.predict(actual, features)
                expected_scores = independent_probe.scores(features, expected)
                np.testing.assert_allclose(scores, expected_scores, atol=ATOL, rtol=0)
                ba = probe.metrics(scores[:, 0], target)["balanced_accuracy"]
                assert ba == independent_probe.metrics(expected_scores[:, 0], target)["balanced_accuracy"]
                if condition == "nominal" and tap == 9:
                    np.testing.assert_array_equal(features[early_left], features[early_right])
                    assert ba == 0.5
                    early_controls += 1
                if condition == "tag_removed":
                    np.testing.assert_array_equal(features[removed_left], features[removed_right])
                    assert ba == 0.5
                    information_controls += 1
                elif condition == "query_only":
                    np.testing.assert_array_equal(features, np.broadcast_to(features[0], features.shape))
                    assert ba == 0.5
                    information_controls += 1
            configurations += 1

    assert (configurations, early_controls, information_controls, scalar_episodes) == (148, 24, 296, 4)
    return dict(fixture_fit_rows=64, fixture_test_rows=128, fixture_seed_fit=FIT_SEED,
                fixture_seed_test=TEST_SEED, fixture_seed_permutation=PERMUTATION_SEED,
                fixture_seed_weights=WEIGHT_SEED, arms=list(ARMS), configurations=configurations,
                early_t9_controls=early_controls, information_controls=information_controls,
                scalar_tan_episodes=scalar_episodes, max_normal_residual=max_residual,
                max_coefficient_error=max_coefficient_error)


def main():
    assert not (R / "FREEZE.json").exists(), "frozen artifacts are read-only"
    start = time.perf_counter()
    try:
        checks = run_checks()
    except Exception as error:
        dump(R / "PRECHECK.json", dict(status="FAIL", exception=type(error).__name__,
                                       reason=str(error), wall_seconds=time.perf_counter() - start))
        raise
    dump(R / "PRECHECK.json", dict(status="PASS", wall_seconds=time.perf_counter() - start, **checks))
    print("PRECHECK PASS", checks, flush=True)


if __name__ == "__main__":
    main()
