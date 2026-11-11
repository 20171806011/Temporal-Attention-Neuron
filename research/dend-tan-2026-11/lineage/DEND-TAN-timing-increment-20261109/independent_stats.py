"""Independent matched timing statistics and augmented least-squares fits."""

import numpy as np


ALPHA = 0.1
SCORE_TOLERANCE = 1e-10
SHUFFLES = 16


def paired_indices(counts, labels, quartets):
    """Return label-0/label-1 rows in stable quartet, low-row, high-row order."""
    ids, sizes = np.unique(quartets, return_counts=True)
    assert np.all(sizes == 4)
    ordered = np.argsort(quartets, kind='stable').reshape(len(ids), 4)
    block_labels = labels[ordered]
    assert np.all((block_labels == 0).sum(axis=1) == 2)
    assert np.all((block_labels == 1).sum(axis=1) == 2)
    zero = ordered[block_labels == 0].reshape(-1, 2)
    one = ordered[block_labels == 1].reshape(-1, 2)
    same = np.all(counts[zero, None] == counts[one][:, None], axis=-1)
    which, row_0, row_1 = np.nonzero(same)
    low, high = zero[which, row_0], one[which, row_1]
    pair_quartets = ids[which]
    counts_per_group = np.count_nonzero(same, axis=(1, 2))
    n_matched = np.count_nonzero(counts_per_group)
    weights = 1.0 / (n_matched * counts_per_group[which]) if len(which) else np.empty(0)
    return low, high, pair_quartets, weights


def permutation_maps(n_rows, seed):
    rng = np.random.default_rng(seed)
    return np.array([[rng.permutation(8) for _ in range(n_rows)] for _ in range(SHUFFLES)], dtype=np.uint8)


def shuffled_raster(raster, order):
    return np.take_along_axis(raster, order[:, :, None], axis=1)


def helmert_features(raster):
    """Use prefix sums rather than a matrix product for the Helmert transform."""
    raster = np.asarray(raster, dtype=np.float64)
    assert raster.shape[1:] == (8, 2)
    j = np.arange(1, 8, dtype=float)[None, :, None]
    differences = np.cumsum(raster, axis=1)[:, :-1] - j * raster[:, 1:]
    timing = differences / np.sqrt(j * (j + 1))
    count = raster.sum(axis=1, keepdims=True) / np.sqrt(8.0)
    return np.concatenate((count, timing), axis=1).reshape(len(raster), 16)


def pair_fit(differences, weights):
    """Solve the weighted target-one ridge with no intercept via augmented SVD."""
    n_pairs, dimensions = differences.shape
    if n_pairs == 0 or not np.any(differences):
        return np.zeros(dimensions)
    scale = np.sqrt(weights)
    design = np.concatenate((differences * scale[:, None], np.sqrt(ALPHA) * np.eye(dimensions)))
    target = np.concatenate((scale, np.zeros(dimensions)))
    return np.linalg.lstsq(design, target, rcond=None)[0]


def class_fit(features, labels):
    """Augment the original feature matrix with an unpenalized constant."""
    n_rows, dimensions = features.shape
    scale = np.sqrt(n_rows)
    observed = np.column_stack((features, np.ones(n_rows))) / scale
    penalty = np.column_stack((np.sqrt(ALPHA) * np.eye(dimensions), np.zeros(dimensions)))
    design = np.concatenate((observed, penalty))
    target = np.r_[(2.0 * labels - 1.0) / scale, np.zeros(dimensions)]
    solution = np.linalg.lstsq(design, target, rcond=None)[0]
    coef, intercept = solution[:-1], solution[-1]
    mean = features.mean(axis=0)
    return mean, coef, intercept + mean @ coef


def rank_rate(scores, pair_quartets):
    """Average paired wins with half credit for ties, then average over quartets."""
    if not len(pair_quartets):
        return None
    outcomes = np.select((scores > SCORE_TOLERANCE, scores < -SCORE_TOLERANCE), (1.0, 0.0), default=0.5)
    _, groups, sizes = np.unique(pair_quartets, return_inverse=True, return_counts=True)
    return float(np.mean(np.bincount(groups, weights=outcomes) / sizes))


def balanced_accuracy(predictions, labels):
    if np.unique(labels).size != 2:
        return None
    return float(sum(np.mean(predictions[labels == cls] == cls) for cls in (0, 1)) / 2)


def classification_metrics(scores, labels):
    return balanced_accuracy(scores > SCORE_TOLERANCE, labels), float(np.mean((scores - (2.0 * labels - 1)) ** 2))


def prior_raster_scores(raster, archive):
    """Replay the prior sealed standardizer and readout in coefficient form."""
    flat = raster.reshape(len(raster), 16)
    z = np.zeros(flat.shape, dtype=float)
    active = archive['active']
    z[:, active] = (flat[:, active] - archive['mean'][active]) / archive['scale'][active]
    coefficients = archive['coef'][:, 0]
    return z @ coefficients[:-1] + coefficients[-1]
