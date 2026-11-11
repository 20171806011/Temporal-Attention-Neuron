"""Fixed orthogonal features, within-quartet matching and ridge readouts."""

import numpy as np


ALPHA = .1
TIE = 1e-10


def helmert():
    basis = np.zeros((8, 8))
    basis[:, 0] = 1/np.sqrt(8)
    for j in range(1, 8):
        scale = np.sqrt(j*(j+1))
        basis[:j, j] = 1/scale
        basis[j, j] = -j/scale
    return basis


BASIS = helmert()


def features(raster):
    return np.einsum('ntc,tk->nkc', raster, BASIS).reshape(len(raster), 16)


def make_permutations(n, seed):
    rng = np.random.default_rng(seed)
    order = np.empty((16, n, 8), dtype=np.uint8)
    for r in range(16):
        for i in range(n):
            order[r, i] = rng.permutation(8)
    return order


def shuffle(raster, mapping):
    return raster[np.arange(len(raster))[:, None], mapping]


def pairs(counts, base):
    low, high, quartet = [], [], []
    y = base['y']
    for q in range(len(y)//4):
        members = np.flatnonzero(base['quartet'] == q)
        for i in members[y[members] == 0]:
            for j in members[y[members] == 1]:
                if np.array_equal(counts[i], counts[j]):
                    low.append(i)
                    high.append(j)
                    quartet.append(q)
    result = {name: np.asarray(value, dtype=int) for name, value in
              (('low', low), ('high', high), ('quartet', quartet))}
    result['weight'] = pair_weights(result['quartet'])
    return result


def pair_weights(quartet):
    if not len(quartet):
        return np.empty(0)
    counts = np.bincount(quartet)
    return 1/(np.count_nonzero(counts)*counts[quartet])


def timing_differences(x, matched):
    return x[matched['high'], 2:] - x[matched['low'], 2:]


def fit_rank(differences, weights):
    if not len(weights):
        return np.zeros(14), 'NO_PAIRS'
    if not np.any(differences):
        return np.zeros(14), 'ZERO_DIFFERENCES'
    weighted = differences * weights[:, None]
    coef = np.linalg.solve(differences.T @ weighted + ALPHA*np.eye(14), weighted.sum(0))
    return coef, 'FITTED'


def rank_rate(scores, weights):
    if not len(scores):
        return None
    wins = np.where(scores > TIE, 1., np.where(scores < -TIE, 0., .5))
    return float(np.average(wins, weights=weights))


def fit_class(x, y):
    mean = x.mean(0)
    centered = x - mean
    target = 2*y - 1
    intercept = float(target.mean())
    coef = np.linalg.solve(centered.T @ centered/len(x) + ALPHA*np.eye(x.shape[1]),
                           centered.T @ (target-intercept)/len(x))
    return mean, coef, intercept


def class_scores(x, mean, coef, intercept):
    return (x-mean) @ coef + intercept


def balanced_accuracy(pred, y):
    return float(np.mean([np.mean(pred[y == label] == label) for label in (0, 1)]))


def classify(scores):
    return (scores > TIE).astype(int)


def native_metrics(counts, y):
    forced = (counts[:, 1] > counts[:, 0]).astype(int)
    tied = counts[:, 0] == counts[:, 1]
    strict = np.where(tied, -1, forced)
    silence = np.all(counts == 0, axis=1)
    sat = np.all(counts == 8, axis=1)
    return dict(native_strict_ba=balanced_accuracy(strict, y), native_forced_ba=balanced_accuracy(forced, y),
                native_coverage=float(np.mean(~tied)), native_silence=float(silence.mean()),
                native_saturated_tie=float(sat.mean()), native_intermediate_tie=float(np.mean(tied & ~silence & ~sat)))
