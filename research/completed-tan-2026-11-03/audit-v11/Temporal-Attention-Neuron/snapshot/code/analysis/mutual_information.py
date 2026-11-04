"""
Statistical Analysis Utilities
================================
Functions for computing t-tests, Cohen's d, bootstrap CIs, and plateau widths.

Author: Li Zexu, University of Leeds
"""
import numpy as np
from scipy import stats


def cohens_d(a, b):
    """Cohen's d with safeguards for degenerate variance."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    var_a = float(a.var(ddof=1)) if len(a) > 1 else 0.0
    var_b = float(b.var(ddof=1)) if len(b) > 1 else 0.0
    pooled_std = np.sqrt(max((var_a + var_b) / 2, 1e-6))
    diff = float(a.mean() - b.mean())
    d = diff / pooled_std
    return float(np.clip(d, -1000.0, 1000.0))


def bootstrap_ci(data, n_boot=10000, ci=0.95, rng=None):
    """Bootstrap confidence interval."""
    if rng is None:
        rng = np.random.RandomState(0)
    data = np.asarray(data)
    n = len(data)
    boot_means = np.array([rng.choice(data, n, replace=True).mean()
                           for _ in range(n_boot)])
    return (float(np.percentile(boot_means, (1 - ci) / 2 * 100)),
            float(np.percentile(boot_means, (1 + ci) / 2 * 100)))


def interpret_d(d):
    """Interpret effect size magnitude."""
    ad = abs(d)
    if ad < 0.2: return "negligible"
    if ad < 0.5: return "small"
    if ad < 0.8: return "medium"
    if ad < 2.0: return "large"
    if ad < 100.0: return "very large"
    return "extreme (degenerate variance)"


def compute_plateau_width(values, means, lower_is_better=True, tolerance=0.10):
    """Compute plateau width: range within tolerance of best."""
    if lower_is_better:
        best_idx = int(np.argmin(means))
        best_val = means[best_idx]
        threshold = best_val + tolerance * max(abs(best_val), 1e-6)
        in_plateau = means <= threshold
    else:
        best_idx = int(np.argmax(means))
        best_val = means[best_idx]
        threshold = best_val - tolerance * max(abs(best_val), 1e-6)
        in_plateau = means >= threshold
    plateau_indices = np.where(in_plateau)[0]
    if len(plateau_indices) == 0:
        return values[best_idx], values[best_idx], 0.0
    width_fraction = len(plateau_indices) / len(values)
    return values[plateau_indices[0]], values[plateau_indices[-1]], width_fraction


def mutual_information_ksg(x, y, k=3):
    """KSG estimator for I(X;Y) with jitter for tie-breaking."""
    from scipy.spatial import cKDTree
    from scipy.special import digamma
    n = len(x)
    if n < 10:
        return 0.0
    rng = np.random.RandomState(42)
    x = np.asarray(x, dtype=float).reshape(-1, 1) + rng.normal(0, 1e-9, (n, 1))
    y = np.asarray(y, dtype=float).reshape(-1, 1) + rng.normal(0, 1e-9, (n, 1))
    xy = np.hstack([x, y])
    tree_xy = cKDTree(xy)
    dists_xy, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = np.maximum(dists_xy[:, -1], 1e-12)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)
    nx = np.array([len(tree_x.query_ball_point(x[i], eps[i], p=np.inf)) - 1 for i in range(n)])
    ny = np.array([len(tree_y.query_ball_point(y[i], eps[i], p=np.inf)) - 1 for i in range(n)])
    mi = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return float(max(mi, 0.0))


def mutual_information_discrete(x_cont, y_disc, n_bins=20):
    """MI for continuous X and discrete Y (plugin + Miller-Madow)."""
    x_cont = np.asarray(x_cont, dtype=float)
    y_disc = np.asarray(y_disc).astype(int)
    n = len(x_cont)
    if n < 10:
        return 0.0
    x_min, x_max = x_cont.min(), x_cont.max()
    if x_max - x_min < 1e-12:
        return 0.0
    x_disc = np.clip(np.digitize(x_cont, np.linspace(x_min, x_max + 1e-9, n_bins + 1)[:-1]), 0, n_bins - 1)
    y_vals = np.unique(y_disc)
    x_vals = np.unique(x_disc)
    joint = np.zeros((len(x_vals), len(y_vals)))
    for i, xv in enumerate(x_vals):
        for j, yv in enumerate(y_vals):
            joint[i, j] = np.sum((x_disc == xv) & (y_disc == yv))
    joint /= n
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = joint / (px * py + 1e-15)
        log_ratio = np.where(joint > 1e-15, np.log2(np.maximum(ratio, 1e-15)), 0)
        mi = np.sum(joint * log_ratio)
    mm_correction = (len(x_vals) - 1) * (len(y_vals) - 1) / (2 * n)
    return float(max(mi + mm_correction, 0.0))
