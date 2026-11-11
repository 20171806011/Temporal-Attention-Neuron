"""Reference streaming readouts for two-channel, eight-step binary pulses."""

from time import perf_counter_ns

import numpy as np

from timing_stats import TIE, fit_class


WEIGHTS = (-7, -5, -3, -1, 1, 3, 5, 7)
SCALES = {'count2': np.array([4., 4.]), 'moment2': np.array([4., np.sqrt(336.)])}
METHODS = ('native', 'count2', 'moment2')


def accumulate(raster, moment=False):
    """Consume each step in order. The readout retains only its accumulators."""
    if raster.ndim != 3 or raster.shape[1:] != (8, 2):
        raise ValueError('expected an N×8×2 pulse raster')
    if raster.dtype.kind not in 'biu':
        raise TypeError('expected binary integer or boolean pulse samples')
    counts = np.zeros((len(raster), 2), dtype=np.int32)
    time_moment = np.zeros(len(raster), dtype=np.int32) if moment else None
    for tick, weight in enumerate(WEIGHTS):
        pulses = raster[:, tick, :].astype(np.int32, copy=False)
        counts += pulses
        if moment:
            difference = pulses[:, 1] - pulses[:, 0]
            time_moment += weight * difference
    return counts, time_moment


def _integer_terms(raster, kind):
    if kind not in SCALES:
        raise ValueError(f'unknown fitted readout: {kind}')
    counts, time_moment = accumulate(raster, moment=(kind == 'moment2'))
    difference = counts[:, 1] - counts[:, 0]
    second = time_moment if kind == 'moment2' else counts[:, 1] + counts[:, 0]
    return difference, second


def raw_features(raster, kind):
    return np.column_stack(_integer_terms(raster, kind))


def fit_rule(raster, y, kind):
    features = raw_features(raster, kind) / SCALES[kind]
    mean, coef, intercept = fit_class(features, np.asarray(y, dtype=float))
    raw_coef = np.r_[coef / SCALES[kind], intercept - mean @ coef]
    return mean, coef, intercept, raw_coef


def stream_scores(raster, kind, raw_coef):
    first, second = _integer_terms(raster, kind)
    a, b, intercept = raw_coef
    return a * first + b * second + intercept


def predict(raster, kind, raw_coef=None):
    if kind == 'native':
        counts, _ = accumulate(raster)
        return (counts[:, 1] > counts[:, 0]).astype(np.uint8)
    return (stream_scores(raster, kind, raw_coef) > TIE).astype(np.uint8)


def benchmark(raster, count_raw, moment_raw):
    """Return elapsed batch ns in native/COUNT2/MOMENT2 order."""
    coefficients = (None, count_raw, moment_raw)
    for method, coef in zip(METHODS, coefficients):
        for _ in range(3):
            predict(raster, method, coef)
    elapsed = np.empty((3, 64), dtype=np.int64)
    for repeat in range(elapsed.shape[1]):
        for offset in range(3):
            index = (repeat + offset) % 3
            start = perf_counter_ns()
            predict(raster, METHODS[index], coefficients[index])
            elapsed[index, repeat] = perf_counter_ns() - start
    return elapsed
