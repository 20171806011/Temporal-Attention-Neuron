"""Cross native and trace states from dynamic and mean pre-cue prefixes."""

import numpy as np

import population
import scale_model
from intervene import (BATCH, ENTRY_STEP, QUERY_STEP, STEPS, _archive,
                       _native_state, _resources, _resume)


# D is the original prefix and M reads both traces with equal weight. Each
# condition names the source of its native state first, then its trace state.
ORIGINS = {
    'SCALE': {
        'nominal': ('D', 'D'),
        'prefix_mean': ('M', 'M'),
        'mean_native': ('M', 'D'),
        'mean_traces': ('D', 'M'),
    },
    'SCALE_MEAN': {
        'nominal': ('D', 'D'),
        'prefix_mean': ('M', 'M'),
    },
    'GAIN2': {'nominal': ('D', 'D')},
}
CONDITIONS = {arm: tuple(origins) for arm, origins in ORIGINS.items()}


def conditions(arm):
    if arm not in CONDITIONS:
        raise ValueError(f'unknown prefix arm: {arm}')
    return CONDITIONS[arm]


def _prefix_bank(x, w, prefix_arm, q, fanout_r, fanout_o):
    n = len(x)
    trace_arm = prefix_arm != 'GAIN2'
    native = np.empty((n, 26))
    traces = np.empty((n, 12, 2)) if trace_arm else None
    resources = np.empty((n, 4))

    for start in range(0, n, BATCH):
        batch = slice(start, start+BATCH)
        _, cache = scale_model.forward(x[batch, :ENTRY_STEP], w, prefix_arm, q)
        native[batch] = _native_state(cache[-1])
        if trace_arm:
            traces[batch] = cache[-1]['traces']
        resources[batch] = _resources(cache, fanout_r, fanout_o)
        if start == 0:
            archive = _archive(cache, trace_arm)

    return dict(native=native, traces=traces, resources=resources, archive=archive)


def _continue(x, weights, arm, q, native_bank, trace_bank, fanout_r, fanout_o):
    native = native_bank['native']
    traces = trace_bank['traces']
    resources = native_bank['resources'].copy()
    counts = np.empty((len(x), 2))

    for start in range(0, len(x), BATCH):
        batch = slice(start, start+BATCH)
        entry_traces = traces[batch] if traces is not None else None
        counts[batch], cache = _resume(x[batch], weights, arm, q, native[batch], entry_traces,
                                       post_mean=False)
        resources[batch] += _resources(cache, fanout_r, fanout_o)
        if start == 0:
            suffix = _archive(cache, traces is not None)

    result = {key: np.concatenate((value, suffix[key]), axis=1)
              for key, value in native_bank['archive'].items()}
    result.update(counts=counts, resources=resources, entry_native=native)
    if traces is not None:
        result['entry_traces'] = traces
    return result


def evaluate(x, w, arm, q):
    names = conditions(arm)
    if x.ndim != 3 or x.shape[1:] != (STEPS, 6) or len(x) == 0 or q != QUERY_STEP:
        raise ValueError('prefix crossing requires nonempty D16 inputs and query at t=28')
    weights = population.unpack(w, 6)
    fanout_r = np.count_nonzero(weights[1], axis=1)
    fanout_o = np.count_nonzero(weights[4], axis=1)

    banks = {'D': _prefix_bank(x, w, arm, q, fanout_r, fanout_o)}
    if arm != 'GAIN2':
        # SCALE_MEAN also runs its second prefix separately, as a sham control.
        banks['M'] = _prefix_bank(x, w, 'SCALE_MEAN', q, fanout_r, fanout_o)

    results = {}
    for name in names:
        native_origin, trace_origin = ORIGINS[arm][name]
        results[name] = _continue(x, weights, arm, q, banks[native_origin], banks[trace_origin],
                                  fanout_r, fanout_o)
    return results
