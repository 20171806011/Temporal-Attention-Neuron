"""Fixed-weight D16 continuation with a single pre-cue state intervention."""

import numpy as np

import population
import scale_model


ENTRY_STEP = 25
QUERY_STEP = 28
STEPS = 36
BATCH = 32
ARCHIVE_ROWS = 8
ARMS = ('SCALE', 'SCALE_MEAN', 'GAIN2')
NATIVE_CONDITIONS = ('nominal', 'identity', 'native_selected', 'native_unselected')
TRACE_CONDITIONS = ('trace_selected', 'trace_unselected', 'all_selected', 'all_unselected')


def donors(base):
    rows = list(zip(base['octet'], base['selected'], base['first_bit'], base['second_bit']))
    lookup = {row: i for i, row in enumerate(rows)}
    if len(lookup) != len(rows):
        raise ValueError('duplicate stimulus metadata')
    selected = np.arange(len(rows), dtype=np.int64)
    unselected = selected.copy()
    for i, (octet, cue, first, second) in enumerate(rows):
        if first != second:
            flip_first = lookup[octet, cue, 1-first, second]
            flip_second = lookup[octet, cue, first, 1-second]
            if cue == 0:
                selected[i], unselected[i] = flip_first, flip_second
            else:
                selected[i], unselected[i] = flip_second, flip_first
    return dict(selected=selected, unselected=unselected)


def conditions(arm):
    if arm not in ARMS:
        raise ValueError(f'unknown retrieval arm: {arm}')
    if arm == 'GAIN2':
        return NATIVE_CONDITIONS
    values = NATIVE_CONDITIONS + TRACE_CONDITIONS
    return values + ('post_mean',) if arm == 'SCALE' else values


def _native_state(step):
    return np.concatenate((step['states'], step['hs']), axis=1)


def _archive(cache, trace_arm):
    result = dict(
        states=np.stack([_native_state(step)[:ARCHIVE_ROWS] for step in cache], axis=1),
        spikes=np.stack([np.concatenate((step['hs'], step['os']), axis=1)[:ARCHIVE_ROWS]
                         for step in cache], axis=1),
        injections=np.stack([step['A'][:ARCHIVE_ROWS] for step in cache], axis=1),
    )
    if trace_arm:
        result['traces'] = np.stack([step['traces'][:ARCHIVE_ROWS] for step in cache], axis=1)
        result['weights_fast'] = np.stack([step['weight_fast'][:ARCHIVE_ROWS] for step in cache], axis=1)
    return result


def _resources(cache, fanout_r, fanout_o):
    values = np.zeros((len(cache[0]['hs']), 4))
    for step in cache:
        values[:, 0] += step['hs'].sum(axis=1)
        values[:, 1] += step['os'].sum(axis=1)
        values[:, 2] += step['prev'] @ fanout_r
        values[:, 3] += step['prev'] @ fanout_o
    return values


def _resume(x, weights, arm, q, native, traces, post_mean):
    wi, wr, db, vb, wo, ob = weights
    d, v, o, p = (part.copy() for part in np.split(native, (12, 18, 20), axis=1))
    counts = np.zeros_like(o)
    trace_arm = traces is not None
    if trace_arm:
        beta = 0. if post_mean else scale_model.BETAS[arm]
        gates = np.minimum(1., np.abs(x).sum(axis=2, keepdims=True))
    cache = []
    for t in range(ENTRY_STEP, STEPS):
        z = x[:, t] @ wi + p @ wr
        current = z + db
        if trace_arm:
            drive = np.tanh(z)
            gate = gates[:, t]
            rate = .5*gate
            retention = (1.-rate)[..., None]*scale_model.RHO
            difference = traces[..., 0]-traces[..., 1]
            weight_fast = 1./(1.+np.exp(-beta*drive*difference))
            context = traces[..., 1]+weight_fast*difference
            injection = gate*context
            traces = retention*traces+(rate*drive)[..., None]
        else:
            injection = current

        d = .95*d+.25*(current+injection)
        local = np.maximum(d, 0.)
        vc = .85*v+.5*local.reshape(len(x), 6, 2).sum(axis=2)+vb
        oc = .85*o+p @ wo+ob
        hs = (vc > .5).astype(float)
        os = (oc > .5).astype(float)
        v = vc*(1.-hs)
        o = oc*(1.-os)
        if t >= q:
            counts += os
        step = dict(A=injection, hs=hs, os=os, prev=p, states=np.concatenate((d, v, o), axis=1))
        if trace_arm:
            step.update(traces=traces, weight_fast=weight_fast)
        cache.append(step)
        p = hs
    return counts, cache


def evaluate(x, w, arm, q, condition, donor_map):
    if condition not in conditions(arm):
        raise ValueError(f'condition {condition} is unavailable for {arm}')
    n = len(x)
    if x.ndim != 3 or x.shape[1:] != (STEPS, 6) or n == 0 or q != QUERY_STEP:
        raise ValueError('retrieval requires nonempty D16 inputs and query at t=28')
    weights = population.unpack(w, 6)
    fanout_r = np.count_nonzero(weights[1], axis=1)
    fanout_o = np.count_nonzero(weights[4], axis=1)
    trace_arm = arm != 'GAIN2'
    entry_native = np.empty((n, 26))
    entry_traces = np.empty((n, 12, 2)) if trace_arm else None
    resources = np.empty((n, 4))

    # Use the sealed implementation until the state just after the last blank.
    # Assemble the entire bank before gathering donors from across the batch.
    for start in range(0, n, BATCH):
        batch = slice(start, start+BATCH)
        _, cache = scale_model.forward(x[batch, :ENTRY_STEP], w, arm, q)
        entry_native[batch] = _native_state(cache[-1])
        if trace_arm:
            entry_traces[batch] = cache[-1]['traces']
        resources[batch] = _resources(cache, fanout_r, fanout_o)
        if start == 0:
            prefix = _archive(cache, trace_arm)

    if condition == 'identity':
        index = np.arange(n)
        entry_native = entry_native[index]
        if trace_arm:
            entry_traces = entry_traces[index]
    elif condition not in ('nominal', 'post_mean'):
        component, direction = condition.split('_', 1)
        index = np.asarray(donor_map[direction])
        if index.shape != (n,) or index.dtype.kind not in 'iu' or np.any(index < 0) or np.any(index >= n):
            raise ValueError(f'invalid {direction} donor index')
        if component in ('native', 'all'):
            entry_native = entry_native[index]
        if component in ('trace', 'all'):
            entry_traces = entry_traces[index]

    counts = np.empty((n, 2))
    for start in range(0, n, BATCH):
        batch = slice(start, start+BATCH)
        traces = entry_traces[batch] if trace_arm else None
        counts[batch], cache = _resume(x[batch], weights, arm, q, entry_native[batch], traces,
                                       condition == 'post_mean')
        resources[batch] += _resources(cache, fanout_r, fanout_o)
        if start == 0:
            suffix = _archive(cache, trace_arm)

    result = {key: np.concatenate((values, suffix[key]), axis=1) for key, values in prefix.items()}
    result.update(counts=counts, resources=resources, entry_native=entry_native)
    if trace_arm:
        result['entry_traces'] = entry_traces
    return result
