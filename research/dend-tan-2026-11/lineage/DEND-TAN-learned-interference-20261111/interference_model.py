"""Cross the native and value states at the first query cue."""

import numpy as np

import learned_model
import population


BOUNDARY = 25
CONDITIONS = {
    'full_full': ('full', 'full'),
    'late_full': ('late', 'full'),
    'full_late': ('full', 'late'),
    'late_late': ('late', 'late'),
}
TRACE_FIELDS = {
    'states': 'states',
    'currents': 'I',
    'injections': 'A',
    'key_bank': 'key_bank',
    'value_bank': 'value_bank',
    'scores': 'scores',
    'weights': 'weights',
    'read_gate': 'read_gate',
    'write_gate': 'write_gate',
    'drive': 'drive',
}
CUE_FIELDS = {
    'cue_scores': 'scores',
    'cue_weights': 'weights',
    'cue_read_vectors': 'A',
}


def _continue(x, weights, matrix, q, native, keys, values):
    wi, wr, db, vb, wo, ob = weights
    d, v, o, p = np.split(native, (12, 18, 20), axis=1)
    queries = x[..., 2:4]
    projected = queries @ matrix
    write_gates = x[..., 5:6]
    read_gates = (1. - write_gates) * np.minimum(1., np.linalg.norm(queries, axis=-1, keepdims=True))
    counts = np.zeros_like(o)
    cache = []

    for step in range(x.shape[1]):
        z = x[:, step] @ wi + p @ wr
        current = z + db
        drive = np.tanh(z)
        read_gate = read_gates[:, step]
        scores = learned_model.TEMPERATURE * (keys @ projected[:, step, :, None])[..., 0]
        exponentials = np.exp(scores - scores.max(axis=1, keepdims=True))
        attention = exponentials / exponentials.sum(axis=1, keepdims=True)
        context = values[..., 0] * attention[:, :1] + values[..., 1] * attention[:, 1:]
        injection = learned_model.READ_GAIN * read_gate * context
        values = learned_model.MEMORY_RETENTION * values

        d = .95 * d + .25 * (current + injection)
        local = np.maximum(d, 0.)
        vc = .85 * v + .5 * local.reshape(len(x), 6, 2).sum(axis=2) + vb
        oc = .85 * o + p @ wo + ob
        hs, os = (vc > .5).astype(float), (oc > .5).astype(float)
        v, o = vc * (1. - hs), oc * (1. - os)
        if BOUNDARY + step >= q:
            counts += os
        cache.append(dict(I=current, A=injection, dc=d, vc=vc, oc=oc, hs=hs, os=os, prev=p,
                          states=np.concatenate((d, v, o), axis=1), key_bank=keys, value_bank=values,
                          scores=scores, weights=attention, read_gate=read_gate,
                          write_gate=write_gates[:, step], drive=drive))
        p = hs
    return counts, cache


def _resources(cache, offset, q, fanout_r, fanout_o):
    totals = np.zeros((len(cache[0]['hs']), 6))
    for tick, state in enumerate(cache, start=offset):
        hidden = state['hs'].sum(axis=1)
        output = state['os'].sum(axis=1)
        totals[:, 0] += hidden
        totals[:, 1] += output
        totals[:, 2] += state['prev'] @ fanout_r
        totals[:, 3] += state['prev'] @ fanout_o
        if 16 <= tick < q:
            totals[:, 4] += hidden
            totals[:, 5] += output
    return totals


def _trace(cache):
    saved = {name: np.stack([state[field][:8] for state in cache], axis=1)
             for name, field in TRACE_FIELDS.items()}
    saved['spikes'] = np.stack([np.concatenate((state['hs'][:8], state['os'][:8]), axis=1)
                                for state in cache], axis=1)
    saved['candidates'] = np.stack([np.concatenate((state['dc'][:8], state['vc'][:8], state['oc'][:8]), axis=1)
                                    for state in cache], axis=1)
    return saved


def evaluate(full_x, late_x, w, arm, q):
    if full_x.shape != late_x.shape or full_x.shape[1:] != (36, 6) or q != 28 or len(full_x) == 0:
        raise ValueError('the crossing requires nonempty paired 36-step inputs and query start 28')
    if not np.array_equal(full_x[:, BOUNDARY:], late_x[:, BOUNDARY:]):
        raise ValueError('paired suffix inputs must agree exactly')
    if np.any(full_x[:, BOUNDARY:, 5]):
        raise ValueError('the common suffix must contain no write markers')

    matrix = w[learned_model.NATIVE_PARAMETERS:].reshape(2, 2) if arm == 'LEARN' else {
        'ORACLE': learned_model.ORACLE_MATRIX, 'MEAN': learned_model.MEAN_MATRIX,
    }[arm]
    weights = population.unpack(w[:learned_model.NATIVE_PARAMETERS], 6)
    fanout_r = np.count_nonzero(weights[1], axis=1)
    fanout_o = np.count_nonzero(weights[4], axis=1)
    fields = ('counts', 'resources', *CUE_FIELDS, 'entry_native', 'entry_keys', 'entry_values')
    chunks = {condition: {field: [] for field in fields} for condition in CONDITIONS}
    traces = {}

    for start in range(0, len(full_x), 32):
        batch = slice(start, start + 32)
        prefixes = {
            'full': learned_model.forward(full_x[batch, :BOUNDARY], w, arm, q)[1],
            'late': learned_model.forward(late_x[batch, :BOUNDARY], w, arm, q)[1],
        }
        boundary = {source: cache[-1] for source, cache in prefixes.items()}
        common_keys = boundary['full']['key_bank']
        if not np.array_equal(common_keys, boundary['late']['key_bank']):
            raise ValueError('boundary keys must agree exactly')
        native_entries = {source: np.concatenate((state['states'], state['hs']), axis=1)
                          for source, state in boundary.items()}
        prefix_resources = {source: _resources(cache, 0, q, fanout_r, fanout_o)
                            for source, cache in prefixes.items()}
        suffix_inputs = full_x[batch, BOUNDARY:]

        for condition, (native_source, value_source) in CONDITIONS.items():
            entry_native = native_entries[native_source]
            entry_values = boundary[value_source]['value_bank']
            counts, suffix = _continue(suffix_inputs, weights, matrix, q, entry_native, common_keys, entry_values)
            saved = chunks[condition]
            saved['counts'].append(counts)
            saved['resources'].append(prefix_resources[native_source] + _resources(suffix, BOUNDARY, q, fanout_r, fanout_o))
            for name, field in CUE_FIELDS.items():
                saved[name].append(np.stack([suffix[tick - BOUNDARY][field] for tick in learned_model.CUE_STEPS], axis=1))
            saved['entry_native'].append(entry_native)
            saved['entry_keys'].append(common_keys)
            saved['entry_values'].append(entry_values)
            if start == 0:
                traces[condition] = _trace(prefixes[native_source] + suffix)

    for condition, fields in chunks.items():
        traces[condition].update({name: np.concatenate(values) for name, values in fields.items()})
    return traces
