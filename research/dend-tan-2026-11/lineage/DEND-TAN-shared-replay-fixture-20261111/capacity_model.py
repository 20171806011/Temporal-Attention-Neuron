"""Frozen learned-key inference with two or four FIFO slots."""

import numpy as np

import learned_model
import population


CONDITIONS = {
    'LEARN': ('base2', 'full2', 'full4', 'full4_hard', 'full4_hard_matched'),
    'ORACLE': ('base2', 'full2', 'full4', 'full4_hard'),
    'MEAN': ('base2', 'full2', 'full4'),
}
TWO_SLOT = ('base2', 'full2')
HARD_READ = ('full4_hard', 'full4_hard_matched')


def conditions(arm):
    return CONDITIONS[arm]


def read_weights(scores, use_hard):
    maxima = scores.max(axis=1, keepdims=True)
    exponentials = np.exp(scores - maxima)
    soft_weights = exponentials / exponentials.sum(axis=1, keepdims=True)
    if use_hard:
        highest = scores == maxima
        return highest / highest.sum(axis=1, keepdims=True), soft_weights
    return soft_weights, soft_weights


def match_norm(direction, target):
    direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
    target_norm = np.linalg.norm(target, axis=1, keepdims=True)
    scale = np.divide(target_norm, direction_norm, out=np.zeros_like(target_norm), where=direction_norm > 0)
    injection = direction * scale
    fallback = (direction_norm[:, 0] == 0) & (target_norm[:, 0] > 0)
    injection[fallback, 0] = target_norm[fallback, 0]
    return injection, fallback


def _forward_four(x, w, arm, q, condition):
    source = learned_model
    matrix = w[source.NATIVE_PARAMETERS:].reshape(2, 2) if arm == 'LEARN' else {
        'ORACLE': source.ORACLE_MATRIX, 'MEAN': source.MEAN_MATRIX,
    }[arm]
    wi, wr, db, vb, wo, ob = population.unpack(w[:source.NATIVE_PARAMETERS], 6)
    queries = x[..., 2:4]
    projected_queries = queries @ matrix
    write_gates = x[..., 5:6]
    read_gates = (1. - write_gates) * np.minimum(1., np.linalg.norm(queries, axis=-1, keepdims=True))
    n, steps, _ = x.shape
    d, v, o = np.zeros((n, 12)), np.zeros((n, 6)), np.zeros((n, 2))
    p = np.zeros_like(v)
    key_bank = np.zeros((n, 4, 2))
    value_bank = np.zeros((n, 12, 4))
    counts = np.zeros_like(o)
    cache = []
    no_fallback = np.zeros(n, dtype=bool)
    use_hard = condition in HARD_READ
    match_amplitude = condition == 'full4_hard_matched'

    for t in range(steps):
        z = x[:, t] @ wi + p @ wr
        current = z + db
        drive = np.tanh(z)
        read_gate = read_gates[:, t]
        write_gate = write_gates[:, t]
        writers = write_gate[:, 0] == 1.
        keys_before = key_bank
        values_before = value_bank

        scores = source.TEMPERATURE * (keys_before @ projected_queries[:, t, :, None])[..., 0]
        weights, soft_weights = read_weights(scores, use_hard)
        gain = source.READ_GAIN * read_gate
        context = np.sum(values_before * weights[:, None, :], axis=2)
        injection = gain * context
        fallback = no_fallback
        if match_amplitude:
            soft_context = np.sum(values_before * soft_weights[:, None, :], axis=2)
            injection, fallback = match_norm(injection, gain * soft_context)

        value_bank = source.MEMORY_RETENTION * values_before
        if writers.any():
            value_bank[writers, :, 1:] = value_bank[writers, :, :-1]
            value_bank[writers, :, 0] = drive[writers]
            key_bank = keys_before.copy()
            key_bank[writers, 1:] = keys_before[writers, :-1]
            key_bank[writers, 0] = queries[writers, t]

        d = .95 * d + .25 * (current + injection)
        local = np.maximum(d, 0.)
        vc = .85 * v + .5 * local.reshape(n, 6, 2).sum(axis=2) + vb
        oc = .85 * o + p @ wo + ob
        hs, os = (vc > .5).astype(float), (oc > .5).astype(float)
        v, o = vc * (1. - hs), oc * (1. - os)
        if t >= q:
            counts += os
        cache.append(dict(I=current, A=injection, dc=d, vc=vc, oc=oc, hs=hs, os=os, prev=p,
                          states=np.concatenate((d, v, o), axis=1), key_bank=key_bank, value_bank=value_bank,
                          scores=scores, weights=weights, soft_weights=soft_weights, fallback=fallback,
                          read_gate=read_gate, write_gate=write_gate, drive=drive))
        p = hs
    return counts, cache


def evaluate(x, w, arm, q, condition):
    if condition not in conditions(arm):
        raise ValueError(f'condition {condition} is unavailable for {arm}')
    _, wr, _, _, wo, _ = population.unpack(w[:learned_model.NATIVE_PARAMETERS], 6)
    fanout_r = np.count_nonzero(wr, axis=1)
    fanout_o = np.count_nonzero(wo, axis=1)
    cue_steps = learned_model.CUE_STEPS
    two_slot = condition in TWO_SLOT
    cue_fields = {'cue_scores': 'scores', 'cue_weights': 'weights', 'cue_read_vectors': 'A'}
    traces = {'states': 'states', 'currents': 'I', 'injections': 'A', 'key_bank': 'key_bank',
              'value_bank': 'value_bank', 'scores': 'scores', 'weights': 'weights',
              'read_gate': 'read_gate', 'write_gate': 'write_gate', 'drive': 'drive'}
    chunks = {key: [] for key in ('counts', 'resources', *cue_fields, 'cue_soft_weights', 'cue_fallback')}

    for start in range(0, len(x), 32):
        batch = x[start:start+32]
        if two_slot:
            counts, cache = learned_model.forward(batch, w, arm, q)
        else:
            counts, cache = _forward_four(batch, w, arm, q, condition)
        chunks['counts'].append(counts)
        for name, field in cue_fields.items():
            chunks[name].append(np.stack([cache[t][field] for t in cue_steps], axis=1))
        soft_field = 'weights' if two_slot else 'soft_weights'
        chunks['cue_soft_weights'].append(np.stack([cache[t][soft_field] for t in cue_steps], axis=1))
        fallback = (np.zeros((len(batch), len(cue_steps)), dtype=bool) if two_slot else
                    np.stack([cache[t]['fallback'] for t in cue_steps], axis=1))
        chunks['cue_fallback'].append(fallback)

        resources = np.zeros((len(batch), 6))
        for t, step in enumerate(cache):
            resources[:, 0] += step['hs'].sum(axis=1)
            resources[:, 1] += step['os'].sum(axis=1)
            resources[:, 2] += step['prev'] @ fanout_r
            resources[:, 3] += step['prev'] @ fanout_o
            if 16 <= t < q:
                resources[:, 4] += step['hs'].sum(axis=1)
                resources[:, 5] += step['os'].sum(axis=1)
        chunks['resources'].append(resources)
        if start == 0:
            saved = {name: np.stack([step[field][:8] for step in cache], axis=1)
                     for name, field in traces.items()}
            saved['spikes'] = np.stack([np.concatenate((step['hs'][:8], step['os'][:8]), axis=1)
                                        for step in cache], axis=1)
            saved['candidates'] = np.stack([np.concatenate((step['dc'][:8], step['vc'][:8], step['oc'][:8]), axis=1)
                                            for step in cache], axis=1)
    saved.update({key: np.concatenate(values) for key, values in chunks.items()})
    return saved
