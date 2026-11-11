"""Evaluate native-state replacement using observed read onsets and slot eviction."""

import numpy as np

import learned_model
import population


RULES = ('original', 'match_zero', 'eviction_zero', 'eviction_warm', 'eviction_late_native')
COSINE_THRESHOLD = 1./np.sqrt(2.)
INVALID_CONFIDENCE = -2.
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
    'cue_native_before': 'native_before',
    'cue_native_after': 'native_after',
}


def cosine_confidence(projected, keys, dots=None):
    """Take the highest valid cosine for each query and its two stored keys."""
    if dots is None:
        dots = (keys @ projected[..., None])[..., 0]
    norms = np.linalg.norm(projected, axis=1, keepdims=True) * np.linalg.norm(keys, axis=2)
    cosines = np.divide(dots, norms, out=np.full_like(norms, INVALID_CONFIDENCE), where=norms > 0.)
    return cosines.max(axis=1)


def warm_reference(w):
    """Return the native state after 25 zero-input steps from the zero state."""
    blank = np.zeros((1, 25, 6))
    _, cache = learned_model.forward(blank, w, 'MEAN', 28)
    final = cache[-1]
    return np.concatenate((final['states'][0], final['hs'][0]))


def _forward(x, weights, matrix, q, rule, reference):
    wi, wr, db, vb, wo, ob = weights
    queries = x[..., 2:4]
    projected_queries = queries @ matrix
    write_gates = x[..., 5:6]
    read_gates = (1. - write_gates) * np.minimum(1., np.linalg.norm(queries, axis=-1, keepdims=True))
    n, steps, _ = x.shape
    d, v, o = np.zeros((n, 12)), np.zeros((n, 6)), np.zeros((n, 2))
    p = np.zeros_like(v)
    key_bank = np.zeros((n, 2, 2))
    value_bank = np.zeros((n, 12, 2))
    previous_read = np.zeros(n, dtype=bool)
    read_onset = np.zeros((n, steps), dtype=bool)
    reset_steps = np.zeros_like(read_onset)
    eviction_seen = np.zeros_like(read_onset)
    evicted = np.zeros(n, dtype=bool)
    match_confidence = np.full((n, steps), INVALID_CONFIDENCE)
    counts = np.zeros_like(o)
    cache = []

    for t in range(steps):
        read_gate = read_gates[:, t]
        active_read = read_gate[:, 0] > 0.
        onset = active_read & ~previous_read
        read_onset[:, t] = onset
        eviction_seen[:, t] = evicted
        previous_read = active_read
        keys_before = key_bank
        values_before = value_bank
        projected = projected_queries[:, t]
        dots = (keys_before @ projected[..., None])[..., 0]
        reset = reset_steps[:, t]
        if rule != 'original':
            eligible = onset if rule == 'match_zero' else onset & evicted
            if eligible.any():
                confidence = cosine_confidence(projected[eligible], keys_before[eligible], dots[eligible])
                match_confidence[eligible, t] = confidence
                reset[eligible] = confidence >= COSINE_THRESHOLD

        archive_cue = t in learned_model.CUE_STEPS
        if archive_cue:
            native_before = np.concatenate((d, v, o, p), axis=1)
        if reset.any():
            # Copy first because the preceding step's archive retains these arrays.
            d, v, o, p = (state.copy() for state in (d, v, o, p))
            if reference is None:
                for state in (d, v, o, p):
                    state[reset] = 0.
            else:
                replacement = reference if reference.ndim == 1 else reference[reset]
                components = np.split(replacement, (12, 18, 20), axis=-1)
                for state, component in zip((d, v, o, p), components):
                    state[reset] = component
        if archive_cue:
            native_after = np.concatenate((d, v, o, p), axis=1)

        z = x[:, t] @ wi + p @ wr
        current = z + db
        drive = np.tanh(z)
        write_gate = write_gates[:, t]
        writers = write_gate[:, 0] == 1.
        scores = learned_model.TEMPERATURE * dots
        exponentials = np.exp(scores - scores.max(axis=1, keepdims=True))
        attention = exponentials / exponentials.sum(axis=1, keepdims=True)
        context = values_before[..., 0] * attention[:, :1] + values_before[..., 1] * attention[:, 1:]
        injection = learned_model.READ_GAIN * read_gate * context

        value_bank = learned_model.MEMORY_RETENTION * values_before
        if writers.any():
            occupied = np.any(keys_before[writers] != 0., axis=2)
            evicted[writers] |= occupied.all(axis=1)
            value_bank[writers, :, 1] = value_bank[writers, :, 0]
            value_bank[writers, :, 0] = drive[writers]
            key_bank = keys_before.copy()
            key_bank[writers, 1] = keys_before[writers, 0]
            key_bank[writers, 0] = queries[writers, t]

        d = .95 * d + .25 * (current + injection)
        local = np.maximum(d, 0.)
        vc = .85 * v + .5 * local.reshape(n, 6, 2).sum(axis=2) + vb
        oc = .85 * o + p @ wo + ob
        hs, os = (vc > .5).astype(float), (oc > .5).astype(float)
        v, o = vc * (1. - hs), oc * (1. - os)
        if t >= q:
            counts += os
        saved = dict(I=current, A=injection, dc=d, vc=vc, oc=oc, hs=hs, os=os, prev=p,
                     states=np.concatenate((d, v, o), axis=1), key_bank=key_bank, value_bank=value_bank,
                     scores=scores, weights=attention, read_gate=read_gate, write_gate=write_gate, drive=drive)
        if archive_cue:
            saved.update(native_before=native_before, native_after=native_after)
        cache.append(saved)
        p = hs

    return counts, cache, dict(read_onset=read_onset, reset_steps=reset_steps, match_confidence=match_confidence,
                               eviction_seen=eviction_seen)


def _trace(cache):
    saved = {name: np.stack([state[field][:8] for state in cache], axis=1)
             for name, field in TRACE_FIELDS.items()}
    saved['spikes'] = np.stack([np.concatenate((state['hs'][:8], state['os'][:8]), axis=1)
                                for state in cache], axis=1)
    saved['candidates'] = np.stack([np.concatenate((state['dc'][:8], state['vc'][:8], state['oc'][:8]), axis=1)
                                    for state in cache], axis=1)
    return saved


def _resources(cache, q, fanout_r, fanout_o):
    resources = np.zeros((len(cache[0]['hs']), 6))
    for t, step in enumerate(cache):
        hidden = step['hs'].sum(axis=1)
        output = step['os'].sum(axis=1)
        resources[:, 0] += hidden
        resources[:, 1] += output
        resources[:, 2] += step['prev'] @ fanout_r
        resources[:, 3] += step['prev'] @ fanout_o
        if 16 <= t < q:
            resources[:, 4] += hidden
            resources[:, 5] += output
    return resources


def evaluate(x, w, arm, q, rule, reference=None):
    if rule not in RULES:
        raise ValueError(f'unknown reset rule {rule}')
    if x.shape[1:] != (36, 6) or len(x) == 0:
        raise ValueError('the read-onset study requires nonempty 36-step inputs')
    if rule in ('eviction_warm', 'eviction_late_native'):
        if reference is None and rule == 'eviction_warm':
            reference = warm_reference(w)
        reference = np.asarray(reference, dtype=float)
        expected = (26,) if rule == 'eviction_warm' else (len(x), 26)
        if reference.shape != expected or not np.isfinite(reference).all():
            raise ValueError(f'{rule} requires a finite reference with shape {expected}')
    elif reference is not None:
        raise ValueError(f'{rule} does not use a reference state')
    matrix = w[learned_model.NATIVE_PARAMETERS:].reshape(2, 2) if arm == 'LEARN' else {
        'ORACLE': learned_model.ORACLE_MATRIX, 'MEAN': learned_model.MEAN_MATRIX,
    }[arm]
    weights = population.unpack(w[:learned_model.NATIVE_PARAMETERS], 6)
    fanout_r = np.count_nonzero(weights[1], axis=1)
    fanout_o = np.count_nonzero(weights[4], axis=1)
    fields = ('counts', 'resources', *CUE_FIELDS, 'read_onset', 'reset_steps', 'match_confidence', 'eviction_seen')
    chunks = {name: [] for name in fields}

    for start in range(0, len(x), 32):
        batch = x[start:start + 32]
        local_reference = reference[start:start + 32] if rule == 'eviction_late_native' else reference
        counts, cache, reset_diagnostics = _forward(batch, weights, matrix, q, rule, local_reference)
        chunks['counts'].append(counts)
        chunks['resources'].append(_resources(cache, q, fanout_r, fanout_o))
        for name, field in CUE_FIELDS.items():
            chunks[name].append(np.stack([cache[t][field] for t in learned_model.CUE_STEPS], axis=1))
        for name, values in reset_diagnostics.items():
            chunks[name].append(values)
        if start == 0:
            saved = _trace(cache)
    saved.update({name: np.concatenate(values) for name, values in chunks.items()})
    return saved
