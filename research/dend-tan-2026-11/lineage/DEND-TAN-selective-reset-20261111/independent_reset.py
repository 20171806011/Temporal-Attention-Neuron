"""Independent replay of native state reset at visible read onset."""

import numpy as np

from reference import split


ARMS = ('LEARN', 'ORACLE', 'MEAN')
RULES = ('original', 'always', 'selective')
ROTATION = np.array([[0., -1.], [1., 0.]])
COSINE_THRESHOLD = 1/np.sqrt(2.)
NO_CONFIDENCE = -2.
TRACES = ('states', 'spikes', 'candidates', 'currents', 'injections', 'key_bank', 'value_bank',
          'scores', 'weights', 'read_gate', 'write_gate', 'drive')
CUE_ARRAYS = ('cue_scores', 'cue_weights', 'cue_read_vectors', 'cue_native_before', 'cue_native_after')


def matched_cosine(projected, keys):
    """Return the greatest valid cosine, using a sentinel for an empty match."""
    dot = np.einsum('ni,nki->nk', projected, keys, optimize=False)
    query_norm = np.sqrt(np.einsum('ni,ni->n', projected, projected, optimize=False))
    key_norm = np.sqrt(np.einsum('nki,nki->nk', keys, keys, optimize=False))
    denominator = query_norm[:, None]*key_norm
    cosine = np.full(dot.shape, NO_CONFIDENCE)
    np.divide(dot, denominator, out=cosine, where=denominator > 0.)
    return cosine.max(axis=1)


def replay(x, w, arm, q, rule):
    assert x.ndim == 3 and x.shape[-1] == 6 and 3 <= q < x.shape[1]
    assert arm in ARMS and rule in RULES and w.shape == (180,)
    wi, wr, db, vb, wo, ob = split(w[:176])
    matcher = {'LEARN': w[176:].reshape(2, 2), 'ORACLE': ROTATION, 'MEAN': np.zeros((2, 2))}[arm]
    n = len(x)
    d, v, o, p = np.zeros((n, 12)), np.zeros((n, 6)), np.zeros((n, 2)), np.zeros((n, 6))
    keys, values = np.zeros((n, 2, 2)), np.zeros((n, 12, 2))
    counts, resources = np.zeros((n, 2)), np.zeros((n, 6))
    recurrent_fanout = np.count_nonzero(wr, axis=1)
    output_fanout = np.count_nonzero(wo, axis=1)
    traces, cues = {name: [] for name in TRACES}, {name: [] for name in CUE_ARRAYS}
    onset_steps, resets, confidence = [], [], []
    previously_reading = np.zeros(n, dtype=bool)
    for tick in range(x.shape[1]):
        external = x[:, tick]
        observed, marker = external[:, 2:4], external[:, 5:6]
        writing = marker == 1.
        read = (1.-marker)*np.minimum(1., np.linalg.norm(observed, axis=1, keepdims=True))
        reading = read[:, 0] > 0.
        onset = reading & ~previously_reading
        previously_reading = reading
        projected = np.einsum('ni,ij->nj', observed, matcher, optimize=False)
        cosine = np.full(n, NO_CONFIDENCE)
        reset = np.zeros(n, dtype=bool)
        if rule == 'always':
            reset = onset.copy()
        if rule == 'selective':
            cosine[onset] = matched_cosine(projected[onset], keys[onset])
            reset = cosine >= COSINE_THRESHOLD
        onset_steps.append(onset)
        resets.append(reset)
        confidence.append(cosine)

        in_cue_archive = q-3 <= tick < q
        if in_cue_archive:
            cues['cue_native_before'].append(np.column_stack((d, v, o, p)))
        d, v, o, p = (np.where(reset[:, None], 0., state) for state in (d, v, o, p))
        if in_cue_archive:
            cues['cue_native_after'].append(np.column_stack((d, v, o, p)))

        scores = 4*np.einsum('ni,nki->nk', projected, keys, optimize=False)
        exp_scores = np.exp(scores-scores.max(axis=1, keepdims=True))
        weights = exp_scores/exp_scores.sum(axis=1, keepdims=True)
        injection = 2*read*np.einsum('nk,ndk->nd', weights, values, optimize=False)
        z = np.einsum('ni,ij->nj', external, wi, optimize=False)
        z += np.einsum('ni,ij->nj', p, wr, optimize=False)
        current, drive = z+db, np.tanh(z)
        earlier_key = np.where(writing, keys[:, 0], keys[:, 1])
        recent_key = np.where(writing, observed, keys[:, 0])
        earlier_value = .97*np.where(writing, values[..., 0], values[..., 1])
        recent_value = np.where(writing, drive, .97*values[..., 0])
        keys = np.stack((recent_key, earlier_key), axis=1)
        values = np.stack((recent_value, earlier_value), axis=-1)

        d = .95*d+.25*(current+injection)
        local = np.maximum(d, 0.)
        vc = .85*v+.5*(local[:, ::2]+local[:, 1::2])+vb
        oc = .85*o+np.einsum('ni,ij->nj', p, wo, optimize=False)+ob
        hs, os = (vc > .5).astype(float), (oc > .5).astype(float)
        v, o = vc*(1.-hs), oc*(1.-os)
        if tick >= q:
            counts += os
        resources[:, 0] += hs.sum(axis=1)
        resources[:, 1] += os.sum(axis=1)
        resources[:, 2] += np.einsum('ni,i->n', p, recurrent_fanout, optimize=False)
        resources[:, 3] += np.einsum('ni,i->n', p, output_fanout, optimize=False)
        if 16 <= tick < q:
            resources[:, 4] += hs.sum(axis=1)
            resources[:, 5] += os.sum(axis=1)
        if in_cue_archive:
            cues['cue_scores'].append(scores)
            cues['cue_weights'].append(weights)
            cues['cue_read_vectors'].append(injection)
        step = dict(states=np.column_stack((d, v, o)), spikes=np.column_stack((hs, os)),
                    candidates=np.column_stack((d, vc, oc)), currents=current, injections=injection,
                    key_bank=keys, value_bank=values, scores=scores, weights=weights,
                    read_gate=read, write_gate=marker, drive=drive)
        for name, array in step.items():
            traces[name].append(array[:8])
        p = hs
    return dict(counts=counts, resources=resources, read_onset=np.stack(onset_steps, axis=1),
                reset_steps=np.stack(resets, axis=1), match_confidence=np.stack(confidence, axis=1),
                **{name: np.stack(arrays, axis=1) for name, arrays in traces.items()},
                **{name: np.stack(arrays, axis=1) for name, arrays in cues.items()})
