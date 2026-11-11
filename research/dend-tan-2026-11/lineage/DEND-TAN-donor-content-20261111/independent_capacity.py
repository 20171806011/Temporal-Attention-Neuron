"""Independent native replay with arbitrary slot contractions and FIFO shift."""

import numpy as np

from reference import split


CONDITIONS = {
    'LEARN': ('base2', 'full2', 'full4', 'full4_hard', 'full4_hard_matched'),
    'ORACLE': ('base2', 'full2', 'full4', 'full4_hard'),
    'MEAN': ('base2', 'full2', 'full4'),
}
ROTATION = np.array([[0., -1.], [1., 0.]])


def slot_count(condition):
    return 2 if condition in ('base2', 'full2') else 4


def hard_weights(scores):
    maximum = scores.max(axis=-1, keepdims=True)
    tied = scores == maximum
    return tied/tied.sum(axis=-1, keepdims=True)


def norm(vectors):
    return np.sqrt(np.einsum('...d,...d->...', vectors, vectors, optimize=False))


def matched_injection(direction, target):
    direction_norm, target_norm = norm(direction), norm(target)
    fallback = (direction_norm == 0.) & (target_norm > 0.)
    ratios = np.divide(target_norm, direction_norm, out=np.zeros_like(target_norm), where=direction_norm > 0.)
    scaled = direction*ratios[..., None]
    scaled[..., 0] = np.where(fallback, target_norm, scaled[..., 0])
    return scaled, fallback


def replay(x, w, arm, q, condition='full4'):
    assert condition in CONDITIONS[arm] and q == 28 and x.shape[1:] == (36, 6) and w.shape == (180,)
    slots = slot_count(condition)
    hard = condition in ('full4_hard', 'full4_hard_matched')
    matched = condition == 'full4_hard_matched'
    matcher = {'LEARN': w[176:].reshape(2, 2), 'ORACLE': ROTATION, 'MEAN': np.zeros((2, 2))}[arm]
    wi, wr, db, vb, wo, ob = split(w[:176])
    n = len(x)
    d, v, o = np.zeros((n, 12)), np.zeros((n, 6)), np.zeros((n, 2))
    p, counts = np.zeros_like(v), np.zeros_like(o)
    key_bank, value_bank = np.zeros((n, slots, 2)), np.zeros((n, 12, slots))
    resources = np.zeros((n, 6))
    recurrent_fanout = np.count_nonzero(wr, axis=1)
    output_fanout = np.count_nonzero(wo, axis=1)
    names = ('states', 'spikes', 'candidates', 'currents', 'injections', 'key_bank', 'value_bank',
             'scores', 'weights', 'read_gate', 'write_gate', 'drive')
    history = {name: [] for name in names}
    cues = {name: [] for name in ('cue_scores', 'cue_weights', 'cue_soft_weights', 'cue_read_vectors', 'cue_fallback')}
    for tick in range(36):
        external = x[:, tick]
        observed, marker = external[:, 2:4], external[:, 5:6]
        writing = marker == 1.
        read = (1-marker)*np.minimum(1., norm(observed)[:, None])
        scores = 4*np.einsum('ni,ij,nkj->nk', observed, matcher, key_bank, optimize=False)
        exponentials = np.exp(scores-scores.max(axis=-1, keepdims=True))
        soft_weights = exponentials/exponentials.sum(axis=-1, keepdims=True)
        weights = hard_weights(scores) if hard else soft_weights
        injection = 2*read*np.einsum('nk,ndk->nd', weights, value_bank, optimize=False)
        if matched:
            soft_injection = 2*read*np.einsum('nk,ndk->nd', soft_weights, value_bank, optimize=False)
            injection, fallback = matched_injection(injection, soft_injection)
        else:
            fallback = np.zeros(n, dtype=bool)

        z = np.einsum('ni,ij->nj', external, wi, optimize=False)+np.einsum('ni,ij->nj', p, wr, optimize=False)
        current, drive = z+db, np.tanh(z)
        key_shift = np.concatenate((observed[:, None, :], key_bank[:, :-1, :]), axis=1)
        key_bank = np.where(writing[..., None], key_shift, key_bank)
        retained = .97*value_bank
        value_shift = np.concatenate((drive[..., None], retained[..., :-1]), axis=-1)
        value_bank = np.where(writing[..., None], value_shift, retained)

        d = .95*d+.25*(current+injection)
        local = np.where(d > 0., d, 0.)
        vc = .85*v+.5*(local[:, ::2]+local[:, 1::2])+vb
        oc = .85*o+np.einsum('ni,ij->nj', p, wo, optimize=False)+ob
        hs, os = (vc > .5).astype(float), (oc > .5).astype(float)
        v, o = vc*(1-hs), oc*(1-os)
        if tick >= q:
            counts += os
        resources[:, 0] += hs.sum(axis=1)
        resources[:, 1] += os.sum(axis=1)
        resources[:, 2] += np.einsum('ni,i->n', p, recurrent_fanout, optimize=False)
        resources[:, 3] += np.einsum('ni,i->n', p, output_fanout, optimize=False)
        if 16 <= tick < q:
            resources[:, 4] += hs.sum(axis=1)
            resources[:, 5] += os.sum(axis=1)
        if 25 <= tick < q:
            cues['cue_scores'].append(scores)
            cues['cue_weights'].append(weights)
            cues['cue_soft_weights'].append(soft_weights)
            cues['cue_read_vectors'].append(injection)
            cues['cue_fallback'].append(fallback)
        step = dict(states=np.column_stack((d, v, o)), spikes=np.column_stack((hs, os)),
                    candidates=np.column_stack((d, vc, oc)), currents=current, injections=injection,
                    key_bank=key_bank, value_bank=value_bank, scores=scores, weights=weights,
                    read_gate=read, write_gate=marker, drive=drive)
        for name, values in step.items():
            history[name].append(values[:8])
        p = hs
    return dict(counts=counts, resources=resources,
                **{name: np.stack(values, axis=1) for name, values in history.items()},
                **{name: np.stack(values, axis=1) for name, values in cues.items()})
