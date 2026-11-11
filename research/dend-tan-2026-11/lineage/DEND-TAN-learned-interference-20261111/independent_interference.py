"""Independent native replay of the two-prefix boundary exchange."""

import numpy as np

from reference import split


ARMS = ('LEARN', 'ORACLE', 'MEAN')
CONDITIONS = ('full_full', 'late_full', 'full_late', 'late_late')
BOUNDARY = 25
ROTATION = np.array([[0., -1.], [1., 0.]])
TRACES = ('states', 'spikes', 'candidates', 'currents', 'injections', 'key_bank', 'value_bank',
          'scores', 'weights', 'read_gate', 'write_gate', 'drive')
CUE_ARRAYS = ('cue_scores', 'cue_weights', 'cue_read_vectors')


def initial_state(n):
    return tuple(np.zeros((n, *shape)) for shape in ((12,), (6,), (2,), (6,), (2, 2), (12, 2)))


def segment(x, w, arm, q, start, stop, entry):
    """Advance from entry and capture the leading eight trajectories."""
    wi, wr, db, vb, wo, ob = split(w[:176])
    matcher = {'LEARN': w[176:].reshape(2, 2), 'ORACLE': ROTATION, 'MEAN': np.zeros((2, 2))}[arm]
    d, v, o, p, keys, values = entry
    counts, resources = np.zeros((len(x), 2)), np.zeros((len(x), 6))
    recurrent_fanout = np.count_nonzero(wr, axis=1)
    output_fanout = np.count_nonzero(wo, axis=1)
    traces, cues = {name: [] for name in TRACES}, {name: [] for name in CUE_ARRAYS}
    for tick in range(start, stop):
        external = x[:, tick]
        observed, marker = external[:, 2:4], external[:, 5:6]
        writing = marker == 1.
        read = (1.-marker)*np.minimum(1., np.linalg.norm(observed, axis=1, keepdims=True))
        scores = 4*np.einsum('ni,ij,nkj->nk', observed, matcher, keys, optimize=False)
        exp_scores = np.exp(scores-scores.max(axis=1, keepdims=True))
        weights = exp_scores/exp_scores.sum(axis=1, keepdims=True)
        injection = 2*read*np.einsum('nk,ndk->nd', weights, values, optimize=False)

        z = np.einsum('ni,ij->nj', external, wi, optimize=False)
        z += np.einsum('ni,ij->nj', p, wr, optimize=False)
        current, drive = z+db, np.tanh(z)
        older_key = np.where(writing, keys[:, 0], keys[:, 1])
        newer_key = np.where(writing, observed, keys[:, 0])
        older_value = .97*np.where(writing, values[..., 0], values[..., 1])
        newer_value = np.where(writing, drive, .97*values[..., 0])
        keys = np.stack((newer_key, older_key), axis=1)
        values = np.stack((newer_value, older_value), axis=-1)

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
        if BOUNDARY <= tick < q:
            cues['cue_scores'].append(scores)
            cues['cue_weights'].append(weights)
            cues['cue_read_vectors'].append(injection)
        step = dict(states=np.column_stack((d, v, o)), spikes=np.column_stack((hs, os)),
                    candidates=np.column_stack((d, vc, oc)), currents=current, injections=injection,
                    key_bank=keys, value_bank=values, scores=scores, weights=weights,
                    read_gate=read, write_gate=marker, drive=drive)
        for name, values_at_step in step.items():
            traces[name].append(values_at_step[:8])
        p = hs
    raw = dict(counts=counts, resources=resources,
               **{name: np.stack(values, axis=1) for name, values in traces.items()},
               **{name: np.stack(values, axis=1) for name, values in cues.items() if values})
    return (d, v, o, p, keys, values), raw


def replay(full_x, late_x, w, arm, q, condition):
    assert condition in CONDITIONS and q == 28 and w.shape == (180,) and arm in ARMS
    assert full_x.shape == late_x.shape and full_x.shape[1:] == (36, 6)
    assert np.array_equal(full_x[:, BOUNDARY:], late_x[:, BOUNDARY:])
    source_n, source_v = condition.split('_')
    sources = {'full': full_x, 'late': late_x}
    entry_n, prefix = segment(sources[source_n], w, arm, q, 0, BOUNDARY, initial_state(len(full_x)))
    if source_n == source_v:
        entry_v = entry_n
    else:
        entry_v, _ = segment(sources[source_v], w, arm, q, 0, BOUNDARY, initial_state(len(full_x)))
    assert np.array_equal(entry_n[4], entry_v[4])
    entry = (*entry_n[:5], entry_v[5])
    _, suffix = segment(full_x, w, arm, q, BOUNDARY, 36, entry)
    return dict(counts=prefix['counts']+suffix['counts'], resources=prefix['resources']+suffix['resources'],
                entry_native=np.column_stack(entry[:4]), entry_keys=entry[4], entry_values=entry[5],
                **{name: np.concatenate((prefix[name], suffix[name]), axis=1) for name in TRACES},
                **{name: suffix[name] for name in CUE_ARRAYS})
