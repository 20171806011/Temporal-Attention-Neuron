"""Independent test geometry, evicted-key history and guarded native replay."""

import numpy as np

import independent_event as event
import independent_warm as native


GEOMETRIES = ('orthogonal', 'near')
HISTORIES = event.HISTORIES
RULES = ('original', 'replay', 'guarded')
SCHEDULES = ('A', 'B', 'C', 'D')
WRITE_TICKS = ((0, 3, 6, 11), (1, 4, 7, 13), (0, 2, 5, 10), (2, 5, 8, 12))
DISTRACTOR_TICKS = ((2, 8), (0, 10), (1, 8), (3, 6))
FIRST_READ = (22, 24, 20, 23)
T, Q = 36, 28


def geometry(base, name):
    assert name in GEOMETRIES
    if name == 'orthogonal':
        return base.copy()
    phases = base['angle'][:, None]+np.deg2rad((0., 45., 180., 225.))
    keys = np.stack((np.cos(phases), np.sin(phases)), axis=-1)
    queries = np.stack((-keys[..., 1], keys[..., 0]), axis=-1)+base['query_noise']
    return dict(base, key_vectors=keys, query_vectors=queries)


def assemble(base, history):
    kept = {'four': (0, 1, 2, 3), 'three0': (0, 2, 3), 'three2': (1, 2, 3), 'two': (2, 3)}[history]
    x = np.zeros((len(base['y']), T, 6))
    for row, (group, selected, bits) in enumerate(zip(base['group'], base['selected'], base['bits'])):
        schedule = group % len(SCHEDULES)
        order = base['write_order'][group]
        for position in kept:
            tick = WRITE_TICKS[schedule][position]
            x[row, tick, bits[position]] = base['payload_amplitude'][group, position]
            x[row, tick, 2:4] = base['key_vectors'][group, order[position]]
            x[row, tick, 5] = 1.
        x[row, DISTRACTOR_TICKS[schedule], 4] = base['distractor'][group]
        first = FIRST_READ[schedule]
        x[row, first:first+3, 2:4] = base['query_vectors'][group, order[selected]]
        x[row, Q:, 4] = 1.
    return x, Q


def first_read_match(x, w):
    key = x[..., 2:4]
    read = (1.-x[..., 5])*np.minimum(1., np.linalg.norm(key, axis=-1))
    active = read > 0.
    assert np.all(active.any(axis=1))
    rows = np.arange(len(x))
    first = active.argmax(axis=1)
    latest = np.zeros((len(x), 2, 2))
    for row, stop in enumerate(first):
        writes = np.flatnonzero(x[row, :stop, 5] == 1.)[-2:][::-1]
        latest[row, :len(writes)] = key[row, writes]
    projected = np.einsum('ni,ij->nj', key[rows, first], w[176:].reshape(2, 2), optimize=False)
    scores = 4*np.einsum('ni,nki->nk', projected, latest, optimize=False)
    e = np.exp(scores-scores.max(axis=1, keepdims=True))
    return first, scores, e/e.sum(axis=1, keepdims=True)


def evicted_log(x, first):
    """Take the valid keys overwritten two writes later before the first read."""
    keys = np.zeros((len(x), 2, 2))
    for row, edge in enumerate(first):
        prefix = x[row, :edge] if edge >= 0 else x[row]
        writes = np.flatnonzero(prefix[:, 5] == 1.)
        overwritten = prefix[writes[:-2], 2:4]
        valid = overwritten[np.any(overwritten != 0., axis=1)][-2:]
        if len(valid):
            keys[row, -len(valid):] = valid
    return keys


def apply_guard(x, w, replayed, original):
    """Combine independent full replay and original rows after deriving the causal veto."""
    n, horizon = x.shape[:2]
    evicted_keys = evicted_log(x, replayed['first_read_tick'])
    evicted_cosine = np.full((n, horizon), -2.)
    comparisons = np.zeros((n, horizon), dtype=np.int64)
    rows, ticks = np.nonzero(replayed['reset_steps'])
    projected = np.einsum('ni,ij->nj', x[rows, ticks, 2:4], w[176:].reshape(2, 2), optimize=False)
    key_norm = np.linalg.norm(evicted_keys[rows], axis=-1)
    denominator = np.linalg.norm(projected, axis=-1)[:, None]*key_norm
    dot = np.sum(projected[:, None, :]*evicted_keys[rows], axis=-1)
    cosine = np.full_like(dot, -2.)
    valid = denominator > 0.
    np.divide(dot, denominator, out=cosine, where=valid)
    evicted_cosine[rows, ticks] = cosine.max(axis=1)
    comparisons[rows, ticks] = valid.sum(axis=1)
    rejected = np.zeros(n, dtype=bool)
    rejected[rows] = replayed['match_confidence'][rows, ticks] <= evicted_cosine[rows, ticks]

    out = replayed.copy()
    for field, baseline in original.items():
        if field == 'match_confidence':
            continue
        mask = rejected[:len(baseline)]
        out[field] = out[field].copy()
        out[field][mask] = baseline[mask]
    for field in ('replay_steps', 'replay_resources', 'replay_reference'):
        out[field] = out[field].copy()
        out[field][rejected] = 0
    out.update(evicted_keys=evicted_keys, evicted_cosine=evicted_cosine, evicted_comparisons=comparisons)
    return out


def replay(x, w, q):
    return apply_guard(x, w, event.replay(x, w, q), native.replay(x, w, 'LEARN', q, 'original'))
