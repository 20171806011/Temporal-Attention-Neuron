"""Reference native replay and state splice without importing the experimental runner."""

import numpy as np

from reference import attention, split


TAPS = (9, 15, 19)
STATE_FIELDS = ('d', 'v', 'p', 'o')
ATTENTION_ARMS = ('FULL', 'NOATT')
GROUPS = {
    'full': ('d', 'v', 'p', 'o', 'H'),
    'hidden': ('d', 'v', 'p', 'H'),
    'branch': ('d',),
    'soma_p': ('v', 'p'),
    'output': ('o',),
    'history': ('H',),
}


def _replay(x, theta, arm, start, state, taps):
    """Return native query pulses, resources and optional post-step state taps."""
    assert arm in ('DEND', 'FULL', 'NOATT', 'GAIN2')
    assert x.shape[1:] == (24, 6) and 0 <= start < 24
    n = len(x)
    wi, wr, db, vb, wo, ob = split(theta)
    recurrent_fanout = np.count_nonzero(wr, axis=1)
    output_fanout = np.count_nonzero(wo, axis=1)
    use_history = arm in ATTENTION_ARMS
    fields = (*STATE_FIELDS, 'H') if use_history else STATE_FIELDS
    assert set(state) == set(fields)
    d, v, previous, o = (state[key].copy() for key in STATE_FIELDS)
    history = state['H'].copy() if use_history else None
    raster = np.empty((n, 24 - max(start, 16), 2), dtype=np.uint8)
    resources = np.zeros((n, 4), dtype=np.int64)
    saved = {key: [] for key in fields}

    for t in range(start, 24):
        current = np.einsum('ni,ij->nj', x[:, t], wi, optimize=False)
        current += np.einsum('ni,ij->nj', previous, wr, optimize=False) + db
        if use_history:
            injection, history, _ = attention(current, history, arm == 'FULL')
        elif arm == 'GAIN2':
            injection = current
        else:
            injection = np.zeros_like(current)

        d = .95 * d + .25 * (current + injection)
        local = np.maximum(d, 0.)
        hidden_candidate = .85 * v + .5 * (local[:, ::2] + local[:, 1::2]) + vb
        hidden = hidden_candidate > .5
        v = hidden_candidate * ~hidden
        output_candidate = .85 * o + np.einsum('ni,ij->nj', previous, wo, optimize=False) + ob
        out = output_candidate > .5
        o = output_candidate * ~out

        resources[:, 0] += hidden.sum(axis=1)
        resources[:, 1] += out.sum(axis=1)
        resources[:, 2] += np.einsum('ni,i->n', previous.astype(np.int64), recurrent_fanout)
        resources[:, 3] += np.einsum('ni,i->n', previous.astype(np.int64), output_fanout)
        if t >= max(start, 16):
            raster[:, t - max(start, 16)] = out
        previous = hidden
        if t in taps:
            for key, value in zip(STATE_FIELDS, (d, v, previous, o)):
                saved[key].append(value.copy())
            if use_history:
                saved['H'].append(history.copy())

    taps_saved = {key: np.stack(values, axis=1) for key, values in saved.items() if values}
    return raster, resources, taps_saved


def capture(x, theta, arm, taps=TAPS):
    """Replay a whole episode from the fixed model's zero initial state."""
    taps = tuple(taps)
    assert tuple(sorted(set(taps))) == taps and all(0 <= t < 24 for t in taps)
    n = len(x)
    state = {key: np.zeros((n, dim)) for key, dim in zip(STATE_FIELDS, (12, 6, 6, 2))}
    if arm in ATTENTION_ARMS:
        state['H'] = np.zeros((n, 12, 5))
    raster, resources, states = _replay(x, theta, arm, 0, state, taps)
    return dict(raster=raster, counts=raster.sum(axis=1), resources=resources, **states)


def transplant(snapshot, tap, group, donor_index):
    """Build a same-time mixture from reference baseline snapshots."""
    take_donor = GROUPS[group]
    t_index = TAPS.index(tap)
    fields = (*STATE_FIELDS, 'H') if 'H' in snapshot else STATE_FIELDS
    assert group != 'history' or 'H' in fields
    return {
        key: snapshot[key][donor_index, t_index].copy()
        if key in take_donor else snapshot[key][:, t_index].copy()
        for key in fields
    }


def continue_capture(x, theta, arm, tap, state):
    """Resume at tap+1. Prefix output counts never enter the dynamic state."""
    assert tap in TAPS
    raster, resources, _ = _replay(x, theta, arm, tap + 1, state, ())
    return raster, resources
