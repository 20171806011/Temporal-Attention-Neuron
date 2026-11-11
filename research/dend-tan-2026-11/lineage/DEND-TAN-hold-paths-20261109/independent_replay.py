"""Independent native and interventional replay, without the scientific runner."""

import numpy as np

from reference import attention, split


ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2')
ATTENTION_ARMS = ('FULL', 'NOATT')
STATE_FIELDS = ('d', 'v', 'p', 'o')
GROUPS = {
    'branch': ('d',),
    'soma': ('v',),
    'previous': ('p',),
    'soma_p': ('v', 'p'),
    'output': ('o',),
    'other': ('v', 'p', 'o', 'H'),
    'full': ('d', 'v', 'p', 'o', 'H'),
    'history': ('H',),
    'branch_history': ('d', 'H'),
}
CURRENT_WINDOWS = {
    'self_blank': (9, 25),
    'mean_blank': (9, 25),
    'flip_first_blank': (9, 25),
    'flip_second_blank': (9, 25),
    'mean_early': (9, 17),
    'mean_late': (17, 25),
    'mean_query': (28, 36),
}


def _simulate(x, theta, arm, q, start, state, taps, recurrence, window, save_recurrence):
    n, steps, channels = x.shape
    assert channels == 6 and arm in ARMS and 0 <= start <= q and steps == q+8
    att = arm in ATTENTION_ARMS
    fields = (*STATE_FIELDS, 'H') if att else STATE_FIELDS
    assert set(state) == set(fields)
    d, v, p, o = (state[key].copy() for key in STATE_FIELDS)
    history = state['H'].copy() if att else None
    wi, wr, db, vb, wo, ob = split(theta)
    nr = np.count_nonzero(wr, axis=1)
    no = np.count_nonzero(wo, axis=1)
    assert (recurrence is None) == (window is None)
    if window is not None:
        assert start <= window[0] < window[1] <= steps and recurrence.shape == (n, steps, 12)
    assert not save_recurrence or (start == 0 and window is None)
    saved = {key: [] for key in (*fields, 'prefix_resources')} if taps else {}
    path = np.empty((n, steps, 12)) if save_recurrence else None
    raster = np.empty((n, 8, 2), dtype=np.uint8)
    resources = np.zeros((n, 5), dtype=np.int64)
    for t in range(start, steps):
        clamped = window is not None and window[0] <= t < window[1]
        if clamped:
            feedback = recurrence[:, t]
            resources[:, 4] += 12
        else:
            feedback = np.einsum('ni,ij->nj', p, wr, optimize=False)
            resources[:, 2] += np.einsum('ni,i->n', p.astype(np.int64), nr)
        if path is not None:
            path[:, t] = feedback
        current = np.einsum('ni,ij->nj', x[:, t], wi, optimize=False) + feedback + db
        if att:
            injection, history, _ = attention(current, history, arm == 'FULL')
        elif arm == 'GAIN2':
            injection = current
        else:
            injection = 0.

        d = .95*d + .25*(current+injection)
        rectified = np.maximum(d, 0.)
        v_candidate = .85*v + .5*(rectified[:, ::2]+rectified[:, 1::2]) + vb
        hidden = v_candidate > .5
        v = np.where(hidden, 0., v_candidate)
        o_candidate = .85*o + np.einsum('ni,ij->nj', p, wo, optimize=False) + ob
        output = o_candidate > .5
        o = np.where(output, 0., o_candidate)
        resources[:, 0] += hidden.sum(1)
        resources[:, 1] += output.sum(1)
        resources[:, 3] += np.einsum('ni,i->n', p.astype(np.int64), no)
        if t >= q:
            raster[:, t-q] = output
        p = hidden
        if t in taps:
            for key, value in zip(STATE_FIELDS, (d, v, p, o)):
                saved[key].append(value.copy())
            saved['prefix_resources'].append(resources.copy())
            if att:
                saved['H'].append(history.copy())

    result = dict(raster=raster, counts=raster.sum(1), resources=resources)
    result.update({key: np.stack(sequence, 1) for key, sequence in saved.items()})
    if path is not None:
        result['recurrence'] = path
    return result


def capture(x, theta, arm, q, taps, save_recurrence=False):
    """Replay from zero; capture only declared post-step taps and optional R."""
    taps = tuple(taps)
    assert tuple(sorted(set(taps))) == taps and all(0 <= t < x.shape[1] for t in taps)
    state = {key: np.zeros((len(x), width)) for key, width in zip(STATE_FIELDS, (12, 6, 6, 2))}
    if arm in ATTENTION_ARMS:
        state['H'] = np.zeros((len(x), 12, 5))
    return _simulate(x, theta, arm, q, 0, state, taps, None, None, save_recurrence)


def resume(x, theta, arm, q, tap, state, recurrence=None, window=None):
    """Resume a pre-query snapshot. Replay R only inside the half-open window."""
    return _simulate(x, theta, arm, q, tap+1, state, (), recurrence, window, False)


def state_at(snapshot, taps, tap):
    index = tuple(taps).index(tap)
    fields = (*STATE_FIELDS, 'H') if 'H' in snapshot else STATE_FIELDS
    return {key: snapshot[key][:, index].copy() for key in fields}


def splice(snapshot, taps, tap, group, donor_index):
    """Construct a same-time intervention and compute its Euclidean distance."""
    state = state_at(snapshot, taps, tap)
    assert group not in ('history', 'branch_history') or 'H' in state
    distance_squared = np.zeros(len(donor_index))
    for key in GROUPS[group]:
        if key in state:
            value = state[key].astype(float)
            delta = (value[donor_index]-value).reshape(len(value), -1)
            distance_squared += np.einsum('ni,ni->n', delta, delta, optimize=False)
            state[key] = state[key][donor_index].copy()
    return state, np.sqrt(distance_squared)


def donors(base):
    """Independent lookup of physical flips and opposite future cue partners."""
    keys = ('octet', 'selected', 'first_bit', 'second_bit')
    cases = [tuple(int(value) for value in values) for values in zip(*(base[key] for key in keys))]
    index = {case: i for i, case in enumerate(cases)}
    assert len(index) == len(cases)
    return tuple(np.array([index[case[:j]+(1-case[j],)+case[j+1:]] for case in cases], dtype=np.int64)
                 for j in (2, 3, 1))


def mean_recurrence(recurrence, base):
    """Keep the native mean across the four contents at fixed octet and cue."""
    result = np.empty_like(recurrence)
    group = 2*base['octet']+base['selected']
    for value in np.unique(group):
        mask = group == value
        assert mask.sum() == 4
        result[mask] = recurrence[mask].mean(0)
    return result
