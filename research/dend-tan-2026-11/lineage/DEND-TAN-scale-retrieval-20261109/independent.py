"""Independent donor indexing and native replay of the delayed interventions."""

import numpy as np

import independent_data
from reference import split


TRANSFER_TICK = 25
NATIVE_CONDITIONS = ('nominal', 'identity', 'native_selected', 'native_unselected')
TRACE_CONDITIONS = ('trace_selected', 'trace_unselected', 'all_selected', 'all_unselected')
CONDITIONS = {
    'GAIN2': NATIVE_CONDITIONS,
    'SCALE': NATIVE_CONDITIONS + TRACE_CONDITIONS + ('post_mean',),
    'SCALE_MEAN': NATIVE_CONDITIONS + TRACE_CONDITIONS,
}


def donors(base):
    rows = independent_data.lookup(base)
    selected_donor = np.arange(len(base['y']))
    unselected_donor = selected_donor.copy()
    for key, receiver in rows.items():
        octet, cue, first, second = key
        if first != second:
            selected_donor[receiver] = rows[octet, cue, first ^ (1-cue), second ^ cue]
            unselected_donor[receiver] = rows[octet, cue, first ^ cue, second ^ (1-cue)]
    return {'selected': selected_donor, 'unselected': unselected_donor}


def replay(x, w, arm, q, condition, donor_map):
    assert q == 28 and x.shape[1:] == (36, 6)
    assert condition in CONDITIONS[arm]
    wi, wr, db, vb, wo, ob = split(w)
    n, steps, _ = x.shape
    d = np.zeros((n, 12))
    v = np.zeros((n, 6))
    o = np.zeros((n, 2))
    p = np.zeros_like(v)
    fast = np.zeros_like(d)
    slow = np.zeros_like(d)
    counts = np.zeros_like(o)
    resources = np.zeros((n, 4))
    recurrent_fanout = np.count_nonzero(wr, axis=1)
    output_fanout = np.count_nonzero(wo, axis=1)
    two_traces = arm in ('SCALE', 'SCALE_MEAN')
    kept = ('states', 'spikes', 'injections')
    if two_traces:
        kept += ('traces', 'weights_fast')
    history = {name: [] for name in kept}

    for tick in range(steps):
        if tick == TRANSFER_TICK:
            scope, _, donor_kind = condition.partition('_')
            if condition == 'identity':
                index = np.arange(n)
            elif donor_kind in donor_map:
                index = donor_map[donor_kind]
            if scope in ('native', 'all', 'identity'):
                d, v, o, p = (array[index] for array in (d, v, o, p))
            if two_traces and scope in ('trace', 'all', 'identity'):
                fast, slow = fast[index], slow[index]
            entry_native = np.column_stack((d, v, o, p))
            if two_traces:
                entry_traces = np.stack((fast, slow), axis=-1)

        external = x[:, tick]
        drive = np.einsum('ni,ij->nj', external, wi, optimize=False)
        drive += np.einsum('ni,ij->nj', p, wr, optimize=False)
        current = drive + db
        injection = current
        if two_traces:
            signed = np.tanh(drive)
            gate = np.minimum(np.linalg.norm(external, ord=1, axis=1, keepdims=True), 1.)
            uniform = arm == 'SCALE_MEAN' or (condition == 'post_mean' and tick >= TRANSFER_TICK)
            weight_fast = np.full_like(fast, .5) if uniform else .5*(1+np.tanh(signed*(fast-slow)))
            context = weight_fast*fast + (1-weight_fast)*slow
            injection = gate*context
            write = .5*gate
            fast = (1-write)*.85*fast + write*signed
            slow = (1-write)*.97*slow + write*signed

        d = .95*d + .25*(current+injection)
        local = np.where(d > 0, d, 0.)
        vc = .85*v + .5*(local[:, ::2]+local[:, 1::2]) + vb
        oc = .85*o + np.einsum('ni,ij->nj', p, wo, optimize=False) + ob
        hs, os = vc > .5, oc > .5
        v, o = vc*np.logical_not(hs), oc*np.logical_not(os)
        if tick >= q:
            counts += os
        resources[:, 0] += hs.sum(axis=1)
        resources[:, 1] += os.sum(axis=1)
        resources[:, 2] += np.einsum('ni,i->n', p, recurrent_fanout, optimize=False)
        resources[:, 3] += np.einsum('ni,i->n', p, output_fanout, optimize=False)
        p = hs.astype(float)

        history['states'].append(np.column_stack((d[:8], v[:8], o[:8], p[:8])))
        history['spikes'].append(np.column_stack((hs[:8], os[:8])))
        history['injections'].append(injection[:8].copy())
        if two_traces:
            history['traces'].append(np.stack((fast[:8], slow[:8]), axis=-1))
            history['weights_fast'].append(weight_fast[:8].copy())

    result = {name: np.stack(values, axis=1) for name, values in history.items()}
    result.update(counts=counts, resources=resources, entry_native=entry_native)
    if two_traces:
        result['entry_traces'] = entry_traces
    return result
