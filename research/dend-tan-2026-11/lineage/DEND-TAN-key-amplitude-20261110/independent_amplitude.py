"""Independent fixed-weight read-direction and norm intervention replay."""

import numpy as np

from reference import split


CONDITIONS = {
    'KEY': ('nominal', 'swap_read', 'swap_direction', 'swap_magnitude', 'mean_read', 'mean_matched'),
    'KEY_MEAN': ('nominal',),
    'GAIN2': ('nominal',),
}
MATCHED = ('swap_direction', 'swap_magnitude', 'mean_matched')
CUE = 25


def match_norm(direction, amplitude):
    wanted = np.sqrt(np.einsum('...i,...i->...', amplitude, amplitude))
    source = np.sqrt(np.einsum('...i,...i->...', direction, direction))
    unit = np.divide(direction, source[..., None], out=np.zeros_like(direction), where=source[..., None] > 0)
    fallback = (source == 0) & (wanted > 0)
    unit[..., 0] = np.where(fallback, 1., unit[..., 0])
    return unit*wanted[..., None], fallback


def replay(x, w, arm, q, condition='nominal'):
    assert condition in CONDITIONS[arm] and q == 28 and x.shape[1:] == (36, 6)
    wi, wr, db, vb, wo, ob = split(w)
    n, steps, _ = x.shape
    d, v, o = np.zeros((n, 12)), np.zeros((n, 6)), np.zeros((n, 2))
    p, counts = np.zeros_like(v), np.zeros_like(o)
    resources = np.zeros((n, 6))
    recurrent_fanout = np.count_nonzero(wr, axis=1)
    output_fanout = np.count_nonzero(wo, axis=1)
    read_vectors = np.zeros((n, 3, 12))
    direction_fallback = np.zeros((n, 3), dtype=bool)
    keyed = arm != 'GAIN2'
    history = {key: [] for key in ('states', 'spikes', 'candidates', 'currents', 'injections')}
    if keyed:
        slot0, slot1 = np.zeros_like(d), np.zeros_like(d)
        cue_memory = np.zeros((n, 3, 12, 2))
        history.update(memory=[], keys=[], selector=[], writes=[], read_gate=[], drive=[])

    for tick in range(steps):
        external = x[:, tick]
        z = np.einsum('ni,ij->nj', external, wi, optimize=False)
        z += np.einsum('ni,ij->nj', p, wr, optimize=False)
        current = z+db
        in_cue = CUE <= tick < q
        injection = current
        if keyed:
            address = np.maximum(external[:, 2:4], 0.)
            mass = address.sum(axis=1, keepdims=True)
            keys = np.divide(address, mass, out=np.zeros_like(address), where=mass > 0.)
            marker = np.clip(external[:, 5:6], 0., 1.)
            writes = marker*keys
            gate = (1-marker)*np.minimum(mass, 1.)
            drive = np.tanh(z)
            other = 1-keys
            correct = 2*gate*(keys[:, :1]*slot0+keys[:, 1:]*slot1)
            swapped = 2*gate*(other[:, :1]*slot0+other[:, 1:]*slot1)
            mean = gate*(slot0+slot1)
            selector = keys
            injection = correct
            fallback = None
            if condition in ('swap_read', 'swap_direction'):
                selector, injection = other, swapped
            if arm == 'KEY_MEAN' or condition in ('mean_read', 'mean_matched'):
                selector, injection = np.full_like(keys, .5), mean
            if condition == 'swap_direction':
                injection, fallback = match_norm(swapped, correct)
            elif condition == 'swap_magnitude':
                injection, fallback = match_norm(correct, swapped)
            elif condition == 'mean_matched':
                injection, fallback = match_norm(mean, correct)
            if in_cue:
                cue_memory[:, tick-CUE, :, 0] = slot0
                cue_memory[:, tick-CUE, :, 1] = slot1
                if fallback is not None:
                    direction_fallback[:, tick-CUE] = fallback
            slot0 = .97*(1-writes[:, :1])*slot0 + writes[:, :1]*drive
            slot1 = .97*(1-writes[:, 1:])*slot1 + writes[:, 1:]*drive
        if in_cue:
            read_vectors[:, tick-CUE] = injection

        d = .95*d+.25*(current+injection)
        local = np.where(d > 0, d, 0.)
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
        step = dict(states=np.column_stack((d[:8], v[:8], o[:8])), spikes=np.column_stack((hs[:8], os[:8])),
                    candidates=np.column_stack((d[:8], vc[:8], oc[:8])), currents=current[:8], injections=injection[:8])
        if keyed:
            step.update(memory=np.stack((slot0[:8], slot1[:8]), axis=-1), keys=keys[:8], selector=selector[:8],
                        writes=writes[:8], read_gate=gate[:8], drive=drive[:8])
        for name, values in step.items():
            history[name].append(values)
        p = hs

    raw = dict(counts=counts, resources=resources, read_vectors=read_vectors, direction_fallback=direction_fallback,
               **{name: np.stack(values, axis=1) for name, values in history.items()})
    if keyed:
        raw['cue_memory'] = cue_memory
    return raw
