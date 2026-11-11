"""Independent keyed-memory recurrence and anchored surrogate."""

import numpy as np

import reference


KEYED = ('KEY', 'KEY_MEAN')


def selection(x, arm, swap_read):
    tags = np.maximum(x[..., 2:4], 0.)
    mass = tags.sum(axis=-1, keepdims=True)
    keys = np.divide(tags, mass, out=np.zeros_like(tags), where=mass > 0)
    write_mark = np.clip(x[..., 5:6], 0., 1.)
    writes = write_mark*keys
    read = (1-write_mark)*np.minimum(mass, 1.)
    if swap_read:
        selector = 1-keys
    elif arm == 'KEY_MEAN':
        selector = np.full_like(keys, .5)
    else:
        selector = keys
    return keys, selector, writes, read


def run(x, w, arm, q, swap_read, mode, anchors=None):
    wi, wr, db, vb, wo, ob = reference.split(w)
    n, steps, _ = x.shape
    d = np.zeros((n, 12))
    v, o = np.zeros((n, 6)), np.zeros((n, 2))
    p, counts = np.zeros_like(v), np.zeros_like(o)
    keyed = arm in KEYED
    if keyed:
        keys, selector, writes, read = selection(x, arm, swap_read)
        slot0, slot1 = np.zeros_like(d), np.zeros_like(d)
    if mode == 'archive':
        resources = np.zeros((n, 6))
        recurrent_fanout = np.count_nonzero(wr, axis=1)
        output_fanout = np.count_nonzero(wo, axis=1)
        names = ('states', 'spikes', 'candidates', 'currents', 'injections')
        if keyed:
            names += ('memory', 'keys', 'selector', 'writes', 'read_gate', 'drive')
        history = {key: [] for key in names}
    snapshots = []
    for tick in range(steps):
        z = np.einsum('ni,ij->nj', x[:, tick], wi, optimize=False)
        z += np.einsum('ni,ij->nj', p, wr, optimize=False)
        current = z+db
        if keyed:
            drive = np.tanh(z)
            mix0, mix1 = selector[:, tick, :1], selector[:, tick, 1:]
            write0, write1 = writes[:, tick, :1], writes[:, tick, 1:]
            gate = read[:, tick]
            injection = (2*gate)*(mix0*slot0+mix1*slot1)
            slot0 = .97*(1-write0)*slot0 + write0*drive
            slot1 = .97*(1-write1)*slot1 + write1*drive
        else:
            injection = current

        d = .95*d+.25*(current+injection)
        frozen = None if anchors is None else anchors[tick]
        local = np.where((d if frozen is None else frozen['d']) > 0, d, 0.)
        vc = .85*v+.5*(local[:, ::2]+local[:, 1::2])+vb
        oc = .85*o+np.einsum('ni,ij->nj', p, wo, optimize=False)+ob
        if frozen is None:
            hs, os = (vc > .5).astype(float), (oc > .5).astype(float)
        else:
            hs = frozen['hs']+(vc-frozen['vc'])/(1+5*np.abs(frozen['vc']-.5))**2
            os = frozen['os']+(oc-frozen['oc'])/(1+5*np.abs(frozen['oc']-.5))**2
        v, o = vc*(1-hs), oc*(1-os)
        if tick >= q:
            counts += os
        if mode == 'anchor':
            snapshots.append(dict(d=d, vc=vc, oc=oc, hs=hs, os=os))
        if mode == 'archive':
            resources[:, 0] += hs.sum(axis=1)
            resources[:, 1] += os.sum(axis=1)
            resources[:, 2] += np.einsum('ni,i->n', p, recurrent_fanout, optimize=False)
            resources[:, 3] += np.einsum('ni,i->n', p, output_fanout, optimize=False)
            if 16 <= tick < q:
                resources[:, 4] += hs.sum(axis=1)
                resources[:, 5] += os.sum(axis=1)
            step = dict(states=np.column_stack((d[:8], v[:8], o[:8])),
                        spikes=np.column_stack((hs[:8], os[:8])),
                        candidates=np.column_stack((d[:8], vc[:8], oc[:8])),
                        currents=current[:8], injections=injection[:8])
            if keyed:
                step.update(memory=np.stack((slot0[:8], slot1[:8]), axis=-1), keys=keys[:8, tick],
                            selector=selector[:8, tick], writes=writes[:8, tick], read_gate=gate[:8], drive=drive[:8])
            for name, values in step.items():
                history[name].append(values)
        p = hs
    if mode == 'archive':
        return dict(counts=counts, resources=resources,
                    **{name: np.stack(values, axis=1) for name, values in history.items()})
    if mode == 'anchor':
        return snapshots
    return counts


def replay(x, w, arm, q, swap_read=False):
    assert arm in KEYED or arm == 'GAIN2'
    assert not swap_read or arm == 'KEY'
    if arm == 'GAIN2':
        return reference.replay(x, w, arm, q)
    return run(x, w, arm, q, swap_read, 'archive')


def anchor(x, w, arm, q):
    return run(x, w, arm, q, False, 'anchor')


def objective(x, y, w, arm, q, anchor):
    counts = run(x, w, arm, q, False, 'objective', anchor)
    margin = .5*(counts[:, 1]-counts[:, 0])
    return float(np.mean(np.logaddexp(0., (1-2*y)*margin))+.0001*np.mean(w*w))
