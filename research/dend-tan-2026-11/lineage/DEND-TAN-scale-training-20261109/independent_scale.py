"""Independent native replay of the two-scale model and the four controls."""

import numpy as np

import reference


SCALE_ARMS = ('SCALE', 'SCALE_MEAN')


def replay(x, w, arm, q):
    if arm not in SCALE_ARMS:
        return reference.replay(x, w, arm, q)

    assert x.shape[1:] == (q+8, 6)
    wi, wr, db, vb, wo, ob = reference.split(w)
    n, steps, _ = x.shape
    d = np.zeros((n, 12))
    v = np.zeros((n, 6))
    o = np.zeros((n, 2))
    p = np.zeros_like(v)
    fast = np.zeros_like(d)
    slow = np.zeros_like(d)
    counts = np.zeros_like(o)
    resources = np.zeros((n, 6))
    recurrent_fanout = np.count_nonzero(wr, axis=1)
    output_fanout = np.count_nonzero(wo, axis=1)
    traces = {name: [] for name in ('states', 'spikes', 'candidates', 'currents', 'injections',
                                    'traces', 'weights_fast', 'contexts', 'gates', 'drive')}

    for tick in range(steps):
        external = x[:, tick]
        external_drive = np.einsum('ni,ij->nj', external, wi, optimize=False)
        recurrent_drive = np.einsum('ni,ij->nj', p, wr, optimize=False)
        drive = external_drive + recurrent_drive
        current = drive + db
        signed = np.tanh(drive)
        gate = np.minimum(np.linalg.norm(external, ord=1, axis=1, keepdims=True), 1.)
        if arm == 'SCALE':
            # sigmoid(2*z) = (1 + tanh(z))/2. Read both previous traces.
            weight_fast = .5*(1+np.tanh(signed*(fast-slow)))
        else:
            weight_fast = np.full_like(fast, .5)
        context = weight_fast*fast + (1-weight_fast)*slow
        injection = gate*context
        write = .5*gate
        fast = (1-write)*.85*fast + write*signed
        slow = (1-write)*.97*slow + write*signed

        d = .95*d + .25*(current+injection)
        local = np.where(d > 0, d, 0.)
        vc = .85*v + .5*(local[:, ::2]+local[:, 1::2]) + vb
        hs = vc > .5
        v = vc*np.logical_not(hs)
        oc = .85*o + np.einsum('ni,ij->nj', p, wo, optimize=False) + ob
        os = oc > .5
        o = oc*np.logical_not(os)
        if tick >= q:
            counts += os
        resources[:, 0] += hs.sum(axis=1)
        resources[:, 1] += os.sum(axis=1)
        resources[:, 2] += np.einsum('ni,i->n', p, recurrent_fanout, optimize=False)
        resources[:, 3] += np.einsum('ni,i->n', p, output_fanout, optimize=False)
        if 16 <= tick < q:
            resources[:, 4] += hs.sum(axis=1)
            resources[:, 5] += os.sum(axis=1)

        step = dict(states=np.column_stack((d, v, o)), spikes=np.column_stack((hs, os)),
                    candidates=np.column_stack((d, vc, oc)), currents=current, injections=injection,
                    traces=np.stack((fast, slow), axis=-1), weights_fast=weight_fast,
                    contexts=context, gates=gate, drive=signed)
        for name, values in step.items():
            traces[name].append(values[:8].copy())
        p = hs.astype(float)

    return dict(counts=counts, resources=resources,
                **{name: np.stack(values, axis=1) for name, values in traces.items()})
