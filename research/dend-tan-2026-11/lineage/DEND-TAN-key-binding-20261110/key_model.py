"""Addressed two-slot memory on the native DEND branch interface."""

import numpy as np

import model as native
import population


ARMS = ('KEY', 'KEY_MEAN', 'GAIN2')
GAMMAS = {'KEY': 1., 'KEY_MEAN': 0.}
MEMORY_RETENTION = .97
READ_GAIN = 2.


def _check_swap(arm, swap_read):
    if swap_read and arm != 'KEY':
        raise ValueError('swap_read requires KEY')


def _address_signals(x, gamma):
    marker = np.clip(x[..., 5:6], 0., 1.)
    addresses = np.maximum(x[..., 2:4], 0.)
    total = addresses.sum(axis=-1, keepdims=True)
    keys = np.divide(addresses, total, out=np.zeros_like(addresses), where=total > 0.)
    selector = .5+gamma*(keys-.5)
    writes = marker*keys
    read_gate = (1.-marker)*np.minimum(1., total)
    return keys, selector, writes, read_gate


def forward(x, w, arm, q, anchor=None, swap_read=False):
    _check_swap(arm, swap_read)
    if arm == 'GAIN2':
        return native.forward(x, w, arm, q, anchor)

    gamma = -1. if swap_read else GAMMAS[arm]
    keys, selectors, writes, read_gates = _address_signals(x, gamma)
    retention = MEMORY_RETENTION*(1.-writes)
    wi, wr, db, vb, wo, ob = population.unpack(w, 6)
    n, steps, _ = x.shape
    d = np.zeros((n, 12))
    v = np.zeros((n, 6))
    p = np.zeros_like(v)
    o = np.zeros((n, 2))
    memory = np.zeros((n, 12, 2))
    counts = np.zeros_like(o)
    cache = []

    for t in range(steps):
        anc = None if anchor is None else anchor[t]
        z = x[:, t] @ wi+p @ wr
        current = z+db
        drive = np.tanh(z)
        selector = selectors[:, t]
        write = writes[:, t]
        read_gate = read_gates[:, t]
        keep = retention[:, t]

        context = memory[..., 0]*selector[:, :1]+memory[..., 1]*selector[:, 1:]
        injection = READ_GAIN*read_gate*context
        memory = keep[:, None, :]*memory+write[:, None, :]*drive[..., None]

        d = .95*d+.25*(current+injection)
        if anc is None:
            local = np.maximum(d, 0.)
        else:
            local = np.maximum(anc['dc'], 0.)+(anc['dc'] > 0.)*(d-anc['dc'])
        vc = .85*v+.5*local.reshape(n, 6, 2).sum(axis=2)+vb
        oc = .85*o+p @ wo+ob
        if anc is None:
            hs = (vc > .5).astype(float)
            os = (oc > .5).astype(float)
        else:
            hs = anc['hs']+population.slope(anc['vc'])*(vc-anc['vc'])
            os = anc['os']+population.slope(anc['oc'])*(oc-anc['oc'])
        v = vc*(1.-hs)
        o = oc*(1.-os)
        if t >= q:
            counts += os
        cache.append(dict(I=current, A=injection, dc=d, vc=vc, oc=oc, hs=hs, os=os, prev=p,
                          states=np.concatenate((d, v, o), axis=1), memory=memory, keys=keys[:, t],
                          selector=selector, writes=write, read_gate=read_gate, drive=drive,
                          retention=keep))
        p = hs
    return counts, cache


def loss_grad(x, y, w, arm, q):
    if arm == 'GAIN2':
        return native.loss_grad(x, y, w, arm, q)
    _, wr, _, _, wo, _ = population.unpack(w, 6)
    counts, cache = forward(x, w, arm, q)
    loss, gcounts = population.loss_counts(counts, y, 'CE')
    grad = np.zeros_like(w)
    gwi, gwr, gdb, gvb, gwo, gob = population.unpack(grad, 6)
    n = len(x)
    gd = np.zeros((n, 12))
    gv = np.zeros((n, 6))
    gp = np.zeros_like(gv)
    go = np.zeros((n, 2))
    gmemory = np.zeros((n, 12, 2))

    for t in reversed(range(x.shape[1])):
        step = cache[t]
        goc = (go*(1.-step['os'])
               + ((gcounts if t >= q else 0)-go*step['oc'])*population.slope(step['oc']))
        gwo += step['prev'].T @ goc
        gob += goc.sum(axis=0)
        gprev_output = goc @ wo.T
        go = .85*goc

        gvc = gv*(1.-step['hs'])+(gp-gv*step['vc'])*population.slope(step['vc'])
        gvb += gvc.sum(axis=0)
        gv = .85*gvc
        gdc = gd+.5*np.repeat(gvc, 2, axis=1)*(step['dc'] > 0.)
        direct = .25*gdc
        gd = .95*gdc

        write = step['writes']
        gu = gmemory[..., 0]*write[:, :1]+gmemory[..., 1]*write[:, 1:]
        gcontext = READ_GAIN*step['read_gate']*direct
        gmemory = (step['retention'][:, None, :]*gmemory
                   + gcontext[..., None]*step['selector'][:, None, :])
        gz = direct+gu*(1.-step['drive']**2)
        gwi += x[:, t].T @ gz
        gwr += step['prev'].T @ gz
        gdb += direct.sum(axis=0)
        gp = gprev_output+gz @ wr.T

    grad += .0002*w/len(w)
    return loss+.0001*np.mean(w*w), grad, counts


def objective(x, y, w, arm, q, anchor=None):
    if arm == 'GAIN2':
        return native.objective(x, y, w, arm, q, anchor)
    counts, _ = forward(x, w, arm, q, anchor)
    return population.loss_counts(counts, y, 'CE')[0]+.0001*np.mean(w*w)


def evaluate(x, w, arm, q, swap_read=False):
    _check_swap(arm, swap_read)
    if arm == 'GAIN2':
        return native.evaluate(x, w, arm, q)
    _, wr, _, _, wo, _ = population.unpack(w, 6)
    fanout_r = np.count_nonzero(wr, axis=1)
    fanout_o = np.count_nonzero(wo, axis=1)
    count_chunks = []
    resource_chunks = []
    for start in range(0, len(x), 32):
        batch = x[start:start+32]
        counts, cache = forward(batch, w, arm, q, swap_read=swap_read)
        count_chunks.append(counts)
        resources = np.zeros((len(batch), 6))
        for t, step in enumerate(cache):
            resources[:, 0] += step['hs'].sum(axis=1)
            resources[:, 1] += step['os'].sum(axis=1)
            resources[:, 2] += step['prev'] @ fanout_r
            resources[:, 3] += step['prev'] @ fanout_o
            if 16 <= t < q:
                resources[:, 4] += step['hs'].sum(axis=1)
                resources[:, 5] += step['os'].sum(axis=1)
        resource_chunks.append(resources)
        if start == 0:
            names = {'states': 'states', 'currents': 'I', 'injections': 'A', 'memory': 'memory',
                     'keys': 'keys', 'selector': 'selector', 'writes': 'writes', 'read_gate': 'read_gate',
                     'drive': 'drive'}
            saved = {key: np.stack([step[field][:8] for step in cache], axis=1) for key, field in names.items()}
            saved['spikes'] = np.stack([np.concatenate((step['hs'][:8], step['os'][:8]), axis=1)
                                       for step in cache], axis=1)
            saved['candidates'] = np.stack([np.concatenate((step['dc'][:8], step['vc'][:8], step['oc'][:8]), axis=1)
                                           for step in cache], axis=1)
    saved['counts'] = np.concatenate(count_chunks)
    saved['resources'] = np.concatenate(resource_chunks)
    return saved
