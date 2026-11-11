"""Bilinear matching over a two-slot FIFO on the native branch interface."""

import numpy as np

import population


ARMS = ('LEARN', 'ORACLE', 'MEAN')
NATIVE_PARAMETERS = 176
PARAMETERS = 180
MEMORY_RETENTION = .97
READ_GAIN = 2.
TEMPERATURE = 4.
ORACLE_MATRIX = np.array([[0., -1.], [1., 0.]])
MEAN_MATRIX = np.zeros((2, 2))
CUE_STEPS = (25, 26, 27)


def forward(x, w, arm, q, anchor=None, swap_read=False):
    if swap_read and arm not in ('LEARN', 'ORACLE'):
        raise ValueError('swap_read requires LEARN or ORACLE')
    matrix = w[NATIVE_PARAMETERS:].reshape(2, 2) if arm == 'LEARN' else {
        'ORACLE': ORACLE_MATRIX, 'MEAN': MEAN_MATRIX,
    }[arm]
    wi, wr, db, vb, wo, ob = population.unpack(w[:NATIVE_PARAMETERS], 6)
    queries = x[..., 2:4]
    projected_queries = queries @ matrix
    write_gates = x[..., 5:6]
    read_gates = (1.-write_gates)*np.minimum(1., np.linalg.norm(queries, axis=-1, keepdims=True))
    n, steps, _ = x.shape
    d = np.zeros((n, 12))
    v = np.zeros((n, 6))
    p = np.zeros_like(v)
    o = np.zeros((n, 2))
    key_bank = np.zeros((n, 2, 2))
    value_bank = np.zeros((n, 12, 2))
    counts = np.zeros_like(o)
    cache = []

    for t in range(steps):
        anc = None if anchor is None else anchor[t]
        z = x[:, t] @ wi+p @ wr
        current = z+db
        drive = np.tanh(z)
        query = queries[:, t]
        read_gate = read_gates[:, t]
        write_gate = write_gates[:, t]
        writers = write_gate[:, 0] == 1.
        has_write = writers.any()
        keys_before = key_bank
        values_before = value_bank

        scores = TEMPERATURE*(keys_before @ projected_queries[:, t, :, None])[..., 0]
        exponentials = np.exp(scores-scores.max(axis=1, keepdims=True))
        weights = exponentials/exponentials.sum(axis=1, keepdims=True)
        if swap_read:
            weights = weights[:, ::-1]
        context = values_before[..., 0]*weights[:, :1]+values_before[..., 1]*weights[:, 1:]
        injection = READ_GAIN*read_gate*context

        value_bank = MEMORY_RETENTION*values_before
        if has_write:
            value_bank[writers, :, 1] = value_bank[writers, :, 0]
            value_bank[writers, :, 0] = drive[writers]
            key_bank = keys_before.copy()
            key_bank[writers, 1] = keys_before[writers, 0]
            key_bank[writers, 0] = query[writers]

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
                          states=np.concatenate((d, v, o), axis=1), key_bank=key_bank,
                          value_bank=value_bank, keys_before=keys_before, values_before=values_before,
                          scores=scores, weights=weights, read_gate=read_gate, write_gate=write_gate,
                          drive=drive, query=query, writers=writers, has_write=has_write))
        p = hs
    return counts, cache


def loss_grad(x, y, w, arm, q):
    _, wr, _, _, wo, _ = population.unpack(w[:NATIVE_PARAMETERS], 6)
    counts, cache = forward(x, w, arm, q)
    loss, gcounts = population.loss_counts(counts, y, 'CE')
    grad = np.zeros_like(w)
    gwi, gwr, gdb, gvb, gwo, gob = population.unpack(grad[:NATIVE_PARAMETERS], 6)
    gmatrix = grad[NATIVE_PARAMETERS:].reshape(2, 2)
    learn_matrix = arm == 'LEARN'
    n = len(x)
    gd = np.zeros((n, 12))
    gv = np.zeros((n, 6))
    gp = np.zeros_like(gv)
    go = np.zeros((n, 2))
    gvalue = np.zeros((n, 12, 2))

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

        gz = direct
        if step['has_write']:
            writers = step['writers']
            gz = direct.copy()
            gz[writers] += gvalue[writers, :, 0]*(1.-step['drive'][writers]**2)
            gvalue[writers, :, 0] = gvalue[writers, :, 1]
            gvalue[writers, :, 1] = 0.
        gcontext = READ_GAIN*step['read_gate']*direct
        weights = step['weights']
        gvalue = MEMORY_RETENTION*gvalue+gcontext[..., None]*weights[:, None, :]
        if learn_matrix:
            gweights = (step['values_before'].transpose(0, 2, 1) @ gcontext[..., None])[..., 0]
            gscores = weights*(gweights-(weights*gweights).sum(axis=1, keepdims=True))
            gprojected = TEMPERATURE*(gscores[:, None, :] @ step['keys_before'])[:, 0, :]
            gmatrix += step['query'].T @ gprojected

        gwi += x[:, t].T @ gz
        gwr += step['prev'].T @ gz
        gdb += direct.sum(axis=0)
        gp = gprev_output+gz @ wr.T

    grad += .0002*w/len(w)
    return loss+.0001*np.mean(w*w), grad, counts


def objective(x, y, w, arm, q, anchor=None):
    counts, _ = forward(x, w, arm, q, anchor)
    return population.loss_counts(counts, y, 'CE')[0]+.0001*np.mean(w*w)


def evaluate(x, w, arm, q, swap_read=False):
    _, wr, _, _, wo, _ = population.unpack(w[:NATIVE_PARAMETERS], 6)
    fanout_r = np.count_nonzero(wr, axis=1)
    fanout_o = np.count_nonzero(wo, axis=1)
    count_chunks = []
    resource_chunks = []
    weight_chunks = []
    for start in range(0, len(x), 32):
        batch = x[start:start+32]
        counts, cache = forward(batch, w, arm, q, swap_read=swap_read)
        count_chunks.append(counts)
        weight_chunks.append(np.stack([cache[t]['weights'] for t in CUE_STEPS], axis=1))
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
            fields = {'states': 'states', 'currents': 'I', 'injections': 'A', 'key_bank': 'key_bank',
                      'value_bank': 'value_bank', 'scores': 'scores', 'weights': 'weights',
                      'read_gate': 'read_gate', 'write_gate': 'write_gate', 'drive': 'drive'}
            saved = {key: np.stack([step[field][:8] for step in cache], axis=1) for key, field in fields.items()}
            saved['spikes'] = np.stack([np.concatenate((step['hs'][:8], step['os'][:8]), axis=1)
                                       for step in cache], axis=1)
            saved['candidates'] = np.stack([np.concatenate((step['dc'][:8], step['vc'][:8], step['oc'][:8]), axis=1)
                                           for step in cache], axis=1)
    saved['counts'] = np.concatenate(count_chunks)
    saved['resources'] = np.concatenate(resource_chunks)
    saved['cue_weights'] = np.concatenate(weight_chunks)
    return saved
