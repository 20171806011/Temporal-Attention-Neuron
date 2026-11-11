"""Input-gated two-timescale traces on the native DEND branch interface."""

import numpy as np

import model as native
import population


LEGACY_ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2')
BETAS = {'SCALE': 2., 'SCALE_MEAN': 0.}
ARMS = LEGACY_ARMS + tuple(BETAS)
RHO = np.array([.85, .97])


def forward(x, w, arm, q, anchor=None):
    if arm in LEGACY_ARMS:
        return native.forward(x, w, arm, q, anchor)
    beta = BETAS[arm]
    wi, wr, db, vb, wo, ob = population.unpack(w, 6)
    n, steps, _ = x.shape
    d = np.zeros((n, 12))
    v = np.zeros((n, 6))
    p = np.zeros_like(v)
    o = np.zeros((n, 2))
    traces = np.zeros((n, 12, 2))
    counts = np.zeros_like(o)
    gates = np.minimum(1., np.abs(x).sum(axis=2, keepdims=True))
    cache = []

    for t in range(steps):
        anc = None if anchor is None else anchor[t]
        z = x[:, t] @ wi + p @ wr
        current = z + db
        drive = np.tanh(z)
        gate = gates[:, t]
        rate = .5*gate
        retention = (1.-rate)[..., None]*RHO

        difference = traces[..., 0]-traces[..., 1]
        weight_fast = 1./(1.+np.exp(-beta*drive*difference))
        context = traces[..., 1]+weight_fast*difference
        injection = gate*context
        traces = retention*traces+(rate*drive)[..., None]

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
                          states=np.concatenate((d, v, o), axis=1), traces=traces, weight_fast=weight_fast,
                          context=context, gate=gate, drive=drive, rate=rate, retention=retention,
                          difference=difference))
        p = hs
    return counts, cache


def loss_grad(x, y, w, arm, q, zero_surrogate=False):
    if arm in LEGACY_ARMS:
        return native.loss_grad(x, y, w, arm, q, zero_surrogate)
    beta = BETAS[arm]
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
    gtraces = np.zeros((n, 12, 2))
    slope = (lambda value: np.zeros_like(value)) if zero_surrogate else population.slope

    for t in reversed(range(x.shape[1])):
        step = cache[t]
        goc = go*(1.-step['os'])+((gcounts if t >= q else 0)-go*step['oc'])*slope(step['oc'])
        gwo += step['prev'].T @ goc
        gob += goc.sum(axis=0)
        gprev_output = goc @ wo.T
        go = .85*goc

        gvc = gv*(1.-step['hs'])+(gp-gv*step['vc'])*slope(step['vc'])
        gvb += gvc.sum(axis=0)
        gv = .85*gvc
        gdc = gd+.5*np.repeat(gvc, 2, axis=1)*(step['dc'] > 0.)
        direct = .25*gdc
        gd = .95*gdc

        gu = step['rate']*gtraces.sum(axis=2)
        gtraces = step['retention']*gtraces
        gc = step['gate']*direct
        a = step['weight_fast']
        glogit = gc*step['difference']*a*(1.-a)
        gu += beta*step['difference']*glogit
        gcompatibility = beta*step['drive']*glogit
        gtraces[..., 0] += a*gc+gcompatibility
        gtraces[..., 1] += (1.-a)*gc-gcompatibility

        gz = direct+gu*(1.-step['drive']**2)
        gwi += x[:, t].T @ gz
        gwr += step['prev'].T @ gz
        gdb += direct.sum(axis=0)
        gp = gprev_output+gz @ wr.T

    grad += .0002*w/len(w)
    return loss+.0001*np.mean(w*w), grad, counts


def objective(x, y, w, arm, q, anchor=None):
    if arm in LEGACY_ARMS:
        return native.objective(x, y, w, arm, q, anchor)
    counts = forward(x, w, arm, q, anchor)[0]
    return population.loss_counts(counts, y, 'CE')[0]+.0001*np.mean(w*w)


def evaluate(x, w, arm, q):
    if arm in LEGACY_ARMS:
        return native.evaluate(x, w, arm, q)
    _, wr, _, _, wo, _ = population.unpack(w, 6)
    fanout_r = np.count_nonzero(wr, axis=1)
    fanout_o = np.count_nonzero(wo, axis=1)
    count_chunks = []
    resource_chunks = []
    for start in range(0, len(x), 32):
        batch = x[start:start+32]
        counts, cache = forward(batch, w, arm, q)
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
            names = {'states': 'states', 'currents': 'I', 'injections': 'A', 'traces': 'traces',
                     'weights_fast': 'weight_fast', 'contexts': 'context', 'gates': 'gate', 'drive': 'drive'}
            saved = {key: np.stack([step[field][:8] for step in cache], axis=1) for key, field in names.items()}
            saved['spikes'] = np.stack([np.concatenate((step['hs'], step['os']), axis=1)[:8]
                                       for step in cache], axis=1)
            saved['candidates'] = np.stack([np.concatenate((step['dc'], step['vc'], step['oc']), axis=1)[:8]
                                           for step in cache], axis=1)
    saved['counts'] = np.concatenate(count_chunks)
    saved['resources'] = np.concatenate(resource_chunks)
    return saved
