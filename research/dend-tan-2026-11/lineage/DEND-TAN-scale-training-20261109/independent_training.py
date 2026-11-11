"""Independent anchored objective for the two-trace surrogate gradients."""

import numpy as np

from reference import split


BETAS = {'DYN': 2., 'DETACH': 2., 'MEAN': 0.}


def count_objective(counts, y, w):
    margin = .5*(counts[:, 1].astype(float)-counts[:, 0].astype(float))
    return float(np.logaddexp(0., (1-2*y)*margin).mean() + .0001*np.mean(w**2))


def spike_line(value, center, hard_spike):
    return hard_spike+(value-center)/(1+5*np.abs(center-.5))**2


def rollout(x, w, arm, q, anchors):
    beta = BETAS[arm]
    wi, wr, db, vb, wo, ob = split(w)
    n = len(x)
    d, v, o = np.zeros((n, 12)), np.zeros((n, 6)), np.zeros((n, 2))
    p, fast, slow, counts = np.zeros_like(v), np.zeros_like(d), np.zeros_like(d), np.zeros_like(o)
    snapshots = []
    for tick in range(x.shape[1]):
        external = x[:, tick]
        z = np.einsum('ni,ij->nj', external, wi, optimize=False)
        z += np.einsum('ni,ij->nj', p, wr, optimize=False)
        drive = np.tanh(z)
        gate = np.minimum(1., np.linalg.norm(external, ord=1, axis=1, keepdims=True))
        frozen = None if anchors is None else anchors[tick]
        if arm == 'DETACH' and frozen is not None:
            alpha = frozen['alpha']
        else:
            alpha = .5*(1+np.tanh((beta/2)*drive*(fast-slow)))
        injection = gate*(alpha*fast+(1-alpha)*slow)
        write = .5*gate
        fast = .85*(1-write)*fast + write*drive
        slow = .97*(1-write)*slow + write*drive

        d = .95*d+.25*(z+db+injection)
        open_branch = d > 0 if frozen is None else frozen['d'] > 0
        local = np.where(open_branch, d, 0.)
        vc = .85*v+.5*(local[:, ::2]+local[:, 1::2])+vb
        oc = .85*o+np.einsum('ni,ij->nj', p, wo, optimize=False)+ob
        if frozen is None:
            hs, os = (vc > .5).astype(float), (oc > .5).astype(float)
            snapshots.append(dict(alpha=alpha, d=d, vc=vc, oc=oc, hs=hs, os=os))
        else:
            hs = spike_line(vc, frozen['vc'], frozen['hs'])
            os = spike_line(oc, frozen['oc'], frozen['os'])
        v, o, p = vc*(1-hs), oc*(1-os), hs
        if tick >= q:
            counts += os
    return counts, snapshots


def anchor(x, w, arm, q):
    return rollout(x, w, arm, q, None)[1]


def objective(x, y, w, arm, q, anchor):
    counts, _ = rollout(x, w, arm, q, anchor)
    return count_objective(counts, y, w)
