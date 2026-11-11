"""Training arms for the two-timescale trace compatibility-gradient study."""

import numpy as np

import population
import scale_model


FORWARD_ARMS = {'DYN': 'SCALE', 'DETACH': 'SCALE', 'MEAN': 'SCALE_MEAN'}
ARMS = tuple(FORWARD_ARMS)


def forward(x, w, arm, q, anchor=None):
    return scale_model.forward(x, w, FORWARD_ARMS[arm], q, anchor)


def loss_grad(x, y, w, arm, q):
    if arm != 'DETACH':
        return scale_model.loss_grad(x, y, w, FORWARD_ARMS[arm], q)

    # Mirror the sealed SCALE recurrence with the read weights held constant
    # in the backward pass. Each call recomputes them in the native forward.
    counts, cache = forward(x, w, arm, q)
    _, wr, _, _, wo, _ = population.unpack(w, 6)
    loss, gcounts = population.loss_counts(counts, y, 'CE')
    grad = np.zeros_like(w)
    gwi, gwr, gdb, gvb, gwo, gob = population.unpack(grad, 6)
    n = len(x)
    gd = np.zeros((n, 12))
    gv = np.zeros((n, 6))
    gp = np.zeros_like(gv)
    go = np.zeros((n, 2))
    gtraces = np.zeros((n, 12, 2))

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

        gu = step['rate']*gtraces.sum(axis=2)
        gtraces = step['retention']*gtraces
        gc = step['gate']*direct
        a = step['weight_fast']
        gtraces[..., 0] += a*gc
        gtraces[..., 1] += (1.-a)*gc

        gz = direct+gu*(1.-step['drive']**2)
        gwi += x[:, t].T @ gz
        gwr += step['prev'].T @ gz
        gdb += direct.sum(axis=0)
        gp = gprev_output+gz @ wr.T

    grad += .0002*w/len(w)
    return loss+.0001*np.mean(w*w), grad, counts
