"""Independent FIFO matching dynamics and fixed-anchor surrogate."""

import numpy as np

from reference import split


ARMS = ('LEARN', 'ORACLE', 'MEAN')
ROTATION = np.array([[0., -1.], [1., 0.]])


def run(x, theta, arm, q, swap_read, mode, anchors=None):
    assert theta.shape == (180,) and arm in ARMS and q == 28 and x.shape[1:] == (36, 6)
    assert not swap_read or arm != 'MEAN'
    wi, wr, db, vb, wo, ob = split(theta[:176])
    matcher = {'LEARN': theta[176:].reshape(2, 2), 'ORACLE': ROTATION, 'MEAN': np.zeros((2, 2))}[arm]
    n = len(x)
    d, v, o = np.zeros((n, 12)), np.zeros((n, 6)), np.zeros((n, 2))
    p, counts = np.zeros_like(v), np.zeros_like(o)
    recent_key = np.zeros((n, 2))
    early_key = np.zeros_like(recent_key)
    recent_value, early_value = np.zeros_like(d), np.zeros_like(d)
    if mode == 'archive':
        resources = np.zeros((n, 6))
        recurrent_fanout = np.count_nonzero(wr, axis=1)
        output_fanout = np.count_nonzero(wo, axis=1)
        names = ('states', 'spikes', 'candidates', 'currents', 'injections', 'key_bank', 'value_bank',
                 'scores', 'weights', 'read_gate', 'write_gate', 'drive')
        history = {key: [] for key in names}
        cue_weights = []
    snapshots = []
    for tick in range(x.shape[1]):
        external = x[:, tick]
        observed = external[:, 2:4]
        marker = external[:, 5:6]
        writing = marker == 1.
        read = (1-marker)*np.minimum(1., np.linalg.norm(observed, axis=1, keepdims=True))
        # Two independent contractions preserve the orientation q^T A k.
        query_row = np.einsum('ni,ij->nj', observed, matcher, optimize=False)
        recent_score = 4*np.einsum('ni,ni->n', query_row, recent_key, optimize=False)
        early_score = 4*np.einsum('ni,ni->n', query_row, early_key, optimize=False)
        scores = np.column_stack((recent_score, early_score))
        exp_scores = np.exp(scores-scores.max(axis=1, keepdims=True))
        weights = exp_scores/exp_scores.sum(axis=1, keepdims=True)
        if swap_read:
            weights = weights[:, ::-1]
        injection = 2*read*(weights[:, :1]*recent_value+weights[:, 1:]*early_value)

        z = np.einsum('ni,ij->nj', external, wi, optimize=False)
        z += np.einsum('ni,ij->nj', p, wr, optimize=False)
        current = z+db
        drive = np.tanh(z)
        early_key = np.where(writing, recent_key, early_key)
        recent_key = np.where(writing, observed, recent_key)
        early_value = .97*np.where(writing, recent_value, early_value)
        recent_value = np.where(writing, drive, .97*recent_value)

        d = .95*d+.25*(current+injection)
        frozen = None if anchors is None else anchors[tick]
        local = np.where((d if frozen is None else frozen['d']) > 0., d, 0.)
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
            if 25 <= tick < q:
                cue_weights.append(weights)
            step = dict(states=np.column_stack((d[:8], v[:8], o[:8])),
                        spikes=np.column_stack((hs[:8], os[:8])),
                        candidates=np.column_stack((d[:8], vc[:8], oc[:8])),
                        currents=current[:8], injections=injection[:8],
                        key_bank=np.stack((recent_key[:8], early_key[:8]), axis=1),
                        value_bank=np.stack((recent_value[:8], early_value[:8]), axis=-1),
                        scores=scores[:8], weights=weights[:8], read_gate=read[:8], write_gate=marker[:8], drive=drive[:8])
            for name, values in step.items():
                history[name].append(values)
        p = hs
    if mode == 'archive':
        return dict(counts=counts, resources=resources, cue_weights=np.stack(cue_weights, axis=1),
                    **{name: np.stack(values, axis=1) for name, values in history.items()})
    if mode == 'anchor':
        return snapshots
    return counts


def replay(x, w, arm, q, swap_read=False):
    return run(x, w, arm, q, swap_read, 'archive')


def anchor(x, w, arm, q):
    return run(x, w, arm, q, False, 'anchor')


def objective(x, y, w, arm, q, anchor):
    counts = run(x, w, arm, q, False, 'objective', anchor)
    margin = .5*(counts[:, 1]-counts[:, 0])
    return float(np.mean(np.logaddexp(0., (1-2*y)*margin))+.0001*np.mean(w*w))
