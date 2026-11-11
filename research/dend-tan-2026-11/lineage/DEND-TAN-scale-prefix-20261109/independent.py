"""Independent segment replay for the D16 prefix-state experiment."""

import numpy as np

from reference import split


ENTRY = 25
CONDITIONS = {
    'GAIN2': ('nominal',),
    'SCALE': ('nominal', 'prefix_mean', 'mean_native', 'mean_traces'),
    'SCALE_MEAN': ('nominal', 'prefix_mean'),
}


def segment(x, parameters, q, start, stop, beta, entry_native, entry_traces):
    wi, wr, db, vb, wo, ob = parameters
    n = len(x)
    d, v, o, p = np.split(entry_native, (12, 18, 20), axis=1)
    counts = np.zeros_like(o)
    resources = np.zeros((n, 4))
    recurrent_fanout = np.count_nonzero(wr, axis=1)
    output_fanout = np.count_nonzero(wo, axis=1)
    history = {key: [] for key in ('states', 'spikes', 'injections')}
    if beta is not None:
        fast, slow = entry_traces[..., 0], entry_traces[..., 1]
        history.update(traces=[], weights_fast=[])

    for tick in range(start, stop):
        external = x[:, tick]
        drive = np.einsum('ni,ij->nj', external, wi, optimize=False)
        drive += np.einsum('ni,ij->nj', p, wr, optimize=False)
        current = drive + db
        injection = current
        if beta is not None:
            signed = np.tanh(drive)
            gate = np.minimum(np.linalg.norm(external, ord=1, axis=1, keepdims=True), 1.)
            # sigmoid(z) = (1+tanh(z/2))/2, with read before write.
            weight_fast = .5*(1+np.tanh((beta/2)*signed*(fast-slow)))
            injection = gate*(weight_fast*fast + (1-weight_fast)*slow)
            rate = .5*gate
            fast = .85*(1-rate)*fast + rate*signed
            slow = .97*(1-rate)*slow + rate*signed

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
        if beta is not None:
            history['traces'].append(np.stack((fast[:8], slow[:8]), axis=-1))
            history['weights_fast'].append(weight_fast[:8].copy())

    final_native = np.column_stack((d, v, o, p))
    final_traces = np.stack((fast, slow), axis=-1) if beta is not None else None
    archive = {name: np.stack(steps, axis=1) for name, steps in history.items()}
    return final_native, final_traces, counts, resources, archive


def replay(x, w, arm, q, condition):
    assert condition in CONDITIONS[arm] and q == 28 and x.shape[1:] == (36, 6)
    parameters = split(w)
    n = len(x)
    beta = {'GAIN2': None, 'SCALE': 2., 'SCALE_MEAN': 0.}[arm]
    native_beta = 0. if condition in ('prefix_mean', 'mean_native') else beta
    trace_beta = 0. if condition in ('prefix_mean', 'mean_traces') else beta
    initial_native = np.zeros((n, 26))
    initial_traces = None if beta is None else np.zeros((n, 12, 2))
    native, traces, _, resources_pre, archive_pre = segment(
        x, parameters, q, 0, ENTRY, native_beta, initial_native, initial_traces)
    if trace_beta != native_beta:
        _, traces, _, _, _ = segment(x, parameters, q, 0, ENTRY, trace_beta, initial_native, initial_traces)
    _, _, counts, resources_post, archive_post = segment(x, parameters, q, ENTRY, 36, beta, native, traces)
    result = {name: np.concatenate((pre, archive_post[name]), axis=1) for name, pre in archive_pre.items()}
    result.update(counts=counts, resources=resources_pre+resources_post, entry_native=native)
    if beta is not None:
        result['entry_traces'] = traces
    return result
