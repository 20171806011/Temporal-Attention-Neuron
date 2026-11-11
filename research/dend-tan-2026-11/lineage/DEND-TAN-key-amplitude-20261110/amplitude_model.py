"""Fixed-weight read direction and amplitude interventions for KEY memory."""

import numpy as np

import key_model
import population


CUE_STEPS = (25, 26, 27)
CONDITIONS = {
    'KEY': ('nominal', 'swap_read', 'swap_direction', 'swap_magnitude', 'mean_read', 'mean_matched'),
    'KEY_MEAN': ('nominal',),
    'GAIN2': ('nominal',),
}
MATCHED_SELECTORS = {
    'swap_direction': (-1., 1.),
    'swap_magnitude': (1., -1.),
    'mean_matched': (0., 1.),
}


def conditions(arm):
    return CONDITIONS[arm]


def _rescale(direction, amplitude):
    direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
    target_norm = np.linalg.norm(amplitude, axis=1, keepdims=True)
    scale = np.divide(target_norm, direction_norm, out=np.zeros_like(target_norm), where=direction_norm > 0.)
    matched = direction*scale
    fallback = (direction_norm[:, 0] == 0.) & (target_norm[:, 0] > 0.)
    matched[fallback, 0] = target_norm[fallback, 0]
    return matched, fallback


def _read(memory, selector, read_gate):
    context = memory[..., 0]*selector[:, :1]+memory[..., 1]*selector[:, 1:]
    return key_model.READ_GAIN*read_gate*context


def _matched_forward(x, w, q, direction_gamma, amplitude_gamma):
    keys, selectors, writes, read_gates = key_model._address_signals(x, direction_gamma)
    amplitude_selectors = .5+amplitude_gamma*(keys-.5)
    retention = key_model.MEMORY_RETENTION*(1.-writes)
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
        z = x[:, t] @ wi+p @ wr
        current = z+db
        drive = np.tanh(z)
        selector = selectors[:, t]
        write = writes[:, t]
        read_gate = read_gates[:, t]
        keep = retention[:, t]

        direction = _read(memory, selector, read_gate)
        amplitude = _read(memory, amplitude_selectors[:, t], read_gate)
        injection, fallback = _rescale(direction, amplitude)
        memory = keep[:, None, :]*memory+write[:, None, :]*drive[..., None]

        d = .95*d+.25*(current+injection)
        local = np.maximum(d, 0.)
        vc = .85*v+.5*local.reshape(n, 6, 2).sum(axis=2)+vb
        oc = .85*o+p @ wo+ob
        hs = (vc > .5).astype(float)
        os = (oc > .5).astype(float)
        v = vc*(1.-hs)
        o = oc*(1.-os)
        if t >= q:
            counts += os
        cache.append(dict(I=current, A=injection, dc=d, vc=vc, oc=oc, hs=hs, os=os, prev=p,
                          states=np.concatenate((d, v, o), axis=1), memory=memory, keys=keys[:, t],
                          selector=selector, writes=write, read_gate=read_gate, drive=drive,
                          fallback=fallback))
        p = hs
    return counts, cache


def evaluate(x, w, arm, q, condition):
    if condition not in conditions(arm):
        raise ValueError(f'{condition!r} is unavailable for {arm!r}')
    keyed = arm != 'GAIN2'
    matched = condition in MATCHED_SELECTORS
    _, wr, _, _, wo, _ = population.unpack(w, 6)
    fanout_r = np.count_nonzero(wr, axis=1)
    fanout_o = np.count_nonzero(wo, axis=1)
    count_chunks = []
    resource_chunks = []
    read_chunks = []
    memory_chunks = []
    fallback_chunks = []

    for start in range(0, len(x), 32):
        batch = x[start:start+32]
        if matched:
            counts, cache = _matched_forward(batch, w, q, *MATCHED_SELECTORS[condition])
        else:
            read_arm = 'KEY_MEAN' if condition == 'mean_read' else arm
            counts, cache = key_model.forward(batch, w, read_arm, q, swap_read=condition == 'swap_read')
        count_chunks.append(counts)
        read_chunks.append(np.stack([cache[t]['A'] for t in CUE_STEPS], axis=1))
        if keyed:
            memory_chunks.append(np.stack([cache[t-1]['memory'] for t in CUE_STEPS], axis=1))
        if matched:
            fallback_chunks.append(np.stack([cache[t]['fallback'] for t in CUE_STEPS], axis=1))

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
            fields = {'states': 'states', 'currents': 'I', 'injections': 'A'}
            if keyed:
                fields.update({name: name for name in ('memory', 'keys', 'selector', 'writes', 'read_gate', 'drive')})
            saved = {key: np.stack([step[field][:8] for step in cache], axis=1) for key, field in fields.items()}
            saved['spikes'] = np.stack([np.concatenate((step['hs'][:8], step['os'][:8]), axis=1)
                                       for step in cache], axis=1)
            saved['candidates'] = np.stack([np.concatenate((step['dc'][:8], step['vc'][:8], step['oc'][:8]), axis=1)
                                           for step in cache], axis=1)

    saved['counts'] = np.concatenate(count_chunks)
    saved['resources'] = np.concatenate(resource_chunks)
    saved['read_vectors'] = np.concatenate(read_chunks)
    saved['direction_fallback'] = (np.concatenate(fallback_chunks) if matched
                                   else np.zeros((len(x), len(CUE_STEPS)), dtype=bool))
    if keyed:
        saved['cue_memory'] = np.concatenate(memory_chunks)
    return saved
