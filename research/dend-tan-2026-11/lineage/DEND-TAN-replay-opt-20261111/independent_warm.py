"""Independent native replay for warmed references and evicted-record gating."""

import numpy as np

from reference import split


ARMS = ('LEARN', 'ORACLE')
RULES = ('original', 'match_zero', 'eviction_zero', 'eviction_warm', 'eviction_late_native')
ROTATION = np.array([[0., -1.], [1., 0.]])
COSINE_THRESHOLD = 1./np.sqrt(2.)
NO_CONFIDENCE = -2.
WARM_STEPS = 25
TRACES = ('states', 'spikes', 'candidates', 'currents', 'injections', 'key_bank', 'value_bank',
          'scores', 'weights', 'read_gate', 'write_gate', 'drive')
CUES = ('cue_scores', 'cue_weights', 'cue_read_vectors', 'cue_native_before', 'cue_native_after')


def matched_cosine(projected, keys):
    dot = np.sum(keys*projected[:, None, :], axis=-1)
    scale = np.linalg.norm(projected, axis=1)[:, None]*np.linalg.norm(keys, axis=2)
    scores = np.full_like(dot, NO_CONFIDENCE)
    np.divide(dot, scale, out=scores, where=scale > 0.)
    return scores.max(axis=1)


def warm_reference(w):
    """Run the intrinsic equations from rest on 25 zero external inputs."""
    assert w.shape == (180,)
    _, wr, db, vb, wo, ob = split(w[:176])
    dend, voltage, output, pulses = np.zeros(12), np.zeros(6), np.zeros(2), np.zeros(6)
    for _ in range(WARM_STEPS):
        dend = .95*dend+.25*(np.einsum('i,ij->j', pulses, wr, optimize=False)+db)
        positive = np.maximum(dend, 0.)
        voltage_candidate = .85*voltage+.5*(positive[::2]+positive[1::2])+vb
        output_candidate = .85*output+np.einsum('i,ij->j', pulses, wo, optimize=False)+ob
        pulses = (voltage_candidate > .5).astype(float)
        voltage = voltage_candidate*(1.-pulses)
        output = output_candidate*(output_candidate <= .5)
    return np.concatenate((dend, voltage, output, pulses))


def replay(x, w, arm, q, rule, reference=None):
    assert x.ndim == 3 and x.shape[-1] == 6 and 3 <= q < x.shape[1]
    assert arm in ARMS and rule in RULES and w.shape == (180,)
    n, steps = x.shape[:2]
    if rule == 'eviction_warm':
        reference = warm_reference(w) if reference is None else reference
        assert reference.shape == (26,)
    elif rule == 'eviction_late_native':
        assert reference.shape == (n, 26)
    else:
        assert reference is None
        reference = np.zeros(26)
    ref_d, ref_v, ref_o, ref_p = np.split(reference, (12, 18, 20), axis=-1)
    wi, wr, db, vb, wo, ob = split(w[:176])
    matcher = w[176:].reshape(2, 2) if arm == 'LEARN' else ROTATION
    d, v, o, p = np.zeros((n, 12)), np.zeros((n, 6)), np.zeros((n, 2)), np.zeros((n, 6))
    keys, values = np.zeros((n, 2, 2)), np.zeros((n, 12, 2))
    occupied = np.zeros((n, 2), dtype=bool)
    evicted, previously_reading = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    counts, resources = np.zeros((n, 2)), np.zeros((n, 6))
    recurrent_fanout, output_fanout = np.count_nonzero(wr, axis=1), np.count_nonzero(wo, axis=1)
    traces, cues = {name: [] for name in TRACES}, {name: [] for name in CUES}
    onset_steps = np.zeros((n, steps), dtype=bool)
    interventions = np.zeros_like(onset_steps)
    eviction_seen = np.zeros_like(onset_steps)
    confidence = np.full((n, steps), NO_CONFIDENCE)
    for tick, external in enumerate(x.swapaxes(0, 1)):
        observed, marker = external[:, 2:4], external[:, 5:6]
        writing = marker[:, 0] == 1.
        read = (1.-marker)*np.minimum(1., np.linalg.norm(observed, axis=1, keepdims=True))
        reading = read[:, 0] > 0.
        onset = reading & ~previously_reading
        previously_reading = reading
        projected = np.einsum('ni,ij->nj', observed, matcher, optimize=False)
        onset_steps[:, tick] = onset
        eviction_seen[:, tick] = evicted
        if rule != 'original':
            eligible = onset if rule == 'match_zero' else onset & evicted
            confidence[eligible, tick] = matched_cosine(projected[eligible], keys[eligible])
        apply = confidence[:, tick] >= COSINE_THRESHOLD
        interventions[:, tick] = apply

        archive_cue = q-3 <= tick < q
        if archive_cue:
            cues['cue_native_before'].append(np.column_stack((d, v, o, p)))
        if apply.any():
            d, v, o, p = (np.where(apply[:, None], replacement, state)
                          for state, replacement in ((d, ref_d), (v, ref_v), (o, ref_o), (p, ref_p)))
        if archive_cue:
            cues['cue_native_after'].append(np.column_stack((d, v, o, p)))

        scores = 4*np.einsum('ni,nki->nk', projected, keys, optimize=False)
        exp_scores = np.exp(scores-scores.max(axis=1, keepdims=True))
        weights = exp_scores/exp_scores.sum(axis=1, keepdims=True)
        injection = 2*read*np.einsum('nk,ndk->nd', weights, values, optimize=False)
        z = np.einsum('ni,ij->nj', external, wi, optimize=False)
        z += np.einsum('ni,ij->nj', p, wr, optimize=False)
        current, drive = z+db, np.tanh(z)

        evicted |= writing & occupied.all(axis=1)
        occupied[writing] = np.column_stack((np.any(observed[writing] != 0., axis=1), occupied[writing, 0]))
        values *= .97
        keys[writing, 1] = keys[writing, 0]
        keys[writing, 0] = observed[writing]
        values[writing, :, 1] = values[writing, :, 0]
        values[writing, :, 0] = drive[writing]

        d = .95*d+.25*(current+injection)
        local = np.maximum(d, 0.)
        vc = .85*v+.5*(local[:, ::2]+local[:, 1::2])+vb
        oc = .85*o+np.einsum('ni,ij->nj', p, wo, optimize=False)+ob
        hs, os = (vc > .5).astype(float), (oc > .5).astype(float)
        v, o = vc*(1.-hs), oc*(1.-os)
        if tick >= q:
            counts += os
        resources[:, 0] += hs.sum(axis=1)
        resources[:, 1] += os.sum(axis=1)
        resources[:, 2] += np.einsum('ni,i->n', p, recurrent_fanout, optimize=False)
        resources[:, 3] += np.einsum('ni,i->n', p, output_fanout, optimize=False)
        if 16 <= tick < q:
            resources[:, 4] += hs.sum(axis=1)
            resources[:, 5] += os.sum(axis=1)
        if archive_cue:
            cues['cue_scores'].append(scores)
            cues['cue_weights'].append(weights)
            cues['cue_read_vectors'].append(injection)
        step = dict(states=np.column_stack((d, v, o)), spikes=np.column_stack((hs, os)),
                    candidates=np.column_stack((d, vc, oc)), currents=current, injections=injection,
                    key_bank=keys, value_bank=values, scores=scores, weights=weights,
                    read_gate=read, write_gate=marker, drive=drive)
        for name, array in step.items():
            traces[name].append(array[:8].copy())
        p = hs
    return dict(counts=counts, resources=resources, read_onset=onset_steps, reset_steps=interventions,
                match_confidence=confidence, eviction_seen=eviction_seen,
                **{name: np.stack(arrays, axis=1) for name, arrays in traces.items()},
                **{name: np.stack(arrays, axis=1) for name, arrays in cues.items()})
