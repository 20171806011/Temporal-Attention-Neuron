"""Independent reconstruction from visible input prefixes and native replay."""

import numpy as np

import independent_capacity_data as data
from independent_warm import CUES, TRACES, matched_cosine
from reference import split


HISTORIES = ('four', 'three0', 'three2', 'two')
RULES = ('original', 'replay', 'dense')
COSINE_THRESHOLD = 1./np.sqrt(2.)
NO_CONFIDENCE = -2.


def assemble(base, history):
    assert history in HISTORIES
    x, q = data.assemble(base, 'full4')
    removed = {'four': (), 'three0': (2,), 'three2': (0,), 'two': (0, 2)}[history]
    if removed:
        x[:, removed] = 0.
    return x, q


def reference(x, w, onsets):
    """Run native state on [0, onset), retaining just the last two visible writes."""
    n, horizon, channels = x.shape
    onsets = np.asarray(onsets, dtype=np.int64)
    assert channels == 6 and w.shape == (180,) and onsets.shape == (n,)
    assert np.all((0 <= onsets) & (onsets <= horizon))
    ticks = np.arange(horizon)
    active = ticks[None, :] < onsets[:, None]
    writes = (x[..., 5] == 1.) & active
    newer = np.cumsum(writes[:, ::-1], axis=1)[:, ::-1]
    retained = np.where((writes & (newer > 2))[..., None], 0., x)
    wi, wr, db, vb, wo, ob = split(w[:176])
    d, v, o, p = np.zeros((n, 12)), np.zeros((n, 6)), np.zeros((n, 2)), np.zeros((n, 6))
    fanout_h, fanout_o = np.count_nonzero(wr, axis=1), np.count_nonzero(wo, axis=1)
    resources = np.zeros((n, 4))
    for tick in range(int(onsets.max(initial=0))):
        z = np.einsum('ni,ij->nj', retained[:, tick], wi, optimize=False)
        z += np.einsum('ni,ij->nj', p, wr, optimize=False)
        dc = .95*d+.25*(z+db)
        positive = np.maximum(dc, 0.)
        vc = .85*v+.5*(positive[:, ::2]+positive[:, 1::2])+vb
        oc = .85*o+np.einsum('ni,ij->nj', p, wo, optimize=False)+ob
        hs, os = vc > .5, oc > .5
        executing = active[:, tick, None]
        delivered = np.column_stack((hs.sum(axis=1), os.sum(axis=1), p @ fanout_h, p @ fanout_o))
        resources += delivered*executing
        d = np.where(executing, dc, d)
        v = np.where(executing, vc*~hs, v)
        o = np.where(executing, oc*~os, o)
        p = np.where(executing, hs, p)
    return np.column_stack((d, v, o, p)), resources


def expected_log(x, first_read):
    """Extract the exact before-first-read log directly from input, without a FIFO implementation."""
    n, horizon, _ = x.shape
    write_values, write_ticks = np.zeros((n, 2, 6)), np.full((n, 2), -1, dtype=np.int64)
    other_values, other_ticks = np.zeros((n, horizon, 6)), np.full((n, horizon), -1, dtype=np.int64)
    other_count = np.zeros(n, dtype=np.int64)
    for row, first in enumerate(first_read):
        prefix = x[row, :horizon if first < 0 else first]
        writing = prefix[:, 5] == 1.
        writes = np.flatnonzero(writing)[-2:]
        others = np.flatnonzero(~writing & np.any(prefix != 0., axis=1))
        if len(writes):
            write_ticks[row, -len(writes):] = writes
            write_values[row, -len(writes):] = prefix[writes]
        other_ticks[row, :len(others)] = others
        other_values[row, :len(others)] = prefix[others]
        other_count[row] = len(others)
    return dict(logged_write_values=write_values, logged_write_ticks=write_ticks,
                logged_nonwrite_values=other_values, logged_nonwrite_ticks=other_ticks,
                logged_nonwrite_count=other_count)


def replay(x, w, q):
    """Replay native candidate with a single first-read decision and direct prefix reconstruction."""
    n, steps, channels = x.shape
    assert channels == 6 and 3 <= q < steps and w.shape == (180,)
    wi, wr, db, vb, wo, ob = split(w[:176])
    matcher = w[176:].reshape(2, 2)
    d, v, o, p = np.zeros((n, 12)), np.zeros((n, 6)), np.zeros((n, 2)), np.zeros((n, 6))
    keys, values = np.zeros((n, 2, 2)), np.zeros((n, 12, 2))
    occupied = np.zeros((n, 2), dtype=bool)
    evicted, previously_reading = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    counts, resources = np.zeros((n, 2)), np.zeros((n, 6))
    fanout_h, fanout_o = np.count_nonzero(wr, axis=1), np.count_nonzero(wo, axis=1)
    traces, cues = {name: [] for name in TRACES}, {name: [] for name in CUES}
    onset_steps = np.zeros((n, steps), dtype=bool)
    interventions, eviction_seen = np.zeros_like(onset_steps), np.zeros_like(onset_steps)
    confidence = np.full((n, steps), NO_CONFIDENCE)
    first_read_tick = np.full(n, -1, dtype=np.int64)
    replay_steps = np.zeros(n, dtype=np.int64)
    replay_resources, replay_reference = np.zeros((n, 4)), np.zeros((n, 26))
    for tick, external in enumerate(x.swapaxes(0, 1)):
        observed, marker = external[:, 2:4], external[:, 5:6]
        writing = marker[:, 0] == 1.
        read = (1.-marker)*np.minimum(1., np.linalg.norm(observed, axis=1, keepdims=True))
        reading = read[:, 0] > 0.
        onset = reading & ~previously_reading
        previously_reading = reading
        first = onset & (first_read_tick < 0)
        first_read_tick[first] = tick
        projected = np.einsum('ni,ij->nj', observed, matcher, optimize=False)
        onset_steps[:, tick], eviction_seen[:, tick] = onset, evicted
        eligible = first & evicted
        confidence[eligible, tick] = matched_cosine(projected[eligible], keys[eligible])
        apply = eligible & (confidence[:, tick] >= COSINE_THRESHOLD)
        interventions[:, tick] = apply

        archive_cue = q-3 <= tick < q
        if archive_cue:
            cues['cue_native_before'].append(np.column_stack((d, v, o, p)))
        if apply.any():
            stop = np.full(np.count_nonzero(apply), tick, dtype=np.int64)
            rebuilt, extra = reference(x[apply, :tick], w, stop)
            replay_steps[apply], replay_resources[apply], replay_reference[apply] = stop, extra, rebuilt
            d[apply], v[apply], o[apply], p[apply] = np.split(rebuilt, (12, 18, 20), axis=1)
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
        keys[writing, 1], keys[writing, 0] = keys[writing, 0], observed[writing]
        values[writing, :, 1], values[writing, :, 0] = values[writing, :, 0], drive[writing]

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
        resources[:, 2] += p @ fanout_h
        resources[:, 3] += p @ fanout_o
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
                **{name: np.stack(arrays, axis=1) for name, arrays in cues.items()},
                **expected_log(x, first_read_tick), first_read_tick=first_read_tick, replay_steps=replay_steps,
                replay_resources=replay_resources, replay_reference=replay_reference)
