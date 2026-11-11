"""Run the learned model with optional causal replay at its first read onset."""

import time

import numpy as np

import learned_model
import population
import warm_model


def _new_log(n, steps):
    return dict(
        logged_write_values=np.zeros((n, 2, 6)),
        logged_write_ticks=np.full((n, 2), -1, dtype=np.int64),
        logged_nonwrite_values=np.zeros((n, steps, 6)),
        logged_nonwrite_ticks=np.full((n, steps), -1, dtype=np.int64),
        logged_nonwrite_count=np.zeros(n, dtype=np.int64),
        replay_steps=np.zeros(n, dtype=np.int64),
        replay_resources=np.zeros((n, 4)),
        first_read_tick=np.full(n, -1, dtype=np.int64),
        replay_reference=np.zeros((n, 26)),
    )


def _new_timing():
    return dict(replay_wall_seconds=0., replay_batches=0, replay_scratch_peak_bytes=0)


def _record(log, frame, writers, tick):
    open_log = log['first_read_tick'] < 0
    write = open_log & writers
    write_values, write_ticks = log['logged_write_values'], log['logged_write_ticks']
    write_values[write, 0] = write_values[write, 1]
    write_ticks[write, 0] = write_ticks[write, 1]
    write_values[write, 1] = frame[write]
    write_ticks[write, 1] = tick

    rows = np.flatnonzero(open_log & ~writers & np.any(frame != 0., axis=1))
    positions = log['logged_nonwrite_count'][rows]
    log['logged_nonwrite_values'][rows, positions] = frame[rows]
    log['logged_nonwrite_ticks'][rows, positions] = tick
    log['logged_nonwrite_count'][rows] += 1


def _replay_prefix(log, triggered, stop, weights, fanout_r, fanout_o):
    rows = np.flatnonzero(triggered)
    n = len(rows)
    prefix = np.zeros((n, stop, 6))
    for name in ('write', 'nonwrite'):
        ticks = log[f'logged_{name}_ticks'][rows]
        row, slot = np.nonzero(ticks >= 0)
        prefix[row, ticks[row, slot]] = log[f'logged_{name}_values'][rows[row], slot]

    wi, wr, db, vb, wo, ob = weights
    d, v, o = np.zeros((n, 12)), np.zeros((n, 6)), np.zeros((n, 2))
    p = np.zeros_like(v)
    resources = np.zeros((n, 4))
    for tick in range(stop):
        z = prefix[:, tick] @ wi + p @ wr
        current = z + db
        d = .95 * d + .25 * current
        local = np.maximum(d, 0.)
        vc = .85 * v + .5 * local.reshape(n, 6, 2).sum(axis=2) + vb
        oc = .85 * o + p @ wo + ob
        hs, os = (vc > .5).astype(float), (oc > .5).astype(float)
        v, o = vc * (1. - hs), oc * (1. - os)
        resources[:, 0] += hs.sum(axis=1)
        resources[:, 1] += os.sum(axis=1)
        resources[:, 2] += p @ fanout_r
        resources[:, 3] += p @ fanout_o
        p = hs
    return np.concatenate((d, v, o, p), axis=1), resources, prefix.nbytes


def _forward(x, weights, matrix, q, fanout_r, fanout_o):
    wi, wr, db, vb, wo, ob = weights
    n, steps, _ = x.shape
    d, v, o = np.zeros((n, 12)), np.zeros((n, 6)), np.zeros((n, 2))
    p = np.zeros_like(v)
    key_bank = np.zeros((n, 2, 2))
    value_bank = np.zeros((n, 12, 2))
    previous_read = np.zeros(n, dtype=bool)
    read_onset = np.zeros((n, steps), dtype=bool)
    reset_steps = np.zeros_like(read_onset)
    eviction_seen = np.zeros_like(read_onset)
    evicted = np.zeros(n, dtype=bool)
    match_confidence = np.full((n, steps), warm_model.INVALID_CONFIDENCE)
    counts = np.zeros_like(o)
    cache = []
    log = _new_log(n, steps)
    timing = _new_timing()

    for t in range(steps):
        frame = x[:, t]
        query = frame[:, 2:4]
        write_gate = frame[:, 5:6]
        writers = write_gate[:, 0] == 1.
        read_gate = (1. - write_gate) * np.minimum(1., np.linalg.norm(query, axis=1, keepdims=True))
        active_read = read_gate[:, 0] > 0.
        onset = active_read & ~previous_read
        read_onset[:, t] = onset
        eviction_seen[:, t] = evicted
        previous_read = active_read
        first = onset & (log['first_read_tick'] < 0)
        log['first_read_tick'][first] = t
        keys_before = key_bank
        values_before = value_bank
        projected = query @ matrix
        dots = (keys_before @ projected[..., None])[..., 0]
        reset = reset_steps[:, t]
        eligible = first & evicted
        if eligible.any():
            confidence = warm_model.cosine_confidence(projected[eligible], keys_before[eligible], dots[eligible])
            match_confidence[eligible, t] = confidence
            reset[eligible] = confidence >= warm_model.COSINE_THRESHOLD

        archive_cue = t in learned_model.CUE_STEPS
        if archive_cue:
            native_before = np.concatenate((d, v, o, p), axis=1)
        if reset.any():
            start = time.perf_counter()
            replacement, resources, scratch = _replay_prefix(log, reset, t, weights, fanout_r, fanout_o)
            log['replay_steps'][reset] = t
            log['replay_resources'][reset] = resources
            log['replay_reference'][reset] = replacement
            # Copy states retained by the preceding step's trace before replacing rows.
            d, v, o, p = (state.copy() for state in (d, v, o, p))
            for state, component in zip((d, v, o, p), np.split(replacement, (12, 18, 20), axis=1)):
                state[reset] = component
            timing['replay_wall_seconds'] += time.perf_counter()-start
            timing['replay_batches'] += 1
            timing['replay_scratch_peak_bytes'] = max(timing['replay_scratch_peak_bytes'], scratch)
        if archive_cue:
            native_after = np.concatenate((d, v, o, p), axis=1)

        z = frame @ wi + p @ wr
        current = z + db
        drive = np.tanh(z)
        _record(log, frame, writers, t)
        scores = learned_model.TEMPERATURE * dots
        exponentials = np.exp(scores - scores.max(axis=1, keepdims=True))
        attention = exponentials / exponentials.sum(axis=1, keepdims=True)
        context = values_before[..., 0] * attention[:, :1] + values_before[..., 1] * attention[:, 1:]
        injection = learned_model.READ_GAIN * read_gate * context

        value_bank = learned_model.MEMORY_RETENTION * values_before
        if writers.any():
            occupied = np.any(keys_before[writers] != 0., axis=2)
            evicted[writers] |= occupied.all(axis=1)
            value_bank[writers, :, 1] = value_bank[writers, :, 0]
            value_bank[writers, :, 0] = drive[writers]
            key_bank = keys_before.copy()
            key_bank[writers, 1] = keys_before[writers, 0]
            key_bank[writers, 0] = query[writers]

        d = .95 * d + .25 * (current + injection)
        local = np.maximum(d, 0.)
        vc = .85 * v + .5 * local.reshape(n, 6, 2).sum(axis=2) + vb
        oc = .85 * o + p @ wo + ob
        hs, os = (vc > .5).astype(float), (oc > .5).astype(float)
        v, o = vc * (1. - hs), oc * (1. - os)
        if t >= q:
            counts += os
        saved = dict(I=current, A=injection, dc=d, vc=vc, oc=oc, hs=hs, os=os, prev=p,
                     states=np.concatenate((d, v, o), axis=1), key_bank=key_bank, value_bank=value_bank,
                     scores=scores, weights=attention, read_gate=read_gate, write_gate=write_gate, drive=drive)
        if archive_cue:
            saved.update(native_before=native_before, native_after=native_after)
        cache.append(saved)
        p = hs

    raw = dict(read_onset=read_onset, reset_steps=reset_steps, match_confidence=match_confidence,
               eviction_seen=eviction_seen)
    return counts, cache, raw, log, timing


def evaluate(x, w, q, rule, reference=None):
    """Return native traces, replay-only diagnostics, and aggregate replay timing."""
    timing = _new_timing()
    if rule in ('original', 'dense'):
        path = 'original' if rule == 'original' else 'eviction_late_native'
        return warm_model.evaluate(x, w, 'LEARN', q, path, reference), {}, timing
    if rule != 'replay' or reference is not None:
        raise ValueError('replay accepts only the recipient input and frozen weights')
    if x.shape[1:] != (36, 6) or len(x) == 0:
        raise ValueError('the replay study requires nonempty 36-step inputs')

    matrix = w[learned_model.NATIVE_PARAMETERS:].reshape(2, 2)
    weights = population.unpack(w[:learned_model.NATIVE_PARAMETERS], 6)
    fanout_r = np.count_nonzero(weights[1], axis=1)
    fanout_o = np.count_nonzero(weights[4], axis=1)
    fields = ('counts', 'resources', *warm_model.CUE_FIELDS, 'read_onset', 'reset_steps',
              'match_confidence', 'eviction_seen')
    chunks = {name: [] for name in fields}
    logs = {name: [] for name in _new_log(0, x.shape[1])}
    for start in range(0, len(x), 32):
        batch = x[start:start+32]
        counts, cache, raw, log, batch_timing = _forward(batch, weights, matrix, q, fanout_r, fanout_o)
        chunks['counts'].append(counts)
        chunks['resources'].append(warm_model._resources(cache, q, fanout_r, fanout_o))
        for name, field in warm_model.CUE_FIELDS.items():
            chunks[name].append(np.stack([cache[t][field] for t in learned_model.CUE_STEPS], axis=1))
        for name, values in raw.items():
            chunks[name].append(values)
        for name, values in log.items():
            logs[name].append(values)
        timing['replay_wall_seconds'] += batch_timing['replay_wall_seconds']
        timing['replay_batches'] += batch_timing['replay_batches']
        timing['replay_scratch_peak_bytes'] = max(timing['replay_scratch_peak_bytes'],
                                                  batch_timing['replay_scratch_peak_bytes'])
        if start == 0:
            trace = warm_model._trace(cache)
    trace.update({name: np.concatenate(values) for name, values in chunks.items()})
    return trace, {name: np.concatenate(values) for name, values in logs.items()}, timing
