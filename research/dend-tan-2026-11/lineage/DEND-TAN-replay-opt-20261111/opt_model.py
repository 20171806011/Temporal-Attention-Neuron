"""Optimize causal event replay with weight-only zero states and sparse input projection."""

import time

import numpy as np

import event_model
import learned_model
import population
import warm_model


MODES = ('full', 'cached', 'sparse')
HORIZON = 36


def _native_step(frame, weights, state, active=None):
    wi, wr, db, vb, wo, ob = weights
    d, v, o, p = state
    if active is None:
        input_current = frame @ wi
    else:
        input_current = np.zeros_like(d)
        if active.any():
            input_current[active] = frame[active] @ wi
    z = input_current + p @ wr
    current = z + db
    d = .95 * d + .25 * current
    local = np.maximum(d, 0.)
    vc = .85 * v + .5 * local.reshape(len(frame), 6, 2).sum(axis=2) + vb
    oc = .85 * o + p @ wo + ob
    hs, os = (vc > .5).astype(float), (oc > .5).astype(float)
    return (d, vc * (1. - hs), oc * (1. - os), hs), os


def build_cache(w):
    """Record the complete native state after each number of all-zero input steps."""
    weights = population.unpack(w[:learned_model.NATIVE_PARAMETERS], 6)
    table = np.zeros((HORIZON + 1, 26))
    frame = np.zeros((1, 6))
    state = tuple(np.split(table[:1], (12, 18, 20), axis=1))
    for tick in range(1, HORIZON + 1):
        state, _ = _native_step(frame, weights, state)
        table[tick] = np.concatenate(state, axis=1)[0]
    return table


def _new_counts(n):
    return dict(earliest_tick=np.full(n, -1, dtype=np.int64),
                replay_input_rows=np.zeros(n, dtype=np.int64),
                replay_nonzero_steps=np.zeros(n, dtype=np.int64))


def _record_support(log, rows):
    write_ticks = log['logged_write_ticks'][rows]
    first_write = np.where(write_ticks[:, 0] >= 0, write_ticks[:, 0], write_ticks[:, 1])
    first_other = log['logged_nonwrite_ticks'][rows, 0]
    earliest = np.minimum(np.where(first_write >= 0, first_write, HORIZON),
                          np.where(first_other >= 0, first_other, HORIZON))
    if np.any(earliest == HORIZON):
        raise AssertionError('eligible replay must contain a retained nonzero write')
    nonzero = np.count_nonzero(write_ticks >= 0, axis=1) + log['logged_nonwrite_count'][rows]
    log['earliest_tick'][rows] = earliest
    log['replay_nonzero_steps'][rows] = nonzero
    return earliest, nonzero


def _replay_suffix(log, triggered, stop, weights, fanout_r, fanout_o, table, sparse):
    rows = np.flatnonzero(triggered)
    earliest, nonzero = _record_support(log, rows)
    log['replay_steps'][rows] = stop-earliest
    log['replay_input_rows'][rows] = nonzero if sparse else stop-earliest
    replacement = np.empty((len(rows), 26))
    scratch_peak = 0

    for first in np.unique(earliest):
        members = earliest == first
        group = rows[members]
        n = len(group)
        prefix = np.zeros((n, stop-first, 6))
        for name in ('write', 'nonwrite'):
            ticks = log[f'logged_{name}_ticks'][group]
            row, slot = np.nonzero(ticks >= 0)
            prefix[row, ticks[row, slot]-first] = log[f'logged_{name}_values'][group[row], slot]
        scratch_peak = max(scratch_peak, prefix.nbytes)
        initial = np.broadcast_to(table[first], (n, 26))
        state = tuple(np.split(initial, (12, 18, 20), axis=1))
        resources = np.zeros((n, 4))
        for frame in prefix.swapaxes(0, 1):
            prior_spikes = state[3]
            active = np.any(frame != 0., axis=1) if sparse else None
            state, output_spikes = _native_step(frame, weights, state, active)
            resources[:, 0] += state[3].sum(axis=1)
            resources[:, 1] += output_spikes.sum(axis=1)
            resources[:, 2] += prior_spikes @ fanout_r
            resources[:, 3] += prior_spikes @ fanout_o
        replacement[members] = np.concatenate(state, axis=1)
        log['replay_resources'][group] = resources
    return replacement, scratch_peak


def _forward(x, weights, matrix, q, fanout_r, fanout_o, cache_table, sparse):
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
    log = event_model._new_log(n, steps)
    log.update(_new_counts(n))
    timing = event_model._new_timing()

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
            replacement, scratch = _replay_suffix(log, reset, t, weights, fanout_r, fanout_o, cache_table, sparse)
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
        event_model._record(log, frame, writers, t)
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

def evaluate(x, w, q, mode, cache=None):
    """Return native traces, causal log and operation diagnostics, and replay timing."""
    if mode not in MODES:
        raise ValueError(f'unknown replay implementation {mode}')
    if mode == 'full':
        raw, log, timing = event_model.evaluate(x, w, q, 'replay')
        log.update(_new_counts(len(x)))
        triggered = np.flatnonzero(log['replay_steps'] > 0)
        _record_support(log, triggered)
        log['replay_input_rows'][triggered] = log['replay_steps'][triggered]
        return raw, log, timing
    if x.shape[1:] != (HORIZON, 6) or len(x) == 0:
        raise ValueError('the replay study requires nonempty 36-step inputs')
    if cache is None:
        cache = build_cache(w)
    if cache.shape != (HORIZON + 1, 26) or cache.dtype != np.float64 or not np.isfinite(cache).all():
        raise ValueError('expected a finite float64 zero-state table with shape (37, 26)')

    matrix = w[learned_model.NATIVE_PARAMETERS:].reshape(2, 2)
    weights = population.unpack(w[:learned_model.NATIVE_PARAMETERS], 6)
    fanout_r = np.count_nonzero(weights[1], axis=1)
    fanout_o = np.count_nonzero(weights[4], axis=1)
    fields = ('counts', 'resources', *warm_model.CUE_FIELDS, 'read_onset', 'reset_steps',
              'match_confidence', 'eviction_seen')
    chunks = {name: [] for name in fields}
    logs = {name: [] for name in (*event_model._new_log(0, HORIZON), *_new_counts(0))}
    timing = event_model._new_timing()
    for start in range(0, len(x), 32):
        batch = x[start:start+32]
        counts, states, raw, log, batch_timing = _forward(batch, weights, matrix, q, fanout_r, fanout_o,
                                                          cache, mode == 'sparse')
        chunks['counts'].append(counts)
        chunks['resources'].append(warm_model._resources(states, q, fanout_r, fanout_o))
        for name, field in warm_model.CUE_FIELDS.items():
            chunks[name].append(np.stack([states[t][field] for t in learned_model.CUE_STEPS], axis=1))
        for name, values in raw.items():
            chunks[name].append(values)
        for name, values in log.items():
            logs[name].append(values)
        timing['replay_wall_seconds'] += batch_timing['replay_wall_seconds']
        timing['replay_batches'] += batch_timing['replay_batches']
        timing['replay_scratch_peak_bytes'] = max(timing['replay_scratch_peak_bytes'],
                                                  batch_timing['replay_scratch_peak_bytes'])
        if start == 0:
            trace = warm_model._trace(states)
    trace.update({name: np.concatenate(values) for name, values in chunks.items()})
    return trace, {name: np.concatenate(values) for name, values in logs.items()}, timing
