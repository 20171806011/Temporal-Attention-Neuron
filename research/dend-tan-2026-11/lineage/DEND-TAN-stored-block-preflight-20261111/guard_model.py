"""Guard causal event replay using the two latest overwritten keys."""

import time

import numpy as np

import event_model
import learned_model
import population
import warm_model


RULES = ('original', 'replay', 'guarded')


def _new_log(n, steps):
    log = event_model._new_log(n, steps)
    log.update(evicted_keys=np.zeros((n, 2, 2)),
               evicted_cosine=np.full((n, steps), warm_model.INVALID_CONFIDENCE),
               evicted_comparisons=np.zeros((n, steps), dtype=np.int64))
    return log


def _new_timing():
    return dict(**event_model._new_timing(), guard_wall_seconds=0., guard_batches=0)


def _live_confidence(projected, keys, dots):
    # Match the original arithmetic while retaining the shared query norm.
    query_norm = np.linalg.norm(projected, axis=1, keepdims=True)
    denominators = query_norm * np.linalg.norm(keys, axis=2)
    cosines = np.divide(dots, denominators, out=np.full_like(denominators, warm_model.INVALID_CONFIDENCE),
                        where=denominators > 0.)
    return cosines.max(axis=1), query_norm[:, 0]


def evicted_confidence(projected, keys, query_norm=None):
    """Compare valid logged keys only and return the best cosine and comparison count."""
    if query_norm is None:
        query_norm = np.linalg.norm(projected, axis=1)
    best = np.full(len(projected), warm_model.INVALID_CONFIDENCE)
    counts = np.zeros(len(projected), dtype=np.int64)
    rows, slots = np.nonzero((query_norm[:, None] > 0.) & np.any(keys != 0., axis=2))
    candidates = keys[rows, slots]
    denominators = query_norm[rows] * np.linalg.norm(candidates, axis=1)
    valid = denominators > 0.
    rows = rows[valid]
    candidates = candidates[valid]
    dots = (candidates[:, None, :] @ projected[rows, :, None])[:, 0, 0]
    cosines = dots / denominators[valid]
    np.maximum.at(best, rows, cosines)
    np.add.at(counts, rows, 1)
    return best, counts


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
            confidence, query_norm = _live_confidence(projected[eligible], keys_before[eligible], dots[eligible])
            match_confidence[eligible, t] = confidence
            passed = confidence >= warm_model.COSINE_THRESHOLD
            reset[eligible] = passed
            if reset.any():
                guard_start = time.perf_counter()
                rows = np.flatnonzero(reset)
                lost_cosine, comparisons = evicted_confidence(projected[rows], log['evicted_keys'][rows], query_norm[passed])
                log['evicted_cosine'][rows, t] = lost_cosine
                log['evicted_comparisons'][rows, t] = comparisons
                reset[rows] = confidence[passed] > lost_cosine
                timing['guard_wall_seconds'] += time.perf_counter()-guard_start
                timing['guard_batches'] += 1

        archive_cue = t in learned_model.CUE_STEPS
        if archive_cue:
            native_before = np.concatenate((d, v, o, p), axis=1)
        if reset.any():
            start = time.perf_counter()
            replacement, resources, scratch = event_model._replay_prefix(log, reset, t, weights, fanout_r, fanout_o)
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
        event_model._record(log, frame, writers, t)
        scores = learned_model.TEMPERATURE * dots
        exponentials = np.exp(scores - scores.max(axis=1, keepdims=True))
        attention = exponentials / exponentials.sum(axis=1, keepdims=True)
        context = values_before[..., 0] * attention[:, :1] + values_before[..., 1] * attention[:, 1:]
        injection = learned_model.READ_GAIN * read_gate * context

        value_bank = learned_model.MEMORY_RETENTION * values_before
        if writers.any():
            dropped = writers & (log['first_read_tick'] < 0) & np.any(keys_before[:, 1] != 0., axis=1)
            log['evicted_keys'][dropped, 0] = log['evicted_keys'][dropped, 1]
            log['evicted_keys'][dropped, 1] = keys_before[dropped, 1]
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

def evaluate(x, w, q, rule):
    """Return online traces, replay/guard diagnostics, and segment timings."""
    if rule not in RULES:
        raise ValueError(f'unknown evicted-key rule {rule}')
    if rule != 'guarded':
        raw, log, timing = event_model.evaluate(x, w, q, rule)
        return raw, log, dict(**timing, guard_wall_seconds=0., guard_batches=0)
    if x.shape[1:] != (36, 6) or len(x) == 0:
        raise ValueError('the evicted-key study requires nonempty 36-step inputs')

    matrix = w[learned_model.NATIVE_PARAMETERS:].reshape(2, 2)
    weights = population.unpack(w[:learned_model.NATIVE_PARAMETERS], 6)
    fanout_r = np.count_nonzero(weights[1], axis=1)
    fanout_o = np.count_nonzero(weights[4], axis=1)
    fields = ('counts', 'resources', *warm_model.CUE_FIELDS, 'read_onset', 'reset_steps',
              'match_confidence', 'eviction_seen')
    chunks = {name: [] for name in fields}
    logs = {name: [] for name in _new_log(0, x.shape[1])}
    timing = _new_timing()
    for start in range(0, len(x), 32):
        batch = x[start:start+32]
        counts, states, raw, log, batch_timing = _forward(batch, weights, matrix, q, fanout_r, fanout_o)
        chunks['counts'].append(counts)
        chunks['resources'].append(warm_model._resources(states, q, fanout_r, fanout_o))
        for name, field in warm_model.CUE_FIELDS.items():
            chunks[name].append(np.stack([states[t][field] for t in learned_model.CUE_STEPS], axis=1))
        for name, values in raw.items():
            chunks[name].append(values)
        for name, values in log.items():
            logs[name].append(values)
        for name in ('replay_wall_seconds', 'replay_batches', 'guard_wall_seconds', 'guard_batches'):
            timing[name] += batch_timing[name]
        timing['replay_scratch_peak_bytes'] = max(timing['replay_scratch_peak_bytes'],
                                                  batch_timing['replay_scratch_peak_bytes'])
        if start == 0:
            trace = warm_model._trace(states)
    trace.update({name: np.concatenate(values) for name, values in chunks.items()})
    return trace, {name: np.concatenate(values) for name, values in logs.items()}, timing
