"""Evaluate three read policies by sharing each recipient's common trajectory."""

import numpy as np

import event_model
import guard_model
import learned_model
import population
import warm_model


DYNAMIC_TRACES = {'states': 'states', 'currents': 'I', 'injections': 'A',
                  'value_bank': 'value_bank', 'drive': 'drive'}
DYNAMIC_CUES = {'cue_native_before': 'native_before', 'cue_native_after': 'native_after',
                'cue_read_vectors': 'A'}
REPLAY_EFFECTS = ('replay_steps', 'replay_resources', 'replay_reference')


def _advance(frame, read_gate, attention, writers, weights, state, values):
    wi, wr, db, vb, wo, ob = weights
    d, v, o, p = state
    z = frame @ wi + p @ wr
    current = z + db
    drive = np.tanh(z)
    context = values[..., 0] * attention[:, :1] + values[..., 1] * attention[:, 1:]
    injection = learned_model.READ_GAIN * read_gate * context
    value_bank = learned_model.MEMORY_RETENTION * values
    if writers.any():
        value_bank[writers, :, 1] = value_bank[writers, :, 0]
        value_bank[writers, :, 0] = drive[writers]
    d = .95 * d + .25 * (current + injection)
    local = np.maximum(d, 0.)
    vc = .85 * v + .5 * local.reshape(len(frame), 6, 2).sum(axis=2) + vb
    oc = .85 * o + p @ wo + ob
    hs, os = (vc > .5).astype(float), (oc > .5).astype(float)
    v, o = vc * (1. - hs), oc * (1. - os)
    step = dict(I=current, A=injection, dc=d, vc=vc, oc=oc, hs=hs, os=os, prev=p,
                states=np.concatenate((d, v, o), axis=1), value_bank=value_bank, drive=drive)
    return (d, v, o, hs), value_bank, step


def _archive(main, branches, q, fanout_r, fanout_o):
    original = warm_model._trace(main)
    original['counts'] = np.zeros_like(main[0]['os'])
    for tick, step in enumerate(main):
        if tick >= q:
            original['counts'] += step['os']
    original['resources'] = warm_model._resources(main, q, fanout_r, fanout_o)
    for name, field in warm_model.CUE_FIELDS.items():
        original[name] = np.stack([main[t][field] for t in learned_model.CUE_STEPS], axis=1)

    replay = original.copy()
    if not any(branch is not None for branch in branches):
        return original, replay
    for name in ('counts', 'resources', 'spikes', 'candidates', *DYNAMIC_TRACES, *DYNAMIC_CUES):
        replay[name] = original[name].copy()
    cue_indices = {tick: slot for slot, tick in enumerate(learned_model.CUE_STEPS)}
    for tick, branch in enumerate(branches):
        if branch is None:
            continue
        rows, step = branch
        baseline = main[tick]
        hidden = (step['hs'] - baseline['hs'][rows]).sum(axis=1)
        output = step['os'] - baseline['os'][rows]
        emitted = output.sum(axis=1)
        delivered = step['prev'] - baseline['prev'][rows]
        if tick >= q:
            replay['counts'][rows] += output
        replay['resources'][rows, 0] += hidden
        replay['resources'][rows, 1] += emitted
        replay['resources'][rows, 2] += delivered @ fanout_r
        replay['resources'][rows, 3] += delivered @ fanout_o
        if 16 <= tick < q:
            replay['resources'][rows, 4] += hidden
            replay['resources'][rows, 5] += emitted

        trace = rows < 8
        indices = rows[trace]
        for name, field in DYNAMIC_TRACES.items():
            replay[name][indices, tick] = step[field][trace]
        replay['spikes'][indices, tick] = np.concatenate((step['hs'][trace], step['os'][trace]), axis=1)
        replay['candidates'][indices, tick] = np.concatenate((step['dc'][trace], step['vc'][trace],
                                                              step['oc'][trace]), axis=1)
        if tick in cue_indices:
            cue = cue_indices[tick]
            for name, field in DYNAMIC_CUES.items():
                replay[name][rows, cue] = step[field]
    return original, replay


def _select(original, replay, accepted):
    guarded = original.copy()
    if accepted.any():
        for name, value in original.items():
            if value is not replay[name]:
                selected = accepted[:len(value)]
                guarded[name] = value.copy()
                guarded[name][selected] = replay[name][selected]
    return guarded


def evaluate(x, w, q):
    """Return flat archives for original, replay and guarded plus actual sample steps."""
    if x.shape[1:] != (36, 6) or not 1 <= len(x) <= 32:
        raise ValueError('shared replay requires a packet of 1 to 32 episodes with 36 steps')
    n, steps, _ = x.shape
    matrix = w[learned_model.NATIVE_PARAMETERS:].reshape(2, 2)
    weights = population.unpack(w[:learned_model.NATIVE_PARAMETERS], 6)
    fanout_r = np.count_nonzero(weights[1], axis=1)
    fanout_o = np.count_nonzero(weights[4], axis=1)
    native = tuple(np.zeros((n, size)) for size in (12, 6, 2, 6))
    values = np.zeros((n, 12, 2))
    keys = np.zeros((n, 2, 2))
    alternative = tuple(np.zeros_like(component) for component in native)
    alternate_values = np.zeros_like(values)
    active = np.zeros(n, dtype=bool)
    accepted = np.zeros(n, dtype=bool)
    previous_read = np.zeros(n, dtype=bool)
    evicted = np.zeros(n, dtype=bool)
    read_onset = np.zeros((n, steps), dtype=bool)
    eviction_seen = np.zeros_like(read_onset)
    legacy_reset = np.zeros_like(read_onset)
    match_confidence = np.full((n, steps), warm_model.INVALID_CONFIDENCE)
    log = event_model._new_log(n, steps)
    evicted_keys = np.zeros_like(keys)
    evicted_cosine = np.full_like(match_confidence, warm_model.INVALID_CONFIDENCE)
    evicted_comparisons = np.zeros((n, steps), dtype=np.int64)
    main, branches = [], []
    branch_steps = 0

    for tick in range(steps):
        frame = x[:, tick]
        query = frame[:, 2:4]
        write_gate = frame[:, 5:6]
        writers = write_gate[:, 0] == 1.
        read_gate = (1. - write_gate) * np.minimum(1., np.linalg.norm(query, axis=1, keepdims=True))
        reading = read_gate[:, 0] > 0.
        onset = reading & ~previous_read
        read_onset[:, tick] = onset
        eviction_seen[:, tick] = evicted
        previous_read = reading
        first = onset & (log['first_read_tick'] < 0)
        log['first_read_tick'][first] = tick
        projected = query @ matrix
        dots = (keys @ projected[..., None])[..., 0]
        reset = legacy_reset[:, tick]
        eligible = first & evicted
        if eligible.any():
            confidence, query_norm = guard_model._live_confidence(projected[eligible], keys[eligible], dots[eligible])
            match_confidence[eligible, tick] = confidence
            passed = confidence >= warm_model.COSINE_THRESHOLD
            reset[eligible] = passed
            if reset.any():
                rows = np.flatnonzero(reset)
                lost, comparisons = guard_model.evicted_confidence(projected[rows], evicted_keys[rows],
                                                                  query_norm[passed])
                evicted_cosine[rows, tick] = lost
                evicted_comparisons[rows, tick] = comparisons
                accepted[rows] = confidence[passed] > lost
                reference, resources, _ = event_model._replay_prefix(log, reset, tick, weights, fanout_r, fanout_o)
                log['replay_steps'][rows] = tick
                log['replay_resources'][rows] = resources
                log['replay_reference'][rows] = reference
                for component, replacement in zip(alternative, np.split(reference, (12, 18, 20), axis=1)):
                    component[rows] = replacement
                alternate_values[rows] = values[rows]
                active[rows] = True

        event_model._record(log, frame, writers, tick)
        scores = learned_model.TEMPERATURE * dots
        exponentials = np.exp(scores - scores.max(axis=1, keepdims=True))
        attention = exponentials / exponentials.sum(axis=1, keepdims=True)
        if writers.any():
            dropped = writers & (log['first_read_tick'] < 0) & np.any(keys[:, 1] != 0., axis=1)
            evicted_keys[dropped, 0] = evicted_keys[dropped, 1]
            evicted_keys[dropped, 1] = keys[dropped, 1]
            occupied = np.any(keys[writers] != 0., axis=2)
            evicted[writers] |= occupied.all(axis=1)
            updated = keys.copy()
            updated[writers, 1] = keys[writers, 0]
            updated[writers, 0] = query[writers]
            keys = updated

        cue = tick in learned_model.CUE_STEPS
        if cue:
            before = np.concatenate(native, axis=1)
        native, values, step = _advance(frame, read_gate, attention, writers, weights, native, values)
        step.update(key_bank=keys, scores=scores, weights=attention, read_gate=read_gate, write_gate=write_gate)
        if cue:
            step.update(native_before=before, native_after=before)
        main.append(step)

        rows = np.flatnonzero(active)
        branch = None
        if len(rows):
            branch_steps += len(rows)
            local = tuple(component[rows] for component in alternative)
            if cue:
                after = np.concatenate(local, axis=1)
                branch_before = after.copy()
                new = reset[rows]
                branch_before[new] = before[rows[new]]
            local, bank, local_step = _advance(frame[rows], read_gate[rows], attention[rows], writers[rows],
                                               weights, local, alternate_values[rows])
            for component, updated in zip(alternative, local):
                component[rows] = updated
            alternate_values[rows] = bank
            if cue:
                local_step.update(native_before=branch_before, native_after=after)
            branch = rows, local_step
        branches.append(branch)

    original, replay = _archive(main, branches, q, fanout_r, fanout_o)
    guarded = _select(original, replay, accepted)
    original.update(read_onset=read_onset, eviction_seen=eviction_seen, reset_steps=np.zeros_like(legacy_reset),
                    match_confidence=np.full_like(match_confidence, warm_model.INVALID_CONFIDENCE))
    replay.update(read_onset=read_onset, eviction_seen=eviction_seen, reset_steps=legacy_reset,
                  match_confidence=match_confidence, **log)
    guarded.update(read_onset=read_onset, eviction_seen=eviction_seen, reset_steps=legacy_reset & accepted[:, None],
                   match_confidence=match_confidence, **log, evicted_keys=evicted_keys,
                   evicted_cosine=evicted_cosine, evicted_comparisons=evicted_comparisons)
    for name in REPLAY_EFFECTS:
        guarded[name] = replay[name].copy()
        guarded[name][~accepted] = 0
    cases = dict(original=original, replay=replay, guarded=guarded)
    ledger = dict(main_sample_steps=n*steps, branch_sample_steps=branch_steps,
                  replay_sample_steps=int(log['replay_steps'].sum()))
    return cases, ledger
