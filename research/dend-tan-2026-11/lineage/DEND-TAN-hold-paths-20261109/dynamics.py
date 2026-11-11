"""Native trajectory taps and an intervention-only continuation of the same dynamics."""

import numpy as np

import model
import population
from core import ATTENTION_ARMS


def capture(x, theta, arm, q, taps, save_recurrence=False):
    _, wr, _, _, wo, _ = population.unpack(theta, 6)
    fan_r, fan_o = np.count_nonzero(wr, 1), np.count_nonzero(wo, 1)
    chunks = []
    for start in range(0, len(x), 32):
        batch = x[start:start+32]
        counts, cache = model.forward(batch, theta, arm, q)
        raster = np.stack([step['os'] for step in cache[q:]], axis=1).astype(np.uint8)
        assert np.array_equal(counts, raster.sum(1))
        events = np.zeros((len(batch), len(cache), 5))
        for t, step in enumerate(cache):
            events[:, t, 0] = step['hs'].sum(1)
            events[:, t, 1] = step['os'].sum(1)
            events[:, t, 2] = step['prev'] @ fan_r
            events[:, t, 3] = step['prev'] @ fan_o
        states = np.stack([cache[t]['states'] for t in taps], axis=1)
        record = dict(raster=raster, counts=counts, resources=events.sum(1),
                      prefix_resources=np.cumsum(events, axis=1)[:, taps],
                      d=states[:, :, :12], v=states[:, :, 12:18], o=states[:, :, 18:],
                      p=np.stack([cache[t]['hs'] for t in taps], axis=1))
        if arm in ATTENTION_ARMS:
            record['H'] = np.stack([cache[t]['attention']['H'] for t in taps], axis=1)
        if save_recurrence:
            record['recurrence'] = np.stack([step['prev'] @ wr for step in cache], axis=1)
        chunks.append(record)
    return {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}


def state_at(snapshot, taps, tap):
    fields = ('d', 'v', 'p', 'o') + (('H',) if 'H' in snapshot else ())
    return {key: snapshot[key][:, taps.index(tap)] for key in fields}


def splice(snapshot, taps, tap, group, donor_index):
    state = state_at(snapshot, taps, tap)
    chosen = {'branch': ('d',), 'soma': ('v',), 'previous': ('p',), 'soma_p': ('v', 'p'),
              'output': ('o',), 'other': tuple(key for key in state if key != 'd'),
              'full': tuple(state), 'history': ('H',), 'branch_history': ('d', 'H')}[group]
    squared = np.zeros(len(donor_index))
    for key in chosen:
        source = state[key][donor_index]
        delta = (source-state[key]).reshape(len(donor_index), -1)
        squared += np.sum(delta*delta, axis=1)
        state[key] = source
    return state, np.sqrt(squared)


def donors(base):
    columns = (base['octet'], base['selected'], base['first_bit'], base['second_bit'])
    lookup = {key: index for index, key in enumerate(zip(*columns))}
    assert len(lookup) == len(columns[0])
    first = np.array([lookup[(k,s,1-b0,b1)] for k,s,b0,b1 in zip(*columns)])
    second = np.array([lookup[(k,s,b0,1-b1)] for k,s,b0,b1 in zip(*columns)])
    cue = np.array([lookup[(k,1-s,b0,b1)] for k,s,b0,b1 in zip(*columns)])
    return first, second, cue


def mean_recurrence(recurrence, base):
    averaged = np.empty_like(recurrence)
    for group in np.unique(base['octet']):
        for cue in (0, 1):
            mask = (base['octet'] == group) & (base['selected'] == cue)
            assert mask.sum() == 4
            averaged[mask] = recurrence[mask].mean(0)
    return averaged


def resume(x, theta, arm, q, tap, state, recurrence=None, window=None):
    assert tap < q and (recurrence is None) == (window is None)
    wi, wr, db, vb, wo, ob = population.unpack(theta, 6)
    fan_r, fan_o = np.count_nonzero(wr, 1), np.count_nonzero(wo, 1)
    d, v, previous, o = (state[key] for key in ('d', 'v', 'p', 'o'))
    history = state.get('H')
    raster = []
    resources = np.zeros((len(x), 5))
    for t in range(tap+1, x.shape[1]):
        clamped = window is not None and window[0] <= t < window[1]
        if clamped:
            recurrent = recurrence[:, t]
            resources[:, 4] += 12
        else:
            recurrent = previous @ wr
            resources[:, 2] += previous @ fan_r
        current = x[:, t] @ wi + recurrent + db
        if arm in ATTENTION_ARMS:
            injection, history, _ = model.attention(current, history, arm == 'FULL')
        else:
            injection = current if arm == 'GAIN2' else np.zeros_like(current)
        d = .95*d + .25*(current + injection)
        local = np.maximum(d, 0)
        candidate = .85*v + .5*local.reshape(len(x), 6, 2).sum(2) + vb
        hidden = (candidate > .5).astype(float)
        v = candidate * (1-hidden)
        candidate = .85*o + previous @ wo + ob
        output = (candidate > .5).astype(float)
        o = candidate * (1-output)
        if t >= q:
            raster.append(output.astype(np.uint8))
        resources[:, 0] += hidden.sum(1)
        resources[:, 1] += output.sum(1)
        resources[:, 3] += previous @ fan_o
        previous = hidden
    raster = np.stack(raster, axis=1)
    assert raster.shape[1:] == (8, 2)
    return dict(raster=raster, counts=raster.sum(1), resources=resources)
