"""Collect native states, splice selected fields, and resume native dynamics."""

import numpy as np

import model
import population
from core import ATTENTION_ARMS, TAPS


def add_resources(resources, hidden, outputs, previous, recurrent_fanout, output_fanout):
    resources[:, 0] += hidden.sum(1)
    resources[:, 1] += outputs.sum(1)
    resources[:, 2] += previous @ recurrent_fanout
    resources[:, 3] += previous @ output_fanout


def collect(x, w, arm, taps=TAPS):
    _, wr, _, _, wo, _ = population.unpack(w, 6)
    recurrent_fanout = np.count_nonzero(wr, axis=1)
    output_fanout = np.count_nonzero(wo, axis=1)
    chunks = []
    for start in range(0, len(x), 32):
        batch = x[start:start+32]
        counts, cache = model.forward(batch, w, arm, 16)
        raster = np.stack([state['os'] for state in cache[16:]], axis=1).astype(np.uint8)
        assert np.array_equal(counts, raster.sum(1))
        resources = np.zeros((len(batch), 4))
        for state in cache:
            add_resources(resources, state['hs'], state['os'], state['prev'], recurrent_fanout, output_fanout)
        record = dict(raster=raster, counts=counts, resources=resources)
        if taps:
            states = np.stack([cache[t]['states'] for t in taps], axis=1)
            record.update(d=states[:, :, :12], v=states[:, :, 12:18], o=states[:, :, 18:],
                          p=np.stack([cache[t]['hs'] for t in taps], axis=1))
            if arm in ATTENTION_ARMS:
                record['H'] = np.stack([cache[t]['attention']['H'] for t in taps], axis=1)
        chunks.append(record)
    return {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}


def splice(base, tap, group, donor_index):
    fields = ('d', 'v', 'p', 'o') + (('H',) if 'H' in base else ())
    chosen = {'full': fields, 'hidden': tuple(key for key in fields if key != 'o'),
              'branch': ('d',), 'soma_p': ('v', 'p'), 'output': ('o',), 'history': ('H',)}[group]
    assert all(key in fields for key in chosen)
    index = TAPS.index(tap)
    return {key: (base[key][donor_index, index] if key in chosen else base[key][:, index].copy())
            for key in fields}


def resume(x, w, arm, tap, state):
    wi, wr, db, vb, wo, ob = population.unpack(w, 6)
    recurrent_fanout = np.count_nonzero(wr, axis=1)
    output_fanout = np.count_nonzero(wo, axis=1)
    d, v, previous, o = (state[key] for key in ('d', 'v', 'p', 'o'))
    history = state.get('H')
    raster = []
    resources = np.zeros((len(x), 4))
    for t in range(tap+1, 24):
        current = x[:, t] @ wi + previous @ wr + db
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
        if t >= 16:
            raster.append(output.astype(np.uint8))
        add_resources(resources, hidden, output, previous, recurrent_fanout, output_fanout)
        previous = hidden
    return dict(suffix_raster=np.stack(raster, axis=1), suffix_resources=resources)


def joint_features(state):
    return np.column_stack([state[key] for key in ('d', 'v', 'p', 'o')])
