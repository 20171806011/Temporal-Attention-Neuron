"""Independently reconstruct replay effects and the operations each mode executes."""

import numpy as np

import independent_event as event
from reference import split


MODES = ('full', 'cached', 'sparse')
HISTORIES = event.HISTORIES


def assemble(base, history):
    x, q = event.assemble(base, history)
    moved = base['group'] % 2 != 0
    x[moved, 1] = x[moved, 5]
    x[moved, 5] = 0.
    return x, q


def zero_prefix(w, steps=36):
    """Return all intrinsic zero-input states and cumulative native resources."""
    assert w.shape == (180,) and steps >= 0
    _, wr, db, vb, wo, ob = split(w[:176])
    dend, soma, output, pulse = np.zeros(12), np.zeros(6), np.zeros(2), np.zeros(6)
    table, resources = np.zeros((steps+1, 26)), np.zeros((steps+1, 4))
    fanout_h, fanout_o = np.count_nonzero(wr, axis=1), np.count_nonzero(wo, axis=1)
    for tick in range(1, steps+1):
        dend = .95*dend+.25*(np.einsum('i,ij->j', pulse, wr, optimize=False)+db)
        positive = np.maximum(dend, 0.)
        soma_candidate = .85*soma+.5*(positive[::2]+positive[1::2])+vb
        output_candidate = .85*output+np.einsum('i,ij->j', pulse, wo, optimize=False)+ob
        hidden_spike, output_spike = soma_candidate > .5, output_candidate > .5
        resources[tick] = resources[tick-1]+[hidden_spike.sum(), output_spike.sum(), pulse @ fanout_h, pulse @ fanout_o]
        soma, output = soma_candidate*~hidden_spike, output_candidate*~output_spike
        pulse = hidden_spike.astype(float)
        table[tick] = np.concatenate((dend, soma, output, pulse))
    return table, resources


def cache(w):
    return zero_prefix(w)[0]


def operation_fields(raw, w, mode):
    """Infer the executed work from visible log times and full-prefix resources."""
    assert mode in MODES
    times = np.column_stack((raw['logged_write_ticks'], raw['logged_nonwrite_ticks']))
    executed = raw['replay_steps'] > 0
    onsets = raw['first_read_tick']
    # All active events have unique ticks and occur before the first read.
    present = (times >= 0) & (times < onsets[:, None])
    earliest = np.where(present, times, onsets[:, None]).min(axis=1)
    earliest_tick = np.where(executed, earliest, -1).astype(np.int64)
    nonzero_steps = np.where(executed, present.sum(axis=1), 0).astype(np.int64)
    steps = raw['replay_steps'].copy()
    resources = raw['replay_resources'].copy()
    if mode != 'full':
        steps[executed] -= earliest_tick[executed]
        _, cumulative = zero_prefix(w)
        resources[executed] -= cumulative[earliest_tick[executed]]
    rows = nonzero_steps.copy() if mode == 'sparse' else steps.copy()
    return dict(earliest_tick=earliest_tick, replay_steps=steps, replay_input_rows=rows,
                replay_nonzero_steps=nonzero_steps, replay_resources=resources)


def replay(x, w, q, mode):
    raw = event.replay(x, w, q)
    raw.update(operation_fields(raw, w, mode))
    return raw
