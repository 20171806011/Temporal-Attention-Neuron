"""Four frozen input schedules for independently fitted replay blocks."""

import numpy as np


T = 36
Q = 28
HISTORIES = {'four': (0, 1, 2, 3), 'three0': (0, 2, 3), 'three2': (1, 2, 3), 'two': (2, 3)}
SCHEDULES = (
    ('A', (0, 3, 6, 11), (2, 8), 22),
    ('B', (1, 4, 7, 13), (0, 10), 24),
    ('C', (0, 2, 5, 10), (1, 8), 20),
    ('D', (2, 5, 8, 12), (3, 6), 23),
)


def assemble(base, history):
    groups, selected = base['group'], base['selected']
    n = len(groups)
    x = np.zeros((n, T, 6))
    addresses = base['write_order'][groups]
    for schedule, (_, writes, nonwrites, first_read) in enumerate(SCHEDULES):
        rows = np.flatnonzero(groups % len(SCHEDULES) == schedule)
        group = groups[rows]
        assert len(rows)*len(SCHEDULES) == n
        for position in HISTORIES[history]:
            tick = writes[position]
            x[rows, tick, base['bits'][rows, position]] = base['payload_amplitude'][group, position]
            x[rows, tick, 2:4] = base['key_vectors'][group, addresses[rows, position]]
            x[rows, tick, 5] = 1.
        x[rows[:, None], nonwrites, 4] = base['distractor'][group]
        query = base['query_vectors'][group, addresses[rows, selected[rows]]]
        x[rows[:, None], np.arange(first_read, first_read+3), 2:4] = query[:, None, :]
    x[:, Q:, 4] = 1.
    return x, Q
