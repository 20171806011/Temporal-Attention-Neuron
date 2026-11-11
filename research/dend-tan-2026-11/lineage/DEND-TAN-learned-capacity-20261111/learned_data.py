"""Rotated continuous queries for two records written in either order."""

import numpy as np


CONDITIONS = ('nominal', 'no_cue', 'no_tags', 'no_contents')


def make_base(n, seed):
    assert n > 0 and n % 16 == 0
    groups = n//16
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-np.pi, np.pi, groups)
    gap = rng.uniform(np.pi/2, 3*np.pi/2, groups)
    query_noise = rng.uniform(-.12, .12, (groups, 2, 2))
    payload_amplitude = rng.uniform(.8, 1.2, (groups, 2))
    distractor = rng.uniform(.10, .45, (groups, 2))
    permutation = rng.permutation(n)
    phase = np.column_stack((angle, angle+gap))
    key_vectors = np.stack((np.cos(phase), np.sin(phase)), axis=-1)
    query_vectors = np.stack((-key_vectors[..., 1], key_vectors[..., 0]), axis=-1)+query_noise
    index = permutation % 16
    group, order = permutation//16, index//8
    selected, first, second = (index//4) % 2, (index//2) % 2, index % 2
    y = np.where(selected == 0, first, second)
    return dict(angle=angle, gap=gap, query_noise=query_noise, payload_amplitude=payload_amplitude,
                distractor=distractor, key_vectors=key_vectors, query_vectors=query_vectors,
                permutation=permutation, group=group, order=order, octet=2*group+order,
                selected=selected, first_bit=first, second_bit=second, y=y, seed=np.array(seed))


def assemble(base, condition='nominal'):
    assert condition in CONDITIONS
    group = base['group']
    rows = np.arange(len(group))
    x = np.zeros((len(rows), 36, 6))
    for position, tick in enumerate((4, 8)):
        record = base['order'] if position == 0 else 1-base['order']
        content = np.where(record == 0, base['first_bit'], base['second_bit'])
        x[:, tick, 5] = 1.
        if condition != 'no_contents':
            x[rows, tick, content] = base['payload_amplitude'][group, position]
        if condition != 'no_tags':
            x[:, tick, 2:4] = base['key_vectors'][group, record]
    x[:, [5, 7], 4] = base['distractor'][group]
    if condition != 'no_cue':
        x[:, 25:28, 2:4] = base['query_vectors'][group, base['selected']][:, None]
    x[:, 28:, 4] = 1.
    return x, 28
