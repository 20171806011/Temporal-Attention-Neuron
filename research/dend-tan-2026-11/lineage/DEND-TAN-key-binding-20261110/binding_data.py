"""Matched 16-way binding groups, with independently varied address order."""

import numpy as np


CONDITIONS = ('nominal', 'no_cue', 'no_tags', 'no_contents')


def make_base(n, seed):
    assert n > 0 and n % 16 == 0
    groups = n//16
    rng = np.random.default_rng(seed)
    payload_amplitude = rng.uniform(.8, 1.2, (groups, 2))
    tag_amplitude = rng.uniform(.8, 1.2, (groups, 3))
    distractor = rng.uniform(.10, .45, (groups, 2))
    permutation = rng.permutation(n)
    index = permutation % 16
    group = permutation//16
    order = index//8
    selected = (index//4) % 2
    first = (index//2) % 2
    second = index % 2
    y = np.where(selected == 0, first, second)
    return dict(payload_amplitude=payload_amplitude, tag_amplitude=tag_amplitude, distractor=distractor,
                permutation=permutation, group=group, order=order, octet=2*group+order,
                selected=selected, first_bit=first, second_bit=second, y=y, seed=np.array(seed))


def assemble(base, condition='nominal'):
    assert condition in CONDITIONS
    group = base['group']
    rows = np.arange(len(group))
    x = np.zeros((len(rows), 36, 6))
    for position, tick in enumerate((4, 8)):
        address = base['order'] if position == 0 else 1-base['order']
        content = np.where(address == 0, base['first_bit'], base['second_bit'])
        x[:, tick, 5] = 1.
        if condition != 'no_contents':
            x[rows, tick, content] = base['payload_amplitude'][group, position]
        if condition != 'no_tags':
            x[rows, tick, 2+address] = base['tag_amplitude'][group, position]
    x[:, [5, 7], 4] = base['distractor'][group]
    if condition != 'no_cue':
        for tick in range(25, 28):
            x[rows, tick, 2+base['selected']] = base['tag_amplitude'][group, 2]
    x[:, 28:, 4] = 1.
    return x, 28
