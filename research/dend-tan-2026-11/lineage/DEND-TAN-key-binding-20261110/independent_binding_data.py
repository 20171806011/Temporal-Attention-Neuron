"""Independent enumeration and information checks for randomized address binding."""

from collections import defaultdict
from itertools import product

import numpy as np


CONDITIONS = ('nominal', 'no_cue', 'no_tags', 'no_contents', 'swap_read')
CEILINGS = dict(nominal=1., no_cue=.75, no_tags=.75, no_contents=.5, swap_read=1.)


def make_base(n, seed):
    assert n > 0 and n % 16 == 0
    groups = n//16
    rng = np.random.default_rng(seed)
    payload = rng.uniform(.8, 1.2, (groups, 2))
    tag = rng.uniform(.8, 1.2, (groups, 3))
    distractor = rng.uniform(.10, .45, (groups, 2))
    permutation = rng.permutation(n)
    rows = np.array(list(product(range(groups), (0, 1), (0, 1), (0, 1), (0, 1))))[permutation]
    group, order, selected, b0, b1 = rows.T
    return dict(payload_amplitude=payload, tag_amplitude=tag, distractor=distractor, permutation=permutation,
                group=group, order=order, octet=2*group+order, selected=selected, first_bit=b0, second_bit=b1,
                y=np.where(selected == 0, b0, b1), seed=np.array(seed))


def row_keys(base):
    return zip(*(base[key] for key in ('group', 'order', 'selected', 'first_bit', 'second_bit')))


def assemble(base, delay=16, condition='nominal'):
    assert delay == 16 and condition in CONDITIONS
    x = np.zeros((len(base['y']), 36, 6))
    for index, (group, order, selected, b0, b1) in enumerate(row_keys(base)):
        bits = (b0, b1)
        for phase, tick in enumerate((4, 8)):
            address = (order+phase) % 2
            x[index, tick, 5] = 1.
            if condition != 'no_contents':
                x[index, tick, bits[address]] = base['payload_amplitude'][group, phase]
            if condition != 'no_tags':
                x[index, tick, 2+address] = base['tag_amplitude'][group, phase]
        x[index, 5, 4], x[index, 7, 4] = base['distractor'][group]
        if condition != 'no_cue':
            x[index, 25:28, 2+selected] = base['tag_amplitude'][group, 2]
        x[index, 28:, 4] = 1.
    return x, 28


def pairs(base, condition):
    rows = {key: index for index, key in enumerate(row_keys(base))}
    matches = []
    for (group, order, selected, b0, b1), index in rows.items():
        if condition == 'no_cue' and b0 != b1:
            target = (group, order, 1-selected, b0, b1)
        elif condition == 'no_tags' and b0 != b1:
            target = (group, 1-order, selected, b1, b0)
        elif condition == 'no_contents':
            target = (group, order, selected, b0 ^ (1-selected), b1 ^ selected)
        else:
            continue
        mate = rows[target]
        if index < mate:
            matches.append((index, mate))
    return matches


def ceiling(x, y):
    classes = defaultdict(lambda: [0, 0])
    for stimulus, label in zip(x, y):
        classes[stimulus.tobytes()][label] += 1
    return sum(max(group) for group in classes.values())/len(y)


def check(base, x, delay=16, condition='nominal'):
    y = base['y']
    n = len(y)
    assert delay == 16 and x.shape == (n, 36, 6)
    assert np.count_nonzero(y == 0) == n//2
    assert not x[:, 9:25].any()
    upper = ceiling(x, y)
    assert upper == CEILINGS[condition], (condition, upper)
    matches = pairs(base, condition)
    count = {'no_cue': n//4, 'no_tags': n//4, 'no_contents': n//2}.get(condition, 0)
    assert len(matches) == count
    if matches:
        first, second = np.asarray(matches).T
        assert np.array_equal(x[first], x[second]) and np.all(y[first] != y[second])
    return dict(balanced_labels=True, ceiling=upper, duplicate_opposite_label_pairs=len(matches))
