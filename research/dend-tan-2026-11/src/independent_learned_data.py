"""Independent enumeration of the signed-key rotation and deletion task."""

from collections import defaultdict
from itertools import product

import numpy as np


CONDITIONS = ('nominal', 'no_cue', 'no_tags', 'no_contents', 'swap_read')
CEILINGS = dict(nominal=1., no_cue=.75, no_tags=.75, no_contents=.5, swap_read=1.)
ORACLE_MARGIN_BOUND = 1-2*np.sqrt(2.)*.12


def make_base(n, seed):
    assert n > 0 and n % 16 == 0
    groups = n//16
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-np.pi, np.pi, groups)
    gap = rng.uniform(np.pi/2, 3*np.pi/2, groups)
    noise = rng.uniform(-.12, .12, (groups, 2, 2))
    payload = rng.uniform(.8, 1.2, (groups, 2))
    distractor = rng.uniform(.10, .45, (groups, 2))
    permutation = rng.permutation(n)
    phases = np.column_stack((angle, angle+gap))
    keys = np.stack((np.cos(phases), np.sin(phases)), axis=-1)
    queries = np.stack((-keys[..., 1], keys[..., 0]), axis=-1)+noise
    rows = np.array(list(product(range(groups), (0, 1), (0, 1), (0, 1), (0, 1))))[permutation]
    group, order, selected, b0, b1 = rows.T
    return dict(angle=angle, gap=gap, query_noise=noise, payload_amplitude=payload, distractor=distractor,
                key_vectors=keys, query_vectors=queries, permutation=permutation, group=group, order=order,
                octet=2*group+order, selected=selected, first_bit=b0, second_bit=b1,
                y=np.where(selected == 0, b0, b1), seed=np.array(seed))


def row_keys(base):
    return zip(*(base[name] for name in ('group', 'order', 'selected', 'first_bit', 'second_bit')))


def assemble(base, delay=16, condition='nominal'):
    assert delay == 16 and condition in CONDITIONS
    x = np.zeros((len(base['y']), 36, 6))
    for i, (group, order, selected, b0, b1) in enumerate(row_keys(base)):
        bits = (b0, b1)
        for phase, tick in enumerate((4, 8)):
            record = (order+phase) % 2
            x[i, tick, 5] = 1.
            if condition != 'no_contents':
                x[i, tick, bits[record]] = base['payload_amplitude'][group, phase]
            if condition != 'no_tags':
                x[i, tick, 2:4] = base['key_vectors'][group, record]
        x[i, 5, 4], x[i, 7, 4] = base['distractor'][group]
        if condition != 'no_cue':
            x[i, 25:28, 2:4] = base['query_vectors'][group, selected]
        x[i, 28:, 4] = 1.
    return x, 28


def pairs(base, condition):
    rows = {key: i for i, key in enumerate(row_keys(base))}
    matches = []
    for (group, order, selected, b0, b1), i in rows.items():
        if condition == 'no_cue' and b0 != b1:
            counterpart = (group, order, 1-selected, b0, b1)
        elif condition == 'no_tags' and b0 != b1:
            counterpart = (group, 1-order, selected, b1, b0)
        elif condition == 'no_contents':
            counterpart = (group, order, selected, b0 ^ (1-selected), b1 ^ selected)
        else:
            continue
        j = rows[counterpart]
        if i < j:
            matches.append((i, j))
    return matches


def ceiling(x, y):
    classes = defaultdict(lambda: [0, 0])
    for stimulus, label in zip(x, y):
        classes[stimulus.tobytes()][label] += 1
    return sum(max(group) for group in classes.values())/len(y)


def oracle_margins(base):
    keys, query = base['key_vectors'], base['query_vectors']
    difference = keys-keys[:, ::-1]
    return query[..., 1]*difference[..., 0]-query[..., 0]*difference[..., 1]


def check(base, x, delay=16, condition='nominal'):
    y = base['y']
    n = len(y)
    assert delay == 16 and x.shape == (n, 36, 6)
    assert np.count_nonzero(y == 0) == n//2 and not x[:, 9:25].any()
    upper = ceiling(x, y)
    assert upper == CEILINGS[condition], (condition, upper)
    matches = pairs(base, condition)
    assert len(matches) == {'no_cue': n//4, 'no_tags': n//4, 'no_contents': n//2}.get(condition, 0)
    if matches:
        first, second = np.asarray(matches).T
        assert np.array_equal(x[first], x[second]) and np.all(y[first] != y[second])
    margin = float(oracle_margins(base).min())
    assert margin >= ORACLE_MARGIN_BOUND-1e-13
    return dict(balanced_labels=True, ceiling=upper, duplicate_opposite_label_pairs=len(matches),
                oracle_margin_min=margin)
