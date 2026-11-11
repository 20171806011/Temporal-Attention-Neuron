"""Independently enumerate the four-record stimuli and information bounds."""

from collections import defaultdict
from itertools import product

import numpy as np


WRITE_STEPS = (0, 2, 4, 8)
CONDITIONS = ('base2', 'full2', 'full4', 'full4_hard', 'full4_hard_matched')
ORACLE_MARGIN_BOUND = 1-2*np.sqrt(2.)*.12


def make_base(n, seed):
    assert n > 0 and n % 64 == 0
    groups = n//64
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-np.pi, np.pi, groups)
    noise = rng.uniform(-.12, .12, (groups, 4, 2))
    payload = rng.uniform(.8, 1.2, (groups, 4))
    distractor = rng.uniform(.10, .45, (groups, 2))
    write_order = np.stack([rng.permutation(4) for _ in range(groups)])
    permutation = rng.permutation(n)

    phases = angle[:, None]+(np.pi/2)*np.arange(4)
    keys = np.stack((np.cos(phases), np.sin(phases)), axis=-1)
    queries = np.stack((-keys[..., 1], keys[..., 0]), axis=-1)+noise
    rows = np.array(list(product(range(groups), range(4), (0, 1), (0, 1), (0, 1), (0, 1))))[permutation]
    group, selected = rows[:, 0], rows[:, 1]
    bits = rows[:, 2:][:, ::-1].copy()
    y = bits[np.arange(n), selected]
    return dict(angle=angle, query_noise=noise, payload_amplitude=payload, distractor=distractor,
                key_vectors=keys, query_vectors=queries, write_order=write_order, permutation=permutation,
                group=group, selected=selected, bits=bits, y=y, seed=np.array(seed))


def assemble(base, condition='full4'):
    assert condition in CONDITIONS
    x = np.zeros((len(base['y']), 36, 6))
    for i, (group, selected, bits) in enumerate(zip(base['group'], base['selected'], base['bits'])):
        order = base['write_order'][group]
        for position, tick in enumerate(WRITE_STEPS):
            if condition == 'base2' and position < 2:
                continue
            x[i, tick, bits[position]] = base['payload_amplitude'][group, position]
            x[i, tick, 2:4] = base['key_vectors'][group, order[position]]
            x[i, tick, 5] = 1.
        x[i, 5, 4], x[i, 7, 4] = base['distractor'][group]
        x[i, 25:28, 2:4] = base['query_vectors'][group, order[selected]]
        x[i, 28:, 4] = 1.
    return x, 28


def pairs(base):
    rows = {(int(group), int(selected), tuple(bits)): i
            for i, (group, selected, bits) in enumerate(zip(base['group'], base['selected'], base['bits']))}
    matched = []
    for (group, selected, bits), i in rows.items():
        if selected < 2 and bits[selected] == 0:
            other = list(bits)
            other[selected] = 1
            matched.append((i, rows[group, selected, tuple(other)]))
    return np.asarray(matched, dtype=np.int64).reshape(-1, 2)


def ceiling(x, y):
    classes = defaultdict(lambda: [0, 0])
    for stimulus, label in zip(x, y):
        classes[stimulus.tobytes()][label] += 1
    return sum(max(counts) for counts in classes.values())/len(y)


def oracle_margins(base):
    keys, queries = base['key_vectors'], base['query_vectors']
    rotated = np.stack((-keys[..., 1], keys[..., 0]), axis=-1)
    similarity = np.einsum('gsi,gki->gsk', queries, rotated, optimize=False)
    diagonal = np.diagonal(similarity, axis1=1, axis2=2)
    margin = diagonal[..., None]-similarity
    return margin[:, ~np.eye(4, dtype=bool)].reshape(-1, 4, 3)


def check(base, x, condition='full4'):
    y, selected = base['y'], base['selected']
    n = len(y)
    assert x.shape == (n, 36, 6) and not x[:, 9:25].any()
    for position in range(4):
        labels = y[selected == position]
        assert len(labels) == n//4 and np.count_nonzero(labels == 0) == n//8
    upper = ceiling(x, y)
    assert upper == (.75 if condition == 'base2' else 1.)
    count = 0
    if condition == 'base2':
        left, right = pairs(base).T
        count = len(left)
        assert count == n//4
        assert np.array_equal(x[left], x[right]) and np.all(y[left] != y[right])
        assert ceiling(x[selected < 2], y[selected < 2]) == .5
    margin = float(oracle_margins(base).min())
    assert margin >= ORACLE_MARGIN_BOUND-1e-13
    return dict(balanced_labels=True, ceiling=upper, duplicate_opposite_label_pairs=count,
                oracle_margin_min=margin)
