"""Four writes and four queries with balanced content in every nuisance group."""

import numpy as np


WRITE_STEPS = (0, 2, 4, 8)


def make_base(n, seed):
    assert n > 0 and n % 64 == 0
    groups = n//64
    rng = np.random.default_rng(seed)
    angle = rng.uniform(-np.pi, np.pi, groups)
    query_noise = rng.uniform(-.12, .12, (groups, 4, 2))
    payload_amplitude = rng.uniform(.8, 1.2, (groups, 4))
    distractor = rng.uniform(.10, .45, (groups, 2))
    write_order = np.stack([rng.permutation(4) for _ in range(groups)])
    permutation = rng.permutation(n)
    phase = angle[:, None]+(np.pi/2)*np.arange(4)
    key_vectors = np.stack((np.cos(phase), np.sin(phase)), axis=-1)
    query_vectors = np.stack((-key_vectors[..., 1], key_vectors[..., 0]), axis=-1)+query_noise
    group, index = permutation//64, permutation % 64
    selected = index//16
    bits = ((index % 16)[:, None] >> np.arange(4)) & 1
    y = bits[np.arange(n), selected]
    return dict(angle=angle, query_noise=query_noise, payload_amplitude=payload_amplitude,
                distractor=distractor, write_order=write_order, permutation=permutation,
                key_vectors=key_vectors, query_vectors=query_vectors,
                group=group, selected=selected, bits=bits, y=y, seed=np.array(seed))


def assemble(base, condition='full4'):
    groups, selected = base['group'], base['selected']
    rows = np.arange(len(groups))
    addresses = base['write_order'][groups]
    x = np.zeros((len(rows), 36, 6))
    first_position = 2 if condition == 'base2' else 0
    for position in range(first_position, 4):
        tick = WRITE_STEPS[position]
        x[rows, tick, base['bits'][:, position]] = base['payload_amplitude'][groups, position]
        x[:, tick, 2:4] = base['key_vectors'][groups, addresses[:, position]]
        x[:, tick, 5] = 1.
    x[:, [5, 7], 4] = base['distractor'][groups]
    x[:, 25:28, 2:4] = base['query_vectors'][groups, addresses[rows, selected]][:, None]
    x[:, 28:, 4] = 1.
    return x, 28
