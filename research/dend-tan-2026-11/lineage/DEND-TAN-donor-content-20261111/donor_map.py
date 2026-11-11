"""Index the exact same-group counterfactual donors for every recipient."""

import numpy as np


def build(base):
    groups = np.asarray(base['group'])
    selected = np.asarray(base['selected'])
    bits = np.asarray(base['bits'])
    n = len(groups)
    if groups.shape != (n,) or selected.shape != (n,) or bits.shape != (n, 4):
        raise ValueError('donor pairing requires one group, one query position, and four bits per row')
    if not np.all(np.isin(selected, (0, 1, 2, 3))) or not np.all(np.isin(bits, (0, 1))):
        raise ValueError('invalid query position or bit values')

    keys = [(group, int(position), tuple(payload)) for group, position, payload in zip(groups, selected, bits)]
    lookup = {key: row for row, key in enumerate(keys)}
    if not n or n != len({key[0] for key in keys})*64 or len(lookup) != n:
        raise ValueError('each donor group must contain all positions and all bit patterns exactly once')

    def flipped(group, position, payload, bit):
        toggled = payload[:bit] + (1 - payload[bit],) + payload[bit+1:]
        return lookup[group, position, toggled]

    maps = {name: np.empty(n, dtype=np.int64) for name in ('target', 'other', 'deleted')}
    for row, (group, position, payload) in enumerate(keys):
        maps['target'][row] = flipped(group, position, payload, position)
        maps['other'][row] = flipped(group, position, payload, 5-position) if position >= 2 else row
        maps['deleted'][row] = flipped(group, position, payload, 0)
    return dict(own=np.arange(n, dtype=np.int64), **maps)
