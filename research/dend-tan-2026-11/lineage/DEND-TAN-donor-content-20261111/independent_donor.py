"""Independently index counterfactual donors in a complete factorial block."""

import numpy as np


def build(base):
    """Map each recipient to its unique own, target, other and deleted donor."""
    group = np.asarray(base['group'], dtype=np.int64)
    selected = np.asarray(base['selected'], dtype=np.int64)
    bits = np.asarray(base['bits'], dtype=np.int64)
    n = len(group)
    assert n > 0 and n % 64 == 0 and selected.shape == (n,) and bits.shape == (n, 4)
    assert np.all((0 <= group) & (group < n//64))
    assert np.all((0 <= selected) & (selected < 4)) and np.all((bits == 0) | (bits == 1))

    addresses = group*64+selected*16+bits @ np.array([1, 2, 4, 8], dtype=np.int64)
    rows = np.argsort(addresses).astype(np.int64, copy=False)
    assert np.array_equal(addresses[rows], np.arange(n, dtype=np.int64))
    target_bit = np.left_shift(np.int64(1), selected)
    other_bit = np.zeros(n, dtype=np.int64)
    recent = selected >= 2
    other_bit[recent] = np.left_shift(np.int64(1), 5-selected[recent])
    return dict(own=rows[addresses], target=rows[addresses ^ target_bit],
                other=rows[addresses ^ other_bit], deleted=rows[addresses ^ np.int64(1)])
