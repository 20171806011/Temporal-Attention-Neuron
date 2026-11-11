"""Independently reconstruct the paired confirmation key geometry."""

import numpy as np


GEOMETRIES = ('orthogonal', 'near')
NEAR_DEGREES = (0., 40., 180., 230.)


def geometry(base, name):
    assert name in GEOMETRIES
    if name == 'orthogonal':
        return base.copy()
    angles = base['angle'][:, None]+np.deg2rad(NEAR_DEGREES)
    keys = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    rotated = np.stack((-keys[..., 1], keys[..., 0]), axis=-1)
    return dict(base, key_vectors=keys, query_vectors=rotated+base['query_noise'])
