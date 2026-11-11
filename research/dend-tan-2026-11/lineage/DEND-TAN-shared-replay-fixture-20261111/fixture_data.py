"""Frozen factorial inputs and hand-constructed shared-replay edge cases."""

import numpy as np

import capacity_data
import confirm_data
import learned_model
import population


DATA_SEED = 4099903201
WEIGHT_SEED = 4099903202
EXAMPLES = 256
Q = 28
GEOMETRIES = ('orthogonal', 'near')
HISTORIES = ('four', 'three0', 'three2', 'two')


def factorial():
    """Build the paired geometries from a single set of four nuisance groups."""
    base = capacity_data.make_base(EXAMPLES, DATA_SEED)
    phase = base['angle'][:, None] + np.deg2rad([0., 40., 180., 230.])
    keys = np.stack((np.cos(phase), np.sin(phase)), axis=-1)
    queries = np.stack((-keys[..., 1], keys[..., 0]), axis=-1) + base['query_noise']
    geometries = {'orthogonal': base, 'near': base | dict(key_vectors=keys, query_vectors=queries)}
    inputs = {}
    for geometry, observed in geometries.items():
        for history in HISTORIES:
            x, q = confirm_data.assemble(observed, history)
            assert q == Q
            inputs[f'{geometry}_{history}'] = x
    w = np.r_[population.initialize(WEIGHT_SEED, 6), learned_model.ORACLE_MATRIX.ravel()]
    return base, inputs, w


def edges():
    """Return twelve fixed rows and their first-read and replacement ticks."""
    x = np.zeros((12, confirm_data.T, 6))
    east, north, zero, west, diagonal = (1., 0.), (0., 1.), (0., 0.), (-1., 0.), (1., 1.)
    histories = (
        (east, north, north),
        (west, north, east, east),
        (east, zero, north, north),
        (east, east, east),
        (east, north, west, east, north),
        (),
        (diagonal, north, north),
        (east, north, north),
        (east, north, west, east),
        (east, north),
        (west, north, east, east),
        (west, north, east, east),
    )
    for row, keys in enumerate(histories):
        for tick, key in zip((0, 2, 4, 8, 9), keys):
            x[row, tick] = [1., 0., *key, 0., 1.]

    x[0, 6, 2:4] = diagonal
    x[1, 9:11, 2:4] = east
    x[1, 11] = [1., 0., *north, 0., 1.]
    x[1, 12, 2:4] = east
    x[2, 9, 2:4] = north
    x[3, 6, 2:4] = west
    x[3, 10, 2:4] = east
    x[4, 10, 2:4] = north
    x[5, 7, 2:4] = east
    x[6, 6, 2:4] = diagonal
    x[7, 1, 2:4] = east
    x[7, 10, 2:4] = north
    x[8, 1, 4] = .31
    x[9, 6, 2:4] = north

    # The last two rows share their entire causal input through the first read.
    x[10:, 1, 4] = .33
    x[10:, 5, 4] = .27
    x[10:, 9:11, 2:4] = east
    x[10, 11] = [1.2, 0., *west, 0., 1.]
    x[11, 11] = [0., .9, *north, 0., 1.]
    x[10:, 12, 2:4] = east

    w = np.r_[population.initialize(WEIGHT_SEED, 6), np.eye(2).ravel()]
    expectations = dict(
        names=['threshold_tie', 'accept_repeat_and_write', 'zero_key', 'first_read_mismatch', 'eviction_fifo',
               'empty_keys', 'stronger_evicted_key', 'read_before_eviction', 'no_read', 'two_writes',
               'future_write_west', 'future_write_north'],
        descriptions=[
            'Live cosine equals threshold and evicted cosine; old replay starts and the guard vetoes.',
            'Two evictions precede an accepted read; a later write and read edge do not replace the first replay.',
            'A zero key participates in a write but cannot fill a valid-key slot.',
            'First read misses; a later matching read remains ineligible.',
            'Three evictions leave only the two newest lost keys; a perfect tie vetoes.',
            'A read with no writes has empty live and lost key banks.',
            'Live cosine equals threshold but the evicted diagonal key matches more strongly.',
            'First read precedes any eviction; a later matching edge cannot start replay.',
            'Evictions and a non-write drive occur, but there is no read.',
            'Two valid writes and a matching read produce no eviction or replay.',
            'Early non-write drive and accepted replay precede a west-key write and later read.',
            'The same first-read prefix precedes a different north-key write and later read.',
        ],
        first_read_tick=[6, 9, 9, 6, 10, 7, 6, 1, -1, 6, 9, 9],
        old_trigger_tick=[6, 9, -1, -1, 10, -1, 6, -1, -1, -1, 9, 9],
        guarded_trigger_tick=[-1, 9, -1, -1, -1, -1, -1, -1, -1, -1, 9, 9],
        future_pair=dict(rows=[10, 11], shared_through_tick=10, changed_write_tick=11),
    )
    return x, w, expectations
