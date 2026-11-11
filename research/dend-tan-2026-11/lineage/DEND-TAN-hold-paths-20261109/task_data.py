"""Two independent candidates with an independently chosen cue after a blank."""

import numpy as np


CONDITIONS = ('nominal', 'no_cue', 'no_candidates', 'no_first', 'no_second')


def make_base(n, seed):
    assert n > 0 and n % 8 == 0
    groups = n//8
    rng = np.random.default_rng(seed)
    amplitude = rng.uniform(.8, 1.2, groups)
    cue_amplitude = rng.uniform(.8, 1.2, groups)
    background = rng.uniform(0, .03, (groups, 6, 2))
    background[:, [1, 5]] = 0
    distractor = rng.uniform(.15, .8, (groups, 2))
    permutation = rng.permutation(n)
    octet = np.repeat(np.arange(groups), 8)[permutation]
    selected = np.tile(np.repeat([0, 1], 4), groups)[permutation]
    first_bit = np.tile([0, 0, 1, 1]*2, groups)[permutation]
    second_bit = np.tile([0, 1, 0, 1]*2, groups)[permutation]
    y = np.where(selected == 0, first_bit, second_bit)
    return dict(amplitude=amplitude, cue_amplitude=cue_amplitude, background=background,
                distractor=distractor, permutation=permutation, octet=octet, selected=selected,
                first_bit=first_bit, second_bit=second_bit, y=y, seed=np.array(seed))


def assemble(base, delay, condition='nominal'):
    assert delay in (0, 16, 32) and condition in CONDITIONS
    groups = base['octet']
    rows = np.arange(len(groups))
    cue, query = 9+delay, 12+delay
    x = np.zeros((len(groups), query+8, 6))
    x[:, 3:9, :2] = base['background'][groups]
    x[:, [5, 7], 5] = base['distractor'][groups]
    if condition not in ('no_candidates', 'no_first'):
        x[rows, 4, base['first_bit']] = base['amplitude'][groups]
    if condition not in ('no_candidates', 'no_second'):
        x[rows, 8, base['second_bit']] = base['amplitude'][groups]
    if condition != 'no_cue':
        x[rows, cue, 2+base['selected']] = base['cue_amplitude'][groups]
    x[:, query:, 4] = 1
    return x, query
