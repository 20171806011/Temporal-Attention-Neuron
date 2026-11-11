"""Independent late-cue construction, duplicate pairs and information ceilings."""

from itertools import product

import numpy as np


DELAYS = (0, 16, 32)
CONDITIONS = ('nominal', 'no_cue', 'no_candidates', 'no_first', 'no_second')


def make_base(n, seed):
    assert n > 0 and n % 8 == 0
    k = n // 8
    rng = np.random.default_rng(seed)
    amplitude = rng.uniform(.8, 1.2, k)
    cue_amplitude = rng.uniform(.8, 1.2, k)
    background = rng.uniform(0., .03, (k, 6, 2))
    background[:, (1, 5), :] = 0.
    distractor = rng.uniform(.15, .8, (k, 2))
    permutation = rng.permutation(n)
    rows = np.array(list(product(range(k), (0, 1), (0, 1), (0, 1))))[permutation]
    octet, selected, first, second = rows.T
    label = np.where(selected, second, first)
    return dict(amplitude=amplitude, cue_amplitude=cue_amplitude, background=background,
                distractor=distractor, permutation=permutation, octet=octet, selected=selected,
                first_bit=first, second_bit=second, y=label, seed=np.array(seed))


def assemble(base, delay, condition='nominal'):
    assert delay in DELAYS and condition in CONDITIONS
    q = delay + 12
    x = np.zeros((len(base['y']), q + 8, 6), dtype=np.float64)
    for i, (octet, selected, first, second) in enumerate(zip(
            base['octet'], base['selected'], base['first_bit'], base['second_bit'])):
        x[i, 3:9, :2] = base['background'][octet]
        x[i, 5, 5], x[i, 7, 5] = base['distractor'][octet]
        if condition not in ('no_candidates', 'no_first'):
            x[i, 4, first] = base['amplitude'][octet]
        if condition not in ('no_candidates', 'no_second'):
            x[i, 8, second] = base['amplitude'][octet]
        if condition != 'no_cue':
            x[i, q-3, 2+selected] = base['cue_amplitude'][octet]
        x[i, q:, 4] = 1.
    return x, q


def lookup(base):
    return {key: i for i, key in enumerate(zip(base['octet'], base['selected'],
                                               base['first_bit'], base['second_bit']))}


def pairs(base, condition):
    """Disjoint label-0/label-1 pairs on the affected subset of each control."""
    assert condition in CONDITIONS[1:]
    rows = lookup(base)
    result = []
    for (octet, selected, first, second), index in rows.items():
        if base['y'][index] != 0:
            continue
        if condition == 'no_cue':
            if first == second:
                continue
            partner = rows[(octet, 1-selected, first, second)]
        elif selected == 0 and condition in ('no_candidates', 'no_first'):
            partner = rows[(octet, selected, 1-first, second)]
        elif selected == 1 and condition in ('no_candidates', 'no_second'):
            partner = rows[(octet, selected, first, 1-second)]
        else:
            continue
        assert base['y'][partner] == 1
        result.append((index, partner))
    return np.asarray(result, dtype=np.int64).reshape(-1, 2)


def ceiling(x, y):
    """Best deterministic classifier on exactly identical observed inputs."""
    groups = {}
    for sample, label in zip(x, y):
        counts = groups.setdefault(sample.tobytes(), [0, 0])
        counts[label] += 1
    return sum(map(max, groups.values())) / len(y)


def check(base, x, delay, condition='nominal'):
    expected, q = assemble(base, delay, condition)
    assert np.array_equal(x, expected), 'stimulus differs from independent construction'
    n = len(base['y'])
    assert n % 8 == 0 and np.all(np.bincount(base['y'], minlength=2) == n//2)
    rows = lookup(base)
    assert len(rows) == n
    assert set(rows) == set(product(range(n//8), (0, 1), (0, 1), (0, 1)))
    assert np.array_equal(base['y'], np.where(base['selected'], base['second_bit'], base['first_bit']))
    assert not np.any(x[:, 9:9+delay]), 'delay includes nonzero external input'
    assert np.all(x[:, q:, 4] == 1.) and not np.any(x[:, :q, 4])
    disagreements = base['first_bit'] != base['second_bit']
    if condition == 'nominal':
        # Switching selection cannot change the prefix before the cue. Flipping
        # the selected content leaves all post-candidate inputs identical.
        for (octet, selected, first, second), i in rows.items():
            cue_partner = rows[(octet, 1-selected, first, second)]
            assert np.array_equal(x[i, :q-3], x[cue_partner, :q-3])
            if selected == 0:
                bit_partner = rows[(octet, selected, 1-first, second)]
            else:
                bit_partner = rows[(octet, selected, first, 1-second)]
            assert base['y'][bit_partner] == 1-base['y'][i]
            assert np.array_equal(x[i, 9:], x[bit_partner, 9:])
        bounds = [ceiling(x[:, :end], base['y']) for end in (4, 5, q-3, q-2)]
        assert bounds == [.5, .75, .75, 1.], 'wrong information release schedule'
        assert ceiling(x[disagreements, :q-3], base['y'][disagreements]) == .5
        return dict(balanced_labels=True, prefix_ceilings=bounds,
                    disagree_pre_cue_ceiling=.5, cue_and_content_required=True)
    paired = pairs(base, condition)
    assert len(paired) == (n//2 if condition == 'no_candidates' else n//4)
    assert len(np.unique(paired)) == paired.size
    assert np.array_equal(x[paired[:, 0]], x[paired[:, 1]])
    bound = ceiling(x, base['y'])
    assert bound == (.5 if condition == 'no_candidates' else .75)
    return dict(balanced_labels=True, opposite_label_pairs=len(paired), accuracy_ceiling=bound)
