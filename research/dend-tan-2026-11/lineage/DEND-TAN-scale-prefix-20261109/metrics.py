"""Native paired-cue accuracy, coverage, and information-deletion bounds."""

import numpy as np


def score(counts, base):
    y = base['y']
    tied = counts[:, 0] == counts[:, 1]
    prediction = np.where(tied, -1, counts.argmax(1))
    correct = prediction == y
    silence = np.all(counts == 0, axis=1)
    saturated = np.all(counts == 8, axis=1)
    disagree = base['first_bit'] != base['second_bit']
    first = base['selected'] == 0
    recalls = [float(correct[y == label].mean()) for label in (0, 1)]

    def subset(mask):
        return float(np.mean([correct[mask & (y == label)].mean() for label in (0, 1)])), float((~tied[mask]).mean())

    agree_ba, agree_coverage = subset(~disagree)
    disagree_ba, disagree_coverage = subset(disagree)
    first_ba, first_coverage = subset(first)
    second_ba, second_coverage = subset(~first)
    joints = []
    for group in np.unique(base['octet']):
        members = (base['octet'] == group) & disagree
        for bit in (0, 1):
            matched = members & (base['first_bit'] == bit)
            assert matched.sum() == 2
            joints.append(np.all(correct[matched]))
    return dict(balanced_accuracy=sum(recalls)/2, accuracy=float(correct.mean()), coverage=float((~tied).mean()),
                silence=float(silence.mean()), saturated_tie=float(saturated.mean()),
                intermediate_tie=float((tied & ~silence & ~saturated).mean()), recalls=recalls,
                prediction_counts=[int(np.sum(prediction == label)) for label in (-1, 0, 1)],
                agree_ba=agree_ba, disagree_ba=disagree_ba, agree_coverage=agree_coverage,
                disagree_coverage=disagree_coverage, select_first_ba=first_ba, select_second_ba=second_ba,
                select_first_coverage=first_coverage, select_second_coverage=second_coverage,
                disagree_joint_correct=float(np.mean(joints)))


def check_control(condition, values):
    relation = {'no_cue': ('disagree_ba', 'disagree_coverage'),
                'no_candidates': ('balanced_accuracy', 'coverage'),
                'no_first': ('select_first_ba', 'select_first_coverage'),
                'no_second': ('select_second_ba', 'select_second_coverage')}
    if condition in relation:
        ba, coverage = relation[condition]
        assert abs(values[ba] - .5*values[coverage]) < 1e-12, condition
        assert values['balanced_accuracy'] <= (.5 if condition == 'no_candidates' else .75)
        if condition == 'no_cue':
            assert values['disagree_joint_correct'] == 0
