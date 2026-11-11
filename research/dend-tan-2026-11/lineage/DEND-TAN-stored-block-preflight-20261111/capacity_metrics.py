"""Native count accuracy and internal routing on balanced write positions."""

import numpy as np


def score(raw, base):
    counts, selected, y = raw['counts'], base['selected'], base['y']
    rows = np.arange(len(y))
    correct = counts[rows, y] > counts[rows, 1-y]
    answered = counts[:, 0] != counts[:, 1]
    logits, weights = raw['cue_scores'][:, 0], raw['cue_weights'][:, 0]
    k = logits.shape[1]
    target = 3-selected
    available = target < k
    routed = np.zeros(len(y), dtype=bool)
    target_weight = np.zeros(len(y))
    chosen = target[available]
    values = logits[available]
    is_target = np.arange(k) == chosen[:, None]
    competitor = np.max(np.where(is_target, -np.inf, values), axis=1)
    routed[available] = values[np.arange(len(chosen)), chosen] > competitor
    target_weight[available] = weights[rows[available], chosen]
    older, recent = selected < 2, selected >= 2

    def positions(values):
        return [float(values[selected == position].mean()) for position in range(4)]

    return dict(balanced_accuracy=float(correct.mean()), older_ba=float(correct[older].mean()),
                recent_ba=float(correct[recent].mean()), coverage=float(answered.mean()),
                older_coverage=float(answered[older].mean()), recent_coverage=float(answered[recent].mean()),
                routing_accuracy=float(routed.mean()), older_routing_accuracy=float(routed[older].mean()),
                recent_routing_accuracy=float(routed[recent].mean()), target_weight=float(target_weight.mean()),
                older_target_weight=float(target_weight[older].mean()),
                recent_target_weight=float(target_weight[recent].mean()), available_fraction=float(available.mean()),
                ba_by_position=positions(correct), coverage_by_position=positions(answered),
                routing_by_position=positions(routed), target_weight_by_position=positions(target_weight))
