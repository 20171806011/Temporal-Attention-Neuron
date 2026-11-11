"""Readout diagnostics on raw native rasters and preserved test prefixes."""

import numpy as np

import probe


def labels_from_scores(scores):
    return (scores > 1e-10).astype(int)


def balanced_accuracy(prediction, labels):
    return float(np.mean([np.mean(prediction[labels == label] == label) for label in (0, 1)]))


def native_predictions(counts):
    forced = (counts[:, 1] > counts[:, 0]).astype(int)
    tied = counts[:, 0] == counts[:, 1]
    return np.where(tied, -1, forced), forced


def native_metrics(counts, y):
    strict, forced = native_predictions(counts)
    tied = strict == -1
    silence = np.all(counts == 0, axis=1)
    saturation = np.all(counts == 8, axis=1)
    return dict(strict_ba=balanced_accuracy(strict, y), forced_ba=balanced_accuracy(forced, y),
                coverage=float(np.mean(~tied)), silence=float(silence.mean()),
                saturated_tie=float(saturation.mean()),
                intermediate_tie=float(np.mean(tied & ~silence & ~saturation)))


def readout_features(raster):
    return {'counts': raster.sum(1).astype(float), 'raster': raster.reshape(len(raster), 16).astype(float)}


def readout_scores(raster, fitted):
    return {name: probe.predict(fitted[name], x)[:, 0] for name, x in readout_features(raster).items()}


def probe_metrics(scores, y):
    result = {}
    for name, value in scores.items():
        entry = probe.metrics(value, y)
        result[f'{name}_probe_ba'] = entry['balanced_accuracy']
        result[f'{name}_probe_mse'] = entry['mse']
    return result


def intervention(base_raster, donor_index, suffix, tap, y, baseline_state, patched_state,
                 baseline_scores, fitted):
    prefix_length = max(0, tap-15)
    patched_raster = np.concatenate((base_raster[:, :prefix_length], suffix), axis=1)
    base_counts = base_raster.sum(1).astype(int)
    counts = patched_raster.sum(1).astype(int)
    donor_label = y[donor_index]
    _, base_forced = native_predictions(base_counts)
    strict, forced = native_predictions(counts)
    scores = readout_scores(patched_raster, fitted)
    donor_sign = 2*donor_label - 1
    base_margin = base_counts[:, 1] - base_counts[:, 0]
    margin = counts[:, 1] - counts[:, 0]
    raster_changed = np.any(patched_raster != base_raster, axis=(1, 2))
    count_changed = np.any(counts != base_counts, axis=1)
    forced_changed = forced != base_forced
    state_transfer = ((donor_label != y) & (labels_from_scores(baseline_state) == y)
                      & (labels_from_scores(patched_state) == donor_label))

    def donor_gain(prediction, before):
        return balanced_accuracy(prediction, donor_label) - balanced_accuracy(before, donor_label)

    return dict(
        **native_metrics(counts, y), **probe_metrics(scores, y),
        donor_strict_ba=balanced_accuracy(strict, donor_label),
        donor_forced_ba=balanced_accuracy(forced, donor_label),
        donor_margin_shift=float(np.mean(donor_sign*(margin-base_margin))),
        donor_forced_gain=donor_gain(forced, base_forced),
        donor_count_probe_gain=donor_gain(labels_from_scores(scores['counts']), labels_from_scores(baseline_scores['counts'])),
        donor_raster_probe_gain=donor_gain(labels_from_scores(scores['raster']), labels_from_scores(baseline_scores['raster'])),
        donor_state_score_shift=float(np.mean(donor_sign*(patched_state-baseline_state))),
        raster_changed=float(raster_changed.mean()), count_changed=float(count_changed.mean()),
        margin_changed=float(np.mean(margin != base_margin)), forced_changed=float(forced_changed.mean()),
        raster_changed_count_same=float(np.mean(raster_changed & ~count_changed)),
        count_changed_forced_same=float(np.mean(count_changed & ~forced_changed)),
        suffix_eq_donor=float(np.mean(np.all(suffix == base_raster[donor_index, prefix_length:], axis=(1, 2)))),
        state_to_donor=float(state_transfer.mean()),
        state_to_donor_raster_same=float(np.mean(state_transfer & ~raster_changed)),
        state_to_donor_count_same=float(np.mean(state_transfer & ~count_changed)),
        state_to_donor_forced_same=float(np.mean(state_transfer & ~forced_changed)))
