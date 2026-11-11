"""Posthoc saved-record check of timing accessibility at exactly [4,4] counts."""

import numpy as np

from core import R, dump, read
from metrics import balanced_accuracy, labels_from_scores
import probe


def main():
    assert read(R / 'AUDIT.json')['status'] == 'PASS'
    rows = []
    for row in read(R / 'test/BASE_ROWS.json'):
        ident, seed = row['id'], row['seed']
        with np.load(R / 'test' / f'{ident}_base.npz') as native:
            same_counts = np.all(native['counts'] == 4, axis=1)
            raster = native['raster'][same_counts].reshape(-1, 16)
        with np.load(R / 'data' / f'test_s{seed}.npz') as base:
            labels = base['y'][same_counts]
        label_counts = np.bincount(labels, minlength=2).tolist()
        ba = None
        if min(label_counts) > 0:
            with np.load(R / 'fit' / f'{ident}_raster.npz') as fitted:
                scores = probe.predict(fitted, raster)[:, 0]
            ba = balanced_accuracy(labels_from_scores(scores), labels)
        rows.append(dict(id=ident, seed=seed, arm=row['arm'], loss=row['loss'],
                         n_equal_4_4=len(labels), label_counts=label_counts, raster_probe_ba=ba,
                         any_forced_count_only_ba=.5 if min(label_counts) > 0 else None))
    dump(R / 'FIXED_COUNT_DIAGNOSIS.json', dict(
        status='POSTHOC_SAVED_RECORDS_ONLY', count_pair=[4, 4], neural_reruns=0, probe_refits=0,
        rows=rows, scope='All 36 models, saved test rasters and frozen calibration coefficients.'))
    print(next(row for row in rows if row['id'] == 's2_GAIN2_COUNT'))


if __name__ == '__main__':
    main()
