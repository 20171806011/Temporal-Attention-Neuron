"""Aggregate audited timing-increment results at the original model-block level."""

import csv
import json
from pathlib import Path

import numpy as np


R = Path(__file__).resolve().parent
ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2')
LOSSES = ('CE', 'COUNT', 'CECOM')
BLOCKS = (0, 1, 2)
SHUFFLES = 16
BOOTSTRAP_SEED = 4021900001
BOOTSTRAP_DRAWS = 20000
RESOURCE_NAMES = ('hidden_spikes', 'output_spikes', 'recurrent_fanout', 'output_fanout')
BOOTSTRAP = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 3, (BOOTSTRAP_DRAWS, 3))

RANK_FIELDS = (
    'pair_rank', 'pair_rank_shuffle_mean', 'pair_rank_delta', 'count_pair_rank',
    'pair_rank_4_4', 'pair_rank_shuffled_4_4_mean',
)
BA_FIELDS = ('native_strict_ba', 'native_forced_ba', 'count_ba', 'raster_ba')
COVERAGE_FIELDS = (
    'fit_pairs', 'fit_matched_quartets', 'test_pairs', 'test_matched_quartets',
    'test_matched_fraction', 'pair_equal_raster_fraction', 'matched_4_4_pairs',
    'matched_4_4_quartets', 'fixed_4_4_n',
)
OTHER_FIELDS = (
    'native_coverage', 'native_silence', 'native_saturated_tie', 'native_intermediate_tie',
    'count_mse', 'raster_mse', 'prior_4_4_raster_ba',
)


def read(name):
    return json.loads((R / name).read_text(encoding='utf-8'))


def mean_available(values):
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def block_statistics(values):
    """Only compute the frozen three-block interval for complete block data."""
    assert len(values) == 3
    available = sum(value is not None for value in values)
    interval = None
    if available == 3:
        sample = np.asarray(values, dtype=float)
        assert np.isfinite(sample).all()
        interval = np.quantile(sample[BOOTSTRAP].mean(axis=1), [.025, .975]).tolist()
    return dict(values=values, mean=mean_available(values), ci95=interval, available_blocks=available)


def close(a, b):
    return abs(a-b) <= 1e-12


def verify(rows):
    expected = {(arm, loss, block) for arm in ARMS for loss in LOSSES for block in BLOCKS}
    assert len(rows) == len({r['id'] for r in rows}) == 36
    assert {(r['arm'], r['loss'], r['seed']) for r in rows} == expected
    for row in rows:
        assert len(row['pair_fit_status']) == SHUFFLES + 1
        assert len(row['pair_rank_shuffled']) == SHUFFLES
        assert len(row['mean_resources']) == 4
        assert 0 <= row['fit_matched_quartets'] <= 128
        assert 0 <= row['test_matched_quartets'] <= 256
        assert row['fit_matched_quartets'] <= row['fit_pairs'] <= 4*row['fit_matched_quartets']
        assert row['test_matched_quartets'] <= row['test_pairs'] <= 4*row['test_matched_quartets']
        assert close(row['test_matched_fraction'], row['test_matched_quartets']/256)
        assert 0 <= row['matched_4_4_quartets'] <= row['test_matched_quartets']
        assert row['matched_4_4_quartets'] <= row['matched_4_4_pairs'] <= row['test_pairs']
        labels = row['fixed_4_4_label_counts']
        assert len(labels) == 2 and sum(labels) == row['fixed_4_4_n'] <= 1024
        assert (row['prior_4_4_raster_ba'] is not None) == (min(labels) > 0)
        assert all(0 <= row[field] <= 1 for field in BA_FIELDS)
        if row['test_pairs'] == 0:
            assert all(row[field] is None for field in RANK_FIELDS)
            assert all(value is None for value in row['pair_rank_shuffled'])
            assert row['pair_equal_raster_fraction'] is None
            continue
        assert row['count_pair_rank'] == .5
        assert 0 <= row['pair_rank'] <= 1
        assert 0 <= row['pair_equal_raster_fraction'] <= 1
        assert all(0 <= value <= 1 for value in row['pair_rank_shuffled'])
        assert close(row['pair_rank_shuffle_mean'], np.mean(row['pair_rank_shuffled']))
        assert close(row['pair_rank_delta'], row['pair_rank']-row['pair_rank_shuffle_mean'])
        if row['fit_pairs'] == 0:
            assert close(row['pair_rank'], .5)
            assert all(close(value, .5) for value in row['pair_rank_shuffled'])
        for field in ('pair_rank_4_4', 'pair_rank_shuffled_4_4_mean'):
            assert (row[field] is None) == (row['matched_4_4_pairs'] == 0)


def summaries(rows):
    block_rows = []
    for block in BLOCKS:
        models = [row for row in rows if row['seed'] == block]
        eligible = [row for row in models if row['test_pairs'] > 0]
        block_rows.append(dict(
            seed=block, models=len(models), eligible_models=len(eligible),
            undefined_test_models=[row['id'] for row in models if row['test_pairs'] == 0],
            zero_fit_pairs_models=[row['id'] for row in models if row['fit_pairs'] == 0],
            **{key: mean_available([row[key] for row in models]) for key in RANK_FIELDS},
            pair_rank_shuffled=[mean_available([row['pair_rank_shuffled'][j] for row in eligible])
                                for j in range(SHUFFLES)],
            mean_test_matched_quartets=mean_available([row['test_matched_quartets'] for row in models]),
            mean_test_matched_fraction=mean_available([row['test_matched_fraction'] for row in models]),
            **{key: mean_available([row[key] for row in models]) for key in BA_FIELDS},
        ))

    groups = []
    for arm in ARMS:
        for loss in LOSSES:
            models = sorted((row for row in rows if row['arm'] == arm and row['loss'] == loss),
                            key=lambda row: row['seed'])
            fields = RANK_FIELDS + BA_FIELDS + COVERAGE_FIELDS + OTHER_FIELDS
            metrics = {key: block_statistics([row[key] for row in models]) for key in fields}
            for j, name in enumerate(RESOURCE_NAMES):
                metrics[name] = block_statistics([row['mean_resources'][j] for row in models])
            for j in range(SHUFFLES):
                metrics[f'shuffle_{j+1:02d}_rank'] = block_statistics([
                    row['pair_rank_shuffled'][j] for row in models
                ])
            groups.append(dict(arm=arm, loss=loss, model_ids=[row['id'] for row in models], metrics=metrics))

    eligible = [row for row in rows if row['test_pairs'] > 0]
    overall = dict(
        eligible_models=len(eligible),
        undefined_test_models=[row['id'] for row in rows if row['test_pairs'] == 0],
        zero_fit_pairs_models=[row['id'] for row in rows if row['fit_pairs'] == 0],
        positive_deltas=sum(row['pair_rank_delta'] > 1e-12 for row in eligible),
        zero_deltas=sum(abs(row['pair_rank_delta']) <= 1e-12 for row in eligible),
        negative_deltas=sum(row['pair_rank_delta'] < -1e-12 for row in eligible),
        matched_test_quartets=sum(row['test_matched_quartets'] for row in rows),
        mean_test_matched_fraction=mean_available([row['test_matched_fraction'] for row in rows]),
        matched_test_pairs=sum(row['test_pairs'] for row in rows),
        fixed_4_4_both_classes=sum(row['prior_4_4_raster_ba'] is not None for row in rows),
        clean_raster_above_count=sum(row['raster_ba']-row['count_ba'] > 1e-12 for row in rows),
        clean_raster_equal_count=sum(abs(row['raster_ba']-row['count_ba']) <= 1e-12 for row in rows),
        clean_raster_below_count=sum(row['raster_ba']-row['count_ba'] < -1e-12 for row in rows),
        test_resources={name: int(round(1024*sum(row['mean_resources'][j] for row in rows)))
                        for j, name in enumerate(RESOURCE_NAMES)},
    )
    primary = {key: block_statistics([row[key] for row in block_rows]) for key in RANK_FIELDS}
    clean = {key: block_statistics([mean_available([row[key] for row in rows if row['seed'] == block])
                                    for block in BLOCKS])
             for key in BA_FIELDS + ('native_coverage', 'count_mse', 'raster_mse')}
    return dict(
        study_status='FRESH_HELD_OUT_TIMING_INCREMENT', model_rows=rows, block_rows=block_rows,
        group_rows=groups, primary=primary, clean=clean, overall=overall,
        historical_case=next(row for row in rows if row['id'] == 's2_GAIN2_COUNT'),
        inference=dict(model_blocks=3, shuffle_count=SHUFFLES, bootstrap_seed=BOOTSTRAP_SEED,
                       bootstrap_draws=BOOTSTRAP_DRAWS, interval='pointwise 95%, unadjusted, complete blocks',
                       primary_rule='equal-weight available models within original blocks; equal-weight blocks',
                       ranking_rule='equal-weight matching quartets, then equal-weight pairs within quartet'),
    )


def export_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (R / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def flatten_group(row):
    result = {key: value for key, value in row.items() if key != 'metrics'}
    for key, stat in row['metrics'].items():
        result[f'{key}_mean'] = stat['mean']
        result[f'{key}_available_blocks'] = stat['available_blocks']
        for seed, value in enumerate(stat['values']):
            result[f'{key}_block{seed}'] = value
        low, high = stat['ci95'] if stat['ci95'] is not None else (None, None)
        result[f'{key}_ci95_low'] = low
        result[f'{key}_ci95_high'] = high
    return result


def main():
    science, audit = read('SCIENCE_SUMMARY.json'), read('AUDIT.json')
    assert science['status'] == 'COMPLETE' and audit['status'] == 'PASS'
    assert (science['models'], science['neural_training_updates'], science['rank_fits'],
            science['class_fits'], science['baseline_captures']) == (36, 0, 612, 72, 72)
    rows = sorted(read('test/ROWS.json'), key=lambda row: (ARMS.index(row['arm']),
                  LOSSES.index(row['loss']), row['seed']))
    verify(rows)
    result = summaries(rows)
    (R / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                             allow_nan=False) + '\n', encoding='utf-8')
    export_csv('model_metrics.csv', rows)
    export_csv('group_summary.csv', [flatten_group(row) for row in result['group_rows']])
    export_csv('block_summary.csv', result['block_rows'])
    export_csv('shuffle_metrics.csv', [
        dict(id=row['id'], arm=row['arm'], loss=row['loss'], seed=row['seed'], shuffle=j+1,
             test_matched_quartets=row['test_matched_quartets'], pair_rank_shuffled=score,
             pair_rank=row['pair_rank'],
             difference=None if score is None else row['pair_rank']-score)
        for row in rows for j, score in enumerate(row['pair_rank_shuffled'])
    ])
    print(f'Summarized {len(rows)} models, {len(result["group_rows"])} groups and 576 shuffled results.')


if __name__ == '__main__':
    main()
