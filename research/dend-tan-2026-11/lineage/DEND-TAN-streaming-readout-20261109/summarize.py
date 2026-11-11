"""Summarize audited streaming readouts using the original model blocks."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2')
LOSSES = ('CE', 'COUNT', 'CECOM')
BLOCKS = (0, 1, 2)
SHUFFLES = 8
METHODS = ('native_forced', 'count2', 'moment2')
RESOURCES = ('hidden_spikes', 'output_spikes', 'recurrent_fanout', 'output_fanout')
BOOTSTRAP_SEED = 4031900001
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 3, (BOOTSTRAP_DRAWS, 3))

BA_FIELDS = ('native_strict_ba', 'native_forced_ba', 'count2_ba', 'moment2_ba',
             'shuffle_ba_mean', 'offline16_ba')
CONTRASTS = ('moment_minus_count', 'moment_minus_shuffle', 'moment_minus_native')
MSE_FIELDS = ('count2_mse', 'moment2_mse', 'shuffle_mse_mean', 'offline16_mse')
MATCH_FIELDS = ('matched_pairs', 'matched_quartets', 'matched_count_rank',
                'matched_moment_rank', 'matched_shuffle_rank_mean', 'matched_offline_rank')
FIXED_FIELDS = ('fixed_4_4_n', 'fixed_4_4_moment_ba', 'fixed_4_4_offline_ba')
OTHER_FIELDS = ('native_coverage', 'native_silence', 'native_saturated_tie',
                'native_intermediate_tie', 'fit_seconds', 'fit_collect_seconds', 'test_collect_seconds')
SUMMARY_FIELDS = BA_FIELDS + CONTRASTS + MSE_FIELDS + MATCH_FIELDS + FIXED_FIELDS + OTHER_FIELDS


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def mean_available(values):
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def block_statistics(values):
    assert len(values) == len(BLOCKS)
    available = sum(value is not None for value in values)
    interval = None
    if available == len(BLOCKS):
        sample = np.asarray(values, dtype=float)
        assert np.isfinite(sample).all()
        interval = np.quantile(sample[BOOTSTRAP].mean(axis=1), [.025, .975]).tolist()
    return dict(values=values, mean=mean_available(values), ci95=interval,
                available_blocks=available)


def close(a, b):
    return abs(a - b) <= 1e-12


def verify(rows):
    expected = {(arm, loss, block) for arm in ARMS for loss in LOSSES for block in BLOCKS}
    assert len(rows) == len({row['id'] for row in rows}) == 36
    assert {(row['arm'], row['loss'], row['seed']) for row in rows} == expected
    for row in rows:
        assert len(row['shuffled_ba']) == len(row['shuffled_mse']) == SHUFFLES
        assert len(row['mean_resources']) == 4
        assert len(row['bench_median_ns']) == len(METHODS)
        assert all(np.isfinite(x) and x > 0 for x in row['bench_median_ns'])
        assert all(0 <= row[key] <= 1 for key in BA_FIELDS)
        assert all(0 <= value <= 1 for value in row['shuffled_ba'])
        assert all(np.isfinite(row[key]) and row[key] >= 0 for key in MSE_FIELDS)
        assert all(np.isfinite(x) and x >= 0 for x in row['shuffled_mse'])
        assert close(row['shuffle_ba_mean'], np.mean(row['shuffled_ba']))
        assert close(row['shuffle_mse_mean'], np.mean(row['shuffled_mse']))
        for key, baseline in zip(CONTRASTS, ('count2_ba', 'shuffle_ba_mean', 'native_forced_ba')):
            assert close(row[key], row['moment2_ba'] - row[baseline])
        assert 0 <= row['matched_quartets'] <= 256
        assert row['matched_quartets'] <= row['matched_pairs'] <= 4*row['matched_quartets']
        for key in MATCH_FIELDS[2:]:
            assert (row[key] is None) == (row['matched_pairs'] == 0)
            assert row[key] is None or 0 <= row[key] <= 1
        if row['matched_pairs']:
            assert row['matched_count_rank'] == .5
        labels = row['fixed_4_4_label_counts']
        assert len(labels) == 2 and sum(labels) == row['fixed_4_4_n'] <= 1024
        for key in FIXED_FIELDS[1:]:
            assert (row[key] is not None) == (min(labels) > 0)
            assert row[key] is None or 0 <= row[key] <= 1


def signs(values):
    present = [x for x in values if x is not None]
    return dict(positive=sum(x > 1e-12 for x in present),
                zero=sum(abs(x) <= 1e-12 for x in present),
                negative=sum(x < -1e-12 for x in present), missing=len(values)-len(present))


def summaries(rows):
    blocks = []
    for seed in BLOCKS:
        models = [row for row in rows if row['seed'] == seed]
        assert len(models) == 12
        blocks.append(dict(
            seed=seed, models=len(models), matched_models=sum(row['matched_pairs'] > 0 for row in models),
            fixed_4_4_models=sum(row['fixed_4_4_moment_ba'] is not None for row in models),
            **{key: mean_available([row[key] for row in models]) for key in SUMMARY_FIELDS},
            **{f'{method}_bench_median_ns': float(np.median([row['bench_median_ns'][j] for row in models]))
               for j, method in enumerate(METHODS)},
        ))

    groups = []
    for arm in ARMS:
        for loss in LOSSES:
            models = [row for row in rows if (row['arm'], row['loss']) == (arm, loss)]
            assert [row['seed'] for row in models] == list(BLOCKS)
            metrics = {key: block_statistics([row[key] for row in models]) for key in SUMMARY_FIELDS}
            for j, method in enumerate(METHODS):
                metrics[f'{method}_bench_median_ns'] = block_statistics([
                    row['bench_median_ns'][j] for row in models
                ])
            for j, name in enumerate(RESOURCES):
                metrics[name] = block_statistics([row['mean_resources'][j] for row in models])
            groups.append(dict(arm=arm, loss=loss, model_ids=[row['id'] for row in models], metrics=metrics))

    overall = {key: block_statistics([row[key] for row in blocks]) for key in SUMMARY_FIELDS}
    benchmarks = []
    for j, method in enumerate(METHODS):
        values = [row['bench_median_ns'][j] for row in rows]
        benchmarks.append(dict(method=method, median_ns=float(np.median(values)),
                               min_ns=min(values), max_ns=max(values), model_medians_ns=values))

    return dict(
        study_status='FROZEN_NATIVE_PULSES_STREAMING_READOUT', model_rows=rows, group_rows=groups,
        block_rows=blocks, overall=overall, benchmarks=benchmarks,
        contrast_signs={key: signs([row[key] for row in rows]) for key in CONTRASTS},
        coverage=dict(matched_models=sum(row['matched_pairs'] > 0 for row in rows),
                      unmatched_models=[row['id'] for row in rows if row['matched_pairs'] == 0],
                      matched_pairs=sum(row['matched_pairs'] for row in rows),
                      matched_quartets=sum(row['matched_quartets'] for row in rows),
                      fixed_4_4_models=sum(row['fixed_4_4_moment_ba'] is not None for row in rows)),
        resources=dict(test_totals={name: int(round(1024*sum(row['mean_resources'][j] for row in rows)))
                                    for j, name in enumerate(RESOURCES)},
                       total_fit_seconds=sum(row['fit_seconds'] for row in rows),
                       total_fit_collect_seconds=sum(row['fit_collect_seconds'] for row in rows),
                       total_test_collect_seconds=sum(row['test_collect_seconds'] for row in rows)),
        inference=dict(model_blocks=3, models_per_block=12, shuffle_count=SHUFFLES,
                       bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=BOOTSTRAP_DRAWS,
                       interval='pointwise 95%, descriptive, complete blocks',
                       primary='MOMENT2 minus COUNT2 whole-test balanced accuracy',
                       ranking='equal matching quartets, equal pairs within quartet'),
    )


def export_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (ROOT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (list, dict)) else value for key, value in row.items()})


def flatten_group(group):
    result = {key: value for key, value in group.items() if key != 'metrics'}
    for key, stat in group['metrics'].items():
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
    assert science['models'] == 36 and science['neural_training_updates'] == 0
    assert science['baseline_captures'] == 72 and science['probe_fits'] == 396
    assert science['benchmark_batches'] == 6912
    rows = sorted(read('test/ROWS.json'), key=lambda row: (ARMS.index(row['arm']),
                  LOSSES.index(row['loss']), row['seed']))
    for row in rows:
        row['shuffle_mse_mean'] = float(np.mean(row['shuffled_mse']))
    verify(rows)
    result = summaries(rows)
    (ROOT / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                allow_nan=False) + '\n', encoding='utf-8')
    export_csv('model_metrics.csv', rows)
    export_csv('group_summary.csv', [flatten_group(group) for group in result['group_rows']])
    export_csv('block_summary.csv', result['block_rows'])
    export_csv('shuffle_metrics.csv', [
        dict(id=row['id'], arm=row['arm'], loss=row['loss'], seed=row['seed'], shuffle=index+1,
             shuffled_ba=value, shuffled_mse=row['shuffled_mse'][index], moment2_ba=row['moment2_ba'],
             difference=row['moment2_ba']-value)
        for row in rows for index, value in enumerate(row['shuffled_ba'])
    ])
    print('Summarized 36 models, 12 groups, three blocks and 288 shuffled controls.')


if __name__ == '__main__':
    main()
