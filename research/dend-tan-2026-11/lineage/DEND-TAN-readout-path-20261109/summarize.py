"""Summarize the audited state-transplant records, without replaying models."""

import csv
import json
from pathlib import Path

import numpy as np


R = Path(__file__).resolve().parent
ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2')
LOSSES = ('CE', 'COUNT', 'CECOM')
BLOCKS = (0, 1, 2)
DONORS = ('cue_flip', 'content_flip', 'both_flip')
OPPOSITE = DONORS[:2]
T15_GROUPS = ('full', 'hidden', 'branch', 'soma_p', 'output', 'history')
T19_GROUPS = ('full', 'hidden', 'output')
NATIVE_FIELDS = (
    'strict_ba', 'forced_ba', 'coverage', 'silence', 'saturated_tie',
    'intermediate_tie', 'counts_probe_ba', 'counts_probe_mse',
    'raster_probe_ba', 'raster_probe_mse',
)
BASE_FIELDS = NATIVE_FIELDS + ('joint9_ba', 'joint15_ba', 'joint19_ba')
PATCH_FIELDS = NATIVE_FIELDS + (
    'donor_strict_ba', 'donor_forced_ba', 'donor_margin_shift',
    'donor_forced_gain', 'donor_count_probe_gain', 'donor_raster_probe_gain',
    'donor_state_score_shift', 'raster_changed', 'count_changed',
    'margin_changed', 'forced_changed', 'raster_changed_count_same',
    'count_changed_forced_same', 'suffix_eq_donor', 'state_to_donor',
    'state_to_donor_raster_same', 'state_to_donor_count_same',
    'state_to_donor_forced_same',
)
RESOURCE_NAMES = ('hidden_spikes', 'output_spikes', 'recurrent_fanout', 'output_fanout')
BOOTSTRAP_SEED = 4011900001
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 3, (BOOTSTRAP_DRAWS, 3))


def read(name):
    return json.loads((R / name).read_text(encoding='utf-8'))


def conditions(arm):
    """All frozen patch conditions, including the implementation controls."""
    yield 9, 'full', 'content_flip'
    t15_groups = T15_GROUPS if arm in ('FULL', 'NOATT') else T15_GROUPS[:-1]
    for time, groups in ((15, t15_groups), (19, T19_GROUPS)):
        for group in groups:
            for donor in DONORS:
                yield time, group, donor
        yield time, 'full', 'self'


def block_statistics(values):
    """Use only the three original model blocks as resampling units."""
    assert len(values) == 3
    if all(value is None for value in values):
        return None
    x = np.asarray(values, dtype=float)
    assert np.all(np.isfinite(x))
    draws = x[BOOTSTRAP].mean(1)
    return dict(
        values=x.tolist(),
        mean=float(x.mean()),
        ci95=np.quantile(draws, [.025, .975]).tolist(),
    )


def export_csv(name, rows):
    assert rows
    columns = list(dict.fromkeys(field for row in rows for field in row))
    with (R / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def metric_rows(rows, fields, resources_field, prefix):
    """Expand resource vectors for numeric grouping without editing the raw rows."""
    out = []
    for row in rows:
        metrics = {field: row[field] for field in fields}
        resources = row[resources_field]
        assert len(resources) == len(RESOURCE_NAMES)
        metrics.update({prefix + key: value for key, value in zip(RESOURCE_NAMES, resources)})
        if 'mean_query_counts' in row:
            counts = row['mean_query_counts']
            assert len(counts) == 2
            metrics.update(query_count0=counts[0], query_count1=counts[1])
        out.append(metrics)
    return out


def group_statistics(metric_records):
    assert len(metric_records) == 3
    return {
        key: block_statistics([row[key] for row in metric_records])
        for key in metric_records[0]
    }


def combine_directions(left, right):
    assert left.keys() == right.keys()
    result = {}
    for key in left:
        if left[key] is None or right[key] is None:
            assert left[key] is None and right[key] is None
            result[key] = None
        else:
            result[key] = (left[key] + right[key]) / 2
    return result


def flatten_group(row):
    out = {key: value for key, value in row.items() if key != 'metrics'}
    for key, stat in row['metrics'].items():
        out[f'{key}_mean'] = None if stat is None else stat['mean']
        for block in BLOCKS:
            out[f'{key}_block{block}'] = None if stat is None else stat['values'][block]
        out[f'{key}_ci95_low'] = None if stat is None else stat['ci95'][0]
        out[f'{key}_ci95_high'] = None if stat is None else stat['ci95'][1]
    return out


def verify_and_index(baselines, patches):
    expected_models = {(arm, loss, seed) for arm in ARMS for loss in LOSSES for seed in BLOCKS}
    baseline_index = {(r['arm'], r['loss'], r['seed']): r for r in baselines}
    assert len(baselines) == len(baseline_index) == 36
    assert set(baseline_index) == expected_models
    assert len({r['id'] for r in baselines}) == 36

    patch_index = {
        (r['arm'], r['loss'], r['seed'], r['time'], r['group'], r['donor']): r
        for r in patches
    }
    expected_patches = {
        (arm, loss, seed, time, group, donor)
        for arm, loss, seed in expected_models
        for time, group, donor in conditions(arm)
    }
    assert len(patches) == len(patch_index) == len({r['patch_id'] for r in patches}) == 1026
    assert set(patch_index) == expected_patches

    early = own = donor_suffix = 0
    for key, row in patch_index.items():
        assert row['id'] == baseline_index[key[:3]]['id']
        raster, counts, forced = (row[k] for k in ('raster_changed', 'count_changed', 'forced_changed'))
        assert 0 <= forced <= counts <= raster <= 1
        assert abs(row['raster_changed_count_same'] - (raster - counts)) <= 1e-12
        assert abs(row['count_changed_forced_same'] - (counts - forced)) <= 1e-12
        if row['time'] == 9 or row['donor'] == 'self':
            assert raster == counts == forced == 0
            if row['time'] == 9:
                early += 1
            else:
                own += 1
        if row['group'] == 'full' and row['time'] in (15, 19) and row['donor'] != 'self':
            assert row['suffix_eq_donor'] == 1
            donor_suffix += 1
    assert (early, own, donor_suffix) == (36, 72, 216)
    return baseline_index, patch_index, dict(
        identical_prefix_t9=early, self_replay=own, full_nonself_donor_suffix=donor_suffix,
    )


def grouped_results(baseline_index, patch_index):
    native_groups = []
    all_groups = []
    primary = []
    for arm in ARMS:
        for loss in LOSSES:
            base = [baseline_index[(arm, loss, seed)] for seed in BLOCKS]
            metrics = metric_rows(base, BASE_FIELDS, 'mean_resources', 'native_')
            native_groups.append(dict(arm=arm, loss=loss, metrics=group_statistics(metrics)))

            block_rows = {}
            for time, group, donor in conditions(arm):
                raw = [patch_index[(arm, loss, seed, time, group, donor)] for seed in BLOCKS]
                metrics = metric_rows(raw, PATCH_FIELDS, 'mean_suffix_resources', 'suffix_')
                block_rows[(time, group, donor)] = metrics
                all_groups.append(dict(
                    arm=arm, loss=loss, time=time, group=group, donor=donor,
                    metrics=group_statistics(metrics),
                ))
            for group in T15_GROUPS:
                if (15, group, OPPOSITE[0]) not in block_rows:
                    continue
                first, second = (block_rows[(15, group, donor)] for donor in OPPOSITE)
                metrics = [combine_directions(a, b) for a, b in zip(first, second)]
                primary.append(dict(
                    arm=arm, loss=loss, time=15, group=group,
                    donor='opposite_equal_weight', metrics=group_statistics(metrics),
                ))
    assert (len(native_groups), len(all_groups), len(primary)) == (12, 342, 66)
    return native_groups, all_groups, primary


def overall_description(baselines, primary):
    fields = ('strict_ba', 'forced_ba', 'coverage', 'counts_probe_ba', 'raster_probe_ba',
              'joint15_ba', 'joint19_ba', 'counts_probe_mse', 'raster_probe_mse')
    means = {key: float(np.mean([row[key] for row in baselines])) for key in fields}
    counts = dict(
        counts_probe_above_forced=sum(r['counts_probe_ba'] > r['forced_ba'] for r in baselines),
        raster_probe_above_counts=sum(r['raster_probe_ba'] > r['counts_probe_ba'] for r in baselines),
        raster_probe_equal_counts=sum(r['raster_probe_ba'] == r['counts_probe_ba'] for r in baselines),
        raster_probe_below_counts=sum(r['raster_probe_ba'] < r['counts_probe_ba'] for r in baselines),
    )
    families = {}
    for group in T15_GROUPS:
        cells = [row for row in primary if row['group'] == group]
        fields = ('donor_margin_shift', 'donor_forced_gain', 'raster_changed', 'count_changed',
                  'forced_changed', 'raster_changed_count_same', 'count_changed_forced_same',
                  'state_to_donor', 'state_to_donor_raster_same', 'state_to_donor_count_same',
                  'state_to_donor_forced_same')
        families[group] = dict(
            model_count=3 * len(cells),
            **{key: float(np.mean([row['metrics'][key]['mean'] for row in cells])) for key in fields},
        )
    return dict(native_means=means, clean_model_comparisons=counts, t15_families=families)


def main():
    audit = read('AUDIT.json')
    science = read('SCIENCE_SUMMARY.json')
    assert audit['status'] == 'PASS' and science['status'] == 'COMPLETE'
    assert (science['models'], science['neural_training_updates'], science['probe_fits'],
            science['baseline_captures'], science['patch_conditions']) == (36, 0, 72, 72, 1026)
    baselines = read('test/BASE_ROWS.json')
    patches = read('test/ROWS.json')
    baseline_index, patch_index, controls = verify_and_index(baselines, patches)
    native_groups, groups, primary = grouped_results(baseline_index, patch_index)
    result = dict(
        study_status='EXPLORATORY_FROZEN_STATE_TRANSPLANTS',
        native_models=baselines, native_groups=native_groups, condition_groups=groups,
        primary_effects=primary, control_checks=controls,
        overall=overall_description(baselines, primary),
        inference=dict(model_blocks=3, bootstrap_seed=BOOTSTRAP_SEED,
                       bootstrap_draws=BOOTSTRAP_DRAWS, interval='pointwise 95%, unadjusted',
                       primary_rule='t15; average cue_flip and content_flip within model block first',
                       proportion_denominator='all 256 recipient episodes per condition'),
    )
    (R / 'RESULTS.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    export_csv('intervention_metrics.csv', patches)
    export_csv('native_metrics.csv', baselines)
    export_csv('group_summary.csv', [flatten_group(row) for row in groups])
    export_csv('primary_effects.csv', [flatten_group(row) for row in primary])
    print(f'Summarized {len(patches)} intervention rows and {len(primary)} primary cells.')


if __name__ == '__main__':
    main()
