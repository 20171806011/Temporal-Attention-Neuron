"""Summarize audited native trajectories, swaps, and current replays."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2')
HISTORY_ARMS = ('FULL', 'NOATT')
BLOCKS = (0, 1, 2)
TIMES = {0: (8,), 16: (8, 16, 24)}
BASE_GROUPS = ('branch', 'soma', 'previous', 'soma_p', 'output', 'other', 'full')
HISTORY_GROUPS = ('history', 'branch_history')
ROUTES = ('self_blank', 'mean_blank', 'flip_first_blank', 'flip_second_blank',
          'mean_early', 'mean_late', 'mean_query')
DONOR_ROUTES = ('flip_first_blank', 'flip_second_blank')
MEAN_ROUTES = ('mean_blank', 'mean_early', 'mean_late', 'mean_query')
NATIVE_FIELDS = (
    'balanced_accuracy', 'accuracy', 'coverage', 'silence', 'saturated_tie',
    'intermediate_tie', 'agree_ba', 'disagree_ba', 'agree_coverage',
    'disagree_coverage', 'select_first_ba', 'select_second_ba',
    'select_first_coverage', 'select_second_coverage', 'disagree_joint_correct',
)
DONOR_FIELDS = (
    'selected_margin_shift', 'unselected_margin_shift', 'selection_specific_margin',
    'selected_donor_gain', 'unselected_recipient_gain', 'selected_count_changed',
    'unselected_count_changed',
)
PATCH_FIELDS = NATIVE_FIELDS + DONOR_FIELDS + (
    'delta_ba', 'delta_disagree_ba', 'raster_changed', 'count_changed',
    'distance_mean', 'max_cue_distance_difference',
    'continuation_steps', 'clamped_steps', 'dense_macs_per_episode',
)
RESOURCE_NAMES = ('hidden_spikes', 'output_spikes', 'recurrent_fanout',
                  'output_fanout', 'replayed_current_values')
PRIMARY = (
    'd16_t24_branch_minus_other', 'd16_blank_donor_replay',
    'd16_mean_blank_disagree_ba',
)
INTERACTIONS = {
    'd_by_other': ('full', 'branch', 'other'),
    'd_by_history': ('branch_history', 'branch', 'history'),
}
BOOTSTRAP_SEED = 4051900001
BOOTSTRAP_DRAWS = 20000
DRAW = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 3, (BOOTSTRAP_DRAWS, 3))


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def groups_for(arm):
    return BASE_GROUPS + HISTORY_GROUPS if arm in HISTORY_ARMS else BASE_GROUPS


def conditions(arm, delay):
    for tap in TIMES[delay]:
        yield 'state', tap, 'full', None, None
        for group in groups_for(arm):
            for bit in (0, 1):
                yield 'state', tap, group, bit, None
    if delay == 16:
        for route in ROUTES:
            tap = {'mean_late': 16, 'mean_query': 27}.get(route, 8)
            bit = {'flip_first_blank': 0, 'flip_second_blank': 1}.get(route)
            yield 'recurrence', tap, None, bit, route


def stats(values):
    assert len(values) == 3
    if all(value is None for value in values):
        return None
    x = np.asarray(values, dtype=float)
    assert np.all(np.isfinite(x))
    return dict(values=x.tolist(), mean=float(x.mean()),
                ci95=np.quantile(x[DRAW].mean(1), [.025, .975]).tolist())


def group_stats(rows):
    assert len(rows) == 3 and all(row.keys() == rows[0].keys() for row in rows)
    return {field: stats([row[field] for row in rows]) for field in rows[0]}


def average_rows(first, second):
    assert first.keys() == second.keys()
    result = {}
    for key, left in first.items():
        right = second[key]
        if left is None or right is None:
            assert left is None and right is None, key
            result[key] = None
        else:
            result[key] = (left + right) / 2
    return result


def expand_resources(row, name):
    raw = row[name]
    assert len(raw) == len(RESOURCE_NAMES)
    return {f'{name}_{metric}': value for metric, value in zip(RESOURCE_NAMES, raw)}


def numeric_baseline(row):
    return ({key: row[key] for key in (*NATIVE_FIELDS, 'persistent_array_bytes', 'history_slots')}
            | expand_resources(row, 'mean_resources'))


def numeric_patch(row, baseline):
    result = {key: row[key] for key in PATCH_FIELDS}
    result['delta_coverage'] = row['coverage'] - baseline['coverage']
    result['delta_disagree_coverage'] = row['disagree_coverage'] - baseline['disagree_coverage']
    for name in ('mean_resources', 'mean_suffix_resources', 'resource_delta'):
        result.update(expand_resources(row, name))
    return result


def index_rows(baselines, interventions):
    model = {(r['arm'], r['train_delay'], r['seed']): r for r in baselines}
    expected_models = {(arm, d, seed) for arm in ARMS for d in TIMES for seed in BLOCKS}
    assert len(baselines) == len(model) == len({r['id'] for r in baselines}) == 24
    assert set(model) == expected_models
    fields = ('arm', 'train_delay', 'seed', 'kind', 'tap', 'group', 'bit', 'route')
    cases = {tuple(r[key] for key in fields): r for r in interventions}
    expected_cases = {
        (arm, d, seed, *case) for arm, d, seed in expected_models for case in conditions(arm, d)
    }
    assert len(cases) == len(interventions) == len({r['case_id'] for r in interventions}) == 900
    assert set(cases) == expected_cases

    control_checks = dict(state_self=0, full_donor=0, current_self_blank=0,
                          bit_matched_rows=0, max_cue_distance_difference=0.)
    for key, row in cases.items():
        original = model[key[:3]]
        assert row['id'] == original['id']
        assert 0 <= row['count_changed'] <= row['raster_changed'] <= 1
        for suffix in ('ba', 'disagree_ba'):
            field = 'balanced_accuracy' if suffix == 'ba' else suffix
            assert abs(row[f'delta_{suffix}'] - (row[field] - original[field])) <= 1e-12
        if row['kind'] == 'state' and row['bit'] is None:
            assert row['count_changed'] == row['raster_changed'] == 0
            control_checks['state_self'] += 1
        elif row['kind'] == 'state' and row['group'] == 'full':
            control_checks['full_donor'] += 1
        elif row['route'] == 'self_blank':
            assert row['count_changed'] == row['raster_changed'] == 0
            control_checks['current_self_blank'] += 1
        if row['bit'] is None:
            assert all(row[field] is None for field in DONOR_FIELDS)
        else:
            assert all(row[field] is not None for field in DONOR_FIELDS)
            expected_margin = row['selected_margin_shift'] - row['unselected_margin_shift']
            assert abs(row['selection_specific_margin'] - expected_margin) <= 1e-12
            difference = row['max_cue_distance_difference']
            assert 0 <= difference <= 1e-12
            control_checks['bit_matched_rows'] += 1
            control_checks['max_cue_distance_difference'] = max(
                control_checks['max_cue_distance_difference'], difference)
    assert tuple(control_checks[key] for key in ('state_self', 'full_donor', 'current_self_blank',
                                                  'bit_matched_rows')) == (48, 96, 12, 792)
    return model, cases, control_checks


def collect_summaries(baselines, cases):
    native_groups, condition_groups, bit_groups, interactions = [], [], [], []
    paired_model_rows = {}
    for arm in ARMS:
        for delay, taps in TIMES.items():
            keys = [(arm, delay, seed) for seed in BLOCKS]
            originals = [baselines[key] for key in keys]
            native_groups.append(dict(arm=arm, train_delay=delay,
                                      metrics=group_stats([numeric_baseline(row) for row in originals])))
            for condition in conditions(arm, delay):
                kind, tap, group, bit, route = condition
                rows = [numeric_patch(cases[(*key, *condition)], original)
                        for key, original in zip(keys, originals)]
                condition_groups.append(dict(arm=arm, train_delay=delay, kind=kind, tap=tap,
                                             group=group, bit=bit, route=route, metrics=group_stats(rows)))

            for tap in taps:
                for group in groups_for(arm):
                    values = [average_rows(*[numeric_patch(cases[(*key, 'state', tap, group, bit, None)], original)
                                             for bit in (0, 1)])
                              for key, original in zip(keys, originals)]
                    paired_model_rows[arm, delay, tap, group] = values
                    bit_groups.append(dict(arm=arm, train_delay=delay, kind='state', tap=tap, group=group,
                                           route=None, metrics=group_stats(values)))

                for name, (combined, first, second) in INTERACTIONS.items():
                    if combined not in groups_for(arm):
                        continue
                    joined, a, b = (paired_model_rows[arm, delay, tap, group] for group in (combined, first, second))
                    values = [row['selection_specific_margin'] - left['selection_specific_margin']
                              - right['selection_specific_margin'] for row, left, right in zip(joined, a, b)]
                    interactions.append(dict(arm=arm, train_delay=delay, tap=tap, interaction=name,
                                             metrics={'selection_specific_margin': stats(values)}))

            if delay == 16:
                values = [average_rows(*[numeric_patch(cases[(*key, 'recurrence', 8, None, bit, route)], original)
                                         for bit, route in enumerate(DONOR_ROUTES)])
                          for key, original in zip(keys, originals)]
                paired_model_rows[arm, delay, 8, 'donor_replay'] = values
                bit_groups.append(dict(arm=arm, train_delay=delay, kind='recurrence', tap=8, group=None,
                                       route='donor_replay', metrics=group_stats(values)))
    assert (len(native_groups), len(condition_groups), len(bit_groups), len(interactions)) == (8, 300, 132, 24)
    return native_groups, condition_groups, bit_groups, interactions, paired_model_rows


def primary_effects(cases, paired):
    per_arm = {name: {} for name in PRIMARY}
    for arm in ARMS:
        branch, other = (paired[arm, 16, 24, group] for group in ('branch', 'other'))
        per_arm[PRIMARY[0]][arm] = [a['selection_specific_margin'] - b['selection_specific_margin']
                                    for a, b in zip(branch, other)]
        per_arm[PRIMARY[1]][arm] = [row['selection_specific_margin']
                                    for row in paired[arm, 16, 8, 'donor_replay']]
        per_arm[PRIMARY[2]][arm] = [cases[arm, 16, seed, 'recurrence', 8, None, None, 'mean_blank']['delta_disagree_ba']
                                    for seed in BLOCKS]
    units = ('spikes', 'spikes', 'fraction')
    return [dict(effect=name, unit=unit, overall=stats(np.mean([per_arm[name][arm] for arm in ARMS], axis=0)),
                 by_arm={arm: stats(values) for arm, values in per_arm[name].items()})
            for name, unit in zip(PRIMARY, units)]


def export(name, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (ROOT / name).open('w', encoding='utf-8-sig', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def flatten(row):
    result = {key: value for key, value in row.items() if key != 'metrics'}
    for key, value in row['metrics'].items():
        result[f'{key}_mean'] = None if value is None else value['mean']
        for seed in BLOCKS:
            result[f'{key}_block{seed}'] = None if value is None else value['values'][seed]
        result[f'{key}_ci95_low'] = None if value is None else value['ci95'][0]
        result[f'{key}_ci95_high'] = None if value is None else value['ci95'][1]
    return result


def main():
    science, audit = read('SCIENCE_SUMMARY.json'), read('AUDIT.json')
    assert science['status'] == 'COMPLETE' and audit['status'] == 'PASS'
    expected = dict(models=24, neural_training_updates=0, baseline_captures=24,
                    state_cases=816, recurrence_cases=84, forward_sample_steps=4220928,
                    clamped_sample_steps=270336, dense_forward_macs=639000576, diagnostic_rec_macs=7962624)
    assert all(science[key] == value for key, value in expected.items())
    baselines, interventions = read('evaluate/BASE_ROWS.json'), read('evaluate/ROWS.json')
    baseline_index, cases, controls = index_rows(baselines, interventions)
    native_groups, condition_groups, bit_groups, interactions, paired = collect_summaries(baseline_index, cases)
    primary = primary_effects(cases, paired)
    result = dict(study_status='EXPLORATORY_FROZEN_LATE_CUE_HOLD_PATHS', native_models=baselines,
                  model_rows=interventions, native_groups=native_groups, group_rows=condition_groups,
                  bit_paired_rows=bit_groups, interactions=interactions, primary_effects=primary,
                  control_checks=controls,
                  inference=dict(model_blocks=3, bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=BOOTSTRAP_DRAWS,
                                 interval='pointwise 95% descriptive', primary_weighting='bit then arm then block'))
    (ROOT / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    export('native_metrics.csv', baselines)
    export('intervention_metrics.csv', interventions)
    export('group_summary.csv', [flatten(row) for row in condition_groups])
    export('bit_paired_summary.csv', [flatten(row) for row in bit_groups])
    export('interactions.csv', [flatten(row) for row in interactions])
    export('primary_effects.csv', [flatten(dict(effect=row['effect'], unit=row['unit'], arm=arm,
                                               metrics={'effect': effect}))
                                   for row in primary for arm, effect in [('all_arms', row['overall']), *row['by_arm'].items()]])
    print(f'Summarized {len(baselines)} baselines, {len(interventions)} interventions and {len(primary)} primary comparisons.')


if __name__ == '__main__':
    main()
