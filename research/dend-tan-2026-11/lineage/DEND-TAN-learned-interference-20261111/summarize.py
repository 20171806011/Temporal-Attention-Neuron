"""Summarize independently audited crosses of native states and stored values."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ARMS = ('LEARN', 'ORACLE', 'MEAN')
CONDITIONS = ('full_full', 'late_full', 'full_late', 'late_late')
RESOURCES = ('soma_events', 'output_events', 'recurrent_deliveries', 'output_deliveries')
PRIMARY = ('native_effect', 'value_effect')
SECONDARY = ('native_at_full_value', 'native_at_late_value', 'value_at_full_native', 'value_at_late_native',
             'interaction', 'natural_recovery')
PAIRS = (('late_full', 'full_full'), ('late_late', 'full_late'), ('full_late', 'full_full'),
         ('late_late', 'late_full'), ('late_late', 'full_full'))
ENTRY_SEGMENTS = {'branch': slice(0, 12), 'soma': slice(12, 18), 'output': slice(18, 20),
                  'previous_pulse': slice(20, 26)}
BOOTSTRAP_SEED = 4079900001
BOOTSTRAP_DRAWS = 50_000
PERCENTILES = (1.25, 98.75)
TOLERANCE = 1e-12


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def export_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (ROOT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (list, dict)) else value for key, value in row.items()})


def predict(counts):
    return np.where(counts[:, 0] == counts[:, 1], -1, counts.argmax(axis=1))


def load_base(seed, n):
    with np.load(ROOT / f'evaluate/test_s{seed}.npz', allow_pickle=False) as archive:
        selected, bits, y = (archive[name] for name in ('selected', 'bits', 'y'))
    assert selected.shape == y.shape == (n,) and bits.shape == (n, 4)
    assert np.all((0 <= selected) & (selected < 4))
    np.testing.assert_array_equal(y, bits[np.arange(n), selected])
    masks = {'all': np.ones(n, dtype=bool), 'older': selected < 2, 'recent': selected >= 2}
    masks.update({f'position{i}': selected == i for i in range(4)})
    for i in range(4):
        assert masks[f'position{i}'].sum() == n // 4
        assert (masks[f'position{i}'] & (y == 0)).sum() == n // 8
    return selected, y, masks


def verify_scores(row, data, selected, y, masks):
    counts, scores, weights = (data[name] for name in ('counts', 'cue_scores', 'cue_weights'))
    true_slot = 3 - selected
    available = true_slot < 2
    first_scores = scores[:, 0]
    first_weights = weights[:, 0]
    order = np.arange(len(y))
    safe_slot = np.where(available, true_slot, 0)
    ranked_first = available & (first_scores[order, safe_slot] > first_scores[order, 1-safe_slot])
    actual_weight = np.where(available, first_weights[order, safe_slot], 0.)
    pred = predict(counts)
    signals = {'ba': pred == y, 'coverage': pred >= 0,
               'routing_accuracy': ranked_first, 'target_weight': actual_weight}
    verified = {'available_fraction': float(available.mean())}
    for name, signal in signals.items():
        overall = 'balanced_accuracy' if name == 'ba' else name
        verified[overall] = float(signal.mean())
        verified[f'older_{name}'] = float(signal[masks['older']].mean())
        verified[f'recent_{name}'] = float(signal[masks['recent']].mean())
        by_position = 'routing' if name == 'routing_accuracy' else name
        verified[f'{by_position}_by_position'] = [float(signal[masks[f'position{i}']].mean()) for i in range(4)]
    for key, expected in verified.items():
        np.testing.assert_allclose(row[key], expected, atol=TOLERANCE, rtol=0, err_msg=key)
    if row['condition'] == 'late_late':
        assert row['older_ba'] == .5 * row['older_coverage']
    if row['arm'] == 'ORACLE':
        np.testing.assert_array_equal(ranked_first, available)


def load_condition(row, selected, y, masks):
    seed, arm, condition = (row[name] for name in ('seed', 'arm', 'condition'))
    assert row['id'] == f's{seed}_{arm}' and (row['T'], row['q'], row['slots']) == (36, 28, 2)
    shapes = {'counts': (2,), 'resources': (6,), 'cue_scores': (3, 2), 'cue_weights': (3, 2),
              'cue_read_vectors': (3, 12), 'entry_native': (26,), 'entry_keys': (2, 2),
              'entry_values': (12, 2)}
    with np.load(ROOT / f'evaluate/{row["id"]}_{condition}.npz', allow_pickle=False) as archive:
        data = {name: archive[name] for name in shapes}
    for name, shape in shapes.items():
        assert data[name].shape == (len(y), *shape), name
        assert np.isfinite(data[name]).all(), name
    counts = data['counts']
    assert np.all((0 <= counts) & (counts <= 8)) and np.array_equal(counts, counts.astype(int))
    weights = data['cue_weights']
    assert np.all(weights >= 0.)
    np.testing.assert_allclose(weights.sum(axis=-1), 1., atol=TOLERANCE, rtol=0)
    np.testing.assert_allclose(data['resources'][:, :4].mean(axis=0), row['mean_resources'], atol=TOLERANCE, rtol=0)
    row.update(zip(RESOURCES, row['mean_resources']))
    verify_scores(row, data, selected, y, masks)
    norms = np.linalg.norm(data['cue_read_vectors'], axis=-1)
    row.update(read_l2_mean=float(norms.mean()), read_l2_max=float(norms.max()),
               older_read_l2_mean=float(norms[masks['older']].mean()),
               recent_read_l2_mean=float(norms[masks['recent']].mean()))
    row.update({f'cost_{key}': value for key, value in row['cost_per_episode'].items()})
    return data


def verify_cross(cases):
    full, late = cases['full_full'], cases['late_late']
    for condition in CONDITIONS:
        case = cases[condition]
        native_source, value_source = condition.split('_')
        np.testing.assert_array_equal(case['entry_native'], (full if native_source == 'full' else late)['entry_native'])
        np.testing.assert_array_equal(case['entry_values'], (full if value_source == 'full' else late)['entry_values'])
        for name in ('entry_keys', 'cue_scores', 'cue_weights'):
            np.testing.assert_array_equal(case[name], full[name])
        np.testing.assert_array_equal(case['cue_read_vectors'],
                                      (full if value_source == 'full' else late)['cue_read_vectors'])


def entry_changes(seed, arm, cases, masks):
    full, late = cases['full_full'], cases['late_late']
    changes = {name: full['entry_native'][:, part] - late['entry_native'][:, part]
               for name, part in ENTRY_SEGMENTS.items()}
    changes['values'] = (full['entry_values'] - late['entry_values']).reshape(len(masks['all']), -1)
    output = []
    for component, delta in changes.items():
        for subset in ('all', 'older', 'recent'):
            selected = delta[masks[subset]]
            distances = np.linalg.norm(selected, axis=1)
            output.append(dict(seed=seed, arm=arm, component=component, subset=subset, n=len(selected),
                               changed=int(np.any(selected != 0., axis=1).sum()),
                               mean_l2=float(distances.mean()), max_l2=float(distances.max()),
                               max_abs=float(np.abs(selected).max())))
    return output


def paired_changes(seed, arm, cases, y, masks):
    records = []
    for left, right in PAIRS:
        left_counts, right_counts = (cases[name]['counts'] for name in (left, right))
        left_pred, right_pred = predict(left_counts), predict(right_counts)
        left_correct, right_correct = left_pred == y, right_pred == y
        changed = np.any(left_counts != right_counts, axis=1)
        for subset, mask in masks.items():
            records.append(dict(seed=seed, arm=arm, left=left, right=right, subset=subset, n=int(mask.sum()),
                                count_changed=int((mask & changed).sum()),
                                prediction_changed=int((mask & (left_pred != right_pred)).sum()),
                                correctness_changed=int((mask & (left_correct != right_correct)).sum()),
                                left_only_correct=int((mask & left_correct & ~right_correct).sum()),
                                right_only_correct=int((mask & right_correct & ~left_correct).sum())))
    return records


def block_summary(seed, rows):
    result = {'seed': seed}
    for arm in ARMS:
        values = {}
        for condition in CONDITIONS:
            row = rows[seed, arm, condition]
            values[condition] = row['recent_ba']
            for name in ('older_ba', 'recent_ba', 'recent_coverage', 'recent_routing_accuracy', 'recent_target_weight',
                         'recent_read_l2_mean'):
                result[f'{arm}_{condition}_{name}'] = row[name]
        ff, lf, fl, ll = (values[name] for name in CONDITIONS)
        effects = dict(native_at_full_value=lf-ff, native_at_late_value=ll-fl, value_at_full_native=fl-ff,
                       value_at_late_native=ll-lf, interaction=ll-lf-fl+ff, natural_recovery=ll-ff,
                       native_effect=((lf-ff)+(ll-fl))/2, value_effect=((fl-ff)+(ll-lf))/2)
        assert effects['native_effect'] + effects['value_effect'] == effects['natural_recovery']
        result.update({f'{arm}_{name}': value for name, value in effects.items()})
    result.update({name: result[f'LEARN_{name}'] for name in PRIMARY})
    return result


def effect_summary(name, block_rows, resamples=None):
    values = np.array([row[name] for row in block_rows], dtype=float)
    out = dict(effect=name, blocks=len(values), values=values.tolist(), mean=float(values.mean()),
               sample_sd=float(values.std(ddof=1)), minimum=float(values.min()), maximum=float(values.max()),
               positive=int((values > 0.).sum()), zero=int((values == 0.).sum()), negative=int((values < 0.).sum()))
    if resamples is not None:
        low, high = np.percentile(values[resamples].mean(axis=1), PERCENTILES)
        out['ci97_5'] = [float(low), float(high)]
        out['direction'] = 'positive' if low > 0 else 'negative' if high < 0 else 'undetermined'
    return out


def verify_primary(primary, audit):
    audited = {row['effect']: row for row in audit['primary_effects']}
    assert set(audited) == set(PRIMARY) and len(audit['primary_effects']) == len(PRIMARY)
    for effect in primary:
        independent = audited[effect['effect']]
        for name in ('blocks', 'positive', 'zero', 'negative', 'direction'):
            assert effect[name] == independent[name], name
        for name in ('values', 'mean', 'sample_sd', 'minimum', 'maximum', 'ci97_5'):
            np.testing.assert_allclose(effect[name], independent[name], atol=TOLERANCE, rtol=0, err_msg=name)


def group_summary(rows, blocks):
    scalars = ('balanced_accuracy', 'older_ba', 'recent_ba', 'coverage', 'older_coverage', 'recent_coverage',
               'routing_accuracy', 'older_routing_accuracy', 'recent_routing_accuracy', 'target_weight',
               'older_target_weight', 'recent_target_weight', 'available_fraction', 'read_l2_mean', 'read_l2_max',
               'older_read_l2_mean', 'recent_read_l2_mean') + RESOURCES
    vectors = ('ba_by_position', 'coverage_by_position', 'routing_by_position', 'target_weight_by_position')
    groups = []
    for arm in ARMS:
        for condition in CONDITIONS:
            cells = [rows[seed, arm, condition] for seed in range(blocks)]
            values = {name: float(np.mean([cell[name] for cell in cells])) for name in scalars}
            values.update({name: np.mean([cell[name] for cell in cells], axis=0).tolist() for name in vectors})
            groups.append(dict(arm=arm, condition=condition, blocks=blocks, slots=cells[0]['slots'],
                               cost_per_episode=cells[0]['cost_per_episode'], **values))
    return groups


def main():
    freeze, science, audit = map(read, ('FREEZE.json', 'SCIENCE_SUMMARY.json', 'AUDIT.json'))
    blocks, n = freeze['blocks'], freeze['examples_per_block']
    assert blocks == 12 and n in (256, 512)
    assert science['status'] == 'COMPLETE' and audit['status'] == 'PASS'
    assert science['evaluations'] == audit['evaluations'] == 12 * blocks
    assert science['examples_per_block'] == n and science['blocks'] == blocks
    assert science['fits'] == science['updates'] == 0
    assert science['forward_sample_steps'] == 3 * blocks * n * 94
    assert science['path_sample_steps'] == 12 * blocks * n * 36
    assert science['dense_forward_macs'] == 156 * science['forward_sample_steps']
    assert science['matcher_macs'] == 8 * science['forward_sample_steps']
    assert science['boundary_bank_bytes'] == 3 * blocks * n * 2 * 54 * 8

    evaluation = sorted(read('evaluate/ROWS.json'), key=lambda row: (
        row['seed'], ARMS.index(row['arm']), CONDITIONS.index(row['condition'])))
    index = {(row['seed'], row['arm'], row['condition']): row for row in evaluation}
    expected = {(seed, arm, condition) for seed in range(blocks) for arm in ARMS for condition in CONDITIONS}
    assert len(evaluation) == len(index) == len(expected) and set(index) == expected
    block_rows, changes, entry = [], [], []
    for seed in range(blocks):
        selected, y, masks = load_base(seed, n)
        for arm in ARMS:
            cases = {name: load_condition(index[seed, arm, name], selected, y, masks) for name in CONDITIONS}
            verify_cross(cases)
            changes.extend(paired_changes(seed, arm, cases, y, masks))
            entry.extend(entry_changes(seed, arm, cases, masks))
        block_rows.append(block_summary(seed, index))

    resamples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, blocks, size=(BOOTSTRAP_DRAWS, blocks))
    primary = [effect_summary(name, block_rows, resamples) for name in PRIMARY]
    verify_primary(primary, audit)
    secondary_names = [f'{arm}_{name}' for arm in ARMS for name in SECONDARY]
    secondary_names.extend(f'{arm}_{name}' for arm in ('ORACLE', 'MEAN') for name in PRIMARY)
    secondary = [effect_summary(name, block_rows) for name in secondary_names]
    result = dict(study_status='FROZEN_NATIVE_VALUE_CROSS',
                  inference=dict(blocks=blocks, examples_per_block=n, bootstrap_seed=BOOTSTRAP_SEED,
                                 bootstrap_draws=BOOTSTRAP_DRAWS, percentiles=PERCENTILES,
                                 nominal_family_confidence=.95, independent_primary_match=True),
                  primary_effects=primary, secondary_effects=secondary, block_rows=block_rows,
                  model_rows=evaluation, group_rows=group_summary(index, blocks), paired_change_rows=changes,
                  entry_change_rows=entry)
    (ROOT / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                       encoding='utf-8')
    for name, records in (('PRIMARY_EFFECTS.csv', primary), ('BLOCK_RESULTS.csv', block_rows), ('EVAL_ROWS.csv', evaluation)):
        export_csv(name, records)
    print(f'Summarized {blocks} paired blocks, {len(evaluation)} evaluations and {len(entry)} entry-state comparisons.')


if __name__ == '__main__':
    main()
