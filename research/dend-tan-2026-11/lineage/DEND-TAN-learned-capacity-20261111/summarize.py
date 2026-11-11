"""Summarize independently audited capacity and matched-norm evaluations."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CONDITIONS = {
    'LEARN': ('base2', 'full2', 'full4', 'full4_hard', 'full4_hard_matched'),
    'ORACLE': ('base2', 'full2', 'full4', 'full4_hard'),
    'MEAN': ('base2', 'full2', 'full4'),
}
RESOURCES = ('soma_events', 'output_events', 'recurrent_deliveries', 'output_deliveries')
PRIMARY = ('capacity_gain', 'matched_hard_gain')
BOOTSTRAP_SEED = 4078900001
BOOTSTRAP_DRAWS = 50_000
PERCENTILES = (1.25, 98.75)
TOLERANCE = 1e-12
PAIRS = (
    ('full4', 'full2'),
    ('full2', 'base2'),
    ('full4_hard', 'full4'),
    ('full4_hard_matched', 'full4'),
    ('full4_hard_matched', 'full4_hard'),
)


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
    assert np.all((bits == 0) | (bits == 1))
    np.testing.assert_array_equal(y, bits[np.arange(n), selected])
    masks = {'all': np.ones(n, dtype=bool), 'older': selected < 2, 'recent': selected >= 2}
    masks.update({f'position{i}': selected == i for i in range(4)})
    for i in range(4):
        assert masks[f'position{i}'].sum() == n // 4
        assert (masks[f'position{i}'] & (y == 0)).sum() == n // 8
    return selected, y, masks


def verify_scores(row, data, selected, y, masks):
    counts, scores, weights, soft_weights = (data[name] for name in
                                            ('counts', 'cue_scores', 'cue_weights', 'cue_soft_weights'))
    slots = row['slots']
    true_slot = 3 - selected
    available = true_slot < slots
    first_scores = scores[:, 0]
    first_weights = weights[:, 0]
    order = np.arange(len(y))
    safe_slot = np.where(available, true_slot, 0)
    alternatives = first_scores.copy()
    alternatives[order, safe_slot] = -np.inf
    ranked_first = available & (first_scores[order, safe_slot] > alternatives.max(axis=1))
    actual_weight = np.where(available, first_weights[order, safe_slot], 0.)
    soft_weight = np.where(available, soft_weights[:, 0][order, safe_slot], 0.)
    pred = predict(counts)
    signals = {
        'ba': pred == y,
        'coverage': pred >= 0,
        'routing_accuracy': ranked_first,
        'target_weight': actual_weight,
    }
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
    if row['condition'] == 'base2':
        assert row['older_ba'] == .5 * row['older_coverage']
    if row['arm'] == 'ORACLE':
        assert np.array_equal(ranked_first, available)
    row['available_by_position'] = [float(available[masks[f'position{i}']].mean()) for i in range(4)]
    row['older_soft_target_weight'] = float(soft_weight[masks['older']].mean())
    row['recent_soft_target_weight'] = float(soft_weight[masks['recent']].mean())


def load_condition(row, selected, y, masks):
    seed, arm, condition = (row[name] for name in ('seed', 'arm', 'condition'))
    assert row['id'] == f's{seed}_{arm}' and (row['T'], row['q']) == (36, 28)
    slots = 2 if condition in ('base2', 'full2') else 4
    assert row['slots'] == slots
    keys = ('counts', 'resources', 'cue_scores', 'cue_weights', 'cue_soft_weights', 'cue_read_vectors', 'cue_fallback')
    with np.load(ROOT / f'evaluate/{row["id"]}_{condition}.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in keys}
    n = len(y)
    counts, resources, scores, weights, soft_weights, vectors, fallback = (data[name] for name in keys)
    assert counts.shape == (n, 2) and resources.shape == (n, 6)
    assert scores.shape == weights.shape == soft_weights.shape == (n, 3, slots)
    assert vectors.shape == (n, 3, 12) and fallback.shape == (n, 3) and fallback.dtype == np.bool_
    assert all(np.isfinite(array).all() for array in data.values())
    assert np.all((0 <= counts) & (counts <= 8)) and np.array_equal(counts, counts.astype(int))
    assert np.all(weights >= 0.) and np.all(soft_weights >= 0.)
    np.testing.assert_allclose(weights.sum(axis=-1), 1., atol=TOLERANCE, rtol=0)
    np.testing.assert_allclose(soft_weights.sum(axis=-1), 1., atol=TOLERANCE, rtol=0)
    assert int(fallback.sum()) == row['fallback_reads']
    if condition != 'full4_hard_matched':
        assert row['fallback_reads'] == 0
    if condition in ('base2', 'full2', 'full4'):
        np.testing.assert_array_equal(weights, soft_weights)
    np.testing.assert_allclose(resources[:, :4].mean(axis=0), row['mean_resources'], atol=TOLERANCE, rtol=0)
    row.update(zip(RESOURCES, row['mean_resources']))
    verify_scores(row, data, selected, y, masks)
    norms = np.linalg.norm(vectors, axis=-1)
    row.update(read_l2_mean=float(norms.mean()), read_l2_max=float(norms.max()),
               older_read_l2_mean=float(norms[masks['older']].mean()),
               recent_read_l2_mean=float(norms[masks['recent']].mean()))
    row.update({f'cost_{key}': value for key, value in row['cost_per_episode'].items()})
    return data


def verify_matched(seed, cases):
    soft, hard, matched = (cases[name] for name in ('full4', 'full4_hard', 'full4_hard_matched'))
    for name in ('cue_scores', 'cue_soft_weights'):
        np.testing.assert_array_equal(soft[name], hard[name])
        np.testing.assert_array_equal(soft[name], matched[name])
    np.testing.assert_array_equal(hard['cue_weights'], matched['cue_weights'])
    target = np.linalg.norm(soft['cue_read_vectors'], axis=-1)
    direction = hard['cue_read_vectors']
    direction_norm = np.linalg.norm(direction, axis=-1)
    fallback = (direction_norm == 0.) & (target > 0.)
    np.testing.assert_array_equal(matched['cue_fallback'], fallback)
    ratio = np.divide(target, direction_norm, out=np.zeros_like(target), where=direction_norm > 0.)
    expected = direction * ratio[..., None]
    expected[..., 0][fallback] = target[fallback]
    actual = matched['cue_read_vectors']
    norm_residual = float(np.abs(np.linalg.norm(actual, axis=-1) - target).max())
    vector_residual = float(np.abs(actual - expected).max())
    assert norm_residual <= TOLERANCE and vector_residual <= TOLERANCE
    return dict(seed=seed, read_steps=target.size, fallback_reads=int(fallback.sum()),
                zero_hard_reads=int((direction_norm == 0.).sum()), zero_soft_reads=int((target == 0.).sum()),
                max_norm_residual=norm_residual, max_vector_residual=vector_residual)


def paired_changes(seed, arm, cases, y, masks):
    records = []
    for left, right in PAIRS:
        if left not in cases or right not in cases:
            continue
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
    for arm, conditions in CONDITIONS.items():
        for condition in conditions:
            row = rows[seed, arm, condition]
            for name in ('older_ba', 'recent_ba', 'older_routing_accuracy', 'recent_routing_accuracy',
                         'older_target_weight', 'recent_target_weight'):
                result[f'{arm}_{condition}_{name}'] = row[name]
        full2, full4, base2 = (rows[seed, arm, name] for name in ('full2', 'full4', 'base2'))
        result[f'{arm}_capacity_gain'] = full4['older_ba'] - full2['older_ba']
        result[f'{arm}_recent_capacity_change'] = full4['recent_ba'] - full2['recent_ba']
        result[f'{arm}_recent_early_write_effect'] = full2['recent_ba'] - base2['recent_ba']
        if 'full4_hard' in conditions:
            hard = rows[seed, arm, 'full4_hard']
            result[f'{arm}_hard_gain'] = hard['older_ba'] - full4['older_ba']
            result[f'{arm}_recent_hard_change'] = hard['recent_ba'] - full4['recent_ba']
    matched = rows[seed, 'LEARN', 'full4_hard_matched']
    soft = rows[seed, 'LEARN', 'full4']
    hard = rows[seed, 'LEARN', 'full4_hard']
    result.update(capacity_gain=result['LEARN_capacity_gain'],
                  matched_hard_gain=matched['older_ba'] - soft['older_ba'],
                  recent_matched_hard_change=matched['recent_ba'] - soft['recent_ba'],
                  matched_minus_raw_hard=matched['older_ba'] - hard['older_ba'])
    return result


def effect_summary(name, block_rows, resamples=None):
    values = np.array([row[name] for row in block_rows], dtype=float)
    out = dict(effect=name, blocks=len(values), values=values.tolist(), mean=float(values.mean()),
               sample_sd=float(values.std(ddof=1)), minimum=float(values.min()), maximum=float(values.max()),
               positive=int((values > 0.).sum()), zero=int((values == 0.).sum()), negative=int((values < 0.).sum()))
    if resamples is not None:
        low, high = np.percentile(values[resamples].mean(axis=1), PERCENTILES)
        out['ci97_5'] = [float(low), float(high)]
        out['direction'] = 'positive' if low > 0. else 'negative' if high < 0. else 'undetermined'
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
               'older_target_weight', 'recent_target_weight', 'available_fraction',
               'older_soft_target_weight', 'recent_soft_target_weight', 'read_l2_mean', 'read_l2_max',
               'older_read_l2_mean', 'recent_read_l2_mean') + RESOURCES
    vectors = ('ba_by_position', 'coverage_by_position', 'routing_by_position', 'target_weight_by_position',
               'available_by_position')
    groups = []
    for arm, conditions in CONDITIONS.items():
        for condition in conditions:
            cells = [rows[seed, arm, condition] for seed in range(blocks)]
            values = {name: float(np.mean([cell[name] for cell in cells])) for name in scalars}
            values.update({name: np.mean([cell[name] for cell in cells], axis=0).tolist() for name in vectors})
            groups.append(dict(arm=arm, condition=condition, blocks=blocks, slots=cells[0]['slots'],
                               cost_per_episode=cells[0]['cost_per_episode'],
                               fallback_reads=sum(cell['fallback_reads'] for cell in cells), **values))
    return groups


def main():
    freeze, science, audit = map(read, ('FREEZE.json', 'SCIENCE_SUMMARY.json', 'AUDIT.json'))
    blocks, n = freeze['blocks'], freeze['examples_per_block']
    assert blocks == 12 and n in (256, 512)
    assert science['status'] == 'COMPLETE' and audit['status'] == 'PASS'
    assert science['evaluations'] == audit['evaluations'] == 12 * blocks
    assert science['examples_per_block'] == n and science['blocks'] == blocks
    assert science['fits'] == science['updates'] == 0
    assert science['forward_sample_steps'] == 12 * blocks * n * 36
    assert science['dense_forward_macs'] == 156 * science['forward_sample_steps']

    evaluation = sorted(read('evaluate/ROWS.json'), key=lambda row: (
        row['seed'], tuple(CONDITIONS).index(row['arm']), CONDITIONS[row['arm']].index(row['condition'])))
    index = {(row['seed'], row['arm'], row['condition']): row for row in evaluation}
    expected = {(seed, arm, condition) for seed in range(blocks)
                for arm, conditions in CONDITIONS.items() for condition in conditions}
    assert len(evaluation) == len(index) == len(expected) and set(index) == expected
    block_rows, changes, matches = [], [], []
    for seed in range(blocks):
        selected, y, masks = load_base(seed, n)
        for arm, conditions in CONDITIONS.items():
            cases = {name: load_condition(index[seed, arm, name], selected, y, masks) for name in conditions}
            changes.extend(paired_changes(seed, arm, cases, y, masks))
            if arm == 'LEARN':
                match = verify_matched(seed, cases)
                matches.append(match)
                index[seed, arm, 'full4_hard_matched'].update(
                    {key: value for key, value in match.items() if key != 'seed'})
        block_rows.append(block_summary(seed, index))

    assert len(matches) == science['matched_cases'] == blocks
    assert sum(row['read_steps'] for row in matches) == science['matched_read_steps'] == blocks * n * 3
    assert sum(row['fallback_reads'] for row in matches) == science['fallback_reads']
    resamples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, blocks, size=(BOOTSTRAP_DRAWS, blocks))
    primary = [effect_summary(name, block_rows, resamples) for name in PRIMARY]
    verify_primary(primary, audit)
    secondary_names = [f'{arm}_{name}' for arm in CONDITIONS for name in
                       ('capacity_gain', 'recent_capacity_change', 'recent_early_write_effect')]
    secondary_names += [f'{arm}_{name}' for arm in ('LEARN', 'ORACLE')
                        for name in ('hard_gain', 'recent_hard_change')]
    secondary_names += ['recent_matched_hard_change', 'matched_minus_raw_hard']
    secondary = [effect_summary(name, block_rows) for name in secondary_names]
    result = dict(study_status='FROZEN_FOUR_SLOT_CAPACITY',
                  inference=dict(blocks=blocks, examples_per_block=n, bootstrap_seed=BOOTSTRAP_SEED,
                                 bootstrap_draws=BOOTSTRAP_DRAWS, percentiles=PERCENTILES,
                                 nominal_family_confidence=.95, independent_primary_match=True),
                  primary_effects=primary, secondary_effects=secondary, block_rows=block_rows,
                  model_rows=evaluation, group_rows=group_summary(index, blocks), paired_change_rows=changes,
                  match_rows=matches, max_norm_residual=max(row['max_norm_residual'] for row in matches),
                  max_vector_residual=max(row['max_vector_residual'] for row in matches))
    (ROOT / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                       encoding='utf-8')
    for name, records in (('PRIMARY_EFFECTS.csv', primary), ('BLOCK_RESULTS.csv', block_rows), ('EVAL_ROWS.csv', evaluation)):
        export_csv(name, records)
    print(f'Summarized {blocks} paired blocks, {len(evaluation)} evaluations, and {len(matches)} norm matches.')


if __name__ == '__main__':
    main()
