"""Summarize audited selective resets with paired record-level comparisons."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ARMS = ('LEARN', 'ORACLE', 'MEAN')
CONDITIONS = ('full_original', 'full_always', 'full_selective', 'late_original', 'late_always', 'late_selective')
RESOURCES = ('soma_events', 'output_events', 'recurrent_deliveries', 'output_deliveries')
PRIMARY = ('recent_gain', 'older_change')
MARGIN = -.02
THRESHOLD = 1. / np.sqrt(2.)
INVALID_CONFIDENCE = -2.
BOOTSTRAP_SEED = 4080900001
BOOTSTRAP_DRAWS = 50_000
PERCENTILES = (1.25, 98.75)
TOLERANCE = 1e-12
PAIRS = (
    ('full_selective', 'full_original'), ('full_always', 'full_original'), ('full_selective', 'full_always'),
    ('late_selective', 'late_original'), ('late_always', 'late_original'), ('late_selective', 'late_always'),
    ('late_original', 'full_original'),
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
    signals = {'ba': pred == y, 'coverage': pred >= 0, 'routing_accuracy': ranked_first,
               'target_weight': actual_weight, 'reset_fraction': data['reset_steps'].any(axis=1)}
    verified = {'available_fraction': float(available.mean()), 'reset_count': int(data['reset_steps'].sum()),
                'onset_count': int(data['read_onset'].sum())}
    for name, signal in signals.items():
        overall = 'balanced_accuracy' if name == 'ba' else name
        verified[overall] = float(signal.mean())
        verified[f'older_{name}'] = float(signal[masks['older']].mean())
        verified[f'recent_{name}'] = float(signal[masks['recent']].mean())
        by_position = {'routing_accuracy': 'routing', 'reset_fraction': 'reset'}.get(name, name)
        verified[f'{by_position}_by_position'] = [float(signal[masks[f'position{i}']].mean()) for i in range(4)]
    for key, expected in verified.items():
        np.testing.assert_allclose(row[key], expected, atol=TOLERANCE, rtol=0, err_msg=key)
    if row['condition'].startswith('late_'):
        assert row['older_ba'] == .5 * row['older_coverage']
    if row['arm'] == 'ORACLE':
        np.testing.assert_array_equal(ranked_first, available)


def verify_resets(data, arm, rule):
    onset, reset, confidence = (data[name] for name in ('read_onset', 'reset_steps', 'match_confidence'))
    assert onset.dtype == reset.dtype == np.bool_
    assert np.all(onset[:, 25]) and onset.sum() == len(onset)
    assert not np.any(reset & ~onset)
    valid = confidence != INVALID_CONFIDENCE
    assert np.all((confidence[valid] >= -1. - TOLERANCE) & (confidence[valid] <= 1. + TOLERANCE))
    assert not np.any(valid & ~onset)
    if rule == 'selective':
        np.testing.assert_array_equal(reset, onset & (confidence >= THRESHOLD))
        if arm == 'MEAN':
            assert not valid.any() and not reset.any()
    else:
        assert not valid.any()
        if rule == 'always':
            np.testing.assert_array_equal(reset, onset)
        else:
            assert rule == 'original' and not reset.any()
    before, after = data['cue_native_before'], data['cue_native_after']
    cue_reset = reset[:, 25:28]
    np.testing.assert_array_equal(after[cue_reset], np.zeros_like(after[cue_reset]))
    np.testing.assert_array_equal(after[~cue_reset], before[~cue_reset])


def load_condition(row, selected, y, masks):
    seed, arm, condition = (row[name] for name in ('seed', 'arm', 'condition'))
    assert row['id'] == f's{seed}_{arm}' and (row['T'], row['q'], row['slots']) == (36, 28, 2)
    shapes = {'counts': (2,), 'resources': (6,), 'cue_scores': (3, 2), 'cue_weights': (3, 2),
              'cue_read_vectors': (3, 12), 'read_onset': (36,), 'reset_steps': (36,),
              'match_confidence': (36,), 'cue_native_before': (3, 26), 'cue_native_after': (3, 26)}
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
    verify_resets(data, arm, condition.split('_')[1])
    verify_scores(row, data, selected, y, masks)
    norms = np.linalg.norm(data['cue_read_vectors'], axis=-1)
    row.update(read_l2_mean=float(norms.mean()), recent_read_l2_mean=float(norms[masks['recent']].mean()))
    cost = row['cost_per_episode']
    np.testing.assert_allclose(cost['reset_scalar_writes_mean'], 26*row['reset_count']/len(y), atol=TOLERANCE, rtol=0)
    row.update({f'cost_{key}': value for key, value in cost.items()})
    return data


def verify_group(arm, cases, masks):
    original = cases['full_original']
    for condition, case in cases.items():
        for name in ('cue_scores', 'cue_weights', 'read_onset'):
            np.testing.assert_array_equal(case[name], original[name])
        own_original = cases[f"{condition.split('_')[0]}_original"]
        np.testing.assert_array_equal(case['cue_native_before'][:, 0], own_original['cue_native_before'][:, 0])
        np.testing.assert_array_equal(case['cue_read_vectors'], own_original['cue_read_vectors'])
        if arm == 'ORACLE' and condition.endswith('_selective'):
            np.testing.assert_array_equal(case['reset_steps'].any(axis=1), masks['recent'])


def confidence_rows(seed, arm, cases, masks):
    output = []
    for condition in ('full_selective', 'late_selective'):
        case = cases[condition]
        confidence = case['match_confidence'][:, 25]
        for name, mask in masks.items():
            valid = confidence[mask & (confidence != INVALID_CONFIDENCE)]
            output.append(dict(seed=seed, arm=arm, condition=condition, subset=name, n=int(mask.sum()),
                               valid=int(valid.size), sum=float(valid.sum()),
                               minimum=float(valid.min()) if valid.size else None,
                               maximum=float(valid.max()) if valid.size else None))
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
        for condition in CONDITIONS:
            row = rows[seed, arm, condition]
            for name in ('older_ba', 'recent_ba', 'older_reset_fraction', 'recent_reset_fraction',
                         'older_coverage', 'recent_coverage'):
                result[f'{arm}_{condition}_{name}'] = row[name]
        for left, right in PAIRS:
            for subset in ('older', 'recent'):
                result[f'{arm}_{left}_minus_{right}_{subset}'] = (
                    rows[seed, arm, left][f'{subset}_ba'] - rows[seed, arm, right][f'{subset}_ba'])
    result.update(recent_gain=result['LEARN_full_selective_minus_full_original_recent'],
                  older_change=result['LEARN_full_selective_minus_full_original_older'])
    return result


def effect_summary(name, block_rows, resamples):
    values = np.array([row[name] for row in block_rows], dtype=float)
    low, high = np.percentile(values[resamples].mean(axis=1), PERCENTILES)
    return dict(effect=name, blocks=len(values), values=values.tolist(), mean=float(values.mean()),
                sample_sd=float(values.std(ddof=1)), minimum=float(values.min()), maximum=float(values.max()),
                positive=int((values > 0.).sum()), zero=int((values == 0.).sum()), negative=int((values < 0.).sum()),
                ci97_5=[float(low), float(high)],
                direction='positive' if low > 0 else 'negative' if high < 0 else 'undetermined')


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
               'older_target_weight', 'recent_target_weight', 'available_fraction', 'reset_fraction',
               'older_reset_fraction', 'recent_reset_fraction', 'read_l2_mean', 'recent_read_l2_mean') + RESOURCES
    vectors = ('ba_by_position', 'coverage_by_position', 'routing_by_position', 'target_weight_by_position',
               'reset_by_position')
    groups = []
    for arm in ARMS:
        for condition in CONDITIONS:
            cells = [rows[seed, arm, condition] for seed in range(blocks)]
            values = {name: float(np.mean([cell[name] for cell in cells])) for name in scalars}
            values.update({name: np.mean([cell[name] for cell in cells], axis=0).tolist() for name in vectors})
            totals = {name: sum(cell[name] for cell in cells) for name in ('reset_count', 'onset_count')}
            costs = cells[0]['cost_per_episode'].copy()
            costs['reset_scalar_writes_mean'] = float(np.mean(
                [cell['cost_per_episode']['reset_scalar_writes_mean'] for cell in cells]))
            groups.append(dict(arm=arm, condition=condition, blocks=blocks, slots=2,
                               cost_per_episode=costs, **totals, **values))
    return groups


def main():
    freeze, science, audit = map(read, ('FREEZE.json', 'SCIENCE_SUMMARY.json', 'AUDIT.json'))
    blocks, n = freeze['blocks'], freeze['examples_per_block']
    assert blocks == 12 and n in (256, 512)
    assert science['status'] == 'COMPLETE' and audit['status'] == 'PASS'
    assert science['evaluations'] == audit['evaluations'] == 18 * blocks
    assert science['examples_per_block'] == n and science['blocks'] == blocks
    assert science['fits'] == science['updates'] == 0
    assert science['forward_sample_steps'] == 18 * blocks * n * 36
    assert science['dense_forward_macs'] == 156 * science['forward_sample_steps']
    assert science['matcher_macs'] == 8 * science['forward_sample_steps']

    evaluation = sorted(read('evaluate/ROWS.json'), key=lambda row: (
        row['seed'], ARMS.index(row['arm']), CONDITIONS.index(row['condition'])))
    index = {(row['seed'], row['arm'], row['condition']): row for row in evaluation}
    expected = {(seed, arm, condition) for seed in range(blocks) for arm in ARMS for condition in CONDITIONS}
    assert len(evaluation) == len(index) == len(expected) and set(index) == expected
    assert sum(row['reset_count'] for row in evaluation) == science['reset_count']
    assert sum(row['onset_count'] for row in evaluation if row['condition'].endswith('_selective')) == science['selective_onsets']
    assert science['reset_scalar_writes'] == 26 * science['reset_count']
    block_rows, changes, confidence = [], [], []
    for seed in range(blocks):
        selected, y, masks = load_base(seed, n)
        for arm in ARMS:
            cases = {name: load_condition(index[seed, arm, name], selected, y, masks) for name in CONDITIONS}
            verify_group(arm, cases, masks)
            changes.extend(paired_changes(seed, arm, cases, y, masks))
            confidence.extend(confidence_rows(seed, arm, cases, masks))
        block_rows.append(block_summary(seed, index))

    resamples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, blocks, size=(BOOTSTRAP_DRAWS, blocks))
    primary = [effect_summary(name, block_rows, resamples) for name in PRIMARY]
    verify_primary(primary, audit)
    recent_pass, older_pass = primary[0]['ci97_5'][0] > 0., primary[1]['ci97_5'][0] > MARGIN
    assert audit['adoption_supported'] == (recent_pass and older_pass)
    for effect, threshold in zip(primary, (0., MARGIN)):
        independent = next(row for row in audit['primary_effects'] if row['effect'] == effect['effect'])
        effect.update(threshold=threshold, passes=effect['ci97_5'][0] > threshold)
        assert independent['threshold'] == effect['threshold'] and independent['passes'] == effect['passes']
    result = dict(study_status='FROZEN_SELECTIVE_NATIVE_RESET',
                  inference=dict(blocks=blocks, examples_per_block=n, bootstrap_seed=BOOTSTRAP_SEED,
                                 bootstrap_draws=BOOTSTRAP_DRAWS, percentiles=PERCENTILES,
                                 nominal_family_confidence=.95, independent_primary_match=True),
                  decision=dict(recent_pass=recent_pass, older_pass=older_pass,
                                both_pass=recent_pass and older_pass, older_margin=MARGIN),
                  primary_effects=primary, block_rows=block_rows, model_rows=evaluation,
                  group_rows=group_summary(index, blocks), paired_change_rows=changes,
                  confidence_rows=confidence)
    (ROOT / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                       encoding='utf-8')
    for name, records in (('PRIMARY_EFFECTS.csv', primary), ('BLOCK_RESULTS.csv', block_rows), ('EVAL_ROWS.csv', evaluation)):
        export_csv(name, records)
    print(f'Summarized {blocks} paired blocks and {len(evaluation)} evaluations with the frozen adoption criteria.')


if __name__ == '__main__':
    main()
