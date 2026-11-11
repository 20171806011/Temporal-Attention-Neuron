"""Summarize the audited fixed-weight direction and norm crossover."""

import csv
import json
from pathlib import Path

import numpy as np

from metrics import score


ROOT = Path(__file__).resolve().parent
KEY_CONDITIONS = ('nominal', 'swap_read', 'swap_direction', 'swap_magnitude', 'mean_read', 'mean_matched')
CONDITIONS = dict(KEY=KEY_CONDITIONS, KEY_MEAN=('nominal',), GAIN2=('nominal',))
MATCHED = {'swap_direction': ('swap_read', 'nominal'), 'swap_magnitude': ('nominal', 'swap_read'),
           'mean_matched': ('mean_read', 'nominal')}
PAIRS = (('swap_direction', 'nominal'), ('swap_read', 'swap_magnitude'),
         ('swap_magnitude', 'nominal'), ('swap_read', 'swap_direction'),
         ('nominal', 'mean_matched'), ('mean_matched', 'mean_read'), ('swap_read', 'nominal'), ('nominal', 'mean_read'))
METRICS = ('balanced_accuracy', 'accuracy', 'coverage', 'silence', 'saturated_tie', 'intermediate_tie',
           'agree_ba', 'disagree_ba', 'agree_coverage', 'disagree_coverage', 'disagree_joint_correct',
           'counterfactual_joint', 'counterfactual_disagree_ba', 'order0_disagree_ba', 'order1_disagree_ba')
RESOURCES = ('soma_events', 'output_events', 'recurrent_deliveries', 'output_deliveries')
NORM_FIELDS = ('read_l2_mean', 'read_l2_max', 'zero_read_norms', 'fallback_reads')
PRIMARY = ('equal_norm_direction_effect', 'selected_direction_minus_matched_mean')
SECONDARY = ('direction_at_selected_norm', 'direction_at_other_norm', 'norm_at_selected_direction',
             'norm_at_other_direction', 'direction_norm_interaction', 'selected_direction_true_joint_norm_effect',
             'other_direction_true_joint_norm_effect', 'mean_norm_matching_gain', 'selected_minus_raw_mean', 'natural_swap_gain')
BOOTSTRAP_SEED = 4076900001
BOOTSTRAP_DRAWS = 50_000
PERCENTILES = (1.25, 98.75)
MATCH_TOLERANCE = 1e-12


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


def base_data(seed, n):
    with np.load(ROOT / f'evaluate/test_s{seed}.npz', allow_pickle=False) as archive:
        keys = ('y', 'first_bit', 'second_bit', 'selected', 'order', 'octet')
        base = {key: archive[key] for key in keys}
    assert all(value.shape == (n,) for value in base.values())
    y, first, second, cue, order = (base[key] for key in keys[:-1])
    disagree = first != second
    assert np.array_equal(y, np.where(cue == 0, first, second))
    assert disagree.sum() == n // 2 and (y == 0).sum() == n // 2
    masks = dict(all=np.ones(n, dtype=bool), disagree=disagree,
                 order0=disagree & (order == 0), order1=disagree & (order == 1))
    for key in ('order0', 'order1'):
        assert masks[key].sum() == n // 4 and np.count_nonzero(masks[key] & (y == 0)) == n // 8
    return base, masks


def predict(counts):
    return np.where(counts[:, 0] == counts[:, 1], -1, counts.argmax(axis=1))


def load_condition(row, base, masks):
    seed, arm, condition = (row[key] for key in ('seed', 'arm', 'condition'))
    assert row['id'] == f's{seed}_{arm}' and (row['T'], row['q']) == (36, 28)
    keys = ('counts', 'resources', 'read_vectors', 'direction_fallback')
    if arm != 'GAIN2':
        keys += ('cue_memory',)
    with np.load(ROOT / f'evaluate/{row["id"]}_{condition}.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in keys}
    n = len(base['y'])
    counts, resources, vectors, fallback = (data[key] for key in keys[:4])
    assert counts.shape == (n, 2) and resources.shape == (n, 6)
    assert vectors.shape == (n, 3, 12) and fallback.shape == (n, 3) and fallback.dtype == np.bool_
    assert np.all((0 <= counts) & (counts <= 8)) and np.array_equal(counts, counts.astype(int))
    assert all(np.isfinite(array).all() for array in data.values())
    if arm != 'GAIN2':
        assert data['cue_memory'].shape == (n, 3, 12, 2)
    assert int(fallback.sum()) == row['fallback_reads']
    if condition not in MATCHED:
        assert row['fallback_reads'] == 0
    np.testing.assert_allclose(resources[:, :4].mean(axis=0), row['mean_resources'], rtol=0, atol=1e-12)
    row.update(zip(RESOURCES, row['mean_resources']))

    values, anti = score(counts, base), score(counts[:, ::-1], base)
    values.update(counterfactual_joint=anti['disagree_joint_correct'], counterfactual_disagree_ba=anti['disagree_ba'])
    correct = predict(counts) == base['y']
    for key in ('order0', 'order1'):
        values[f'{key}_disagree_ba'] = correct[masks[key]].mean()
    assert all(np.allclose(row[key], value, rtol=0, atol=1e-12) for key, value in values.items())
    norms = np.linalg.norm(vectors, axis=-1)
    row.update(read_l2_mean=float(norms.mean()), read_l2_max=float(norms.max()), zero_read_norms=int((norms == 0).sum()))
    cost = row['cost_per_episode']
    slots = 0 if arm == 'GAIN2' else 24
    assert (cost['parameters'], cost['trace_slots'], cost['continuous_state_slots']) == (176, slots, 20 + slots)
    assert cost['dense_forward_macs'] == 156 * 36
    row.update({f'cost_{key}': value for key, value in cost.items()})
    return data


def match_diagnostics(seed, key_cases):
    output = []
    for condition, (direction, norm_target) in MATCHED.items():
        d, target = (key_cases[case]['read_vectors'] for case in (direction, norm_target))
        actual = key_cases[condition]['read_vectors']
        d_norm = np.linalg.norm(d, axis=-1)
        target_norm = np.linalg.norm(target, axis=-1)
        fallback = (d_norm == 0) & (target_norm > 0)
        ratios = np.divide(target_norm, d_norm, out=np.zeros_like(d_norm), where=d_norm > 0)
        expected = d * ratios[..., None]
        expected[..., 0][fallback] = target_norm[fallback]
        vector_residual = float(np.abs(expected - actual).max())
        norm_residual = float(np.abs(np.linalg.norm(actual, axis=-1) - target_norm).max())
        np.testing.assert_array_equal(key_cases[condition]['direction_fallback'], fallback)
        assert vector_residual <= MATCH_TOLERANCE and norm_residual <= MATCH_TOLERANCE
        output.append(dict(seed=seed, condition=condition, direction_source=direction, norm_target=norm_target,
                           read_steps=target_norm.size, zero_direction=int((d_norm == 0).sum()),
                           zero_target_norm=int((target_norm == 0).sum()), fallback_reads=int(fallback.sum()),
                           max_abs_l2_residual=norm_residual, max_abs_vector_residual=vector_residual))
    return output


def paired_changes(seed, cases, y, masks):
    output = []
    for left, right in PAIRS:
        before, after = (cases[name]['counts'] for name in (right, left))
        old, new = predict(before), predict(after)
        correct_old, correct_new = old == y, new == y
        reverse_old, reverse_new = old == 1 - y, new == 1 - y
        changed = np.any(before != after, axis=1)
        for subset, mask in masks.items():
            count_n = int((mask & changed).sum())
            class_n = int((mask & (new != old)).sum())
            correct_n = int((mask & (correct_new != correct_old)).sum())
            assert correct_n <= class_n <= count_n
            output.append(dict(seed=seed, left=left, right=right, subset=subset, n=int(mask.sum()),
                               count_changed=count_n, class_changed=class_n, correctness_changed=correct_n,
                               left_only_correct=int((mask & correct_new & ~correct_old).sum()),
                               right_only_correct=int((mask & correct_old & ~correct_new).sum()),
                               left_opposite=int((mask & reverse_new).sum()), right_opposite=int((mask & reverse_old).sum())))
    return output


def block_summary(seed, rows):
    anti = {case: rows[seed, 'KEY', case]['counterfactual_joint'] for case in KEY_CONDITIONS}
    true = {case: rows[seed, 'KEY', case]['disagree_joint_correct'] for case in KEY_CONDITIONS}
    d_selected = anti['swap_direction'] - anti['nominal']
    d_other = anti['swap_read'] - anti['swap_magnitude']
    n_selected = anti['swap_magnitude'] - anti['nominal']
    n_other = anti['swap_read'] - anti['swap_direction']
    assert d_other - d_selected == n_other - n_selected
    row = dict(seed=seed, equal_norm_direction_effect=(d_selected + d_other) / 2,
               selected_direction_minus_matched_mean=true['nominal'] - true['mean_matched'],
               direction_at_selected_norm=d_selected, direction_at_other_norm=d_other,
               norm_at_selected_direction=n_selected, norm_at_other_direction=n_other,
               direction_norm_interaction=d_other - d_selected,
               selected_direction_true_joint_norm_effect=true['swap_magnitude'] - true['nominal'],
               other_direction_true_joint_norm_effect=true['swap_read'] - true['swap_direction'],
               mean_norm_matching_gain=true['mean_matched'] - true['mean_read'],
               selected_minus_raw_mean=true['nominal'] - true['mean_read'], natural_swap_gain=anti['swap_read'] - anti['nominal'])
    for case in KEY_CONDITIONS:
        row[f'{case}_true_joint'], row[f'{case}_opposite_joint'] = true[case], anti[case]
        row[f'{case}_disagree_ba'] = rows[seed, 'KEY', case]['disagree_ba']
    for arm in ('KEY_MEAN', 'GAIN2'):
        row[f'{arm}_disagree_ba'] = rows[seed, arm, 'nominal']['disagree_ba']
    return row


def effect_statistics(block_rows):
    blocks = len(block_rows)
    samples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, blocks, (BOOTSTRAP_DRAWS, blocks))
    output = {}
    for key in PRIMARY + SECONDARY:
        values = np.array([row[key] for row in block_rows])
        item = dict(effect=key, blocks=blocks, values=values.tolist(), mean=float(values.mean()),
                    sample_sd=float(values.std(ddof=1)), minimum=float(values.min()), maximum=float(values.max()),
                    positive=int((values > 0).sum()), zero=int((values == 0).sum()), negative=int((values < 0).sum()))
        if key in PRIMARY:
            low, high = np.percentile(values[samples].mean(axis=1), PERCENTILES).tolist()
            direction = 'positive' if low > 0 else 'negative' if high < 0 else 'undetermined'
            item.update(ci97_5=[low, high], direction=direction)
        output[key] = item
    return [output[key] for key in PRIMARY], [output[key] for key in SECONDARY]


def verify_primary(effects, audit):
    replay = {item['effect']: item for item in audit['primary_effects']}
    assert set(replay) == set(PRIMARY) and len(audit['primary_effects']) == len(PRIMARY)
    for item in effects:
        other = replay[item['effect']]
        assert all(item[key] == other[key] for key in ('blocks', 'positive', 'zero', 'negative', 'direction'))
        for key in ('values', 'mean', 'sample_sd', 'minimum', 'maximum', 'ci97_5'):
            np.testing.assert_allclose(item[key], other[key], rtol=0, atol=1e-12)


def main():
    freeze, science, audit = (read(name) for name in ('FREEZE.json', 'SCIENCE_SUMMARY.json', 'AUDIT.json'))
    assert science['status'] == 'COMPLETE' and audit['status'] == 'PASS'
    blocks, n = freeze['blocks'], freeze['examples_per_block']
    assert science['blocks'] == blocks == 12 and science['examples_per_block'] == n and n in (128, 256)
    assert science['evaluations'] == audit['evaluations'] == 8 * blocks
    assert science['fits'] == science['updates'] == 0
    assert science['forward_sample_steps'] == 8 * blocks * n * 36
    assert science['dense_forward_macs'] == 156 * science['forward_sample_steps']
    assert science['matched_cases'] == 3 * blocks and science['matched_read_steps'] == 3 * blocks * n * 3
    evaluation = sorted(read('evaluate/ROWS.json'), key=lambda row: (
        row['seed'], tuple(CONDITIONS).index(row['arm']), CONDITIONS[row['arm']].index(row['condition'])))
    rows = {(row['seed'], row['arm'], row['condition']): row for row in evaluation}
    expected = {(seed, arm, condition) for seed in range(blocks) for arm, conditions in CONDITIONS.items()
                for condition in conditions}
    assert len(evaluation) == len(rows) == len(expected) and set(rows) == expected

    block_rows, matched, pairs = [], [], []
    for seed in range(blocks):
        base, masks = base_data(seed, n)
        data = {(arm, condition): load_condition(rows[seed, arm, condition], base, masks)
                for arm, conditions in CONDITIONS.items() for condition in conditions}
        keyed = {case: data['KEY', case] for case in KEY_CONDITIONS}
        for case in KEY_CONDITIONS[1:]:
            np.testing.assert_array_equal(keyed[case]['cue_memory'], keyed['nominal']['cue_memory'])
        diagnostics = match_diagnostics(seed, keyed)
        for row in diagnostics:
            rows[seed, 'KEY', row['condition']].update({key: row[key] for key in (
                'zero_direction', 'zero_target_norm', 'max_abs_l2_residual', 'max_abs_vector_residual')})
        matched.extend(diagnostics)
        pairs.extend(paired_changes(seed, keyed, base['y'], masks))
        block_rows.append(block_summary(seed, rows))
    assert len(matched) == science['matched_cases']
    assert sum(row['read_steps'] for row in matched) == science['matched_read_steps']
    assert sum(row['fallback_reads'] for row in matched) == science['fallback_reads']

    groups = []
    for arm, conditions in CONDITIONS.items():
        for condition in conditions:
            cases = [rows[seed, arm, condition] for seed in range(blocks)]
            metrics = {key: float(np.mean([row[key] for row in cases])) for key in METRICS + RESOURCES + NORM_FIELDS}
            groups.append(dict(arm=arm, condition=condition, blocks=blocks, cost_per_episode=cases[0]['cost_per_episode'], **metrics))
    primary, secondary = effect_statistics(block_rows)
    verify_primary(primary, audit)
    inference = dict(blocks=blocks, examples_per_block=n, bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=BOOTSTRAP_DRAWS,
                     percentiles=PERCENTILES, interval='whole-training-block paired percentile',
                     nominal_family_confidence=.95, independent_primary_match=True)
    result = dict(study_status='FIXED_WEIGHT_ADDRESS_NORM_DIRECTION_CROSSOVER', inference=inference, primary_effects=primary,
                  secondary_effects=secondary, block_rows=block_rows, model_rows=evaluation, group_rows=groups,
                  match_rows=matched, paired_change_rows=pairs,
                  max_abs_l2_residual=max(row['max_abs_l2_residual'] for row in matched),
                  max_abs_vector_residual=max(row['max_abs_vector_residual'] for row in matched))
    (ROOT / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for name, records in (('PRIMARY_EFFECTS.csv', primary), ('BLOCK_RESULTS.csv', block_rows), ('EVAL_ROWS.csv', evaluation)):
        export_csv(name, records)
    print(f'Summarized {blocks} paired blocks, {len(evaluation)} evaluations and {len(matched)} norm matches.')


if __name__ == '__main__':
    main()
