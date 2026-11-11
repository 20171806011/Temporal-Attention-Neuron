"""Summarize the audited fixed-budget address-binding study."""

import csv
import json
from pathlib import Path

import numpy as np

from metrics import score


ROOT = Path(__file__).resolve().parent
ARMS = ('KEY', 'KEY_MEAN', 'GAIN2')
CONTROLS = {'no_cue': ('disagree_ba', 'disagree_coverage', .75),
            'no_tags': ('disagree_ba', 'disagree_coverage', .75),
            'no_contents': ('balanced_accuracy', 'coverage', .50)}
CONDITIONS = {arm: ('nominal', *CONTROLS, 'swap_read') if arm == 'KEY' else ('nominal', *CONTROLS)
              for arm in ARMS}
METRICS = ('balanced_accuracy', 'accuracy', 'coverage', 'silence', 'saturated_tie', 'intermediate_tie',
           'agree_ba', 'disagree_ba', 'agree_coverage', 'disagree_coverage', 'disagree_joint_correct',
           'counterfactual_joint', 'counterfactual_disagree_ba', 'order0_disagree_ba', 'order1_disagree_ba')
RESOURCES = ('soma_events', 'output_events', 'recurrent_deliveries', 'output_deliveries')
PRIMARY = ('key_minus_mean', 'swap_counterfactual_joint_gain')
SECONDARY = ('key_minus_gain2', 'key_mean_minus_gain2', 'order0_key_minus_mean', 'order1_key_minus_mean',
             'swap_counterfactual_disagree_ba_gain')
BOOTSTRAP_SEED = 4075900001
BOOTSTRAP_DRAWS = 50_000
PERCENTILES = (1.25, 98.75)


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


def verify_cost(row):
    slots = 24 if row['arm'] in ('KEY', 'KEY_MEAN') else 0
    expected = dict(parameters=176, spiking_cells=8, graded_branches=12, history_slots=0,
                    trace_slots=slots, continuous_state_slots=20 + slots, persistent_array_bytes=8 * (28 + slots),
                    dense_forward_macs=156 * row['T'], graded_transfers=12 * row['T'])
    cost = row['cost_per_episode']
    assert all(cost[key] == value for key, value in expected.items()), row['id']
    row.update({f'cost_{key}': value for key, value in cost.items()})


def verify_train(row, updates):
    assert row['id'] == f"s{row['seed']}_{row['arm']}"
    assert (row['train_delay'], row['T'], row['q']) == (16, 36, 28)
    assert (row['h'], row['loss'], row['lr'], row['updates']) == (6, 'CE', .0025, updates)
    assert np.isfinite(row['seconds']) and row['seconds'] >= 0
    steps = 128 * 36 * updates
    assert (row['dense_forward_macs'], row['dense_backward_macs']) == (156 * steps, 240 * steps)
    assert row['memory_backward_unit_steps'] == (0 if row['arm'] == 'GAIN2' else 12 * steps)
    verify_cost(row)


def base_data(seed):
    with np.load(ROOT / f'evaluate/test_s{seed}.npz', allow_pickle=False) as archive:
        keys = ('y', 'first_bit', 'second_bit', 'selected', 'order', 'octet')
        base = {key: archive[key] for key in keys}
    assert all(value.shape == (256,) for value in base.values())
    y, first, second, cue, order = (base[key] for key in keys[:-1])
    disagree = first != second
    assert np.array_equal(y, np.where(cue == 0, first, second))
    assert np.unique(order).tolist() == [0, 1] and disagree.sum() == 128
    assert np.count_nonzero(disagree & (order == 0)) == 64
    return base, disagree


def prediction(counts):
    return np.where(counts[:, 0] == counts[:, 1], -1, counts.argmax(axis=1))


def evaluate_row(row, train, base, disagree):
    meta = ('id', 'arm', 'seed', 'train_delay', 'h', 'loss', 'lr', 'updates', 'T', 'q')
    assert all(row[key] == train[key] for key in meta) and row['test_delay'] == 16
    verify_cost(row)
    with np.load(ROOT / f'evaluate/{row["id"]}_{row["condition"]}.npz', allow_pickle=False) as archive:
        counts, resources = archive['counts'], archive['resources']
    assert counts.shape == (256, 2) and resources.shape == (256, 6)
    assert np.all((0 <= counts) & (counts <= 8)) and np.array_equal(counts, counts.astype(int))
    assert np.isfinite(resources).all() and np.all(resources >= 0)
    np.testing.assert_allclose(resources[:, :4].mean(axis=0), row['mean_resources'], rtol=0, atol=1e-12)
    row.update(zip(RESOURCES, row['mean_resources']))

    values = score(counts, base)
    reverse = score(counts[:, ::-1], base)
    values.update(counterfactual_joint=reverse['disagree_joint_correct'], counterfactual_disagree_ba=reverse['disagree_ba'])
    y = base['y']
    correct = prediction(counts) == y
    for order in (0, 1):
        subset = disagree & (base['order'] == order)
        values[f'order{order}_disagree_ba'] = np.mean([correct[subset & (y == label)].mean() for label in (0, 1)])
    assert all(np.allclose(row[key], value, rtol=0, atol=1e-12) for key, value in values.items())
    assert np.isclose(row['disagree_ba'], .5 * (row['order0_disagree_ba'] + row['order1_disagree_ba']), rtol=0, atol=1e-12)
    return counts


def negative_control(row):
    ba_key, coverage_key, ceiling = CONTROLS[row['condition']]
    residual = row[ba_key] - .5 * row[coverage_key]
    assert abs(residual) <= 1e-12 and row['balanced_accuracy'] <= ceiling + 1e-12
    if row['condition'] == 'no_cue':
        assert row['disagree_joint_correct'] == 0
    fields = {key: row[key] for key in ('id', 'seed', 'arm', 'condition', 'disagree_joint_correct')}
    return dict(**fields, subset_ba_field=ba_key, subset_coverage_field=coverage_key,
                subset_ba=row[ba_key], subset_coverage=row[coverage_key], residual=residual,
                full_ba=row['balanced_accuracy'], full_ba_ceiling=ceiling)


def swap_details(seed, nominal_counts, swap_counts, base, disagree):
    y, order = base['y'], base['order']
    nominal, swapped = prediction(nominal_counts), prediction(swap_counts)
    opposite_nominal, opposite_swap = nominal == 1 - y, swapped == 1 - y
    inverted = (nominal >= 0) & (swapped == 1 - nominal)
    changed = np.any(nominal_counts != swap_counts, axis=1)
    joints = []
    for octet in np.unique(base['octet']):
        for bit in (0, 1):
            subset = (base['octet'] == octet) & disagree & (base['first_bit'] == bit)
            assert subset.sum() == 2 and sorted(base['selected'][subset]) == [0, 1]
            orders = np.unique(order[subset])
            assert len(orders) == 1
            joints.append((int(orders[0]), bool(opposite_nominal[subset].all()), bool(opposite_swap[subset].all()),
                           bool((nominal[subset] == y[subset]).all()), bool(inverted[subset].all())))
    output = []
    for stratum, mask in (('all_disagree', disagree), ('order0', disagree & (order == 0)), ('order1', disagree & (order == 1))):
        selected = [record for record in joints if stratum == 'all_disagree' or record[0] == int(stratum[-1])]
        n = int(mask.sum())
        assert len(selected) * 2 == n
        output.append(dict(seed=seed, stratum=stratum, receivers=n, cue_pairs=len(selected),
                           count_changed=int((mask & changed).sum()), class_changed=int((mask & (nominal != swapped)).sum()),
                           nominal_opposite=int((mask & opposite_nominal).sum()), swap_opposite=int((mask & opposite_swap).sum()),
                           both_predictions_inverted=sum(row[4] for row in selected),
                           nominal_opposite_joint=sum(row[1] for row in selected), swap_opposite_joint=sum(row[2] for row in selected),
                           nominal_correct_and_swap_opposite_joint=sum(row[2] and row[3] for row in selected)))
    return output


def block_summary(seed, by_case):
    nominal = {arm: by_case[seed, arm, 'nominal'] for arm in ARMS}
    swapped = by_case[seed, 'KEY', 'swap_read']
    row = dict(seed=seed)
    for arm, values in nominal.items():
        for key in ('disagree_ba', 'order0_disagree_ba', 'order1_disagree_ba'):
            row[f'{arm}_{key}'] = values[key]
    for key in ('disagree_ba', 'disagree_joint_correct', 'counterfactual_disagree_ba', 'counterfactual_joint'):
        row[f'KEY_nominal_{key}'], row[f'KEY_swap_{key}'] = nominal['KEY'][key], swapped[key]
    row.update(key_minus_mean=nominal['KEY']['disagree_ba'] - nominal['KEY_MEAN']['disagree_ba'],
               key_minus_gain2=nominal['KEY']['disagree_ba'] - nominal['GAIN2']['disagree_ba'],
               key_mean_minus_gain2=nominal['KEY_MEAN']['disagree_ba'] - nominal['GAIN2']['disagree_ba'],
               swap_counterfactual_joint_gain=swapped['counterfactual_joint'] - nominal['KEY']['counterfactual_joint'],
               swap_counterfactual_disagree_ba_gain=swapped['counterfactual_disagree_ba'] - nominal['KEY']['counterfactual_disagree_ba'])
    for order in (0, 1):
        field = f'order{order}_disagree_ba'
        row[f'order{order}_key_minus_mean'] = nominal['KEY'][field] - nominal['KEY_MEAN'][field]
    return row


def effect_statistics(block_rows):
    blocks = len(block_rows)
    samples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, blocks, (BOOTSTRAP_DRAWS, blocks))
    result = {}
    for key in PRIMARY + SECONDARY:
        values = np.array([row[key] for row in block_rows])
        item = dict(effect=key, blocks=blocks, values=values.tolist(), mean=float(values.mean()),
                    sample_sd=float(values.std(ddof=1)), minimum=float(values.min()), maximum=float(values.max()),
                    positive=int((values > 0).sum()), zero=int((values == 0).sum()), negative=int((values < 0).sum()))
        if key in PRIMARY:
            low, high = np.percentile(values[samples].mean(axis=1), PERCENTILES).tolist()
            direction = 'positive' if low > 0 else 'negative' if high < 0 else 'undetermined'
            item.update(ci97_5=[low, high], direction=direction)
        result[key] = item
    return [result[key] for key in PRIMARY], [result[key] for key in SECONDARY]


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
    blocks, updates = freeze['blocks'], freeze['updates_per_fit']
    assert science['blocks'] == blocks and blocks in (8, 12) and science['updates_per_fit'] == updates == 600
    assert science['fits'] == audit['fits'] == 3 * blocks and science['evaluations'] == audit['evaluations'] == 13 * blocks
    assert science['updates'] == audit['optimizer_updates'] == 3 * blocks * updates
    train_steps, eval_steps = 3 * blocks * updates * 128 * 36, 13 * blocks * 256 * 36
    assert science['forward_sample_steps'] == train_steps + eval_steps
    assert science['dense_forward_macs'] == 156 * (train_steps + eval_steps) and science['dense_backward_macs'] == 240 * train_steps
    training = sorted(read('train/ROWS.json'), key=lambda row: (row['seed'], ARMS.index(row['arm'])))
    evaluation = sorted(read('evaluate/ROWS.json'), key=lambda row: (
        row['seed'], ARMS.index(row['arm']), CONDITIONS[row['arm']].index(row['condition'])))
    trains = {(row['seed'], row['arm']): row for row in training}
    cases = {(row['seed'], row['arm'], row['condition']): row for row in evaluation}
    assert len(training) == len(trains) == 3 * blocks and len(evaluation) == len(cases) == 13 * blocks
    assert set(trains) == {(seed, arm) for seed in range(blocks) for arm in ARMS}
    assert set(cases) == {(seed, arm, condition) for seed in range(blocks) for arm in ARMS for condition in CONDITIONS[arm]}
    for row in training:
        verify_train(row, updates)

    controls, swaps, block_rows = [], [], []
    for seed in range(blocks):
        base, disagree = base_data(seed)
        raw = {}
        for arm in ARMS:
            for condition in CONDITIONS[arm]:
                row = cases[seed, arm, condition]
                counts = evaluate_row(row, trains[seed, arm], base, disagree)
                if arm == 'KEY' and condition in ('nominal', 'swap_read'):
                    raw[condition] = counts
                if condition in CONTROLS:
                    controls.append(negative_control(row))
        details = swap_details(seed, raw['nominal'], raw['swap_read'], base, disagree)
        block_row = block_summary(seed, cases)
        for condition in ('nominal', 'swap'):
            assert details[0][f'{condition}_opposite_joint'] / details[0]['cue_pairs'] == block_row[f'KEY_{condition}_counterfactual_joint']
            assert details[0][f'{condition}_opposite'] / details[0]['receivers'] == block_row[f'KEY_{condition}_counterfactual_disagree_ba']
        swaps.extend(details)
        block_rows.append(block_row)
    assert len(controls) == 9 * blocks

    groups = []
    for arm in ARMS:
        for condition in CONDITIONS[arm]:
            rows = [cases[seed, arm, condition] for seed in range(blocks)]
            metrics = {key: float(np.mean([row[key] for row in rows])) for key in METRICS + RESOURCES}
            groups.append(dict(arm=arm, condition=condition, blocks=blocks, **metrics))
    primary, secondary = effect_statistics(block_rows)
    verify_primary(primary, audit)
    inference = dict(blocks=blocks, bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=BOOTSTRAP_DRAWS,
                     percentiles=PERCENTILES, interval='whole-training-block paired percentile',
                     nominal_family_confidence=.95, independent_primary_match=True,
                     both_positive=all(effect['direction'] == 'positive' for effect in primary))
    result = dict(study_status='RANDOM_ORDER_ADDRESS_BINDING', updates_per_fit=updates, inference=inference,
                  primary_effects=primary, secondary_effects=secondary, block_rows=block_rows,
                  train_rows=training, model_rows=evaluation, group_rows=groups, swap_rows=swaps,
                  negative_controls=dict(rows=controls, max_abs_residual=max(abs(row['residual']) for row in controls)))
    (ROOT / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                       encoding='utf-8')
    for name, rows in (('PRIMARY_EFFECTS.csv', primary), ('BLOCK_RESULTS.csv', block_rows), ('EVAL_ROWS.csv', evaluation),
                       ('TRAIN_ROWS.csv', training), ('NEGATIVE_CONTROLS.csv', controls)):
        export_csv(name, rows)
    print(f'Summarized {blocks} paired blocks, {len(training)} fits and {len(evaluation)} evaluations.')


if __name__ == '__main__':
    main()
