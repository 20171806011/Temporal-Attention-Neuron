"""Summarize audited learned matching, fixed routing, and swap outcomes."""

import csv
import json
from pathlib import Path

import numpy as np

from metrics import score


ROOT = Path(__file__).resolve().parent
ARMS = ('LEARN', 'ORACLE', 'MEAN')
SWAP_ARMS = ('LEARN', 'ORACLE')
ROTATION = np.array([[0., -1.], [1., 0.]])
CONTROLS = {'no_cue': ('disagree_ba', 'disagree_coverage', .75),
            'no_tags': ('disagree_ba', 'disagree_coverage', .75),
            'no_contents': ('balanced_accuracy', 'coverage', .50)}
CONDITIONS = {arm: ('nominal', *CONTROLS, 'swap_read') if arm in SWAP_ARMS else ('nominal', *CONTROLS) for arm in ARMS}
METRICS = ('balanced_accuracy', 'accuracy', 'coverage', 'silence', 'saturated_tie', 'intermediate_tie',
           'agree_ba', 'disagree_ba', 'agree_coverage', 'disagree_coverage', 'disagree_joint_correct',
           'counterfactual_joint', 'counterfactual_disagree_ba', 'order0_disagree_ba', 'order1_disagree_ba',
           'routing_accuracy', 'routing_tie_rate', 'routing_weight_mean', 'routing_inverse_rate')
RESOURCES = ('soma_events', 'output_events', 'recurrent_deliveries', 'output_deliveries')
PRIMARY = ('learn_minus_mean', 'learn_swap_counterfactual_joint_gain')
SECONDARY = ('oracle_minus_learn', 'oracle_minus_mean', 'oracle_swap_counterfactual_joint_gain',
             'learn_swap_counterfactual_ba_gain', 'oracle_swap_counterfactual_ba_gain',
             'order0_learn_minus_mean', 'order1_learn_minus_mean')
BOOTSTRAP_SEED = 4077900001
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
    cost = row['cost_per_episode']
    learned = row['arm'] == 'LEARN'
    expected = dict(parameters=180 if learned else 176, allocated_parameter_slots=180,
                    fixed_match_constants=0 if learned else 4, spiking_cells=8, graded_branches=12,
                    continuous_state_slots=48, history_slots=0, trace_slots=24, key_slots=4,
                    persistent_array_bytes=448, dense_forward_macs=156 * row['T'], matcher_dense_macs=8 * row['T'])
    assert all(cost[key] == value for key, value in expected.items())
    row.update({f'cost_{key}': value for key, value in cost.items()})


def training_row(row, updates):
    seed, arm = row['seed'], row['arm']
    assert row['id'] == f's{seed}_{arm}' and (row['train_delay'], row['T'], row['q']) == (16, 36, 28)
    assert (row['h'], row['loss'], row['lr'], row['updates']) == (6, 'CE', .0025, updates)
    assert np.isfinite(row['seconds']) and row['seconds'] >= 0
    steps = 128 * 36 * updates
    assert (row['dense_forward_macs'], row['dense_backward_macs']) == (156 * steps, 240 * steps)
    assert row['matcher_backward_unit_steps'] == (2 * steps if arm == 'LEARN' else 0)
    assert row['memory_backward_unit_steps'] == 12 * steps
    verify_cost(row)
    with np.load(ROOT / f'train/{row["id"]}.npz', allow_pickle=False) as archive:
        initial, final = archive['initial'], archive['theta']
    assert initial.shape == final.shape == (180,) and np.isfinite(final).all()
    assert np.count_nonzero(initial[-4:]) == 0
    if arm == 'LEARN':
        matrix = final[-4:].reshape(2, 2)
    else:
        assert np.count_nonzero(final[-4:]) == 0
        matrix = ROTATION if arm == 'ORACLE' else np.zeros((2, 2))
    norm = float(np.linalg.norm(matrix))
    cosine = float(np.sum(matrix * ROTATION) / (norm * np.sqrt(2.))) if norm else None
    matrix_info = dict(seed=seed, arm=arm, matrix=matrix.tolist(), frobenius_norm=norm, cosine_to_rotation=cosine,
                       active_trainable_parameters=180 if arm == 'LEARN' else 176, stored_parameters=180)
    row.update(A=matrix_info['matrix'], A_frobenius_norm=norm, A_cosine_to_R=cosine,
               active_trainable_parameters=matrix_info['active_trainable_parameters'], stored_parameters=180)
    return initial, matrix_info


def base_data(seed):
    with np.load(ROOT / f'evaluate/test_s{seed}.npz', allow_pickle=False) as archive:
        keys = ('y', 'first_bit', 'second_bit', 'selected', 'order', 'octet')
        base = {key: archive[key] for key in keys}
    assert all(value.shape == (256,) for value in base.values())
    y, first, second, selected, order = (base[key] for key in keys[:-1])
    disagree = first != second
    assert np.array_equal(y, np.where(selected == 0, first, second)) and disagree.sum() == 128
    masks = dict(all=np.ones(256, dtype=bool), disagree=disagree,
                 order0=disagree & (order == 0), order1=disagree & (order == 1))
    for key in ('order0', 'order1'):
        assert masks[key].sum() == 64 and np.count_nonzero(masks[key] & (y == 0)) == 32
    return base, masks


def predict(counts):
    return np.where(counts[:, 0] == counts[:, 1], -1, counts.argmax(axis=1))


def evaluation_row(row, train, base, masks):
    meta = ('id', 'arm', 'seed', 'train_delay', 'h', 'loss', 'lr', 'updates', 'T', 'q')
    assert all(row[key] == train[key] for key in meta) and row['test_delay'] == 16
    verify_cost(row)
    with np.load(ROOT / f'evaluate/{row["id"]}_{row["condition"]}.npz', allow_pickle=False) as archive:
        counts, resources, weights = (archive[key] for key in ('counts', 'resources', 'cue_weights'))
    assert counts.shape == (256, 2) and weights.shape == (256, 3, 2)
    assert resources.shape == (256, 6)
    assert np.all((0 <= counts) & (counts <= 8)) and np.array_equal(counts, counts.astype(int))
    assert np.isfinite(resources).all() and np.all(resources >= 0)
    assert np.all((0 <= weights) & (weights <= 1))
    np.testing.assert_allclose(weights.sum(axis=-1), 1., rtol=0, atol=1e-12)
    np.testing.assert_allclose(resources[:, :4].mean(axis=0), row['mean_resources'], rtol=0, atol=1e-12)
    row.update(zip(RESOURCES, row['mean_resources']))

    values, reverse = score(counts, base), score(counts[:, ::-1], base)
    values.update(counterfactual_joint=reverse['disagree_joint_correct'], counterfactual_disagree_ba=reverse['disagree_ba'])
    correct = predict(counts) == base['y']
    for key in ('order0', 'order1'):
        values[f'{key}_disagree_ba'] = correct[masks[key]].mean()
    true_slot = (base['selected'] == base['order']).astype(int)
    example = np.arange(len(true_slot))
    true_weight, wrong_weight = weights[example, 0, true_slot], weights[example, 0, 1 - true_slot]
    values.update(routing_accuracy=float((true_weight > wrong_weight).mean()),
                  routing_tie_rate=float((true_weight == wrong_weight).mean()), routing_weight_mean=float(true_weight.mean()))
    assert all(np.allclose(row[key], value, rtol=0, atol=1e-12) for key, value in values.items())
    row['routing_inverse_rate'] = float((true_weight < wrong_weight).mean())
    chosen = np.where(true_weight == wrong_weight, -1, weights[:, 0, :].argmax(axis=-1))
    routing = {key: row[key] for key in ('id', 'arm', 'seed', 'condition')}
    routing.update(true_slot=true_slot.tolist(), true_slot_weight=true_weight.tolist(), chosen_slot=chosen.tolist())
    return counts, weights, routing


def negative_control(row):
    ba, coverage, ceiling = CONTROLS[row['condition']]
    residual = row[ba] - .5 * row[coverage]
    assert abs(residual) <= 1e-12 and row['balanced_accuracy'] <= ceiling + 1e-12
    if row['condition'] == 'no_cue':
        assert row['disagree_joint_correct'] == 0
    identifiers = {key: row[key] for key in ('id', 'seed', 'arm', 'condition', 'disagree_joint_correct')}
    return dict(**identifiers, subset_ba_field=ba, subset_coverage_field=coverage,
                subset_ba=row[ba], subset_coverage=row[coverage], residual=residual,
                full_ba=row['balanced_accuracy'], full_ba_ceiling=ceiling)


def swap_details(seed, arm, counts, base, masks):
    old, new = (predict(counts[key]) for key in ('nominal', 'swap_read'))
    y = base['y']
    changed = np.any(counts['nominal'] != counts['swap_read'], axis=1)
    joints = []
    for octet in np.unique(base['octet']):
        for bit in (0, 1):
            mask = (base['octet'] == octet) & masks['disagree'] & (base['first_bit'] == bit)
            assert mask.sum() == 2 and sorted(base['selected'][mask]) == [0, 1]
            order = np.unique(base['order'][mask])
            assert len(order) == 1
            joints.append((int(order[0]), bool(np.all(old[mask] == 1 - y[mask])), bool(np.all(new[mask] == 1 - y[mask])),
                           bool(np.all(old[mask] == y[mask])), bool(np.all((old[mask] >= 0) & (new[mask] == 1 - old[mask])))))
    result = []
    for stratum in ('disagree', 'order0', 'order1'):
        mask = masks[stratum]
        chosen = [row for row in joints if stratum == 'disagree' or row[0] == int(stratum[-1])]
        assert 2 * len(chosen) == mask.sum()
        result.append(dict(seed=seed, arm=arm, stratum=stratum, examples=int(mask.sum()), cue_pairs=len(chosen),
                           count_changed=int((mask & changed).sum()), class_changed=int((mask & (old != new)).sum()),
                           nominal_opposite=int((mask & (old == 1 - y)).sum()), swap_opposite=int((mask & (new == 1 - y)).sum()),
                           nominal_opposite_joint=sum(row[1] for row in chosen), swap_opposite_joint=sum(row[2] for row in chosen),
                           nominal_correct_and_swap_opposite_joint=sum(row[2] and row[3] for row in chosen),
                           both_predictions_inverted=sum(row[4] for row in chosen)))
    return result


def block_summary(seed, cases, matrix):
    nominal = {arm: cases[seed, arm, 'nominal'] for arm in ARMS}
    row = dict(seed=seed, learn_A=matrix['matrix'], learn_A_norm=matrix['frobenius_norm'], learn_A_cosine_to_R=matrix['cosine_to_rotation'])
    for arm, entry in nominal.items():
        for key in ('disagree_ba', 'order0_disagree_ba', 'order1_disagree_ba', 'routing_accuracy', 'routing_tie_rate',
                    'routing_inverse_rate', 'routing_weight_mean'):
            row[f'{arm}_{key}'] = entry[key]
    for arm in SWAP_ARMS:
        swapped = cases[seed, arm, 'swap_read']
        for key in ('disagree_ba', 'disagree_joint_correct', 'counterfactual_disagree_ba', 'counterfactual_joint'):
            row[f'{arm}_nominal_{key}'], row[f'{arm}_swap_{key}'] = nominal[arm][key], swapped[key]
        row[f'{arm.lower()}_swap_counterfactual_joint_gain'] = swapped['counterfactual_joint'] - nominal[arm]['counterfactual_joint']
        row[f'{arm.lower()}_swap_counterfactual_ba_gain'] = swapped['counterfactual_disagree_ba'] - nominal[arm]['counterfactual_disagree_ba']
    row.update(learn_minus_mean=nominal['LEARN']['disagree_ba'] - nominal['MEAN']['disagree_ba'],
               oracle_minus_learn=nominal['ORACLE']['disagree_ba'] - nominal['LEARN']['disagree_ba'],
               oracle_minus_mean=nominal['ORACLE']['disagree_ba'] - nominal['MEAN']['disagree_ba'])
    for order in (0, 1):
        key = f'order{order}_disagree_ba'
        row[f'order{order}_learn_minus_mean'] = nominal['LEARN'][key] - nominal['MEAN'][key]
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
    assert science['fits'] == audit['fits'] == 3 * blocks and science['evaluations'] == audit['evaluations'] == 14 * blocks
    assert science['updates'] == audit['optimizer_updates'] == 3 * blocks * updates
    train_steps, eval_steps = 3 * blocks * updates * 128 * 36, 14 * blocks * 256 * 36
    assert science['forward_sample_steps'] == train_steps + eval_steps
    assert science['dense_forward_macs'] == 156 * (train_steps + eval_steps) and science['dense_backward_macs'] == 240 * train_steps
    training = sorted(read('train/ROWS.json'), key=lambda row: (row['seed'], ARMS.index(row['arm'])))
    evaluation = sorted(read('evaluate/ROWS.json'), key=lambda row: (
        row['seed'], ARMS.index(row['arm']), CONDITIONS[row['arm']].index(row['condition'])))
    trains = {(row['seed'], row['arm']): row for row in training}
    cases = {(row['seed'], row['arm'], row['condition']): row for row in evaluation}
    assert len(training) == len(trains) == 3 * blocks and len(evaluation) == len(cases) == 14 * blocks
    assert set(trains) == {(seed, arm) for seed in range(blocks) for arm in ARMS}
    assert set(cases) == {(seed, arm, condition) for seed in range(blocks) for arm in ARMS for condition in CONDITIONS[arm]}

    controls, routing, matrices, swaps, block_rows = [], [], [], [], []
    for seed in range(blocks):
        base, masks = base_data(seed)
        initials, block_matrices = {}, {}
        for arm in ARMS:
            train = trains[seed, arm]
            initials[arm], block_matrices[arm] = training_row(train, updates)
            counts, weights = {}, {}
            for condition in CONDITIONS[arm]:
                row = cases[seed, arm, condition]
                counts[condition], weights[condition], examples = evaluation_row(row, train, base, masks)
                routing.append(examples)
                if condition in CONTROLS:
                    controls.append(negative_control(row))
            if arm in SWAP_ARMS:
                np.testing.assert_allclose(weights['swap_read'], weights['nominal'][:, :, ::-1], rtol=0, atol=1e-12)
                detail = swap_details(seed, arm, counts, base, masks)
                for label, condition in (('nominal', 'nominal'), ('swap', 'swap_read')):
                    assert detail[0][f'{label}_opposite_joint'] / detail[0]['cue_pairs'] == cases[seed, arm, condition]['counterfactual_joint']
                swaps.extend(detail)
        np.testing.assert_array_equal(initials['LEARN'], initials['ORACLE'])
        np.testing.assert_array_equal(initials['LEARN'], initials['MEAN'])
        matrices.extend(block_matrices.values())
        block_rows.append(block_summary(seed, cases, block_matrices['LEARN']))
    assert len(controls) == 9 * blocks

    groups = []
    for arm in ARMS:
        for condition in CONDITIONS[arm]:
            rows = [cases[seed, arm, condition] for seed in range(blocks)]
            means = {key: float(np.mean([row[key] for row in rows])) for key in METRICS + RESOURCES}
            groups.append(dict(arm=arm, condition=condition, blocks=blocks, **means))
    primary, secondary = effect_statistics(block_rows)
    verify_primary(primary, audit)
    inference = dict(blocks=blocks, bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=BOOTSTRAP_DRAWS,
                     percentiles=PERCENTILES, interval='whole-training-block paired percentile',
                     nominal_family_confidence=.95, independent_primary_match=True,
                     both_positive=all(effect['direction'] == 'positive' for effect in primary))
    result = dict(study_status='LEARNED_ROTATED_QUERY_MATCHING', updates_per_fit=updates, inference=inference,
                  primary_effects=primary, secondary_effects=secondary, block_rows=block_rows,
                  train_rows=training, model_rows=evaluation, group_rows=groups, swap_rows=swaps, matrix_rows=matrices,
                  routing_example_rows=routing,
                  negative_controls=dict(rows=controls, max_abs_residual=max(abs(row['residual']) for row in controls)))
    (ROOT / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                       encoding='utf-8')
    for name, rows in (('PRIMARY_EFFECTS.csv', primary), ('BLOCK_RESULTS.csv', block_rows), ('EVAL_ROWS.csv', evaluation),
                       ('TRAIN_ROWS.csv', training), ('NEGATIVE_CONTROLS.csv', controls)):
        export_csv(name, rows)
    print(f'Summarized {blocks} paired blocks, {len(training)} fits and {len(evaluation)} evaluations.')


if __name__ == '__main__':
    main()
