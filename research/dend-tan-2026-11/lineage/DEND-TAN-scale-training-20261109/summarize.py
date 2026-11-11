"""Summarize the independently audited training and evaluation rule crossover."""

import csv
import json
from pathlib import Path

import numpy as np

from metrics import score


ROOT = Path(__file__).resolve().parent
TRAIN_RULES = dict(DYN='dynamic', DETACH='dynamic', MEAN='mean')
ARMS = tuple(TRAIN_RULES)
EVAL_RULES = ('mean', 'dynamic')
TRAIN_PAIRS = (('DYN', 'MEAN'), ('DYN', 'DETACH'), ('DETACH', 'MEAN'))
UPDATE_NUMBERS = (1, 50, 100, 200, 400, 600)
METRICS = ('balanced_accuracy', 'accuracy', 'coverage', 'silence', 'saturated_tie', 'intermediate_tie',
           'agree_ba', 'disagree_ba', 'agree_coverage', 'disagree_coverage', 'select_first_ba',
           'select_second_ba', 'select_first_coverage', 'select_second_coverage', 'disagree_joint_correct')
RESOURCES = ('soma_events', 'output_events', 'recurrent_deliveries', 'output_deliveries')
PRIMARY = ('dyn_minus_mean_at_mean', 'dyn_minus_detach_at_mean')
SECONDARY = ('dyn_swap', 'detach_swap', 'mean_swap', 'dyn_minus_mean_at_dynamic', 'dyn_minus_detach_at_dynamic',
             'dyn_vs_mean_interaction', 'dyn_vs_detach_interaction', 'dyn_minus_mean_in_rule',
             'detach_minus_mean_in_rule')
BOOTSTRAP_SEED = 4074900001
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
                             if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def base_data(name, examples):
    with np.load(ROOT / name, allow_pickle=False) as archive:
        base = {key: archive[key] for key in ('y', 'selected', 'first_bit', 'second_bit', 'octet')}
    assert all(value.shape == (examples,) for value in base.values())
    y, first, second, cue = (base[key] for key in ('y', 'first_bit', 'second_bit', 'selected'))
    disagree = first != second
    assert np.array_equal(y, np.where(cue == 0, first, second))
    assert (y == 0).sum() == examples // 2 and disagree.sum() == examples // 2
    assert np.count_nonzero(disagree & (y == 0)) == examples // 4
    return base, disagree


def verify_cost(row):
    units = 12 * row['T']
    expected = dict(parameters=176, spiking_cells=8, graded_branches=12, trace_slots=24, history_slots=0,
                    continuous_state_slots=44, persistent_array_bytes=416, dense_forward_macs=156 * row['T'],
                    graded_transfers=units, trace_tanh_evaluations=units, trace_logistic_evaluations=units,
                    trace_retained_multiplies=2 * units, trace_write_multiplies=units,
                    trace_update_adds=2 * units, trace_context_unit_steps=units)
    cost = row['cost_per_episode']
    assert all(cost[key] == value for key, value in expected.items()), row['id']
    row.update({f'cost_{key}': value for key, value in cost.items()})


def load_training(row, base, updates):
    seed, arm = row['seed'], row['arm']
    assert row['id'] == f's{seed}_{arm}' and (row['train_delay'], row['T'], row['q']) == (16, 36, 28)
    assert row['train_rule'] == TRAIN_RULES[arm] and row['detached'] is (arm == 'DETACH')
    assert row['h'] == 6 and row['loss'] == 'CE' and row['lr'] == .0025 and row['updates'] == updates
    assert np.isfinite(row['seconds']) and row['seconds'] >= 0
    steps = 64 * 36 * updates
    assert (row['dense_forward_macs'], row['dense_backward_macs']) == (156 * steps, 240 * steps)
    assert row['memory_backward_unit_steps'] == 12 * steps
    assert row['compatibility_backward_unit_steps'] == (0 if arm == 'DETACH' else 12 * steps)
    verify_cost(row)
    with np.load(ROOT / f'train/{row["id"]}.npz', allow_pickle=False) as archive:
        initial, final, gradients, counts, history = (archive[key] for key in ('initial', 'theta', 'gradients', 'counts', 'history'))
    assert initial.shape == final.shape == (176,)
    assert gradients.shape == (updates, 176) and counts.shape == (updates, 64, 2) and history.shape == (updates, 4)
    assert all(np.isfinite(value).all() for value in (initial, final, gradients, counts, history))
    assert counts.dtype == np.uint8 and counts.max() <= 8
    curve = []
    for update in UPDATE_NUMBERS:
        k = update - 1
        metrics = score(counts[k], base)
        curve.append(dict(id=row['id'], seed=seed, arm=arm, update=update, count_timing='before_update',
                          loss=float(history[k, 0]), disagree_ba=metrics['disagree_ba'], coverage=metrics['coverage'],
                          disagree_coverage=metrics['disagree_coverage'], balanced_accuracy=metrics['balanced_accuracy']))
    return dict(initial=initial, final=final, first_gradient=gradients[0].copy(), first_counts=counts[0].copy()), curve


def load_evaluation(row, train, base):
    identity = ('id', 'seed', 'arm', 'train_rule', 'detached', 'train_delay', 'h', 'lr', 'loss', 'updates', 'T', 'q')
    assert all(row[key] == train[key] for key in identity)
    assert row['eval_rule'] in EVAL_RULES and row['test_delay'] == 16 and row['condition'] == 'nominal'
    verify_cost(row)
    with np.load(ROOT / f'evaluate/{row["id"]}_{row["eval_rule"]}.npz', allow_pickle=False) as archive:
        counts, resources = archive['counts'], archive['resources']
    assert counts.shape == (256, 2) and resources.shape == (256, 6)
    assert np.all((0 <= counts) & (counts <= 8)) and np.array_equal(counts, counts.astype(int))
    assert np.isfinite(resources).all() and np.all(resources >= 0)
    np.testing.assert_allclose(resources[:, :4].mean(axis=0), row['mean_resources'], rtol=0, atol=1e-12)
    assert all(np.allclose(row[key], value, rtol=0, atol=1e-12) for key, value in score(counts, base).items())
    row.update(zip(RESOURCES, row['mean_resources']))
    return counts


def paired_distances(seed, trains):
    for arm in ARMS[1:]:
        np.testing.assert_array_equal(trains['DYN']['initial'], trains[arm]['initial'])
    np.testing.assert_array_equal(trains['DYN']['first_counts'], trains['DETACH']['first_counts'])
    rows = []
    for left, right in TRAIN_PAIRS:
        a, b = trains[left], trains[right]
        grad, weights = a['first_gradient'] - b['first_gradient'], a['final'] - b['final']
        first_count_changed = np.any(a['first_counts'] != b['first_counts'], axis=1)
        rows.append(dict(seed=seed, left=left, right=right, first_train_count_changed=int(first_count_changed.sum()),
                         first_clipped_gradient_l2=float(np.linalg.norm(grad)), first_clipped_gradient_linf=float(np.abs(grad).max()),
                         final_weight_l2=float(np.linalg.norm(weights)), final_weight_linf=float(np.abs(weights).max())))
    return rows


def predict(counts):
    return np.where(counts[:, 0] == counts[:, 1], -1, counts.argmax(axis=1))


def paired_changes(seed, left, right, comparison, counts, y, masks):
    left_counts, right_counts = counts[left], counts[right]
    left_class, right_class = predict(left_counts), predict(right_counts)
    left_good, right_good = left_class == y, right_class == y
    count_change = np.any(left_counts != right_counts, axis=1)
    output = []
    for subset, mask in masks.items():
        gains = int(np.count_nonzero(mask & left_good & ~right_good))
        losses = int(np.count_nonzero(mask & right_good & ~left_good))
        n = int(mask.sum())
        count_n = int(np.count_nonzero(mask & count_change))
        class_n = int(np.count_nonzero(mask & (left_class != right_class)))
        assert gains + losses <= class_n <= count_n
        output.append(dict(seed=seed, comparison=comparison, left_arm=left[0], left_rule=left[1], right_arm=right[0],
                           right_rule=right[1], subset=subset, examples=n, count_changed=count_n, class_changed=class_n,
                           correctness_changed=gains + losses, left_only_correct=gains, right_only_correct=losses,
                           accuracy_difference=(gains - losses) / n))
    return output


def block_summary(seed, evaluations):
    rates = {(arm, rule): evaluations[seed, arm, rule]['disagree_ba'] for arm in ARMS for rule in EVAL_RULES}
    row = dict(seed=seed, **{f'{arm}_{rule}_disagree_ba': value for (arm, rule), value in rates.items()})
    for arm in ARMS:
        row[f'{arm.lower()}_swap'] = rates[arm, 'dynamic'] - rates[arm, 'mean']
        row[f'{arm}_in_rule_disagree_ba'] = rates[arm, TRAIN_RULES[arm]]
    for right in ('MEAN', 'DETACH'):
        arm = right.lower()
        for rule in EVAL_RULES:
            row[f'dyn_minus_{arm}_at_{rule}'] = rates['DYN', rule] - rates[right, rule]
        row[f'dyn_vs_{arm}_interaction'] = row[f'dyn_minus_{arm}_at_dynamic'] - row[f'dyn_minus_{arm}_at_mean']
        assert row[f'dyn_vs_{arm}_interaction'] == row['dyn_swap'] - row[f'{arm}_swap']
    row['dyn_minus_mean_in_rule'] = rates['DYN', 'dynamic'] - rates['MEAN', 'mean']
    row['detach_minus_mean_in_rule'] = rates['DETACH', 'dynamic'] - rates['MEAN', 'mean']
    return row


def effect_statistics(block_rows):
    blocks = len(block_rows)
    samples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, blocks, (BOOTSTRAP_DRAWS, blocks))
    results = {}
    for key in PRIMARY + SECONDARY:
        values = np.array([row[key] for row in block_rows])
        item = dict(effect=key, blocks=blocks, values=values.tolist(), mean=float(values.mean()),
                    sample_sd=float(values.std(ddof=1)), minimum=float(values.min()), maximum=float(values.max()),
                    positive=int((values > 0).sum()), zero=int((values == 0).sum()), negative=int((values < 0).sum()))
        if key in PRIMARY:
            low, high = np.percentile(values[samples].mean(axis=1), PERCENTILES).tolist()
            direction = 'positive' if low > 0 else 'negative' if high < 0 else 'undetermined'
            item.update(ci97_5=[low, high], direction=direction)
        results[key] = item
    return [results[key] for key in PRIMARY], [results[key] for key in SECONDARY]


def verify_primary_replay(effects, audit):
    independent = {effect['effect']: effect for effect in audit['primary_effects']}
    assert set(independent) == set(PRIMARY) and len(audit['primary_effects']) == len(PRIMARY)
    for effect in effects:
        replay = independent[effect['effect']]
        assert all(effect[key] == replay[key] for key in ('blocks', 'positive', 'zero', 'negative', 'direction'))
        for key in ('values', 'mean', 'sample_sd', 'minimum', 'maximum', 'ci97_5'):
            np.testing.assert_allclose(effect[key], replay[key], rtol=0, atol=1e-12)


def mean_row(rows, metrics):
    return {key: float(np.mean([row[key] for row in rows])) for key in metrics}


def main():
    freeze, science, audit = (read(name) for name in ('FREEZE.json', 'SCIENCE_SUMMARY.json', 'AUDIT.json'))
    assert science['status'] == 'COMPLETE' and audit['status'] == 'PASS'
    blocks, updates = freeze['blocks'], freeze['updates_per_fit']
    assert science['blocks'] == blocks and blocks in (8, 12) and science['updates_per_fit'] == updates == 600
    assert science['fits'] == audit['fits'] == 3 * blocks and science['evaluations'] == audit['evaluations'] == 6 * blocks
    assert science['updates'] == audit['optimizer_updates'] == 3 * blocks * updates
    train_steps, eval_steps = 3 * blocks * updates * 64 * 36, 6 * blocks * 256 * 36
    assert science['forward_sample_steps'] == train_steps + eval_steps
    assert science['dense_forward_macs'] == 156 * (train_steps + eval_steps) and science['dense_backward_macs'] == 240 * train_steps
    training = sorted(read('train/ROWS.json'), key=lambda row: (row['seed'], ARMS.index(row['arm'])))
    evaluation = sorted(read('evaluate/ROWS.json'), key=lambda row: (
        row['seed'], ARMS.index(row['arm']), EVAL_RULES.index(row['eval_rule'])))
    trains = {(row['seed'], row['arm']): row for row in training}
    evaluations = {(row['seed'], row['arm'], row['eval_rule']): row for row in evaluation}
    assert len(training) == len(trains) == 3 * blocks and len(evaluation) == len(evaluations) == 6 * blocks
    assert set(trains) == {(seed, arm) for seed in range(blocks) for arm in ARMS}
    assert set(evaluations) == {(seed, arm, rule) for seed in range(blocks) for arm in ARMS for rule in EVAL_RULES}

    learning, distances, pairs, block_rows = [], [], [], []
    for seed in range(blocks):
        train_base, _ = base_data(f'train/base_s{seed}.npz', 64)
        test_base, disagree = base_data(f'evaluate/test_s{seed}.npz', 256)
        masks = dict(all=np.ones(256, dtype=bool), disagree=disagree)
        train_arrays, counts = {}, {}
        for arm in ARMS:
            train_row = trains[seed, arm]
            train_arrays[arm], curve = load_training(train_row, train_base, updates)
            learning.extend(curve)
            for rule in EVAL_RULES:
                counts[arm, rule] = load_evaluation(evaluations[seed, arm, rule], train_row, test_base)
        distances.extend(paired_distances(seed, train_arrays))
        block_rows.append(block_summary(seed, evaluations))
        comparisons = [((arm, 'dynamic'), (arm, 'mean'), 'fixed_eval_swap') for arm in ARMS]
        comparisons += [((left, rule), (right, rule), 'common_rule') for left, right in TRAIN_PAIRS[:2] for rule in EVAL_RULES]
        comparisons += [((left, TRAIN_RULES[left]), (right, TRAIN_RULES[right]), 'in_rule')
                        for left, right in TRAIN_PAIRS if TRAIN_RULES[left] != TRAIN_RULES[right]]
        for left, right, comparison in comparisons:
            changes = paired_changes(seed, left, right, comparison, counts, test_base['y'], masks)
            ba_difference = evaluations[(seed, *left)]['disagree_ba'] - evaluations[(seed, *right)]['disagree_ba']
            assert changes[1]['accuracy_difference'] == ba_difference
            pairs.extend(changes)

    groups = [dict(arm=arm, train_rule=TRAIN_RULES[arm], eval_rule=rule, blocks=blocks,
                   **mean_row([evaluations[seed, arm, rule] for seed in range(blocks)], METRICS + RESOURCES))
              for arm in ARMS for rule in EVAL_RULES]
    learning_groups = [dict(arm=arm, update=update, **mean_row(
        [row for row in learning if row['arm'] == arm and row['update'] == update],
        ('loss', 'disagree_ba', 'coverage', 'disagree_coverage', 'balanced_accuracy')))
        for arm in ARMS for update in UPDATE_NUMBERS]
    primary, secondary = effect_statistics(block_rows)
    verify_primary_replay(primary, audit)
    inference = dict(blocks=blocks, bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=BOOTSTRAP_DRAWS,
                     percentiles=PERCENTILES, interval='whole-training-block paired percentile',
                     nominal_family_confidence=.95, independent_primary_match=True)
    result = dict(study_status='DYNAMIC_COMPATIBILITY_TRAINING_INTERVENTION', updates_per_fit=updates, inference=inference,
                  primary_effects=primary, secondary_effects=secondary, block_rows=block_rows,
                  train_rows=training, model_rows=evaluation, group_rows=groups, learning_rows=learning,
                  learning_group_rows=learning_groups, path_distance_rows=distances, paired_change_rows=pairs)
    (ROOT / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                       encoding='utf-8')
    for name, rows in (('PRIMARY_EFFECTS.csv', primary), ('BLOCK_RESULTS.csv', block_rows), ('EVAL_ROWS.csv', evaluation),
                       ('TRAIN_ROWS.csv', training), ('LEARNING_CURVES.csv', learning)):
        export_csv(name, rows)
    print(f'Summarized {blocks} paired blocks, {len(training)} fits and {len(evaluation)} evaluations.')


if __name__ == '__main__':
    main()
