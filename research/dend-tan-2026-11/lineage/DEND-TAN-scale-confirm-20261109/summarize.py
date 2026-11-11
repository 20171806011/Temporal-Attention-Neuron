"""Summarize the independent D16 replication after the replay audit passes."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ARMS = ('GAIN2', 'SCALE', 'SCALE_MEAN')
CONTROLS = {
    'no_cue': ('disagree_ba', 'disagree_coverage', .75),
    'no_candidates': ('balanced_accuracy', 'coverage', .50),
    'no_first': ('select_first_ba', 'select_first_coverage', .75),
    'no_second': ('select_second_ba', 'select_second_coverage', .75),
}
CONDITIONS = ('nominal', *CONTROLS)
METRICS = ('balanced_accuracy', 'accuracy', 'coverage', 'silence', 'saturated_tie',
           'intermediate_tie', 'agree_ba', 'disagree_ba', 'agree_coverage',
           'disagree_coverage', 'select_first_ba', 'select_second_ba',
           'select_first_coverage', 'select_second_coverage', 'disagree_joint_correct')
RESOURCES = ('soma_events', 'output_events', 'recurrent_deliveries', 'output_deliveries')
BOOTSTRAP_SEED = 4071900001
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


def close(a, b):
    return abs(a - b) <= 1e-12


def verify_cost(row):
    trace_slots = 24 if row['arm'] in ('SCALE', 'SCALE_MEAN') else 0
    trace_steps = 12 * row['T'] if trace_slots else 0
    expected = dict(history_slots=0, trace_slots=trace_slots,
                    continuous_state_slots=20 + trace_slots,
                    persistent_array_bytes=8 * (28 + trace_slots),
                    dense_forward_macs=156 * row['T'], graded_transfers=12 * row['T'],
                    history_roll_writes=0, softmax_exponentials=0,
                    trace_tanh_evaluations=trace_steps, trace_logistic_evaluations=trace_steps,
                    trace_retained_multiplies=2 * trace_steps, trace_write_multiplies=trace_steps,
                    trace_update_adds=2 * trace_steps, trace_context_unit_steps=trace_steps)
    cost = row['cost_per_episode']
    assert all(cost[key] == value for key, value in expected.items()), row['id']


def verify_records(training, evaluation, blocks, updates):
    expected_train = {(arm, seed) for arm in ARMS for seed in range(blocks)}
    assert len(training) == len({row['id'] for row in training}) == 3 * blocks
    assert {(row['arm'], row['seed']) for row in training} == expected_train
    for row in training:
        assert (row['train_delay'], row['T'], row['q']) == (16, 36, 28)
        assert row['h'] == 6 and row['loss'] == 'CE' and row['updates'] == updates
        assert close(row['lr'], .0025)
        verify_cost(row)
        steps = 64 * 36 * updates
        assert row['dense_forward_macs'] == 156 * steps
        assert row['dense_backward_macs'] == 240 * steps
        assert row['memory_backward_unit_steps'] == 64 * updates * row['cost_per_episode']['trace_context_unit_steps']

    by_id = {row['id']: row for row in training}
    expected_eval = {(model_id, condition) for model_id in by_id for condition in CONDITIONS}
    assert len(evaluation) == len(expected_eval) == 15 * blocks
    assert {(row['id'], row['condition']) for row in evaluation} == expected_eval
    negative = []
    for row in evaluation:
        train = by_id[row['id']]
        assert all(row[key] == train[key] for key in ('arm', 'seed', 'train_delay', 'T', 'q', 'h', 'loss', 'updates', 'lr'))
        assert row['test_delay'] == 16
        verify_cost(row)
        assert all(np.isfinite(row[key]) and 0 <= row[key] <= 1 for key in METRICS)
        counts = row['prediction_counts']
        assert len(counts) == 3 and sum(counts) == 256 and all(value >= 0 for value in counts)
        assert close(row['coverage'], sum(counts[1:]) / 256)
        assert close(row['balanced_accuracy'], row['accuracy'])
        assert close(row['coverage'] + row['silence'] + row['saturated_tie'] + row['intermediate_tie'], 1)
        assert close(row['balanced_accuracy'], .5 * (row['agree_ba'] + row['disagree_ba']))
        assert close(row['balanced_accuracy'], .5 * (row['select_first_ba'] + row['select_second_ba']))
        assert close(row['coverage'], .5 * (row['agree_coverage'] + row['disagree_coverage']))
        assert row['balanced_accuracy'] <= row['coverage'] + 1e-12
        assert 2 * row['disagree_ba'] - 1 - 1e-12 <= row['disagree_joint_correct'] <= row['disagree_ba'] + 1e-12
        assert len(row['mean_resources']) == len(RESOURCES)
        assert all(np.isfinite(value) and value >= 0 for value in row['mean_resources'])
        row.update(zip(RESOURCES, row['mean_resources']))
        row.update({f'cost_{key}': value for key, value in row['cost_per_episode'].items()})
        if row['condition'] in CONTROLS:
            ba_key, coverage_key, ceiling = CONTROLS[row['condition']]
            residual = row[ba_key] - .5 * row[coverage_key]
            assert close(residual, 0) and row['balanced_accuracy'] <= ceiling + 1e-12
            if row['condition'] == 'no_cue':
                assert row['disagree_joint_correct'] == 0
            identifiers = {key: row[key] for key in ('id', 'arm', 'seed', 'condition')}
            negative.append(dict(**identifiers, subset_ba_field=ba_key, subset_coverage_field=coverage_key,
                                 subset_ba=row[ba_key], subset_coverage=row[coverage_key], residual=residual,
                                 full_ba=row['balanced_accuracy'], full_ba_ceiling=ceiling,
                                 disagree_joint_correct=row['disagree_joint_correct']))
    assert len(negative) == 12 * blocks
    return negative


def summarize(evaluation, blocks):
    index = {(row['arm'], row['seed'], row['condition']): row for row in evaluation}
    groups = []
    for arm in ARMS:
        for condition in CONDITIONS:
            rows = [index[arm, seed, condition] for seed in range(blocks)]
            groups.append(dict(arm=arm, condition=condition, n=blocks,
                               **{key: float(np.mean([row[key] for row in rows])) for key in METRICS + RESOURCES}))

    nominal = {arm: np.array([index[arm, seed, 'nominal']['disagree_ba'] for seed in range(blocks)])
               for arm in ARMS}
    paired = {right: nominal['SCALE'] - nominal[right] for right in ('SCALE_MEAN', 'GAIN2')}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, blocks, (BOOTSTRAP_DRAWS, blocks))
    effects = []
    for right, values in paired.items():
        interval = np.percentile(values[samples].mean(axis=1), PERCENTILES).tolist()
        direction = 'positive' if interval[0] > 0 else 'negative' if interval[1] < 0 else 'undetermined'
        effects.append(dict(left='SCALE', right=right, metric='disagree_ba', n=blocks,
                            values=values.tolist(), mean=float(values.mean()), sample_sd=float(values.std(ddof=1)),
                            minimum=float(values.min()), maximum=float(values.max()),
                            positive=int((values > 0).sum()), zero=int((values == 0).sum()),
                            negative=int((values < 0).sum()), ci97_5=interval, direction=direction))

    block_rows = []
    for seed in range(blocks):
        scores = {f'{arm}_disagree_ba': float(nominal[arm][seed]) for arm in ARMS}
        differences = {f'SCALE_minus_{right}': float(values[seed]) for right, values in paired.items()}
        block_rows.append(dict(seed=seed, **scores, **differences))
    return groups, effects, block_rows


def main():
    freeze, science, audit = (read(name) for name in ('FREEZE.json', 'SCIENCE_SUMMARY.json', 'AUDIT.json'))
    assert science['status'] == 'COMPLETE' and audit['status'] == 'PASS'
    blocks, updates = freeze['blocks'], freeze['updates_per_fit']
    assert blocks in (8, 12) and updates == 600
    assert science['blocks'] == blocks and science['updates_per_fit'] == updates
    assert science['fits'] == audit['fits'] == 3 * blocks
    assert science['evaluations'] == audit['evaluations'] == 15 * blocks
    assert science['updates'] == audit['optimizer_updates'] == 3 * blocks * updates
    train_steps, eval_steps = 3 * blocks * updates * 64 * 36, 15 * blocks * 256 * 36
    assert science['forward_sample_steps'] == train_steps + eval_steps
    assert science['dense_forward_macs'] == 156 * (train_steps + eval_steps)
    assert science['dense_backward_macs'] == 240 * train_steps
    training = sorted(read('train/ROWS.json'), key=lambda row: (row['seed'], ARMS.index(row['arm'])))
    evaluation = sorted(read('evaluate/ROWS.json'), key=lambda row: (
        row['seed'], ARMS.index(row['arm']), CONDITIONS.index(row['condition'])))
    negative = verify_records(training, evaluation, blocks, updates)
    groups, effects, block_rows = summarize(evaluation, blocks)
    inference = dict(blocks=blocks, bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=BOOTSTRAP_DRAWS,
                     percentiles=PERCENTILES, interval='whole-block paired percentile',
                     nominal_family_confidence=.95, both_positive=all(row['direction'] == 'positive' for row in effects),
                     prior_blocks_pooled=False)
    result = dict(study_status='INDEPENDENT_D16_CONFIRMATION', updates_per_fit=updates,
                  inference=inference, train_rows=training, model_rows=evaluation, group_rows=groups,
                  block_rows=block_rows, primary_effects=effects,
                  negative_controls=dict(status='PASS', rows=negative,
                                         max_abs_residual=max(abs(row['residual']) for row in negative)))
    (ROOT / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                       encoding='utf-8')
    export_csv('TRAIN_ROWS.csv', training)
    export_csv('EVAL_ROWS.csv', evaluation)
    export_csv('GROUP_METRICS.csv', groups)
    export_csv('BLOCK_RESULTS.csv', block_rows)
    export_csv('PRIMARY_EFFECTS.csv', effects)
    export_csv('NEGATIVE_CONTROLS.csv', negative)
    print(f'Summarized {blocks} matched blocks, {len(training)} fits and {len(evaluation)} evaluations.')


if __name__ == '__main__':
    main()
