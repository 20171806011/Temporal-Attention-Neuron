"""Summarize the frozen comparisons from complete, audited native records."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2')
TRAIN_DELAYS = (0, 16)
TEST_DELAYS = (0, 16, 32)
BLOCKS = (0, 1, 2)
CONTROLS = ('no_cue', 'no_candidates', 'no_first', 'no_second')
CONTROL_SUBSETS = {
    'no_cue': ('disagree_ba', 'disagree_coverage', .75),
    'no_candidates': ('balanced_accuracy', 'coverage', .50),
    'no_first': ('select_first_ba', 'select_first_coverage', .75),
    'no_second': ('select_second_ba', 'select_second_coverage', .75),
}
METRICS = ('balanced_accuracy', 'accuracy', 'coverage', 'silence', 'saturated_tie',
           'intermediate_tie', 'agree_ba', 'disagree_ba', 'agree_coverage',
           'disagree_coverage', 'select_first_ba', 'select_second_ba',
           'select_first_coverage', 'select_second_coverage', 'disagree_joint_correct')
RESOURCES = ('soma_events', 'output_events', 'recurrent_deliveries', 'output_deliveries')
BOOTSTRAP_SEED = 4041900001
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 3, (BOOTSTRAP_DRAWS, 3))


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                        allow_nan=False) + '\n', encoding='utf-8')


def export_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (ROOT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def block_statistics(values):
    x = np.asarray(values, dtype=float)
    assert x.shape == (3,) and np.isfinite(x).all()
    interval = np.quantile(x[BOOTSTRAP].mean(axis=1), [.025, .975])
    return dict(values=x.tolist(), mean=float(x.mean()), ci95=interval.tolist(), n=3)


def expected_conditions(train_delay):
    return [(d, 'nominal') for d in TEST_DELAYS] + [(train_delay, c) for c in CONTROLS]


def close(a, b):
    return abs(a-b) <= 1e-12


def verify(training, evaluation, updates):
    expected_train = {(arm, delay, seed) for arm in ARMS for delay in TRAIN_DELAYS for seed in BLOCKS}
    assert len(training) == len({row['id'] for row in training}) == 24
    assert {(r['arm'], r['train_delay'], r['seed']) for r in training} == expected_train
    by_id = {row['id']: row for row in training}
    for row in training:
        delay = row['train_delay']
        assert row['T'] == 20+delay and row['q'] == 12+delay
        assert row['updates'] == updates and row['h'] == 6 and row['loss'] == 'CE'
        assert close(row['lr'], .0025)
    expected_eval = {(row['id'], delay, condition) for row in training
                     for delay, condition in expected_conditions(row['train_delay'])}
    assert len(evaluation) == len(expected_eval) == 168
    assert {(r['id'], r['test_delay'], r['condition']) for r in evaluation} == expected_eval
    negative = []
    for row in evaluation:
        train = by_id[row['id']]
        assert all(row[key] == train[key] for key in ('arm', 'train_delay', 'seed', 'lr', 'h', 'loss', 'updates'))
        assert row['T'] == 20+row['test_delay'] and row['q'] == 12+row['test_delay']
        assert all(np.isfinite(row[key]) and 0 <= row[key] <= 1 for key in METRICS)
        assert len(row['mean_resources']) == len(RESOURCES)
        assert len(row['prediction_counts']) == 3 and sum(row['prediction_counts']) == 256
        assert close(row['coverage'], sum(row['prediction_counts'][1:])/256)
        assert close(row['balanced_accuracy'], row['accuracy'])
        assert close(row['coverage']+row['silence']+row['saturated_tie']+row['intermediate_tie'], 1)
        assert close(row['balanced_accuracy'], .5*(row['agree_ba']+row['disagree_ba']))
        assert close(row['balanced_accuracy'], .5*(row['select_first_ba']+row['select_second_ba']))
        assert close(row['coverage'], .5*(row['agree_coverage']+row['disagree_coverage']))
        assert row['balanced_accuracy'] <= row['coverage']+1e-12
        assert row['disagree_joint_correct'] <= row['disagree_ba']+1e-12
        assert row['disagree_joint_correct'] >= 2*row['disagree_ba']-1-1e-12
        if row['condition'] in CONTROL_SUBSETS:
            ba_key, coverage_key, ceiling = CONTROL_SUBSETS[row['condition']]
            residual = row[ba_key]-.5*row[coverage_key]
            assert abs(residual) <= 1e-12 and row['balanced_accuracy'] <= ceiling+1e-12
            if row['condition'] == 'no_cue':
                assert row['disagree_joint_correct'] == 0
            record = {key: row[key] for key in ('id', 'arm', 'seed', 'train_delay', 'test_delay', 'condition')}
            negative.append(dict(**record, subset_ba_field=ba_key, subset_coverage_field=coverage_key,
                                 subset_ba=row[ba_key], subset_coverage=row[coverage_key],
                                 residual=residual, full_ba=row['balanced_accuracy'], full_ba_ceiling=ceiling))
        row.update(zip(RESOURCES, row['mean_resources']))
        row.update({f'cost_{key}': value for key, value in row['cost_per_episode'].items()})
    assert len(negative) == 96
    return dict(status='PASS', rows=negative, max_abs_residual=max(abs(r['residual']) for r in negative))


def summarize(evaluation):
    index = {(r['arm'], r['train_delay'], r['test_delay'], r['condition'], r['seed']): r for r in evaluation}
    groups = []
    for arm in ARMS:
        for train_delay in TRAIN_DELAYS:
            for test_delay, condition in expected_conditions(train_delay):
                rows = [index[arm, train_delay, test_delay, condition, seed] for seed in BLOCKS]
                groups.append(dict(arm=arm, train_delay=train_delay, test_delay=test_delay,
                                   condition=condition, ids=[r['id'] for r in rows],
                                   metrics={key: block_statistics([r[key] for r in rows])
                                            for key in METRICS+RESOURCES}))

    differences = {}
    for delay in TRAIN_DELAYS:
        differences[delay] = np.array([index['FULL', delay, delay, 'nominal', seed]['disagree_ba']
                                       - index['NOATT', delay, delay, 'nominal', seed]['disagree_ba']
                                       for seed in BLOCKS])
    effects = [dict(contrast='full_minus_noatt_d16', metric='disagree_ba',
                    **block_statistics(differences[16])),
               dict(contrast='d16_minus_d0_full_minus_noatt', metric='disagree_ba',
                    **block_statistics(differences[16]-differences[0]))]
    return groups, effects, block_statistics(differences[0])


def diagnostics(training, updates):
    rows = []
    snapshots = sorted({0, 1, 2, 5, 10, 25, 50, 100, 200, updates//2, updates-1})
    for train in training:
        ident, seed = train['id'], train['seed']
        with np.load(ROOT / 'train' / f'base_s{seed}.npz') as base:
            y = base['y']
            disagree = base['first_bit'] != base['second_bit']
        with np.load(ROOT / 'train' / f'{ident}.npz') as saved:
            counts, history = saved['counts'], saved['history']
        assert counts.shape == (updates, 64, 2) and history.shape == (updates, 4)
        assert np.all((0 <= counts) & (counts <= 8)) and np.isfinite(history).all()
        for k in snapshots:
            c = counts[k]
            tied = c[:, 0] == c[:, 1]
            pred = np.where(tied, -1, c.argmax(axis=1))
            correct = pred == y
            quiet, saturated = (c == 0).all(1), (c == 8).all(1)
            rows.append(dict(id=ident, arm=train['arm'], train_delay=train['train_delay'], seed=seed,
                             completed_updates=k, next_update=k+1, balanced_accuracy=float(correct.mean()),
                             disagree_ba=float(correct[disagree].mean()), coverage=float((~tied).mean()),
                             disagree_coverage=float((~tied[disagree]).mean()), silence=float(quiet.mean()),
                             saturated_tie=float(saturated.mean()), intermediate_tie=float((tied & ~quiet & ~saturated).mean()),
                             query_output_spikes_mean=float(c.sum(axis=1).mean()), loss=float(history[k, 0]),
                             gradient_norm=float(history[k, 1]), task_gradient_norm=float(history[k, 2]),
                             regularizer_gradient_norm=float(history[k, 3])))
    return rows


def flatten_group(group):
    row = {key: value for key, value in group.items() if key != 'metrics'}
    for key, stat in group['metrics'].items():
        row.update({f'{key}_mean': stat['mean'], f'{key}_ci95_low': stat['ci95'][0],
                    f'{key}_ci95_high': stat['ci95'][1]})
        row.update({f'{key}_s{i}': value for i, value in enumerate(stat['values'])})
    return row


def main():
    science, audit = read('SCIENCE_SUMMARY.json'), read('AUDIT.json')
    assert science['status'] == 'COMPLETE' and audit['status'] == 'PASS'
    assert science['fits'] == 24 and science['evaluations'] == 168
    updates = science['updates_per_fit']
    assert updates in (400, 600) and science['updates'] == 24*updates
    training = sorted(read('train/ROWS.json'), key=lambda r: (r['train_delay'], ARMS.index(r['arm']), r['seed']))
    evaluation = sorted(read('evaluate/ROWS.json'), key=lambda r: (
        r['train_delay'], ARMS.index(r['arm']), r['seed'], r['condition'] != 'nominal', r['test_delay'], r['condition']))
    negative = verify(training, evaluation, updates)
    groups, effects, short_difference = summarize(evaluation)
    detail = diagnostics(training, updates)
    matched = [r for r in evaluation if r['condition'] == 'nominal' and r['train_delay'] == r['test_delay']]
    assert len(groups) == 56 and len(matched) == 24
    result = dict(study_status='EXPLORATORY_LATE_CUE_RETRAINING', updates_per_fit=updates,
                  train_rows=training, model_rows=evaluation, group_rows=groups, primary_effects=effects,
                  matched_nominal_rows=matched, full_minus_noatt_d0=short_difference, negative_controls=negative,
                  inference=dict(blocks=3, bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=BOOTSTRAP_DRAWS,
                                 interval='pointwise descriptive percentile interval across three paired training blocks'),
                  diagnostic_rows=len(detail),
                  diagnostic_timing='Saved training counts at index k precede update k+1. Test uses the final weight.')
    write_json('RESULTS.json', result)
    write_json('DIAGNOSTICS.json', detail)
    export_csv('metrics.csv', evaluation)
    export_csv('group_summary.csv', [flatten_group(group) for group in groups])
    export_csv('primary_effects.csv', effects)
    export_csv('diagnostics.csv', detail)
    print(f'Summarized {len(training)} fits, {len(evaluation)} evaluations, {len(negative["rows"])} information controls.')


if __name__ == '__main__':
    main()
