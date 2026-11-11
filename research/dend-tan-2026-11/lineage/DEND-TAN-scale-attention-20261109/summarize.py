"""Summarize the frozen comparisons from complete, audited native records."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2', 'SCALE', 'SCALE_MEAN')
TRACE_ARMS = ('SCALE', 'SCALE_MEAN')
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
BOOTSTRAP_SEED = 4061900001
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


def verify_cost(row):
    cost = row['cost_per_episode']
    steps = row['T']
    history = 60 if row['arm'] in ('FULL', 'NOATT') else 0
    trace = 24 if row['arm'] in TRACE_ARMS else 0
    trace_units = 12*steps if trace else 0
    expected = dict(
        history_slots=history, trace_slots=trace, continuous_state_slots=20+history+trace,
        persistent_array_bytes=8*(28+history+trace), dense_forward_macs=156*steps,
        graded_transfers=12*steps, history_roll_writes=72*steps if history else 0,
        softmax_exponentials=60*steps if row['arm'] == 'FULL' else 0,
        trace_tanh_evaluations=trace_units, trace_logistic_evaluations=trace_units,
        trace_retained_multiplies=2*trace_units, trace_write_multiplies=trace_units,
        trace_update_adds=2*trace_units, trace_context_unit_steps=trace_units,
    )
    assert all(cost[key] == value for key, value in expected.items()), row['id']


def verify(training, evaluation, updates):
    expected_train = {(arm, delay, seed) for arm in ARMS for delay in TRAIN_DELAYS for seed in BLOCKS}
    assert len(training) == len({row['id'] for row in training}) == 36
    assert {(r['arm'], r['train_delay'], r['seed']) for r in training} == expected_train
    by_id = {row['id']: row for row in training}
    for row in training:
        delay = row['train_delay']
        assert row['T'] == 20+delay and row['q'] == 12+delay
        assert row['updates'] == updates and row['h'] == 6 and row['loss'] == 'CE'
        assert close(row['lr'], .0025)
        verify_cost(row)
        trace_units = 12*row['T']*64*updates if row['arm'] in TRACE_ARMS else 0
        assert row['memory_backward_unit_steps'] == trace_units
    expected_eval = {(row['id'], delay, condition) for row in training
                     for delay, condition in expected_conditions(row['train_delay'])}
    assert len(evaluation) == len(expected_eval) == 252
    assert {(r['id'], r['test_delay'], r['condition']) for r in evaluation} == expected_eval
    negative = []
    for row in evaluation:
        train = by_id[row['id']]
        assert all(row[key] == train[key] for key in ('arm', 'train_delay', 'seed', 'lr', 'h', 'loss', 'updates'))
        assert row['T'] == 20+row['test_delay'] and row['q'] == 12+row['test_delay']
        verify_cost(row)
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
    assert len(negative) == 144
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

    def contrast(left, right, delay):
        return np.array([index[left, delay, delay, 'nominal', seed]['disagree_ba']
                         - index[right, delay, delay, 'nominal', seed]['disagree_ba']
                         for seed in BLOCKS])

    def effect(left, right, delay):
        return dict(contrast=f'{left.lower()}_minus_{right.lower()}_d{delay}',
                    left=left, right=right, delay=delay, metric='disagree_ba',
                    **block_statistics(contrast(left, right, delay)))

    main_pairs = [('SCALE', 'SCALE_MEAN'), ('SCALE', 'FULL')]
    primary = [effect(left, right, 16) for left, right in main_pairs]
    descriptive = [effect(left, right, 0) for left, right in main_pairs]
    for left, right in main_pairs:
        descriptive.append(dict(contrast=f'd16_minus_d0_{left.lower()}_minus_{right.lower()}',
                                left=left, right=right, delay='D16−D0', metric='disagree_ba',
                                **block_statistics(contrast(left, right, 16)-contrast(left, right, 0))))
    descriptive_pairs = [('SCALE_MEAN', 'DEND'), ('SCALE', 'DEND'), ('SCALE', 'NOATT'),
                         ('SCALE', 'GAIN2'), ('FULL', 'NOATT')]
    descriptive += [effect(left, right, delay) for left, right in descriptive_pairs for delay in TRAIN_DELAYS]
    return groups, primary, descriptive


def scale_snapshots(evaluation):
    """Describe the first-eight saved episodes without evaluating the model again."""
    rows = []
    for case in evaluation:
        if case['arm'] not in TRACE_ARMS or case['condition'] != 'nominal':
            continue
        ident, delay = case['id'], case['test_delay']
        with np.load(ROOT / 'evaluate' / f'{ident}_d{delay}_nominal.npz') as archive:
            trace, weight = archive['traces'], archive['weights_fast']
            gate, injection = archive['gates'], archive['injections']
        episodes, steps = trace.shape[:2]
        assert (episodes, steps) == (8, case['T']) and trace.shape[2:] == (12, 2)
        assert weight.shape == injection.shape == (episodes, steps, 12)
        assert gate.shape == (episodes, steps, 1)
        before = np.concatenate((np.zeros_like(trace[:, :1]), trace[:, :-1]), axis=1)
        split = before[..., 0] - before[..., 1]
        # Analytic read difference at the same state; downstream state stays fixed.
        nonuniform_injection = gate*(weight-.5)*split
        identity = {key: case[key] for key in ('id', 'arm', 'seed', 'train_delay', 'test_delay')}
        for stage, selection in [('cue', slice(9+delay, 10+delay)),
                                 ('query', slice(case['q'], case['q']+8))]:
            w = weight[:, selection]
            m = before[:, selection]
            rows.append(dict(**identity, stage=stage, saved_episodes=episodes,
                             fast_weight_mean=float(w.mean()),
                             fast_weight_abs_from_half=float(np.abs(w-.5).mean()),
                             fast_abs_trace_before_read=float(np.abs(m[..., 0]).mean()),
                             slow_abs_trace_before_read=float(np.abs(m[..., 1]).mean()),
                             absolute_trace_difference_before_read=float(np.abs(split[:, selection]).mean()),
                             absolute_injection=float(np.abs(injection[:, selection]).mean()),
                             absolute_injection_vs_uniform_same_state=float(np.abs(nonuniform_injection[:, selection]).mean())))
    assert len(rows) == 72
    return rows


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
    assert science['fits'] == 36 and science['evaluations'] == 252
    updates = science['updates_per_fit']
    assert updates in (400, 600) and science['updates'] == 36*updates
    assert science['forward_sample_steps'] == 18*updates*64*(20+36)+2027520
    training = sorted(read('train/ROWS.json'), key=lambda r: (r['train_delay'], ARMS.index(r['arm']), r['seed']))
    evaluation = sorted(read('evaluate/ROWS.json'), key=lambda r: (
        r['train_delay'], ARMS.index(r['arm']), r['seed'], r['condition'] != 'nominal', r['test_delay'], r['condition']))
    negative = verify(training, evaluation, updates)
    groups, effects, descriptive = summarize(evaluation)
    detail = diagnostics(training, updates)
    snapshots = scale_snapshots(evaluation)
    matched = [r for r in evaluation if r['condition'] == 'nominal' and r['train_delay'] == r['test_delay']]
    assert len(groups) == 84 and len(matched) == 36
    result = dict(study_status='EXPLORATORY_SCALE_ATTENTION_RETRAINING', updates_per_fit=updates,
                  train_rows=training, model_rows=evaluation, group_rows=groups, primary_effects=effects,
                  matched_nominal_rows=matched, descriptive_effects=descriptive, negative_controls=negative,
                  scale_snapshot_rows=snapshots,
                  inference=dict(blocks=3, bootstrap_seed=BOOTSTRAP_SEED, bootstrap_draws=BOOTSTRAP_DRAWS,
                                 interval='pointwise descriptive percentile interval across three paired training blocks'),
                  diagnostic_rows=len(detail),
                  diagnostic_timing='Saved training counts at index k precede update k+1. Test uses the final weight.')
    write_json('RESULTS.json', result)
    write_json('DIAGNOSTICS.json', detail)
    export_csv('metrics.csv', evaluation)
    export_csv('group_summary.csv', [flatten_group(group) for group in groups])
    export_csv('primary_effects.csv', effects)
    export_csv('descriptive_effects.csv', descriptive)
    export_csv('diagnostics.csv', detail)
    export_csv('scale_snapshots.csv', snapshots)
    print(f'Summarized {len(training)} fits, {len(evaluation)} evaluations, {len(negative["rows"])} information controls.')


if __name__ == '__main__':
    main()
