"""Summarize audited implementation equivalence, arithmetic, and paired latency."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
HISTORIES = ('four', 'three0', 'three2', 'two')
MODES = ('full', 'cached', 'sparse')
CONDITIONS = tuple(f'{history}_{mode}' for history in HISTORIES for mode in MODES)
SCHEDULES = ('standard', 'early')
BOOTSTRAP_SEED = 4084900001
BOOTSTRAP_DRAWS = 50_000
ABS_TOLERANCE = 5e-12
RESOURCES = ('soma_events', 'output_events', 'recurrent_deliveries', 'output_deliveries')


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def load_archive(name):
    with np.load(ROOT / name, allow_pickle=False) as archive:
        return dict(archive)


def export_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (ROOT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, allow_nan=False)
                             if isinstance(value, (list, dict)) else value for key, value in row.items()})


def predictions(counts):
    return np.where(counts[:, 0] == counts[:, 1], -1, counts.argmax(axis=1))


def mean_or_none(values):
    return float(np.mean(values)) if len(values) else None


def effect_summary(scope, mode, block_rows, draws):
    values = np.array([row[scope][mode] - row[scope]['full'] for row in block_rows])
    low, high = np.quantile(values[draws].mean(axis=1), [.025, .975])
    return dict(scope=scope, mode=mode, contrast=f'{mode}_minus_full', blocks=len(values), values=values.tolist(),
                mean=float(values.mean()), ci95=[float(low), float(high)],
                minimum=float(values.min()), maximum=float(values.max()),
                faster_blocks=int((values < 0.).sum()), tied_blocks=int((values == 0.).sum()),
                slower_blocks=int((values > 0.).sum()),
                direction='faster' if high < 0. else 'slower' if low > 0. else 'undetermined')


def condition_summary(counts, y, selected, triggered, mask):
    prediction = predictions(counts)
    recent = mask & (selected >= 2)
    older = mask & (selected < 2)
    return dict(examples=int(mask.sum()), triggered=int((mask & triggered).sum()),
                ba=mean_or_none(prediction[mask] == y[mask]), recent_ba=mean_or_none(prediction[recent] == y[recent]),
                older_ba=mean_or_none(prediction[older] == y[older]), coverage=mean_or_none(prediction[mask] >= 0),
                ba_by_position=[mean_or_none(prediction[mask & (selected == i)] == y[mask & (selected == i)])
                                for i in range(4)])


def verify_parity(full, actual):
    for key in ('counts', 'spikes', 'resources', 'read_onset', 'reset_steps', 'eviction_seen'):
        np.testing.assert_array_equal(actual[key], full[key], err_msg=key)
    for key in ('cue_native_before', 'cue_native_after', 'cue_read_vectors'):
        np.testing.assert_allclose(actual[key], full[key], atol=ABS_TOLERANCE, rtol=0., err_msg=key)


def load_block(seed, n, rows):
    base = load_archive(f'evaluate/test_s{seed}.npz')
    y, selected, groups = (base[key] for key in ('y', 'selected', 'group'))
    assert y.shape == selected.shape == groups.shape == (n,)
    np.testing.assert_array_equal(y, base['bits'][np.arange(n), selected])
    standard = groups % 2 == 0
    assert standard.sum() == n // 2
    four_triggered = None
    for history in HISTORIES:
        full = None
        for mode in MODES:
            condition = f'{history}_{mode}'
            row = rows[seed, condition]
            assert row['id'] == f's{seed}_LEARN' and (row['T'], row['q'], row['slots']) == (36, 28, 2)
            assert (row['history'], row['mode']) == (history, mode)
            raw = load_archive(f'evaluate/{row["id"]}_{condition}.npz')
            if mode == 'full':
                full = raw
            else:
                verify_parity(full, raw)
            counts, reset = (raw[key] for key in ('counts', 'reset_steps'))
            assert counts.shape == (n, 2) and reset.shape == (n, 36)
            triggered = reset.any(axis=1)
            assert reset.sum() == row['reset_count'] == triggered.sum()
            assert not triggered.any() or history != 'two'
            assert not np.any(reset[:, :25]) and not np.any(reset[:, 26:])
            if condition == 'four_full':
                four_triggered = triggered
            np.testing.assert_allclose(raw['resources'][:, :4].mean(axis=0), row['mean_resources'], atol=1e-12, rtol=0.)
            observed = condition_summary(counts, y, selected, triggered, np.ones(n, dtype=bool))
            for key in ('recent_ba', 'older_ba', 'coverage', 'ba_by_position'):
                np.testing.assert_array_equal(observed[key], row[key])
            assert observed['ba'] == row['balanced_accuracy']
            for schedule, mask, first in (('standard', standard, 4), ('early', ~standard, 1)):
                summary = condition_summary(counts, y, selected, triggered, mask)
                archived = row['replay_by_schedule'][schedule]
                examples = summary['triggered']
                steps = (25 if mode == 'full' else 25 - first) * examples
                input_rows = 4 * examples if mode == 'sparse' else steps
                assert (archived['episodes'], archived['triggers']) == (summary['examples'], examples)
                assert (archived['steps'], archived['input_rows'], archived['nonzero_steps'], archived['native_macs']) == (
                    steps, input_rows, 4 * examples, 84 * steps + 72 * input_rows)
                summary.update(archived)
                row.setdefault('schedule_metrics', {})[schedule] = summary
    assert four_triggered is not None
    return {schedule: np.flatnonzero(mask & four_triggered)[:8].tolist()
            for schedule, mask in (('standard', standard), ('early', ~standard))}


def block_summary(seed, rows, measurements, expected):
    order = list(MODES[seed % 3:] + MODES[:seed % 3])
    for schedule in SCHEDULES:
        assert [row['sample'] for row in measurements if row['schedule'] == schedule] == expected[schedule]
    assert len(measurements) == sum(len(indices) for indices in expected.values()) and measurements
    for row in measurements:
        assert row['order'] == order and set(row['seconds']) == set(MODES)
        assert all(np.isfinite(value) and value > 0. for value in row['seconds'].values())
    single = {mode: 1000 * float(np.mean([row['seconds'][mode] for row in measurements])) for mode in MODES}
    batch, calls, by_history = {}, {}, {}
    for mode in MODES:
        cells = [rows[seed, f'{history}_{mode}'] for history in HISTORIES]
        calls[mode] = sum(cell['replay_batches'] for cell in cells)
        assert calls[mode] > 0
        batch[mode] = 1000 * sum(cell['replay_wall_seconds'] for cell in cells) / calls[mode]
    for history in HISTORIES:
        by_history[history] = {}
        for mode in MODES:
            cell = rows[seed, f'{history}_{mode}']
            timed = cell['replay_batches']
            by_history[history][mode] = 1000 * cell['replay_wall_seconds'] / timed if timed else None
    return dict(seed=seed, batch=batch, single=single, batch_calls=calls, batch_by_history=by_history,
                single_examples={schedule: len(indices) for schedule, indices in expected.items()})


def schedule_summary(rows, blocks):
    records = []
    for condition in CONDITIONS:
        history, mode = condition.split('_')
        for schedule in SCHEDULES:
            cells = [rows[seed, condition]['schedule_metrics'][schedule] for seed in range(blocks)]
            totals = {key: sum(cell[key] for cell in cells) for key in
                      ('episodes', 'triggers', 'steps', 'input_rows', 'nonzero_steps', 'native_macs')}
            sums = np.asarray([cell['resources'] for cell in cells]).sum(axis=0)
            metrics = {key: float(np.mean([cell[key] for cell in cells]))
                       for key in ('ba', 'recent_ba', 'older_ba', 'coverage')}
            records.append(dict(history=history, mode=mode, schedule=schedule, **totals, **metrics,
                                ba_by_position=np.mean([cell['ba_by_position'] for cell in cells], axis=0).tolist(),
                                resources=sums.tolist()))
    return records


def cost_summary(rows, blocks):
    result = []
    for condition in CONDITIONS:
        history, mode = condition.split('_')
        cells = [rows[seed, condition] for seed in range(blocks)]
        costs = {key: float(np.mean([cell['cost_per_episode'][key] for cell in cells], axis=0))
                 for key in cells[0]['cost_per_episode'] if key != 'replay_resources_mean'}
        costs['replay_resources_mean'] = np.mean([cell['cost_per_episode']['replay_resources_mean']
                                                 for cell in cells], axis=0).tolist()
        result.append(dict(history=history, mode=mode, cost_per_episode=costs,
                           replay_batches=sum(cell['replay_batches'] for cell in cells),
                           replay_wall_seconds=sum(cell['replay_wall_seconds'] for cell in cells),
                           replay_scratch_peak_bytes=max(cell['replay_scratch_peak_bytes'] for cell in cells),
                           online_resources_mean=np.mean([cell['mean_resources'] for cell in cells], axis=0).tolist()))
    return result


def verify_latency(effects, audit):
    independent = {row['effect']: row for row in audit['latency_effects']}
    assert set(independent) == {row['effect'] for row in effects}
    for row in effects:
        for key in ('mean', 'ci95', 'values'):
            np.testing.assert_allclose(row[key], independent[row['effect']][key], atol=1e-12, rtol=0., err_msg=key)


def main():
    freeze, science, audit = map(read, ('FREEZE.json', 'SCIENCE_SUMMARY.json', 'AUDIT.json'))
    blocks, n = freeze['blocks'], freeze['examples_per_block']
    assert blocks == 12 and n in (256, 512)
    assert science['status'] == 'COMPLETE' and audit['status'] == 'PASS' and audit['equivalence_supported']
    assert science['evaluations'] == audit['evaluations'] == blocks * len(CONDITIONS)
    assert science['fits'] == science['updates'] == 0
    assert science['forward_sample_steps'] == science['evaluations'] * n * 36
    assert science['dense_forward_macs'] == 156 * science['forward_sample_steps']
    assert science['matcher_macs'] == 8 * science['forward_sample_steps']
    model_rows = sorted(read('evaluate/ROWS.json'), key=lambda row: (row['seed'], CONDITIONS.index(row['condition'])))
    rows = {(row['seed'], row['condition']): row for row in model_rows}
    expected = {(seed, condition) for seed in range(blocks) for condition in CONDITIONS}
    assert len(model_rows) == len(rows) and rows.keys() == expected
    microbench = read('MICROBENCH.json')
    assert len(microbench) == science['microbench_records']
    assert len(microbench) * len(MODES) == science['microbench_evaluations']
    assert science['microbench_forward_sample_steps'] == 36 * science['microbench_evaluations']
    assert {row['seed'] for row in microbench} == set(range(blocks))
    block_rows = []
    for seed in range(blocks):
        expected_samples = load_block(seed, n, rows)
        block_rows.append(block_summary(seed, rows, [row for row in microbench if row['seed'] == seed], expected_samples))
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, blocks, (BOOTSTRAP_DRAWS, blocks))
    effects = [dict(effect=f'{mode}_{scope}', **effect_summary(scope, mode, block_rows, draws))
               for scope in ('batch', 'single') for mode in MODES[1:]]
    verify_latency(effects, audit)
    schedules = schedule_summary(rows, blocks)
    cost_rows = cost_summary(rows, blocks)
    for mode in MODES:
        cells = [row for row in schedules if row['mode'] == mode]
        for key in ('triggers', 'steps', 'input_rows', 'nonzero_steps', 'native_macs'):
            assert sum(row[key] for row in cells) == science['mode_totals'][mode][key]
        np.testing.assert_array_equal(np.sum([row['resources'] for row in cells], axis=0),
                                      science['mode_totals'][mode]['resources'])
    result = dict(study_status='FROZEN_REPLAY_OPT', equivalence_supported=True,
                  inference=dict(blocks=blocks, examples_per_block=n, bootstrap_seed=BOOTSTRAP_SEED,
                                 bootstrap_draws=BOOTSTRAP_DRAWS, percentiles=[2.5, 97.5],
                                 independent_latency_match=True),
                  latency_effects=effects, block_rows=block_rows, model_rows=model_rows, schedule_rows=schedules,
                  cost_rows=cost_rows, microbench=microbench)
    (ROOT / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                       encoding='utf-8')
    for name, records in (('LATENCY_EFFECTS.csv', effects), ('BLOCK_RESULTS.csv', block_rows), ('EVAL_ROWS.csv', model_rows)):
        export_csv(name, records)
    print(f'Summarized {blocks} paired blocks and {len(model_rows)} equivalent replay paths.')


if __name__ == '__main__':
    main()
