"""Summarize independent evicted-key confirmation on new training blocks."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
GEOMETRIES = ('orthogonal', 'near')
HISTORIES = ('four', 'three0', 'three2', 'two')
RULES = ('original', 'replay', 'guarded')
CONDITIONS = tuple((geometry, history, rule) for geometry in GEOMETRIES for history in HISTORIES for rule in RULES)
SCHEDULES = {'A': 22, 'B': 24, 'C': 20, 'D': 23}
EVICTED = {'three0': 0, 'three2': 1}
PRIMARY = {'evicted_gain': 0., 'recent_change': -.02}
BOOTSTRAP_SEED = 4087900001
BOOTSTRAP_DRAWS = 50_000
PERCENTILES = (1.25, 98.75)
TOLERANCE = 1e-12
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


def episode_masks(selected, history):
    masks = dict(all=np.ones(len(selected), dtype=bool), older=selected < 2, recent=selected >= 2)
    masks.update({f'position{i}': selected == i for i in range(4)})
    if history in EVICTED:
        position = EVICTED[history]
        masks.update(evicted=selected == position, missing=selected == 1 - position)
    elif history == 'four':
        masks['evicted'] = masks['older']
    else:
        masks['missing'] = masks['older']
    return masks


def scores(pred, y, triggered, mask):
    return dict(n=int(mask.sum()), correct=int((mask & (pred == y)).sum()),
                ties=int((mask & (pred < 0)).sum()), triggered=int((mask & triggered).sum()))


def pair_scores(old, guarded, original, y, old_trigger, blocked, mask):
    left, right, baseline = guarded == y, old == y, original == y
    return dict(n=int(mask.sum()), old_triggered=int((mask & old_trigger).sum()),
                blocked=int((mask & blocked).sum()), guard_only_correct=int((mask & left & ~right).sum()),
                replay_only_correct=int((mask & right & ~left).sum()),
                guard_only_vs_original=int((mask & left & ~baseline).sum()),
                original_only_vs_guard=int((mask & baseline & ~left).sum()),
                replay_only_vs_original=int((mask & right & ~baseline).sum()),
                original_only_vs_replay=int((mask & baseline & ~right).sum()),
                prediction_changed=int((mask & (old != guarded)).sum()))


def effect_summary(name, values, resamples):
    values = np.asarray(values, dtype=float)
    low, high = np.percentile(values[resamples].mean(axis=1), PERCENTILES)
    threshold = PRIMARY[name]
    return dict(effect=name, blocks=len(values), values=values.tolist(), mean=float(values.mean()),
                sample_sd=float(values.std(ddof=1)), minimum=float(values.min()), maximum=float(values.max()),
                positive=int((values > 0.).sum()), zero=int((values == 0.).sum()), negative=int((values < 0.).sum()),
                ci97_5=[float(low), float(high)], threshold=threshold, passes=bool(low > threshold),
                direction='positive' if low > 0. else 'negative' if high < 0. else 'undetermined')


def verify_primary(primary, audit):
    independent = {row['effect']: row for row in audit['primary_effects']}
    assert set(independent) == set(PRIMARY) and len(audit['primary_effects']) == len(PRIMARY)
    for effect in primary:
        actual = independent[effect['effect']]
        for key in ('blocks', 'positive', 'zero', 'negative', 'direction', 'threshold', 'passes'):
            assert effect[key] == actual[key], key
        for key in ('values', 'mean', 'sample_sd', 'minimum', 'maximum', 'ci97_5'):
            np.testing.assert_allclose(effect[key], actual[key], atol=TOLERANCE, rtol=0., err_msg=key)


def aggregate(rows, dimensions):
    groups = {}
    for row in rows:
        group = tuple(row[key] for key in dimensions)
        cells = groups.setdefault(group, {})
        for key, value in row.items():
            if key not in dimensions and key != 'seed':
                cells[key] = cells.get(key, 0) + value
    return [dict(zip(dimensions, keys), **values) for keys, values in groups.items()]


def load_path(seed, geometry, history, rule, row, y, selected, read_ticks):
    condition = f'{geometry}_{history}_{rule}'
    assert row['id'] == f's{seed}_LEARN' and row['condition'] == condition
    assert (row['T'], row['q'], row['slots']) == (36, 28, 2)
    raw = load_archive(f'evaluate/{row["id"]}_{condition}.npz')
    counts, reset, onset = (raw[key] for key in ('counts', 'reset_steps', 'read_onset'))
    n = len(y)
    assert counts.shape == (n, 2) and reset.shape == onset.shape == (n, 36)
    assert np.all((0 <= counts) & (counts <= 8)) and np.array_equal(counts, counts.astype(int))
    assert onset.sum() == row['onset_count'] == n and onset[np.arange(n), read_ticks].all()
    assert not np.any(reset & ~onset) and reset.sum() == row['reset_count']
    pred = predictions(counts)
    trigger = reset.any(axis=1)
    masks = episode_masks(selected, history)
    for subset, key in (('all', 'balanced_accuracy'), ('older', 'older_ba'), ('recent', 'recent_ba')):
        assert (pred[masks[subset]] == y[masks[subset]]).mean() == row[key], (seed, condition, key)
    np.testing.assert_array_equal(row['ba_by_position'],
                                  [(pred[mask] == y[mask]).mean() for mask in (masks[f'position{i}'] for i in range(4))])
    np.testing.assert_array_equal(row['reset_by_position'], [trigger[mask].mean() for mask in
                                                            (masks[f'position{i}'] for i in range(4))])
    np.testing.assert_allclose(raw['resources'][:, :4].mean(axis=0), row['mean_resources'], atol=TOLERANCE, rtol=0.)
    if rule != 'original':
        np.testing.assert_array_equal(raw['replay_steps'], read_ticks * trigger)
        np.testing.assert_allclose(raw['replay_resources'].mean(axis=0), row['replay_resources_mean'], atol=TOLERANCE, rtol=0.)
    if history in EVICTED:
        assert (pred[masks['evicted']] == y[masks['evicted']]).mean() == row['evicted_ba']
        assert row['missing_ba'] == .5 * row['missing_coverage']
    return raw, pred, trigger


def collect_block(seed, n, rows, microbench):
    base = load_archive(f'evaluate/test_s{seed}.npz')
    selected, y, group = (base[name] for name in ('selected', 'y', 'group'))
    assert selected.shape == y.shape == group.shape == (n,)
    np.testing.assert_array_equal(y, base['bits'][np.arange(n), selected])
    schedule_masks = {name: group % 4 == index for index, name in enumerate(SCHEDULES)}
    read_ticks = np.array(tuple(SCHEDULES.values()))[group % 4]
    for mask in schedule_masks.values():
        assert mask.sum() == n // 4
        for i in range(4):
            position = mask & (selected == i)
            assert position.sum() == n // 16 and (position & (y == 0)).sum() == n // 32
    outcomes, pairs, differences = [], [], {}
    expected_micro = {}
    for geometry in GEOMETRIES:
        differences[geometry] = {}
        for history in HISTORIES:
            masks = episode_masks(selected, history)
            paths = {rule: load_path(seed, geometry, history, rule, rows[seed, geometry, history, rule], y, selected, read_ticks)
                     for rule in RULES}
            original, old, guard = (paths[rule] for rule in RULES)
            assert not original[2].any() and not np.any(guard[2] & ~old[2])
            blocked = old[2] & ~guard[2]
            for key in ('counts', 'resources'):
                np.testing.assert_array_equal(guard[0][key][~blocked], old[0][key][~blocked])
                np.testing.assert_array_equal(guard[0][key][blocked], original[0][key][blocked])
            if history == 'two':
                assert not old[2].any()
                for key in original[0]:
                    np.testing.assert_array_equal(old[0][key], original[0][key], err_msg=key)
                    np.testing.assert_array_equal(guard[0][key], original[0][key], err_msg=key)
            if history == 'four':
                expected_micro[geometry] = {name: np.flatnonzero(mask & old[2])[:4].tolist()
                                            for name, mask in schedule_masks.items()}
                for row in microbench:
                    if row['geometry'] != geometry:
                        continue
                    sample = row['sample']
                    assert row['replay_steps'] == old[0]['replay_steps'][sample]
                    assert row['guarded_steps'] == guard[0]['replay_steps'][sample]
            for schedule, subset in dict(all=masks['all'], **schedule_masks).items():
                for name, mask in masks.items():
                    mask = mask & subset
                    identity = dict(seed=seed, geometry=geometry, history=history, schedule=schedule, subset=name)
                    for rule, (raw, pred, trigger) in paths.items():
                        steps = int(raw['replay_steps'][mask].sum()) if rule != 'original' else 0
                        outcomes.append(dict(**identity, rule=rule, replay_steps=steps, **scores(pred, y, trigger, mask)))
                    pairs.append(dict(**identity, **pair_scores(old[1], guard[1], original[1], y, old[2], blocked, mask)))
            old_row, guard_row, original_row = (rows[seed, geometry, history, rule] for rule in ('replay', 'guarded', 'original'))
            keys = ('recent_ba', 'older_ba') + (('evicted_ba',) if history in EVICTED else ())
            differences[geometry][history] = {key: dict(guarded_minus_replay=guard_row[key] - old_row[key],
                                                       guarded_minus_original=guard_row[key] - original_row[key],
                                                       replay_minus_original=old_row[key] - original_row[key]) for key in keys}
    verify_microbench(seed, microbench, expected_micro)
    near = differences['near']
    effects = {effect: float(np.mean([near[history][metric]['guarded_minus_replay'] for history in EVICTED]))
               for effect, metric in (('evicted_gain', 'evicted_ba'), ('recent_change', 'recent_ba'))}
    return dict(seed=seed, differences=differences, **effects), outcomes, pairs


def verify_microbench(seed, rows, expected):
    order = RULES[seed % len(RULES):] + RULES[:seed % len(RULES)]
    expected_order = [rule for rule in order if rule != 'original']
    assert len(rows) == sum(len(samples) for geometry in expected.values() for samples in geometry.values())
    for geometry, schedules in expected.items():
        for schedule, samples in schedules.items():
            records = [row for row in rows if (row['geometry'], row['schedule']) == (geometry, schedule)]
            assert [row['sample'] for row in records] == samples
            for row in records:
                assert row['order'] == expected_order
                assert row['replay_steps'] == SCHEDULES[schedule] and row['guarded_steps'] in (0, row['replay_steps'])
                assert np.isfinite(row['guard_seconds']) and row['guard_seconds'] > 0.
                assert np.isfinite(row['replay_seconds']) and row['replay_seconds'] > 0.
                assert np.isfinite(row['guarded_replay_seconds']) and row['guarded_replay_seconds'] >= 0.
                assert bool(row['guarded_replay_seconds']) == bool(row['guarded_steps'])


def cost_summary(rows, blocks):
    grouped = []
    for geometry, history, rule in CONDITIONS:
        cells = [rows[seed, geometry, history, rule] for seed in range(blocks)]
        costs = {key: float(np.mean([cell['cost_per_episode'][key] for cell in cells]))
                 for key in cells[0]['cost_per_episode']}
        means = {key: np.mean([cell[key] for cell in cells], axis=0).tolist()
                 for key in ('mean_resources', 'replay_resources_mean')}
        totals = {key: sum(cell[key] for cell in cells) for key in
                  ('reset_count', 'onset_count', 'replay_batches', 'replay_wall_seconds',
                   'evaluation_wall_seconds', 'guard_wall_seconds', 'guard_batches')}
        grouped.append(dict(geometry=geometry, history=history, rule=rule, **means, **totals,
                            cost_per_episode=costs, replay_scratch_peak_bytes=max(cell['replay_scratch_peak_bytes'] for cell in cells)))
    return grouped


def main():
    freeze, train, science, audit = map(read, ('FREEZE.json', 'TRAIN_SUMMARY.json', 'SCIENCE_SUMMARY.json', 'AUDIT.json'))
    blocks, n = freeze['blocks'], freeze['examples_per_block']
    assert blocks == 12 and n in (256, 512)
    assert train['status'] == science['status'] == 'COMPLETE' and audit['status'] == 'PASS'
    assert science['evaluations'] == audit['evaluations'] == blocks * len(CONDITIONS)
    assert train['blocks'] == train['fits'] == science['fits'] == blocks
    assert train['updates'] == science['updates'] == blocks * 600 and train['examples_per_fit'] == 128
    training_lock, test_start = map(read, ('train/TRAIN_LOCK.json', 'TEST_START.json'))
    assert training_lock['test_generated'] is False and test_start['test_generated'] is False
    assert training_lock['blocks'] == test_start['blocks'] == blocks and training_lock['updates'] == train['updates']
    assert test_start['status'] == 'READY'
    assert test_start['train_lock_sha256'] == science['train_lock_sha256'] == audit['train_lock_sha256']
    training_rows = sorted(read('train/ROWS.json'), key=lambda row: row['seed'])
    assert [row['seed'] for row in training_rows] == list(range(blocks))
    for seed, row in enumerate(training_rows):
        assert row['id'] == f's{seed}_LEARN' and row['arm'] == 'LEARN' and row['examples'] == 128
        assert row['updates'] == 600 and all(np.isfinite(row[key]) for key in ('first_loss', 'last_loss', 'final_loss'))
    steps = science['evaluations'] * n * 36
    assert science['forward_sample_steps'] == steps
    assert science['dense_forward_macs'] == 156 * steps and science['matcher_macs'] == 8 * steps
    assert science['replay_native_macs'] == 156 * science['replay_sample_steps']
    model_rows = sorted(read('evaluate/ROWS.json'), key=lambda row: (
        row['seed'], CONDITIONS.index((row['geometry'], row['history'], row['rule']))))
    rows = {(row['seed'], row['geometry'], row['history'], row['rule']): row for row in model_rows}
    expected = {(seed, *condition) for seed in range(blocks) for condition in CONDITIONS}
    assert len(model_rows) == len(rows) and rows.keys() == expected
    microbench = read('MICROBENCH.json')
    assert len(microbench) == science['microbench_records']
    assert science['microbench_evaluations'] == 2 * len(microbench)
    assert science['microbench_forward_sample_steps'] == 36 * science['microbench_evaluations']
    assert science['microbench_replay_steps'] == sum(row['replay_steps'] + row['guarded_steps'] for row in microbench)
    assert science['microbench_replay_native_macs'] == 156 * science['microbench_replay_steps']
    assert {row['seed'] for row in microbench} <= set(range(blocks))
    block_rows, outcomes, pairs = [], [], []
    for seed in range(blocks):
        row, scores, changes = collect_block(seed, n, rows, [record for record in microbench if record['seed'] == seed])
        row['training'] = training_rows[seed]
        block_rows.append(row)
        outcomes.extend(scores)
        pairs.extend(changes)
    performance = aggregate(outcomes, ('geometry', 'history', 'schedule', 'subset', 'rule'))
    paired = aggregate(pairs, ('geometry', 'history', 'schedule', 'subset'))
    overall = [row for row in performance if row['schedule'] == row['subset'] == 'all']
    assert sum(row['replay_steps'] for row in overall) == science['replay_sample_steps']
    assert sum(row['triggered'] for row in overall) == science['reset_count']
    resamples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, blocks, (BOOTSTRAP_DRAWS, blocks))
    primary = [effect_summary(name, [row[name] for row in block_rows], resamples) for name in PRIMARY]
    verify_primary(primary, audit)
    decision = all(effect['passes'] for effect in primary)
    assert decision == audit['adoption_supported']
    result = dict(study_status='FROZEN_EVICTED_KEY_CONFIRMATION',
                  inference=dict(blocks=blocks, examples_per_block=n, bootstrap_seed=BOOTSTRAP_SEED,
                                 bootstrap_draws=BOOTSTRAP_DRAWS, percentiles=PERCENTILES,
                                 nominal_family_confidence=.95, independent_primary_match=True),
                  decision=dict(both_pass=decision, evicted_pass=primary[0]['passes'], recent_pass=primary[1]['passes']),
                  primary_effects=primary, block_rows=block_rows, model_rows=model_rows, training_rows=training_rows,
                  training_archive=dict(locked_utc=training_lock['utc'], test_start_utc=test_start['utc'],
                                        train_lock_sha256=test_start['train_lock_sha256']),
                  performance_rows=performance, paired_rows=paired, block_paired_rows=pairs,
                  cost_rows=cost_summary(rows, blocks), microbench=microbench)
    (ROOT / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                       encoding='utf-8')
    for name, records in (('PRIMARY_EFFECTS.csv', primary), ('BLOCK_RESULTS.csv', block_rows), ('EVAL_ROWS.csv', model_rows)):
        export_csv(name, records)
    print(f'Summarized {blocks} paired blocks and {len(model_rows)} evicted-key evaluations.')


if __name__ == '__main__':
    main()
