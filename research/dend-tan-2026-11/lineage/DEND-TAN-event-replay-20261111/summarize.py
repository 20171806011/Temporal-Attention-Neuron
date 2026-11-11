"""Summarize the frozen three-write endpoints and the replay parity controls."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
HISTORIES = ('four', 'three0', 'three2', 'two')
RULES = ('original', 'replay', 'dense')
CONDITIONS = tuple(f'{history}_{rule}' for history in HISTORIES for rule in RULES)
EVICTED = {'three0': 0, 'three2': 1}
PRIMARY = {'recent_gain': 0., 'evicted_change': -.02}
FOCUS_SEEDS = (1, 9)
BOOTSTRAP_SEED = 4083900001
BOOTSTRAP_DRAWS = 50_000
PERCENTILES = (1.25, 98.75)
TOLERANCE = 1e-12
RESOURCES = ('soma_events', 'output_events', 'recurrent_deliveries', 'output_deliveries')
REPLAY_RESOURCES = tuple(f'replay_{key}' for key in RESOURCES)
SCALARS = ('balanced_accuracy', 'older_ba', 'recent_ba', 'coverage', 'older_coverage', 'recent_coverage',
           'routing_accuracy', 'older_routing_accuracy', 'recent_routing_accuracy', 'target_weight',
           'older_target_weight', 'recent_target_weight', 'available_fraction', 'reset_fraction',
           'older_reset_fraction', 'recent_reset_fraction', 'eviction_fraction')
VECTORS = ('ba_by_position', 'coverage_by_position', 'routing_by_position', 'target_weight_by_position',
           'reset_by_position')


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


def load_archive(name):
    with np.load(ROOT / name, allow_pickle=False) as archive:
        return dict(archive)


def predictions(counts):
    return np.where(counts[:, 0] == counts[:, 1], -1, counts.argmax(axis=1))


def load_block(seed, n, rows):
    base = load_archive(f'evaluate/test_s{seed}.npz')
    selected, y = (base[name] for name in ('selected', 'y'))
    assert selected.shape == y.shape == (n,)
    np.testing.assert_array_equal(y, base['bits'][np.arange(n), selected])
    masks = dict(all=np.ones(n, dtype=bool), older=selected < 2, recent=selected >= 2)
    masks.update({f'position{i}': selected == i for i in range(4)})
    for i in range(4):
        mask = masks[f'position{i}']
        assert mask.sum() == n // 4 and (mask & (y == 0)).sum() == n // 8

    cases = {}
    for condition in CONDITIONS:
        row = rows[seed, condition]
        assert row['id'] == f's{seed}_LEARN' and (row['T'], row['q'], row['slots']) == (36, 28, 2)
        raw = load_archive(f'evaluate/{row["id"]}_{condition}.npz')
        counts, reset, onset = (raw[key] for key in ('counts', 'reset_steps', 'read_onset'))
        assert counts.shape == (n, 2) and reset.shape == onset.shape == (n, 36)
        assert np.all((0 <= counts) & (counts <= 8)) and np.array_equal(counts, counts.astype(int))
        assert onset.sum() == row['onset_count'] == n and onset[:, 25].all()
        assert not np.any(reset & ~onset) and reset.sum() == row['reset_count']
        np.testing.assert_allclose(raw['resources'][:, :4].mean(axis=0), row['mean_resources'],
                                   atol=TOLERANCE, rtol=0)
        row.update(zip(RESOURCES, row['mean_resources']))
        pred = predictions(counts)
        for subset, key in (('all', 'balanced_accuracy'), ('older', 'older_ba'), ('recent', 'recent_ba')):
            assert (pred[masks[subset]] == y[masks[subset]]).mean() == row[key], (seed, condition, key)
        np.testing.assert_array_equal(row['ba_by_position'],
                                      [(pred[mask] == y[mask]).mean() for mask in (masks[f'position{i}'] for i in range(4))])
        np.testing.assert_array_equal(row['coverage_by_position'],
                                      [(pred[masks[f'position{i}']] >= 0).mean() for i in range(4)])
        history = condition.split('_')[0]
        if history in EVICTED:
            position = EVICTED[history]
            assert row['evicted_ba'] == row['ba_by_position'][position]
            assert row['evicted_coverage'] == row['coverage_by_position'][position]
            assert row['missing_ba'] == row['ba_by_position'][1-position] == .5 * row['missing_coverage']
            assert row['missing_coverage'] == row['coverage_by_position'][1-position]
        else:
            assert all(row[key] is None for key in ('evicted_ba', 'evicted_coverage', 'missing_ba', 'missing_coverage'))
        row.update(zip(REPLAY_RESOURCES, row['replay_resources_mean']))
        if row['rule'] == 'replay':
            np.testing.assert_allclose(raw['replay_resources'].mean(axis=0), row['replay_resources_mean'],
                                       atol=TOLERANCE, rtol=0)
        else:
            assert not any(row['replay_resources_mean']) and row['replay_batches'] == row['replay_wall_seconds'] == 0
        cases[condition] = raw
    return cases, y, masks


def paired_changes(seed, cases, y, masks):
    results = []
    for history in HISTORIES:
        original, replay, dense = (cases[f'{history}_{rule}'] for rule in RULES)
        for field in ('counts', 'resources', 'reset_steps'):
            np.testing.assert_array_equal(replay[field], dense[field])
            if history == 'two':
                np.testing.assert_array_equal(replay[field], original[field])
        triggered = replay['reset_steps'].any(axis=1)
        np.testing.assert_array_equal(replay['counts'][~triggered], original['counts'][~triggered])
        subset_masks = dict(masks, recent_triggered=masks['recent'] & triggered,
                            recent_untriggered=masks['recent'] & ~triggered)
        if history in EVICTED:
            position = EVICTED[history]
            subset_masks.update(evicted=masks[f'position{position}'], absent=masks[f'position{1-position}'])
        left, right = (predictions(raw['counts']) for raw in (replay, original))
        left_correct, right_correct = left == y, right == y
        changed = np.any(replay['counts'] != original['counts'], axis=1)
        for subset, mask in subset_masks.items():
            results.append(dict(seed=seed, history=history, subset=subset, n=int(mask.sum()),
                                count_changed=int((mask & changed).sum()),
                                prediction_changed=int((mask & (left != right)).sum()),
                                replay_only_correct=int((mask & left_correct & ~right_correct).sum()),
                                original_only_correct=int((mask & right_correct & ~left_correct).sum())))
    return results


def block_summary(seed, rows):
    cells = {condition: rows[seed, condition] for condition in CONDITIONS}
    changes = {}
    for history in HISTORIES:
        replay, original = (cells[f'{history}_{rule}'] for rule in ('replay', 'original'))
        fields = ('recent_ba', 'older_ba') + (('evicted_ba',) if history in EVICTED else ())
        changes[history] = {key: replay[key] - original[key] for key in fields}
    return dict(seed=seed, recent_gain=float(np.mean([changes[key]['recent_ba'] for key in EVICTED])),
                evicted_change=float(np.mean([changes[key]['evicted_ba'] for key in EVICTED])),
                changes=changes,
                conditions={condition: {key: cell[key] for key in ('recent_ba', 'older_ba', 'evicted_ba',
                            'recent_reset_fraction', 'recent_coverage')} for condition, cell in cells.items()})


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
    audited = {row['effect']: row for row in audit['primary_effects']}
    assert set(audited) == set(PRIMARY) and len(audit['primary_effects']) == len(PRIMARY)
    for effect in primary:
        independent = audited[effect['effect']]
        for name in ('blocks', 'positive', 'zero', 'negative', 'direction', 'threshold', 'passes'):
            assert effect[name] == independent[name], name
        for name in ('values', 'mean', 'sample_sd', 'minimum', 'maximum', 'ci97_5'):
            np.testing.assert_allclose(effect[name], independent[name], atol=TOLERANCE, rtol=0, err_msg=name)


def group_summary(rows, blocks):
    grouped = []
    for condition in CONDITIONS:
        cells = [rows[seed, condition] for seed in range(blocks)]
        values = {key: float(np.mean([cell[key] for cell in cells]))
                  for key in (*SCALARS, *RESOURCES, *REPLAY_RESOURCES)}
        values.update({key: np.mean([cell[key] for cell in cells], axis=0).tolist() for key in VECTORS})
        values.update({key: sum(cell[key] for cell in cells) for key in
                       ('reset_count', 'onset_count', 'replay_batches', 'replay_wall_seconds')})
        values['replay_scratch_peak_bytes'] = max(cell['replay_scratch_peak_bytes'] for cell in cells)
        for key in ('evicted_ba', 'evicted_coverage', 'missing_ba', 'missing_coverage'):
            values[key] = float(np.mean([cell[key] for cell in cells])) if cells[0][key] is not None else None
        costs = {key: float(np.mean([cell['cost_per_episode'][key] for cell in cells]))
                 for key in cells[0]['cost_per_episode']}
        grouped.append(dict(condition=condition, blocks=blocks, cost_per_episode=costs, **values))
    return grouped


def main():
    freeze, science, audit = map(read, ('FREEZE.json', 'SCIENCE_SUMMARY.json', 'AUDIT.json'))
    blocks, n = freeze['blocks'], freeze['examples_per_block']
    assert blocks == 12 and n in (256, 512)
    assert science['status'] == 'COMPLETE' and audit['status'] == 'PASS'
    assert science['evaluations'] == audit['evaluations'] == blocks * len(CONDITIONS)
    assert science['fits'] == science['updates'] == 0
    steps = science['evaluations'] * n * 36
    assert science['forward_sample_steps'] == steps
    assert science['dense_forward_macs'] == 156 * steps and science['matcher_macs'] == 8 * steps
    assert science['replay_native_macs'] == 156 * science['replay_sample_steps']
    assert science['replay_sample_steps'] == 25 * science['replay_examples']
    assert science['cosine_onsets'] == 6 * blocks * n
    assert science['dense_diagnostic_evaluations'] == len(HISTORIES) * blocks
    assert science['dense_diagnostic_sample_steps'] == len(HISTORIES) * blocks * n * 36
    model_rows = sorted(read('evaluate/ROWS.json'), key=lambda row: (row['seed'], CONDITIONS.index(row['condition'])))
    rows = {(row['seed'], row['condition']): row for row in model_rows}
    expected = {(seed, condition) for seed in range(blocks) for condition in CONDITIONS}
    assert len(model_rows) == len(rows) and rows.keys() == expected
    assert sum(row['reset_count'] for row in model_rows) == science['reset_count']
    assert sum(row['reset_count'] for row in model_rows if row['rule'] == 'replay') == science['replay_examples']
    assert sum(row['replay_batches'] for row in model_rows) == science['replay_batches']
    assert max(row['replay_scratch_peak_bytes'] for row in model_rows) == science['replay_scratch_peak_bytes']
    np.testing.assert_allclose(sum(row['replay_wall_seconds'] for row in model_rows), science['replay_wall_seconds'],
                               atol=TOLERANCE, rtol=0)
    block_rows, changes = [], []
    for seed in range(blocks):
        cases, y, masks = load_block(seed, n, rows)
        changes.extend(paired_changes(seed, cases, y, masks))
        block_rows.append(block_summary(seed, rows))
    resamples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, blocks, size=(BOOTSTRAP_DRAWS, blocks))
    primary = [effect_summary(name, [row[name] for row in block_rows], resamples) for name in PRIMARY]
    verify_primary(primary, audit)
    decision = all(effect['passes'] for effect in primary)
    assert decision == audit['adoption_supported']
    result = dict(study_status='FROZEN_EVENT_REPLAY',
                  inference=dict(blocks=blocks, examples_per_block=n, bootstrap_seed=BOOTSTRAP_SEED,
                                 bootstrap_draws=BOOTSTRAP_DRAWS, percentiles=PERCENTILES,
                                 nominal_family_confidence=.95, independent_primary_match=True),
                  decision=dict(both_pass=decision, recent_pass=primary[0]['passes'], evicted_pass=primary[1]['passes']),
                  primary_effects=primary, block_rows=block_rows, focus_rows=[block_rows[seed] for seed in FOCUS_SEEDS],
                  model_rows=model_rows, group_rows=group_summary(rows, blocks), paired_change_rows=changes)
    (ROOT / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                       encoding='utf-8')
    for name, records in (('PRIMARY_EFFECTS.csv', primary), ('BLOCK_RESULTS.csv', block_rows),
                          ('EVAL_ROWS.csv', model_rows)):
        export_csv(name, records)
    print(f'Summarized {blocks} paired blocks and {len(model_rows)} event-replay evaluations.')


if __name__ == '__main__':
    main()
