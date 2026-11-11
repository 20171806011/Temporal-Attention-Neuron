"""Summarize audited donor interventions in paired native spike-count units."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CONDITIONS = ('late_original', 'full_original', 'full_eviction_warm', 'full_own_natural',
              'full_target_flip', 'full_other_flip', 'full_deleted_flip')
PRIMARY = {'own_minus_target': ('full_own_natural', 'full_target_flip'),
           'other_minus_target': ('full_other_flip', 'full_target_flip')}
DESCRIPTIVE = {'own_minus_original': ('full_own_natural', 'full_original'),
               'own_minus_warm': ('full_own_natural', 'full_eviction_warm'),
               'other_minus_own': ('full_other_flip', 'full_own_natural'),
               'deleted_minus_own': ('full_deleted_flip', 'full_own_natural')}
SUBSETS = ('all', 'older', 'recent', 'position0', 'position1', 'position2', 'position3',
           'recent_triggered', 'recent_untriggered', 'recent_y0', 'recent_y1')
FOCUS_SEEDS = (1, 9)
BOOTSTRAP_SEED = 4082900001
BOOTSTRAP_DRAWS = 50_000
PERCENTILES = (1.25, 98.75)
TOLERANCE = 1e-12
RESOURCES = ('soma_events', 'output_events', 'recurrent_deliveries', 'output_deliveries')


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


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def load_block(seed, n, rows):
    folder = ROOT / 'evaluate'
    base = load_archive(folder / f'test_s{seed}.npz')
    y, selected = (base[key].astype(np.int64) for key in ('y', 'selected'))
    assert y.shape == selected.shape == (n,)
    np.testing.assert_array_equal(y, base['bits'][np.arange(n), selected])
    assert np.all((0 <= selected) & (selected < 4)) and np.all((y == 0) | (y == 1))
    cases = {}
    for condition in CONDITIONS:
        row = rows[seed, condition]
        assert row['id'] == f's{seed}_LEARN' and (row['T'], row['q'], row['slots']) == (36, 28, 2)
        raw = load_archive(folder / f'{row["id"]}_{condition}.npz')
        counts, reset = raw['counts'], raw['reset_steps']
        assert counts.shape == (n, 2) and reset.shape == (n, 36) and reset.dtype == np.bool_
        assert np.all((0 <= counts) & (counts <= 8)) and np.array_equal(counts, counts.astype(int))
        assert row['reset_count'] == int(reset.sum()) and row['onset_count'] == n
        assert np.all(reset.sum(axis=1) <= 1)
        np.testing.assert_allclose(raw['resources'][:, :4].mean(axis=0), row['mean_resources'],
                                   atol=TOLERANCE, rtol=0)
        row.update(zip(RESOURCES, row['mean_resources']))
        cases[condition] = raw

    original, own = (cases[key] for key in ('full_original', 'full_own_natural'))
    triggered = own['reset_steps'].any(axis=1)
    assert not np.any(own['reset_steps'][:, :25]) and not np.any(own['reset_steps'][:, 26:])
    donors = load_archive(folder / f'donors_s{seed}.npz')
    assert set(donors) == {'own', 'target', 'other', 'deleted'}
    donor_before = cases['late_original']['cue_native_before'][:, 0]
    for kind, suffix in (('own', 'own_natural'), ('target', 'target_flip'), ('other', 'other_flip'),
                         ('deleted', 'deleted_flip')):
        donor = donors[kind]
        assert donor.shape == (n,) and np.all((0 <= donor) & (donor < n))
        np.testing.assert_array_equal(cases[f'full_{suffix}']['cue_native_after'][triggered, 0],
                                      donor_before[donor[triggered]])
    for condition in CONDITIONS[2:]:
        raw = cases[condition]
        np.testing.assert_array_equal(raw['reset_steps'], own['reset_steps'])
        np.testing.assert_array_equal(raw['counts'][~triggered], original['counts'][~triggered])
        for field in ('cue_scores', 'cue_weights', 'cue_read_vectors'):
            np.testing.assert_array_equal(raw[field], original[field])
    for field in ('counts', 'resources', 'cue_native_before', 'cue_native_after'):
        np.testing.assert_array_equal(cases['full_deleted_flip'][field], own[field])
        for condition in ('full_target_flip', 'full_other_flip'):
            np.testing.assert_array_equal(cases[condition][field][selected < 2], own[field][selected < 2])

    masks = dict(all=np.ones(n, dtype=bool), older=selected < 2, recent=selected >= 2)
    masks.update({f'position{i}': selected == i for i in range(4)})
    for i in range(4):
        mask = masks[f'position{i}']
        assert mask.sum() == n // 4 and (mask & (y == 0)).sum() == n // 8
    masks.update(recent_triggered=masks['recent'] & triggered, recent_untriggered=masks['recent'] & ~triggered,
                 recent_y0=masks['recent'] & (y == 0), recent_y1=masks['recent'] & (y == 1))
    return cases, y, masks


def subset_statistics(raw, y, mask):
    counts = raw['counts'][mask].astype(np.int64)
    labels = y[mask]
    signed = counts[:, 1] - counts[:, 0]
    aligned = (2 * labels - 1) * signed
    record = dict(n=len(labels), correct=int((aligned > 0).sum()), ties=int((aligned == 0).sum()),
                  predicted_0=int((signed < 0).sum()), predicted_1=int((signed > 0).sum()),
                  triggered=int(raw['reset_steps'][mask].any(axis=1).sum()), margin_sum=int(aligned.sum()),
                  count_0_sum=int(counts[:, 0].sum()), count_1_sum=int(counts[:, 1].sum()))
    for label in (0, 1):
        chosen = labels == label
        record.update({f'n_y{label}': int(chosen.sum()), f'correct_y{label}': int((aligned[chosen] > 0).sum()),
                       f'signed_sum_y{label}': int(signed[chosen].sum())})
    return add_rates(record)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def add_rates(record):
    n = record['n']
    recalls = [ratio(record[f'correct_y{label}'], record[f'n_y{label}']) for label in (0, 1)]
    return dict(record, ba=sum(recalls) / 2 if None not in recalls else None,
                tie_fraction=ratio(record['ties'], n), trigger_fraction=ratio(record['triggered'], n),
                margin_mean=ratio(record['margin_sum'], n), count_0_mean=ratio(record['count_0_sum'], n),
                count_1_mean=ratio(record['count_1_sum'], n),
                signed_mean_y0=ratio(record['signed_sum_y0'], record['n_y0']),
                signed_mean_y1=ratio(record['signed_sum_y1'], record['n_y1']))


def paired_statistics(left, right, y, mask):
    left_counts, right_counts = (raw['counts'][mask].astype(np.int64) for raw in (left, right))
    left_signed = left_counts[:, 1] - left_counts[:, 0]
    right_signed = right_counts[:, 1] - right_counts[:, 0]
    labels = y[mask]
    delta = (2 * labels - 1) * (left_signed - right_signed)
    return dict(n=len(labels), margin_diff_sum=int(delta.sum()), margin_diff_mean=ratio(int(delta.sum()), len(labels)),
                positive=int((delta > 0).sum()), zero=int((delta == 0).sum()), negative=int((delta < 0).sum()),
                count_changed=int(np.any(left_counts != right_counts, axis=1).sum()),
                prediction_changed=int((np.sign(left_signed) != np.sign(right_signed)).sum()))


def effect_summary(name, values, resamples):
    values = np.asarray(values, dtype=float)
    low, high = np.percentile(values[resamples].mean(axis=1), PERCENTILES)
    return dict(effect=name, blocks=len(values), values=values.tolist(), mean=float(values.mean()),
                sample_sd=float(values.std(ddof=1)), minimum=float(values.min()), maximum=float(values.max()),
                positive=int((values > 0.).sum()), zero=int((values == 0.).sum()), negative=int((values < 0.).sum()),
                ci97_5=[float(low), float(high)], threshold=0., passes=bool(low > 0.),
                direction='positive' if low > 0. else 'negative' if high < 0. else 'undetermined')


def verify_primary(primary, audit):
    audited = {row['effect']: row for row in audit['primary_effects']}
    assert set(audited) == set(PRIMARY) and len(audit['primary_effects']) == len(PRIMARY)
    for effect in primary:
        independent = audited[effect['effect']]
        for key in ('blocks', 'positive', 'zero', 'negative', 'direction', 'threshold', 'passes'):
            assert independent[key] == effect[key], key
        for key in ('values', 'mean', 'sample_sd', 'minimum', 'maximum', 'ci97_5'):
            np.testing.assert_allclose(independent[key], effect[key], atol=TOLERANCE, rtol=0, err_msg=key)


def group_statistics(subset_rows, pairs, rows, blocks):
    sums = ('n', 'correct', 'ties', 'predicted_0', 'predicted_1', 'triggered', 'margin_sum', 'count_0_sum',
            'count_1_sum', 'n_y0', 'n_y1', 'correct_y0', 'correct_y1', 'signed_sum_y0', 'signed_sum_y1')
    grouped = []
    for condition in CONDITIONS:
        path_rows = [rows[seed, condition] for seed in range(blocks)]
        cost = {key: float(np.mean([row['cost_per_episode'][key] for row in path_rows]))
                for key in path_rows[0]['cost_per_episode']}
        resources = {key: float(np.mean([row[key] for row in path_rows])) for key in RESOURCES}
        grouped.append(dict(condition=condition, blocks=blocks, cost_per_episode=cost, **resources,
                            subsets={subset: add_rates({key: sum(subset_rows[seed, condition, subset][key]
                                                                 for seed in range(blocks)) for key in sums})
                                     for subset in SUBSETS}))
    pair_groups = []
    for name in (*PRIMARY, *DESCRIPTIVE):
        for subset in SUBSETS:
            cells = [pairs[seed, name, subset] for seed in range(blocks)]
            totals = {key: sum(cell[key] for cell in cells) for key in
                      ('n', 'margin_diff_sum', 'positive', 'zero', 'negative', 'count_changed', 'prediction_changed')}
            pair_groups.append(dict(effect=name, subset=subset, **totals,
                                    margin_diff_mean=ratio(totals['margin_diff_sum'], totals['n'])))
    return grouped, pair_groups


def main():
    freeze, science, audit = map(read, ('FREEZE.json', 'SCIENCE_SUMMARY.json', 'AUDIT.json'))
    blocks, n = freeze['blocks'], freeze['examples_per_block']
    assert blocks == 12 and n in (256, 512)
    assert science['status'] == 'COMPLETE' and audit['status'] == 'PASS'
    assert science['evaluations'] == audit['evaluations'] == blocks * len(CONDITIONS)
    assert science['fits'] == science['updates'] == 0
    steps = blocks * n * 36
    donor_steps = blocks * (n // 4) * 36
    assert science['forward_sample_steps'] == (len(CONDITIONS) - 1) * steps + donor_steps
    assert science['dense_forward_macs'] == 156 * science['forward_sample_steps']
    assert science['matcher_macs'] == 8 * science['forward_sample_steps']
    assert science['cosine_onsets'] == 5 * blocks * n
    assert science['donor_forward_evaluations'] == blocks and science['donor_forward_sample_steps'] == donor_steps
    assert science['donor_forward_unique_episodes'] == blocks * (n // 4)
    assert science['donor_index_bytes'] == blocks * 4 * n * 8
    assert science['mechanism_diagnostic_evaluations'] == 4 * blocks
    assert science['mechanism_diagnostic_sample_steps'] == 4 * steps
    assert science['warm_reference_count'] == blocks and science['warm_reference_steps'] == 25 * blocks
    model_rows = sorted(read('evaluate/ROWS.json'), key=lambda row: (row['seed'], CONDITIONS.index(row['condition'])))
    rows = {(row['seed'], row['condition']): row for row in model_rows}
    expected = {(seed, condition) for seed in range(blocks) for condition in CONDITIONS}
    assert len(model_rows) == len(rows) and rows.keys() == expected
    subsets, pairs, block_rows = {}, {}, []
    for seed in range(blocks):
        cases, y, masks = load_block(seed, n, rows)
        for condition, raw in cases.items():
            for subset, mask in masks.items():
                subsets[seed, condition, subset] = subset_statistics(raw, y, mask)
            row = rows[seed, condition]
            for subset, key in (('all', 'balanced_accuracy'), ('older', 'older_ba'), ('recent', 'recent_ba')):
                assert subsets[seed, condition, subset]['ba'] == row[key], (seed, condition, key)
                margin_key = 'margin' if subset == 'all' else f'{subset}_margin'
                assert subsets[seed, condition, subset]['margin_mean'] == row[margin_key], (seed, condition, margin_key)
            np.testing.assert_array_equal(row['margin_by_position'],
                                          [subsets[seed, condition, f'position{i}']['margin_mean'] for i in range(4)])
            np.testing.assert_array_equal(row['recent_margin_by_label'],
                                          [subsets[seed, condition, f'recent_y{i}']['margin_mean'] for i in range(2)])
        for name, (left, right) in (PRIMARY | DESCRIPTIVE).items():
            for subset, mask in masks.items():
                pairs[seed, name, subset] = paired_statistics(cases[left], cases[right], y, mask)
        block_rows.append(dict(seed=seed, **{name: pairs[seed, name, 'recent']['margin_diff_mean']
                                             for name in (*PRIMARY, *DESCRIPTIVE)},
                               conditions={condition: {subset: subsets[seed, condition, subset]
                                                       for subset in ('older', 'recent', 'recent_triggered')}
                                           for condition in CONDITIONS}))

    resamples = np.random.default_rng(BOOTSTRAP_SEED).integers(0, blocks, size=(BOOTSTRAP_DRAWS, blocks))
    primary = [effect_summary(name, [row[name] for row in block_rows], resamples) for name in PRIMARY]
    verify_primary(primary, audit)
    decision = all(effect['passes'] for effect in primary)
    assert decision == audit['mechanism_supported']
    group_rows, pair_groups = group_statistics(subsets, pairs, rows, blocks)
    result = dict(study_status='FROZEN_DONOR_CONTENT',
                  inference=dict(blocks=blocks, examples_per_block=n, bootstrap_seed=BOOTSTRAP_SEED,
                                 bootstrap_draws=BOOTSTRAP_DRAWS, percentiles=PERCENTILES,
                                 nominal_family_confidence=.95, independent_primary_match=True),
                  decision=dict(both_pass=decision, target_dependence_supported=decision),
                  primary_effects=primary, block_rows=block_rows,
                  focus_rows=[block_rows[seed] for seed in FOCUS_SEEDS], model_rows=model_rows,
                  group_rows=group_rows, pair_groups=pair_groups,
                  paired_rows=[dict(seed=seed, effect=effect, subset=subset, **record)
                               for (seed, effect, subset), record in pairs.items()],
                  subset_rows=[dict(seed=seed, condition=condition, subset=subset, **record)
                               for (seed, condition, subset), record in subsets.items()])
    (ROOT / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                       encoding='utf-8')
    for name, records in (('PRIMARY_EFFECTS.csv', primary), ('BLOCK_RESULTS.csv', block_rows),
                          ('EVAL_ROWS.csv', model_rows)):
        export_csv(name, records)
    print(f'Summarized {blocks} paired blocks and {len(model_rows)} evaluations in native spike counts.')


if __name__ == '__main__':
    main()
