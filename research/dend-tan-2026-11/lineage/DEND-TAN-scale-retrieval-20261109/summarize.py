"""Summarize the frozen-weight D16 interventions after independent replay."""

import csv
import json
from pathlib import Path

import numpy as np

from metrics import score


ROOT = Path(__file__).resolve().parent
ARMS = ('GAIN2', 'SCALE', 'SCALE_MEAN')
TRACE_ARMS = ('SCALE', 'SCALE_MEAN')
BASE_CONDITIONS = ('nominal', 'identity', 'native_selected', 'native_unselected')
TRACE_CONDITIONS = ('trace_selected', 'trace_unselected', 'all_selected', 'all_unselected')
CONDITIONS = dict(GAIN2=BASE_CONDITIONS, SCALE=BASE_CONDITIONS + TRACE_CONDITIONS + ('post_mean',),
                  SCALE_MEAN=BASE_CONDITIONS + TRACE_CONDITIONS)
METRICS = ('balanced_accuracy', 'accuracy', 'coverage', 'silence', 'saturated_tie',
           'intermediate_tie', 'agree_ba', 'disagree_ba', 'agree_coverage', 'disagree_coverage',
           'select_first_ba', 'select_second_ba', 'select_first_coverage', 'select_second_coverage',
           'disagree_joint_correct')
RESOURCES = ('soma_events', 'output_events', 'recurrent_deliveries', 'output_deliveries')
COUNT_METRICS = ('mean_count_0', 'mean_count_1', 'mean_label_margin', 'disagree_label_margin',
                 'disagree_margin_change', 'nominal_count_match', 'disagree_nominal_count_match')
BOOTSTRAP_SEED = 4072900001
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


def mean(rows, key):
    return float(np.mean([row[key] for row in rows]))


def prediction(counts):
    return np.where(counts[:, 0] == counts[:, 1], -1, counts.argmax(axis=1))


def block_inputs(seed, examples):
    with np.load(ROOT / f'evaluate/test_s{seed}.npz', allow_pickle=False) as archive:
        keys = ('y', 'selected', 'first_bit', 'second_bit', 'octet')
        base = {key: archive[key] for key in keys}
    with np.load(ROOT / f'evaluate/donors_s{seed}.npz', allow_pickle=False) as archive:
        donors = {side: archive[side] for side in ('selected', 'unselected')}
    rows = np.arange(examples)
    assert all(value.shape == (examples,) for value in (*base.values(), *donors.values()))
    y, first, second, cue = (base[key] for key in ('y', 'first_bit', 'second_bit', 'selected'))
    disagree = first != second
    assert np.array_equal(y, np.where(cue == 0, first, second)) and disagree.sum() == examples // 2
    masks = {'all': disagree, 'label_0_to_1': disagree & (y == 0), 'label_1_to_0': disagree & (y == 1),
             'select_first': disagree & (cue == 0), 'select_second': disagree & (cue == 1)}
    assert all(mask.sum() == examples // 4 for key, mask in masks.items() if key != 'all')
    for side, donor in donors.items():
        assert np.issubdtype(donor.dtype, np.integer) and np.all((0 <= donor) & (donor < examples))
        assert np.array_equal(donor[~disagree], rows[~disagree])
        assert np.array_equal(cue[donor], cue) and np.array_equal(base['octet'][donor], base['octet'])
        assert not disagree[donor].any()
        assert np.array_equal(y[donor], np.where(disagree & (side == 'selected'), 1 - y, y))
        assert np.array_equal((first[donor] != first).astype(int) + (second[donor] != second), disagree)
    return base, donors, masks


def block_conditions(seed, arm, base, donors, disagree, row_index):
    examples = len(disagree)
    rows = np.arange(examples)
    y = base['y']
    counts = {}
    for condition in CONDITIONS[arm]:
        entry = row_index[seed, arm, condition]
        assert entry['id'] == f's{seed}_{arm}' and (entry['T'], entry['q']) == (36, 28)
        with np.load(ROOT / f'evaluate/{entry["id"]}_{condition}.npz', allow_pickle=False) as archive:
            counts[condition] = archive['counts']
            resources = archive['resources']
        values = counts[condition]
        assert values.shape == (examples, 2) and np.all((0 <= values) & (values <= 8))
        assert np.array_equal(values, values.astype(int))
        assert resources.ndim == 2 and resources.shape[0] == examples and resources.shape[1] >= 4
        np.testing.assert_allclose(resources[:, :4].mean(axis=0), entry['mean_resources'], rtol=0, atol=1e-12)
        computed = score(values, base)
        assert all(np.allclose(entry[key], value, atol=1e-12, rtol=0) for key, value in computed.items())
        entry.update(zip(RESOURCES, entry['mean_resources']))
        entry.update({f'cost_{key}': value for key, value in entry['cost_per_episode'].items()})
        cost = entry['cost_per_episode']
        assert (cost['parameters'], cost['dense_forward_macs']) == (176, 36 * 156)
        trace_slots = 24 if arm in TRACE_ARMS else 0
        assert (cost['trace_slots'], cost['continuous_state_slots']) == (trace_slots, 20 + trace_slots)

    baseline = counts['nominal']
    assert np.array_equal(counts['identity'], baseline)
    native_margin = baseline[rows, y].astype(int) - baseline[rows, 1 - y].astype(int)
    for condition, values in counts.items():
        entry = row_index[seed, arm, condition]
        margin = values[rows, y].astype(int) - values[rows, 1 - y].astype(int)
        same = np.all(values == baseline, axis=1)
        entry.update(mean_count_0=float(values[:, 0].mean()), mean_count_1=float(values[:, 1].mean()),
                     mean_label_margin=float(margin.mean()), disagree_label_margin=float(margin[disagree].mean()),
                     disagree_margin_change=float((margin - native_margin)[disagree].mean()),
                     nominal_count_match=float(same.mean()), disagree_nominal_count_match=float(same[disagree].mean()))
        if condition.endswith(('_selected', '_unselected')):
            family, side = condition.split('_')
            donor_equal = np.all(values == baseline[donors[side]], axis=1)
            if family == 'all' or (arm == 'GAIN2' and family == 'native'):
                assert donor_equal.all(), (seed, arm, condition)
            entry.update(donor_count_match=float(donor_equal.mean()),
                         disagree_donor_count_match=float(donor_equal[disagree].mean()))
    return counts


def transfer_rows(seed, arm, base, donors, masks, counts):
    y = base['y']
    nominal = counts['nominal']
    correct = prediction(nominal) == y
    selected_eligible = correct & correct[donors['selected']]
    families = ('native', 'trace', 'all') if arm in TRACE_ARMS else ('native',)
    output = []
    for family in families:
        selected = counts[f'{family}_selected']
        unselected = counts[f'{family}_unselected']
        reverse_selected = prediction(selected) == 1 - y
        reverse_unselected = prediction(unselected) == 1 - y
        match_selected = np.all(selected == nominal[donors['selected']], axis=1)
        match_unselected = np.all(unselected == nominal[donors['unselected']], axis=1)
        for stratum, mask in masks.items():
            n = int(mask.sum())
            n_selected = int((mask & reverse_selected).sum())
            n_unselected = int((mask & reverse_unselected).sum())
            eligible = mask & selected_eligible
            denominator = int(eligible.sum())
            numerator = int((eligible & reverse_selected).sum())
            output.append(dict(seed=seed, arm=arm, family=family, stratum=stratum, receivers=n,
                               selected_reverse_count=n_selected, unselected_reverse_count=n_unselected,
                               selected_reverse_rate=n_selected / n, unselected_reverse_rate=n_unselected / n,
                               transfer_effect=(n_selected - n_unselected) / n,
                               selected_donor_count_match=float(match_selected[mask].mean()),
                               unselected_donor_count_match=float(match_unselected[mask].mean()),
                               conditional_selected_numerator=numerator, conditional_selected_denominator=denominator,
                               conditional_selected_rate=numerator / denominator if denominator else None))
    return output


def group_conditions(evaluation):
    groups = {}
    for row in evaluation:
        groups.setdefault((row['arm'], row['condition']), []).append(row)
    output = []
    for (arm, condition), rows in groups.items():
        stats = {key: mean(rows, key) for key in METRICS + RESOURCES + COUNT_METRICS}
        if 'donor_count_match' in rows[0]:
            stats.update({key: mean(rows, key) for key in ('donor_count_match', 'disagree_donor_count_match')})
        output.append(dict(arm=arm, condition=condition, blocks=len(rows),
                           cost_per_episode=rows[0]['cost_per_episode'], **stats))
    return output


def group_transfers(transfers):
    groups = {}
    for row in transfers:
        groups.setdefault((row['arm'], row['family'], row['stratum']), []).append(row)
    output = []
    for (arm, family, stratum), rows in groups.items():
        stats = {key: mean(rows, key) for key in ('selected_reverse_rate', 'unselected_reverse_rate',
                                                 'transfer_effect', 'selected_donor_count_match',
                                                 'unselected_donor_count_match')}
        numerator = sum(row['conditional_selected_numerator'] for row in rows)
        denominator = sum(row['conditional_selected_denominator'] for row in rows)
        output.append(dict(arm=arm, family=family, stratum=stratum, blocks=len(rows), **stats,
                           conditional_selected_numerator=numerator, conditional_selected_denominator=denominator,
                           conditional_selected_rate_pooled=numerator / denominator if denominator else None,
                           conditional_selected_nonempty_blocks=sum(row['conditional_selected_denominator'] > 0 for row in rows)))
    return output


def primary_effects(block_rows):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    blocks = len(block_rows)
    samples = rng.integers(0, blocks, (BOOTSTRAP_DRAWS, blocks))
    output = []
    for key in ('trace_transfer_effect', 'scale_nominal_minus_post_mean'):
        values = np.array([row[key] for row in block_rows], dtype=float)
        interval = np.percentile(values[samples].mean(axis=1), PERCENTILES).tolist()
        direction = 'positive' if interval[0] > 0 else 'negative' if interval[1] < 0 else 'undetermined'
        output.append(dict(effect=key, blocks=blocks, values=values.tolist(), mean=float(values.mean()),
                           sample_sd=float(values.std(ddof=1)), minimum=float(values.min()), maximum=float(values.max()),
                           positive=int((values > 0).sum()), zero=int((values == 0).sum()), negative=int((values < 0).sum()),
                           ci97_5=interval, direction=direction))
    return output


def main():
    science, audit = read('SCIENCE_SUMMARY.json'), read('AUDIT.json')
    assert science['status'] == 'COMPLETE' and audit['status'] == 'PASS'
    blocks, examples = science['blocks'], science['examples_per_block']
    assert blocks == 12 and examples in (128, 256) and science['fits'] == science['updates'] == 0
    assert science['evaluations'] == audit['evaluations'] == 21 * blocks
    assert science['forward_sample_steps'] == 21 * blocks * examples * 36
    assert science['dense_forward_macs'] == 156 * science['forward_sample_steps']
    evaluation = sorted(read('evaluate/ROWS.json'), key=lambda row: (
        row['seed'], ARMS.index(row['arm']), CONDITIONS[row['arm']].index(row['condition'])))
    index = {(row['seed'], row['arm'], row['condition']): row for row in evaluation}
    expected = {(seed, arm, condition) for seed in range(blocks) for arm in ARMS for condition in CONDITIONS[arm]}
    assert len(evaluation) == len(index) == len(expected) and set(index) == expected

    transfers = []
    for seed in range(blocks):
        base, donors, masks = block_inputs(seed, examples)
        for arm in ARMS:
            counts = block_conditions(seed, arm, base, donors, masks['all'], index)
            transfers.extend(transfer_rows(seed, arm, base, donors, masks, counts))
    block_rows = []
    transfer_index = {(row['seed'], row['arm'], row['family']): row for row in transfers if row['stratum'] == 'all'}
    for seed in range(blocks):
        trace = {arm: transfer_index[seed, arm, 'trace']['transfer_effect'] for arm in TRACE_ARMS}
        nominal = index[seed, 'SCALE', 'nominal']['disagree_ba']
        fixed = index[seed, 'SCALE', 'post_mean']['disagree_ba']
        block_rows.append(dict(seed=seed, SCALE_trace_transfer=trace['SCALE'],
                               SCALE_MEAN_trace_transfer=trace['SCALE_MEAN'],
                               trace_transfer_effect=float(np.mean(list(trace.values()))),
                               SCALE_nominal_disagree_ba=nominal, SCALE_post_mean_disagree_ba=fixed,
                               scale_nominal_minus_post_mean=nominal - fixed))

    groups = group_conditions(evaluation)
    transfer_groups = group_transfers(transfers)
    effects = primary_effects(block_rows)
    assert len(effects) == len(audit['primary_effects'])
    for effect, verified in zip(effects, audit['primary_effects']):
        for key in ('effect', 'blocks', 'positive', 'zero', 'negative', 'direction'):
            assert effect[key] == verified[key], key
        for key in ('values', 'mean', 'minimum', 'maximum', 'ci97_5'):
            np.testing.assert_allclose(effect[key], verified[key], rtol=0, atol=1e-12)
    inference = dict(blocks=blocks, examples_per_block=examples, bootstrap_seed=BOOTSTRAP_SEED,
                     bootstrap_draws=BOOTSTRAP_DRAWS, percentiles=PERCENTILES,
                     interval='whole-training-block paired percentile', nominal_family_confidence=.95)
    result = dict(study_status='FIXED_WEIGHT_D16_RETRIEVAL_INTERVENTIONS', inference=inference,
                  primary_effects=effects, block_rows=block_rows, model_rows=evaluation, group_rows=groups,
                  transfer_rows=transfers, transfer_group_rows=transfer_groups)
    (ROOT / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                       encoding='utf-8')
    for name, rows in (('EVAL_ROWS.csv', evaluation), ('GROUP_METRICS.csv', groups), ('TRANSFER_ROWS.csv', transfers),
                       ('TRANSFER_GROUPS.csv', transfer_groups), ('BLOCK_RESULTS.csv', block_rows),
                       ('PRIMARY_EFFECTS.csv', effects)):
        export_csv(name, rows)
    print(f'Summarized {blocks} complete paired blocks, {len(evaluation)} evaluations, {len(transfers)} transfer rows.')


if __name__ == '__main__':
    main()
