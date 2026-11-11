"""Summarize the audited prefix clamp and boundary-state crossover."""

import csv
import json
from pathlib import Path

import numpy as np

from metrics import score


ROOT = Path(__file__).resolve().parent
SCALE_CASES = dict(DD='nominal', MM='prefix_mean', MD='mean_native', DM='mean_traces')
CONDITIONS = dict(SCALE=tuple(SCALE_CASES.values()), SCALE_MEAN=('nominal', 'prefix_mean'), GAIN2=('nominal',))
PAIRS = (('DD', 'MM'), ('DD', 'DM'), ('MD', 'MM'), ('DD', 'MD'), ('DM', 'MM'))
PRIMARY = ('prefix_total_effect', 'trace_source_effect')
SECONDARY = ('trace_on_dynamic_native', 'trace_on_mean_native', 'native_on_dynamic_trace',
             'native_on_mean_trace', 'native_source_effect', 'interaction')
METRICS = ('balanced_accuracy', 'accuracy', 'coverage', 'silence', 'saturated_tie',
           'intermediate_tie', 'agree_ba', 'disagree_ba', 'agree_coverage', 'disagree_coverage',
           'select_first_ba', 'select_second_ba', 'select_first_coverage', 'select_second_coverage',
           'disagree_joint_correct')
RESOURCES = ('soma_events', 'output_events', 'recurrent_deliveries', 'output_deliveries')
MARGINS = ('mean_count_0', 'mean_count_1', 'mean_label_margin', 'disagree_label_margin')
BOOTSTRAP_SEED = 4073900001
BOOTSTRAP_DRAWS = 50_000
PERCENTILES = (1.25, 98.75)
CONTINUOUS_TOLERANCE = 5e-12


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


def predict(counts):
    return np.where(counts[:, 0] == counts[:, 1], -1, counts.argmax(axis=1))


def block_input(seed, examples):
    with np.load(ROOT / f'evaluate/test_s{seed}.npz', allow_pickle=False) as archive:
        base = {key: archive[key] for key in ('y', 'selected', 'first_bit', 'second_bit', 'octet')}
    assert all(value.shape == (examples,) for value in base.values())
    y, first, second, cue = (base[key] for key in ('y', 'first_bit', 'second_bit', 'selected'))
    disagree = first != second
    assert np.array_equal(y, np.where(cue == 0, first, second))
    assert disagree.sum() == examples // 2 and (y == 0).sum() == examples // 2
    assert np.count_nonzero(disagree & (y == 0)) == examples // 4
    return base, disagree


def load_condition(row, base, disagree):
    seed, arm, condition = (row[key] for key in ('seed', 'arm', 'condition'))
    assert row['id'] == f's{seed}_{arm}' and (row['T'], row['q']) == (36, 28)
    trace_arm = arm != 'GAIN2'
    keys = ('counts', 'resources', 'entry_native') + (('entry_traces',) if trace_arm else ())
    with np.load(ROOT / f'evaluate/{row["id"]}_{condition}.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in keys}
    examples = len(disagree)
    counts, resources, native = (data[key] for key in ('counts', 'resources', 'entry_native'))
    assert counts.shape == (examples, 2) and np.all((0 <= counts) & (counts <= 8))
    assert np.array_equal(counts, counts.astype(int))
    assert resources.shape == (examples, 4) and native.shape == (examples, 26)
    assert np.isin(native[:, 20:], [0, 1]).all()
    if trace_arm:
        assert data['entry_traces'].shape == (examples, 12, 2)
    assert all(np.isfinite(value).all() for value in data.values())
    np.testing.assert_allclose(resources.mean(axis=0), row['mean_resources'], rtol=0, atol=1e-12)
    assert all(np.allclose(row[key], value, rtol=0, atol=1e-12) for key, value in score(counts, base).items())
    cost = row['cost_per_episode']
    trace_slots = 24 if trace_arm else 0
    assert (cost['parameters'], cost['dense_forward_macs']) == (176, 36 * 156)
    assert (cost['trace_slots'], cost['continuous_state_slots']) == (trace_slots, 20 + trace_slots)
    row.update(zip(RESOURCES, row['mean_resources']))
    row.update({f'cost_{key}': value for key, value in cost.items()})
    y, samples = base['y'], np.arange(examples)
    margin = counts[samples, y].astype(int) - counts[samples, 1 - y].astype(int)
    row.update(mean_count_0=float(counts[:, 0].mean()), mean_count_1=float(counts[:, 1].mean()),
               mean_label_margin=float(margin.mean()), disagree_label_margin=float(margin[disagree].mean()))
    return data


def verify_entries(data):
    scale = {case: data['SCALE', condition] for case, condition in SCALE_CASES.items()}
    for case in ('MD', 'DM'):
        for index, field in enumerate(('entry_native', 'entry_traces')):
            source = 'DD' if case[index] == 'D' else 'MM'
            np.testing.assert_array_equal(scale[case][field], scale[source][field])
    mean_nominal, mean_prefix = (data['SCALE_MEAN', condition] for condition in CONDITIONS['SCALE_MEAN'])
    for field in mean_nominal:
        np.testing.assert_array_equal(mean_nominal[field], mean_prefix[field])
    return scale


def paired_changes(seed, arm, left, right, before, after, y, masks):
    left_class, right_class = predict(before), predict(after)
    left_good, right_good = left_class == y, right_class == y
    count_change = np.any(before != after, axis=1)
    rows = []
    for subset, mask in masks.items():
        n = int(mask.sum())
        gains = int(np.count_nonzero(mask & left_good & ~right_good))
        losses = int(np.count_nonzero(mask & right_good & ~left_good))
        count_n = int(np.count_nonzero(mask & count_change))
        class_n = int(np.count_nonzero(mask & (left_class != right_class)))
        assert gains + losses <= class_n <= count_n
        rows.append(dict(seed=seed, arm=arm, left=left, right=right, subset=subset, n=n,
                         count_changed=count_n, class_changed=class_n, correctness_changed=gains + losses,
                         left_only_correct=gains, right_only_correct=losses, accuracy_difference=(gains - losses) / n))
    return rows


def boundary_changes(seed, scale, masks):
    dd, mm = scale['DD'], scale['MM']
    native = np.abs(dd['entry_native'] - mm['entry_native'])
    traces = np.abs(dd['entry_traces'] - mm['entry_traces'])
    components = dict(native_continuous=native[:, :20], branches=native[:, :12],
                      soma_voltage=native[:, 12:18], output_voltage=native[:, 18:20],
                      previous_spikes=native[:, 20:], traces=traces.reshape(len(traces), 24),
                      fast_trace=traces[:, :, 0], slow_trace=traces[:, :, 1])
    output = []
    for subset, mask in masks.items():
        for component, difference in components.items():
            differences = difference[mask]
            tolerance = 0. if component == 'previous_spikes' else CONTINUOUS_TOLERANCE
            changed = differences > tolerance
            output.append(dict(seed=seed, subset=subset, component=component, examples=len(differences),
                               slots_per_example=differences.shape[1], tolerance=tolerance,
                               examples_changed=int(changed.any(axis=1).sum()), components_changed=int(changed.sum()),
                               max_abs_difference=float(differences.max()), mean_abs_difference=float(differences.mean())))
    return output


def effects_for_block(seed, rates):
    dd, mm, md, dm = (rates[case] for case in SCALE_CASES)
    trace_d, trace_m, native_d, native_m = dd - dm, md - mm, dd - md, dm - mm
    total, trace, native = dd - mm, (trace_d + trace_m) / 2, (native_d + native_m) / 2
    assert total == trace + native
    return dict(seed=seed, DD_disagree_ba=dd, MM_disagree_ba=mm, MD_disagree_ba=md, DM_disagree_ba=dm,
                prefix_total_effect=total, trace_source_effect=trace, native_source_effect=native,
                trace_on_dynamic_native=trace_d, trace_on_mean_native=trace_m,
                native_on_dynamic_trace=native_d, native_on_mean_trace=native_m, interaction=trace_d - trace_m)


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
        for key in ('values', 'mean', 'minimum', 'maximum', 'ci97_5'):
            np.testing.assert_allclose(effect[key], replay[key], rtol=0, atol=1e-12)


def group_conditions(evaluation):
    groups = {}
    for row in evaluation:
        groups.setdefault((row['arm'], row['condition']), []).append(row)
    return [dict(arm=arm, condition=condition, blocks=len(rows), cost_per_episode=rows[0]['cost_per_episode'],
                 **{key: float(np.mean([row[key] for row in rows])) for key in METRICS + RESOURCES + MARGINS})
            for (arm, condition), rows in groups.items()]


def main():
    freeze, science, audit = (read(name) for name in ('FREEZE.json', 'SCIENCE_SUMMARY.json', 'AUDIT.json'))
    assert science['status'] == 'COMPLETE' and audit['status'] == 'PASS'
    blocks, examples = science['blocks'], science['examples_per_block']
    assert blocks == freeze['blocks'] == 12 and examples == freeze['examples_per_block'] and examples in (128, 256)
    assert science['fits'] == science['updates'] == 0
    assert science['evaluations'] == audit['evaluations'] == freeze['evaluations'] == 7 * blocks
    assert science['forward_sample_steps'] == blocks * examples * (5 * 25 + 7 * 11)
    assert science['path_sample_steps'] == blocks * examples * 7 * 36
    assert science['dense_forward_macs'] == 156 * science['forward_sample_steps']
    assert science['boundary_bank_bytes'] == blocks * examples * 8 * (26 + 4 * 50)
    evaluation = sorted(read('evaluate/ROWS.json'), key=lambda row: (
        row['seed'], tuple(CONDITIONS).index(row['arm']), CONDITIONS[row['arm']].index(row['condition'])))
    index = {(row['seed'], row['arm'], row['condition']): row for row in evaluation}
    expected = {(seed, arm, condition) for seed in range(blocks) for arm, conditions in CONDITIONS.items()
                for condition in conditions}
    assert len(evaluation) == len(index) == len(expected) and set(index) == expected

    blocks_out, pairs, boundaries = [], [], []
    for seed in range(blocks):
        base, disagree = block_input(seed, examples)
        masks = dict(all=np.ones(examples, dtype=bool), disagree=disagree)
        data = {(arm, condition): load_condition(index[seed, arm, condition], base, disagree)
                for arm, conditions in CONDITIONS.items() for condition in conditions}
        scale = verify_entries(data)
        rates = {case: index[seed, 'SCALE', condition]['disagree_ba'] for case, condition in SCALE_CASES.items()}
        blocks_out.append(effects_for_block(seed, rates))
        for left, right in PAIRS:
            changes = paired_changes(seed, 'SCALE', left, right, scale[left]['counts'], scale[right]['counts'], base['y'], masks)
            assert changes[1]['accuracy_difference'] == rates[left] - rates[right]
            pairs.extend(changes)
        nominal = data['SCALE_MEAN', 'nominal']['counts']
        prefix = data['SCALE_MEAN', 'prefix_mean']['counts']
        pairs.extend(paired_changes(seed, 'SCALE_MEAN', 'nominal', 'prefix_mean', nominal, prefix, base['y'], masks))
        boundaries.extend(boundary_changes(seed, scale, masks))

    primary, secondary = effect_statistics(blocks_out)
    verify_primary_replay(primary, audit)
    inference = dict(blocks=blocks, examples_per_block=examples, bootstrap_seed=BOOTSTRAP_SEED,
                     bootstrap_draws=BOOTSTRAP_DRAWS, percentiles=PERCENTILES,
                     interval='whole-training-block paired percentile', nominal_family_confidence=.95,
                     independent_primary_match=True)
    result = dict(study_status='FIXED_WEIGHT_D16_PREFIX_INTERVENTIONS', inference=inference,
                  primary_effects=primary, secondary_effects=secondary, block_rows=blocks_out,
                  model_rows=evaluation, group_rows=group_conditions(evaluation), paired_change_rows=pairs,
                  boundary_change_rows=boundaries)
    (ROOT / 'RESULTS.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                                       encoding='utf-8')
    export_csv('PRIMARY_EFFECTS.csv', primary)
    export_csv('BLOCK_RESULTS.csv', blocks_out)
    export_csv('EVAL_ROWS.csv', evaluation)
    print(f'Summarized {blocks} paired blocks, {len(evaluation)} conditions and {len(pairs)} paired-change rows.')


if __name__ == '__main__':
    main()
