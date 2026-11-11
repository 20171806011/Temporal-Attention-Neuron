"""Verify the published DEND-TAN confirmation from the bundled evidence only."""

import json
from pathlib import Path
import sys

import numpy as np


BLOCKS, EXAMPLES, STEPS = 12, 256, 36
GEOMETRIES = ('orthogonal', 'near')
HISTORIES = ('four', 'three0', 'three2', 'two')
RULES = ('original', 'replay', 'guarded')
SCHEDULES = 'ABCD'
EVICTED = ((0, 1), (0,), (1,), ())
MISSING = ((), (1,), (0,), (0, 1))
RESOURCE_NAMES = ('soma_events', 'output_events', 'recurrent_deliveries', 'output_deliveries',
                  'soma_events_t16_to27', 'output_events_t16_to27')
THRESHOLDS = {'evicted_gain': 0., 'recent_change': -.02}
DRAW_SEED, DRAWS, PERCENTILES = 4087900001, 50_000, [1.25, 98.75]
PATH_SHAPE = (BLOCKS, len(GEOMETRIES), len(HISTORIES), len(RULES), EXAMPLES)
TOLERANCE = 1e-12


def require(condition, context):
    if not condition:
        raise ValueError(context)


def fields(saved, computed, context):
    for name, value in computed.items():
        require(name in saved, f'{context}: missing {name}')
        observed = saved[name]
        if isinstance(value, (float, list, np.ndarray)):
            np.testing.assert_allclose(observed, value, atol=TOLERANCE, rtol=0., err_msg=f'{context}: {name}')
        else:
            require(observed == value, f'{context}: {name}: {observed!r} != {value!r}')


def row_index(rows, names, context):
    index = {tuple(row[name] for name in names): row for row in rows}
    require(len(rows) == len(index), f'{context}: duplicate rows')
    return index


def read_inputs(directory):
    shapes = dict(counts=PATH_SHAPE+(2,), triggered=PATH_SHAPE, replay_steps=PATH_SHAPE,
                  selected=(BLOCKS, EXAMPLES), y=(BLOCKS, EXAMPLES), schedule=(BLOCKS, EXAMPLES),
                  resources=PATH_SHAPE+(6,), replay_resources=PATH_SHAPE+(4,), evicted_comparisons=PATH_SHAPE,
                  first_read_scores=PATH_SHAPE+(2,), first_read_weights=PATH_SHAPE+(2,))
    with np.load(directory/'confirmation_counts.npz', allow_pickle=False) as archive:
        data = {name: archive[name] for name in shapes if name in archive}
    for name in ('counts', 'triggered', 'replay_steps', 'selected', 'y'):
        require(name in data, f'evidence: missing {name}')
    for name, array in data.items():
        require(array.shape == shapes[name] and array.dtype.kind in 'biuf' and np.isfinite(array).all(),
                f'evidence: invalid shape, type or nonfinite value in {name}')
        if name not in ('first_read_scores', 'first_read_weights'):
            integer = array.astype(np.int64)
            require(np.array_equal(array, integer) and (integer >= 0).all(), f'evidence: invalid integer {name}')
            data[name] = integer
    require(np.all(data['counts'] <= 8), 'evidence: counts outside the eight-step readout')
    require(np.all(data['triggered'] <= 1), 'evidence: trigger values must be binary')
    require(np.all(data['replay_steps'] < STEPS), 'evidence: invalid replay length')
    require(np.all(data['selected'] < 4) and np.all(data['y'] <= 1), 'evidence: invalid position or label')
    data['triggered'] = data['triggered'].astype(bool)
    if 'schedule' in data:
        require(np.all(data['schedule'] < 4), 'evidence: invalid schedule')
    for seed in range(BLOCKS):
        cell = 2*data['selected'][seed]+data['y'][seed]
        require(np.array_equal(np.bincount(cell, minlength=8), np.full(8, 32)), f'block {seed}: class balance')
        if 'schedule' in data:
            cell = 8*data['schedule'][seed]+cell
            require(np.array_equal(np.bincount(cell, minlength=32), np.full(32, 8)),
                    f'block {seed}: schedule/position/label balance')
    return data


def verify_paths(data):
    triggers, steps, counts = (data[name] for name in ('triggered', 'replay_steps', 'counts'))
    require(np.array_equal(triggers, steps > 0), 'paths: triggers and replay lengths disagree')
    require(not triggers[:, :, :, 0].any(), 'paths: original model triggered replay')
    old, guarded = triggers[:, :, :, 1], triggers[:, :, :, 2]
    require(not np.any(guarded & ~old), 'paths: guard introduced a trigger')
    require(np.array_equal(steps[:, :, :, 2], np.where(guarded, steps[:, :, :, 1], 0)),
            'paths: accepted replay length changed')
    if 'schedule' in data:
        expected = np.array([22, 24, 20, 23])[data['schedule']][:, None, None, None, :]
        require(np.array_equal(steps, np.where(triggers, expected, 0)), 'paths: replay length differs from first read')
    require(not triggers[:, :, 3].any(), 'paths: two-write history triggered replay')
    for name in ('counts', 'resources'):
        if name in data:
            values = data[name]
            for rule in (1, 2):
                require(np.array_equal(values[:, :, 3, rule], values[:, :, 3, 0]), f'paths: two-write {name} changed')
                unchanged = ~triggers[:, :, :, rule]
                require(np.array_equal(values[:, :, :, rule][unchanged], values[:, :, :, 0][unchanged]),
                        f'paths: nontriggered {name} differ from original')
            require(np.array_equal(values[:, :, :, 2][guarded], values[:, :, :, 1][guarded]),
                    f'paths: accepted {name} differ from replay')
    if 'replay_resources' in data:
        resources = data['replay_resources']
        require(not resources[~triggers].any() and np.array_equal(resources[:, :, :, 2][guarded], resources[:, :, :, 1][guarded]),
                'paths: replay resources inconsistent with trigger')
    comparisons = np.zeros_like(steps)
    comparisons[:, :, :, 2] = old*np.array([2, 1, 1, 0])[None, None, :, None]
    if 'evicted_comparisons' in data:
        require(np.array_equal(data['evicted_comparisons'], comparisons), 'paths: comparison counts disagree')
    else:
        data['evicted_comparisons'] = comparisons
    if 'first_read_scores' in data and 'first_read_weights' in data:
        scores, weights = data['first_read_scores'], data['first_read_weights']
        exponentials = np.exp(scores-scores.max(axis=-1, keepdims=True))
        np.testing.assert_allclose(weights, exponentials/exponentials.sum(axis=-1, keepdims=True), rtol=0., atol=TOLERANCE)
    require(counts.shape == PATH_SHAPE+(2,), 'paths: invalid counts shape')


def masks(selected, history):
    result = dict(all=np.ones(EXAMPLES, dtype=bool), older=selected < 2, recent=selected >= 2,
                  evicted=np.isin(selected, EVICTED[history]), missing=np.isin(selected, MISSING[history]))
    return result | {f'position{position}': selected == position for position in range(4)}


def mean(values, mask):
    return float(values[mask].mean()) if mask.any() else None


def model_values(path, subsets, history, group):
    output = dict(examples=EXAMPLES, T=STEPS, q=28, evicted_positions=list(EVICTED[history]),
                  missing_positions=list(MISSING[history]), onset_count=EXAMPLES, eviction_fraction=float(history != 3),
                  reset_count=int(path['triggered'].sum()), reset_fraction=float(path['triggered'].mean()),
                  older_reset_fraction=mean(path['triggered'], subsets['older']),
                  recent_reset_fraction=mean(path['triggered'], subsets['recent']),
                  reset_by_position=[mean(path['triggered'], subsets[f'position{i}']) for i in range(4)],
                  replay_steps=int(path['replay_steps'].sum()), evicted_comparisons=int(path['evicted_comparisons'].sum()),
                  cross_entropy_mean=float(path['loss'].mean()), cross_entropy_p95=float(np.percentile(path['loss'], 95)),
                  cross_entropy_max=float(path['loss'].max()), shared_group_seconds=group['seconds'])
    for source, metric, position_field in (('correct', 'ba', 'ba_by_position'),
                                            ('covered', 'coverage', 'coverage_by_position')):
        values = path[source]
        output['balanced_accuracy' if metric == 'ba' else metric] = mean(values, subsets['all'])
        for subset in ('older', 'recent', 'evicted', 'missing'):
            output[f'{subset}_{metric}'] = mean(values, subsets[subset])
        output[position_field] = [mean(values, subsets[f'position{i}']) for i in range(4)]
    for name, field in (('resources', 'mean_resources'), ('replay_resources', 'replay_resources_mean')):
        if name in path:
            output[field] = path[name].mean(axis=0)
    if 'first_read_scores' in path and 'first_read_weights' in path:
        scores, weights = path['first_read_scores'], path['first_read_weights']
        eligible = subsets['recent']
        target = np.where(subsets['position2'], 1, 0)
        row = np.arange(EXAMPLES)
        routed = eligible & (scores[row, target] > scores[row, 1-target])
        target_weight = np.where(eligible, weights[row, target], 0.)
        output['available_fraction'] = float(eligible.mean())
        for values, metric, position_field in ((routed, 'routing_accuracy', 'routing_by_position'),
                                                (target_weight, 'target_weight', 'target_weight_by_position')):
            output[metric] = mean(values, subsets['all'])
            for subset in ('older', 'recent'):
                output[f'{subset}_{metric}'] = mean(values, subsets[subset])
            output[position_field] = [mean(values, subsets[f'position{i}']) for i in range(4)]
    return output


def performance_values(y, path, mask):
    correct, covered, triggered = (path[name][mask] for name in ('correct', 'covered', 'triggered'))
    labels = y[mask]
    result = dict(n=len(labels), n0=int((labels == 0).sum()), n1=int((labels == 1).sum()),
                  correct0=int(((labels == 0) & correct).sum()), correct1=int(((labels == 1) & correct).sum()),
                  covered=int(covered.sum()), ties=int((~covered).sum()), triggered=int(triggered.sum()),
                  replay_steps=int(path['replay_steps'][mask].sum()), evicted_comparisons=int(path['evicted_comparisons'][mask].sum()),
                  cross_entropy_sum=float(path['loss'][mask].sum()))
    for name, prefix, columns in (('resources', '', RESOURCE_NAMES), ('replay_resources', 'replay_', RESOURCE_NAMES[:4])):
        if name in path:
            result.update({prefix+key: int(value) for key, value in zip(columns, path[name][mask].sum(axis=0))})
    return result


def paired_values(paths, mask):
    original, old, guarded = paths
    old_triggered, new_triggered = old['triggered'][mask], guarded['triggered'][mask]
    losses = guarded['loss'][mask]-old['loss'][mask]
    canceled_steps = int((old['replay_steps'][mask]-guarded['replay_steps'][mask]).sum())
    output = dict(n=int(mask.sum()), old_triggered=int(old_triggered.sum()), new_triggered=int(new_triggered.sum()),
                  blocked=int((old_triggered & ~new_triggered).sum()), canceled_replay_steps=canceled_steps,
                  guarded_minus_replay_ce_sum=float(losses.sum()), guarded_loss_increased=int((losses > 0).sum()),
                  guarded_loss_decreased=int((losses < 0).sum()), canceled_native_macs=156*canceled_steps)
    for label, a, b in (('guarded_vs_replay', guarded, old), ('guarded_vs_original', guarded, original),
                        ('replay_vs_original', old, original)):
        x, y = a['correct'][mask], b['correct'][mask]
        output[label+'_gained_correct'] = int((x & ~y).sum())
        output[label+'_lost_correct'] = int((~x & y).sum())
        output[label+'_changed_prediction'] = int((a['prediction'][mask] != b['prediction'][mask]).sum())
    return output


def add(table, key, values):
    row = table.setdefault(key, dict.fromkeys(values, 0))
    for name, value in values.items():
        row[name] += value


def verify_tables(data, report):
    models = row_index(report['model_rows'], ('seed', 'geometry', 'history', 'rule'), 'model rows')
    blocks = row_index(report['block_rows'], ('seed',), 'block rows')
    shared = row_index(report['shared_execution']['rows'], ('seed', 'geometry', 'history'), 'shared rows')
    require(len(models) == 288 and len(blocks) == BLOCKS and len(shared) == 96, 'report: incorrect row counts')
    performance, paired, shared_work = {}, {}, {}
    effects = {name: [] for name in THRESHOLDS}
    for seed in range(BLOCKS):
        y, selected = data['y'][seed], data['selected'][seed]
        schedules = {'all': np.ones(EXAMPLES, dtype=bool)}
        if 'schedule' in data:
            schedules.update({name: data['schedule'][seed] == s for s, name in enumerate(SCHEDULES)})
        block_effects = {name: [] for name in THRESHOLDS}
        differences = row_index(blocks[seed,]['differences'], ('geometry', 'history', 'subset'), f'block {seed}')
        difference_keys = set()
        for g, geometry in enumerate(GEOMETRIES):
            for h, history in enumerate(HISTORIES):
                subset = masks(selected, h)
                group = shared[seed, geometry, history]
                require(np.isfinite(group['seconds']) and group['seconds'] > 0., 'shared execution time is invalid')
                replay_steps = data['replay_steps'][seed, g, h, 1]
                main = EXAMPLES*STEPS
                branches = int(np.where(replay_steps > 0, STEPS-replay_steps, 0).sum())
                replay = int(replay_steps.sum())
                work = dict(main_sample_steps=main, branch_sample_steps=branches, replay_sample_steps=replay,
                            online_sample_steps=main+branches, native_macs=156*(main+branches+replay))
                fields(group['work'], work, 'shared group work')
                add(shared_work, 'all', work)
                paths = []
                for r, rule in enumerate(RULES):
                    key = seed, g, h, r
                    counts = data['counts'][key]
                    margin = counts[:, 1]-counts[:, 0]
                    covered = margin != 0
                    prediction = np.where(covered, margin > 0, -1)
                    path = dict(covered=covered, prediction=prediction, correct=prediction == y,
                                loss=np.logaddexp(0., -.5*margin*(2*y-1)))
                    for name in ('triggered', 'replay_steps', 'evicted_comparisons', 'resources', 'replay_resources',
                                 'first_read_scores', 'first_read_weights'):
                        if name in data:
                            path[name] = data[name][key]
                    paths.append(path)
                    row = models[seed, geometry, history, rule]
                    fields(row, model_values(path, subset, h, group) | {'condition': f'{geometry}_{history}_{rule}'},
                           f'block {seed} {geometry}/{history}/{rule}')
                    fields(row['shared_group_work'], work, 'model group work')
                    if subset['missing'].any():
                        require(mean(path['correct'], subset['missing']) == .5*mean(covered, subset['missing']),
                                f'block {seed} {history}: missing labels exceed paired ceiling')
                for name in ('all', 'evicted', 'missing', 'recent'):
                    mask = subset[name]
                    if not mask.any():
                        continue
                    a, b, c = (mean(path['correct'], mask) for path in paths)
                    key = geometry, history, name
                    difference_keys.add(key)
                    fields(differences[key], dict(original=a, replay=b, guarded=c, guarded_minus_replay=c-b,
                                                  guarded_minus_original=c-a, replay_minus_original=b-a), f'block {seed} {key}')
                    if geometry == 'near' and history in ('three0', 'three2') and name in ('evicted', 'recent'):
                        block_effects['evicted_gain' if name == 'evicted' else 'recent_change'].append(c-b)
                for schedule, scheduled in schedules.items():
                    for name, mask in subset.items():
                        combined = mask & scheduled
                        if combined.any():
                            key = geometry, history, schedule, name
                            add(paired, key, paired_values(paths, combined))
                            for rule, path in zip(RULES, paths):
                                add(performance, (*key, rule), performance_values(y, path, combined))
        require(set(differences) == difference_keys, f'block {seed}: incomplete difference scopes')
        for name, values in block_effects.items():
            require(len(values) == 2, f'block {seed}: incomplete primary histories')
            effect = float(np.mean(values))
            fields(blocks[seed,], {name: effect}, f'block {seed} endpoint')
            effects[name].append(effect)

    group_fields = ('geometry', 'history', 'schedule', 'subset')
    for label, computed, keys in (('performance_rows', performance, (*group_fields, 'rule')),
                                   ('paired_rows', paired, group_fields)):
        archived = row_index(report[label], keys, label)
        expected_keys = set(archived) if 'schedule' in data else {key for key in archived if key[2] == 'all'}
        require(set(computed) == expected_keys, f'{label}: incomplete scopes')
        for key, values in computed.items():
            if label == 'performance_rows':
                values.update(balanced_accuracy=.5*(values['correct0']/values['n0']+values['correct1']/values['n1']),
                              coverage=values['covered']/values['n'], mean_cross_entropy=values['cross_entropy_sum']/values['n'])
            fields(archived[key], values, f'{label} {key}')
    fields(report['shared_execution'], dict(groups=len(shared), seconds=sum(row['seconds'] for row in shared.values())),
           'shared execution aggregate')
    fields(report['shared_execution']['work'], shared_work['all'], 'shared work aggregate')
    return effects, len(performance), len(paired)


def verify_primary(effects, report):
    fields(report['inference'], dict(blocks=BLOCKS, examples_per_block=EXAMPLES, bootstrap_seed=DRAW_SEED,
                                    bootstrap_draws=DRAWS, percentiles=PERCENTILES), 'inference design')
    saved = row_index(report['primary_effects'], ('effect',), 'primary effects')
    require(set(saved) == {(name,) for name in THRESHOLDS}, 'primary endpoint scopes')
    draws = np.random.default_rng(DRAW_SEED).integers(0, BLOCKS, size=(DRAWS, BLOCKS))
    checks = []
    for name, threshold in THRESHOLDS.items():
        values = np.asarray(effects[name], dtype=np.float64)
        require(values.shape == (BLOCKS,), f'{name}: expected all 12 block effects')
        low, high = np.percentile(values[draws].mean(axis=1), PERCENTILES)
        row = dict(effect=name, blocks=BLOCKS, values=values, mean=float(values.mean()),
                   sample_sd=float(values.std(ddof=1)), minimum=float(values.min()), maximum=float(values.max()),
                   positive=int((values > 0).sum()), zero=int((values == 0).sum()), negative=int((values < 0).sum()),
                   ci97_5=[float(low), float(high)], threshold=threshold, passes=bool(low > threshold),
                   direction='positive' if low > 0 else 'negative' if high < 0 else 'undetermined')
        fields(saved[name,], row, name)
        checks.append(row)
    evicted, recent = (row['passes'] for row in checks)
    fields(report['decision'], dict(evicted_pass=evicted, recent_pass=recent, both_pass=evicted and recent), 'joint decision')
    return checks


def describe(values):
    if not len(values):
        return dict(n=0, mean=None, median=None, p95=None, minimum=None, maximum=None)
    return dict(n=len(values), mean=float(np.mean(values)), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), minimum=float(np.min(values)), maximum=float(np.max(values)))


def verify_isolation(data, saved):
    records = saved['raw_records']
    keys = ('seed', 'geometry', 'schedule', 'sample')
    index = row_index(records, keys, 'isolated records')
    cells = row_index(saved['cells'], ('seed', 'geometry', 'schedule'), 'isolated cells')
    require(len(cells) == BLOCKS*len(GEOMETRIES)*len(SCHEDULES), 'isolated timing cell count')
    expected = []
    for seed in range(BLOCKS):
        for g, geometry in enumerate(GEOMETRIES):
            for s, schedule in enumerate(SCHEDULES):
                cell = [row for row in records if (row['seed'], row['geometry'], row['schedule']) == (seed, geometry, schedule)]
                if 'schedule' in data:
                    eligible = (data['schedule'][seed] == s) & data['triggered'][seed, g, 0, 1]
                    samples = np.flatnonzero(eligible)[:4].tolist()
                    require([row['sample'] for row in cell] == samples, 'isolated timing sample selection')
                require(len(cell) <= 4, 'isolated timing quota exceeded')
                fields(cells[seed, geometry, schedule], dict(pairs=len(cell), maximum_pairs=4, fewer_than_four=len(cell) < 4),
                       'isolated cell')
                for row in cell:
                    sample = row['sample']
                    require(isinstance(sample, int) and 0 <= sample < EXAMPLES, 'isolated sample out of range')
                    old, new = (int(data['replay_steps'][seed, g, 0, rule, sample]) for rule in (1, 2))
                    comparisons = int(data['evicted_comparisons'][seed, g, 0, 2, sample])
                    order = ['guarded', 'replay'] if seed % 3 == 2 else ['replay', 'guarded']
                    require(row['order'] == order, 'isolated timing pair order')
                    fields(row, dict(replay_steps=old, guarded_steps=new, evicted_comparisons=comparisons), 'isolated work')
                    times = [row[name] for name in ('replay_call_seconds', 'guarded_call_seconds', 'replay_seconds',
                                                    'guard_seconds', 'guarded_replay_seconds')]
                    old_call, new_call, old_replay, guard, new_replay = times
                    require(np.isfinite(times).all() and old > 0 and 0 < old_replay <= old_call and 0 < guard <= new_call
                            and guard+new_replay <= new_call+TOLERANCE
                            and (new_replay > 0 if new else new_replay == 0), 'invalid isolated timing scope')
                    expected.append((seed, geometry, schedule, sample))
    require(list(index) == expected, 'isolated records have unexpected order or scopes')
    fields(saved, dict(pairs=len(records), maximum_pairs=BLOCKS*len(GEOMETRIES)*len(SCHEDULES)*4), 'isolated total')
    summaries = row_index(saved['summaries'], ('geometry', 'schedule', 'subset'), 'isolated timing summaries')
    require(len(summaries) == 2*5*2, 'isolated timing summary scope')
    for geometry in GEOMETRIES:
        for schedule in ('all', *SCHEDULES):
            rows = [row for row in records if row['geometry'] == geometry and (schedule == 'all' or row['schedule'] == schedule)]
            for subset in ('all_old_triggered', 'actually_canceled'):
                selected = rows if subset == 'all_old_triggered' else [row for row in rows if row['guarded_steps'] < row['replay_steps']]
                old = np.array([row['replay_call_seconds'] for row in selected])
                new = np.array([row['guarded_call_seconds'] for row in selected])
                summary = summaries[geometry, schedule, subset]
                for name, values in (('replay_call_seconds', old), ('guarded_call_seconds', new),
                                     ('paired_guarded_minus_replay_seconds', new-old)):
                    fields(summary[name], describe(values), 'isolated timing aggregate')
    return len(records)


def main():
    directory = Path(__file__).resolve().parent/'evidence'
    result_path = directory/'RESULTS.json'
    if not result_path.is_file():
        result_path = directory/'confirmation_results.json'
    try:
        report = json.loads(result_path.read_text(encoding='utf-8'))
        data = read_inputs(directory)
        verify_paths(data)
        effects, performance_rows, paired_rows = verify_tables(data, report)
        primary = verify_primary(effects, report)
        isolated = verify_isolation(data, report['isolated_timing'])
    except (AssertionError, ValueError, OSError, KeyError, TypeError, IndexError) as error:
        print(f'Verification failed: {error}', file=sys.stderr)
        return 1

    print(f'PASS: {BLOCKS} paired blocks, {EXAMPLES} examples per condition, 288 model rows.')
    print('PASS: two-write count equivalence and guard-trigger subset.')
    for row in primary:
        low, high = row['ci97_5']
        decision = 'pass' if row['passes'] else 'fail'
        print(f"{row['effect']}: {row['mean']*100:+.4f} percentage points; "
              f"97.5% interval [{low*100:+.4f}, {high*100:+.4f}]; "
              f"strict lower bound > {row['threshold']*100:+.2f}: {decision}.")
    print(f'Joint decision: {"pass" if all(row["passes"] for row in primary) else "fail"}.')
    print(f'Checked {performance_rows} performance rows, {paired_rows} paired rows, shared work, and '
          f'timing summaries from {isolated} recorded isolated pairs.')
    optional = ('schedule', 'resources', 'replay_resources', 'first_read_scores', 'first_read_weights')
    missing = [name for name in optional if name not in data]
    if missing:
        print('Optional split, event or routing checks omitted for missing arrays: '+', '.join(missing)+'.')
    print('Scope: read-only verification of the bundled counts, recorded activity, and summary calculations.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
