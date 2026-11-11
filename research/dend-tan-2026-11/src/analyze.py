"""Frozen count-based confirmation statistics; reads archives without model calls."""

import json
from pathlib import Path

import numpy as np


GEOMETRIES = ('orthogonal', 'near')
HISTORIES = ('four', 'three0', 'three2', 'two')
RULES = ('original', 'replay', 'guarded')
EVICTED = {'four': (0, 1), 'three0': (0,), 'three2': (1,), 'two': ()}
MISSING = {'four': (), 'three0': (1,), 'three2': (0,), 'two': (0, 1)}
THRESHOLDS = {'evicted_gain': 0., 'recent_change': -.02}
BOOTSTRAP_SEED = 4087900001
BOOTSTRAP_DRAWS = 50_000
RESOURCE_NAMES = ('soma_events', 'output_events', 'recurrent_deliveries', 'output_deliveries',
                  'soma_events_t16_to27', 'output_events_t16_to27')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def arrays(path, names):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in names}


def assert_close(actual, expected, context):
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0., err_msg=context)


def balanced(correct, y, mask):
    if not mask.any():
        return None
    values = [correct[mask & (y == label)] for label in (0, 1)]
    assert all(len(part) for part in values)
    return float((values[0].mean()+values[1].mean())/2)


def primary_summary(name, values, draws):
    values = np.asarray(values, dtype=float)
    assert values.shape == (12,) and np.isfinite(values).all()
    low, high = np.percentile(values[draws].mean(axis=1), [1.25, 98.75])
    return dict(effect=name, blocks=12, values=values.tolist(), mean=float(values.mean()),
                sample_sd=float(values.std(ddof=1)), minimum=float(values.min()), maximum=float(values.max()),
                positive=int((values > 0).sum()), zero=int((values == 0).sum()), negative=int((values < 0).sum()),
                ci97_5=[float(low), float(high)], threshold=THRESHOLDS[name], passes=bool(low > THRESHOLDS[name]),
                direction='positive' if low > 0 else 'negative' if high < 0 else 'undetermined')


def independent_primary(effects, guard):
    """Old independent audit's sum/quantile calculation, given independently rebuilt effects.

    This helper does not supply or independently derive the underlying effects.
    The caller must obtain those from its independent raw-path reconstruction.
    """
    guard()
    assert set(effects) == set(THRESHOLDS)
    rng = np.random.default_rng(4087900001)
    draws = rng.integers(0, 12, (50_000, 12))
    result = {}
    for name, block_values in effects.items():
        values = np.asarray(block_values)
        assert values.shape == (12,) and np.isfinite(values).all()
        bootstrap = np.sum(values[draws], axis=1)/12
        low, high = np.quantile(bootstrap, [.0125, .9875])
        result[name] = dict(effect=name, blocks=12, values=values.tolist(), mean=float(values.mean()),
                            sample_sd=float(values.std(ddof=1)), minimum=float(values.min()), maximum=float(values.max()),
                            positive=int((values > 0).sum()), zero=int((values == 0).sum()), negative=int((values < 0).sum()),
                            ci97_5=[float(low), float(high)], threshold=THRESHOLDS[name],
                            passes=bool(low > THRESHOLDS[name]),
                            direction='positive' if low > 0 else 'negative' if high < 0 else 'undetermined')
    guard()
    return result


def masks(selected, history):
    result = dict(all=np.ones(len(selected), dtype=bool), older=selected < 2, recent=selected >= 2,
                  evicted=np.isin(selected, EVICTED[history]), missing=np.isin(selected, MISSING[history]))
    result.update({f'position{position}': selected == position for position in range(4)})
    return result


def add(grouped, key, values):
    target = grouped.setdefault(key, {name: 0 for name in values})
    assert target.keys() == values.keys()
    for name, value in values.items():
        target[name] += value


def performance_counts(y, path, mask):
    correct, coverage, trigger = (path[name] for name in ('correct', 'coverage', 'trigger'))
    result = dict(n=int(mask.sum()), n0=int((mask & (y == 0)).sum()), n1=int((mask & (y == 1)).sum()),
                  correct0=int((mask & (y == 0) & correct).sum()), correct1=int((mask & (y == 1) & correct).sum()),
                  covered=int((mask & coverage).sum()), ties=int((mask & ~coverage).sum()),
                  triggered=int((mask & trigger).sum()), replay_steps=int(path['replay_steps'][mask].sum()),
                  evicted_comparisons=int(path['evicted_comparisons'][mask].sum()),
                  cross_entropy_sum=float(path['loss'][mask].sum()))
    for position, name in enumerate(RESOURCE_NAMES):
        result[name] = int(path['resources'][mask, position].sum())
    for position, name in enumerate(RESOURCE_NAMES[:4]):
        result['replay_'+name] = int(path['replay_resources'][mask, position].sum())
    return result


def pair_counts(paths, mask):
    original, old, new = (paths[rule] for rule in RULES)
    blocked = old['trigger'] & ~new['trigger']
    result = dict(n=int(mask.sum()), old_triggered=int((mask & old['trigger']).sum()),
                  new_triggered=int((mask & new['trigger']).sum()), blocked=int((mask & blocked).sum()),
                  canceled_replay_steps=int((old['replay_steps'][mask]-new['replay_steps'][mask]).sum()),
                  guarded_minus_replay_ce_sum=float((new['loss'][mask]-old['loss'][mask]).sum()),
                  guarded_loss_increased=int((mask & (new['loss'] > old['loss'])).sum()),
                  guarded_loss_decreased=int((mask & (new['loss'] < old['loss'])).sum()))
    for left, right in (('guarded', 'replay'), ('guarded', 'original'), ('replay', 'original')):
        a, b = paths[left], paths[right]
        label = left+'_vs_'+right
        result[label+'_gained_correct'] = int((mask & a['correct'] & ~b['correct']).sum())
        result[label+'_lost_correct'] = int((mask & ~a['correct'] & b['correct']).sum())
        result[label+'_changed_prediction'] = int((mask & (a['prediction'] != b['prediction'])).sum())
    return result


def describe(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(n=0, mean=None, median=None, p95=None, minimum=None, maximum=None)
    assert np.isfinite(values).all()
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), minimum=float(values.min()), maximum=float(values.max()))


def execute(root, audit_effects, guard):
    """Write RESULTS.json after the caller's full independent audit succeeds.

    audit_effects is a dict keyed by the two endpoints, or a list of endpoint
    dictionaries. Every endpoint must include values[12] and ci97_5[2].
    """
    root = Path(root)
    target = root/'RESULTS.json'
    assert not target.exists()
    assert {path.name for path in (root/'evaluate').iterdir() if path.is_dir()} == {f's{s}' for s in range(12)}
    if isinstance(audit_effects, list):
        assert len(audit_effects) == 2
        independent = {row['effect']: row for row in audit_effects}
    else:
        independent = audit_effects
    assert set(independent) == set(THRESHOLDS)
    performance, paired = {}, {}
    block_rows, path_rows, shared_groups, isolated_rows, isolation_cells = [], [], [], [], []
    source_fields = ('counts', 'resources', 'reset_steps')
    for seed in range(12):
        guard()
        directory = root/'evaluate'/f's{seed}'
        base = arrays(directory/'base.npz', ('y', 'selected', 'group', 'seed'))
        y, selected = base['y'], base['selected']
        assert y.shape == selected.shape == (256,) and int(base['seed']) == 4087020000+100*seed
        schedules = base['group'] % 4
        assert np.array_equal(np.bincount(schedules, minlength=4), np.full(4, 64))
        saved_rows = read(directory/'ROWS.json')
        rows = {row['condition']: row for row in saved_rows}
        expected = {f'{g}_{h}_{r}' for g in GEOMETRIES for h in HISTORIES for r in RULES}
        assert len(rows) == len(saved_rows) == 24 and set(rows) == expected
        block = dict(seed=seed, differences=[])
        effects = {name: [] for name in THRESHOLDS}
        for geometry in GEOMETRIES:
            for history in HISTORIES:
                guard()
                paths, sub = {}, masks(selected, history)
                group_times, group_works = [], []
                for rule in RULES:
                    condition = f'{geometry}_{history}_{rule}'
                    row = rows[condition]
                    fields = source_fields + (('replay_steps', 'replay_resources') if rule != 'original' else ())
                    fields += ('evicted_comparisons',) if rule == 'guarded' else ()
                    raw = arrays(directory/(condition+'.npz'), fields)
                    if rule == 'original':
                        raw.update(replay_steps=np.zeros(256, dtype=np.int64), replay_resources=np.zeros((256, 4), dtype=np.int64))
                    if rule != 'guarded':
                        raw['evicted_comparisons'] = np.zeros((256, 36), dtype=np.int64)
                    counts = raw['counts'].astype(float)
                    assert counts.shape == (256, 2) and np.isfinite(counts).all()
                    assert np.array_equal(counts, counts.astype(int)) and ((counts >= 0) & (counts <= 8)).all()
                    assert raw['resources'].shape == (256, 6) and raw['replay_resources'].shape == (256, 4)
                    assert raw['reset_steps'].shape == (256, 36)
                    coverage = counts[:, 0] != counts[:, 1]
                    prediction = np.where(coverage, counts.argmax(axis=1), -1)
                    correct = prediction == y
                    margin = counts[np.arange(256), y]-counts[np.arange(256), 1-y]
                    path = raw | dict(prediction=prediction, correct=correct, coverage=coverage,
                                      trigger=raw['reset_steps'].any(axis=1), loss=np.logaddexp(0., -.5*margin))
                    paths[rule] = path
                    score = {name: balanced(correct, y, mask) for name, mask in sub.items()}
                    for name, field in (('all', 'balanced_accuracy'), ('older', 'older_ba'), ('recent', 'recent_ba'),
                                        ('evicted', 'evicted_ba'), ('missing', 'missing_ba')):
                        if score[name] is None:
                            assert row[field] is None
                        else:
                            assert_close(score[name], row[field], condition+'/'+field)
                    assert_close([score[f'position{p}'] for p in range(4)], row['ba_by_position'], condition+'/positions')
                    assert_close(coverage.mean(), row['coverage'], condition+'/coverage')
                    path_rows.append(dict(seed=seed, **row, cross_entropy_mean=float(path['loss'].mean()),
                                          cross_entropy_p95=float(np.percentile(path['loss'], 95)),
                                          cross_entropy_max=float(path['loss'].max())))
                    group_times.append(row['shared_group_seconds'])
                    group_works.append(row['shared_group_work'])
                assert group_times[0] == group_times[1] == group_times[2] and group_works[0] == group_works[1] == group_works[2]
                shared_groups.append(dict(seed=seed, geometry=geometry, history=history,
                                          seconds=group_times[0], work=group_works[0]))
                assert not (paths['guarded']['trigger'] & ~paths['replay']['trigger']).any()
                assert (paths['guarded']['replay_steps'] <= paths['replay']['replay_steps']).all()
                for schedule in ('all', 'A', 'B', 'C', 'D'):
                    schedule_mask = np.ones(256, dtype=bool) if schedule == 'all' else schedules == 'ABCD'.index(schedule)
                    for subset, subset_mask in sub.items():
                        mask = schedule_mask & subset_mask
                        if not mask.any():
                            continue
                        for rule in RULES:
                            add(performance, (geometry, history, schedule, subset, rule), performance_counts(y, paths[rule], mask))
                        add(paired, (geometry, history, schedule, subset), pair_counts(paths, mask))
                for subset in ('all', 'evicted', 'missing', 'recent'):
                    mask = sub[subset]
                    if not mask.any():
                        continue
                    scores = {rule: balanced(paths[rule]['correct'], y, mask) for rule in RULES}
                    difference = dict(geometry=geometry, history=history, subset=subset, **scores,
                                      guarded_minus_replay=scores['guarded']-scores['replay'],
                                      guarded_minus_original=scores['guarded']-scores['original'],
                                      replay_minus_original=scores['replay']-scores['original'])
                    block['differences'].append(difference)
                    if geometry == 'near' and history in ('three0', 'three2') and subset in ('evicted', 'recent'):
                        effects['evicted_gain' if subset == 'evicted' else 'recent_change'].append(difference['guarded_minus_replay'])
        for name, values in effects.items():
            assert len(values) == 2
            block[name] = float(np.mean(values))
        block_rows.append(block)
        singles = read(directory/'MICROBENCH.json')
        assert len(singles) <= 32
        isolated_rows.extend(dict(seed=seed, **row) for row in singles)
        for geometry in GEOMETRIES:
            for schedule in 'ABCD':
                cell = [row for row in singles if row['geometry'] == geometry and row['schedule'] == schedule]
                assert len(cell) <= 4
                isolation_cells.append(dict(seed=seed, geometry=geometry, schedule=schedule, pairs=len(cell),
                                            maximum_pairs=4, fewer_than_four=len(cell) < 4))
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 12, (BOOTSTRAP_DRAWS, 12))
    primary = [primary_summary(name, [row[name] for row in block_rows], draws) for name in THRESHOLDS]
    for result in primary:
        audited = independent[result['effect']]
        for key in ('values', 'ci97_5'):
            assert_close(result[key], audited[key], 'independent primary/'+result['effect']+'/'+key)
        for key in ('mean', 'sample_sd', 'minimum', 'maximum'):
            if key in audited:
                assert_close(result[key], audited[key], 'independent primary/'+key)
        for key in ('passes', 'threshold', 'blocks', 'positive', 'zero', 'negative', 'direction'):
            if key in audited:
                assert result[key] == audited[key], key
    performance_rows = []
    for key, values in performance.items():
        assert values['n0'] == values['n1'] and values['n'] == values['n0']+values['n1']
        performance_rows.append(dict(zip(('geometry', 'history', 'schedule', 'subset', 'rule'), key)) |
                                values | dict(balanced_accuracy=.5*(values['correct0']/values['n0']+values['correct1']/values['n1']),
                                              coverage=values['covered']/values['n'], mean_cross_entropy=values['cross_entropy_sum']/values['n']))
    paired_rows = [dict(zip(('geometry', 'history', 'schedule', 'subset'), key)) | values |
                   dict(canceled_native_macs=156*values['canceled_replay_steps']) for key, values in paired.items()]
    timings = []
    for geometry in GEOMETRIES:
        for schedule in ('all', 'A', 'B', 'C', 'D'):
            all_rows = [row for row in isolated_rows if row['geometry'] == geometry and (schedule == 'all' or row['schedule'] == schedule)]
            for subset in ('all_old_triggered', 'actually_canceled'):
                group = all_rows if subset == 'all_old_triggered' else [row for row in all_rows if row['guarded_steps'] < row['replay_steps']]
                timings.append(dict(geometry=geometry, schedule=schedule, subset=subset,
                                    replay_call_seconds=describe([row['replay_call_seconds'] for row in group]),
                                    guarded_call_seconds=describe([row['guarded_call_seconds'] for row in group]),
                                    paired_guarded_minus_replay_seconds=describe([row['guarded_call_seconds']-row['replay_call_seconds'] for row in group])))
    shared_work = {name: sum(row['work'][name] for row in shared_groups) for name in shared_groups[0]['work']}
    both = all(row['passes'] for row in primary)
    result = dict(study_status='FROZEN_EVICTED_KEY_SHARED_CONFIRMATION',
                  inference=dict(blocks=12, examples_per_block=256, bootstrap_seed=BOOTSTRAP_SEED,
                                 bootstrap_draws=BOOTSTRAP_DRAWS, percentiles=[1.25, 98.75],
                                 nominal_family_confidence=.95, independent_primary_match=True,
                                 inference_unit='paired training and held-out block',
                                 coverage_caveat='Percentile bootstrap coverage is nominal; 12 blocks do not give a finite-sample coverage guarantee.'),
                  decision=dict(both_pass=both, evicted_pass=primary[0]['passes'], recent_pass=primary[1]['passes']),
                  primary_effects=primary, block_rows=block_rows, model_rows=path_rows,
                  performance_rows=performance_rows, paired_rows=paired_rows,
                  shared_execution=dict(groups=len(shared_groups), seconds=sum(row['seconds'] for row in shared_groups),
                                        work=shared_work, rows=shared_groups),
                  isolated_timing=dict(pairs=len(isolated_rows), maximum_pairs=384, cells=isolation_cells,
                                       summaries=timings, raw_records=isolated_rows),
                  resource_fields=list(RESOURCE_NAMES),
                  rule_labels=dict(original='原模型', replay='旧全前缀重放', guarded='逐出键门控'),
                  notes=['Both primary thresholds use strict lower-bound comparisons; all 12 blocks are retained.',
                         'All non-primary geometry, schedule, position, baseline and cost results are descriptive without confirmatory multiplicity claims.',
                         'Tied native counts count as incorrect; coverage is reported separately from accuracy.',
                         'Subsets and all-schedule rows overlap and must not be added together.',
                         'shared_group_seconds is one common execution time for three outputs; repeated model-row fields must not be summed across rules.',
                         'Isolated timings are complete from-zero calls selected by the frozen old-trigger rule, not random latency samples.',
                         'Online path resources and replay resources are distinct logical accounts; shared actual execution work is reported once.',
                         'The extra allocated evicted-key log is 32 bytes per episode; events and MACs are not physical energy.',
                         'Continuous full-time traces are retained for the first eight examples per condition; all-example counts and event records remain archived.'])
    guard()
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    guard(disk=True)
    return result
