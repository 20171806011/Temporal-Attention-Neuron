"""Independent audit of a single archived full-block timing fixture."""

import json
from pathlib import Path
import time
import zipfile

import numpy as np

import independent_capacity_data as data
import independent_event as event
import independent_guard as gate
import independent_warm as native


T, Q = 36, 28
FIXTURE_SEED, WEIGHT_SEED = 4099903301, 4099903302
PRIOR_TRAINING_AUDIT_ESTIMATE = 9.150149999186397
GEOMETRIES, HISTORIES, RULES = gate.GEOMETRIES, gate.HISTORIES, gate.RULES
GROUPS = tuple(f'{geometry}_{history}' for geometry in GEOMETRIES for history in HISTORIES)
DISCRETE = {'counts', 'resources', 'spikes', 'read_onset', 'reset_steps', 'eviction_seen', 'key_bank', 'write_gate',
            'logged_write_values', 'logged_write_ticks', 'logged_nonwrite_values', 'logged_nonwrite_ticks',
            'logged_nonwrite_count', 'first_read_tick', 'replay_steps', 'replay_resources', 'evicted_keys', 'evicted_comparisons'}


class Audit:
    def __init__(self, guard):
        self.guard = guard
        self.checks = 0
        self.max_error = 0.

    def check(self, condition, context):
        self.checks += 1
        assert bool(condition), context

    def exact(self, actual, expected, context):
        self.check(np.array_equal(actual, expected), context)

    def close(self, actual, expected, context, tolerance=5e-12):
        actual, expected = np.asarray(actual), np.asarray(expected)
        self.check(actual.shape == expected.shape, f'{context}: shape')
        self.check(np.isfinite(actual).all() and np.isfinite(expected).all(), f'{context}: finite')
        error = float(np.max(np.abs(actual-expected), initial=0.))
        self.max_error = max(self.max_error, error)
        self.check(error <= tolerance, f'{context}: {error:.17g} exceeds {tolerance}')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load(audit, path):
    with zipfile.ZipFile(path) as compressed:
        entries = compressed.infolist()
        audit.check(entries and all(item.compress_type == zipfile.ZIP_DEFLATED and item.filename.endswith('.npy') for item in entries)
                    and len({item.filename for item in entries}) == len(entries), f'{path.name}: lossless DEFLATE arrays')
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def initialize():
    rng = np.random.default_rng(WEIGHT_SEED)
    wi = rng.normal(0., .45, (6, 12))
    wr = rng.normal(0., .25, (6, 12))
    wo = rng.normal(0., .5, (6, 2))
    return np.r_[wi.ravel(), wr.ravel(), np.full(12, .15), np.full(6, .05), wo.ravel(), np.full(2, .1),
                 0., -1., 1., 0.]


def fixture(n):
    base = data.make_base(n, FIXTURE_SEED)
    phases = base['angle'][:, None]+np.deg2rad((0., 40., 180., 230.))
    keys = np.stack((np.cos(phases), np.sin(phases)), axis=-1)
    queries = np.stack((-keys[..., 1], keys[..., 0]), axis=-1)+base['query_noise']
    near = dict(base, key_vectors=keys, query_vectors=queries)
    inputs = {f'{geometry}_{history}': gate.assemble(observed, history)[0]
              for geometry, observed in (('orthogonal', base), ('near', near)) for history in HISTORIES}
    return base, inputs, initialize()


def independent_paths(x, w):
    """Recompute whole-block diagnostics and the first eight global trajectories."""
    original = native.replay(x, w, 'LEARN', Q, 'original')
    replay = event.replay(x, w, Q)
    guarded = gate.apply_guard(x, w, replay, original)
    return dict(original=original, replay=replay, guarded=guarded)


def compare_raw(audit, actual, expected, context):
    audit.check(set(actual) == set(expected), f'{context}: exact raw fields')
    for key, value in expected.items():
        array = actual[key]
        audit.check(array.shape == value.shape and array.dtype == value.dtype, f'{context}/{key}: exact shape and dtype')
        if key in DISCRETE or value.dtype.kind in 'biu':
            audit.exact(array, value, f'{context}: exact {key}')
        else:
            audit.close(array, value, f'{context}: {key}')
    for key in ('cue_native_before', 'cue_native_after', 'replay_reference'):
        if key in expected:
            audit.exact(actual[key][..., 20:], expected[key][..., 20:], f'{context}: exact previous pulses {key}')


def scores(raw, base, first_scores, first_weights):
    counts, y, selected = raw['counts'], base['y'], base['selected']
    covered = counts[:, 0] != counts[:, 1]
    correct = covered & (np.argmax(counts, axis=1) == y)
    target = 3-selected
    available = target < 2
    unique = (first_scores == first_scores.max(axis=1, keepdims=True)).sum(axis=1) == 1
    routing = available & unique & (first_scores.argmax(axis=1) == target)
    target_weight = np.zeros(len(y))
    target_weight[available] = first_weights[available, target[available]]
    older = selected < 2

    def ba(mask):
        return float(sum(correct[mask & (y == label)].mean() for label in (0, 1))/2)

    by_position = [selected == pos for pos in range(4)]
    return dict(balanced_accuracy=ba(np.ones(len(y), dtype=bool)), older_ba=ba(older), recent_ba=ba(~older),
                coverage=float(covered.mean()), older_coverage=float(covered[older].mean()), recent_coverage=float(covered[~older].mean()),
                routing_accuracy=float(routing.mean()), older_routing_accuracy=float(routing[older].mean()),
                recent_routing_accuracy=float(routing[~older].mean()), target_weight=float(target_weight.mean()),
                older_target_weight=float(target_weight[older].mean()), recent_target_weight=float(target_weight[~older].mean()),
                available_fraction=float(available.mean()), ba_by_position=[ba(mask) for mask in by_position],
                coverage_by_position=[float(covered[mask].mean()) for mask in by_position],
                routing_by_position=[float(routing[mask].mean()) for mask in by_position],
                target_weight_by_position=[float(target_weight[mask].mean()) for mask in by_position])


EVICTED = {'four': [0, 1], 'three0': [0], 'three2': [1], 'two': []}
MISSING = {'four': [], 'three0': [1], 'three2': [0], 'two': [0, 1]}


def group_work(paths):
    starts = paths['replay']['replay_steps']
    main = len(starts)*T
    branch = int((T-starts[starts > 0]).sum())
    replay = int(starts.sum())
    return dict(main_sample_steps=main, branch_sample_steps=branch, replay_sample_steps=replay,
                online_sample_steps=main+branch, native_macs=156*(main+branch+replay))


def row_metrics(audit, row, raw, base, first_scores, first_weights, geometry, history, rule, work):
    n = len(base['y'])
    group, condition = f'{geometry}_{history}', f'{geometry}_{history}_{rule}'
    identity = dict(condition=condition, geometry=geometry, history=history, rule=rule, examples=n, T=T, q=Q,
                    evicted_positions=EVICTED[history], missing_positions=MISSING[history])
    result = scores(raw, base, first_scores, first_weights)
    for name, positions in (('evicted', EVICTED[history]), ('missing', MISSING[history])):
        for field in ('ba', 'coverage'):
            result[f'{name}_{field}'] = float(np.mean([result[f'{field}_by_position'][pos] for pos in positions])) if positions else None
    selected = base['selected']
    applied = raw['reset_steps'].any(axis=1)
    result.update(mean_resources=raw['resources'].mean(axis=0),
                  replay_resources_mean=raw['replay_resources'].mean(axis=0) if rule != 'original' else np.zeros(4),
                  reset_count=int(raw['reset_steps'].sum()), onset_count=int(raw['read_onset'].sum()),
                  eviction_fraction=float(raw['eviction_seen'].any(axis=1).mean()),
                  reset_fraction=float(applied.mean()), older_reset_fraction=float(applied[selected < 2].mean()),
                  recent_reset_fraction=float(applied[selected >= 2].mean()),
                  reset_by_position=[float(applied[selected == pos].mean()) for pos in range(4)],
                  replay_steps=int(raw['replay_steps'].sum()) if rule != 'original' else 0,
                  evicted_comparisons=int(raw['evicted_comparisons'].sum()) if rule == 'guarded' else 0)
    audit.check(set(row) == set(identity) | set(result) | {'shared_group_seconds', 'shared_group_work'}, f'{condition}: row schema')
    for key, value in identity.items():
        audit.check(row[key] == value, f'{condition}: exact {key}')
    for key, value in result.items():
        if value is None:
            audit.check(row[key] is None, f'{condition}: {key} unavailable in this history')
        else:
            audit.close(row[key], value, f'{condition}: independently scored {key}')
    audit.check(np.isfinite(row['shared_group_seconds']) and row['shared_group_seconds'] > 0., f'{group}: positive shared call time')
    audit.check(row['shared_group_work'] == work, f'{group}: exact shared work')
    return result


def verify_scope(audit, paths, x, base, first, history, group):
    original, replay, guarded = (paths[rule] for rule in RULES)
    old, new = replay['reset_steps'], guarded['reset_steps']
    audit.check(not (new & ~old).any(), f'{group}: guarded replacement only removes old triggers')
    audit.exact(new, old & (guarded['match_confidence'] > guarded['evicted_cosine']), f'{group}: strict visible key veto')
    audit.exact(guarded['evicted_comparisons'], old*len(EVICTED[history]), f'{group}: exact valid keys compared at old triggers')
    audit.check(np.all(guarded['evicted_cosine'][~old] == -2.), f'{group}: no comparison off old trigger')
    vetoed = (old & ~new).any(axis=1)
    expected_first = np.asarray(gate.FIRST_READ)[base['group'] % len(gate.SCHEDULES)]
    audit.exact(first, expected_first, f'{group}: observed schedule-specific first read')
    rows = np.arange(len(x))
    for rule in ('replay', 'guarded'):
        path = paths[rule]
        audit.exact(path['first_read_tick'], first, f'{group}/{rule}: first read archive')
        audit.exact(path['replay_steps'], np.where(path['reset_steps'][rows, first], first, 0),
                    f'{group}/{rule}: only actual replay steps counted')
    for key in original:
        if history == 'two':
            for rule in ('replay', 'guarded'):
                audit.exact(paths[rule][key], original[key], f'{group}/{rule}: exact two-write no-op {key}')
        elif key != 'match_confidence':
            canceled = vetoed[:len(original[key])]
            if key in DISCRETE or original[key].dtype.kind in 'biu':
                audit.exact(guarded[key][canceled], original[key][canceled], f'{group}: veto exact original {key}')
                audit.exact(guarded[key][~canceled], replay[key][~canceled], f'{group}: unchanged trigger exact {key}')
            else:
                audit.close(guarded[key][canceled], original[key][canceled], f'{group}: veto original {key}')
                audit.close(guarded[key][~canceled], replay[key][~canceled], f'{group}: unchanged trigger {key}')
    left, right = data.pairs(base).T
    missing_pairs = np.isin(base['selected'][left], MISSING[history])
    left, right = left[missing_pairs], right[missing_pairs]
    audit.exact(x[left], x[right], f'{group}: missing-label inputs identical')
    audit.exact(base['y'][left], 1-base['y'][right], f'{group}: missing labels opposite')
    for rule in RULES:
        audit.exact(paths[rule]['counts'][left], paths[rule]['counts'][right], f'{group}/{rule}: exact absent-label counts')
    return int(vetoed.sum()), len(left)


def single_expectations(audit, geometry, base, x, w, paths):
    schedule = base['group'] % len(gate.SCHEDULES)
    old_trigger = paths['replay']['replay_steps'] > 0
    entries = []
    for code, name in enumerate(gate.SCHEDULES):
        for sample in np.flatnonzero(old_trigger & (schedule == code))[:4]:
            audit.guard()
            independent = independent_paths(x[sample:sample+1], w)
            for rule in ('replay', 'guarded'):
                for key in ('counts', 'resources', 'replay_steps', 'replay_resources'):
                    audit.exact(independent[rule][key][0], paths[rule][key][sample],
                                f'{geometry}/{name}/{sample}: exact isolated native {rule}/{key}')
            compared = independent['guarded']['evicted_comparisons'][0]
            audit.exact(compared, paths['guarded']['evicted_comparisons'][sample], f'{geometry}/{name}/{sample}: guard comparisons')
            entries.append(dict(geometry=geometry, sample=int(sample), schedule=name,
                                replay_steps=int(independent['replay']['replay_steps'][0]),
                                guarded_steps=int(independent['guarded']['replay_steps'][0]),
                                evicted_comparisons=int(compared.sum())))
    return entries


def verify_singles(audit, measured, expected):
    audit.check(isinstance(measured, list) and len(measured) == len(expected), 'complete isolated timing samples')
    keys = {'order', 'replay_call_seconds', 'guarded_call_seconds', 'replay_seconds', 'guard_seconds', 'guarded_replay_seconds'}
    total = 0.
    for i, (row, expected_row) in enumerate(zip(measured, expected)):
        context = f'isolated record {i}'
        audit.check(set(row) == set(expected_row) | keys and all(row[key] == value for key, value in expected_row.items()),
                    f'{context}: earliest old trigger and executed work')
        order = ['replay', 'guarded'] if i % 2 == 0 else ['guarded', 'replay']
        audit.check(row['order'] == order, f'{context}: alternating full-call order')
        replay_call, guarded_call = row['replay_call_seconds'], row['guarded_call_seconds']
        replay, guarded, comparison = row['replay_seconds'], row['guarded_replay_seconds'], row['guard_seconds']
        audit.check(np.isfinite([replay_call, guarded_call, replay, guarded, comparison]).all()
                    and 0. < replay <= replay_call and 0. < comparison <= guarded_call and guarded+comparison <= guarded_call,
                    f'{context}: measured subphases within full isolated call')
        audit.check((guarded > 0.) if row['guarded_steps'] else guarded == 0., f'{context}: no canceled replay time')
        total += replay_call+guarded_call
    return total


def verify_block_summary(audit, directory, n, account, rows, raw_bytes, isolated_seconds, base_bytes, spot_checks):
    block = read(directory/'BLOCK.json')
    phase_names = ('input_seconds', 'shared_seconds', 'verify_score_seconds', 'archive_seconds', 'isolation_seconds', 'other_seconds')
    expected_fields = {'examples', 'paths', 'total_seconds', 'work', 'metrics'} | set(phase_names)
    audit.check(set(block) == expected_fields and block['examples'] == n and block['paths'] == len(rows), 'complete science block ledger')
    phases = [block[name] for name in phase_names]
    audit.check(np.isfinite(phases).all() and min(phases) >= 0. and block['total_seconds'] > 0., 'valid science phase durations')
    audit.close(block['total_seconds'], sum(phases), 'sum of science phase durations', 1e-10)
    audit.close(block['shared_seconds'], sum(rows[f'{group}_original']['shared_group_seconds'] for group in GROUPS),
                'shared time counts each group once', 1e-10)
    audit.check(block['isolation_seconds'] >= isolated_seconds, 'isolated full calls contained in isolated phase')
    work = account['shared'] | dict(shared_calls=len(GROUPS)*((n+31)//32), spot_calls=account['spot_calls'],
                                    spot_online_sample_steps=account['spot_online_sample_steps'],
                                    spot_replay_sample_steps=account['spot_replay_sample_steps'],
                                    spot_native_macs=156*(account['spot_online_sample_steps']+account['spot_replay_sample_steps']),
                                    evicted_comparisons=account['guard_comparisons'], isolated_calls=account['isolated_calls'],
                                    isolated_online_sample_steps=account['isolated_online_sample_steps'],
                                    isolated_replay_sample_steps=account['isolated_replay_sample_steps'],
                                    isolated_native_macs=156*(account['isolated_online_sample_steps']+account['isolated_replay_sample_steps']),
                                    isolated_evicted_comparisons=account['isolated_evicted_comparisons'])
    audit.check(block['work'] == work, 'actual shared, production spot and isolated resource ledgers')
    metrics = dict(input_array_bytes=len(GROUPS)*n*T*6*8, base_array_bytes=base_bytes, weights_array_bytes=180*8,
                   retained_raw_array_bytes=raw_bytes, spot_checks=spot_checks, isolated_records=account['isolated_pairs'],
                   archived_bytes_before_block=sum(path.stat().st_size for path in directory.iterdir()
                                                   if path.is_file() and path.name != 'BLOCK.json'))
    actual = block['metrics']
    audit.check(set(actual) == set(metrics) | {'max_numeric_error'}, 'complete archive and spot metrics')
    audit.check(all(actual[key] == value for key, value in metrics.items()), 'exact science archive, isolated, and spot ledger')
    audit.check(np.isfinite(actual['max_numeric_error']) and 0. <= actual['max_numeric_error'] <= 5e-12,
                'production spot residual respects continuous tolerance')


def execute(directory, n, guard):
    """Read, independently reproduce and verify one trial before saving its audit."""
    started = time.perf_counter()
    directory = Path(directory)
    audit = Audit(guard)
    audit.check(n in (512, 256), 'allowed factorial fixture size')
    audit.check(not (directory/'AUDIT.json').exists(), 'one independent audit per trial')
    base, inputs, w = fixture(n)
    stored_base = load(audit, directory/'base.npz')
    audit.check(set(stored_base) == set(base), 'exact fixture base fields')
    for field, expected in base.items():
        audit.check(stored_base[field].dtype == expected.dtype, f'base {field}: exact dtype')
        audit.exact(stored_base[field], expected, f'base {field}: independent reproduction')
    stored_w = np.load(directory/'weights.npy', allow_pickle=False)
    audit.check(stored_w.shape == (180,) and stored_w.dtype == np.float64, 'fixed native weight format')
    audit.exact(stored_w, w, 'exact fresh initializer and fixed Oracle matrix')
    audit.check(np.all(np.bincount(base['group'] % len(gate.SCHEDULES), minlength=4) == n//4), 'balanced fixed schedules')

    row_list = read(directory/'ROWS.json')
    rows = {row['condition']: row for row in row_list}
    expected_conditions = {f'{group}_{rule}' for group in GROUPS for rule in RULES}
    audit.check(len(row_list) == len(rows) == len(expected_conditions) and set(rows) == expected_conditions,
                'all 24 paired condition rows')
    archive_names = {f'{condition}.npz' for condition in expected_conditions}
    audit.check({path.name for path in directory.glob('*.npz')} == archive_names | {'base.npz'}, 'exact raw archive list')

    group_totals, isolated = [], []
    raw_bytes = spot_replay = spot_checks = blocked = missing_pairs = guard_comparisons = 0
    ceilings = {'four': 1., 'three0': .875, 'three2': .875, 'two': .75}
    for group, x in inputs.items():
        audit.guard()
        geometry, history = group.split('_', 1)
        audit.check(data.ceiling(x, base['y']) == ceilings[history], f'{group}: independent information bound')
        first, first_scores, first_weights = gate.first_read_match(x, w)
        independent = independent_paths(x, w)
        work = group_work(independent)
        group_totals.append(work)
        archived = {}
        for rule in RULES:
            condition = f'{group}_{rule}'
            path = directory/f'{condition}.npz'
            raw = load(audit, path)
            raw_bytes += sum(array.nbytes for array in raw.values())
            expected = independent[rule] | dict(first_read_scores=first_scores, first_read_weights=first_weights)
            compare_raw(audit, raw, expected, condition)
            spot_checks += len(independent[rule])+len(set(independent[rule]) & {'cue_native_before', 'cue_native_after', 'replay_reference'})
            score = row_metrics(audit, rows[condition], raw, base, first_scores, first_weights, geometry, history, rule, work)
            if MISSING[history]:
                audit.check(score['missing_ba'] == .5*score['missing_coverage'], f'{condition}: missing-label BA equals half coverage')
            if rule != 'original':
                spot_replay += int(raw['replay_steps'][:8].sum())
            if rule == 'guarded':
                guard_comparisons += int(raw['evicted_comparisons'].sum())
            archived[rule] = raw
        audit.check(len({rows[f'{group}_{rule}']['shared_group_seconds'] for rule in RULES}) == 1,
                    f'{group}: one shared group measurement for its three outputs')
        vetoes, duplicates = verify_scope(audit, archived, x, base, first, history, group)
        blocked += vetoes
        missing_pairs += duplicates
        if history == 'four':
            isolated.extend(single_expectations(audit, geometry, base, x, w, archived))
    audit.check(missing_pairs == n, 'all paired missing-label inputs across geometries')
    observed_isolated = read(directory/'MICROBENCH.json')
    isolated_seconds = verify_singles(audit, observed_isolated, isolated)
    isolated_steps = sum(row['replay_steps']+row['guarded_steps'] for row in isolated)
    isolated_comparisons = sum(row['evicted_comparisons'] for row in isolated)
    shared = {key: sum(row[key] for row in group_totals) for key in group_totals[0]}
    account = dict(shared=shared, spot_calls=len(GROUPS)*len(RULES), spot_online_sample_steps=len(GROUPS)*len(RULES)*8*T,
                   spot_replay_sample_steps=spot_replay, isolated_pairs=len(isolated), isolated_calls=2*len(isolated),
                   isolated_online_sample_steps=2*len(isolated)*T, isolated_replay_sample_steps=isolated_steps,
                   isolated_evicted_comparisons=isolated_comparisons, guard_comparisons=guard_comparisons,
                   blocked_triggers=blocked, missing_opposite_label_pairs=missing_pairs)
    verify_block_summary(audit, directory, n, account, rows, raw_bytes, isolated_seconds,
                         sum(value.nbytes for value in base.values()), spot_checks)
    audit.guard(disk=True)
    result = dict(status='PASS', examples=n, paths=len(rows), checks=audit.checks, max_numeric_error=audit.max_error,
                  retained_raw_array_bytes=raw_bytes, work=account,
                  prior_training_audit_forecast_seconds=PRIOR_TRAINING_AUDIT_ESTIMATE,
                  wall_seconds=time.perf_counter()-started,
                  scope='Independent complete fixture, exact discrete replay, first-eight traces, native scoring, '
                        'guard scope, isolated measurements and block work. Prior training audit time is a separate estimate.')
    with (directory/'AUDIT.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return result
