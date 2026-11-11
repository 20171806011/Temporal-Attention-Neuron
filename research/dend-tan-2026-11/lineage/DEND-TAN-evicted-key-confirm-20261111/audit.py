"""Independent audit of new fits and held-out evicted-key guard confirmation."""

import ctypes
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import independent_event as event
import independent_guard as guard
import independent_guard_confirm as geometry_reference
import independent_confirm as training_reference
import independent_learned_data as train_data
import independent_warm as native
import independent_capacity_data as data


ROOT = Path(__file__).resolve().parent
ARM = 'LEARN'
T, Q, CUE, BLOCKS = 36, 28, 25, 12
SCIENCE_CAP, TRAIN_CAP = 180, 600
UPDATES, TRAIN_N = 600, 128
HISTORIES, RULES = guard.HISTORIES, guard.RULES
GEOMETRIES = guard.GEOMETRIES
CONDITIONS = tuple(f'{geometry}_{history}_{rule}' for geometry in GEOMETRIES for history in HISTORIES for rule in RULES)
EVALUATIONS = len(CONDITIONS)*BLOCKS
EVICTED = {'three0': 0, 'three2': 1}
MEMORY_CAP = 1024**3
ARTIFACT_CAP = 3*1024**3//2


class ProcessCounters(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
        (field, ctypes.c_size_t) for field in (
            'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
            'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]


def peak_memory():
    counter = ProcessCounters()
    counter.cb = ctypes.sizeof(counter)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(counter), counter.cb):
        raise OSError('Could not read audit peak working set')
    return counter.PeakWorkingSetSize


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


class Audit:
    def __init__(self):
        self.start = time.perf_counter()
        self.cpu_start = time.process_time()
        self.checks = 0
        self.stage = 'provenance'
        self.max_error = 0.

    def guard(self):
        if time.perf_counter() - self.start > 300:
            raise TimeoutError(f'300 second independent audit cap during {self.stage}')
        if peak_memory() > MEMORY_CAP:
            raise MemoryError('1 GiB audit peak working-set limit')

    def check(self, condition, context):
        self.checks += 1
        if time.perf_counter() - self.start > 300:
            raise TimeoutError(f'300 second independent audit cap during {self.stage}')
        assert bool(condition), context

    def exact(self, actual, expected, context):
        self.check(np.array_equal(actual, expected), context)

    def close(self, actual, expected, context, tolerance=1e-8):
        actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.check(actual.shape == expected.shape, f'{context}: shape')
        self.check(np.isfinite(actual).all() and np.isfinite(expected).all(), f'{context}: finite')
        error = float(np.max(np.abs(actual-expected))) if actual.size else 0.
        self.max_error = max(self.max_error, error)
        self.check(error <= tolerance, f'{context}: error {error:.17g} > {tolerance}')
        return error

    def elapsed(self):
        return dict(wall_seconds=time.perf_counter()-self.start, cpu_seconds=time.process_time()-self.cpu_start)


def regenerated_base(audit, path, n, seed):
    actual = load(path)
    expected = data.make_base(n, seed)
    audit.check(set(actual) == set(expected), f'{path.name}: data fields')
    for key in expected:
        audit.exact(actual[key], expected[key], f'{path.name}: independent {key}')
    return expected


def primary_inference(audit, effects, blocks):
    audit.stage = 'primary inference'
    rng = np.random.default_rng(4087900001)
    draws = rng.integers(0, blocks, (50_000, blocks))
    result = []
    thresholds = {'evicted_gain': 0., 'recent_change': -.02}
    for name, block_values in effects.items():
        values = np.asarray(block_values)
        audit.check(values.shape == (blocks,) and np.isfinite(values).all(), f'{name}: complete primary values')
        bootstrap = np.sum(values[draws], axis=1)/blocks
        low, high = np.quantile(bootstrap, [.0125, .9875])
        direction = 'positive' if low > 0 else 'negative' if high < 0 else 'undetermined'
        result.append(dict(effect=name, blocks=blocks, values=block_values, mean=float(values.mean()),
                           sample_sd=float(values.std(ddof=1)), minimum=float(values.min()), maximum=float(values.max()),
                           positive=int((values > 0).sum()), zero=int((values == 0).sum()),
                           negative=int((values < 0).sum()), ci97_5=[float(low), float(high)], direction=direction,
                           threshold=thresholds[name], passes=bool(low > thresholds[name])))
    return result


def task_scores(raw, base, slots):
    counts, y, selected = raw['counts'], base['y'], base['selected']
    covered = counts[:, 0] != counts[:, 1]
    correct = covered & (np.argmax(counts, axis=1) == y)
    target_slot = 3-selected
    available = target_slot < slots
    cue_scores = raw['first_read_scores']
    maxima = cue_scores.max(axis=1, keepdims=True)
    unique = (cue_scores == maxima).sum(axis=1) == 1
    routes = available & unique & (cue_scores.argmax(axis=1) == target_slot)
    target_weight = np.zeros(len(y))
    target_weight[available] = raw['first_read_weights'][available, target_slot[available]]

    def balanced(mask):
        return float(sum(correct[mask & (y == label)].mean() for label in (0, 1))/2)

    older = selected < 2
    scores = dict(balanced_accuracy=balanced(np.ones(len(y), dtype=bool)), older_ba=balanced(older),
                  recent_ba=balanced(~older), coverage=float(covered.mean()), older_coverage=float(covered[older].mean()),
                  recent_coverage=float(covered[~older].mean()), routing_accuracy=float(routes.mean()),
                  older_routing_accuracy=float(routes[older].mean()), recent_routing_accuracy=float(routes[~older].mean()),
                  target_weight=float(target_weight.mean()), older_target_weight=float(target_weight[older].mean()),
                  recent_target_weight=float(target_weight[~older].mean()), available_fraction=float(available.mean()))
    positions = [selected == position for position in range(4)]
    scores['ba_by_position'] = [balanced(mask) for mask in positions]
    for key, vector in (('coverage', covered), ('routing', routes), ('target_weight', target_weight)):
        scores[f'{key}_by_position'] = [float(vector[mask].mean()) for mask in positions]
    return scores


def compare_raw(audit, saved, expected, context):
    audit.check(set(saved) == set(expected), f'{context}: raw fields')
    discrete = {'counts', 'resources', 'spikes', 'read_onset', 'reset_steps', 'eviction_seen', 'key_bank', 'write_gate',
                'logged_write_values', 'logged_write_ticks', 'logged_nonwrite_values', 'logged_nonwrite_ticks',
                'logged_nonwrite_count', 'first_read_tick', 'replay_steps', 'replay_resources',
                'evicted_keys', 'evicted_comparisons'}
    for key in saved:
        audit.check(saved[key].shape == expected[key].shape and saved[key].dtype == expected[key].dtype,
                    f'{context}: {key} shape and dtype')
        if key in discrete:
            audit.exact(saved[key], expected[key], f'{context}: exact {key}')
        else:
            audit.close(saved[key], expected[key], f'{context}: {key}', 5e-12)
    for key in ('cue_native_before', 'cue_native_after'):
        audit.exact(saved[key][..., 20:], expected[key][..., 20:], f'{context}: exact prior pulse {key}')
    counts = saved['counts']
    audit.check(np.all((0 <= counts) & (counts <= 8)) and np.array_equal(counts, np.rint(counts)),
                f'{context}: native count bounds')


def expected_gates(x):
    observed, marker = x[..., 2:4], x[..., 5:6]
    gate = (1.-marker)*np.minimum(1., np.linalg.norm(observed, axis=-1, keepdims=True))
    reading = gate[..., 0] > 0.
    onset = reading & ~np.column_stack((np.zeros(len(x), dtype=bool), reading[:, :-1]))
    return gate, onset


def cost(history, rule, raw, eligible):
    writes = {'four': 4, 'three0': 3, 'three2': 3, 'two': 2}[history]
    replay = rule != 'original'
    live_bytes = steps = 0.
    if replay:
        live_events = (raw['logged_write_ticks'] >= 0).sum(axis=1)+raw['logged_nonwrite_count']
        live_bytes = float((56*live_events+8).mean())
        steps = float(raw['replay_steps'].mean())
    return dict(parameters=180, allocated_parameter_slots=180, fixed_match_constants=0, spiking_cells=8,
                graded_branches=12, continuous_state_slots=48, key_slots=4, value_slots=24,
                persistent_array_bytes=448, dense_forward_macs=156*T, graded_transfers=12*T,
                matcher_dense_macs=8*T, matcher_logit_scale_multiplies=2*T, softmax_exponentials=2*T,
                softmax_divisions=2*T, query_norm_square_multiplies=2*T, query_norm_adds=T, query_norm_sqrts=T,
                value_tanh_evaluations=12*T, value_decay_multiplies=24*T, value_read_multiplies=24*T,
                value_read_adds=12*T, injection_multiplies=12*T, fifo_scalar_writes=28*writes,
                hard_equalities=0, hard_tie_adds=0, hard_divisions=0, extra_value_read_multiplies=0,
                extra_value_read_adds=0, extra_injection_multiplies=0, normalization_norms=0,
                normalization_square_multiplies=0, normalization_sum_adds=0, normalization_sqrts=0,
                normalization_ratio_divisions_upper=0, normalization_scale_multiplies=0,
                logical_latch_bits=int(rule != 'original'), prototype_onset_unit_steps=T,
                eviction_latch_bits=int(rule != 'original'), prototype_eviction_unit_steps=T,
                occupancy_checks_per_episode=writes, stored_reference_scalars=0,
                cosine_norms_per_episode=3*eligible, cosine_denominator_multiplies_per_episode=2*eligible,
                cosine_divisions_upper_per_episode=2*eligible, reset_scalar_writes_mean=26*raw['reset_steps'].sum()/len(raw['counts']),
                replay_allocated_log_bytes=2136 if replay else 0, replay_live_log_bytes_mean=live_bytes,
                replay_logical_first_read_bits=int(replay), replay_prototype_first_read_tick_bytes=8 if replay else 0,
                replay_native_steps_mean=steps, replay_native_macs_mean=156*steps,
                evicted_key_bytes=32 if rule == 'guarded' else 0, evicted_key_metadata_bytes=0,
                evicted_key_cosines_mean=float(raw['evicted_comparisons'].sum()/len(raw['counts'])) if rule == 'guarded' else 0,
                evicted_key_log_writes_mean=(writes-2) if rule == 'guarded' else 0)


def verify_row(audit, row, raw, base, seed, geometry, history, rule):
    ident, condition = f's{seed}_{ARM}', f'{geometry}_{history}_{rule}'
    context = f'{ident}/{condition}'
    identity = dict(id=ident, seed=seed, arm=ARM, condition=condition, geometry=geometry, history=history, rule=rule, T=T, q=Q, slots=2)
    scores = task_scores(raw, base, 2)
    selected, y = base['selected'], base['y']
    n = len(y)
    reset = raw['reset_steps'].any(axis=1)
    reset_count, onset_count = int(raw['reset_steps'].sum()), int(raw['read_onset'].sum())
    scores.update(reset_count=reset_count, onset_count=onset_count,
                  reset_fraction=float(reset.mean()), older_reset_fraction=float(reset[selected < 2].mean()),
                  recent_reset_fraction=float(reset[selected >= 2].mean()),
                  reset_by_position=[float(reset[selected == pos].mean()) for pos in range(4)],
                  eviction_fraction=float(raw['eviction_seen'].any(axis=1).mean()))
    for field, position in (('evicted', EVICTED.get(history)), ('missing', 1-EVICTED[history] if history in EVICTED else None)):
        scores[f'{field}_ba'] = None if position is None else scores['ba_by_position'][position]
        scores[f'{field}_coverage'] = None if position is None else scores['coverage_by_position'][position]
    replay_resources = raw['replay_resources'].mean(axis=0) if rule != 'original' else np.zeros(4)
    scores['replay_resources_mean'] = replay_resources
    timing_names = {'replay_wall_seconds', 'replay_batches', 'replay_scratch_peak_bytes',
                    'evaluation_wall_seconds', 'guard_wall_seconds', 'guard_batches'}
    audit.check(set(row) == set(identity) | set(scores) | timing_names | {'mean_resources', 'cost_per_episode'},
                f'{context}: row schema')
    for key, value in identity.items():
        audit.check(row[key] == value, f'{context}: {key}')
    for key, value in scores.items():
        if value is None:
            audit.check(row[key] is None, f'{context}: {key} absent outside three writes')
        else:
            audit.close(row[key], value, f'{context}: native {key}', 1e-13)
    audit.close(row['mean_resources'], raw['resources'][:, :4].mean(axis=0), f'{context}: resource means', 1e-12)
    opportunities = int(np.count_nonzero(raw['read_onset'] & raw['eviction_seen'])) if rule != 'original' else 0
    audit.check(row['cost_per_episode'] == cost(history, rule, raw, opportunities/n), f'{context}: per-path structural costs')
    steps = raw['replay_steps'] if rule != 'original' else np.zeros(n, dtype=np.int64)
    chunks = steps.reshape(-1, 32)
    batches = scratch = 0
    for onset in guard.FIRST_READ:
        sizes = (chunks == onset).sum(axis=1)
        batches += int(np.count_nonzero(sizes))
        scratch = max(scratch, int(sizes.max(initial=0))*onset*6*8)
    wall = row['replay_wall_seconds']
    audit.check(np.isfinite(wall) and (wall > 0 if batches else wall == 0.), f'{context}: replay-only time')
    audit.check(row['replay_batches'] == batches and row['replay_scratch_peak_bytes'] == scratch,
                f'{context}: exact replay batches and prefix scratch bytes')
    evaluation_wall, guard_wall = row['evaluation_wall_seconds'], row['guard_wall_seconds']
    guard_batches = 0
    if rule == 'guarded':
        candidates = (raw['match_confidence'] >= native.COSINE_THRESHOLD).reshape(-1, 32, T)
        guard_batches = int(np.count_nonzero(candidates.any(axis=1)))
    audit.check(np.isfinite(guard_wall) and (guard_wall > 0. if guard_batches else guard_wall == 0.),
                f'{context}: guard-only comparison duration')
    audit.check(row['guard_batches'] == guard_batches, f'{context}: exact guard comparison call count')
    audit.check(np.isfinite(evaluation_wall) and evaluation_wall > 0. and wall+guard_wall <= evaluation_wall+1e-10,
                f'{context}: contained substage timings')
    return scores, dict(cosine_onsets=opportunities, replay_sample_steps=int(steps.sum()),
                        replay_examples=int(np.count_nonzero(steps)), replay_wall_seconds=wall, replay_batches=batches,
                        replay_scratch_peak_bytes=scratch, evaluation_wall_seconds=evaluation_wall,
                        guard_wall_seconds=guard_wall, guard_batches=guard_batches)


def verify_timing(audit, n):
    precheck, timing = read(ROOT/'PRECHECK.json'), read(ROOT/'TIMING.json')
    audit.check(precheck['status'] == timing['status'] == 'PASS'
                and precheck['scientific_data_generated'] is timing['scientific_data_generated'] is False,
                'completed timing fixtures precede science')
    audit.check(timing['blocks'] == BLOCKS and timing['examples_per_block'] == n, 'selected sample size')
    training = precheck['train_fixtures']
    seeds = [(row['data_seed'], row['init_seed']) for row in training]
    audit.check(seeds == [(4099903101, 4099903102), (4099903103, 4099903104)], 'exact paired training fixtures')
    audit.check(precheck['test_seed'] == 4099903105, 'exact repeated test fixture seed')
    for row in training:
        steps = np.asarray(row['step_seconds'])
        audit.check(row['updates'] == 24 and steps.shape == (24,) and np.isfinite(steps).all() and np.all(steps > 0.),
                    'all fixed fixture update timings')
        audit.close(row['tail12_median_seconds'], np.median(steps[-12:]), 'fixture tail median')
        audit.check(row['training_audit_seconds'] > 0., 'positive archived-update audit timing')
    train_forecast = 1.5*BLOCKS*UPDATES*max(row['tail12_median_seconds'] for row in training)+30
    training_audit = BLOCKS*UPDATES/24*max(row['training_audit_seconds'] for row in training)
    audit.close(timing['training_forecast_seconds'], train_forecast, 'conservative training forecast')
    audit.close(timing['training_audit_forecast_seconds'], training_audit, 'conservative training audit forecast')
    audit.check(train_forecast < TRAIN_CAP, 'training forecast inside cap')
    candidates = timing['candidates']
    audit.check(candidates == precheck['test_fixture_candidates'], 'both timing archives include all trials')
    audit.exact([item['examples'] for item in candidates], [512] if n == 512 else [512, 256], 'larger size tried first')
    for idx, item in enumerate(candidates):
        trials = item['trials']
        audit.check(item['fixed_gap_seconds'] == 30 and [row['trial'] for row in trials] == [1, 2], 'exactly two timed trials')
        for key in ('checks', 'vetoes', 'evicted_comparisons', 'isolated_records', 'isolated_replay_steps',
                    'isolated_replay_macs', 'isolated_evicted_comparisons'):
            audit.check(trials[0][key] == trials[1][key], f'same deterministic fixture {key}')
        earlier = None
        for trial in trials:
            start, finish = (trial[f'monotonic_{stage}_seconds'] for stage in ('start', 'finish'))
            start_utc, finish_utc = (datetime.fromisoformat(trial[f'utc_{stage}']) for stage in ('start', 'finish'))
            audit.check(0. <= start < finish and start_utc <= finish_utc, 'bounded monotonic and UTC trial interval')
            audit.check(all(np.isfinite(trial[f'{kind}_seconds']) and trial[f'{kind}_seconds'] > 0. for kind in ('science', 'audit')),
                        'positive raw science and audit trial durations')
            gap = trial['gap_from_previous_seconds']
            if earlier is None:
                audit.check(gap is None, 'first trial has no predecessor')
            else:
                audit.check(gap >= 30., 'fixed minimum gap completed')
                audit.close(gap, start-earlier, 'measured gap from previous trial', 1e-6)
            earlier = finish
            for stage in ('before', 'after'):
                snapshot = trial[f'load_{stage}']
                load_utc = datetime.fromisoformat(snapshot['utc'])
                audit.check(snapshot['status'] in ('ok', 'unavailable'), 'archived host load observation')
                if snapshot['status'] == 'ok':
                    percentages = snapshot['percentages']
                    audit.check(isinstance(percentages, list) and bool(percentages)
                                and all(np.isfinite(value) and 0. <= value <= 100. for value in percentages),
                                'bounded observed host load values')
                else:
                    audit.check(snapshot['percentages'] is None and bool(snapshot.get('error')),
                                'unavailable host observation preserved')
                audit.check(load_utc <= start_utc if stage == 'before' else finish_utc <= load_utc,
                            'host load observations outside the timed interval')
        for kind in ('science', 'audit'):
            worst = max(trial[f'{kind}_seconds'] for trial in trials)
            audit.close(item[f'{kind}_seconds'], worst, f'{kind}: slower trial controls decision')
            forecast = 3*BLOCKS*worst+30+(training_audit if kind == 'audit' else 0.)
            audit.close(item[f'{kind}_forecast_seconds'], forecast, f'{kind}: conservative complete forecast')
        eligible = item['science_forecast_seconds'] < SCIENCE_CAP and item['audit_forecast_seconds'] < 300
        audit.check(eligible == (idx == len(candidates)-1), 'first admissible size within both forecast caps')


def provenance(audit):
    frozen = read(ROOT/'FREEZE.json')
    n = frozen['examples_per_block']
    audit.check(frozen['blocks'] == BLOCKS and frozen['evaluations'] == EVALUATIONS and n in (512, 256)
                and frozen['training_examples'] == TRAIN_N and frozen['updates_per_fit'] == UPDATES
                and frozen['scientific_data_generated'] is False, 'fixed train and test scope')
    required = {'PROTOCOL.md', 'REPRODUCE.md', 'SOURCE_LOCK.json', 'PRECHECK.json', 'TIMING.json',
                'run.py', 'trainer.py', 'independent_confirm.py', 'independent_guard_confirm.py',
                'audit.py', 'summarize.py', 'report.py'}
    audit.check(required <= set(frozen['files']), 'frozen train, test, audit, and report sources')
    audit.check(all('/' not in path and path.endswith(('.py', '.json', '.md')) for path in frozen['files']),
                'freeze excludes weights and science output directories')
    for name, expected in frozen['files'].items():
        audit.check(digest(ROOT/name) == expected, f'frozen {name}')
    verify_timing(audit, n)

    lock = read(ROOT/'SOURCE_LOCK.json')
    directories = {'replay': '../DEND-TAN-evicted-key-20261111', 'training': '../DEND-TAN-replay-confirm-20261111'}
    audit.check(set(lock) == {'sources', 'files'} and set(lock['sources']) == set(directories), 'exact dual-source scope')
    manifests = {}
    for name, directory in directories.items():
        previous = ROOT/directory
        record = lock['sources'][name]
        audit.check(record['directory'] == directory, f'{name}: fixed predecessor directory')
        audit.check(digest(previous/'MANIFEST.json') == record['manifest_sha256'], f'{name}: exact predecessor seal')
        manifest = read(previous/'MANIFEST.json')['files']
        manifests[name] = manifest
        for artifact in ('AUDIT.json', 'SCIENCE_SUMMARY.json', 'DELIVERY_CHECKS.json', 'RESULTS.json'):
            audit.check(digest(previous/artifact) == manifest[artifact]['sha256'], f'{name}: sealed prior {artifact}')
        old_audit, old_science = read(previous/'AUDIT.json'), read(previous/'SCIENCE_SUMMARY.json')
        delivery, result = read(previous/'DELIVERY_CHECKS.json'), read(previous/'RESULTS.json')
        evaluations, fits = (288, 0) if name == 'replay' else (96, 12)
        audit.check(old_audit['status'] == delivery['status'] == 'PASS' and old_science['status'] == 'COMPLETE'
                    and old_science['evaluations'] == delivery['verified_evaluations'] == evaluations
                    and old_science['blocks'] == BLOCKS and old_science['fits'] == fits, f'{name}: complete delivered predecessor')
        audit.check(old_audit['adoption_supported'] is result['decision']['both_pass'] is True, f'{name}: predecessor decision sealed')
    replay_sources = {'capacity_data.py', 'capacity_metrics.py', 'capacity_model.py', 'independent_capacity_data.py',
                      'independent_capacity.py', 'learned_model.py', 'population.py', 'reference.py', 'original_tan.py',
                      'reset_model.py', 'independent_reset.py', 'warm_model.py', 'independent_warm.py', 'event_model.py',
                      'independent_event.py', 'confirm_data.py', 'guard_model.py', 'independent_guard.py'}
    training_sources = {'learned_data.py', 'independent_learned_data.py', 'independent_learned.py',
                        'training_recipe_source.py', 'trainer.py', 'independent_confirm.py'}
    expected = {name: 'replay' for name in replay_sources} | {name: 'training' for name in training_sources}
    audit.check(set(lock['files']) == set(expected), '24 exact sealed source copies, with no pretrained parameters')
    for name, source in expected.items():
        item = lock['files'][name]
        audit.check(item['source'] == source and item['path'] == name, f'{name}: exact provenance entry')
        previous = ROOT/directories[source]/name
        audit.check(item['sha256'] == frozen['files'][name] == manifests[source][name]['sha256'] == digest(previous),
                    f'{name}: byte-identical sealed implementation')
    for name in ('learned_model.py', 'population.py', 'reference.py', 'original_tan.py', 'confirm_data.py'):
        audit.check(manifests['replay'][name]['sha256'] == manifests['training'][name]['sha256'],
                    f'{name}: matching predecessor implementation')
    audit.guard()
    return n


def training(audit):
    audit.stage = 'training archive'
    directory = ROOT/'train'
    lock_path = directory/'TRAIN_LOCK.json'
    train_lock = read(lock_path)
    audit.check(train_lock['test_generated'] is False and train_lock['blocks'] == BLOCKS
                and train_lock['updates'] == BLOCKS*UPDATES, 'completed training lock predates test generation')
    expected_files = {'ROWS.json'} | {f'base_s{s}.npz' for s in range(BLOCKS)} | {f's{s}_{ARM}.npz' for s in range(BLOCKS)}
    audit.check(set(train_lock['files']) == expected_files, 'exact training archive lock set')
    audit.check({path.name for path in directory.iterdir() if path.is_file()} == expected_files | {'TRAIN_LOCK.json'},
                'exact complete training archive directory')
    for name, expected in train_lock['files'].items():
        audit.check(digest(directory/name) == expected, f'locked training archive {name}')
    rows = read(directory/'ROWS.json')
    index = {row['seed']: row for row in rows}
    audit.check(len(rows) == len(index) == BLOCKS and set(index) == set(range(BLOCKS)), 'all fresh training blocks')
    weights = {}
    update_steps, final_steps = UPDATES*TRAIN_N*T, TRAIN_N*T
    train_audit_checks = first_last = post_update = 0
    for seed in range(BLOCKS):
        audit.guard()
        ident, row = f's{seed}_{ARM}', index[seed]
        path = directory/f'base_s{seed}.npz'
        saved = load(path)
        base = train_data.make_base(TRAIN_N, 4087010000+100*seed)
        audit.check(set(saved) == set(base), f'{ident}: training base fields')
        for key in base:
            audit.exact(saved[key], base[key], f'{ident}: independent train data {key}')
        x, q = train_data.assemble(base)
        audit.check(q == Q and train_data.check(base, x)['ceiling'] == 1., f'{ident}: nominal training data')
        archive = load(directory/f'{ident}.npz')
        audit.exact(archive['initial'], training_reference.initialize(4087030000+seed), f'{ident}: independently generated new weights')
        checked = training_reference.verify_training_archive(x, base['y'], q, archive)
        train_audit_checks += checked['checks']
        audit.max_error = max(audit.max_error, checked['max_numeric_error'])
        first_last += checked['endpoint_replays']
        post_update += 1
        audit.check(checked['updates'] == UPDATES, f'{ident}: all 600 updates')
        meta = dict(id=ident, seed=seed, arm=ARM, T=T, q=Q, examples=TRAIN_N, updates=UPDATES,
                    dense_forward_macs=156*(update_steps+final_steps), dense_backward_macs=240*update_steps)
        audit.check(set(row) == set(meta) | {'seconds', 'cpu_seconds', 'first_loss', 'last_loss', 'final_loss'},
                    f'{ident}: exact train row schema')
        for key, value in meta.items():
            audit.check(row[key] == value, f'{ident}: fixed training {key}')
        audit.check(0 < archive['step_seconds'].sum() <= row['seconds'] <= TRAIN_CAP and row['cpu_seconds'] > 0.,
                    f'{ident}: positive fit and step durations')
        audit.close(row['first_loss'], archive['history'][0, 0], f'{ident}: first objective', 1e-10)
        audit.close(row['last_loss'], archive['history'][-1, 0], f'{ident}: last update objective', 1e-10)
        audit.close(row['final_loss'], checked['final_loss'], f'{ident}: final parameter objective', 1e-10)
        weights[ident] = archive['theta']
        print(f'Independently verified new LEARN training block {seed+1}/{BLOCKS}', flush=True)

    summary = read(ROOT/'TRAIN_SUMMARY.json')
    update_steps, final_steps = BLOCKS*update_steps, BLOCKS*final_steps
    expected = dict(blocks=BLOCKS, fits=BLOCKS, updates=BLOCKS*UPDATES, examples_per_fit=TRAIN_N,
                    update_sample_steps=update_steps, final_forward_sample_steps=final_steps,
                    dense_forward_macs=156*(update_steps+final_steps), dense_backward_macs=240*update_steps,
                    matcher_forward_macs=8*(update_steps+final_steps), matcher_backward_unit_steps=2*update_steps)
    audit.check(summary['status'] == 'COMPLETE' and summary['cap_seconds'] == TRAIN_CAP
                and 0 < summary['wall_seconds'] <= TRAIN_CAP and summary['cpu_seconds'] > 0., 'training duration and completion')
    audit.check(0 < summary['peak_working_set_bytes'] <= MEMORY_CAP, 'training peak memory cap')
    audit.check(0 < summary['artifact_bytes_at_train_end'] <= ARTIFACT_CAP, 'training artifact cap')
    for key, value in expected.items():
        audit.check(summary[key] == value, f'training ledger {key}')
    audit.check(first_last == 2*BLOCKS and post_update == BLOCKS, 'all first, last and final forward verifications')
    diagnostics = dict(optimizer_updates=BLOCKS*UPDATES, training_checks=train_audit_checks,
                       first_last_training_replays=first_last, final_training_replays=post_update)
    return weights, digest(lock_path), diagnostics


def archive_scope(audit, raw, x, base, first, history, rule, context):
    gate, onset = expected_gates(x)
    n = len(x)
    sample = np.arange(n)
    schedule = base['group'] % len(guard.SCHEDULES)
    audit.exact(first, np.asarray(guard.FIRST_READ)[schedule], f'{context}: schedule-specific read edge')
    audit.exact(raw['read_onset'], onset, f'{context}: visible rising read gate')
    audit.check(np.all(onset.sum(axis=1) == 1) and onset[sample, first].all(), f'{context}: exactly one read')
    writes = np.asarray(guard.WRITE_TICKS)[schedule]
    first_eviction = writes[:, 2] if history == 'four' else writes[:, 3] if history in EVICTED else np.full(n, T)
    evicted = np.arange(T)[None] > first_eviction[:, None]
    audit.exact(raw['eviction_seen'], evicted, f'{context}: exact write-history latch')
    reset, confidence = raw['reset_steps'], raw['match_confidence']
    eligible = onset & evicted if rule != 'original' else np.zeros_like(onset)
    audit.check(np.all(confidence[~eligible] == native.NO_CONFIDENCE), f'{context}: match only at eligible first read')
    old_trigger = eligible & (confidence >= native.COSINE_THRESHOLD)
    expected = old_trigger
    if rule == 'guarded':
        lost, comparisons = raw['evicted_cosine'], raw['evicted_comparisons']
        audit.check(np.all(lost[~old_trigger] == -2.) and not comparisons[~old_trigger].any(),
                    f'{context}: compare evicted keys only for old triggers')
        valid = np.any(raw['evicted_keys'] != 0., axis=2).sum(axis=1)
        audit.exact(comparisons, np.where(old_trigger, valid[:, None], 0), f'{context}: compare only valid keys')
        expected = old_trigger & (confidence > lost)
    audit.exact(reset, expected, f'{context}: exact trigger')
    before, after = raw['cue_native_before'], raw['cue_native_after']
    audit.exact(before, after, f'{context}: legacy t25..27 archive occurs after replacement')
    audit.exact(before[:8, :, :20], raw['states'][:, CUE-1:Q-1], f'{context}: archived preceding native state')
    audit.exact(before[:8, :, 20:], raw['spikes'][:, CUE-1:Q-1, :6], f'{context}: archived preceding pulse')
    audit.close(raw['read_gate'], gate[:8], f'{context}: visible read gate trace', 5e-12)
    audit.exact(raw['write_gate'], x[:8, :, 5:6], f'{context}: visible write gate trace')
    audit.exact(raw['counts'][:8], raw['spikes'][:, Q:, 6:].sum(axis=1), f'{context}: original native counts')
    for archive_key, trace_key in (('cue_scores', 'scores'), ('cue_weights', 'weights'), ('cue_read_vectors', 'injections')):
        audit.exact(raw[archive_key][:8], raw[trace_key][:, CUE:Q], f'{context}: legacy cue {trace_key}')
    audit.close(raw['first_read_scores'][:8], raw['scores'][sample[:8], first[:8]],
                f'{context}: actual first-read score traces', 5e-12)
    audit.close(raw['first_read_weights'][:8], raw['weights'][sample[:8], first[:8]],
                f'{context}: actual first-read softmax traces', 5e-12)
    if rule != 'original':
        distractors = np.asarray(guard.DISTRACTOR_TICKS)[schedule]
        audit.exact(raw['first_read_tick'], first, f'{context}: no clock-forced onset')
        audit.exact(raw['logged_write_ticks'], writes[:, 2:], f'{context}: exact recent-write timestamps')
        audit.exact(raw['logged_write_values'], x[sample[:, None], writes[:, 2:]], f'{context}: exact recent-write vectors')
        audit.exact(raw['logged_nonwrite_count'], np.full(n, 2), f'{context}: all nonwrite events')
        audit.exact(raw['logged_nonwrite_ticks'][:, :2], distractors, f'{context}: nonwrite times including t0')
        audit.exact(raw['logged_nonwrite_values'][:, :2], x[sample[:, None], distractors], f'{context}: exact nonwrite values')
        audit.check(np.all(raw['logged_nonwrite_ticks'][:, 2:] == -1) and not raw['logged_nonwrite_values'][:, 2:].any(),
                    f'{context}: no cue, future or zero inputs logged')
        applied = reset[sample, first]
        audit.exact(raw['replay_steps'], np.where(applied, first, 0), f'{context}: exact replay duration in sample steps')
        audit.check(not raw['replay_reference'][~applied].any() and not raw['replay_resources'][~applied].any(),
                    f'{context}: no replay work off trigger')


def path_equivalence(audit, original, replay, history, first, context):
    n = len(first)
    same = ~replay['reset_steps'].any(axis=1)
    discrete = {'counts', 'resources', 'spikes', 'read_onset', 'reset_steps', 'eviction_seen', 'key_bank', 'write_gate'}
    for key, expected in original.items():
        actual = replay[key]
        if history == 'two':
            audit.exact(actual, expected, f'{context}: exact two-write no-op {key}')
        elif key != 'match_confidence':
            selection = same if len(actual) == n else same[:8]
            if key in discrete:
                audit.exact(actual[selection], expected[selection], f'{context}: exact untriggered {key}')
            else:
                audit.close(actual[selection], expected[selection], f'{context}: untriggered {key}', 5e-12)
    unchanged = ('key_bank', 'value_bank', 'scores', 'weights', 'read_gate', 'write_gate', 'injections',
                 'cue_scores', 'cue_weights', 'cue_read_vectors', 'first_read_scores', 'first_read_weights', 'eviction_seen')
    for key in unchanged:
        if key in discrete:
            audit.exact(replay[key], original[key], f'{context}: same recipient {key}')
        else:
            audit.close(replay[key], original[key], f'{context}: same recipient {key}', 5e-12)


def guard_equivalence(audit, paths, base, context):
    replay, guarded = paths['replay'], paths['guarded']
    old, new = replay['reset_steps'], guarded['reset_steps']
    audit.check(not (new & ~old).any(), f'{context}: guarded trigger is subset')
    audit.close(guarded['match_confidence'], replay['match_confidence'], f'{context}: same live-key confidence', 5e-12)
    audit.exact(new, old & (guarded['match_confidence'] > guarded['evicted_cosine']), f'{context}: strict evicted-key veto')
    canceled = (old & ~new).any(axis=1)
    n = len(canceled)
    discrete = {'counts', 'resources', 'spikes', 'read_onset', 'reset_steps', 'eviction_seen', 'key_bank', 'write_gate',
                'logged_write_values', 'logged_write_ticks', 'logged_nonwrite_values', 'logged_nonwrite_ticks',
                'logged_nonwrite_count', 'first_read_tick', 'replay_steps', 'replay_resources'}
    for field, expected in replay.items():
        same = ~canceled[:len(expected)]
        audit.check(len(expected) in (8, n), f'{context}: {field} trace support')
        if field in discrete:
            audit.exact(guarded[field][same], expected[same], f'{context}: unchanged trigger exact {field}')
        else:
            audit.close(guarded[field][same], expected[same], f'{context}: unchanged trigger {field}', 5e-12)
    for field in ('cue_native_before', 'cue_native_after', 'replay_reference'):
        same = ~canceled
        audit.exact(guarded[field][same, ..., 20:], replay[field][same, ..., 20:], f'{context}: unchanged prior pulses {field}')
    old_steps, new_steps = replay['replay_steps'], guarded['replay_steps']
    audit.exact(new_steps, np.where(canceled, 0, old_steps), f'{context}: exact canceled replay steps')
    by_position = [int(np.count_nonzero(canceled & (base['selected'] == position))) for position in range(4)]
    return by_position


def verify_microbench(audit, records, x, weights, base, cases, seed, geometry):
    context = f's{seed}/{geometry}: isolated timing'
    schedule = base['group'] % len(guard.SCHEDULES)
    replay, guarded = cases['replay'], cases['guarded']
    triggered = replay['replay_steps'] > 0
    chosen = [(name, int(index)) for code, name in enumerate(guard.SCHEDULES)
              for index in np.flatnonzero((schedule == code) & triggered)[:4]]
    audit.check([(row['schedule'], row['sample']) for row in records] == chosen, f'{context}: old trigger timing subset')
    shift = seed % len(RULES)
    order = [rule for rule in RULES[shift:]+RULES[:shift] if rule != 'original']
    fields = {'seed', 'geometry', 'sample', 'schedule', 'order', 'replay_seconds', 'guarded_replay_seconds',
              'guard_seconds', 'replay_steps', 'guarded_steps'}
    steps = comparisons = 0
    for row in records:
        audit.check(set(row) == fields and row['seed'] == seed and row['geometry'] == geometry, f'{context}: record schema')
        audit.check(row['order'] == order, f'{context}: paired rotated rule order')
        index = row['sample']
        old_steps, new_steps = row['replay_steps'], row['guarded_steps']
        audit.check(old_steps == replay['replay_steps'][index] and new_steps == guarded['replay_steps'][index],
                    f'{context}: actual paired sample steps')
        times = [row[key] for key in ('replay_seconds', 'guarded_replay_seconds', 'guard_seconds')]
        audit.check(np.isfinite(times).all() and times[0] > 0. and times[2] > 0.
                    and (times[1] > 0. if new_steps else times[1] == 0.), f'{context}: isolated time only when executed')
        single_x = x[index:index+1]
        original = native.replay(single_x, weights, ARM, Q, 'original')
        single_replay = event.replay(single_x, weights, Q)
        single_guard = guard.apply_guard(single_x, weights, single_replay, original)
        for rule, actual in (('replay', single_replay), ('guarded', single_guard)):
            for key in ('counts', 'resources', 'replay_resources', 'replay_steps'):
                audit.exact(actual[key][0], cases[rule][key][index], f'{context}: isolated/batch exact {rule} {key}')
        audit.exact(single_guard['evicted_comparisons'][0], guarded['evicted_comparisons'][index], f'{context}: paired guard comparisons')
        comparisons += int(single_guard['evicted_comparisons'].sum())
        steps += old_steps+new_steps
    return steps, comparisons


def verify_geometry(audit, base, geometric, geometry, context):
    for key, value in base.items():
        if key not in ('key_vectors', 'query_vectors') or geometry == 'orthogonal':
            audit.exact(geometric[key], value, f'{context}: paired {key}')
    angles = np.deg2rad((0., 90., 180., 270.) if geometry == 'orthogonal' else geometry_reference.NEAR_DEGREES)
    key = geometric['key_vectors']
    gram = np.einsum('gsi,gki->gsk', key, key, optimize=False)
    expected = np.broadcast_to(np.cos(angles[:, None]-angles[None, :]), gram.shape)
    audit.close(gram, expected, f'{context}: fixed inter-key relative angles', 5e-12)
    rotated = np.stack((-key[..., 1], key[..., 0]), axis=-1)
    audit.close(geometric['query_vectors']-rotated, base['query_noise'], f'{context}: paired rotated query noise', 5e-12)


def evaluate(audit, n, weights):
    audit.stage = 'guarded replay'
    rows = read(ROOT/'evaluate'/'ROWS.json')
    index = {(row['id'], row['condition']): row for row in rows}
    expected = {(f's{s}_{ARM}', condition) for s in range(BLOCKS) for condition in CONDITIONS}
    audit.check(len(rows) == len(index) == EVALUATIONS and set(index) == expected, 'complete 288 paired paths')
    files = {f'{ident}_{condition}.npz' for ident, condition in expected} | {f'test_s{s}.npz' for s in range(BLOCKS)}
    audit.check({p.name for p in (ROOT/'evaluate').glob('*.npz')} == files, 'exact paired raw archive set')
    measurements = read(ROOT/'MICROBENCH.json')
    micro = {(seed, geometry): [] for seed in range(BLOCKS) for geometry in GEOMETRIES}
    audit.check(isinstance(measurements, list), 'isolated timing archive list')
    for record in measurements:
        key = record['seed'], record['geometry']
        audit.check(key in micro, 'isolated timing paired scope')
        micro[key].append(record)
    effects = {'evicted_gain': [], 'recent_change': []}
    block_cancels = {geometry: {history: np.zeros(4, dtype=np.int64) for history in HISTORIES} for geometry in GEOMETRIES}
    ceilings = {'four': 1., 'three0': .875, 'three2': .875, 'two': .75}
    ledger = dict(input_array_bytes=0, retained_raw_array_bytes=0, reset_count=0, cosine_onsets=0,
                  replay_sample_steps=0, replay_examples=0, replay_wall_seconds=0., replay_batches=0,
                  replay_scratch_peak_bytes=0, evaluation_wall_seconds=0., guard_wall_seconds=0., guard_batches=0,
                  evicted_key_comparisons=0, evicted_key_log_writes=0)
    duplicates = micro_steps = micro_comparisons = 0
    for seed in range(BLOCKS):
        ident, base_seed = f's{seed}_{ARM}', 4087020000+100*seed
        w = weights[ident]
        base = regenerated_base(audit, ROOT/'evaluate'/f'test_s{seed}.npz', n, base_seed)
        schedule = base['group'] % len(guard.SCHEDULES)
        audit.check(np.all(np.bincount(schedule, minlength=len(guard.SCHEDULES)) == n//4), f'{ident}: balanced schedules')
        for geometry in GEOMETRIES:
            geometric = geometry_reference.geometry(base, geometry)
            verify_geometry(audit, base, geometric, geometry, f'{ident}/{geometry}')
            block_scores = {}
            for history in HISTORIES:
                x, q = guard.assemble(geometric, history)
                audit.check(q == Q and data.ceiling(x, base['y']) == ceilings[history], f'{ident}/{geometry}/{history}: input ceiling')
                ledger['input_array_bytes'] += x.nbytes
                first, score_matrix, softmax = guard.first_read_match(x, w)
                orig = native.replay(x, w, ARM, q, 'original')
                replayed = event.replay(x, w, q)
                expected_paths = dict(original=orig, replay=replayed, guarded=guard.apply_guard(x, w, replayed, orig))
                paths, scores = {}, {}
                for rule in RULES:
                    audit.guard()
                    condition = f'{geometry}_{history}_{rule}'
                    context = f'{ident}/{condition}'
                    raw = load(ROOT/'evaluate'/f'{ident}_{condition}.npz')
                    expected_raw = expected_paths[rule] | dict(first_read_scores=score_matrix, first_read_weights=softmax)
                    compare_raw(audit, raw, expected_raw, context)
                    archive_scope(audit, raw, x, base, first, history, rule, context)
                    score, accounting = verify_row(audit, index[ident, condition], raw, base, seed, geometry, history, rule)
                    for key, independent in task_scores(expected_raw, base, 2).items():
                        audit.close(index[ident, condition][key], independent, f'{context}: independent first-read score {key}', 5e-12)
                    if history != 'four':
                        left, right = data.pairs(base).T
                        if history in EVICTED:
                            absent = base['selected'][left] == 1-EVICTED[history]
                            left, right = left[absent], right[absent]
                            audit.check(score['missing_ba'] == .5*score['missing_coverage'], f'{context}: missing-position BA')
                        else:
                            audit.check(score['older_ba'] == .5*score['older_coverage'], f'{context}: missing-positions BA')
                        audit.exact(x[left], x[right], f'{context}: exact missing-label stimuli')
                        audit.exact(base['y'][left], 1-base['y'][right], f'{context}: opposite absent labels')
                        audit.exact(raw['counts'][left], raw['counts'][right], f'{context}: exact missing-label count pairs')
                        duplicates += len(left)
                    for key, value in accounting.items():
                        if key == 'replay_scratch_peak_bytes':
                            ledger[key] = max(ledger[key], value)
                        else:
                            ledger[key] += value
                    ledger['reset_count'] += score['reset_count']
                    ledger['retained_raw_array_bytes'] += sum(value.nbytes for value in raw.values())
                    if rule == 'guarded':
                        ledger['evicted_key_comparisons'] += int(raw['evicted_comparisons'].sum())
                        ledger['evicted_key_log_writes'] += int(np.any(raw['evicted_keys'] != 0., axis=2).sum())
                    paths[rule], scores[rule] = raw, score
                for rule in ('replay', 'guarded'):
                    path_equivalence(audit, paths['original'], paths[rule], history, first, f'{ident}/{geometry}/{history}/{rule}')
                block_cancels[geometry][history] += guard_equivalence(audit, paths, base, f'{ident}/{geometry}/{history}')
                block_scores[history] = scores
                if history == 'four':
                    steps, comparisons = verify_microbench(audit, micro[seed, geometry], x, w, base, paths, seed, geometry)
                    micro_steps += steps
                    micro_comparisons += comparisons
            if geometry == 'near':
                for name, field in (('evicted_gain', 'evicted_ba'), ('recent_change', 'recent_ba')):
                    changes = [block_scores[h]['guarded'][field]-block_scores[h]['replay'][field] for h in EVICTED]
                    effects[name].append(float(np.mean(changes)))
        print(f'Independently replayed evicted-key block {seed+1}/{BLOCKS}', flush=True)
    audit.check(ledger['cosine_onsets'] == 12*BLOCKS*n, 'all live cosine opportunities')
    audit.check(ledger['evicted_key_log_writes'] == 8*BLOCKS*n, 'all visible evicted-key log writes')
    audit.check(duplicates == 3*BLOCKS*n, 'all missing-label pairs')
    ledger.update(reset_scalar_writes=26*ledger['reset_count'], replay_native_macs=156*ledger['replay_sample_steps'],
                  microbench_records=len(measurements), microbench_evaluations=2*len(measurements),
                  microbench_forward_sample_steps=2*len(measurements)*T, microbench_replay_steps=micro_steps,
                  microbench_replay_native_macs=156*micro_steps, microbench_evicted_key_comparisons=micro_comparisons)
    diagnostics = dict(duplicate_opposite_label_pairs=duplicates,
                       blocked_by_position={geom: {h: values.tolist() for h, values in rows.items()}
                                            for geom, rows in block_cancels.items()})
    return effects, ledger, diagnostics


def main(audit):
    thread_keys = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
    audit.check(all(os.environ.get(key) == '1' for key in thread_keys), 'single-thread independent audit')
    n = provenance(audit)
    weights, training_sha, training_checks = training(audit)
    test_start, train_lock = read(ROOT/'TEST_START.json'), read(ROOT/'train'/'TRAIN_LOCK.json')
    audit.check(test_start['status'] == 'READY' and test_start['test_generated'] is False
                and test_start['train_lock_sha256'] == training_sha and test_start['blocks'] == BLOCKS
                and datetime.fromisoformat(train_lock['utc']) <= datetime.fromisoformat(test_start['utc']),
                'held-out boundary follows the complete new training lock')
    effects, ledgers, diagnostics = evaluate(audit, n, weights)
    inference = primary_inference(audit, effects, BLOCKS)
    audit.stage = 'resource ledger'
    science = read(ROOT/'SCIENCE_SUMMARY.json')
    steps = EVALUATIONS*n*T
    expected = dict(blocks=BLOCKS, examples_per_block=n, evaluations=EVALUATIONS, fits=BLOCKS, updates=BLOCKS*UPDATES,
                    forward_sample_steps=steps, dense_forward_macs=156*steps, matcher_macs=8*steps, train_lock_sha256=training_sha, **ledgers)
    audit.check(ledgers['input_array_bytes'] == len(GEOMETRIES)*len(HISTORIES)*BLOCKS*n*T*6*8, 'independent input-byte ledger')
    audit.check(science['status'] == 'COMPLETE' and science['cap_seconds'] == SCIENCE_CAP
                and 0 < science['wall_seconds'] <= SCIENCE_CAP and science['cpu_seconds'] > 0., 'science status and time')
    audit.check(0 < science['peak_working_set_bytes'] <= MEMORY_CAP, 'science peak memory cap')
    audit.check(0 < science['artifact_bytes_at_science_end'] <= ARTIFACT_CAP, 'science artifact cap')
    audit.check(sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) < ARTIFACT_CAP, 'current artifact cap')
    completion = read(ROOT/'RUN_STATE.json')
    audit.check(completion['status'] == 'complete' and completion['evaluations'] == EVALUATIONS
                and completion['train_lock_sha256'] == training_sha, 'science completion with newly trained locked weights')
    audit.check(read(ROOT/'ENVIRONMENT.json')['threads'] == {key: '1' for key in thread_keys}, 'single-thread science environment')
    for key, value in expected.items():
        if key.endswith('_seconds'):
            audit.close(science[key], value, f'science timing {key}', 1e-10)
        else:
            audit.check(science[key] == value, f'science ledger {key}')
    audit.check(0 < science['replay_wall_seconds']+science['guard_wall_seconds'] <= science['evaluation_wall_seconds']
                <= science['wall_seconds'], 'nested comparison, replay and evaluation time scopes')
    audit.guard()
    result = dict(status='PASS', checks=audit.checks+training_checks['training_checks'], **expected, **diagnostics, **training_checks,
                  max_numeric_error=audit.max_error, cap_seconds=300, peak_working_set_bytes=peak_memory(), **audit.elapsed(),
                  primary_effects=inference, adoption_supported=all(effect['passes'] for effect in inference),
                  scope='New deterministic initialization, all archived Adam steps and count objectives, first/last/final '
                        'native count verification, held-out geometry and visible evicted-key logs, strict veto and native '
                        'output parity, paired costs and inference. Scientific gradients were not independently recomputed.')
    with (ROOT/'AUDIT.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    assert not any((ROOT/name).exists() for name in ('AUDIT.json', 'AUDIT_FAILURE.json')), 'audit already attempted'
    auditor = Audit()
    try:
        main(auditor)
    except Exception as exc:
        failure = dict(status='FAIL', stage=auditor.stage, checks=auditor.checks, reason=str(exc),
                       exception=type(exc).__name__, **auditor.elapsed())
        with (ROOT/'AUDIT_FAILURE.json').open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(failure, indent=2, allow_nan=False)+'\n')
        raise


