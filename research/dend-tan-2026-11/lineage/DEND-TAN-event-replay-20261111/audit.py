"""Independent audit of frozen visible-event replay."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import independent_event as event
import independent_warm as native
import independent_capacity_data as data


ROOT = Path(__file__).resolve().parent
ARM = 'LEARN'
T, Q, CUE, BLOCKS = 36, 28, 25, 12
SCIENCE_CAP = 180
HISTORIES, RULES = event.HISTORIES, event.RULES
CONDITIONS = tuple(f'{history}_{rule}' for history in HISTORIES for rule in RULES)
EVALUATIONS = len(CONDITIONS)*BLOCKS
SOURCE_FILES = {'capacity_data.py', 'capacity_metrics.py', 'capacity_model.py', 'independent_capacity_data.py',
                'independent_capacity.py', 'learned_model.py', 'population.py', 'reference.py', 'original_tan.py',
                'reset_model.py', 'independent_reset.py', 'warm_model.py', 'independent_warm.py'}
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
    rng = np.random.default_rng(4083900001)
    draws = rng.integers(0, blocks, (50_000, blocks))
    result = []
    thresholds = {'recent_gain': 0., 'evicted_change': -.02}
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


def provenance(audit):
    frozen = read(ROOT/'FREEZE.json')
    examples = frozen['examples_per_block']
    audit.check(frozen['scientific_data_generated'] is False and frozen['blocks'] == BLOCKS
                and frozen['evaluations'] == EVALUATIONS and examples in (512, 256), 'frozen scope')
    audit.check({'PROTOCOL.md', 'REPRODUCE.md', 'SOURCE_LOCK.json', 'PRECHECK.json', 'TIMING.json',
                 'run.py', 'event_model.py', 'audit.py', 'independent_warm.py', 'independent_capacity_data.py',
                 'independent_event.py', 'summarize.py', 'report.py'}
                <= set(frozen['files']), 'frozen scripts and prerequisites')
    audit.check(all(not name.startswith('evaluate/') for name in frozen['files']), 'freeze excludes scientific outputs')
    for name, expected in frozen['files'].items():
        audit.check(digest(ROOT/name) == expected, f'frozen file {name}')
    precheck, timing = read(ROOT/'PRECHECK.json'), read(ROOT/'TIMING.json')
    audit.check(precheck['status'] == timing['status'] == 'PASS', 'fixture and timing passed')
    audit.check(precheck.get('scientific_data_generated', False) is False
                and timing['scientific_data_generated'] is False, 'fixtures precede scientific data')
    audit.check(timing['blocks'] == BLOCKS and timing['examples_per_block'] == examples, 'timing selection')
    candidates = timing['candidates']
    audit.exact([row['examples'] for row in candidates], [512] if examples == 512 else [512, 256],
                'timing tried larger size first')
    for i, row in enumerate(candidates):
        for kind in ('science', 'audit'):
            seconds = row[f'{kind}_seconds']
            audit.check(seconds > 0, f'positive {kind} fixture duration')
            audit.close(row[f'{kind}_forecast_seconds'], 36*seconds+30, f'{kind} forecast')
        passed = row['science_forecast_seconds'] < SCIENCE_CAP and row['audit_forecast_seconds'] < 300
        audit.check(passed == (i == len(candidates)-1), 'selected first size satisfying both caps')

    lock = read(ROOT/'SOURCE_LOCK.json')
    audit.check(lock['previous_directory'] == '../DEND-TAN-donor-content-20261111', 'exact prior source directory')
    previous = ROOT/lock['previous_directory']
    audit.check(digest(previous/'MANIFEST.json') == lock['previous_manifest_sha256'], 'prior manifest seal')
    manifest = read(previous/'MANIFEST.json')['files']
    for name in ('AUDIT.json', 'SCIENCE_SUMMARY.json', 'NEXT_PLAN.md'):
        audit.check(digest(previous/name) == manifest[name]['sha256'], f'sealed prior {name}')
    earlier_audit = read(previous/'AUDIT.json')
    earlier_science = read(previous/'SCIENCE_SUMMARY.json')
    audit.check(earlier_audit['status'] == 'PASS' and earlier_science['status'] == 'COMPLETE'
                and earlier_science['blocks'] == BLOCKS and earlier_science['evaluations'] == 84
                and earlier_science['fits'] == 0, 'source is the complete audited 12-block donor-content study')
    expected_ids = {f's{s}_{ARM}' for s in range(BLOCKS)}
    weight_names = {f'weights/{ident}.npy' for ident in expected_ids}
    audit.check(set(lock['files']) == SOURCE_FILES | weight_names, 'complete copied sources and weights')
    for name, sha in lock['files'].items():
        audit.check(frozen['files'][name] == sha == manifest[name]['sha256'] == digest(previous/name),
                    f'byte-identical sealed source: {name}')
    weights = {}
    for ident in sorted(expected_ids):
        array = np.load(ROOT/f'weights/{ident}.npy', allow_pickle=False)
        audit.check(array.shape == (180,) and array.dtype == np.float64 and np.isfinite(array).all(),
                    f'{ident}: unchanged finite native weights')
        weights[ident] = array
    audit.guard()
    return examples, weights


def task_scores(raw, base, slots):
    counts, y, selected = raw['counts'], base['y'], base['selected']
    covered = counts[:, 0] != counts[:, 1]
    correct = covered & (np.argmax(counts, axis=1) == y)
    target_slot = 3-selected
    available = target_slot < slots
    cue_scores = raw['cue_scores'][:, 0]
    maxima = cue_scores.max(axis=1, keepdims=True)
    unique = (cue_scores == maxima).sum(axis=1) == 1
    routes = available & unique & (cue_scores.argmax(axis=1) == target_slot)
    target_weight = np.zeros(len(y))
    target_weight[available] = raw['cue_weights'][available, 0, target_slot[available]]

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
                'logged_nonwrite_count', 'first_read_tick', 'replay_steps', 'replay_resources'}
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


def verify_archive(audit, raw, x, rule, eviction, reference, context):
    n = len(x)
    observed, marker = x[..., 2:4], x[..., 5:6]
    gate, expected_onset = expected_gates(x)
    onset, reset, confidence = (raw[key] for key in ('read_onset', 'reset_steps', 'match_confidence'))
    audit.exact(onset, expected_onset, f'{context}: rising observed read edge')
    audit.check(np.all(onset.sum(axis=1) == 1) and onset[:, CUE].all(), f'{context}: one scientific read')
    audit.exact(raw['eviction_seen'], eviction, f'{context}: exact write-history latch')
    if rule == 'original':
        eligible = np.zeros_like(onset)
    elif rule == 'match_zero':
        eligible = onset
    else:
        eligible = onset & eviction
    audit.check(np.all(confidence[~eligible] == native.NO_CONFIDENCE), f'{context}: confidence eligibility scope')
    valid = confidence != native.NO_CONFIDENCE
    audit.check(np.all(np.abs(confidence[valid]) <= 1.+1e-13), f'{context}: bounded valid cosine')
    audit.exact(reset, eligible & (confidence >= native.COSINE_THRESHOLD), f'{context}: exact applied trigger')

    before, after = raw['cue_native_before'], raw['cue_native_after']
    audit.check(before.shape == after.shape == (n, 3, 26), f'{context}: complete cue native states')
    if reference is None:
        reference = np.zeros((n, 26))
    reference = np.broadcast_to(reference, (n, 26))
    audit.exact(after, np.where(reset[:, CUE:Q, None], reference[:, None], before), f'{context}: exact replacement scope')
    audit.exact(before[:8, :, :20], raw['states'][:, CUE-1:Q-1], f'{context}: pre-replacement native trajectory')
    audit.exact(before[:8, :, 20:], raw['spikes'][:, CUE-1:Q-1, :6], f'{context}: pre-replacement previous pulse')

    keys, values = raw['key_bank'], raw['value_bank']
    audit.check(keys.shape == (8, T, 2, 2) and values.shape == (8, T, 12, 2), f'{context}: complete FIFO archive')
    audit.exact(raw['write_gate'], marker[:8], f'{context}: visible write marker')
    audit.close(raw['read_gate'], gate[:8], f'{context}: visible read gate', 1e-12)
    previous_keys = np.concatenate((np.zeros_like(keys[:, :1]), keys[:, :-1]), axis=1)
    previous_values = np.concatenate((np.zeros_like(values[:, :1]), values[:, :-1]), axis=1)
    key_shift = np.stack((observed[:8], previous_keys[:, :, 0]), axis=2)
    retained = .97*previous_values
    value_shift = np.stack((raw['drive'], retained[..., 0]), axis=-1)
    writing = marker[:8, ..., None] == 1.
    audit.exact(keys, np.where(writing, key_shift, previous_keys), f'{context}: unchanged FIFO keys')
    audit.close(values, np.where(writing, value_shift, retained), f'{context}: unchanged FIFO values', 1e-12)

    scores, weights, injected = (raw[name] for name in ('cue_scores', 'cue_weights', 'cue_read_vectors'))
    audit.check(scores.shape == weights.shape == (n, 3, 2) and injected.shape == (n, 3, 12),
                f'{context}: complete cue reads')
    exp_scores = np.exp(scores-scores.max(axis=-1, keepdims=True))
    audit.close(weights, exp_scores/exp_scores.sum(axis=-1, keepdims=True), f'{context}: recorded cue softmax', 1e-12)
    from_bank = 2*gate[:8, CUE:Q]*np.einsum('ntk,ntdk->ntd', weights[:8], previous_values[:, CUE:Q], optimize=False)
    audit.close(injected[:8], from_bank, f'{context}: archived-bank cue reads', 1e-12)
    for saved_name, trace_name in (('cue_scores', 'scores'), ('cue_weights', 'weights'), ('cue_read_vectors', 'injections')):
        audit.exact(raw[saved_name][:8], raw[trace_name][:, CUE:Q], f'{context}: cue trace {trace_name}')
    audit.exact(raw['counts'][:8], raw['spikes'][:, Q:, 6:].sum(axis=1), f'{context}: native query count archive')


EVICTED = {'three0': 0, 'three2': 1}
FIRST_EVICTION = {'four': 4, 'three0': 8, 'three2': 8, 'two': T}
REPLAY_DIAGNOSTICS = {'logged_write_values', 'logged_write_ticks', 'logged_nonwrite_values',
                      'logged_nonwrite_ticks', 'logged_nonwrite_count', 'first_read_tick',
                      'replay_steps', 'replay_resources', 'replay_reference'}


def verify_log(audit, raw, x, context):
    n = len(x)
    audit.exact(raw['first_read_tick'], np.full(n, CUE, dtype=np.int64), f'{context}: exact first read tick')
    audit.exact(raw['logged_write_ticks'], np.broadcast_to((4, 8), (n, 2)), f'{context}: last two visible writes')
    audit.exact(raw['logged_write_values'], x[:, (4, 8)], f'{context}: exact last-two write inputs')
    audit.exact(raw['logged_nonwrite_count'], np.full(n, 2, dtype=np.int64), f'{context}: two nonwrite inputs')
    audit.exact(raw['logged_nonwrite_ticks'][:, :2], np.broadcast_to((5, 7), (n, 2)), f'{context}: distractor timing')
    audit.exact(raw['logged_nonwrite_values'][:, :2], x[:, (5, 7)], f'{context}: exact distractor amplitudes')
    audit.check(np.all(raw['logged_nonwrite_ticks'][:, 2:] == -1) and not raw['logged_nonwrite_values'][:, 2:].any(),
                f'{context}: no zeros, cue or future logged')
    triggered = raw['reset_steps'][:, CUE]
    audit.exact(raw['replay_steps'], np.where(triggered, CUE, 0), f'{context}: exact extra native steps')
    audit.check(not raw['replay_reference'][~triggered].any() and not raw['replay_resources'][~triggered].any(),
                f'{context}: no extra work without trigger')
    audit.exact(raw['cue_native_after'][triggered, 0], raw['replay_reference'][triggered],
                f'{context}: exact reconstructed state installed before read update')


def verify_paths(audit, paths, history, late_boundary, context):
    original, replay, dense = (paths[rule] for rule in RULES)
    for rule in ('replay', 'dense'):
        candidate = paths[rule]
        for key in ('key_bank', 'value_bank', 'scores', 'weights', 'read_gate', 'write_gate', 'injections',
                    'cue_scores', 'cue_weights', 'cue_read_vectors', 'read_onset', 'eviction_seen'):
            if key in ('key_bank', 'write_gate', 'read_onset', 'eviction_seen'):
                audit.exact(candidate[key], original[key], f'{context}/{rule}: unchanged {key}')
            else:
                audit.close(candidate[key], original[key], f'{context}/{rule}: unchanged {key}', 5e-12)
        audit.close(candidate['cue_native_before'][:, 0], original['cue_native_before'][:, 0],
                    f'{context}/{rule}: unchanged recipient boundary', 5e-12)
    audit.check(set(replay)-set(dense) == REPLAY_DIAGNOSTICS, f'{context}: exact replay-only fields')
    audit.exact(replay['reset_steps'], dense['reset_steps'], f'{context}: shared trigger')
    applied = replay['reset_steps'][:, CUE]
    audit.close(replay['replay_reference'][applied], late_boundary[applied], f'{context}: dense boundary reconstruction', 5e-12)
    audit.exact(replay['replay_reference'][applied, 20:], late_boundary[applied, 20:], f'{context}: reconstructed pulses')
    n = len(applied)
    discrete = {'counts', 'resources', 'spikes', 'read_onset', 'reset_steps', 'eviction_seen', 'key_bank', 'write_gate'}
    for key in dense:
        if key in discrete:
            audit.exact(replay[key], dense[key], f'{context}: replay/dense {key}')
        else:
            audit.close(replay[key], dense[key], f'{context}: replay/dense {key}', 5e-12)
        if history == 'two':
            audit.exact(replay[key], original[key], f'{context}: two-write replay no-op {key}')
            audit.exact(dense[key], original[key], f'{context}: two-write dense no-op {key}')
        elif key != 'match_confidence':
            mask = ~applied if len(replay[key]) == n else ~applied[:8]
            if key in discrete:
                audit.exact(replay[key][mask], original[key][mask], f'{context}: untriggered no-op {key}')
            else:
                audit.close(replay[key][mask], original[key][mask], f'{context}: untriggered no-op {key}', 5e-12)
    for key in ('cue_native_before', 'cue_native_after'):
        audit.exact(replay[key][..., 20:], dense[key][..., 20:], f'{context}: exact replay/dense previous pulses {key}')


def cost(history, rule, raw, eligible):
    writes = {'four': 4, 'three0': 3, 'three2': 3, 'two': 2}[history]
    replay = rule == 'replay'
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
                replay_native_steps_mean=steps, replay_native_macs_mean=156*steps)


def verify_row(audit, row, raw, base, seed, history, rule):
    ident, condition = f's{seed}_{ARM}', f'{history}_{rule}'
    context = f'{ident}/{condition}'
    identity = dict(id=ident, seed=seed, arm=ARM, condition=condition, history=history, rule=rule, T=T, q=Q, slots=2)
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
    replay_resources = raw['replay_resources'].mean(axis=0) if rule == 'replay' else np.zeros(4)
    scores['replay_resources_mean'] = replay_resources
    timing_names = {'replay_wall_seconds', 'replay_batches', 'replay_scratch_peak_bytes'}
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
    steps = raw['replay_steps'] if rule == 'replay' else np.zeros(n, dtype=np.int64)
    chunk_sizes = (steps > 0).reshape(-1, 32).sum(axis=1)
    batches = int(np.count_nonzero(chunk_sizes))
    scratch = int(chunk_sizes.max(initial=0))*CUE*6*8
    wall = row['replay_wall_seconds']
    audit.check(np.isfinite(wall) and (wall > 0 if batches else wall == 0.), f'{context}: replay-only time')
    audit.check(row['replay_batches'] == batches and row['replay_scratch_peak_bytes'] == scratch,
                f'{context}: exact replay batches and prefix scratch bytes')
    return scores, opportunities, int(steps.sum()), int(np.count_nonzero(steps)), wall, batches, scratch


def evaluate(audit, n, weights):
    audit.stage = 'replay'
    rows = read(ROOT/'evaluate'/'ROWS.json')
    index = {(row['id'], row['condition']): row for row in rows}
    expected = {(f's{s}_{ARM}', condition) for s in range(BLOCKS) for condition in CONDITIONS}
    audit.check(len(rows) == len(index) == EVALUATIONS and set(index) == expected, 'complete 144 paired evaluation rows')
    files = {f'{ident}_{condition}.npz' for ident, condition in expected}
    files |= {f'test_s{s}.npz' for s in range(BLOCKS)}
    audit.check({p.name for p in (ROOT/'evaluate').glob('*.npz')} == files, 'exact evaluation archive set')
    effects = {'recent_gain': [], 'evicted_change': []}
    input_bytes = raw_bytes = resets = opportunities = replay_steps = replay_examples = batches = scratch = duplicates = 0
    wall = 0.
    ceilings = {'four': 1., 'three0': .875, 'three2': .875, 'two': .75}
    for seed in range(BLOCKS):
        ident = f's{seed}_{ARM}'
        base = regenerated_base(audit, ROOT/'evaluate'/f'test_s{seed}.npz', n, 4083020000+100*seed)
        stimuli = {}
        full, q = event.assemble(base, 'four')
        audit.check(q == Q and data.check(base, full)['ceiling'] == 1., f'{ident}: base geometry and full ceiling')
        for history in HISTORIES:
            x, q = event.assemble(base, history)
            removed = {'four': (), 'three0': (2,), 'three2': (0,), 'two': (0, 2)}[history]
            kept = [tick for tick in range(T) if tick not in removed]
            audit.exact(x[:, kept], full[:, kept], f'{ident}/{history}: all other inputs exact')
            audit.check(not x[:, removed].any() and q == Q, f'{ident}/{history}: missing writes only')
            audit.check(data.ceiling(x, base['y']) == ceilings[history], f'{ident}/{history}: exact input ceiling')
            stimuli[history] = x
            input_bytes += x.nbytes
        scores_by_history = {}
        for history in ('two', 'four', 'three0', 'three2'):
            x = stimuli[history]
            cases, scores = {}, {}
            eviction = np.broadcast_to(np.arange(T) > FIRST_EVICTION[history], (n, T))
            for rule in RULES:
                audit.guard()
                condition = f'{history}_{rule}'
                context = f'{ident}/{condition}'
                raw = load(ROOT/'evaluate'/f'{ident}_{condition}.npz')
                if rule == 'original':
                    replayed = native.replay(x, weights[ident], ARM, q, 'original')
                    reference = None
                    if history == 'two':
                        late_boundary = raw['cue_native_before'][:, 0]
                        independent_late = replayed['cue_native_before'][:, 0]
                elif rule == 'dense':
                    replayed = native.replay(x, weights[ident], ARM, q, 'eviction_late_native', independent_late)
                    reference = late_boundary
                else:
                    replayed = event.replay(x, weights[ident], q)
                    reference = raw['replay_reference']
                    verify_log(audit, raw, x, context)
                compare_raw(audit, raw, replayed, context)
                verify_archive(audit, raw, x, rule, eviction, reference, context)
                score, opts, steps, applied, duration, block_batches, block_scratch = verify_row(
                    audit, index[ident, condition], raw, base, seed, history, rule)
                if history != 'four':
                    left, right = data.pairs(base).T
                    if history in EVICTED:
                        missing = 1-EVICTED[history]
                        paired = base['selected'][left] == missing
                        left, right = left[paired], right[paired]
                        audit.check(score['missing_ba'] == .5*score['missing_coverage'], f'{context}: missing-label BA')
                    else:
                        audit.check(score['older_ba'] == .5*score['older_coverage'], f'{context}: missing-label BA')
                    audit.exact(x[left], x[right], f'{context}: identical inputs with opposite absent labels')
                    audit.exact(base['y'][left], 1-base['y'][right], f'{context}: opposite absent labels')
                    audit.exact(raw['counts'][left], raw['counts'][right], f'{context}: exact absent-label counts')
                    duplicates += len(left)
                resets += score['reset_count']
                opportunities += opts
                replay_steps += steps
                replay_examples += applied
                wall += duration
                batches += block_batches
                scratch = max(scratch, block_scratch)
                cases[rule], scores[rule] = raw, score
                raw_bytes += sum(value.nbytes for value in raw.values())
            verify_paths(audit, cases, history, late_boundary, f'{ident}/{history}')
            scores_by_history[history] = scores
        for effect, field in (('recent_gain', 'recent_ba'), ('evicted_change', 'evicted_ba')):
            delta = [scores_by_history[name]['replay'][field]-scores_by_history[name]['original'][field] for name in EVICTED]
            effects[effect].append(float(np.mean(delta)))
        print(f'Independently replayed visible-event block {seed+1}/{BLOCKS}', flush=True)
    audit.check(opportunities == 6*BLOCKS*n, 'all cosine opportunities')
    audit.check(duplicates == 3*BLOCKS*n//2, 'all opposite-label pair checks')
    audit.check(replay_steps == CUE*replay_examples, 'first-read-only extra step count')
    ledgers = dict(input_array_bytes=input_bytes, retained_raw_array_bytes=raw_bytes, reset_count=resets,
                   reset_scalar_writes=26*resets, cosine_onsets=opportunities, replay_sample_steps=replay_steps,
                   replay_native_macs=156*replay_steps, replay_examples=replay_examples, replay_wall_seconds=wall,
                   replay_batches=batches, replay_scratch_peak_bytes=scratch)
    return effects, ledgers, duplicates


def main(audit):
    thread_keys = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
    audit.check(all(os.environ.get(key) == '1' for key in thread_keys), 'single-thread independent audit')
    n, weights = provenance(audit)
    effects, ledgers, duplicates = evaluate(audit, n, weights)
    inference = primary_inference(audit, effects, BLOCKS)
    audit.stage = 'resource ledger'
    science = read(ROOT/'SCIENCE_SUMMARY.json')
    steps = EVALUATIONS*n*T
    dense_evaluations = len(HISTORIES)*BLOCKS
    expected = dict(blocks=BLOCKS, examples_per_block=n, evaluations=EVALUATIONS, fits=0, updates=0,
                    forward_sample_steps=steps, dense_forward_macs=156*steps, matcher_macs=8*steps,
                    dense_diagnostic_evaluations=dense_evaluations, dense_diagnostic_sample_steps=dense_evaluations*n*T,
                    **ledgers)
    audit.check(ledgers['input_array_bytes'] == len(HISTORIES)*BLOCKS*n*T*6*8, 'independent input-byte ledger')
    audit.check(science['status'] == 'COMPLETE' and science['cap_seconds'] == SCIENCE_CAP
                and 0 < science['wall_seconds'] <= SCIENCE_CAP and science['cpu_seconds'] > 0, 'science wall and CPU time')
    audit.check(0 < science['peak_working_set_bytes'] <= MEMORY_CAP, 'science memory cap')
    audit.check(0 < science['artifact_bytes_at_science_end'] <= ARTIFACT_CAP, 'science artifact cap')
    audit.check(sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) < ARTIFACT_CAP, 'current artifact cap')
    completion = read(ROOT/'RUN_STATE.json')
    audit.check(completion['status'] == 'complete' and completion['evaluations'] == EVALUATIONS, 'science completion state')
    audit.check(read(ROOT/'ENVIRONMENT.json')['threads'] == {key: '1' for key in thread_keys}, 'science thread scope')
    for key, value in expected.items():
        if key == 'replay_wall_seconds':
            audit.close(science[key], value, f'science ledger {key}', 1e-10)
        else:
            audit.check(science[key] == value, f'science ledger {key}')
    audit.check(0 <= science['replay_wall_seconds'] <= science['wall_seconds'], 'replay time contained in science time')
    audit.guard()
    result = dict(status='PASS', checks=audit.checks, **expected, max_numeric_error=audit.max_error,
                  cap_seconds=300, peak_working_set_bytes=peak_memory(), **audit.elapsed(),
                  duplicate_opposite_label_pairs=duplicates, primary_effects=inference,
                  adoption_supported=all(effect['passes'] for effect in inference),
                  scope='Independent visible prefixes, input and source provenance, exact event logs, reconstructed native '
                        'state, dense diagnostic equality, no-eviction parity, native count/event and resource checks, paired inference.')
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


