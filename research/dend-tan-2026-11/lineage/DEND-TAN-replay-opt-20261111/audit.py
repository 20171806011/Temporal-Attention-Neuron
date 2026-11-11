"""Independent audit of frozen zero-prefix replay implementations."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import independent_event as event
import independent_opt as opt
import independent_warm as native
import independent_capacity_data as data


ROOT = Path(__file__).resolve().parent
ARM = 'LEARN'
T, Q, CUE, BLOCKS = 36, 28, 25, 12
SCIENCE_CAP = 180
HISTORIES, MODES = opt.HISTORIES, opt.MODES
SCHEDULES = ('standard', 'early')
CONDITIONS = tuple(f'{history}_{mode}' for history in HISTORIES for mode in MODES)
EVALUATIONS = len(CONDITIONS)*BLOCKS
SOURCE_FILES = {'capacity_data.py', 'capacity_metrics.py', 'capacity_model.py', 'independent_capacity_data.py',
                'independent_capacity.py', 'learned_model.py', 'population.py', 'reference.py', 'original_tan.py',
                'reset_model.py', 'independent_reset.py', 'warm_model.py', 'independent_warm.py',
                'event_model.py', 'independent_event.py'}
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


def provenance(audit):
    frozen = read(ROOT/'FREEZE.json')
    examples = frozen['examples_per_block']
    audit.check(frozen['scientific_data_generated'] is False and frozen['blocks'] == BLOCKS
                and frozen['evaluations'] == EVALUATIONS and examples in (512, 256), 'frozen scope')
    audit.check({'PROTOCOL.md', 'REPRODUCE.md', 'SOURCE_LOCK.json', 'PRECHECK.json', 'TIMING.json',
                 'run.py', 'opt_model.py', 'audit.py', 'independent_warm.py', 'independent_capacity_data.py',
                 'independent_event.py', 'independent_opt.py', 'summarize.py', 'report.py'}
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
    audit.check(lock['previous_directory'] == '../DEND-TAN-event-replay-20261111', 'exact prior source directory')
    previous = ROOT/lock['previous_directory']
    audit.check(digest(previous/'MANIFEST.json') == lock['previous_manifest_sha256'], 'prior manifest seal')
    manifest = read(previous/'MANIFEST.json')['files']
    for name in ('AUDIT.json', 'SCIENCE_SUMMARY.json', 'NEXT_PLAN.md'):
        audit.check(digest(previous/name) == manifest[name]['sha256'], f'sealed prior {name}')
    earlier_audit = read(previous/'AUDIT.json')
    earlier_science = read(previous/'SCIENCE_SUMMARY.json')
    audit.check(earlier_audit['status'] == 'PASS' and earlier_science['status'] == 'COMPLETE'
                and earlier_science['blocks'] == BLOCKS and earlier_science['evaluations'] == 144
                and earlier_science['fits'] == 0, 'source is the complete audited 12-block event replay study')
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
                'logged_nonwrite_count', 'first_read_tick', 'replay_steps', 'replay_resources',
                'earliest_tick', 'replay_input_rows', 'replay_nonzero_steps'}
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
                      'replay_steps', 'replay_resources', 'replay_reference', 'earliest_tick', 'replay_input_rows', 'replay_nonzero_steps'}


def schedule_masks(base):
    early = base['group'] % 2 != 0
    return {'standard': ~early, 'early': early}


def verify_log(audit, raw, x, mode, masks, context):
    n = len(x)
    early = masks['early']
    audit.exact(raw['first_read_tick'], np.full(n, CUE, dtype=np.int64), f'{context}: exact first read tick')
    audit.exact(raw['logged_write_ticks'], np.broadcast_to((4, 8), (n, 2)), f'{context}: exact retained write ticks')
    audit.exact(raw['logged_write_values'], x[:, (4, 8)], f'{context}: exact retained write values')
    audit.exact(raw['logged_nonwrite_count'], np.full(n, 2, dtype=np.int64), f'{context}: two nonwrite inputs')
    first_distractor = np.where(early, 1, 5)
    audit.exact(raw['logged_nonwrite_ticks'][:, :2], np.column_stack((first_distractor, np.full(n, 7))),
                f'{context}: parity-scheduled nonwrite ticks')
    expected_nonwrite = np.stack((x[np.arange(n), first_distractor], x[:, 7]), axis=1)
    audit.exact(raw['logged_nonwrite_values'][:, :2], expected_nonwrite, f'{context}: original nonwrite amplitudes')
    audit.check(np.all(raw['logged_nonwrite_ticks'][:, 2:] == -1) and not raw['logged_nonwrite_values'][:, 2:].any(),
                f'{context}: no cue, future or zero event logged')
    triggered = raw['reset_steps'][:, CUE]
    earliest = np.where(early, 1, 4)
    steps = CUE if mode == 'full' else CUE-earliest
    rows = 4 if mode == 'sparse' else steps
    for name, expected in (('earliest_tick', np.where(triggered, earliest, -1)),
                           ('replay_steps', np.where(triggered, steps, 0)),
                           ('replay_nonzero_steps', np.where(triggered, 4, 0)),
                           ('replay_input_rows', np.where(triggered, rows, 0))):
        audit.exact(raw[name], expected, f'{context}: {name} accounts only for actual execution')
    audit.check(not raw['replay_reference'][~triggered].any() and not raw['replay_resources'][~triggered].any(),
                f'{context}: untriggered rows do no replay work')
    audit.exact(raw['cue_native_after'][triggered, 0], raw['replay_reference'][triggered],
                f'{context}: reconstructed state installed before read update')


def verify_equivalence(audit, cases, original, history, context):
    full = cases['full']
    shared = set(full)-{'replay_steps', 'replay_resources', 'replay_input_rows'}
    discrete = {'counts', 'resources', 'spikes', 'read_onset', 'reset_steps', 'eviction_seen', 'key_bank', 'write_gate'}
    discrete |= (REPLAY_DIAGNOSTICS-{'replay_reference'})
    for mode in ('cached', 'sparse'):
        raw = cases[mode]
        for key in sorted(shared):
            if history == 'two' or key in discrete:
                audit.exact(raw[key], full[key], f'{context}/{mode}: mode equivalence {key}')
            else:
                audit.close(raw[key], full[key], f'{context}/{mode}: mode equivalence {key}', 5e-12)
        for key in ('cue_native_before', 'cue_native_after', 'replay_reference'):
            audit.exact(raw[key][..., 20:], full[key][..., 20:], f'{context}/{mode}: exact previous pulses {key}')
    no_trigger = ~full['reset_steps'][:, CUE]
    n = len(no_trigger)
    for key in sorted(original):
        if key == 'match_confidence':
            continue
        mask = no_trigger if len(original[key]) == n else no_trigger[:8]
        if key in discrete:
            audit.exact(full[key][mask], original[key][mask], f'{context}: untriggered original native {key}')
        else:
            audit.close(full[key][mask], original[key][mask], f'{context}: untriggered original native {key}', 5e-12)


def work(raw, mask):
    steps = int(raw['replay_steps'][mask].sum())
    rows = int(raw['replay_input_rows'][mask].sum())
    return dict(triggers=int(raw['reset_steps'][mask, CUE].sum()), steps=steps, input_rows=rows,
                nonzero_steps=int(raw['replay_nonzero_steps'][mask].sum()), native_macs=84*steps+72*rows,
                resources=raw['replay_resources'][mask].sum(axis=0).tolist())


def zero_work():
    return dict(triggers=0, steps=0, input_rows=0, nonzero_steps=0, native_macs=0, resources=[0.]*4,
                replay_wall_seconds=0., replay_batches=0, replay_scratch_peak_bytes=0)


def combine_work(total, path):
    for key in ('triggers', 'steps', 'input_rows', 'nonzero_steps', 'native_macs', 'replay_wall_seconds', 'replay_batches'):
        total[key] += path[key]
    total['replay_scratch_peak_bytes'] = max(total['replay_scratch_peak_bytes'], path['replay_scratch_peak_bytes'])
    total['resources'] = [a+b for a, b in zip(total['resources'], path['resources'])]


def timing_shape(raw, mode):
    batches = peak = 0
    for start in range(0, len(raw['replay_steps']), 32):
        chunk = raw['replay_steps'][start:start+32]
        active = chunk[chunk > 0]
        if len(active):
            batches += 1
            if mode == 'full':
                peak = max(peak, len(active)*CUE*6*8)
            else:
                lengths, counts = np.unique(active, return_counts=True)
                peak = max(peak, int(np.max(lengths*counts))*6*8)
    return batches, peak


def verify_row(audit, row, raw, base, masks, seed, history, mode):
    ident, condition = f's{seed}_{ARM}', f'{history}_{mode}'
    context = f'{ident}/{condition}'
    identity = dict(id=ident, seed=seed, arm=ARM, condition=condition, history=history, mode=mode, T=T, q=Q, slots=2)
    scores = task_scores(raw, base, 2)
    scores.update(reset_count=int(raw['reset_steps'].sum()), onset_count=int(raw['read_onset'].sum()),
                  eviction_fraction=float(raw['eviction_seen'].any(axis=1).mean()))
    timing_names = {'replay_wall_seconds', 'replay_batches', 'replay_scratch_peak_bytes'}
    audit.check(set(row) == set(identity) | set(scores) | timing_names | {'mean_resources', 'cost_per_episode', 'replay_by_schedule'},
                f'{context}: row schema')
    for key, value in identity.items():
        audit.check(row[key] == value, f'{context}: {key}')
    for key, value in scores.items():
        audit.close(row[key], value, f'{context}: native {key}', 1e-13)
    audit.close(row['mean_resources'], raw['resources'][:, :4].mean(axis=0), f'{context}: resource means', 1e-12)
    per_schedule = {name: dict(episodes=int(mask.sum()), **work(raw, mask)) for name, mask in masks.items()}
    audit.check(row['replay_by_schedule'] == per_schedule, f'{context}: exact per-schedule operations')
    n = len(base['y'])
    total = work(raw, np.ones(n, dtype=bool))
    cost = dict(persistent_array_bytes=448, log_allocated_bytes=2136, log_live_bytes_mean=232,
                first_read_tick_bytes=8, logical_latch_bits=3, cache_bytes_per_model=0 if mode == 'full' else 7696,
                replay_native_steps_mean=total['steps']/n, replay_input_rows_mean=total['input_rows']/n,
                replay_native_macs_mean=total['native_macs']/n, replay_resources_mean=[value/n for value in total['resources']])
    audit.check(row['cost_per_episode'] == cost, f'{context}: exact structural costs')
    batches, scratch = timing_shape(raw, mode)
    wall = row['replay_wall_seconds']
    audit.check(np.isfinite(wall) and (wall > 0 if batches else wall == 0.), f'{context}: replay-only time')
    audit.check(row['replay_batches'] == batches and row['replay_scratch_peak_bytes'] == scratch,
                f'{context}: exact replay call and peak scratch count')
    return dict(**total, replay_wall_seconds=wall, replay_batches=batches, replay_scratch_peak_bytes=scratch)


def verify_microbench(audit, measurements, cases, masks, seed, x, w):
    context = f's{seed}: isolated timing'
    triggered = cases['full']['reset_steps'][:, CUE]
    chosen = [(name, int(sample)) for name, mask in masks.items() for sample in np.flatnonzero(triggered & mask)[:8]]
    audit.check([(row['schedule'], row['sample']) for row in measurements] == chosen, f'{context}: exact first triggered rows')
    shift = seed % len(MODES)
    order = list(MODES[shift:]+MODES[:shift])
    times = {mode: [] for mode in MODES}
    steps = macs = 0
    for row in measurements:
        sample = row['sample']
        audit.check(set(row) == {'seed', 'sample', 'schedule', 'order', 'seconds'} and row['seed'] == seed,
                    f'{context}: record schema and block')
        audit.check(row['order'] == order and set(row['seconds']) == set(MODES), f'{context}: paired rotated mode order')
        single = event.replay(x[sample:sample+1], w, Q)
        audit.check(single['reset_steps'][0, CUE], f'{context}: independently replayed selected trigger')
        for mode in MODES:
            batch = cases[mode]
            timed = row['seconds'][mode]
            audit.check(np.isfinite(timed) and timed > 0., f'{context}: valid {mode} duration')
            for key in ('counts', 'resources'):
                audit.exact(single[key][0], batch[key][sample], f'{context}/{mode}: single/batch {key}')
            operations = opt.operation_fields(single, w, mode)
            for key, expected in operations.items():
                audit.exact(expected[0], batch[key][sample], f'{context}/{mode}: single/batch {key}')
            executed = int(operations['replay_steps'][0])
            input_rows = int(operations['replay_input_rows'][0])
            steps += executed
            macs += 84*executed+72*input_rows
            times[mode].append(timed)
    audit.check(bool(measurements), f'{context}: at least one pair for the block timing contrast')
    return {mode: 1000*float(np.mean(values)) for mode, values in times.items()}, steps, macs


def timing_effects(audit, block_times):
    draws = np.random.default_rng(4084900001).integers(0, BLOCKS, size=(50_000, BLOCKS))
    result = []
    for scope in ('batch', 'single'):
        for mode in MODES[1:]:
            values = np.asarray([row[scope][mode]-row[scope]['full'] for row in block_times], dtype=float)
            audit.check(values.shape == (BLOCKS,) and np.isfinite(values).all(), f'{mode}/{scope}: complete timing contrasts')
            low, high = np.percentile(values[draws].sum(axis=1)/BLOCKS, [2.5, 97.5])
            result.append(dict(effect=f'{mode}_{scope}', scope=scope, mode=mode, blocks=BLOCKS, values=values.tolist(),
                               mean=float(values.mean()), sample_sd=float(values.std(ddof=1)),
                               minimum=float(values.min()), maximum=float(values.max()),
                               faster_blocks=int((values < 0.).sum()), tied_blocks=int((values == 0.).sum()),
                               slower_blocks=int((values > 0.).sum()), ci95=[float(low), float(high)],
                               direction='faster' if high < 0 else 'slower' if low > 0 else 'undetermined'))
    return result


def evaluate(audit, n, weights):
    audit.stage = 'replay'
    rows = read(ROOT/'evaluate'/'ROWS.json')
    index = {(row['id'], row['condition']): row for row in rows}
    expected = {(f's{s}_{ARM}', condition) for s in range(BLOCKS) for condition in CONDITIONS}
    audit.check(len(rows) == len(index) == EVALUATIONS and set(index) == expected, 'complete 144 paired evaluation rows')
    expected_files = {f'{ident}_{condition}.npz' for ident, condition in expected} | {f'test_s{s}.npz' for s in range(BLOCKS)}
    audit.check({p.name for p in (ROOT/'evaluate').glob('*.npz')} == expected_files, 'exact evaluation archive set')
    audit.check({p.name for p in (ROOT/'prefix_cache').glob('*.npy')} == {f'{ident}.npy' for ident in weights},
                'exact cache archive set')
    measurements = read(ROOT/'MICROBENCH.json')
    audit.check(isinstance(measurements, list), 'isolated timing archive schema')
    micro_by_seed = {seed: [] for seed in range(BLOCKS)}
    for row in measurements:
        audit.check(row['seed'] in micro_by_seed, 'isolated timing block scope')
        micro_by_seed[row['seed']].append(row)
    totals = {mode: zero_work() for mode in MODES}
    block_times = []
    input_bytes = raw_bytes = cache_bytes = micro_steps = micro_macs = 0
    for seed in range(BLOCKS):
        ident = f's{seed}_{ARM}'
        w = weights[ident]
        saved_cache = np.load(ROOT/'prefix_cache'/f'{ident}.npy', allow_pickle=False)
        expected_cache = opt.cache(w)
        audit.check(saved_cache.shape == (T+1, 26) and saved_cache.dtype == np.float64, f'{ident}: exact cache shape and dtype')
        audit.close(saved_cache, expected_cache, f'{ident}: independent 37-entry cache', 5e-12)
        audit.exact(saved_cache[:, 20:], expected_cache[:, 20:], f'{ident}: exact cached pulse states')
        audit.check(not saved_cache[0].any(), f'{ident}: cached zero start')
        cache_bytes += saved_cache.nbytes

        base = regenerated_base(audit, ROOT/'evaluate'/f'test_s{seed}.npz', n, 4084020000+100*seed)
        masks = schedule_masks(base)
        audit.check(all(int(mask.sum()) == n//2 for mask in masks.values()), f'{ident}: balanced schedules')
        per_block = {mode: zero_work() for mode in MODES}
        for history in HISTORIES:
            x, q = opt.assemble(base, history)
            standard, standard_q = event.assemble(base, history)
            audit.check(q == standard_q == Q, f'{ident}/{history}: fixed count window')
            audit.exact(x[masks['standard']], standard[masks['standard']], f'{ident}/{history}: exact even-group schedule')
            early = masks['early']
            audit.exact(x[early, 1], standard[early, 5], f'{ident}/{history}: original drive moved to tick 1')
            audit.check(not x[early, 5].any(), f'{ident}/{history}: moved drive removed from tick 5')
            untouched = [tick for tick in range(T) if tick not in (1, 5)]
            audit.exact(x[early][:, untouched], standard[early][:, untouched], f'{ident}/{history}: other events fixed')
            input_bytes += x.nbytes
            original = native.replay(x, w, ARM, q, 'original')
            baseline = event.replay(x, w, q)
            eviction = np.broadcast_to(np.arange(T) > FIRST_EVICTION[history], (n, T))
            cases = {}
            for mode in MODES:
                audit.guard()
                condition = f'{history}_{mode}'
                context = f'{ident}/{condition}'
                raw = load(ROOT/'evaluate'/f'{ident}_{condition}.npz')
                expected_raw = baseline.copy()
                expected_raw.update(opt.operation_fields(baseline, w, mode))
                compare_raw(audit, raw, expected_raw, context)
                verify_log(audit, raw, x, mode, masks, context)
                verify_archive(audit, raw, x, 'replay', eviction, raw['replay_reference'], context)
                path = verify_row(audit, index[ident, condition], raw, base, masks, seed, history, mode)
                combine_work(totals[mode], path)
                combine_work(per_block[mode], path)
                raw_bytes += sum(value.nbytes for value in raw.values())
                cases[mode] = raw
            verify_equivalence(audit, cases, original, history, f'{ident}/{history}')
            if history == 'four':
                single, steps, macs = verify_microbench(audit, micro_by_seed[seed], cases, masks, seed, x, w)
                micro_steps += steps
                micro_macs += macs
        audit.check(all(per_block[mode]['replay_batches'] > 0 for mode in MODES), f'{ident}: nonempty batch timing pairs')
        batch = {mode: 1000*item['replay_wall_seconds']/item['replay_batches'] for mode, item in per_block.items()}
        block_times.append(dict(batch=batch, single=single))
        print(f'Independently verified replay optimization block {seed+1}/{BLOCKS}', flush=True)
    ledgers = dict(input_array_bytes=input_bytes, retained_raw_array_bytes=raw_bytes, cache_bytes=cache_bytes,
                   microbench_records=len(measurements), microbench_evaluations=3*len(measurements),
                   microbench_forward_sample_steps=3*len(measurements)*T,
                   microbench_replay_steps=micro_steps, microbench_replay_native_macs=micro_macs, mode_totals=totals)
    return block_times, ledgers


def main(audit):
    thread_keys = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
    audit.check(all(os.environ.get(key) == '1' for key in thread_keys), 'single-thread independent audit')
    n, weights = provenance(audit)
    block_times, ledgers = evaluate(audit, n, weights)
    effects = timing_effects(audit, block_times)
    audit.stage = 'resource ledger'
    science = read(ROOT/'SCIENCE_SUMMARY.json')
    steps, cache_steps = EVALUATIONS*n*T, len(weights)*T
    expected = dict(blocks=BLOCKS, examples_per_block=n, evaluations=EVALUATIONS, fits=0, updates=0,
                    forward_sample_steps=steps, dense_forward_macs=156*steps, matcher_macs=8*steps,
                    cache_count=len(weights), cache_build_steps=cache_steps, cache_build_native_macs=156*cache_steps,
                    **ledgers)
    audit.check(ledgers['input_array_bytes'] == len(HISTORIES)*BLOCKS*n*T*6*8, 'independent input-byte ledger')
    audit.check(ledgers['cache_bytes'] == len(weights)*37*26*8, 'independent cache-byte ledger')
    audit.check(science['status'] == 'COMPLETE' and science['cap_seconds'] == SCIENCE_CAP
                and 0 < science['wall_seconds'] <= SCIENCE_CAP and science['cpu_seconds'] > 0, 'science wall and CPU time')
    audit.check(0 < science['cache_build_wall_seconds'] <= science['wall_seconds'], 'cache-build time within science time')
    audit.check(0 < science['peak_working_set_bytes'] <= MEMORY_CAP, 'science memory cap')
    audit.check(0 < science['artifact_bytes_at_science_end'] <= ARTIFACT_CAP, 'science artifact cap')
    audit.check(sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) < ARTIFACT_CAP, 'current artifact cap')
    completion = read(ROOT/'RUN_STATE.json')
    audit.check(completion['status'] == 'complete' and completion['evaluations'] == EVALUATIONS, 'science completion state')
    audit.check(read(ROOT/'ENVIRONMENT.json')['threads'] == {key: '1' for key in thread_keys}, 'science thread scope')
    for key, value in expected.items():
        if key == 'mode_totals':
            audit.check(set(science[key]) == set(value), 'all mode resource totals')
            for mode, totals in value.items():
                audit.check(set(science[key][mode]) == set(totals), f'{mode}: complete resource totals')
                for field, expected_value in totals.items():
                    if field == 'replay_wall_seconds':
                        audit.close(science[key][mode][field], expected_value, f'{mode}: replay duration', 1e-10)
                    else:
                        audit.check(science[key][mode][field] == expected_value, f'{mode}: resource total {field}')
        else:
            audit.check(science[key] == value, f'science ledger {key}')
    audit.guard()
    result = dict(status='PASS', checks=audit.checks, **expected, max_numeric_error=audit.max_error,
                  cap_seconds=300, peak_working_set_bytes=peak_memory(), **audit.elapsed(),
                  equivalence_supported=True, latency_effects=effects,
                  scope='Independent visible input reconstruction, cached states, mode equivalence, actually executed '
                        'operations, isolated timing scope and descriptive paired block timing effects.')
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


