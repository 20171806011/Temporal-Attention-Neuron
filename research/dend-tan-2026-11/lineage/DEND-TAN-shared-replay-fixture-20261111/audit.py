"""Independently verify the small shared executor's archived native outputs."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import independent_capacity_data as data
import independent_event as event
import independent_guard as guard
import independent_warm as native


ROOT = Path(__file__).resolve().parent
N, T, Q = 256, 36, 28
SEED, INITIAL_SEED = 4099903201, 4099903202
CAP, MEMORY_CAP, ARTIFACT_CAP = 60, 768*1024**2, 256*1024**2
GEOMETRIES, HISTORIES, RULES = guard.GEOMETRIES, guard.HISTORIES, guard.RULES
ORACLE = np.array([[0., -1.], [1., 0.]])
DISCRETE = {'counts', 'resources', 'spikes', 'read_onset', 'reset_steps', 'eviction_seen', 'key_bank', 'write_gate',
            'logged_write_values', 'logged_write_ticks', 'logged_nonwrite_values', 'logged_nonwrite_ticks',
            'logged_nonwrite_count', 'first_read_tick', 'replay_steps', 'replay_resources',
            'evicted_keys', 'evicted_comparisons'}


class ProcessCounters(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
        (field, ctypes.c_size_t) for field in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                                             'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
                                             'PagefileUsage', 'PeakPagefileUsage')]


def peak_memory():
    counters = ProcessCounters()
    counters.cb = ctypes.sizeof(counters)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(counters), counters.cb):
        raise OSError('Could not read peak audit working set')
    return counters.PeakWorkingSetSize


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Audit:
    def __init__(self):
        self.started = time.perf_counter()
        self.cpu_started = time.process_time()
        self.checks = 0
        self.stage = 'provenance'
        self.max_error = 0.

    def guard(self):
        if time.perf_counter()-self.started > CAP:
            raise TimeoutError(f'{CAP} second independent audit cap during {self.stage}')
        if peak_memory() > MEMORY_CAP:
            raise MemoryError('768 MiB audit peak working set cap')

    def check(self, condition, context):
        self.checks += 1
        if time.perf_counter()-self.started > CAP:
            raise TimeoutError(f'{CAP} second independent audit cap during {self.stage}')
        assert bool(condition), context

    def exact(self, actual, expected, context):
        self.check(np.array_equal(actual, expected), context)

    def close(self, actual, expected, context, tolerance=5e-12):
        a, b = np.asarray(actual), np.asarray(expected)
        self.check(a.shape == b.shape, f'{context}: shape')
        self.check(np.isfinite(a).all() and np.isfinite(b).all(), f'{context}: finite')
        delta = float(np.max(np.abs(a-b), initial=0.))
        self.max_error = max(self.max_error, delta)
        self.check(delta <= tolerance, f'{context}: {delta:.17g} exceeds {tolerance}')

    def elapsed(self):
        return dict(wall_seconds=time.perf_counter()-self.started, cpu_seconds=time.process_time()-self.cpu_started)


def initialize(matrix):
    rng = np.random.default_rng(INITIAL_SEED)
    wi = rng.normal(0., .45, (6, 12))
    wr = rng.normal(0., .25, (6, 12))
    wo = rng.normal(0., .5, (6, 2))
    return np.r_[wi.ravel(), wr.ravel(), np.full(12, .15), np.full(6, .05), wo.ravel(), np.full(2, .1), matrix.ravel()]


def independent_paths(x, w):
    """Packet every eight examples so the sealed trace window spans every row."""
    chunks = {rule: [] for rule in RULES}
    for start in range(0, len(x), 8):
        packet = x[start:start+8]
        original = native.replay(packet, w, 'LEARN', Q, 'original')
        replayed = event.replay(packet, w, Q)
        chunks['original'].append(original)
        chunks['replay'].append(replayed)
        chunks['guarded'].append(guard.apply_guard(packet, w, replayed, original))
    return {rule: {key: np.concatenate([chunk[key] for chunk in rows], axis=0) for key in rows[0]}
            for rule, rows in chunks.items()}


def compare_raw(audit, actual, expected, context):
    audit.check(set(actual) == set(expected), f'{context}: exact archive fields')
    for key, value in expected.items():
        audit.check(actual[key].shape == value.shape and actual[key].dtype == value.dtype, f'{context}/{key}: shape and dtype')
        if key in DISCRETE:
            audit.exact(actual[key], value, f'{context}: exact {key}')
        else:
            audit.close(actual[key], value, f'{context}: {key}')
    for key in ('cue_native_before', 'cue_native_after', 'replay_reference'):
        if key in actual:
            audit.exact(actual[key][..., 20:], expected[key][..., 20:], f'{context}: exact previous pulses {key}')


def factorial():
    base = data.make_base(N, SEED)
    angles = base['angle'][:, None]+np.deg2rad((0., 40., 180., 230.))
    keys = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    queries = np.stack((-keys[..., 1], keys[..., 0]), axis=-1)+base['query_noise']
    paired = {'orthogonal': base, 'near': dict(base, key_vectors=keys, query_vectors=queries)}
    inputs = {f'{name}_{history}': guard.assemble(geometric, history)[0]
              for name, geometric in paired.items() for history in HISTORIES}
    return base, inputs


def edge_inputs():
    x = np.zeros((12, T, 6))
    keys = {'E': (1., 0.), 'N': (0., 1.), 'Z': (0., 0.), 'W': (-1., 0.), 'D': (1., 1.)}
    sequences = ('ENN', 'WNEE', 'EZNN', 'EEE', 'ENWEN', '', 'DNN', 'ENN', 'ENWE', 'EN', 'WNEE', 'WNEE')
    for row, sequence in enumerate(sequences):
        for tick, key in zip((0, 2, 4, 8, 9), sequence):
            x[row, tick, 0] = 1.
            x[row, tick, 2:4] = keys[key]
            x[row, tick, 5] = 1.
    cues = ({6: 'D'}, {9: 'E', 10: 'E', 12: 'E'}, {9: 'N'}, {6: 'W', 10: 'E'}, {10: 'N'},
            {7: 'E'}, {6: 'D'}, {1: 'E', 10: 'N'}, {}, {6: 'N'}, {9: 'E', 10: 'E', 12: 'E'},
            {9: 'E', 10: 'E', 12: 'E'})
    for row, reads in enumerate(cues):
        for tick, key in reads.items():
            x[row, tick, 2:4] = keys[key]
    x[8, 1, 4] = .31
    x[10:, 1, 4], x[10:, 5, 4] = .33, .27
    for row, payload, key in ((1, (1., 0.), 'N'), (10, (1.2, 0.), 'W'), (11, (0., .9), 'N')):
        x[row, 11] = (*payload, *keys[key], 0., 1.)
    return x


def verify_edges(audit, x, paths, declared):
    first = [6, 9, 9, 6, 10, 7, 6, 1, -1, 6, 9, 9]
    replay_tick = [6, 9, -1, -1, 10, -1, 6, -1, -1, -1, 9, 9]
    guarded_tick = [-1, 9, -1, -1, -1, -1, -1, -1, -1, -1, 9, 9]
    names = ['threshold_tie', 'accept_repeat_and_write', 'zero_key', 'first_read_mismatch', 'eviction_fifo',
             'empty_keys', 'stronger_evicted_key', 'read_before_eviction', 'no_read', 'two_writes',
             'future_write_west', 'future_write_north']
    audit.check(declared['names'] == names and len(declared['descriptions']) == len(names), 'complete named edge coverage')
    audit.exact(declared['first_read_tick'], first, 'declared edge first-read expectations')
    for rule, expected in (('replay', replay_tick), ('guarded', guarded_tick)):
        key = 'old_trigger_tick' if rule == 'replay' else 'guarded_trigger_tick'
        audit.exact(declared[key], expected, f'declared {rule} trigger expectations')
        raw = paths[rule]
        audit.exact(raw['first_read_tick'], first, f'{rule}: independently visible first edges')
        ticks = np.asarray(expected)
        audit.exact(raw['reset_steps'], np.arange(T) == ticks[:, None], f'{rule}: exact edge replacement times')
        audit.exact(raw['replay_steps'], np.maximum(ticks, 0), f'{rule}: exact edge replay work')
    replay, guarded = paths['replay'], paths['guarded']
    audit.check(replay['match_confidence'][0, 6] == native.COSINE_THRESHOLD == guarded['evicted_cosine'][0, 6],
                'threshold equality triggers old replay and strict lost-key equality vetoes')
    audit.exact(guarded['evicted_keys'][4], ((0., 1.), (-1., 0.)), 'third eviction overwrites oldest logged key')
    audit.check(guarded['evicted_cosine'][6, 6] > replay['match_confidence'][6, 6], 'stronger evicted key vetoes')
    audit.check(not replay['eviction_seen'][2].any() and np.count_nonzero(guarded['evicted_keys'][2].any(axis=1)) == 1,
                'zero key cannot make two valid slots')
    audit.check(not replay['read_onset'][8].any() and not replay['reset_steps'][8].any(), 'no read, no replacement')
    pair = declared['future_pair']
    audit.check(pair == dict(rows=[10, 11], shared_through_tick=10, changed_write_tick=11), 'frozen future-write edge pair')
    audit.exact(x[10, :11], x[11, :11], 'future pair identical through first read')
    audit.check(not np.array_equal(x[10, 11], x[11, 11]), 'future pair later writes differ')
    for rule, raw in paths.items():
        for key in native.TRACES:
            audit.exact(raw[key][10, :11], raw[key][11, :11], f'{rule}: no future dependence {key}')
        audit.check(not np.array_equal(raw['value_bank'][10, 11], raw['value_bank'][11, 11]),
                    f'{rule}: later write changes its own value branch')
    audit.exact(replay['replay_reference'][10], replay['replay_reference'][11], 'future write excluded from replay reference')


def provenance(audit):
    frozen = read(ROOT/'FREEZE.json')
    fixed = dict(study='FIXTURE_ONLY', fixture_data_generated=False, factorial_samples=N, edge_samples=12,
                 parity_packet=8, timing_packet=32, timing_repeats=3, cap_seconds=90, audit_cap_seconds=CAP,
                 memory_cap_bytes=MEMORY_CAP, artifact_cap_bytes=ARTIFACT_CAP)
    audit.check(all(frozen[key] == value for key, value in fixed.items()), 'fixture-only scope and caps frozen')
    names = {'PROTOCOL.md', 'REPRODUCE.md', 'SOURCE_LOCK.json', 'fixture_data.py', 'shared_model.py', 'run.py', 'audit.py'}
    audit.check(names <= set(frozen['files']), 'frozen fixture, executor and independent audit sources')
    audit.check(all('/' not in name for name in frozen['files']), 'frozen inputs exclude generated fixture and evaluations')
    for name, expected in frozen['files'].items():
        audit.check(digest(ROOT/name) == expected, f'frozen {name}')
    lock = read(ROOT/'SOURCE_LOCK.json')
    audit.check(lock['previous_directory'] == '../DEND-TAN-evicted-key-20261111'
                and lock['blocked_directory'] == '../DEND-TAN-evicted-key-confirm-20261111', 'exact predecessor directories')
    previous, blocked = ROOT/lock['previous_directory'], ROOT/lock['blocked_directory']
    audit.check(digest(previous/'MANIFEST.json') == lock['previous_manifest_sha256'], 'previous guard manifest')
    audit.check(digest(blocked/'PREFLIGHT_MANIFEST.json') == lock['blocked_manifest_sha256'], 'stopped confirmation preflight manifest')
    old, stop = read(previous/'MANIFEST.json')['files'], read(blocked/'PREFLIGHT_MANIFEST.json')
    audit.check(stop['status'] == 'SEALED_PREFLIGHT_ONLY', 'preflight-only stopping boundary preserved')
    for name in ('TIMING.json', 'PREFLIGHT_VERIFICATION.json', 'RESOURCE_BLOCKER.md'):
        audit.check(digest(blocked/name) == stop['files'][name]['sha256'], f'sealed stopped confirmation {name}')
    blocked_status = read(blocked/'PREFLIGHT_VERIFICATION.json')
    audit.check(blocked_status['status'] == 'VERIFIED_INFEASIBLE'
                and all(not (blocked/name).exists() for name in blocked_status['absent_formal_artifacts']),
                'confirmation remains stopped before formal data')
    sources = {'capacity_data.py', 'capacity_metrics.py', 'capacity_model.py', 'independent_capacity_data.py',
               'independent_capacity.py', 'learned_model.py', 'population.py', 'reference.py', 'original_tan.py',
               'reset_model.py', 'independent_reset.py', 'warm_model.py', 'independent_warm.py', 'event_model.py',
               'independent_event.py', 'confirm_data.py', 'guard_model.py', 'independent_guard.py'}
    audit.check(set(lock['files']) == sources, '18 sealed sources without any trained weights')
    for name in sources:
        expected = lock['files'][name]
        audit.check(expected == frozen['files'][name] == old[name]['sha256'] == stop['files'][name]['sha256']
                    == digest(previous/name) == digest(blocked/name), f'{name}: same sealed source across predecessors')
    audit.guard()


def load_fixture(audit):
    folder = ROOT/'fixture'
    base, inputs = factorial()
    weights = {'factorial': initialize(ORACLE), 'edge': initialize(np.eye(2))}
    expected = {'base.npz': base, 'inputs.npz': inputs, 'weights.npz': weights, 'edge.npz': {'x': edge_inputs()}}
    audit.check({path.name for path in folder.iterdir() if path.is_file()} == set(expected) | {'edge_expectations.json'},
                'exact fixture archive list')
    for name, arrays in expected.items():
        archive = load(folder/name)
        audit.check(set(archive) == set(arrays), f'{name}: fixture field list')
        for key, value in arrays.items():
            audit.check(archive[key].dtype == value.dtype, f'{name}/{key}: fixture dtype')
            audit.exact(archive[key], value, f'{name}/{key}: independently rebuilt fixture')
    audit.check(not np.array_equal(weights['factorial'][176:], weights['edge'][176:]), 'fixed separate matcher matrices')
    audit.exact(weights['factorial'][:176], weights['edge'][:176], 'same seeded native parameters for both fixtures')
    return base, inputs | {'edge': expected['edge.npz']['x']}, weights, read(folder/'edge_expectations.json')


def expected_work(paths):
    n = len(paths['original']['counts'])
    old, guarded = (paths[rule]['replay_steps'] for rule in ('replay', 'guarded'))
    first = paths['replay']['first_read_tick']
    prefix = int(np.where(first >= 0, first, T).sum())
    suffix = int((T-old[old > 0]).sum())
    old_steps, guarded_steps = int(old.sum()), int(guarded.sum())
    totals = {}
    for engine, early, main, branch, replacement in (('shared', prefix, n*T, suffix, old_steps),
                                                      ('separate', 3*prefix, 3*n*T, 0, old_steps+guarded_steps)):
        online = main+branch
        totals[engine] = dict(prefix_sample_steps=early, main_sample_steps=main, branch_sample_steps=branch,
                              online_sample_steps=online, replay_sample_steps=replacement, native_macs=156*(online+replacement))
    return totals


def sum_work(totals):
    items = list(totals)
    return {engine: {name: sum(item[engine][name] for item in items) for name in fields}
            for engine, fields in items[0].items()}


def evaluate(audit):
    audit.stage = 'archived native parity'
    base, inputs, weights, edge_expectations = load_fixture(audit)
    evaluate_dir = ROOT/'evaluate'
    expected_files = {f'{group}_{rule}_{engine}.npz' for group in inputs for rule in RULES for engine in ('shared', 'separate')}
    audit.check({path.name for path in evaluate_dir.iterdir() if path.is_file()} == expected_files,
                'exact archived paths and engines')
    work, single, archive_bytes = {}, [], 0
    schedule = base['group'] % len(guard.SCHEDULES)
    for group, x in inputs.items():
        audit.guard()
        w = weights['edge' if group == 'edge' else 'factorial']
        independent = independent_paths(x, w)
        pair_max = 0.
        path_checks = 0
        for rule in RULES:
            shared = load(evaluate_dir/f'{group}_{rule}_shared.npz')
            separate = load(evaluate_dir/f'{group}_{rule}_separate.npz')
            compare_raw(audit, shared, independent[rule], f'{group}/{rule}/shared')
            compare_raw(audit, separate, independent[rule], f'{group}/{rule}/separate')
            compare_raw(audit, shared, separate, f'{group}/{rule}/pair')
            for key, value in shared.items():
                if key not in DISCRETE and value.dtype.kind not in 'biu':
                    pair_max = max(pair_max, float(np.max(np.abs(value-separate[key]), initial=0.)))
            path_checks += len(shared)+len(set(shared) & {'cue_native_before', 'cue_native_after', 'replay_reference'})
            archive_bytes += sum(value.nbytes for actual in (shared, separate) for value in actual.values())
        original, replay, guarded = (independent[rule] for rule in RULES)
        old, new = replay['reset_steps'], guarded['reset_steps']
        audit.check(not (new & ~old).any(), f'{group}: guard cannot add triggers')
        vetoed = (old & ~new).any(axis=1)
        for key in ('counts', 'resources', 'spikes'):
            audit.exact(guarded[key][vetoed], original[key][vetoed], f'{group}: veto selects original {key}')
            audit.exact(guarded[key][~vetoed], replay[key][~vetoed], f'{group}: equal trigger selects same {key}')
        work[group] = dict(episodes=len(x), checks=((len(x)+7)//8)*path_checks, max_numeric_error=pair_max,
                           old_triggers=int(old.sum()), guarded_triggers=int(new.sum()), vetoes=int(vetoed.sum()),
                           work=expected_work(independent))
        if group == 'edge':
            verify_edges(audit, x, independent, edge_expectations)
        elif group.endswith('_four'):
            for idx, name in enumerate(guard.SCHEDULES):
                candidates = np.flatnonzero((schedule == idx) & (replay['replay_steps'] > 0))
                if not len(candidates):
                    continue
                sample = int(candidates[0])
                one = independent_paths(x[sample:sample+1], w)
                for rule in ('replay', 'guarded'):
                    for key in ('counts', 'resources', 'replay_steps', 'replay_resources'):
                        audit.exact(one[rule][key][0], independent[rule][key][sample], f'{group}/{name}: exact single/batch {rule}/{key}')
                single.append(dict(group=group, schedule=name, sample=sample,
                                   replay_steps={rule: int(one[rule]['replay_steps'][0]) for rule in ('replay', 'guarded')}))
        print(f'Independently checked all native traces for {group}', flush=True)
    return work, single, archive_bytes


def verify_timing(audit, groups, selected):
    timing = read(ROOT/'TIMING.json')
    audit.check(set(timing) == {'batch_size', 'repeats', 'single'} and timing['batch_size'] == 32,
                'frozen timing archive and packet size')
    repeats = timing['repeats']
    expected_orders = (['shared', 'separate'], ['separate', 'shared'], ['shared', 'separate'])
    batch_work = sum_work(row['work'] for name, row in groups.items() if name != 'edge')
    audit.check(len(repeats) == len(expected_orders), 'all three timed repeats retained')
    total_seconds = 0.
    for index, (row, order) in enumerate(zip(repeats, expected_orders), 1):
        audit.check(set(row) == {'trial', 'order', 'seconds', 'work'} and row['trial'] == index and row['order'] == order,
                    f'timing repeat {index}: fixed paired order')
        audit.check(row['work'] == batch_work, f'timing repeat {index}: exact full factorial work')
        audit.check(set(row['seconds']) == set(order) and all(np.isfinite(value) and value > 0. for value in row['seconds'].values()),
                    f'timing repeat {index}: paired positive measurements')
        total_seconds += sum(row['seconds'].values())
    singles = timing['single']
    audit.check(len(singles) == len(selected), 'all fixed isolated selections retained')
    fields = {'group', 'schedule', 'sample', 'order', 'full_seconds', 'replay_seconds', 'guard_seconds', 'replay_steps'}
    for index, (record, expected) in enumerate(zip(singles, selected)):
        audit.check(set(record) == fields and all(record[key] == value for key, value in expected.items()),
                    f'isolated record {index}: first old trigger in geometry/schedule and exact steps')
        order = ['replay', 'guarded'] if index % 2 == 0 else ['guarded', 'replay']
        audit.check(record['order'] == order, f'isolated record {index}: alternating paired order')
        for key in ('full_seconds', 'replay_seconds', 'guard_seconds'):
            audit.check(set(record[key]) == set(order) and all(np.isfinite(value) and value >= 0. for value in record[key].values()),
                        f'isolated record {index}: valid {key}')
        for rule in order:
            total = record['full_seconds'][rule]
            replay = record['replay_seconds'][rule]
            guard_seconds = record['guard_seconds'][rule]
            audit.check(0. < total and replay+guard_seconds <= total, f'isolated record {index}: nested timed operations')
            audit.check((replay > 0.) == (record['replay_steps'][rule] > 0) and (guard_seconds > 0.) == (rule == 'guarded'),
                        f'isolated record {index}: positive time only for executed operations')
            total_seconds += total
    return repeats, singles, total_seconds


def verify_summary(audit, groups, repeats, singles, timing_seconds):
    summary = read(ROOT/'SUMMARY.json')
    rows = read(ROOT/'PARITY_ROWS.json')
    audit.check(summary['parity_rows'] == rows and [row['group'] for row in rows] == list(groups),
                'complete parity accounting list')
    subtotal = 0.
    for row in rows:
        group = row['group']
        expected = groups[group]
        for key, value in expected.items():
            if key == 'max_numeric_error':
                audit.close(row[key], value, f'{group}: archived pair max numeric error')
            else:
                audit.check(row[key] == value, f'{group}: exact parity accounting {key}')
        elapsed = row['engine_seconds']
        audit.check(set(elapsed) == {'shared', 'separate'}, f'{group}: both parity engine times')
        times = [*elapsed.values(), row['comparison_seconds'], row['archive_seconds']]
        audit.check(all(np.isfinite(value) and value > 0. for value in times), f'{group}: recorded positive parity timings')
        subtotal += sum(times)
    totals = sum_work(row['work'] for row in groups.values())
    fixed = dict(status='COMPLETE_FIXTURE', groups=len(groups), factorial_samples=N, edge_samples=12,
                 factorial_seed=SEED, native_init_seed=INITIAL_SEED, scientific_data_generated=False,
                 fits=0, updates=0, logical_paths=len(groups)*len(RULES),
                 checked_episodes=sum(row['episodes'] for row in groups.values()), checks=sum(row['checks'] for row in groups.values()),
                 work=totals, timing_repeats=len(repeats), isolated_pairs=len(singles), cap_seconds=90)
    for key, value in fixed.items():
        audit.check(summary[key] == value, f'fixture summary {key}')
    audit.close(summary['max_numeric_error'], max(row['max_numeric_error'] for row in groups.values()), 'pair maximum across groups')
    audit.check(np.isfinite(summary['input_seconds']) and summary['input_seconds'] > 0., 'positive fixture input construction time')
    audit.check(summary['input_seconds']+subtotal+timing_seconds <= summary['wall_seconds'] <= 90
                and summary['cpu_seconds'] > 0., 'fixture wall and CPU time within cap')
    audit.check(0 < summary['peak_working_set_bytes'] <= MEMORY_CAP, 'fixture working set within cap')
    audit.check(0 < summary['artifact_bytes'] <= ARTIFACT_CAP, 'fixture physical archives within cap')
    audit.check(sum(path.stat().st_size for path in ROOT.rglob('*') if path.is_file()) <= ARTIFACT_CAP, 'current artifact cap')
    state = read(ROOT/'RUN_STATE.json')
    audit.check(state['status'] == 'complete_fixture' and state['scientific_data_generated'] is False, 'fixture-only completion')
    audit.check(all(not (ROOT/name).exists() for name in ('train', 'TEST_START.json', 'TRAIN_SUMMARY.json', 'SCIENCE_SUMMARY.json')),
                'no confirmation stage or fitted parameters')
    return totals


def main(audit):
    thread_keys = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
    audit.check(all(os.environ.get(key) == '1' for key in thread_keys), 'single-thread independent audit')
    provenance(audit)
    groups, selected, array_bytes = evaluate(audit)
    audit.stage = 'fixture accounting and timing'
    repeats, singles, duration = verify_timing(audit, groups, selected)
    work = verify_summary(audit, groups, repeats, singles, duration)
    audit.guard()
    result = dict(status='PASS', fixture_only=True, checks=audit.checks, max_numeric_error=audit.max_error,
                  checked_episodes=sum(row['episodes'] for row in groups.values()), logical_paths=len(groups)*len(RULES),
                  work=work, retained_raw_array_bytes=array_bytes, timing_repeats=len(repeats), isolated_pairs=len(singles),
                  cap_seconds=CAP, peak_working_set_bytes=peak_memory(), **audit.elapsed(),
                  scope='Independently reconstructed factorial and boundary inputs, every archived trace, exact discrete '
                        'outputs and first-read semantics, later-write branch isolation, logical work and fixed timing scope.')
    with (ROOT/'AUDIT.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    assert not any((ROOT/name).exists() for name in ('AUDIT.json', 'AUDIT_FAILURE.json')), 'audit already attempted'
    auditor = Audit()
    try:
        main(auditor)
    except Exception as exc:
        failure = dict(status='FAIL', stage=auditor.stage, checks=auditor.checks, reason=str(exc),
                       exception=type(exc).__name__, **auditor.elapsed())
        with (ROOT/'AUDIT_FAILURE.json').open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(failure, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        raise
