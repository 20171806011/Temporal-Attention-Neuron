"""Freeze and evaluate the saved bilinear matcher checkpoints for small shared-trajectory equivalence."""

import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

import fixture_data
import guard_model
import shared_model


ROOT = Path(__file__).resolve().parent
T = 36
Q = 28
CAP = 90
MEMORY_CAP = 768*1024**2
ARTIFACT_CAP = 256*1024**2
THREAD_KEYS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
RULES = ('original', 'replay', 'guarded')
GROUPS = tuple(f'{geometry}_{history}' for geometry in ('orthogonal', 'near')
               for history in ('four', 'three0', 'three2', 'two'))
EXACT = {'counts', 'resources', 'spikes', 'key_bank', 'write_gate', 'read_onset', 'reset_steps',
         'eviction_seen', 'first_read_tick', 'replay_steps', 'replay_resources', 'evicted_keys', 'evicted_comparisons'}
WORK = ('prefix_sample_steps', 'main_sample_steps', 'branch_sample_steps', 'online_sample_steps',
        'replay_sample_steps', 'native_macs')

def utc():
    return datetime.now(timezone.utc).isoformat()


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def dump(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                             encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_lock(name):
    for relative, digest in read(name)['files'].items():
        assert sha(ROOT / relative) == digest, relative


def peak_memory():
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
             'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
             'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    counter = Counters()
    counter.cb = ctypes.sizeof(counter)
    assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(counter), counter.cb)
    return counter.PeakWorkingSetSize


def artifact_size():
    return sum(path.stat().st_size for path in ROOT.rglob('*') if path.is_file())


def guard(start, disk=False, cap=CAP):
    if time.perf_counter()-start > cap:
        raise TimeoutError(f'{cap} second stage limit')
    if peak_memory() > MEMORY_CAP:
        raise MemoryError('1 GiB peak working-set limit')
    if disk and artifact_size() > ARTIFACT_CAP:
        raise OSError('1.5 GiB artifact limit')


def freeze():
    assert not any((ROOT/name).exists() for name in ('FREEZE.json', 'RUN_STATE.json', 'fixture', 'evaluate'))
    source = read('SOURCE_LOCK.json')
    verify_lock('SOURCE_LOCK.json')
    assert len(source['files']) == 18
    for name in ('PROTOCOL.md', 'REPRODUCE.md', 'fixture_data.py', 'shared_model.py', 'audit.py'):
        assert (ROOT/name).is_file()
    files = {path.name: sha(path) for path in sorted(ROOT.iterdir())
             if path.is_file() and path.suffix in ('.py', '.md', '.json')}
    dump('FREEZE.json', dict(utc=utc(), study='FIXTURE_ONLY', fixture_data_generated=False,
         factorial_samples=256, edge_samples=12, parity_packet=8, timing_packet=32, timing_repeats=3,
         cap_seconds=CAP, audit_cap_seconds=60, memory_cap_bytes=MEMORY_CAP, artifact_cap_bytes=ARTIFACT_CAP, files=files))
    print('FROZEN fixture only: 256 factorial + 12 boundary examples', flush=True)


def checks_between(actual, expected, context):
    assert actual.keys() == expected.keys(), context
    checks, maximum = 0, 0.
    for name, array in actual.items():
        target = expected[name]
        assert array.shape == target.shape and array.dtype == target.dtype, (context, name)
        assert np.isfinite(array).all() and np.isfinite(target).all(), (context, name)
        if name in EXACT or name.startswith('logged_') or array.dtype.kind in 'biu':
            assert np.array_equal(array, target), (context, name)
        else:
            residual = float(np.max(np.abs(array-target), initial=0.))
            assert residual <= 5e-12, (context, name, residual)
            maximum = max(maximum, residual)
        checks += 1
        if name in ('cue_native_before', 'cue_native_after', 'replay_reference'):
            assert np.array_equal(array[..., 20:], target[..., 20:]), (context, name, 'previous pulse')
            checks += 1
    return checks, maximum


def separate(x, w):
    cases = {}
    for rule in RULES:
        raw, diag, _ = guard_model.evaluate(x, w, Q, rule)
        cases[rule] = raw | diag
    return cases, {}


def shared(x, w):
    return shared_model.evaluate(x, w, Q)


def work_account(cases, ledger, n, engine):
    first = cases['replay']['first_read_tick']
    prefix = int(np.where(first >= 0, first, T).sum())
    legacy = cases['replay']['replay_steps']
    replacement = int(legacy.sum())
    active = cases['replay']['reset_steps'].any(axis=1)
    assert np.array_equal(legacy, np.where(active, first, 0))
    if engine == 'shared':
        branch = int(np.where(active, T-first, 0).sum())
        main = n*T
        assert ledger['main_sample_steps'] == main
        assert ledger['branch_sample_steps'] == branch
        assert ledger['replay_sample_steps'] == replacement
    else:
        main, branch = 3*n*T, 0
        prefix *= 3
        replacement += int(cases['guarded']['replay_steps'].sum())
    online = main+branch
    return dict(prefix_sample_steps=prefix, main_sample_steps=main, branch_sample_steps=branch,
                online_sample_steps=online, replay_sample_steps=replacement, native_macs=156*(online+replacement))


def zero_work():
    return {field: 0 for field in WORK}


def add_work(total, added):
    for field in WORK:
        total[field] += added[field]


def trigger_ticks(mask):
    return np.where(mask.any(axis=1), mask.argmax(axis=1), -1)


def boundary_expectations(cases, expected):
    for name, actual in (
        ('first_read_tick', cases['replay']['first_read_tick']),
        ('old_trigger_tick', trigger_ticks(cases['replay']['reset_steps'])),
        ('guarded_trigger_tick', trigger_ticks(cases['guarded']['reset_steps'])),
    ):
        assert np.array_equal(actual, expected[name]), name
    assert np.any(cases['replay']['value_bank'] != cases['original']['value_bank']), 'post-read branch-local value coverage'
    pair = expected['future_pair']
    a, b = pair['rows']
    stop = pair['shared_through_tick']+1
    changed = pair['changed_write_tick']
    for rule, raw in cases.items():
        for field in ('states', 'spikes', 'currents', 'injections', 'key_bank', 'value_bank', 'reset_steps'):
            assert np.array_equal(raw[field][a, :stop], raw[field][b, :stop]), (rule, field, 'future isolation')
        assert not np.array_equal(raw['key_bank'][a, changed], raw['key_bank'][b, changed]), rule
    for rule in ('replay', 'guarded'):
        for field, array in cases[rule].items():
            if field.startswith('logged_') or field in ('first_read_tick', 'replay_reference', 'evicted_keys'):
                assert np.array_equal(array[a], array[b]), (rule, field, 'pre-read logs')



def parity_group(folder, name, x, w, start, expected=None):
    saved = {engine: {rule: {} for rule in RULES} for engine in ('shared', 'separate')}
    ledgers = {engine: zero_work() for engine in saved}
    checks, maximum, compare_seconds = 0, 0., 0.
    elapsed = {engine: 0. for engine in saved}
    for left in range(0, len(x), 8):
        guard(start)
        packet = x[left:left+8]
        outputs = {}
        for engine, fn in (('separate', separate), ('shared', shared)):
            tick = time.perf_counter()
            cases, ledger = fn(packet, w)
            elapsed[engine] += time.perf_counter()-tick
            add_work(ledgers[engine], work_account(cases, ledger, len(packet), engine))
            outputs[engine] = cases
        tick = time.perf_counter()
        try:
            for rule in RULES:
                got, error = checks_between(outputs['shared'][rule], outputs['separate'][rule], (name, left, rule))
                checks += got
                maximum = max(maximum, error)
        except Exception:
            dump('FAILED_PACKET.json', dict(group=name, first_sample=left))
            for engine, cases in outputs.items():
                for rule, raw in cases.items():
                    np.savez_compressed(ROOT/f'FAILED_{engine}_{rule}.npz', **raw)
            raise
        compare_seconds += time.perf_counter()-tick
        for engine, cases in outputs.items():
            for rule, raw in cases.items():
                for key, values in raw.items():
                    saved[engine][rule].setdefault(key, []).append(values)
    started_archive = time.perf_counter()
    combined = {}
    for engine, cases in saved.items():
        combined[engine] = {}
        for rule, raw in cases.items():
            joined = {key: np.concatenate(values, axis=0) for key, values in raw.items()}
            assert all(len(array) == len(x) for array in joined.values())
            combined[engine][rule] = joined
            np.savez_compressed(folder/f'{name}_{rule}_{engine}.npz', **joined)
            guard(start, disk=True)
    if expected is not None:
        for cases in combined.values():
            boundary_expectations(cases, expected)
    old, new = (combined['shared'][rule]['reset_steps'] for rule in ('replay', 'guarded'))
    assert not np.any(new & ~old)
    row = dict(group=name, episodes=len(x), checks=checks, max_numeric_error=maximum,
               old_triggers=int(old.sum()), guarded_triggers=int(new.sum()), vetoes=int((old & ~new).sum()),
               work=ledgers, engine_seconds=elapsed, comparison_seconds=compare_seconds,
               archive_seconds=time.perf_counter()-started_archive)
    print(f"Parity {name}: {checks} checks, old {row['old_triggers']}, guarded {row['guarded_triggers']}", flush=True)
    return row


def time_batches(inputs, w, start):
    records = []
    for trial, order in enumerate((('shared', 'separate'), ('separate', 'shared'), ('shared', 'separate')), 1):
        elapsed, ledgers = {}, {}
        discrete = {}
        for engine in order:
            fn = shared if engine == 'shared' else separate
            total = zero_work()
            ticks = 0.
            cells = []
            for group in GROUPS:
                x = inputs[group]
                for left in range(0, len(x), 32):
                    guard(start)
                    packet = x[left:left+32]
                    tick = time.perf_counter()
                    cases, ledger = fn(packet, w)
                    ticks += time.perf_counter()-tick
                    add_work(total, work_account(cases, ledger, len(packet), engine))
                    cells.append({rule: {key: cases[rule][key] for key in ('counts', 'spikes', 'resources', 'reset_steps')}
                                  for rule in RULES})
            elapsed[engine], ledgers[engine], discrete[engine] = ticks, total, cells
        for left, right in zip(discrete['shared'], discrete['separate']):
            for rule in RULES:
                for key in left[rule]:
                    assert np.array_equal(left[rule][key], right[rule][key]), (trial, rule, key)
        records.append(dict(trial=trial, order=list(order), seconds=elapsed, work=ledgers))
        print(f"Timed fixed repeat {trial}/3", flush=True)
    return records


def timed_single(folder, inputs, base, w, start):
    rows = []
    group_id = base['group'] % 4
    for geometry in ('orthogonal', 'near'):
        group = f'{geometry}_four'
        with np.load(folder/f'{group}_replay_separate.npz', allow_pickle=False) as raw:
            trigger = raw['reset_steps'].any(axis=1)
        references = {}
        for rule in ('replay', 'guarded'):
            with np.load(folder/f'{group}_{rule}_separate.npz', allow_pickle=False) as archive:
                references[rule] = {name: archive[name] for name in ('counts', 'resources', 'replay_steps', 'replay_resources')}
        for schedule, label in enumerate(('A', 'B', 'C', 'D')):
            sample = np.flatnonzero(trigger & (group_id == schedule))
            if not sample.size:
                continue
            idx = int(sample[0])
            order = ('replay', 'guarded') if len(rows) % 2 == 0 else ('guarded', 'replay')
            result = dict(group=group, schedule=label, sample=idx, order=list(order), full_seconds={},
                          replay_seconds={}, guard_seconds={}, replay_steps={})
            for rule in order:
                guard(start)
                tick = time.perf_counter()
                raw, diag, timing = guard_model.evaluate(inputs[group][idx:idx+1], w, Q, rule)
                result['full_seconds'][rule] = time.perf_counter()-tick
                result['replay_seconds'][rule] = timing['replay_wall_seconds']
                result['guard_seconds'][rule] = timing['guard_wall_seconds']
                result['replay_steps'][rule] = int(diag['replay_steps'][0])
                for name, values in references[rule].items():
                    assert np.array_equal((raw | diag)[name][0], values[idx]), (group, idx, rule, name)
            rows.append(result)
    return rows


def run():
    assert all(os.environ.get(key) == '1' for key in THREAD_KEYS)
    assert not any((ROOT/name).exists() for name in ('RUN_STATE.json', 'fixture', 'evaluate', 'SUMMARY.json'))
    verify_lock('FREEZE.json')
    start, cpu = time.perf_counter(), time.process_time()
    stage = 'fixture inputs'
    dump('RUN_STATE.json', dict(status='running_fixture', utc=utc(), scientific_data_generated=False))
    try:
        fixture, folder = ROOT/'fixture', ROOT/'evaluate'
        fixture.mkdir()
        folder.mkdir()
        tick = time.perf_counter()
        base, inputs, w = fixture_data.factorial()
        edge, edge_w, expected = fixture_data.edges()
        assert set(inputs) == set(GROUPS)
        assert all(x.shape == (256, T, 6) for x in inputs.values()) and edge.shape == (12, T, 6)
        input_seconds = time.perf_counter()-tick
        np.savez_compressed(fixture/'base.npz', **base)
        np.savez_compressed(fixture/'inputs.npz', **inputs)
        np.savez_compressed(fixture/'weights.npz', factorial=w, edge=edge_w)
        np.savez_compressed(fixture/'edge.npz', x=edge)
        dump('fixture/edge_expectations.json', expected)
        stage = 'packet parity'
        rows = []
        for group in GROUPS:
            rows.append(parity_group(folder, group, inputs[group], w, start))
        rows.append(parity_group(folder, 'edge', edge, edge_w, start, expected))
        dump('PARITY_ROWS.json', rows)
        assert sum(row['old_triggers'] for row in rows) > 0
        assert sum(row['guarded_triggers'] for row in rows) > 0 and sum(row['vetoes'] for row in rows) > 0
        stage = 'fixed paired timing'
        repeats = time_batches(inputs, w, start)
        singles = timed_single(folder, inputs, base, w, start)
        dump('TIMING.json', dict(batch_size=32, repeats=repeats, single=singles))
        stage = 'fixture ledger'
        guard(start, disk=True)
        total = {name: zero_work() for name in ('shared', 'separate')}
        for row in rows:
            for name in total:
                add_work(total[name], row['work'][name])
        dump('SUMMARY.json', dict(status='COMPLETE_FIXTURE', groups=len(rows), factorial_samples=256, edge_samples=12,
             factorial_seed=4099903201, native_init_seed=4099903202, scientific_data_generated=False, fits=0, updates=0,
             logical_paths=3*len(rows), checked_episodes=sum(row['episodes'] for row in rows),
             checks=sum(row['checks'] for row in rows), max_numeric_error=max(row['max_numeric_error'] for row in rows),
             work=total, parity_rows=rows, input_seconds=input_seconds, timing_repeats=len(repeats), isolated_pairs=len(singles),
             wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu, cap_seconds=CAP,
             peak_working_set_bytes=peak_memory(), artifact_bytes=artifact_size(), host=platform.node()))
        dump('RUN_STATE.json', dict(status='complete_fixture', utc=utc(), scientific_data_generated=False))
    except Exception as error:
        record = dict(status='stopped_fixture', stage=stage, reason=str(error), exception=type(error).__name__,
                      utc=utc(), wall_seconds=time.perf_counter()-start, scientific_data_generated=False)
        dump('FIXTURE_FAILURE.json', record)
        dump('RUN_STATE.json', record)
        raise


if __name__ == '__main__':
    {'freeze': freeze, 'fixture': run}[sys.argv[1]]()
