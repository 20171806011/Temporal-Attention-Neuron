"""Freeze and evaluate the saved bilinear matcher checkpoints with equivalent replay implementations."""

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

import capacity_data
import capacity_metrics
import opt_model


ROOT = Path(__file__).resolve().parent
HISTORIES = ('four', 'three0', 'three2', 'two')
MODES = ('full', 'cached', 'sparse')
CONDITIONS = tuple(f'{history}_{mode}' for history in HISTORIES for mode in MODES)
BLOCKS = 12
EVALUATIONS = len(CONDITIONS)*BLOCKS
T = 36
ONLINE_DISCRETE = {'counts', 'resources', 'spikes', 'read_onset', 'reset_steps', 'eviction_seen',
                   'write_gate', 'key_bank'}
CAP = 180
MEMORY_CAP = 1024**3
ARTIFACT_CAP = int(1.5*1024**3)
THREAD_KEYS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')


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


def guard(start, disk=False):
    if time.perf_counter()-start > CAP:
        raise TimeoutError('180 second science limit')
    if peak_memory() > MEMORY_CAP:
        raise MemoryError('1 GiB peak working-set limit')
    if disk and artifact_size() > ARTIFACT_CAP:
        raise OSError('1.5 GiB artifact limit')


def freeze():
    assert not any((ROOT / name).exists() for name in ('FREEZE.json', 'RUN_STATE.json', 'evaluate', 'prefix_cache'))
    assert read('PRECHECK.json')['status'] == 'PASS'
    timing = read('TIMING.json')
    assert timing['status'] == 'PASS' and timing['blocks'] == BLOCKS
    assert timing['scientific_data_generated'] is False
    selected = next(row['examples'] for row in timing['candidates']
                    if row['science_forecast_seconds'] < CAP and row['audit_forecast_seconds'] < 300)
    assert selected == timing['examples_per_block'] and selected in (512, 256)
    assert len(read('SOURCE_LOCK.json')['files']) == 15+BLOCKS
    verify_lock('SOURCE_LOCK.json')
    for name in ('independent_opt.py', 'opt_model.py', 'independent_capacity_data.py', 'audit.py', 'PROTOCOL.md', 'REPRODUCE.md', 'summarize.py', 'report.py'):
        assert (ROOT / name).is_file(), name
    paths = [p for p in ROOT.iterdir() if p.suffix in ('.py', '.md', '.json') and p.is_file()]
    paths.extend((ROOT / 'weights').glob('*.npy'))
    dump('FREEZE.json', dict(utc=utc(), scientific_data_generated=False,
         blocks=BLOCKS, examples_per_block=selected, evaluations=EVALUATIONS,
         files={path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}))
    print(f'FROZEN {BLOCKS} blocks, {selected} new cases per block, {EVALUATIONS} evaluations', flush=True)


def assemble_histories(base):
    four, q = capacity_data.assemble(base, 'full4')
    two, q_two = capacity_data.assemble(base, 'base2')
    assert four.shape == two.shape == (len(base['y']), T, 6) and q == q_two == 28
    assert np.array_equal(four[:, 3:], two[:, 3:])
    early = base['group'] % 2 == 1
    assert early.sum() * 2 == len(early)
    for x in (four, two):
        x[early, 1, 4] = x[early, 5, 4]
        x[early, 5, 4] = 0.
    three0, three2 = four.copy(), four.copy()
    three0[:, 2], three2[:, 0] = 0., 0.
    return dict(four=four, three0=three0, three2=three2, two=two), q


def mode_order(seed):
    rotation = seed % len(MODES)
    return MODES[rotation:] + MODES[:rotation]


def verify_modes(history, outputs, early):
    full, logs, _ = outputs['full']
    reset = full['reset_steps'][:, 25]
    n = len(reset)
    assert full['read_onset'][:, 25].all() and full['read_onset'].sum() == n
    expected_earliest = np.where(reset, np.where(early, 1, 4), -1)
    if history == 'two':
        assert not reset.any() and not full['eviction_seen'].any()
    for mode, (raw, diag, timing) in outputs.items():
        assert raw.keys() == full.keys() and diag.keys() == logs.keys()
        for name, array in raw.items():
            if name in ONLINE_DISCRETE:
                assert np.array_equal(array, full[name]), (history, mode, name)
            else:
                np.testing.assert_allclose(array, full[name], rtol=0., atol=5e-12,
                                           err_msg=f'{history} {mode} {name}')
        for name in ('cue_native_before', 'cue_native_after'):
            assert np.array_equal(raw[name][..., 20:], full[name][..., 20:]), (history, mode, name, 'p')
        for name, array in diag.items():
            if name.startswith('logged_') or name == 'first_read_tick':
                assert np.array_equal(array, logs[name]), (history, mode, name)
        np.testing.assert_allclose(diag['replay_reference'], logs['replay_reference'], atol=5e-12, rtol=0.)
        assert np.array_equal(diag['replay_reference'][:, 20:], logs['replay_reference'][:, 20:])
        assert np.array_equal(diag['earliest_tick'], expected_earliest)
        assert np.array_equal(diag['replay_nonzero_steps'], reset*4)
        executed = np.where(reset, 25 if mode == 'full' else 25-expected_earliest, 0)
        assert np.array_equal(diag['replay_steps'], executed)
        assert np.array_equal(diag['replay_input_rows'], reset*4 if mode == 'sparse' else executed)
        if not reset.any():
            assert not any(timing.values()) and not diag['replay_resources'].any()
    assert np.array_equal(outputs['cached'][1]['replay_resources'], outputs['sparse'][1]['replay_resources'])


def evaluate_cases(inputs, q, w, cache, early, order):
    results = {}
    for history, x in inputs.items():
        modes = {mode: opt_model.evaluate(x, w, q, mode, cache if mode != 'full' else None) for mode in order}
        verify_modes(history, modes, early)
        results.update({f'{history}_{mode}': value for mode, value in modes.items()})
    assert set(results) == set(CONDITIONS)
    return results


def replay_totals(diag, mask):
    steps = int(diag['replay_steps'][mask].sum())
    rows = int(diag['replay_input_rows'][mask].sum())
    return dict(triggers=int(np.count_nonzero(diag['replay_steps'][mask])), steps=steps, input_rows=rows,
                nonzero_steps=int(diag['replay_nonzero_steps'][mask].sum()), native_macs=84*steps+72*rows,
                resources=diag['replay_resources'][mask].sum(axis=0).tolist())


def add_totals(total, values, timing):
    for name in ('triggers', 'steps', 'input_rows', 'nonzero_steps', 'native_macs'):
        total[name] += values[name]
    total['resources'] = [a+b for a, b in zip(total['resources'], values['resources'])]
    total['replay_wall_seconds'] += timing['replay_wall_seconds']
    total['replay_batches'] += timing['replay_batches']
    total['replay_scratch_peak_bytes'] = max(total['replay_scratch_peak_bytes'], timing['replay_scratch_peak_bytes'])


def cost(mode, diag, totals, n):
    used = (diag['logged_write_ticks'] >= 0).sum(axis=1) + diag['logged_nonwrite_count']
    allocated = sum(diag[name].nbytes for name in ('logged_write_values', 'logged_write_ticks',
                    'logged_nonwrite_values', 'logged_nonwrite_ticks', 'logged_nonwrite_count'))//n
    return dict(persistent_array_bytes=448, log_allocated_bytes=allocated,
                log_live_bytes_mean=float((used*56+8).mean()), first_read_tick_bytes=8, logical_latch_bits=3,
                cache_bytes_per_model=0 if mode == 'full' else 37*26*8,
                replay_native_steps_mean=totals['steps']/n, replay_input_rows_mean=totals['input_rows']/n,
                replay_native_macs_mean=totals['native_macs']/n,
                replay_resources_mean=[value/n for value in totals['resources']])


def measure_single(seed, x, q, w, cache, early, order, results, start):
    records, steps, macs = [], 0, 0
    triggered = results['four_full'][0]['reset_steps'][:, 25]
    for schedule, mask in (('standard', ~early), ('early', early)):
        for sample in np.flatnonzero(triggered & mask)[:8]:
            guard(start)
            values = {}
            for mode in order:
                raw, diag, timing = opt_model.evaluate(x[sample:sample+1], w, q, mode, cache if mode != 'full' else None)
                batch, batch_diag, _ = results[f'four_{mode}']
                assert np.array_equal(raw['counts'][0], batch['counts'][sample])
                assert np.array_equal(raw['resources'][0], batch['resources'][sample])
                assert np.array_equal(diag['replay_resources'][0], batch_diag['replay_resources'][sample])
                assert timing['replay_batches'] == 1 and timing['replay_wall_seconds'] > 0.
                native_steps = int(diag['replay_steps'][0])
                input_rows = int(diag['replay_input_rows'][0])
                assert native_steps == batch_diag['replay_steps'][sample]
                assert input_rows == batch_diag['replay_input_rows'][sample]
                steps += native_steps
                macs += 84*native_steps+72*input_rows
                values[mode] = timing['replay_wall_seconds']
            records.append(dict(seed=seed, sample=int(sample), schedule=schedule, order=list(order), seconds=values))
    return records, steps, macs


def evaluate_block(folder, seed, n, w, cache, rows, records, ledger, start):
    guard(start)
    ident = f's{seed}_LEARN'
    base = capacity_data.make_base(n, 4084020000+100*seed)
    inputs, q = assemble_histories(base)
    np.savez_compressed(folder / f'test_s{seed}.npz', **base)
    ledger['input_array_bytes'] += sum(array.nbytes for array in inputs.values())
    early = base['group'] % 2 == 1
    order = mode_order(seed)
    cases = evaluate_cases(inputs, q, w, cache, early, order)
    for condition, (raw, diag, timing) in cases.items():
        history, mode = condition.split('_', 1)
        assert all(np.isfinite(array).all() for array in (*raw.values(), *diag.values()))
        scores = capacity_metrics.score(raw, base)
        mask = np.ones(n, dtype=bool)
        values = replay_totals(diag, mask)
        add_totals(ledger['mode_totals'][mode], values, timing)
        schedules = {name: dict(episodes=int(selection.sum()), **replay_totals(diag, selection))
                     for name, selection in (('standard', ~early), ('early', early))}
        np.savez_compressed(folder / f'{ident}_{condition}.npz', **raw, **diag)
        ledger['retained_raw_array_bytes'] += sum(array.nbytes for array in (*raw.values(), *diag.values()))
        rows.append(dict(id=ident, seed=seed, arm='LEARN', condition=condition, history=history, mode=mode, T=T, q=q,
                         slots=2, **scores, mean_resources=raw['resources'][:, :4].mean(axis=0).tolist(),
                         reset_count=int(raw['reset_steps'].sum()), onset_count=int(raw['read_onset'].sum()),
                         eviction_fraction=float(raw['eviction_seen'].any(axis=1).mean()),
                         **timing, cost_per_episode=cost(mode, diag, values, n), replay_by_schedule=schedules))
        guard(start, disk=True)
    subset, steps, macs = measure_single(seed, inputs['four'], q, w, cache, early, order, cases, start)
    records.extend(subset)
    ledger['microbench_replay_steps'] += steps
    ledger['microbench_replay_native_macs'] += macs
    dump('evaluate/ROWS.json', rows)
    dump('MICROBENCH.json', records)
    print(f'Evaluated {ident}: {len(rows)}/{EVALUATIONS}, paired isolated timings {len(records)}', flush=True)


def build_caches(start):
    folder = ROOT / 'prefix_cache'
    folder.mkdir()
    items, seconds = [], 0.
    for seed in range(BLOCKS):
        ident = f's{seed}_LEARN'
        w = np.load(ROOT / 'weights' / (ident+'.npy'), allow_pickle=False)
        tick = time.perf_counter()
        cache = opt_model.build_cache(w)
        seconds += time.perf_counter()-tick
        assert cache.shape == (T+1, 26) and cache.dtype == np.float64 and np.isfinite(cache).all()
        assert not cache[0].any() and np.all((cache[:, 20:] == 0) | (cache[:, 20:] == 1))
        np.save(folder / (ident+'.npy'), cache, allow_pickle=False)
        items.append((w, cache))
        guard(start)
    return items, seconds


def science():
    assert all(os.environ.get(key) == '1' for key in THREAD_KEYS)
    assert not any((ROOT / name).exists() for name in ('RUN_STATE.json', 'evaluate', 'prefix_cache', 'MICROBENCH.json'))
    verify_lock('FREEZE.json')
    frozen = read('FREEZE.json')
    assert frozen['blocks'] == BLOCKS and frozen['evaluations'] == EVALUATIONS
    n = frozen['examples_per_block']
    assert n in (512, 256)
    start, cpu = time.perf_counter(), time.process_time()
    rows, records = [], []
    ledger = dict(input_array_bytes=0, retained_raw_array_bytes=0, microbench_replay_steps=0, microbench_replay_native_macs=0,
                  mode_totals={mode: dict(triggers=0, steps=0, input_rows=0, nonzero_steps=0, native_macs=0,
                               resources=[0.]*4, replay_wall_seconds=0., replay_batches=0, replay_scratch_peak_bytes=0)
                               for mode in MODES})
    dump('RUN_STATE.json', dict(status='running', utc=utc()))
    try:
        folder = ROOT / 'evaluate'
        folder.mkdir()
        dump('ENVIRONMENT.json', dict(machine=platform.node(), python=sys.version, numpy=np.__version__,
             threads={key: os.environ[key] for key in THREAD_KEYS}))
        items, cache_seconds = build_caches(start)
        for seed, (w, cache) in enumerate(items):
            evaluate_block(folder, seed, n, w, cache, rows, records, ledger, start)
        assert len(rows) == EVALUATIONS
        guard(start, disk=True)
        steps = EVALUATIONS*n*T
        cache_steps = len(items)*T
        dump('SCIENCE_SUMMARY.json', dict(status='COMPLETE', utc=utc(), blocks=BLOCKS, examples_per_block=n,
             fits=0, updates=0, evaluations=len(rows), forward_sample_steps=steps, dense_forward_macs=156*steps, matcher_macs=8*steps,
             wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu, cap_seconds=CAP,
             peak_working_set_bytes=peak_memory(), artifact_bytes_at_science_end=artifact_size(),
             cache_count=len(items), cache_bytes=sum(cache.nbytes for _, cache in items), cache_build_steps=cache_steps,
             cache_build_native_macs=156*cache_steps, cache_build_wall_seconds=cache_seconds,
             microbench_records=len(records), microbench_evaluations=3*len(records), microbench_forward_sample_steps=3*len(records)*T,
             **ledger))
        dump('RUN_STATE.json', dict(status='complete', utc=utc(), evaluations=len(rows)))
    except Exception as error:
        failure = dict(status='stopped', utc=utc(), reason=str(error), exception=type(error).__name__,
                       evaluations=len(rows), wall_seconds=time.perf_counter()-start)
        dump('SCIENCE_FAILURE.json', failure)
        dump('RUN_STATE.json', failure)
        raise


if __name__ == '__main__':
    {'freeze': freeze, 'science': science}[sys.argv[1]]()
