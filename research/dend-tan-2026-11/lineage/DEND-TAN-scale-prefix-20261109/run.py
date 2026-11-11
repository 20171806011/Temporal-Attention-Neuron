"""Freeze and evaluate the saved D16 checkpoints under prefix rule and state-source interventions."""

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

import prefix
import metrics
import task_data


ROOT = Path(__file__).resolve().parent
ARMS = ('GAIN2', 'SCALE', 'SCALE_MEAN')
BLOCKS = 12
EVALUATIONS = 7*BLOCKS
CAP = 840
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
        raise TimeoutError('840 second science limit')
    if peak_memory() > MEMORY_CAP:
        raise MemoryError('1 GiB peak working-set limit')
    if disk and artifact_size() > ARTIFACT_CAP:
        raise OSError('1.5 GiB artifact limit')


def cost(arm, steps):
    traces = 0 if arm == 'GAIN2' else 24
    units = 12*steps if traces else 0
    return dict(parameters=176, spiking_cells=8, graded_branches=12,
                continuous_state_slots=20+traces, history_slots=0, trace_slots=traces,
                persistent_array_bytes=8*(28+traces), dense_forward_macs=156*steps,
                graded_transfers=12*steps, history_roll_writes=0, attention_unit_steps=0,
                softmax_exponentials=0, trace_tanh_evaluations=units, trace_logistic_evaluations=units,
                trace_retained_multiplies=2*units, trace_write_multiplies=units,
                trace_update_adds=2*units, trace_context_unit_steps=units)


def freeze():
    assert not any((ROOT / name).exists() for name in ('FREEZE.json', 'RUN_STATE.json', 'evaluate'))
    assert read('PRECHECK.json')['status'] == 'PASS'
    timing = read('TIMING.json')
    assert timing['status'] == 'PASS' and timing['blocks'] == BLOCKS
    assert timing['scientific_data_generated'] is False
    selected = next(row['examples'] for row in timing['candidates']
                    if row['science_forecast_seconds'] < CAP and row['audit_forecast_seconds'] < 300)
    assert selected == timing['examples_per_block'] and selected in (256, 128)
    assert len(read('SOURCE_LOCK.json')['files']) == 9+3*BLOCKS
    verify_lock('SOURCE_LOCK.json')
    for name in ('independent.py', 'audit.py', 'PROTOCOL.md', 'REPRODUCE.md', 'summarize.py', 'report.py'):
        assert (ROOT / name).is_file(), name
    paths = [p for p in ROOT.iterdir() if p.suffix in ('.py', '.md', '.json') and p.is_file()]
    paths.extend((ROOT / 'weights').glob('*.npy'))
    dump('FREEZE.json', dict(utc=utc(), scientific_data_generated=False,
         blocks=BLOCKS, examples_per_block=selected, evaluations=EVALUATIONS,
         files={path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}))
    print(f'FROZEN {BLOCKS} blocks, {selected} new cases per block, {EVALUATIONS} evaluations', flush=True)


def save_block(folder, seed, n, ledger):
    base = task_data.make_base(n, 4073020000+100*seed)
    x, q = task_data.assemble(base, 16, 'nominal')
    assert x.shape == (n, 36, 6) and q == 28 and not x[:, 9:25].any()
    np.savez_compressed(folder / f'test_s{seed}.npz', **base)
    ledger['input_array_bytes'] += x.nbytes
    # Four 50-value trace banks and one 26-value GAIN2 bank, per sample.
    ledger['boundary_bank_bytes'] += n*8*(26+4*50)
    return base, x, q


def evaluate_model(folder, seed, arm, base, x, q, rows, ledger, start):
    ident = f's{seed}_{arm}'
    w = np.load(ROOT / 'weights' / (ident+'.npy'), allow_pickle=False)
    guard(start)
    results = prefix.evaluate(x, w, arm, q)
    assert set(results) == set(prefix.conditions(arm))
    if arm == 'SCALE_MEAN':
        assert all(np.array_equal(array, results['prefix_mean'][key])
                   for key, array in results['nominal'].items()), ident
    if arm == 'SCALE':
        nominal, mean = (results[key] for key in ('nominal', 'prefix_mean'))
        md, dm = (results[key] for key in ('mean_native', 'mean_traces'))
        assert np.array_equal(md['entry_native'], mean['entry_native'])
        assert np.array_equal(md['entry_traces'], nominal['entry_traces'])
        assert np.array_equal(dm['entry_native'], nominal['entry_native'])
        assert np.array_equal(dm['entry_traces'], mean['entry_traces'])
    for condition, raw in results.items():
        assert all(np.isfinite(array).all() for array in raw.values()), (ident, condition)
        np.savez_compressed(folder / f'{ident}_{condition}.npz', **raw)
        ledger['retained_raw_array_bytes'] += sum(array.nbytes for array in raw.values())
        rows.append(dict(id=ident, seed=seed, arm=arm, condition=condition, T=36, q=q,
                         **metrics.score(raw['counts'], base), mean_resources=raw['resources'].mean(axis=0).tolist(),
                         cost_per_episode=cost(arm, 36)))
    dump('evaluate/ROWS.json', rows)
    guard(start, disk=True)
    print(f'Evaluated {ident}: {len(rows)}/{EVALUATIONS}', flush=True)


def science():
    assert all(os.environ.get(key) == '1' for key in THREAD_KEYS)
    assert not any((ROOT / name).exists() for name in ('RUN_STATE.json', 'evaluate'))
    verify_lock('FREEZE.json')
    frozen = read('FREEZE.json')
    assert frozen['blocks'] == BLOCKS and frozen['evaluations'] == EVALUATIONS
    n = frozen['examples_per_block']
    assert n in (256, 128)
    start, cpu = time.perf_counter(), time.process_time()
    rows, ledger = [], dict(boundary_bank_bytes=0, input_array_bytes=0,
                            retained_raw_array_bytes=0)
    dump('RUN_STATE.json', dict(status='running', utc=utc()))
    try:
        folder = ROOT / 'evaluate'
        folder.mkdir()
        dump('ENVIRONMENT.json', dict(machine=platform.node(), python=sys.version, numpy=np.__version__,
             threads={key: os.environ[key] for key in THREAD_KEYS}))
        for seed in range(BLOCKS):
            base, x, q = save_block(folder, seed, n, ledger)
            for arm in ARMS:
                evaluate_model(folder, seed, arm, base, x, q, rows, ledger, start)
        assert len(rows) == EVALUATIONS
        guard(start, disk=True)
        steps = BLOCKS*n*(5*25+7*11)
        dump('SCIENCE_SUMMARY.json', dict(status='COMPLETE', utc=utc(), blocks=BLOCKS, examples_per_block=n,
             fits=0, updates=0, evaluations=len(rows), forward_sample_steps=steps, path_sample_steps=len(rows)*n*36, dense_forward_macs=156*steps,
             wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu, cap_seconds=CAP,
             peak_working_set_bytes=peak_memory(), artifact_bytes_at_science_end=artifact_size(), **ledger))
        dump('RUN_STATE.json', dict(status='complete', utc=utc(), evaluations=len(rows)))
    except Exception as error:
        failure = dict(status='stopped', utc=utc(), reason=str(error), exception=type(error).__name__,
                       evaluations=len(rows), wall_seconds=time.perf_counter()-start)
        dump('SCIENCE_FAILURE.json', failure)
        dump('RUN_STATE.json', failure)
        raise


if __name__ == '__main__':
    {'freeze': freeze, 'science': science}[sys.argv[1]]()
