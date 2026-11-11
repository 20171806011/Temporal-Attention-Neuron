"""Freeze and evaluate the saved key-binding checkpoints with matched-norm reads."""

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

import amplitude_model
import binding_data
import metrics


ROOT = Path(__file__).resolve().parent
ARMS = ('GAIN2', 'KEY', 'KEY_MEAN')
MATCHED = {'swap_direction': 'nominal', 'swap_magnitude': 'swap_read', 'mean_matched': 'nominal'}
TOLERANCE = 1e-12
BLOCKS = 12
EVALUATIONS = 8*BLOCKS
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


def cost(arm, condition, steps):
    slots = 24 if arm in ('KEY', 'KEY_MEAN') else 0
    units = 12*steps if slots else 0
    selectors = 2*steps if slots else 0
    matched = steps if condition in MATCHED else 0
    return dict(parameters=176, spiking_cells=8, graded_branches=12, continuous_state_slots=20+slots,
                history_slots=0, trace_slots=slots, persistent_array_bytes=8*(28+slots),
                dense_forward_macs=156*steps, graded_transfers=12*steps,
                key_selection_unit_steps=selectors, key_normalization_divisions_upper=selectors,
                memory_tanh_evaluations=units, memory_read_adds=units, injection_multiplies=units,
                memory_retained_multiplies=2*units, memory_write_multiplies=2*units,
                memory_update_adds=2*units, memory_read_multiplies=2*units,
                extra_source_read_multiplies=24*matched, extra_source_read_adds=12*matched,
                extra_selector_unit_steps=2*matched, extra_source_read_gate_multiplies=matched,
                extra_source_injection_multiplies=12*matched, normalization_norms=2*matched,
                normalization_square_multiplies=24*matched, normalization_sum_adds=22*matched,
                normalization_sqrts=2*matched, normalization_ratio_divisions_upper=matched,
                normalization_scale_multiplies=12*matched)


def freeze():
    assert not any((ROOT / name).exists() for name in ('FREEZE.json', 'RUN_STATE.json', 'evaluate'))
    assert read('PRECHECK.json')['status'] == 'PASS'
    timing = read('TIMING.json')
    assert timing['status'] == 'PASS' and timing['blocks'] == BLOCKS
    assert timing['scientific_data_generated'] is False
    selected = next(row['examples'] for row in timing['candidates']
                    if row['science_forecast_seconds'] < CAP and row['audit_forecast_seconds'] < 300)
    assert selected == timing['examples_per_block'] and selected in (256, 128)
    assert len(read('SOURCE_LOCK.json')['checkpoints']) == 3*BLOCKS
    verify_lock('SOURCE_LOCK.json')
    for name in ('independent_amplitude.py', 'audit.py', 'PROTOCOL.md', 'REPRODUCE.md', 'summarize.py', 'report.py'):
        assert (ROOT / name).is_file(), name
    paths = [p for p in ROOT.iterdir() if p.suffix in ('.py', '.md', '.json') and p.is_file()]
    paths.extend((ROOT / 'weights').glob('*.npy'))
    dump('FREEZE.json', dict(utc=utc(), scientific_data_generated=False,
         blocks=BLOCKS, examples_per_block=selected, evaluations=EVALUATIONS,
         files={path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}))
    print(f'FROZEN {BLOCKS} blocks, {selected} new cases per block, {EVALUATIONS} evaluations', flush=True)


def scores_with_binding(counts, base):
    primary = metrics.score(counts, base)
    counterfactual = metrics.score(counts[:, ::-1], base)
    correct = counts[np.arange(len(counts)), base['y']] > counts[np.arange(len(counts)), 1-base['y']]
    disagree = base['first_bit'] != base['second_bit']
    return dict(**primary, counterfactual_joint=counterfactual['disagree_joint_correct'],
                counterfactual_disagree_ba=counterfactual['disagree_ba'],
                order0_disagree_ba=float(correct[disagree & (base['order'] == 0)].mean()),
                order1_disagree_ba=float(correct[disagree & (base['order'] == 1)].mean()))


def save_block(folder, seed, n, ledger):
    base = binding_data.make_base(n, 4076020000+100*seed)
    x, q = binding_data.assemble(base)
    assert x.shape == (n, 36, 6) and q == 28 and not x[:, 9:25].any()
    np.savez_compressed(folder / f'test_s{seed}.npz', **base)
    ledger['input_array_bytes'] += x.nbytes
    return base, x, q


def verify_key_conditions(raw):
    nominal = raw['nominal']
    for condition, values in raw.items():
        assert np.array_equal(nominal['cue_memory'], values['cue_memory']), condition
        for field in ('states', 'spikes', 'currents', 'injections', 'candidates', 'memory', 'drive'):
            assert np.array_equal(nominal[field][:, :25], values[field][:, :25]), (condition, field)
        assert np.array_equal(nominal['memory'], values['memory']), condition
        if condition in MATCHED:
            target = raw[MATCHED[condition]]['read_vectors']
            a = np.linalg.norm(values['read_vectors'], axis=-1)
            b = np.linalg.norm(target, axis=-1)
            assert np.max(np.abs(a-b)) <= TOLERANCE, condition


def evaluate_model(folder, seed, arm, base, x, q, rows, ledger, start):
    ident = f's{seed}_{arm}'
    w = np.load(ROOT / 'weights' / (ident+'.npy'), allow_pickle=False)
    raw_by_condition = {}
    for condition in amplitude_model.conditions(arm):
        guard(start)
        raw = amplitude_model.evaluate(x, w, arm, q, condition)
        assert all(np.isfinite(array).all() for array in raw.values()), (ident, condition)
        assert raw['read_vectors'].shape == (len(x), 3, 12)
        assert raw['direction_fallback'].shape == (len(x), 3) and raw['direction_fallback'].dtype == np.bool_
        fallback = int(raw['direction_fallback'].sum())
        if condition in MATCHED:
            ledger['matched_cases'] += 1
            ledger['matched_read_steps'] += len(x)*3
        else:
            assert fallback == 0
        ledger['fallback_reads'] += fallback
        np.savez_compressed(folder / f'{ident}_{condition}.npz', **raw)
        ledger['retained_raw_array_bytes'] += sum(array.nbytes for array in raw.values())
        rows.append(dict(id=ident, seed=seed, arm=arm, condition=condition, T=36, q=q,
                         **scores_with_binding(raw['counts'], base), mean_resources=raw['resources'][:, :4].mean(axis=0).tolist(),
                         cost_per_episode=cost(arm, condition, 36), fallback_reads=fallback))
        raw_by_condition[condition] = raw
    if arm == 'KEY':
        verify_key_conditions(raw_by_condition)
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
    rows, ledger = [], dict(matched_cases=0, matched_read_steps=0, fallback_reads=0,
                            input_array_bytes=0, retained_raw_array_bytes=0)
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
        steps = len(rows)*n*36
        dump('SCIENCE_SUMMARY.json', dict(status='COMPLETE', utc=utc(), blocks=BLOCKS, examples_per_block=n,
             fits=0, updates=0, evaluations=len(rows), forward_sample_steps=steps, dense_forward_macs=156*steps,
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
