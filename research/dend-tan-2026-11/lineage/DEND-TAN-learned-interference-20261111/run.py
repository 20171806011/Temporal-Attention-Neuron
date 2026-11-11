"""Freeze and evaluate the saved bilinear matcher checkpoints with crossed states."""

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
import interference_model


ROOT = Path(__file__).resolve().parent
ARMS = ('MEAN', 'ORACLE', 'LEARN')
CONDITIONS = ('full_full', 'late_full', 'full_late', 'late_late')
PREFIX_STEPS = 25
SUFFIX_STEPS = 11
BLOCKS = 12
EVALUATIONS = 12*BLOCKS
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


def cost(arm, condition, slots, steps):
    hard = matched = 0
    writes = 2 if condition.startswith('late_') else 4
    return dict(parameters=180 if arm == 'LEARN' else 176, allocated_parameter_slots=180,
                fixed_match_constants=0 if arm == 'LEARN' else 4, spiking_cells=8, graded_branches=12,
                continuous_state_slots=20+14*slots, key_slots=2*slots, value_slots=12*slots,
                persistent_array_bytes=8*(28+14*slots), dense_forward_macs=156*steps, graded_transfers=12*steps,
                matcher_dense_macs=(4+2*slots)*steps, matcher_logit_scale_multiplies=slots*steps,
                softmax_exponentials=slots*steps, softmax_divisions=slots*steps,
                query_norm_square_multiplies=2*steps, query_norm_adds=steps, query_norm_sqrts=steps,
                value_tanh_evaluations=12*steps, value_decay_multiplies=12*slots*steps,
                value_read_multiplies=12*slots*steps, value_read_adds=12*(slots-1)*steps,
                injection_multiplies=12*steps, fifo_scalar_writes=14*slots*writes,
                hard_equalities=slots*hard, hard_tie_adds=(slots-1)*hard, hard_divisions=slots*hard,
                extra_value_read_multiplies=12*slots*matched, extra_value_read_adds=12*(slots-1)*matched,
                extra_injection_multiplies=12*matched, normalization_norms=2*matched,
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
    assert selected == timing['examples_per_block'] and selected in (512, 256)
    assert len(read('SOURCE_LOCK.json')['files']) == 9+3*BLOCKS
    verify_lock('SOURCE_LOCK.json')
    for name in ('independent_interference.py', 'independent_capacity_data.py', 'audit.py', 'PROTOCOL.md', 'REPRODUCE.md', 'summarize.py', 'report.py'):
        assert (ROOT / name).is_file(), name
    paths = [p for p in ROOT.iterdir() if p.suffix in ('.py', '.md', '.json') and p.is_file()]
    paths.extend((ROOT / 'weights').glob('*.npy'))
    dump('FREEZE.json', dict(utc=utc(), scientific_data_generated=False,
         blocks=BLOCKS, examples_per_block=selected, evaluations=EVALUATIONS,
         files={path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}))
    print(f'FROZEN {BLOCKS} blocks, {selected} new cases per block, {EVALUATIONS} evaluations', flush=True)


def save_block(folder, seed, n, ledger):
    base = capacity_data.make_base(n, 4079020000+100*seed)
    full_x, q = capacity_data.assemble(base, 'full4')
    late_x, late_q = capacity_data.assemble(base, 'base2')
    assert full_x.shape == late_x.shape == (n, 36, 6) and q == late_q == 28
    assert np.array_equal(full_x[:, 3:], late_x[:, 3:])
    assert not full_x[:, 9:25].any()
    np.savez_compressed(folder / f'test_s{seed}.npz', **base)
    ledger['input_array_bytes'] += full_x.nbytes+late_x.nbytes
    return base, full_x, late_x, q


def verify_crossing(raw):
    full, late = raw['full_full'], raw['late_late']
    for condition, case in raw.items():
        native_source = late if condition.startswith('late_') else full
        value_source = late if condition.endswith('_late') else full
        assert np.array_equal(case['entry_native'], native_source['entry_native']), condition
        assert np.array_equal(case['entry_values'], value_source['entry_values']), condition
        assert np.array_equal(case['cue_read_vectors'], value_source['cue_read_vectors']), condition
        for field in ('entry_keys', 'cue_scores', 'cue_weights'):
            assert np.array_equal(case[field], full[field]), (condition, field)
        for field in ('states', 'spikes', 'currents', 'injections', 'candidates', 'drive', 'key_bank', 'value_bank'):
            assert np.array_equal(case[field][:, :PREFIX_STEPS],
                                  native_source[field][:, :PREFIX_STEPS]), (condition, field)


def evaluate_model(folder, seed, arm, base, full_x, late_x, q, rows, ledger, start):
    ident = f's{seed}_{arm}'
    w = np.load(ROOT / 'weights' / (ident+'.npy'), allow_pickle=False)
    guard(start)
    cases = interference_model.evaluate(full_x, late_x, w, arm, q)
    assert set(cases) == set(CONDITIONS)
    verify_crossing(cases)
    ledger['boundary_bank_bytes'] += sum(cases[case][field].nbytes
                                         for case in ('full_full', 'late_late')
                                         for field in ('entry_native', 'entry_keys', 'entry_values'))
    for condition, raw in cases.items():
        assert all(np.isfinite(array).all() for array in raw.values()), (ident, condition)
        scores = capacity_metrics.score(raw, base)
        if condition == 'late_late':
            assert scores['older_ba'] == .5*scores['older_coverage']
        np.savez_compressed(folder / f'{ident}_{condition}.npz', **raw)
        ledger['retained_raw_array_bytes'] += sum(array.nbytes for array in raw.values())
        rows.append(dict(id=ident, seed=seed, arm=arm, condition=condition, T=36, q=q, slots=2,
                         **scores, mean_resources=raw['resources'][:, :4].mean(axis=0).tolist(),
                         cost_per_episode=cost(arm, condition, 2, 36)))
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
    assert n in (512, 256)
    start, cpu = time.perf_counter(), time.process_time()
    rows, ledger = [], dict(boundary_bank_bytes=0, input_array_bytes=0, retained_raw_array_bytes=0)
    dump('RUN_STATE.json', dict(status='running', utc=utc()))
    try:
        folder = ROOT / 'evaluate'
        folder.mkdir()
        dump('ENVIRONMENT.json', dict(machine=platform.node(), python=sys.version, numpy=np.__version__,
             threads={key: os.environ[key] for key in THREAD_KEYS}))
        for seed in range(BLOCKS):
            base, full_x, late_x, q = save_block(folder, seed, n, ledger)
            for arm in ARMS:
                evaluate_model(folder, seed, arm, base, full_x, late_x, q, rows, ledger, start)
        assert len(rows) == EVALUATIONS
        guard(start, disk=True)
        steps = 3*BLOCKS*n*(2*PREFIX_STEPS+4*SUFFIX_STEPS)
        dump('SCIENCE_SUMMARY.json', dict(status='COMPLETE', utc=utc(), blocks=BLOCKS, examples_per_block=n,
             fits=0, updates=0, evaluations=len(rows), forward_sample_steps=steps, path_sample_steps=len(rows)*n*36, dense_forward_macs=156*steps, matcher_macs=8*steps,
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
