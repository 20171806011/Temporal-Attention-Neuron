"""Freeze and evaluate the saved bilinear matcher checkpoints with four FIFO slots."""

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
import capacity_model


ROOT = Path(__file__).resolve().parent
ARMS = ('MEAN', 'ORACLE', 'LEARN')
MATCHED = 'full4_hard_matched'
TOLERANCE = 1e-12
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
    hard = steps if condition in ('full4_hard', MATCHED) else 0
    matched = steps if condition == MATCHED else 0
    writes = 2 if condition == 'base2' else 4
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
    assert len(read('SOURCE_LOCK.json')['checkpoints']) == 3*BLOCKS
    verify_lock('SOURCE_LOCK.json')
    for name in ('independent_capacity.py', 'independent_capacity_data.py', 'audit.py', 'PROTOCOL.md', 'REPRODUCE.md', 'summarize.py', 'report.py'):
        assert (ROOT / name).is_file(), name
    paths = [p for p in ROOT.iterdir() if p.suffix in ('.py', '.md', '.json') and p.is_file()]
    paths.extend((ROOT / 'weights').glob('*.npy'))
    dump('FREEZE.json', dict(utc=utc(), scientific_data_generated=False,
         blocks=BLOCKS, examples_per_block=selected, evaluations=EVALUATIONS,
         files={path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}))
    print(f'FROZEN {BLOCKS} blocks, {selected} new cases per block, {EVALUATIONS} evaluations', flush=True)


def save_block(folder, seed, n, ledger):
    base = capacity_data.make_base(n, 4078020000+100*seed)
    inputs = {condition: capacity_data.assemble(base, condition) for condition in ('full4', 'base2')}
    assert all(x.shape == (n, 36, 6) and q == 28 and not x[:, 9:25].any() for x, q in inputs.values())
    np.savez_compressed(folder / f'test_s{seed}.npz', **base)
    ledger['input_array_bytes'] += sum(x.nbytes for x, _ in inputs.values())
    return base, inputs


def verify_variants(raw):
    baseline = raw['full4']
    for condition in ('full4_hard', MATCHED):
        if condition not in raw:
            continue
        values = raw[condition]
        for field in ('key_bank', 'value_bank', 'cue_scores', 'cue_soft_weights', 'read_gate', 'write_gate'):
            assert np.array_equal(baseline[field], values[field]), (condition, field)
        for field in ('states', 'spikes', 'currents', 'injections', 'candidates', 'drive'):
            assert np.array_equal(baseline[field][:, :25], values[field][:, :25]), (condition, field)
        if condition == MATCHED:
            left = np.linalg.norm(values['cue_read_vectors'], axis=-1)
            right = np.linalg.norm(baseline['cue_read_vectors'], axis=-1)
            assert np.max(np.abs(left-right)) <= TOLERANCE, condition


def evaluate_model(folder, seed, arm, base, inputs, rows, ledger, start):
    ident = f's{seed}_{arm}'
    w = np.load(ROOT / 'weights' / (ident+'.npy'), allow_pickle=False)
    raw_by_condition = {}
    for condition in capacity_model.conditions(arm):
        guard(start)
        x, q = inputs['base2' if condition == 'base2' else 'full4']
        raw = capacity_model.evaluate(x, w, arm, q, condition)
        assert all(np.isfinite(array).all() for array in raw.values()), (ident, condition)
        slots = 2 if condition in ('base2', 'full2') else 4
        assert raw['cue_weights'].shape == (len(x), 3, slots)
        assert raw['cue_read_vectors'].shape == (len(x), 3, 12)
        assert raw['cue_fallback'].shape == (len(x), 3) and raw['cue_fallback'].dtype == np.bool_
        fallback = int(raw['cue_fallback'].sum())
        if condition == MATCHED:
            ledger['matched_cases'] += 1
            ledger['matched_read_steps'] += len(x)*3
        else:
            assert fallback == 0
        ledger['fallback_reads'] += fallback
        scores = capacity_metrics.score(raw, base)
        if condition == 'base2':
            assert scores['older_ba'] == .5*scores['older_coverage']
        if arm == 'ORACLE':
            assert scores['recent_routing_accuracy'] == 1.
            if slots == 4:
                assert scores['routing_accuracy'] == 1.
        np.savez_compressed(folder / f'{ident}_{condition}.npz', **raw)
        ledger['retained_raw_array_bytes'] += sum(array.nbytes for array in raw.values())
        rows.append(dict(id=ident, seed=seed, arm=arm, condition=condition, T=36, q=q, slots=slots,
                         **scores, mean_resources=raw['resources'][:, :4].mean(axis=0).tolist(),
                         cost_per_episode=cost(arm, condition, slots, 36), fallback_reads=fallback))
        raw_by_condition[condition] = raw
    verify_variants(raw_by_condition)
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
    rows, ledger = [], dict(matched_cases=0, matched_read_steps=0, fallback_reads=0,
                            input_array_bytes=0, retained_raw_array_bytes=0)
    dump('RUN_STATE.json', dict(status='running', utc=utc()))
    try:
        folder = ROOT / 'evaluate'
        folder.mkdir()
        dump('ENVIRONMENT.json', dict(machine=platform.node(), python=sys.version, numpy=np.__version__,
             threads={key: os.environ[key] for key in THREAD_KEYS}))
        for seed in range(BLOCKS):
            base, inputs = save_block(folder, seed, n, ledger)
            for arm in ARMS:
                evaluate_model(folder, seed, arm, base, inputs, rows, ledger, start)
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
