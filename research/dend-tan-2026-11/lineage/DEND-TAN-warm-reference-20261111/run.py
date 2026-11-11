"""Freeze and evaluate the saved bilinear matcher checkpoints with eviction-gated reference states."""

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
import warm_model


ROOT = Path(__file__).resolve().parent
ARMS = ('ORACLE', 'LEARN')
HISTORIES = ('full', 'late')
RULES = ('original', 'match_zero', 'eviction_zero', 'eviction_warm')
DIAGNOSTIC = 'full_eviction_late_native'
REFERENCE_STEPS = 25
CONDITIONS = tuple(f'{history}_{rule}' for history in HISTORIES for rule in RULES)
BLOCKS = 12
EVALUATIONS = (len(ARMS)*len(CONDITIONS)+1)*BLOCKS
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


def cost(arm, condition, slots, steps, reset_count, eligible_count, n):
    hard = matched = 0
    rule = condition.split('_', 1)[1]
    eviction = rule.startswith('eviction_')
    cosine = eligible_count/n
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
                normalization_scale_multiplies=12*matched,
                logical_latch_bits=int(rule != 'original'), eviction_latch_bits=int(eviction),
                prototype_onset_unit_steps=steps, prototype_eviction_unit_steps=steps,
                occupancy_checks_per_episode=writes, stored_reference_scalars=26 if rule == 'eviction_warm' else 0,
                cosine_norms_per_episode=3*cosine, cosine_denominator_multiplies_per_episode=2*cosine,
                cosine_divisions_upper_per_episode=2*cosine, reset_scalar_writes_mean=26*reset_count/n)


def freeze():
    assert not any((ROOT / name).exists() for name in ('FREEZE.json', 'RUN_STATE.json', 'evaluate', 'warm_reference'))
    assert read('PRECHECK.json')['status'] == 'PASS'
    timing = read('TIMING.json')
    assert timing['status'] == 'PASS' and timing['blocks'] == BLOCKS
    assert timing['scientific_data_generated'] is False
    selected = next(row['examples'] for row in timing['candidates']
                    if row['science_forecast_seconds'] < CAP and row['audit_forecast_seconds'] < 300)
    assert selected == timing['examples_per_block'] and selected in (512, 256)
    assert len(read('SOURCE_LOCK.json')['files']) == 11+len(ARMS)*BLOCKS
    verify_lock('SOURCE_LOCK.json')
    for name in ('independent_warm.py', 'independent_capacity_data.py', 'audit.py', 'PROTOCOL.md', 'REPRODUCE.md', 'summarize.py', 'report.py'):
        assert (ROOT / name).is_file(), name
    paths = [p for p in ROOT.iterdir() if p.suffix in ('.py', '.md', '.json') and p.is_file()]
    paths.extend((ROOT / 'weights').glob('*.npy'))
    dump('FREEZE.json', dict(utc=utc(), scientific_data_generated=False,
         blocks=BLOCKS, examples_per_block=selected, evaluations=EVALUATIONS,
         files={path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}))
    print(f'FROZEN {BLOCKS} blocks, {selected} new cases per block, {EVALUATIONS} evaluations', flush=True)


def save_block(folder, seed, n, ledger):
    base = capacity_data.make_base(n, 4081020000+100*seed)
    full_x, q = capacity_data.assemble(base, 'full4')
    late_x, late_q = capacity_data.assemble(base, 'base2')
    assert full_x.shape == late_x.shape == (n, 36, 6) and q == late_q == 28
    assert np.array_equal(full_x[:, 3:], late_x[:, 3:])
    assert not full_x[:, 9:25].any()
    np.savez_compressed(folder / f'test_s{seed}.npz', **base)
    ledger['input_array_bytes'] += full_x.nbytes+late_x.nbytes
    return base, full_x, late_x, q


def verify_rules(cases, arm, base, warm):
    expected_conditions = set(CONDITIONS)
    if arm == 'LEARN':
        expected_conditions.add(DIAGNOSTIC)
    assert set(cases) == expected_conditions
    assert warm.shape == (26,) and np.isfinite(warm).all()
    assert np.all((warm[20:] == 0.) | (warm[20:] == 1.))
    natural = cases['late_original']['cue_native_before'][:, 0]
    expected_eviction = np.broadcast_to(np.arange(36) > 4, (len(base['y']), 36))
    for condition, case in cases.items():
        history, rule = condition.split('_', 1)
        original = cases[f'{history}_original']
        onset = case['read_onset']
        reset = case['reset_steps']
        eviction = case['eviction_seen']
        assert onset.dtype == reset.dtype == eviction.dtype == np.bool_
        assert onset[:, 25].all() and onset.sum() == len(base['y'])
        assert np.array_equal(eviction, expected_eviction if history == 'full' else np.zeros_like(eviction))
        for field in ('read_onset', 'key_bank', 'value_bank', 'cue_scores', 'cue_weights', 'cue_read_vectors'):
            assert np.array_equal(case[field], original[field]), (arm, condition, field)
        for field in ('states', 'spikes', 'currents', 'injections', 'candidates', 'drive'):
            assert np.array_equal(case[field][:, :25], original[field][:, :25]), (arm, condition, field)
        confidence = case['match_confidence']
        if rule == 'original':
            assert not reset.any() and np.all(confidence == -2.)
        else:
            eligible = onset if rule == 'match_zero' else onset & eviction
            assert np.all(confidence[~eligible] == -2.)
            assert np.array_equal(reset, eligible & (confidence >= warm_model.COSINE_THRESHOLD))
            if arm == 'ORACLE':
                assert np.array_equal(reset[:, 25], eligible[:, 25] & (base['selected'] >= 2))
        reference = warm if rule == 'eviction_warm' else natural[:, None, :] if rule == 'eviction_late_native' else 0.
        expected = np.where(reset[:, 25:28, None], reference, case['cue_native_before'])
        assert np.array_equal(case['cue_native_after'], expected), (arm, condition, 'native replacement')
    for rule in ('eviction_zero', 'eviction_warm'):
        case = cases[f'late_{rule}']
        assert all(np.array_equal(case[field], array) for field, array in cases['late_original'].items()), (arm, rule)
    matched = cases['full_match_zero']
    assert all(np.array_equal(cases['full_eviction_zero'][key], value) for key, value in matched.items())
    for rule in ('eviction_warm', 'eviction_late_native'):
        condition = f'full_{rule}'
        if condition in cases:
            for field in ('match_confidence', 'reset_steps'):
                assert np.array_equal(cases[condition][field], matched[field]), (arm, rule, field)


def evaluate_model(folder, seed, arm, w, warm, base, full_x, late_x, q, rows, ledger, start):
    ident = f's{seed}_{arm}'
    n = len(base['y'])
    cases = {}
    for history, x in (('late', late_x), ('full', full_x)):
        for rule in RULES:
            guard(start)
            reference = warm if rule == 'eviction_warm' else None
            cases[f'{history}_{rule}'] = warm_model.evaluate(x, w, arm, q, rule, reference)
    if arm == 'LEARN':
        guard(start)
        natural = cases['late_original']['cue_native_before'][:, 0]
        cases[DIAGNOSTIC] = warm_model.evaluate(full_x, w, arm, q, 'eviction_late_native', natural)
    verify_rules(cases, arm, base, warm)
    for condition, raw in cases.items():
        assert all(np.isfinite(array).all() for array in raw.values()), (ident, condition)
        scores = capacity_metrics.score(raw, base)
        history, rule = condition.split('_', 1)
        if history == 'late':
            assert scores['older_ba'] == .5*scores['older_coverage']
        onset, eviction = raw['read_onset'], raw['eviction_seen']
        reset_count = int(raw['reset_steps'].sum())
        onset_count = int(onset.sum())
        eligible_count = onset_count if rule == 'match_zero' else int((onset & eviction).sum()) if rule != 'original' else 0
        reset = raw['reset_steps'].any(axis=1)
        position = base['selected']
        ledger['reset_count'] += reset_count
        ledger['cosine_onsets'] += eligible_count
        np.savez_compressed(folder / f'{ident}_{condition}.npz', **raw)
        ledger['retained_raw_array_bytes'] += sum(array.nbytes for array in raw.values())
        rows.append(dict(id=ident, seed=seed, arm=arm, condition=condition, T=36, q=q, slots=2,
                         **scores, mean_resources=raw['resources'][:, :4].mean(axis=0).tolist(),
                         reset_count=reset_count, onset_count=onset_count, eviction_fraction=float(eviction.any(axis=1).mean()),
                         reset_fraction=float(reset.mean()), older_reset_fraction=float(reset[position<2].mean()),
                         recent_reset_fraction=float(reset[position>=2].mean()),
                         reset_by_position=[float(reset[position==index].mean()) for index in range(4)],
                         cost_per_episode=cost(arm, condition, 2, 36, reset_count, eligible_count, n)))
    dump('evaluate/ROWS.json', rows)
    guard(start, disk=True)
    print(f'Evaluated {ident}: {len(rows)}/{EVALUATIONS}', flush=True)


def build_references(ledger, start):
    folder = ROOT / 'warm_reference'
    folder.mkdir()
    references = {}
    for seed in range(BLOCKS):
        for arm in ARMS:
            ident = f's{seed}_{arm}'
            w = np.load(ROOT / 'weights' / (ident+'.npy'), allow_pickle=False)
            warm = warm_model.warm_reference(w)
            assert warm.shape == (26,) and warm.dtype == np.float64 and np.isfinite(warm).all(), ident
            np.save(folder / (ident+'.npy'), warm, allow_pickle=False)
            ledger['warm_reference_bytes'] += warm.nbytes
            references[seed, arm] = w, warm
            guard(start)
    return references


def science():
    assert all(os.environ.get(key) == '1' for key in THREAD_KEYS)
    assert not any((ROOT / name).exists() for name in ('RUN_STATE.json', 'evaluate', 'warm_reference'))
    verify_lock('FREEZE.json')
    frozen = read('FREEZE.json')
    assert frozen['blocks'] == BLOCKS and frozen['evaluations'] == EVALUATIONS
    n = frozen['examples_per_block']
    assert n in (512, 256)
    start, cpu = time.perf_counter(), time.process_time()
    rows, ledger = [], dict(input_array_bytes=0, retained_raw_array_bytes=0, reset_count=0, cosine_onsets=0, warm_reference_bytes=0)
    dump('RUN_STATE.json', dict(status='running', utc=utc()))
    try:
        folder = ROOT / 'evaluate'
        folder.mkdir()
        dump('ENVIRONMENT.json', dict(machine=platform.node(), python=sys.version, numpy=np.__version__,
             threads={key: os.environ[key] for key in THREAD_KEYS}))
        references = build_references(ledger, start)
        for seed in range(BLOCKS):
            base, full_x, late_x, q = save_block(folder, seed, n, ledger)
            for arm in ARMS:
                w, warm = references[seed, arm]
                evaluate_model(folder, seed, arm, w, warm, base, full_x, late_x, q, rows, ledger, start)
        assert len(rows) == EVALUATIONS
        guard(start, disk=True)
        steps = EVALUATIONS*n*36
        warm_steps = len(references)*REFERENCE_STEPS
        dump('SCIENCE_SUMMARY.json', dict(status='COMPLETE', utc=utc(), blocks=BLOCKS, examples_per_block=n,
             fits=0, updates=0, evaluations=len(rows), forward_sample_steps=steps, dense_forward_macs=156*steps, matcher_macs=8*steps,
             wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu, cap_seconds=CAP,
             peak_working_set_bytes=peak_memory(), artifact_bytes_at_science_end=artifact_size(),
             reset_scalar_writes=26*ledger['reset_count'],
             warm_reference_count=len(references), warm_reference_steps=warm_steps,
             warm_reference_native_macs=156*warm_steps, warm_reference_matcher_macs=8*warm_steps,
             mechanism_diagnostic_evaluations=BLOCKS, mechanism_diagnostic_sample_steps=BLOCKS*n*36, **ledger))
        dump('RUN_STATE.json', dict(status='complete', utc=utc(), evaluations=len(rows)))
    except Exception as error:
        failure = dict(status='stopped', utc=utc(), reason=str(error), exception=type(error).__name__,
                       evaluations=len(rows), wall_seconds=time.perf_counter()-start)
        dump('SCIENCE_FAILURE.json', failure)
        dump('RUN_STATE.json', failure)
        raise


if __name__ == '__main__':
    {'freeze': freeze, 'science': science}[sys.argv[1]]()
