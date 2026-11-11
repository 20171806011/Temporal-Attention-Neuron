"""Freeze and evaluate the saved bilinear matcher checkpoints with exact paired natural-state donors."""

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
import donor_map
import warm_model


ROOT = Path(__file__).resolve().parent
ARMS = ('LEARN',)
DONOR_CASES = {'full_own_natural': 'own', 'full_target_flip': 'target',
               'full_other_flip': 'other', 'full_deleted_flip': 'deleted'}
CONDITIONS = ('late_original', 'full_original', 'full_eviction_warm', *DONOR_CASES)
REFERENCE_STEPS = 25
BLOCKS = 12
EVALUATIONS = len(CONDITIONS)*BLOCKS
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
    rule = 'eviction_late_native' if condition in DONOR_CASES else condition.split('_', 1)[1]
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
    assert len(read('SOURCE_LOCK.json')['files']) == 13+len(ARMS)*BLOCKS
    verify_lock('SOURCE_LOCK.json')
    for name in ('independent_donor.py', 'independent_warm.py', 'independent_capacity_data.py', 'audit.py', 'PROTOCOL.md', 'REPRODUCE.md', 'summarize.py', 'report.py'):
        assert (ROOT / name).is_file(), name
    paths = [p for p in ROOT.iterdir() if p.suffix in ('.py', '.md', '.json') and p.is_file()]
    paths.extend((ROOT / 'weights').glob('*.npy'))
    dump('FREEZE.json', dict(utc=utc(), scientific_data_generated=False,
         blocks=BLOCKS, examples_per_block=selected, evaluations=EVALUATIONS,
         files={path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}))
    print(f'FROZEN {BLOCKS} blocks, {selected} new cases per block, {EVALUATIONS} evaluations', flush=True)


def save_block(folder, seed, n, ledger):
    base = capacity_data.make_base(n, 4082020000+100*seed)
    full_x, q = capacity_data.assemble(base, 'full4')
    late_x, late_q = capacity_data.assemble(base, 'base2')
    assert full_x.shape == late_x.shape == (n, 36, 6) and q == late_q == 28
    assert np.array_equal(full_x[:, 3:], late_x[:, 3:])
    assert not full_x[:, 9:25].any()
    np.savez_compressed(folder / f'test_s{seed}.npz', **base)
    ledger['input_array_bytes'] += full_x.nbytes+late_x.nbytes
    return base, full_x, late_x, q


def verify_cases(cases, base, warm, maps):
    assert set(cases) == set(CONDITIONS)
    assert warm.shape == (26,) and np.isfinite(warm).all()
    assert np.all((warm[20:] == 0.) | (warm[20:] == 1.))
    full = cases['full_original']
    late = cases['late_original']
    own = cases['full_own_natural']
    onset = full['read_onset']
    assert onset.dtype == np.bool_ and onset[:, 25].all() and onset.sum() == len(base['y'])
    assert np.array_equal(full['eviction_seen'], np.broadcast_to(np.arange(36) > 4, onset.shape))
    assert not late['eviction_seen'].any()
    assert not full['reset_steps'].any() and not late['reset_steps'].any()
    mask = own['reset_steps']
    assert np.array_equal(mask, onset & full['eviction_seen'] & (own['match_confidence'] >= warm_model.COSINE_THRESHOLD))
    reference = late['cue_native_before'][:, 0]
    assert np.array_equal(reference, reference[maps['deleted']])
    older = base['selected'] < 2
    assert np.array_equal(reference[older], reference[maps['target'][older]])
    for condition, case in cases.items():
        assert np.array_equal(case['read_onset'], onset), condition
        if condition in ('late_original', 'full_original'):
            continue
        for field in ('eviction_seen', 'key_bank', 'value_bank', 'cue_scores', 'cue_weights', 'cue_read_vectors'):
            assert np.array_equal(case[field], full[field]), (condition, field)
        for field in ('states', 'spikes', 'currents', 'injections', 'candidates', 'drive'):
            assert np.array_equal(case[field][:, :25], full[field][:, :25]), (condition, field)
        for field in ('reset_steps', 'match_confidence'):
            assert np.array_equal(case[field], own[field]), (condition, field)
        assert np.array_equal(case['cue_native_before'][:, 0], full['cue_native_before'][:, 0]), condition
        replacement = warm if condition == 'full_eviction_warm' else reference[maps[DONOR_CASES[condition]], None, :]
        expected = np.where(mask[:, 25:28, None], replacement, case['cue_native_before'])
        assert np.array_equal(case['cue_native_after'], expected), (condition, 'native replacement')
        untouched = ~mask.any(axis=1)
        for field in ('counts', 'resources', 'cue_native_before', 'cue_native_after'):
            assert np.array_equal(case[field][untouched], full[field][untouched]), (condition, field, 'no trigger')
    assert all(np.array_equal(array, cases['full_deleted_flip'][field]) for field, array in own.items())
    for condition in ('full_target_flip', 'full_other_flip'):
        case = cases[condition]
        for field in ('counts', 'resources', 'cue_native_before', 'cue_native_after'):
            assert np.array_equal(case[field][older], own[field][older]), (condition, field, 'early query')


def late_original(x, w, q):
    # Keep representatives in first-appearance order so all eight traced rows can be restored.
    n = len(x)
    _, first, inverse = np.unique(x.reshape(n, -1), axis=0, return_index=True, return_inverse=True)
    assert len(first) == n//4 and np.array_equal(np.bincount(inverse), np.full(n//4, 4))
    order = np.argsort(first)
    representatives = first[order]
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    lookup = rank[inverse]
    assert np.array_equal(x[representatives][lookup], x)
    unique = warm_model.evaluate(x[representatives], w, 'LEARN', q, 'original')
    assert lookup[:8].max() < 8
    return {name: array[lookup if len(array) == len(representatives) else lookup[:8]]
            for name, array in unique.items()}


def evaluate_cases(base, full_x, late_x, q, w, warm, maps):
    late = late_original(late_x, w, q)
    full = warm_model.evaluate(full_x, w, 'LEARN', q, 'original')
    warmed = warm_model.evaluate(full_x, w, 'LEARN', q, 'eviction_warm', warm)
    cases = dict(late_original=late, full_original=full, full_eviction_warm=warmed)
    reference = late['cue_native_before'][:, 0]
    for condition, donor in DONOR_CASES.items():
        cases[condition] = warm_model.evaluate(full_x, w, 'LEARN', q, 'eviction_late_native', reference[maps[donor]])
    verify_cases(cases, base, warm, maps)
    return cases


def evaluate_model(folder, seed, arm, w, warm, base, full_x, late_x, q, rows, ledger, start):
    ident = f's{seed}_{arm}'
    n = len(base['y'])
    position, labels = base['selected'], base['y']
    recent = position >= 2
    guard(start)
    maps = donor_map.build(base)
    assert np.array_equal(late_x[maps['deleted']], late_x)
    assert np.array_equal(late_x[maps['target'][~recent]], late_x[~recent])
    np.savez_compressed(folder / f'donors_s{seed}.npz', **maps)
    ledger['donor_index_bytes'] += sum(array.nbytes for array in maps.values())
    cases = evaluate_cases(base, full_x, late_x, q, w, warm, maps)
    for condition, raw in cases.items():
        assert all(np.isfinite(array).all() for array in raw.values()), (ident, condition)
        scores = capacity_metrics.score(raw, base)
        if condition == 'late_original':
            assert scores['older_ba'] == .5*scores['older_coverage']
        onset, eviction = raw['read_onset'], raw['eviction_seen']
        reset_count = int(raw['reset_steps'].sum())
        onset_count = int(onset.sum())
        eligible_count = int((onset & eviction).sum()) if not condition.endswith('_original') else 0
        reset = raw['reset_steps'].any(axis=1)
        counts = raw['counts']
        margin = (2*labels.astype(float)-1)*(counts[:, 1]-counts[:, 0])
        answered = counts[:, 1] != counts[:, 0]
        ledger['reset_count'] += reset_count
        ledger['cosine_onsets'] += eligible_count
        np.savez_compressed(folder / f'{ident}_{condition}.npz', **raw)
        ledger['retained_raw_array_bytes'] += sum(array.nbytes for array in raw.values())
        rows.append(dict(id=ident, seed=seed, arm=arm, condition=condition, T=36, q=q, slots=2,
                         **scores, mean_resources=raw['resources'][:, :4].mean(axis=0).tolist(),
                         margin=float(margin.mean()), older_margin=float(margin[~recent].mean()),
                         recent_margin=float(margin[recent].mean()),
                         margin_by_position=[float(margin[position==i].mean()) for i in range(4)],
                         recent_margin_by_label=[float(margin[recent & (labels==i)].mean()) for i in range(2)],
                         recent_coverage_by_label=[float(answered[recent & (labels==i)].mean()) for i in range(2)],
                         reset_count=reset_count, onset_count=onset_count, eviction_fraction=float(eviction.any(axis=1).mean()),
                         reset_fraction=float(reset.mean()), older_reset_fraction=float(reset[~recent].mean()),
                         recent_reset_fraction=float(reset[recent].mean()),
                         reset_by_position=[float(reset[position==i].mean()) for i in range(4)],
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
    rows, ledger = [], dict(input_array_bytes=0, retained_raw_array_bytes=0, reset_count=0, cosine_onsets=0, warm_reference_bytes=0, donor_index_bytes=0)
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
        donor_steps = BLOCKS*(n//4)*36
        steps = 6*BLOCKS*n*36 + donor_steps
        warm_steps = len(references)*REFERENCE_STEPS
        dump('SCIENCE_SUMMARY.json', dict(status='COMPLETE', utc=utc(), blocks=BLOCKS, examples_per_block=n,
             fits=0, updates=0, evaluations=len(rows), forward_sample_steps=steps, dense_forward_macs=156*steps, matcher_macs=8*steps,
             wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu, cap_seconds=CAP,
             peak_working_set_bytes=peak_memory(), artifact_bytes_at_science_end=artifact_size(),
             reset_scalar_writes=26*ledger['reset_count'],
             warm_reference_count=len(references), warm_reference_steps=warm_steps,
             warm_reference_native_macs=156*warm_steps, warm_reference_matcher_macs=8*warm_steps,
             mechanism_diagnostic_evaluations=4*BLOCKS, mechanism_diagnostic_sample_steps=4*BLOCKS*n*36,
             donor_forward_evaluations=BLOCKS, donor_forward_sample_steps=donor_steps,
             donor_forward_unique_episodes=BLOCKS*(n//4), **ledger))
        dump('RUN_STATE.json', dict(status='complete', utc=utc(), evaluations=len(rows)))
    except Exception as error:
        failure = dict(status='stopped', utc=utc(), reason=str(error), exception=type(error).__name__,
                       evaluations=len(rows), wall_seconds=time.perf_counter()-start)
        dump('SCIENCE_FAILURE.json', failure)
        dump('RUN_STATE.json', failure)
        raise


if __name__ == '__main__':
    {'freeze': freeze, 'science': science}[sys.argv[1]]()
