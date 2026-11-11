"""Freeze and evaluate the saved bilinear matcher checkpoints with evicted-key competition."""

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
import confirm_data
import guard_model
import learned_model


ROOT = Path(__file__).resolve().parent
HISTORIES = ('four', 'three0', 'three2', 'two')
GEOMETRIES = ('orthogonal', 'near')
RULES = ('original', 'replay', 'guarded')
CONDITIONS = tuple(f'{geometry}_{history}_{rule}' for geometry in GEOMETRIES for history in HISTORIES for rule in RULES)
WRITES = {'four': 4, 'three0': 3, 'three2': 3, 'two': 2}
EVICTED = {'three0': 0, 'three2': 1}
BLOCKS = 12
EVALUATIONS = len(CONDITIONS)*BLOCKS
T = 36
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


def guard(start, disk=False, cap=CAP):
    if time.perf_counter()-start > cap:
        raise TimeoutError(f'{cap} second stage limit')
    if peak_memory() > MEMORY_CAP:
        raise MemoryError('1 GiB peak working-set limit')
    if disk and artifact_size() > ARTIFACT_CAP:
        raise OSError('1.5 GiB artifact limit')


def cost(history, rule, reset_count, eligible_count, diagnostics, n):
    slots, steps = 2, T
    cosine = eligible_count/n
    hard = matched = 0
    writes = WRITES[history]
    replay = rule != 'original'
    allocated = live = extra_steps = 0
    if replay:
        allocated = sum(diagnostics[field].nbytes for field in (
            'logged_write_values', 'logged_write_ticks', 'logged_nonwrite_values',
            'logged_nonwrite_ticks', 'logged_nonwrite_count'))//n
        used = (diagnostics['logged_write_ticks'] >= 0).sum(axis=1) + diagnostics['logged_nonwrite_count']
        live = float((used*56+8).mean())
        extra_steps = float(diagnostics['replay_steps'].mean())
    return dict(parameters=180, allocated_parameter_slots=180,
                fixed_match_constants=0, spiking_cells=8, graded_branches=12,
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
                logical_latch_bits=int(rule != 'original'), eviction_latch_bits=int(rule != 'original'),
                prototype_onset_unit_steps=steps, prototype_eviction_unit_steps=steps,
                occupancy_checks_per_episode=writes, stored_reference_scalars=0,
                cosine_norms_per_episode=3*cosine, cosine_denominator_multiplies_per_episode=2*cosine,
                cosine_divisions_upper_per_episode=2*cosine, reset_scalar_writes_mean=26*reset_count/n,
                replay_allocated_log_bytes=allocated, replay_live_log_bytes_mean=live,
                replay_logical_first_read_bits=int(replay), replay_prototype_first_read_tick_bytes=8*replay,
                replay_native_steps_mean=extra_steps, replay_native_macs_mean=156*extra_steps,
                evicted_key_bytes=32 if rule == 'guarded' else 0, evicted_key_metadata_bytes=0,
                evicted_key_cosines_mean=float(diagnostics['evicted_comparisons'].sum()/n) if rule == 'guarded' else 0.,
                evicted_key_log_writes_mean=writes-2 if rule == 'guarded' else 0)


def freeze():
    assert not any((ROOT / name).exists() for name in ('FREEZE.json', 'RUN_STATE.json', 'evaluate', 'MICROBENCH.json'))
    assert read('PRECHECK.json')['status'] == 'PASS'
    timing = read('TIMING.json')
    assert timing['status'] == 'PASS' and timing['blocks'] == BLOCKS
    assert timing['scientific_data_generated'] is False
    selected = next(row['examples'] for row in timing['candidates']
                    if row['science_forecast_seconds'] < CAP and row['audit_forecast_seconds'] < 300)
    assert selected == timing['examples_per_block'] and selected in (512, 256)
    assert len(read('SOURCE_LOCK.json')['files']) == 16+BLOCKS
    verify_lock('SOURCE_LOCK.json')
    for name in ('guard_model.py', 'independent_guard.py', 'audit.py', 'PROTOCOL.md', 'REPRODUCE.md', 'summarize.py', 'report.py'):
        assert (ROOT / name).is_file(), name
    paths = [path for path in ROOT.iterdir() if path.suffix in ('.py', '.md', '.json') and path.is_file()]
    paths.extend((ROOT/'weights').glob('*.npy'))
    dump('FREEZE.json', dict(utc=utc(), scientific_data_generated=False, blocks=BLOCKS, examples_per_block=selected,
         evaluations=EVALUATIONS, files={path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(paths)}))
    print(f'FROZEN {BLOCKS} new stimulus blocks × {selected} samples; {EVALUATIONS} evaluations', flush=True)


def geometry_base(base, geometry):
    if geometry == 'orthogonal':
        return base
    assert geometry == 'near'
    phase = base['angle'][:, None] + np.array([0., np.pi/4, np.pi, 5*np.pi/4])
    keys = np.stack((np.cos(phase), np.sin(phase)), axis=-1)
    queries = np.stack((-keys[..., 1], keys[..., 0]), axis=-1) + base['query_noise']
    return base | dict(key_vectors=keys, query_vectors=queries)


def assemble_histories(base):
    inputs = {}
    for geometry in GEOMETRIES:
        observed = geometry_base(base, geometry)
        for history in HISTORIES:
            inputs[f'{geometry}_{history}'] = confirm_data.assemble(observed, history)[0]
    return inputs, confirm_data.Q


def rule_order(seed):
    start = seed % len(RULES)
    return RULES[start:] + RULES[:start]


def verify_paths(history, base, x, outputs):
    original = outputs['original'][0]
    replay, diagnostics, timing = outputs['replay']
    n = len(x)
    groups = base['group'] % 4
    first = np.array([row[3] for row in confirm_data.SCHEDULES])[groups]
    writes = np.array([row[1] for row in confirm_data.SCHEDULES])[groups]
    nonwrites = np.array([row[2] for row in confirm_data.SCHEDULES])[groups]
    rows = np.arange(n)
    onset = replay['read_onset']
    assert onset[rows, first].all() and onset.sum() == n
    assert np.array_equal(diagnostics['first_read_tick'], first)
    assert np.array_equal(diagnostics['logged_write_ticks'], writes[:, 2:])
    assert np.array_equal(diagnostics['logged_write_values'], x[rows[:, None], writes[:, 2:]])
    assert np.all(diagnostics['logged_nonwrite_count'] == 2)
    assert np.array_equal(diagnostics['logged_nonwrite_ticks'][:, :2], nonwrites)
    assert np.array_equal(diagnostics['logged_nonwrite_values'][:, :2], x[rows[:, None], nonwrites])
    assert np.all(diagnostics['logged_nonwrite_ticks'][:, 2:] == -1)
    assert not diagnostics['logged_nonwrite_values'][:, 2:].any()
    eviction_tick = writes[:, 2] if history == 'four' else writes[:, 3] if history != 'two' else np.full(n, T)
    expected_eviction = np.arange(T)[None, :] > eviction_tick[:, None]
    assert np.array_equal(replay['eviction_seen'], expected_eviction)
    mask = replay['reset_steps'][rows, first]
    assert np.array_equal(replay['reset_steps'], onset & expected_eviction & (replay['match_confidence'] >= 1./np.sqrt(2.)))
    assert np.array_equal(diagnostics['replay_steps'], first*mask)
    assert not diagnostics['replay_reference'][~mask].any() and not diagnostics['replay_resources'][~mask].any()
    assert not original['reset_steps'].any()
    for field in ('key_bank', 'value_bank', 'read_onset', 'eviction_seen', 'cue_scores', 'cue_weights', 'cue_read_vectors'):
        assert np.array_equal(replay[field], original[field]), (history, field)
    for field in ('counts', 'resources', 'cue_native_before', 'cue_native_after'):
        assert np.array_equal(replay[field][~mask], original[field][~mask]), (history, field)
    for i, tick in enumerate(first[:8]):
        for field in ('states', 'spikes', 'currents', 'injections', 'candidates', 'drive'):
            assert np.array_equal(replay[field][i, :tick], original[field][i, :tick]), (history, field, i)
    if history == 'two':
        assert not mask.any() and not timing['replay_batches']
        assert all(np.array_equal(replay[field], array) for field, array in original.items())


def verify_guard(history, base, x, outputs):
    original = outputs['original'][0]
    replay, logs, _ = outputs['replay']
    guarded, guard, timing = outputs['guarded']
    old_mask, new_mask = replay['reset_steps'], guarded['reset_steps']
    assert not np.any(new_mask & ~old_mask)
    assert np.array_equal(guarded['match_confidence'], replay['match_confidence'])
    assert np.array_equal(new_mask, old_mask & (replay['match_confidence'] > guard['evicted_cosine']))
    assert np.array_equal(guard['replay_steps'], np.where(new_mask.any(axis=1), logs['replay_steps'], 0))
    assert np.all(guard['evicted_cosine'][~old_mask] == -2.)
    assert np.array_equal(guard['evicted_comparisons'], old_mask*(WRITES[history]-2))
    assert guard['evicted_comparisons'].dtype == np.int64
    rows = np.arange(len(x))
    schedules = base['group'] % 4
    writes = np.array([row[1] for row in confirm_data.SCHEDULES])[schedules]
    lost = np.zeros((len(x), 2, 2))
    for slot, position in {'four': ((0, 0), (1, 1)), 'three0': ((1, 0),), 'three2': ((1, 1),), 'two': ()}[history]:
        lost[:, slot] = x[rows, writes[:, position], 2:4]
    assert np.array_equal(guard['evicted_keys'], lost), history
    for field in ('key_bank', 'value_bank', 'read_onset', 'eviction_seen', 'cue_scores', 'cue_weights', 'cue_read_vectors'):
        assert np.array_equal(guarded[field], replay[field]), (history, field)
    for field in ('logged_write_values', 'logged_write_ticks', 'logged_nonwrite_values',
                  'logged_nonwrite_ticks', 'logged_nonwrite_count', 'first_read_tick'):
        assert np.array_equal(guard[field], logs[field]), (history, field)
    kept = new_mask.any(axis=1)
    for field in ('counts', 'resources'):
        assert np.array_equal(guarded[field][kept], replay[field][kept]), (history, field, 'unchanged trigger')
        assert np.array_equal(guarded[field][~kept], original[field][~kept]), (history, field, 'original fallback')
    for field in ('cue_native_before', 'cue_native_after'):
        np.testing.assert_allclose(guarded[field][kept], replay[field][kept], atol=5e-12, rtol=0.)
        assert np.array_equal(guarded[field][kept, :, 20:], replay[field][kept, :, 20:])
        assert np.array_equal(guarded[field][~kept], original[field][~kept])
    assert np.array_equal(guard['replay_resources'][kept], logs['replay_resources'][kept])
    np.testing.assert_allclose(guard['replay_reference'][kept], logs['replay_reference'][kept], atol=5e-12, rtol=0.)
    assert np.array_equal(guard['replay_reference'][kept, 20:], logs['replay_reference'][kept, 20:])
    assert not guard['replay_reference'][~kept].any() and not guard['replay_resources'][~kept].any()
    for i in range(min(8, len(x))):
        comparator = replay if kept[i] else original
        assert np.array_equal(guarded['spikes'][i], comparator['spikes'][i])
    if history == 'two':
        assert not timing['guard_batches'] and not timing['replay_batches']
        assert all(np.array_equal(guarded[field], array) for field, array in original.items())


def timed_evaluate(x, w, q, rule):
    start = time.perf_counter()
    raw, diag, timing = guard_model.evaluate(x, w, q, rule)
    return raw, diag, timing | dict(evaluation_wall_seconds=time.perf_counter()-start)


def evaluate_cases(base, inputs, q, w, order):
    results = {}
    for group, x in inputs.items():
        _, history = group.split('_', 1)
        paths = {rule: timed_evaluate(x, w, q, rule) for rule in order}
        verify_paths(history, base, x, paths)
        verify_guard(history, base, x, paths)
        results.update({f'{group}_{rule}': value for rule, value in paths.items()})
    assert set(results) == set(CONDITIONS)
    return results


def first_read_match(x, diagnostics, w):
    rows = np.arange(len(x))
    query = x[rows, diagnostics['first_read_tick'], 2:4]
    keys = diagnostics['logged_write_values'][:, ::-1, 2:4]
    projected = query @ w[176:].reshape(2, 2)
    scores = learned_model.TEMPERATURE*(keys @ projected[..., None])[..., 0]
    numerators = np.exp(scores-scores.max(axis=1, keepdims=True))
    weights = numerators/numerators.sum(axis=1, keepdims=True)
    return dict(first_read_scores=scores, first_read_weights=weights)


def measure_single(seed, inputs, q, w, base, order, cases, start):
    paired_order = tuple(rule for rule in order if rule != 'original')
    records, steps, comparisons = [], 0, 0
    schedules = base['group'] % 4
    for geometry in GEOMETRIES:
        x = inputs[f'{geometry}_four']
        triggered = cases[f'{geometry}_four_replay'][1]['replay_steps'] > 0
        for schedule, (label, _, _, _) in enumerate(confirm_data.SCHEDULES):
            for sample in np.flatnonzero(triggered & (schedules == schedule))[:4]:
                guard(start)
                record = dict(seed=seed, geometry=geometry, sample=int(sample), schedule=label, order=list(paired_order))
                for rule in paired_order:
                    raw, diag, timing = guard_model.evaluate(x[sample:sample+1], w, q, rule)
                    batch, logs, _ = cases[f'{geometry}_four_{rule}']
                    assert np.array_equal(raw['counts'][0], batch['counts'][sample])
                    assert np.array_equal(raw['resources'][0], batch['resources'][sample])
                    assert np.array_equal(diag['replay_resources'][0], logs['replay_resources'][sample])
                    replay_steps = int(diag['replay_steps'][0])
                    assert replay_steps == logs['replay_steps'][sample] and timing['replay_batches'] == int(replay_steps > 0)
                    assert (timing['replay_wall_seconds'] > 0.) == (replay_steps > 0)
                    steps += replay_steps
                    if rule == 'replay':
                        record.update(replay_seconds=timing['replay_wall_seconds'], replay_steps=replay_steps)
                    else:
                        assert timing['guard_batches'] == 1 and timing['guard_wall_seconds'] > 0.
                        comparisons += int(diag['evicted_comparisons'].sum())
                        record.update(guarded_replay_seconds=timing['replay_wall_seconds'],
                                      guard_seconds=timing['guard_wall_seconds'], guarded_steps=replay_steps)
                records.append(record)
    return records, steps, 156*steps, comparisons


def evaluate_block(folder, seed, n, rows, records, ledger, start):
    guard(start)
    ident = f's{seed}_LEARN'
    w = np.load(ROOT/'weights'/f'{ident}.npy', allow_pickle=False)
    base = capacity_data.make_base(n, 4086020000+100*seed)
    inputs, q = assemble_histories(base)
    np.savez_compressed(folder / f'test_s{seed}.npz', **base)
    ledger['input_array_bytes'] += sum(array.nbytes for array in inputs.values())
    order = rule_order(seed)
    cases = evaluate_cases(base, inputs, q, w, order)
    match = {group: first_read_match(x, cases[f'{group}_replay'][1], w) for group, x in inputs.items()}
    position = base['selected']
    for condition, (raw, diagnostics, timing) in cases.items():
        geometry, history, rule = condition.split('_', 2)
        first = match[f'{geometry}_{history}']
        assert all(np.isfinite(array).all() for array in (*raw.values(), *diagnostics.values(), *first.values()))
        scores = capacity_metrics.score(raw | dict(cue_scores=first['first_read_scores'][:, None, :],
                                                  cue_weights=first['first_read_weights'][:, None, :]), base)
        evicted_ba = evicted_coverage = missing_ba = missing_coverage = None
        if history in EVICTED:
            evicted, missing = EVICTED[history], 1-EVICTED[history]
            evicted_ba, missing_ba = (scores['ba_by_position'][i] for i in (evicted, missing))
            evicted_coverage, missing_coverage = (scores['coverage_by_position'][i] for i in (evicted, missing))
            assert missing_ba == .5*missing_coverage
        elif history == 'two':
            assert scores['older_ba'] == .5*scores['older_coverage']
        onset, eviction = raw['read_onset'], raw['eviction_seen']
        reset_count, onset_count = int(raw['reset_steps'].sum()), int(onset.sum())
        eligible_count = int((onset & eviction).sum()) if rule != 'original' else 0
        reset = raw['reset_steps'].any(axis=1)
        ledger['reset_count'] += reset_count
        ledger['cosine_onsets'] += eligible_count
        if diagnostics:
            ledger['replay_sample_steps'] += int(diagnostics['replay_steps'].sum())
            ledger['replay_examples'] += int(np.count_nonzero(diagnostics['replay_steps']))
        if rule == 'guarded':
            ledger['evicted_key_comparisons'] += int(diagnostics['evicted_comparisons'].sum())
            ledger['evicted_key_log_writes'] += n*(WRITES[history]-2)
        for name in ('replay_wall_seconds', 'replay_batches', 'guard_wall_seconds', 'guard_batches', 'evaluation_wall_seconds'):
            ledger[name] += timing[name]
        ledger['replay_scratch_peak_bytes'] = max(ledger['replay_scratch_peak_bytes'], timing['replay_scratch_peak_bytes'])
        np.savez_compressed(folder / f'{ident}_{condition}.npz', **raw, **diagnostics, **first)
        ledger['retained_raw_array_bytes'] += sum(array.nbytes for array in (*raw.values(), *diagnostics.values(), *first.values()))
        rows.append(dict(id=ident, seed=seed, arm='LEARN', condition=condition, geometry=geometry, history=history, rule=rule,
                         T=T, q=q, slots=2, **scores, mean_resources=raw['resources'][:, :4].mean(axis=0).tolist(),
                         evicted_ba=evicted_ba, evicted_coverage=evicted_coverage,
                         missing_ba=missing_ba, missing_coverage=missing_coverage,
                         reset_count=reset_count, onset_count=onset_count, eviction_fraction=float(eviction.any(axis=1).mean()),
                         reset_fraction=float(reset.mean()), older_reset_fraction=float(reset[position<2].mean()),
                         recent_reset_fraction=float(reset[position>=2].mean()),
                         reset_by_position=[float(reset[position==i].mean()) for i in range(4)],
                         replay_resources_mean=diagnostics['replay_resources'].mean(axis=0).tolist() if diagnostics else [0.]*4,
                         **timing, cost_per_episode=cost(history, rule, reset_count, eligible_count, diagnostics, n)))
        guard(start, disk=True)
    subset, steps, macs, comparisons = measure_single(seed, inputs, q, w, base, order, cases, start)
    records.extend(subset)
    ledger['microbench_replay_steps'] += steps
    ledger['microbench_replay_native_macs'] += macs
    ledger['microbench_evicted_key_comparisons'] += comparisons
    dump('evaluate/ROWS.json', rows)
    dump('MICROBENCH.json', records)
    print(f'Evaluated new {ident}: {len(rows)}/{EVALUATIONS}, isolated timings {len(records)}', flush=True)


def science():
    assert all(os.environ.get(key) == '1' for key in THREAD_KEYS)
    assert not any((ROOT/name).exists() for name in ('RUN_STATE.json', 'evaluate', 'MICROBENCH.json'))
    verify_lock('FREEZE.json')
    frozen = read('FREEZE.json')
    assert frozen['blocks'] == BLOCKS and frozen['evaluations'] == EVALUATIONS
    n = frozen['examples_per_block']
    assert n in (512, 256)
    start, cpu = time.perf_counter(), time.process_time()
    rows, records = [], []
    ledger = dict(input_array_bytes=0, retained_raw_array_bytes=0, reset_count=0, cosine_onsets=0,
                  replay_sample_steps=0, replay_examples=0, replay_wall_seconds=0., replay_batches=0,
                  replay_scratch_peak_bytes=0, microbench_replay_steps=0, microbench_replay_native_macs=0,
                  guard_wall_seconds=0., guard_batches=0, evaluation_wall_seconds=0., evicted_key_comparisons=0,
                  evicted_key_log_writes=0, microbench_evicted_key_comparisons=0)
    dump('RUN_STATE.json', dict(status='running', utc=utc()))
    try:
        folder = ROOT/'evaluate'
        folder.mkdir()
        dump('ENVIRONMENT.json', dict(machine=platform.node(), python=sys.version, numpy=np.__version__,
             threads={key: os.environ[key] for key in THREAD_KEYS}))
        for seed in range(BLOCKS):
            evaluate_block(folder, seed, n, rows, records, ledger, start)
        assert len(rows) == EVALUATIONS
        guard(start, disk=True)
        steps = EVALUATIONS*n*T
        dump('SCIENCE_SUMMARY.json', dict(status='COMPLETE', utc=utc(), blocks=BLOCKS, examples_per_block=n,
             fits=0, updates=0, evaluations=len(rows), forward_sample_steps=steps,
             dense_forward_macs=156*steps, matcher_macs=8*steps, wall_seconds=time.perf_counter()-start,
             cpu_seconds=time.process_time()-cpu, cap_seconds=CAP, peak_working_set_bytes=peak_memory(),
             artifact_bytes_at_science_end=artifact_size(), reset_scalar_writes=26*ledger['reset_count'],
             replay_native_macs=156*ledger['replay_sample_steps'], microbench_records=len(records),
             microbench_evaluations=2*len(records), microbench_forward_sample_steps=2*len(records)*T, **ledger))
        dump('RUN_STATE.json', dict(status='complete', utc=utc(), evaluations=len(rows)))
    except Exception as error:
        failure = dict(status='stopped', stage='science', utc=utc(), reason=str(error), exception=type(error).__name__,
                       evaluations=len(rows), wall_seconds=time.perf_counter()-start)
        dump('SCIENCE_FAILURE.json', failure)
        dump('RUN_STATE.json', failure)
        raise


if __name__ == '__main__':
    {'freeze': freeze, 'science': science}[sys.argv[1]]()
