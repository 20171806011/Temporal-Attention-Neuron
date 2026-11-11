"""Freeze and evaluate the saved bilinear matcher checkpoints with causal event replay."""

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
import event_model


ROOT = Path(__file__).resolve().parent
HISTORIES = ('four', 'three0', 'three2', 'two')
RULES = ('original', 'replay', 'dense')
CONDITIONS = tuple(f'{history}_{rule}' for history in HISTORIES for rule in RULES)
WRITES = {'four': 4, 'three0': 3, 'three2': 3, 'two': 2}
EVICTED = {'three0': 0, 'three2': 1}
BLOCKS = 12
EVALUATIONS = len(CONDITIONS)*BLOCKS
T = 36
REPLAY_FIELDS = ('logged_write_values', 'logged_write_ticks', 'logged_nonwrite_values',
                 'logged_nonwrite_ticks', 'logged_nonwrite_count', 'replay_steps',
                 'replay_resources', 'first_read_tick', 'replay_reference')
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


def cost(history, rule, reset_count, eligible_count, diagnostics, n):
    slots, steps = 2, T
    cosine = eligible_count/n
    hard = matched = 0
    writes = WRITES[history]
    replay = rule == 'replay'
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
                replay_native_steps_mean=extra_steps, replay_native_macs_mean=156*extra_steps)


def freeze():
    assert not any((ROOT / name).exists() for name in ('FREEZE.json', 'RUN_STATE.json', 'evaluate'))
    assert read('PRECHECK.json')['status'] == 'PASS'
    timing = read('TIMING.json')
    assert timing['status'] == 'PASS' and timing['blocks'] == BLOCKS
    assert timing['scientific_data_generated'] is False
    selected = next(row['examples'] for row in timing['candidates']
                    if row['science_forecast_seconds'] < CAP and row['audit_forecast_seconds'] < 300)
    assert selected == timing['examples_per_block'] and selected in (512, 256)
    assert len(read('SOURCE_LOCK.json')['files']) == 13+BLOCKS
    verify_lock('SOURCE_LOCK.json')
    for name in ('independent_event.py', 'event_model.py', 'independent_capacity_data.py', 'audit.py', 'PROTOCOL.md', 'REPRODUCE.md', 'summarize.py', 'report.py'):
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
    three0, three2 = four.copy(), four.copy()
    three0[:, 2], three2[:, 0] = 0., 0.
    return dict(four=four, three0=three0, three2=three2, two=two), q


def verify_paths(history, x, cases, diagnostics):
    n = len(x)
    original, replay, dense = (cases[rule] for rule in RULES)
    assert set(diagnostics) == set(REPLAY_FIELDS)
    expected_eviction = np.broadcast_to(np.arange(T) > {'four': 4, 'three0': 8, 'three2': 8, 'two': T}[history], (n, T))
    onset = original['read_onset']
    assert onset.dtype == np.bool_ and onset[:, 25].all() and onset.sum() == n
    assert not original['reset_steps'].any()
    assert np.array_equal(replay['reset_steps'], dense['reset_steps'])
    mask = replay['reset_steps'][:, 25]
    assert np.array_equal(replay['reset_steps'], onset & expected_eviction & (replay['match_confidence'] >= 1./np.sqrt(2.)))
    assert np.array_equal(diagnostics['replay_steps'], mask*25)
    for field in ('logged_write_ticks', 'logged_nonwrite_ticks', 'first_read_tick', 'replay_steps', 'logged_nonwrite_count'):
        assert diagnostics[field].dtype == np.int64, field
    assert np.all(diagnostics['first_read_tick'] == 25)
    assert np.array_equal(diagnostics['logged_write_ticks'], np.broadcast_to((4, 8), (n, 2)))
    assert np.array_equal(diagnostics['logged_write_values'], x[:, (4, 8)])
    assert np.all(diagnostics['logged_nonwrite_count'] == 2)
    assert np.array_equal(diagnostics['logged_nonwrite_ticks'][:, :2], np.broadcast_to((5, 7), (n, 2)))
    assert np.all(diagnostics['logged_nonwrite_ticks'][:, 2:] == -1)
    assert np.array_equal(diagnostics['logged_nonwrite_values'][:, :2], x[:, (5, 7)])
    assert not diagnostics['logged_nonwrite_values'][:, 2:].any()
    assert not diagnostics['replay_reference'][~mask].any()
    assert not diagnostics['replay_resources'][~mask].any()
    assert np.array_equal(diagnostics['replay_reference'][mask], replay['cue_native_after'][mask, 0])
    for rule, raw in cases.items():
        assert np.array_equal(raw['eviction_seen'], expected_eviction), (history, rule, 'eviction')
        for field in ('key_bank', 'value_bank', 'read_onset', 'cue_scores', 'cue_weights', 'cue_read_vectors'):
            assert np.array_equal(raw[field], original[field]), (history, rule, field)
        for field in ('states', 'spikes', 'currents', 'injections', 'candidates', 'drive'):
            assert np.array_equal(raw[field][:, :25], original[field][:, :25]), (history, rule, field)
        for field in ('counts', 'resources', 'cue_native_before', 'cue_native_after'):
            assert np.array_equal(raw[field][~mask], original[field][~mask]), (history, rule, field, 'no replacement')
    for field in replay:
        if field in ('counts', 'resources', 'spikes', 'read_onset', 'reset_steps', 'eviction_seen'):
            assert np.array_equal(replay[field], dense[field]), (history, field)
        else:
            np.testing.assert_allclose(replay[field], dense[field], atol=5e-12, rtol=0, err_msg=f'{history} {field}')
    for field in ('cue_native_before', 'cue_native_after'):
        assert np.array_equal(replay[field][..., 20:], dense[field][..., 20:]), (history, field, 'p')
    if history == 'two':
        assert not mask.any()
        for rule in ('replay', 'dense'):
            assert all(np.array_equal(cases[rule][field], array) for field, array in original.items()), rule


def evaluate_cases(inputs, q, w):
    output = {}
    two = event_model.evaluate(inputs['two'], w, q, 'original')
    reference = two[0]['cue_native_before'][:, 0]
    for history, x in inputs.items():
        paths = {
            'original': two if history == 'two' else event_model.evaluate(x, w, q, 'original'),
            'replay': event_model.evaluate(x, w, q, 'replay'),
            'dense': event_model.evaluate(x, w, q, 'dense', reference),
        }
        verify_paths(history, x, {name: result[0] for name, result in paths.items()}, paths['replay'][1])
        output.update({f'{history}_{rule}': value for rule, value in paths.items()})
    assert set(output) == set(CONDITIONS)
    return output


def evaluate_block(folder, seed, n, rows, ledger, start):
    guard(start)
    ident = f's{seed}_LEARN'
    w = np.load(ROOT / 'weights' / (ident+'.npy'), allow_pickle=False)
    base = capacity_data.make_base(n, 4083020000+100*seed)
    inputs, q = assemble_histories(base)
    np.savez_compressed(folder / f'test_s{seed}.npz', **base)
    ledger['input_array_bytes'] += sum(array.nbytes for array in inputs.values())
    cases = evaluate_cases(inputs, q, w)
    position = base['selected']
    for condition, (raw, diagnostics, timing) in cases.items():
        history, rule = condition.split('_', 1)
        assert all(np.isfinite(array).all() for array in (*raw.values(), *diagnostics.values())), (ident, condition)
        scores = capacity_metrics.score(raw, base)
        evicted_ba = evicted_coverage = missing_ba = missing_coverage = None
        if history in EVICTED:
            evicted, missing = EVICTED[history], 1-EVICTED[history]
            evicted_ba, missing_ba = (scores['ba_by_position'][i] for i in (evicted, missing))
            evicted_coverage, missing_coverage = (scores['coverage_by_position'][i] for i in (evicted, missing))
            assert missing_ba == .5*missing_coverage
        elif history == 'two':
            assert scores['older_ba'] == .5*scores['older_coverage']
        onset, eviction = raw['read_onset'], raw['eviction_seen']
        reset_count = int(raw['reset_steps'].sum())
        onset_count = int(onset.sum())
        eligible_count = int((onset & eviction).sum()) if rule != 'original' else 0
        reset = raw['reset_steps'].any(axis=1)
        ledger['reset_count'] += reset_count
        ledger['cosine_onsets'] += eligible_count
        if diagnostics:
            ledger['replay_sample_steps'] += int(diagnostics['replay_steps'].sum())
            ledger['replay_examples'] += int(np.count_nonzero(diagnostics['replay_steps']))
        ledger['replay_wall_seconds'] += timing['replay_wall_seconds']
        ledger['replay_batches'] += timing['replay_batches']
        ledger['replay_scratch_peak_bytes'] = max(ledger['replay_scratch_peak_bytes'], timing['replay_scratch_peak_bytes'])
        np.savez_compressed(folder / f'{ident}_{condition}.npz', **raw, **diagnostics)
        ledger['retained_raw_array_bytes'] += sum(array.nbytes for array in (*raw.values(), *diagnostics.values()))
        rows.append(dict(id=ident, seed=seed, arm='LEARN', condition=condition, history=history, rule=rule,
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
    dump('evaluate/ROWS.json', rows)
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
    rows = []
    ledger = dict(input_array_bytes=0, retained_raw_array_bytes=0, reset_count=0, cosine_onsets=0,
                  replay_sample_steps=0, replay_examples=0, replay_wall_seconds=0., replay_batches=0,
                  replay_scratch_peak_bytes=0)
    dump('RUN_STATE.json', dict(status='running', utc=utc()))
    try:
        folder = ROOT / 'evaluate'
        folder.mkdir()
        dump('ENVIRONMENT.json', dict(machine=platform.node(), python=sys.version, numpy=np.__version__,
             threads={key: os.environ[key] for key in THREAD_KEYS}))
        for seed in range(BLOCKS):
            evaluate_block(folder, seed, n, rows, ledger, start)
        assert len(rows) == EVALUATIONS
        guard(start, disk=True)
        steps = EVALUATIONS*n*T
        dump('SCIENCE_SUMMARY.json', dict(status='COMPLETE', utc=utc(), blocks=BLOCKS, examples_per_block=n,
             fits=0, updates=0, evaluations=len(rows), forward_sample_steps=steps, dense_forward_macs=156*steps, matcher_macs=8*steps,
             wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu, cap_seconds=CAP,
             peak_working_set_bytes=peak_memory(), artifact_bytes_at_science_end=artifact_size(),
             reset_scalar_writes=26*ledger['reset_count'], replay_native_macs=156*ledger['replay_sample_steps'],
             dense_diagnostic_evaluations=4*BLOCKS, dense_diagnostic_sample_steps=4*BLOCKS*n*T, **ledger))
        dump('RUN_STATE.json', dict(status='complete', utc=utc(), evaluations=len(rows)))
    except Exception as error:
        failure = dict(status='stopped', utc=utc(), reason=str(error), exception=type(error).__name__,
                       evaluations=len(rows), wall_seconds=time.perf_counter()-start)
        dump('SCIENCE_FAILURE.json', failure)
        dump('RUN_STATE.json', failure)
        raise


if __name__ == '__main__':
    {'freeze': freeze, 'science': science}[sys.argv[1]]()
