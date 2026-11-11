"""Verify fixed training, guarded replay, and the two-trial feasibility gate."""

from datetime import datetime, timezone
import json
import subprocess
import time

import numpy as np

import capacity_data
import independent_capacity_data
import independent_confirm
import independent_event
import independent_guard
import independent_guard_confirm
import independent_learned
import independent_learned_data
import independent_warm
import learned_data
import population
import trainer
from run import BLOCKS, CAP, ROOT, assemble_histories, dump, evaluate_cases, measure_single, peak_memory, rule_order


SEED_PAIRS = ((4099903101, 4099903102), (4099903103, 4099903104))
TEST_SEED = 4099903105
FIXTURE_UPDATES = 24
FORMAL_UPDATES = 600
TRAIN_CAP, AUDIT_CAP = 600, 300
FIXED_GAP_SECONDS = 30.
LOAD_SNAPSHOT_LEAD_SECONDS = 6.
GEOMETRIES, HISTORIES, RULES = independent_guard.GEOMETRIES, independent_guard.HISTORIES, independent_guard.RULES
EPSILON, FD_TOLERANCE = 1e-5, 2e-7
TOLERANCE = 5e-12
EXACT = ('counts', 'spikes', 'resources', 'read_onset', 'reset_steps', 'eviction_seen',
         'logged_write_values', 'logged_write_ticks', 'logged_nonwrite_values', 'logged_nonwrite_ticks',
         'logged_nonwrite_count', 'first_read_tick', 'replay_steps', 'replay_resources',
         'evicted_keys', 'evicted_comparisons')


def check_train_fixture(data_seed, init_seed):
    base = learned_data.make_base(128, data_seed)
    independent_base = independent_learned_data.make_base(128, data_seed)
    assert base.keys() == independent_base.keys()
    assert all(np.array_equal(values, independent_base[name]) for name, values in base.items())
    x, q = learned_data.assemble(base)
    ix, iq = independent_learned_data.assemble(independent_base)
    assert q == iq == 28 and np.array_equal(x, ix)
    y = base['y']
    initial = np.r_[population.initialize(init_seed, 6), np.zeros(4)]
    np.testing.assert_array_equal(initial, independent_confirm.initialize(init_seed))
    archive = trainer.fit_one(x, y, q, initial, FIXTURE_UPDATES, lambda: None)
    times = archive['step_seconds']
    assert times.shape == (FIXTURE_UPDATES,) and np.all(times > 0.)

    tick = time.perf_counter()
    checked = independent_confirm.verify_training_archive(ix, y, iq, archive)
    audit_seconds = time.perf_counter()-tick
    checked = {name: value for name, value in checked.items() if name not in ('theta', 'final_counts')}
    rng = np.random.default_rng(init_seed)
    directions = rng.normal(size=(12, initial.size))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    directions = np.vstack((directions, np.eye(initial.size)[176:]))
    anchor = independent_learned.anchor(ix, initial, 'LEARN', iq)
    np.testing.assert_array_equal(archive['counts'][0], np.sum([step['os'] for step in anchor[iq:]], axis=0))
    initial_objective = independent_learned.objective(ix, y, initial, 'LEARN', iq, anchor)
    assert abs(initial_objective-archive['history'][0, 0]) <= TOLERANCE
    gradient = archive['raw_gradients'][0]
    maximum = 0.
    for number, direction in enumerate(directions):
        high = independent_learned.objective(ix, y, initial+EPSILON*direction, 'LEARN', iq, anchor)
        low = independent_learned.objective(ix, y, initial-EPSILON*direction, 'LEARN', iq, anchor)
        error = abs((high-low)/(2*EPSILON)-float(gradient @ direction))
        assert error <= FD_TOLERANCE, (data_seed, init_seed, number, error)
        maximum = max(maximum, error)
    matcher_norm = float(np.linalg.norm(gradient[176:]))
    tail_median = float(np.median(times[-12:]))
    return archive['theta'], dict(data_seed=data_seed, init_seed=init_seed, updates=FIXTURE_UPDATES,
                                  step_seconds=times.tolist(), tail12_median_seconds=tail_median,
                                  training_audit_seconds=audit_seconds, independent_checks=checked,
                                  gradient_directions=len(directions), max_direction_error=maximum,
                                  initial_matcher_gradient_l2=matcher_norm)


def compare(actual, expected, context):
    assert actual.keys() == expected.keys(), context
    maximum = 0.
    checks = 0
    for name, value in actual.items():
        wanted = expected[name]
        assert value.shape == wanted.shape, (context, name)
        if name in EXACT:
            np.testing.assert_array_equal(value, wanted, err_msg=f'{context} {name}')
        else:
            error = float(np.max(np.abs(value-wanted)))
            assert error <= TOLERANCE, (context, name, error)
            maximum = max(maximum, error)
        if name in ('cue_native_before', 'cue_native_after', 'replay_reference'):
            np.testing.assert_array_equal(value[..., 20:], wanted[..., 20:], err_msg=f'{context} {name} pulses')
            checks += 1
        checks += 1
    return checks, maximum


def check_block(n, w):
    tick = time.perf_counter()
    base = capacity_data.make_base(n, TEST_SEED)
    inputs, q = assemble_histories(base)
    order = rule_order(0)
    cases = evaluate_cases(base, inputs, q, w, order)
    isolated, isolated_steps, isolated_macs, isolated_compares = measure_single(0, inputs, q, w, base, order, cases, tick)
    science_seconds = time.perf_counter()-tick

    tick = time.perf_counter()
    ibase = independent_capacity_data.make_base(n, TEST_SEED)
    assert base.keys() == ibase.keys()
    assert all(np.array_equal(values, ibase[name]) for name, values in base.items())
    assert set(cases) == {f'{geometry}_{history}_{rule}' for geometry in GEOMETRIES for history in HISTORIES for rule in RULES}
    checks, maximum = 0, 0.
    triggered, vetoed, compared = 0, 0, 0
    for geometry in GEOMETRIES:
        expected_base = independent_guard_confirm.geometry(ibase, geometry)
        for history in HISTORIES:
            prefix = f'{geometry}_{history}'
            x, iq = independent_guard.assemble(expected_base, history)
            assert q == iq == 28 and np.array_equal(inputs[prefix], x), prefix
            original = independent_warm.replay(x, w, 'LEARN', q, 'original')
            replay = independent_event.replay(x, w, q)
            guarded = independent_guard.apply_guard(x, w, replay, original)
            for rule, expected in (('original', original), ('replay', replay), ('guarded', guarded)):
                raw, diag, _ = cases[f'{prefix}_{rule}']
                count, residual = compare({**raw, **diag}, expected, f'{prefix}_{rule}')
                checks += count
                maximum = max(maximum, residual)
            original, _, _ = cases[f'{prefix}_original']
            replay, replay_diag, replay_timing = cases[f'{prefix}_replay']
            guarded, guard_diag, guard_timing = cases[f'{prefix}_guarded']
            old_mask, new_mask = replay['reset_steps'], guarded['reset_steps']
            assert not np.any(new_mask & ~old_mask)
            assert np.array_equal(guarded['match_confidence'], replay['match_confidence'])
            assert np.all(guard_diag['evicted_cosine'][~old_mask] == -2.)
            assert np.all(guard_diag['evicted_comparisons'][~old_mask] == 0)
            assert np.array_equal(new_mask, old_mask & (replay['match_confidence'] > guard_diag['evicted_cosine']))
            assert guard_timing['guard_batches'] == replay_timing['replay_batches']
            rejected = (old_mask & ~new_mask).any(axis=1)
            unchanged = ~rejected
            for field in ('counts', 'resources', 'spikes'):
                np.testing.assert_array_equal(guarded[field][rejected[:len(guarded[field])]],
                                              original[field][rejected[:len(guarded[field])]])
                np.testing.assert_array_equal(guarded[field][unchanged[:len(guarded[field])]],
                                              replay[field][unchanged[:len(guarded[field])]])
            for field in ('replay_steps', 'replay_resources'):
                np.testing.assert_array_equal(guard_diag[field][unchanged], replay_diag[field][unchanged])
                assert not guard_diag[field][rejected].any()
            if history == 'two':
                assert not guard_diag['evicted_keys'].any()
                for field in original:
                    np.testing.assert_array_equal(guarded[field], original[field], err_msg=f'{prefix} {field}')
                    np.testing.assert_array_equal(replay[field], original[field], err_msg=f'{prefix} {field}')
            triggered += int(old_mask.sum())
            vetoed += int(rejected.sum())
            compared += int(guard_diag['evicted_comparisons'].sum())

    audit_seconds = time.perf_counter()-tick
    assert peak_memory() < 1024**3
    return dict(examples=n, science_seconds=science_seconds, audit_seconds=audit_seconds,
                checks=checks, max_numeric_error=maximum, paths_verified=len(cases), old_triggers=triggered,
                vetoes=vetoed, evicted_comparisons=compared, isolated_records=len(isolated),
                isolated_replay_steps=isolated_steps, isolated_replay_macs=isolated_macs,
                isolated_evicted_comparisons=isolated_compares)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _processor_load():
    snapshot = dict(utc=_utc(), status='unavailable', percentages=None)
    command = ('$ErrorActionPreference = "Stop"; '
               '$loads = @(Get-CimInstance -ClassName Win32_Processor | '
               'Select-Object -ExpandProperty LoadPercentage); '
               'ConvertTo-Json -InputObject $loads -Compress')
    try:
        result = subprocess.run(['powershell.exe', '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', command],
                                capture_output=True, text=True, check=True, timeout=5,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        loads = json.loads(result.stdout)
        if not isinstance(loads, list) or not loads:
            raise ValueError('Win32_Processor returned no processor records')
        snapshot.update(status='ok', percentages=loads)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        snapshot['error'] = f'{type(error).__name__}: {error}'
    return snapshot


def measure_candidate(candidate, w, training_audit_forecast):
    trials = candidate['trials']
    previous_finish = None
    for number in (1, 2):
        if previous_finish is not None:
            before_snapshot = previous_finish+FIXED_GAP_SECONDS-LOAD_SNAPSHOT_LEAD_SECONDS
            delay = before_snapshot-time.monotonic()
            if delay > 0.:
                time.sleep(delay)
        before = _processor_load()
        if previous_finish is not None:
            # Read load outside the timed call, then wait for the fixed monotonic deadline.
            remaining = previous_finish+FIXED_GAP_SECONDS-time.monotonic()
            if remaining > 0.:
                time.sleep(remaining)
        utc_start = _utc()
        start = time.monotonic()
        trial = dict(trial=number, utc_start=utc_start, monotonic_start_seconds=start, load_before=before,
                     gap_from_previous_seconds=None if previous_finish is None else start-previous_finish)
        trials.append(trial)
        try:
            trial.update(check_block(candidate['examples'], w))
        finally:
            previous_finish = time.monotonic()
            trial.update(utc_finish=_utc(), monotonic_finish_seconds=previous_finish,
                         load_after=_processor_load())
    science_seconds = max(trial['science_seconds'] for trial in trials)
    audit_seconds = max(trial['audit_seconds'] for trial in trials)
    candidate.update(science_seconds=science_seconds, audit_seconds=audit_seconds,
                     science_forecast_seconds=3*BLOCKS*science_seconds+30,
                     audit_forecast_seconds=3*BLOCKS*audit_seconds+30+training_audit_forecast)


def main():
    assert not any((ROOT / name).exists() for name in ('PRECHECK.json', 'TIMING.json', 'FREEZE.json',
                                                       'RUN_STATE.json', 'train', 'evaluate'))
    started = time.perf_counter()
    candidates, training = [], []
    chosen = None
    try:
        weights = []
        for data_seed, init_seed in SEED_PAIRS:
            w, result = check_train_fixture(data_seed, init_seed)
            weights.append(w)
            training.append(result)
        assert max(fixture['initial_matcher_gradient_l2'] for fixture in training) > 1e-8
        median = max(fixture['tail12_median_seconds'] for fixture in training)
        train_forecast = 1.5*BLOCKS*FORMAL_UPDATES*median+30
        audit_seconds = max(fixture['training_audit_seconds'] for fixture in training)
        training_audit_forecast = BLOCKS*FORMAL_UPDATES/FIXTURE_UPDATES*audit_seconds
        if train_forecast < TRAIN_CAP:
            for n in (512, 256):
                candidate = dict(examples=n, fixed_gap_seconds=FIXED_GAP_SECONDS, trials=[])
                candidates.append(candidate)
                measure_candidate(candidate, weights[0], training_audit_forecast)
                if candidate['science_forecast_seconds'] < CAP and candidate['audit_forecast_seconds'] < AUDIT_CAP:
                    chosen = n
                    break
    except Exception as error:
        attempt = 1+len(list(ROOT.glob('PRECHECK_FAILURE_*.json')))
        dump(f'PRECHECK_FAILURE_{attempt}.json', dict(status='FAIL', reason=str(error),
             exception=type(error).__name__, scientific_data_generated=False,
             train_fixtures=training, test_seed=TEST_SEED, candidates=candidates))
        raise

    status = 'PASS' if chosen else 'INFEASIBLE'
    dump('PRECHECK.json', dict(status=status, scientific_data_generated=False, train_fixtures=training,
         test_seed=TEST_SEED, test_fixture_candidates=candidates, peak_working_set_bytes=peak_memory(),
         wall_seconds=time.perf_counter()-started))
    dump('TIMING.json', dict(status=status, blocks=BLOCKS, examples_per_block=chosen,
         candidates=candidates, training_forecast_seconds=train_forecast,
         training_audit_forecast_seconds=training_audit_forecast, scientific_data_generated=False))
    if chosen is None:
        raise SystemExit('No allowed evicted-key confirmation fixture fits all budgets')
    print(f'PRECHECK PASS: {chosen} examples; train {train_forecast:.3f} s forecast; '
          f'test {candidates[-1]["science_forecast_seconds"]:.3f} s; '
          f'audit {candidates[-1]["audit_forecast_seconds"]:.3f} s', flush=True)


if __name__ == '__main__':
    main()
