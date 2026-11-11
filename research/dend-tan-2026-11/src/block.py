"""Execute and archive one full shared-replay budget fixture block."""

import json
from pathlib import Path
import time
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np

import capacity_data
import capacity_metrics
import confirm_data
import guard_model
import learned_model
import population
import shared_model
import warm_model


DATA_SEED, WEIGHT_SEED = 4099903301, 4099903302
GEOMETRIES = ('orthogonal', 'near')
HISTORIES = ('four', 'three0', 'three2', 'two')
RULES = ('original', 'replay', 'guarded')
EVICTED = {'four': (0, 1), 'three0': (0,), 'three2': (1,), 'two': ()}
MISSING = {'four': (), 'three0': (1,), 'three2': (0,), 'two': (0, 1)}
TRACE = frozenset((*warm_model.TRACE_FIELDS, 'spikes', 'candidates'))
EXACT = frozenset(('counts', 'spikes', 'resources', 'read_onset', 'reset_steps', 'eviction_seen',
                   'logged_write_values', 'logged_write_ticks', 'logged_nonwrite_values', 'logged_nonwrite_ticks',
                   'logged_nonwrite_count', 'first_read_tick', 'replay_steps', 'replay_resources',
                   'evicted_keys', 'evicted_comparisons'))
PULSES = ('cue_native_before', 'cue_native_after', 'replay_reference')
TOLERANCE = 5e-12


def _save_npz(path, arrays, compression=ZIP_DEFLATED):
    level = 1 if compression == ZIP_DEFLATED else None
    with ZipFile(path, 'w', compression=compression, compresslevel=level) as archive:
        for name, values in arrays.items():
            with archive.open(name+'.npy', 'w', force_zip64=True) as member:
                np.lib.format.write_array(member, values, allow_pickle=False)


def _dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def build_inputs(base):
    """Assemble all paired geometry and history inputs from a supplied base."""
    angle = base['angle'][:, None]+np.deg2rad([0., 40., 180., 230.])
    keys = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    queries = np.stack((-keys[..., 1], keys[..., 0]), axis=-1)+base['query_noise']
    geometries = dict(orthogonal=base, near=base | dict(key_vectors=keys, query_vectors=queries))
    inputs = {}
    for geometry, observed in geometries.items():
        for history in HISTORIES:
            x, q = confirm_data.assemble(observed, history)
            assert q == confirm_data.Q
            inputs[f'{geometry}_{history}'] = x
    return inputs


def _fixture(n):
    base = capacity_data.make_base(n, DATA_SEED)
    inputs = build_inputs(base)
    w = np.r_[population.initialize(WEIGHT_SEED, 6), learned_model.ORACLE_MATRIX.ravel()]
    return base, inputs, w


def _merge(packets):
    return {rule: {name: (value if name in TRACE else np.concatenate([packet[rule][name] for packet in packets]))
                   for name, value in packets[0][rule].items()} for rule in RULES}


def _first_read(x, replay, w):
    rows = np.arange(len(x))
    query = x[rows, replay['first_read_tick'], 2:4]
    keys = replay['logged_write_values'][:, ::-1, 2:4]
    projected = query @ w[176:].reshape(2, 2)
    scores = learned_model.TEMPERATURE*(keys @ projected[..., None])[..., 0]
    exp = np.exp(scores-scores.max(axis=1, keepdims=True))
    return dict(first_read_scores=scores, first_read_weights=exp/exp.sum(axis=1, keepdims=True))


def _verify_paths(history, x, base, cases):
    original, replay, guarded = (cases[rule] for rule in RULES)
    for rule, values in cases.items():
        for field, array in values.items():
            assert len(array) == (8 if field in TRACE else len(x)), (history, rule, field, array.shape)
            assert np.isfinite(array).all(), (history, rule, field, 'nonfinite')
    rows = np.arange(len(x))
    schedules = base['group'] % len(confirm_data.SCHEDULES)
    first = np.array([entry[3] for entry in confirm_data.SCHEDULES])[schedules]
    writes = np.array([entry[1] for entry in confirm_data.SCHEDULES])[schedules]
    nonwrites = np.array([entry[2] for entry in confirm_data.SCHEDULES])[schedules]
    onset = replay['read_onset']
    assert onset[rows, first].all() and onset.sum() == len(x)
    assert np.array_equal(replay['first_read_tick'], first)
    assert np.array_equal(replay['logged_write_ticks'], writes[:, 2:])
    assert np.array_equal(replay['logged_write_values'], x[rows[:, None], writes[:, 2:]])
    assert np.all(replay['logged_nonwrite_count'] == 2)
    assert np.array_equal(replay['logged_nonwrite_ticks'][:, :2], nonwrites)
    assert np.array_equal(replay['logged_nonwrite_values'][:, :2], x[rows[:, None], nonwrites])
    assert np.all(replay['logged_nonwrite_ticks'][:, 2:] == -1) and not replay['logged_nonwrite_values'][:, 2:].any()
    eviction_tick = writes[:, 2] if history == 'four' else writes[:, 3] if history != 'two' else np.full(len(x), 36)
    expected_eviction = np.arange(confirm_data.T)[None, :] > eviction_tick[:, None]
    assert np.array_equal(replay['eviction_seen'], expected_eviction)
    old, new = replay['reset_steps'], guarded['reset_steps']
    assert np.array_equal(old, onset & expected_eviction & (replay['match_confidence'] >= warm_model.COSINE_THRESHOLD))
    assert np.array_equal(new, old & (replay['match_confidence'] > guarded['evicted_cosine']))
    assert not original['reset_steps'].any()
    assert np.array_equal(guarded['match_confidence'], replay['match_confidence'])
    assert np.all(guarded['evicted_cosine'][~old] == warm_model.INVALID_CONFIDENCE)
    assert np.array_equal(guarded['evicted_comparisons'], old*len(EVICTED[history]))
    lost = np.zeros((len(x), 2, 2))
    for slot, position in enumerate(EVICTED[history], start=2-len(EVICTED[history])):
        lost[:, slot] = x[rows, writes[:, position], 2:4]
    assert np.array_equal(guarded['evicted_keys'], lost)
    for field in ('key_bank', 'value_bank', 'read_onset', 'eviction_seen', 'cue_scores', 'cue_weights', 'cue_read_vectors'):
        assert np.array_equal(original[field], replay[field]) and np.array_equal(guarded[field], replay[field]), field
    for field in ('logged_write_values', 'logged_write_ticks', 'logged_nonwrite_values', 'logged_nonwrite_ticks',
                  'logged_nonwrite_count', 'first_read_tick'):
        assert np.array_equal(guarded[field], replay[field]), field
    for rule, current in (('replay', replay), ('guarded', guarded)):
        triggered = current['reset_steps'].any(axis=1)
        assert np.array_equal(current['replay_steps'], first*triggered), rule
        assert not current['replay_reference'][~triggered].any() and not current['replay_resources'][~triggered].any()
        for field in ('counts', 'resources', 'spikes', 'cue_native_before', 'cue_native_after'):
            mask = triggered[:len(current[field])]
            assert np.array_equal(current[field][~mask], original[field][~mask]), (rule, field)
            if rule == 'guarded':
                assert np.array_equal(current[field][mask], replay[field][mask]), (rule, field)
        if history == 'two':
            assert all(np.array_equal(current[name], values) for name, values in original.items()), rule


def _spot_check(x, w, q, cases):
    maximum, checks, replay_steps = 0., 0, 0
    for rule in RULES:
        raw, diagnostics, _ = guard_model.evaluate(x[:8], w, q, rule)
        expected = raw | diagnostics
        assert cases[rule].keys() == expected.keys(), rule
        for field, values in expected.items():
            observed = cases[rule][field][:8]
            assert observed.shape == values.shape and observed.dtype == values.dtype, (rule, field)
            if field in EXACT:
                np.testing.assert_array_equal(observed, values, err_msg=f'{rule} {field}')
            else:
                residual = float(np.max(np.abs(observed-values)))
                assert residual <= TOLERANCE, (rule, field, residual)
                maximum = max(maximum, residual)
            if field in PULSES:
                np.testing.assert_array_equal(observed[..., 20:], values[..., 20:], err_msg=f'{rule} {field} pulses')
                checks += 1
            checks += 1
        if diagnostics:
            replay_steps += int(diagnostics['replay_steps'].sum())
    return checks, maximum, replay_steps


def _position_mean(scores, field, positions):
    return float(np.mean([scores[field][position] for position in positions])) if positions else None


def _score(base, geometry, history, rule, raw, first, shared_seconds, work):
    scores = capacity_metrics.score(raw | dict(cue_scores=first['first_read_scores'][:, None, :],
                                               cue_weights=first['first_read_weights'][:, None, :]), base)
    evicted, missing = EVICTED[history], MISSING[history]
    evicted_ba = _position_mean(scores, 'ba_by_position', evicted)
    evicted_coverage = _position_mean(scores, 'coverage_by_position', evicted)
    missing_ba = _position_mean(scores, 'ba_by_position', missing)
    missing_coverage = _position_mean(scores, 'coverage_by_position', missing)
    if missing:
        assert missing_ba == .5*missing_coverage
    selected = base['selected']
    reset = raw['reset_steps'].any(axis=1)
    return dict(condition=f'{geometry}_{history}_{rule}', geometry=geometry, history=history, rule=rule,
                examples=len(selected), T=confirm_data.T, q=confirm_data.Q, **scores,
                evicted_positions=list(evicted), missing_positions=list(missing), evicted_ba=evicted_ba,
                evicted_coverage=evicted_coverage, missing_ba=missing_ba, missing_coverage=missing_coverage,
                mean_resources=raw['resources'].mean(axis=0).tolist(),
                replay_resources_mean=raw['replay_resources'].mean(axis=0).tolist() if 'replay_resources' in raw else [0.]*4,
                reset_count=int(raw['reset_steps'].sum()), onset_count=int(raw['read_onset'].sum()),
                eviction_fraction=float(raw['eviction_seen'].any(axis=1).mean()), reset_fraction=float(reset.mean()),
                older_reset_fraction=float(reset[selected < 2].mean()), recent_reset_fraction=float(reset[selected >= 2].mean()),
                reset_by_position=[float(reset[selected == i].mean()) for i in range(4)],
                replay_steps=int(raw['replay_steps'].sum()) if 'replay_steps' in raw else 0,
                evicted_comparisons=int(raw['evicted_comparisons'].sum()) if 'evicted_comparisons' in raw else 0,
                shared_group_seconds=shared_seconds, shared_group_work=work)


def _isolate(base, inputs, w, retained, guard, single_order=None):
    records = []
    schedules = base['group'] % len(confirm_data.SCHEDULES)
    for geometry in GEOMETRIES:
        x = inputs[f'{geometry}_four']
        cases = retained[geometry]
        triggered = cases['replay']['replay_steps'] > 0
        for schedule, (label, _, _, _) in enumerate(confirm_data.SCHEDULES):
            for sample in np.flatnonzero(triggered & (schedules == schedule))[:4]:
                order = single_order
                if order is None:
                    order = ('replay', 'guarded') if len(records) % 2 == 0 else ('guarded', 'replay')
                row = dict(geometry=geometry, sample=int(sample), schedule=label, order=list(order))
                for rule in order:
                    guard()
                    started = time.perf_counter()
                    raw, diag, timing = guard_model.evaluate(x[sample:sample+1], w, confirm_data.Q, rule)
                    row[f'{rule}_call_seconds'] = time.perf_counter()-started
                    case = cases[rule]
                    for field in ('counts', 'resources', 'reset_steps'):
                        np.testing.assert_array_equal(raw[field][0], case[field][sample], err_msg=f'isolated {field}')
                    np.testing.assert_array_equal(diag['replay_resources'][0], case['replay_resources'][sample])
                    steps = int(diag['replay_steps'][0])
                    assert steps == case['replay_steps'][sample] and timing['replay_batches'] == int(steps > 0)
                    assert (timing['replay_wall_seconds'] > 0.) == (steps > 0)
                    if rule == 'replay':
                        row.update(replay_steps=steps, replay_seconds=timing['replay_wall_seconds'])
                    else:
                        comparisons = int(diag['evicted_comparisons'].sum())
                        assert comparisons == case['evicted_comparisons'][sample].sum()
                        assert timing['guard_batches'] == 1 and timing['guard_wall_seconds'] > 0.
                        row.update(guarded_steps=steps, guard_seconds=timing['guard_wall_seconds'],
                                   guarded_replay_seconds=timing['replay_wall_seconds'], evicted_comparisons=comparisons)
                records.append(row)
    return records


def execute(directory, n, guard, packet_size=32, compression=ZIP_DEFLATED, inputs_factory=None, single_order=None):
    """Run, verify and archive a complete supplied block or the default fixture."""
    if n not in (512, 256):
        raise ValueError('only the frozen 512- and 256-example fixture sizes are allowed')
    if packet_size not in (32, 256) or compression not in (ZIP_DEFLATED, ZIP_STORED):
        raise ValueError('expected the frozen packet size and lossless NPZ method')
    if single_order is not None and single_order not in (('replay', 'guarded'), ('guarded', 'replay')):
        raise ValueError('isolated calls require both replay and guarded rules once')
    started = time.perf_counter()
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    tick = time.perf_counter()
    base, inputs, w = _fixture(n) if inputs_factory is None else inputs_factory()
    assert len(base['y']) == n and w.shape == (180,) and np.isfinite(w).all()
    assert set(inputs) == {f'{geometry}_{history}' for geometry in GEOMETRIES for history in HISTORIES}
    assert all(x.shape == (n, confirm_data.T, 6) for x in inputs.values())
    input_seconds = time.perf_counter()-tick
    tick = time.perf_counter()
    _save_npz(directory/'base.npz', base, compression)
    np.save(directory/'weights.npy', w, allow_pickle=False)
    archive_seconds = time.perf_counter()-tick

    rows, retained = [], {}
    shared_seconds, verify_score_seconds = 0., 0.
    work = dict(main_sample_steps=0, branch_sample_steps=0, replay_sample_steps=0, online_sample_steps=0,
                native_macs=0, shared_calls=0, spot_calls=0, spot_online_sample_steps=0,
                spot_replay_sample_steps=0, spot_native_macs=0, evicted_comparisons=0)
    metrics = dict(input_array_bytes=sum(x.nbytes for x in inputs.values()),
                   base_array_bytes=sum(x.nbytes for x in base.values()), weights_array_bytes=w.nbytes,
                   retained_raw_array_bytes=0, spot_checks=0, max_numeric_error=0.)
    for group, x in inputs.items():
        geometry, history = group.split('_', 1)
        packets = []
        group_seconds = 0.
        group_work = dict(main_sample_steps=0, branch_sample_steps=0, replay_sample_steps=0)
        for start in range(0, n, packet_size):
            guard()
            tick = time.perf_counter()
            cases, ledger = shared_model.evaluate(x[start:start+packet_size], w, confirm_data.Q)
            group_seconds += time.perf_counter()-tick
            packets.append(cases)
            for name, value in ledger.items():
                group_work[name] += value
            work['shared_calls'] += 1
        shared_seconds += group_seconds
        group_work['online_sample_steps'] = group_work['main_sample_steps']+group_work['branch_sample_steps']
        group_work['native_macs'] = 156*(group_work['online_sample_steps']+group_work['replay_sample_steps'])
        for name, value in group_work.items():
            work[name] += value

        tick = time.perf_counter()
        cases = _merge(packets)
        try:
            _verify_paths(history, x, base, cases)
            checks, residual, spot_steps = _spot_check(x, w, confirm_data.Q, cases)
            first = _first_read(x, cases['replay'], w)
            assert all(np.isfinite(values).all() for values in first.values())
            rows.extend(_score(base, geometry, history, rule, cases[rule], first, group_seconds, group_work) for rule in RULES)
        except AssertionError:
            for rule, arrays in cases.items():
                _save_npz(directory/f'{group}_{rule}_FAILURE.npz', arrays, compression)
            raise
        metrics['spot_checks'] += checks
        metrics['max_numeric_error'] = max(metrics['max_numeric_error'], residual)
        work['spot_calls'] += len(RULES)
        work['spot_online_sample_steps'] += len(RULES)*8*confirm_data.T
        work['spot_replay_sample_steps'] += spot_steps
        work['spot_native_macs'] += 156*(len(RULES)*8*confirm_data.T+spot_steps)
        work['evicted_comparisons'] += int(cases['guarded']['evicted_comparisons'].sum())
        verify_score_seconds += time.perf_counter()-tick

        tick = time.perf_counter()
        for rule, values in cases.items():
            archive = values | first
            _save_npz(directory/f'{group}_{rule}.npz', archive, compression)
            metrics['retained_raw_array_bytes'] += sum(array.nbytes for array in archive.values())
        archive_seconds += time.perf_counter()-tick
        if history == 'four':
            retained[geometry] = {rule: cases[rule] for rule in ('replay', 'guarded')}
        guard(disk=True)

    tick = time.perf_counter()
    isolated = _isolate(base, inputs, w, retained, guard, single_order)
    isolation_seconds = time.perf_counter()-tick
    isolated_steps = sum(row['replay_steps']+row['guarded_steps'] for row in isolated)
    isolated_online = 2*len(isolated)*confirm_data.T
    work.update(isolated_calls=2*len(isolated), isolated_online_sample_steps=isolated_online,
                isolated_replay_sample_steps=isolated_steps, isolated_native_macs=156*(isolated_online+isolated_steps),
                isolated_evicted_comparisons=sum(row['evicted_comparisons'] for row in isolated))
    metrics['isolated_records'] = len(isolated)
    assert len(rows) == len(GEOMETRIES)*len(HISTORIES)*len(RULES) == 24
    tick = time.perf_counter()
    _dump(directory/'ROWS.json', rows)
    _dump(directory/'MICROBENCH.json', isolated)
    archive_seconds += time.perf_counter()-tick
    guard(disk=True)
    metrics['archived_bytes_before_block'] = sum(path.stat().st_size for path in directory.iterdir() if path.is_file())
    total_seconds = time.perf_counter()-started
    other_seconds = total_seconds-(input_seconds+shared_seconds+verify_score_seconds+archive_seconds+isolation_seconds)
    record = dict(examples=n, paths=len(rows), input_seconds=input_seconds, shared_seconds=shared_seconds,
                  verify_score_seconds=verify_score_seconds, archive_seconds=archive_seconds,
                  isolation_seconds=isolation_seconds, other_seconds=other_seconds, total_seconds=total_seconds,
                  work=work, metrics=metrics)
    _dump(directory/'BLOCK.json', record)
    return record
