"""Verify factorial donor pairing, native replay, and workload before freezing."""

import time

import numpy as np

import capacity_data
import donor_map
import independent_capacity_data
import independent_donor
import independent_warm
import population
import warm_model
from run import BLOCKS, CAP, ROOT, dump, evaluate_cases, peak_memory


DATA_SEED = 4099902601
WEIGHT_SEED = 4099902602
DONOR_CASES = dict(own='full_own_natural', target='full_target_flip',
                   other='full_other_flip', deleted='full_deleted_flip')
DISCRETE = ('counts', 'spikes', 'resources', 'read_onset', 'reset_steps', 'eviction_seen')


def check_maps(base, full_x, late_x, maps, independent_maps):
    n = len(base['y'])
    rows = np.arange(n)
    selected = base['selected']
    recent = selected >= 2
    assert set(maps) == set(independent_maps) == set(DONOR_CASES)
    assert np.array_equal(maps['own'], rows)
    for name, donors in maps.items():
        assert donors.shape == (n,) and donors.dtype == np.int64
        assert np.array_equal(donors, independent_maps[name]), name
        assert np.array_equal(donors[donors], rows), name
        for field in ('group', 'selected'):
            assert np.array_equal(base[field][donors], base[field]), (name, field)

        change = np.zeros_like(base['bits'], dtype=bool)
        if name == 'target':
            change[rows, selected] = True
        if name == 'other':
            change[rows[recent], 5-selected[recent]] = True
        if name == 'deleted':
            change[:, 0] = True
        assert np.array_equal(base['bits'][donors] != base['bits'], change), name
        assert np.array_equal(base['y'][donors] != base['y'], change[rows, selected]), name

        for x, first_position in ((full_x, 0), (late_x, 2)):
            expected = x.copy()
            for position in range(first_position, 4):
                flip = change[:, position]
                tick = capacity_data.WRITE_STEPS[position]
                expected[flip, tick, :2] = x[flip, tick, :2][:, ::-1]
            assert np.array_equal(x[donors], expected), (name, first_position)


def verify_fixture(n, started):
    production_start = time.perf_counter()
    base = capacity_data.make_base(n, DATA_SEED)
    full_x, q = capacity_data.assemble(base, 'full4')
    late_x, late_q = capacity_data.assemble(base, 'base2')
    maps = donor_map.build(base)
    w = np.r_[population.initialize(WEIGHT_SEED, 6), [.03, -.4, .45, -.02]]
    warm = warm_model.warm_reference(w)
    cases = evaluate_cases(base, full_x, late_x, q, w, warm, maps)
    production_seconds = time.perf_counter()-production_start

    audit_start = time.perf_counter()
    independent_base = independent_capacity_data.make_base(n, DATA_SEED)
    ifull, iq = independent_capacity_data.assemble(independent_base, 'full4')
    ilate, ilate_q = independent_capacity_data.assemble(independent_base, 'base2')
    independent_maps = independent_donor.build(independent_base)
    iwarm = independent_warm.warm_reference(w)
    expected = {
        'late_original': independent_warm.replay(ilate, w, 'LEARN', iq, 'original'),
        'full_original': independent_warm.replay(ifull, w, 'LEARN', iq, 'original'),
        'full_eviction_warm': independent_warm.replay(ifull, w, 'LEARN', iq, 'eviction_warm', iwarm),
    }
    boundary = expected['late_original']['cue_native_before'][:, 0]
    for name, condition in DONOR_CASES.items():
        expected[condition] = independent_warm.replay(ifull, w, 'LEARN', iq, 'eviction_late_native',
                                                      boundary[independent_maps[name]])
    audit_seconds = time.perf_counter()-audit_start

    checks, maximum = 0, 0.

    def exact(a, b, context):
        nonlocal checks
        assert np.array_equal(a, b), context
        checks += 1

    def close(a, b, context):
        nonlocal checks, maximum
        assert a.shape == b.shape, context
        error = float(np.max(np.abs(a-b)))
        assert error <= 5e-12, (context, error)
        maximum = max(maximum, error)
        checks += 1

    assert q == late_q == iq == ilate_q == 28
    assert base.keys() == independent_base.keys()
    for name, values in base.items():
        exact(values, independent_base[name], name)
    exact(full_x, ifull, 'full input')
    exact(late_x, ilate, 'late input')
    exact(full_x[:, 3:], late_x[:, 3:], 'shared suffix')
    assert independent_capacity_data.check(independent_base, ifull, 'full4')['ceiling'] == 1.
    assert independent_capacity_data.check(independent_base, ilate, 'base2')['ceiling'] == .75
    check_maps(base, full_x, late_x, maps, independent_maps)
    close(warm, iwarm, 'warm reference')
    exact(warm[20:], iwarm[20:], 'warm pulses')

    assert cases.keys() == expected.keys()
    for condition, actual in cases.items():
        assert actual.keys() == expected[condition].keys(), condition
        for name, values in actual.items():
            (exact if name in DISCRETE else close)(values, expected[condition][name], (condition, name))
        for name in ('cue_native_before', 'cue_native_after'):
            exact(actual[name][..., 20:], expected[condition][name][..., 20:], (condition, name, 'pulses'))

    own = cases['full_own_natural']
    full = cases['full_original']
    late = cases['late_original']
    reset_steps = own['reset_steps']
    assert reset_steps[:, 25].any() and not reset_steps[:, 25].all()
    assert reset_steps.sum() == reset_steps[:, 25].sum()
    assert own['read_onset'][:, 25].all() and own['read_onset'].sum() == n
    assert not late['eviction_seen'].any()
    assert not full['reset_steps'].any()
    exact(full['eviction_seen'], np.broadcast_to(np.arange(36) > 4, (n, 36)), 'full eviction')
    no_reset = ~reset_steps.any(axis=1)
    older = base['selected'] < 2
    natural = late['cue_native_before'][:, 0]
    exact(natural[maps['deleted']], natural, 'deleted donor boundary')
    exact(natural[maps['target']][older], natural[older], 'older target donor boundary')
    exact(natural[maps['other']][older], natural[older], 'older other donor boundary')
    for name, values in own.items():
        exact(cases['full_deleted_flip'][name], values, (name, 'deleted donor'))
        for condition in ('full_target_flip', 'full_other_flip'):
            mask = older[:len(values)]
            exact(cases[condition][name][mask], values[mask], (condition, name, 'older donor'))

    references = {condition: natural[maps[name], None, :] for name, condition in DONOR_CASES.items()}
    references['full_eviction_warm'] = warm
    for condition, reference in references.items():
        actual = cases[condition]
        for name in ('read_onset', 'key_bank', 'value_bank', 'cue_scores', 'cue_weights',
                     'cue_read_vectors', 'eviction_seen'):
            exact(actual[name], full[name], (condition, name, 'recipient path'))
        for name in ('reset_steps', 'match_confidence'):
            exact(actual[name], own[name], (condition, name, 'shared gate'))
        replaced = np.where(reset_steps[:, 25:28, None], reference, actual['cue_native_before'])
        exact(actual['cue_native_after'], replaced, (condition, 'replacement'))
        for name, values in actual.items():
            if name in ('match_confidence', 'reset_steps'):
                continue
            mask = no_reset[:len(values)]
            exact(values[mask], full[name][mask], (condition, name, 'untriggered recipient'))

    assert time.perf_counter()-started < 120 and peak_memory() < 1024**3
    return dict(examples=n, science_seconds=production_seconds, audit_seconds=audit_seconds,
                science_forecast_seconds=3*BLOCKS*production_seconds+30,
                audit_forecast_seconds=3*BLOCKS*audit_seconds+30, checks=checks, max_numeric_error=maximum,
                paths_verified=len(cases), maps_verified=len(maps), triggers=int(reset_steps.sum()))


def main():
    assert not any((ROOT / name).exists() for name in ('PRECHECK.json', 'TIMING.json', 'FREEZE.json',
                                                       'RUN_STATE.json', 'evaluate'))
    start = time.perf_counter()
    candidates = []
    chosen = None
    try:
        for n in (512, 256):
            result = verify_fixture(n, start)
            candidates.append(result)
            if result['science_forecast_seconds'] < CAP and result['audit_forecast_seconds'] < 300:
                chosen = n
                break
    except Exception as error:
        attempt = 1+len(list(ROOT.glob('PRECHECK_FAILURE_*.json')))
        dump(f'PRECHECK_FAILURE_{attempt}.json', dict(status='FAIL', reason=str(error),
             exception=type(error).__name__, scientific_data_generated=False))
        raise
    dump('PRECHECK.json', dict(status='PASS', scientific_data_generated=False, donor_fixture='PASS',
         data_seed=DATA_SEED, weight_seed=WEIGHT_SEED, candidates=candidates,
         peak_working_set_bytes=peak_memory(), wall_seconds=time.perf_counter()-start))
    dump('TIMING.json', dict(status='PASS' if chosen else 'INFEASIBLE', blocks=BLOCKS,
         examples_per_block=chosen, candidates=candidates, scientific_data_generated=False))
    if chosen is None:
        raise SystemExit('No allowed example count fits both time budgets')
    print(f'PRECHECK PASS: {candidates[-1]}', flush=True)


if __name__ == '__main__':
    main()
