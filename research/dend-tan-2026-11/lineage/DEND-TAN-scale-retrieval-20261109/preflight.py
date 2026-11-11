"""Verify native continuation, matched donor states, and workload on independent fixtures."""

from pathlib import Path
import time

import numpy as np

import independent
import independent_data
import intervene
import population
from run import ARMS, BLOCKS, dump, peak_memory
import scale_model
import task_data


ROOT = Path(__file__).resolve().parent
DATA_SEED = 4099901601
WEIGHT_SEED = 4099901602
TOL = 5e-12


def verify_fixture(n, start):
    base = task_data.make_base(n, DATA_SEED)
    ibase = independent_data.make_base(n, DATA_SEED)
    assert all(np.array_equal(array, ibase[name]) for name, array in base.items())
    x, q = task_data.assemble(base, 16)
    ix, iq = independent_data.assemble(ibase, 16)
    assert q == iq == 28 and np.array_equal(x, ix)
    assert not x[:, 9:25].any()
    donors, idonors = intervene.donors(base), independent.donors(ibase)
    assert all(np.array_equal(array, idonors[name]) for name, array in donors.items())
    lookup = independent_data.lookup(ibase)
    twin = np.asarray([lookup[octet, 1-cue, a, b] for octet, cue, a, b in zip(
        base['octet'], base['selected'], base['first_bit'], base['second_bit'])])
    assert np.array_equal(x[:, :25], x[twin, :25])
    assert np.array_equal(donors['selected'][twin], twin[donors['unselected']])
    assert np.array_equal(donors['unselected'][twin], twin[donors['selected']])
    disagree = base['first_bit'] != base['second_bit']
    w = population.initialize(WEIGHT_SEED, 6)
    production, independent_time, checks, maximum, crossover_error = 0., 0., 0, 0., 0.

    def exact(a, b):
        nonlocal checks
        assert np.array_equal(a, b)
        checks += 1

    def close(a, b):
        nonlocal checks, maximum
        err = float(np.max(np.abs(a-b)))
        assert err <= TOL, err
        maximum = max(maximum, err)
        checks += 1
        return err

    for arm in ARMS:
        official = scale_model.evaluate(x, w, arm, q)
        evaluations = {}
        for case in intervene.conditions(arm):
            assert time.perf_counter()-start < 120, '120 second fixture limit'
            tick = time.perf_counter()
            got = intervene.evaluate(x, w, arm, q, case, donors)
            production += time.perf_counter()-tick
            tick = time.perf_counter()
            ref = independent.replay(ix, w, arm, iq, case, idonors)
            independent_time += time.perf_counter()-tick
            assert got.keys() == ref.keys()
            for name, value in got.items():
                if name in ('counts', 'spikes', 'resources'):
                    exact(value, ref[name])
                else:
                    close(value, ref[name])
            exact(got['entry_native'][:, 20:], ref['entry_native'][:, 20:])
            evaluations[case] = got
        nominal = evaluations['nominal']
        for key in ('counts', 'spikes', 'injections'):
            exact(nominal[key], official[key])
        exact(nominal['resources'], official['resources'][:, :4])
        exact(nominal['states'][..., :20], official['states'])
        exact(nominal['states'][..., 20:], official['spikes'][..., :6])
        for key in ('traces', 'weights_fast'):
            if key in nominal:
                exact(nominal[key], official[key])
        for key in nominal:
            exact(evaluations['identity'][key], nominal[key])
        close(nominal['entry_native'][:, :20], nominal['entry_native'][twin, :20])
        exact(nominal['entry_native'][:, 20:], nominal['entry_native'][twin, 20:])
        for case, got in evaluations.items():
            for key in ('states', 'spikes', 'injections', 'traces', 'weights_fast'):
                if key in got:
                    exact(got[key][:, :25], nominal[key][:, :25])
            if case.endswith(('_selected', '_unselected')):
                family, role = case.rsplit('_', 1)
                index = donors[role]
                expected = nominal['entry_native'][index] if family in ('native', 'all') else nominal['entry_native']
                exact(got['entry_native'], expected)
                exact(got['counts'][~disagree], nominal['counts'][~disagree])
                if family == 'all' or arm == 'GAIN2':
                    exact(got['counts'], nominal['counts'][index])
                if 'entry_traces' in got:
                    expected = nominal['entry_traces'][index] if family in ('trace', 'all') else nominal['entry_traces']
                    exact(got['entry_traces'], expected)
                if role == 'selected':
                    other = evaluations[family+'_unselected']
                    err = close(got['entry_native'][:, :20], other['entry_native'][twin, :20])
                    crossover_error = max(crossover_error, err)
                    exact(got['entry_native'][:, 20:], other['entry_native'][twin, 20:])
                    if 'entry_traces' in got:
                        crossover_error = max(crossover_error, close(got['entry_traces'], other['entry_traces'][twin]))
            if case == 'post_mean':
                exact(got['entry_native'], nominal['entry_native'])
                exact(got['entry_traces'], nominal['entry_traces'])
                exact(got['weights_fast'][:, 25:], np.full_like(got['weights_fast'][:, 25:], .5))
                assert np.any(got['injections'][:, 25:] != nominal['injections'][:, 25:])
    assert peak_memory() < 1024**3
    return dict(examples=n, science_seconds=production, audit_seconds=independent_time,
                science_forecast_seconds=3*BLOCKS*production+30,
                audit_forecast_seconds=3*BLOCKS*independent_time+30,
                checks=checks, max_numeric_error=maximum, max_crossover_error=crossover_error)


def main():
    assert not any((ROOT / name).exists() for name in ('PRECHECK.json', 'TIMING.json', 'FREEZE.json',
                                                       'RUN_STATE.json', 'evaluate'))
    start = time.perf_counter()
    candidates = []
    chosen = None
    try:
        for n in (256, 128):
            result = verify_fixture(n, start)
            candidates.append(result)
            if result['science_forecast_seconds'] < 840 and result['audit_forecast_seconds'] < 300:
                chosen = n
                break
    except Exception as error:
        attempt = 1+len(list(ROOT.glob('PRECHECK_FAILURE_*.json')))
        dump(f'PRECHECK_FAILURE_{attempt}.json', dict(status='FAIL', reason=str(error),
             exception=type(error).__name__, scientific_data_generated=False))
        raise
    dump('PRECHECK.json', dict(status='PASS', scientific_data_generated=False,
         data_seed=DATA_SEED, weight_seed=WEIGHT_SEED, candidates=candidates,
         peak_working_set_bytes=peak_memory(), wall_seconds=time.perf_counter()-start))
    dump('TIMING.json', dict(status='PASS' if chosen else 'INFEASIBLE', blocks=BLOCKS,
         examples_per_block=chosen, candidates=candidates, scientific_data_generated=False))
    if chosen is None:
        raise SystemExit('No allowed example count fits both time budgets')
    print(f'PRECHECK PASS: {candidates[-1]}', flush=True)


if __name__ == '__main__':
    main()
