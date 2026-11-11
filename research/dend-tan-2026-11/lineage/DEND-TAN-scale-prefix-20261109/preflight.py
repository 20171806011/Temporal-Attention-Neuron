"""Check both prefix replays against source and time a full fixture block."""

from pathlib import Path
import time

import numpy as np

import independent
import independent_data
import population
import prefix
from run import ARMS, BLOCKS, dump, peak_memory
import scale_model
import task_data


ROOT = Path(__file__).resolve().parent
DATA_SEED = 4099901701
WEIGHT_SEED = 4099901702
TOL = 5e-12


def verify_fixture(n, start):
    base = task_data.make_base(n, DATA_SEED)
    ibase = independent_data.make_base(n, DATA_SEED)
    assert all(np.array_equal(array, ibase[name]) for name, array in base.items())
    x, q = task_data.assemble(base, 16)
    ix, iq = independent_data.assemble(ibase, 16)
    assert q == iq == 28 and np.array_equal(x, ix) and not x[:, 9:25].any()
    w = population.initialize(WEIGHT_SEED, 6)
    production, independent_time, checks, maximum = 0., 0., 0, 0.

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

    for arm in ARMS:
        assert time.perf_counter()-start < 120, '120 second fixture limit'
        official = scale_model.evaluate(x, w, arm, q)
        tick = time.perf_counter()
        cases = prefix.evaluate(x, w, arm, q)
        production += time.perf_counter()-tick
        assert set(cases) == set(prefix.conditions(arm))
        for condition, got in cases.items():
            tick = time.perf_counter()
            ref = independent.replay(ix, w, arm, iq, condition)
            independent_time += time.perf_counter()-tick
            assert got.keys() == ref.keys()
            for name, value in got.items():
                if name in ('counts', 'spikes', 'resources'):
                    exact(value, ref[name])
                else:
                    close(value, ref[name])
            exact(got['entry_native'][:, 20:], ref['entry_native'][:, 20:])
            exact(got['counts'][:8], got['spikes'][:, q:, 6:].sum(axis=1))
            if arm != 'GAIN2':
                exact(got['injections'][:, 9:25], np.zeros((8, 16, 12)))
                close(got['traces'][:, 9:25], got['traces'][:, 8:24]*np.array([.85, .97]))
        nominal = cases['nominal']
        for key in ('counts', 'spikes', 'injections'):
            exact(nominal[key], official[key])
        exact(nominal['resources'], official['resources'][:, :4])
        exact(nominal['states'][..., :20], official['states'])
        exact(nominal['states'][..., 20:], official['spikes'][..., :6])
        if arm != 'GAIN2':
            for key in ('traces', 'weights_fast'):
                exact(nominal[key], official[key])
        if arm == 'SCALE_MEAN':
            for key, values in nominal.items():
                exact(values, cases['prefix_mean'][key])
        if arm == 'SCALE':
            mean, md, dm = (cases[k] for k in ('prefix_mean', 'mean_native', 'mean_traces'))
            exact(md['entry_native'], mean['entry_native'])
            exact(md['entry_traces'], nominal['entry_traces'])
            exact(dm['entry_native'], nominal['entry_native'])
            exact(dm['entry_traces'], mean['entry_traces'])
            for key in ('states', 'spikes', 'injections', 'traces', 'weights_fast'):
                exact(md[key][:, :25], mean[key][:, :25])
                exact(dm[key][:, :25], nominal[key][:, :25])
            for values in (mean, md):
                a = values['weights_fast'][:, :25]
                exact(a, np.full_like(a, .5))
            assert np.any(nominal['injections'][:, :25] != mean['injections'][:, :25])
    assert peak_memory() < 1024**3 and time.perf_counter()-start < 120
    return dict(examples=n, science_seconds=production, audit_seconds=independent_time,
                science_forecast_seconds=3*BLOCKS*production+30,
                audit_forecast_seconds=3*BLOCKS*independent_time+30,
                checks=checks, max_numeric_error=maximum)


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
