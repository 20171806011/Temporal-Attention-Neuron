"""Check first-read semantics, independent references, and the frozen timing gate."""

import time

import numpy as np

import capacity_data
import event_model
import independent_capacity_data
import independent_event
import independent_warm
import learned_model
import population
from run import BLOCKS, CAP, ROOT, T, assemble_histories, dump, evaluate_cases, peak_memory


DATA_SEED = 4099902701
WEIGHT_SEED = 4099902702
DISCRETE = {'counts', 'resources', 'spikes', 'read_onset', 'reset_steps', 'eviction_seen',
            'replay_steps', 'replay_resources', 'first_read_tick', 'key_bank', 'write_gate'}


class Checks:
    def __init__(self):
        self.count = 0
        self.maximum = 0.

    def exact(self, actual, expected, context):
        assert np.array_equal(actual, expected), context
        self.count += 1

    def compare(self, actual, expected, context):
        assert actual.keys() == expected.keys(), context
        for field, array in actual.items():
            other = expected[field]
            assert array.shape == other.shape and array.dtype == other.dtype, (context, field)
            if field in DISCRETE or field.startswith('logged_'):
                self.exact(array, other, (context, field))
            else:
                error = float(np.max(np.abs(array-other)))
                assert error <= 5e-12, (context, field, error)
                self.maximum = max(self.maximum, error)
                self.count += 1
            if field in ('cue_native_before', 'cue_native_after', 'replay_reference'):
                self.exact(array[..., 20:], other[..., 20:], (context, field, 'p'))


def shifted_fixture(w, checks):
    w = w.copy()
    w[176:] = np.eye(2).ravel()
    x = np.zeros((7, T, 6))
    for t, key in ((0, (-1., 0.)), (2, (1., 0.)), (4, (0., 1.))):
        x[:, t, 0], x[:, t, 2:4], x[:, t, 5] = 1., key, 1.
    x[:, (1, 8), 4] = (.2, .4)
    x[0, 3, 2:4] = (1., 0.)
    x[0, 12, 2:4] = (0., 1.)
    x[1, 10:12, 2:4] = (0., 1.)
    x[1, 18, 2:4] = (1., 0.)
    x[1, 14] = x[1, 2]
    x[2, 10, 2:4] = (-1., 0.)
    x[2, 18, 2:4] = (0., 1.)
    x[3, 33, 4] = .2
    x[4, 2, 2:4] = 0.
    x[4, 10, 2:4] = (0., 1.)
    x[5, 10, 2:4] = (1., 1.)
    x[6] = x[1]
    x[6, 11, 4] = .8
    x[6, 14] = x[6, 4]
    raw, diag, timing = event_model.evaluate(x, w, 28, 'replay')
    checks.compare(raw | diag, independent_event.replay(x, w, 28), 'shifted/multiple read')
    checks.exact(diag['first_read_tick'], [3, 10, 10, -1, 10, 10, 10], 'first read')
    checks.exact(diag['replay_steps'], [0, 10, 0, 0, 0, 10, 10], 'first read only')
    checks.exact(np.argwhere(raw['reset_steps']), [[1, 10], [5, 10], [6, 10]], 'read policy')
    checks.exact(raw['match_confidence'][5, 10], 1./np.sqrt(2.), 'threshold equality')
    checks.exact(diag['logged_nonwrite_ticks'][1, :2], [1, 8], 'nonwrite before oldest retained write')
    checks.exact(diag['logged_write_ticks'][1], [2, 4], 'post-read write excluded')
    for name in diag:
        checks.exact(diag[name][1], diag[name][6], (name, 'same prefix/different suffix'))
    checks.exact(timing['replay_batches'], 1, 'grouped first read call')
    checks.exact(timing['replay_scratch_peak_bytes'], 3*10*6*8, 'prefix input allocation')


def verify_fixture(n, start):
    checks = Checks()
    w = np.r_[population.initialize(WEIGHT_SEED, 6), learned_model.ORACLE_MATRIX.ravel()]
    tick = time.perf_counter()
    base = capacity_data.make_base(n, DATA_SEED)
    inputs, q = assemble_histories(base)
    results = evaluate_cases(inputs, q, w)
    science = time.perf_counter()-tick

    tick = time.perf_counter()
    ibase = independent_capacity_data.make_base(n, DATA_SEED)
    checks.exact(set(base), set(ibase), 'fixture data fields')
    for name in base:
        checks.exact(base[name], ibase[name], name)
    two, iq = independent_event.assemble(ibase, 'two')
    own = independent_warm.replay(two, w, 'LEARN', iq, 'original')
    reference = own['cue_native_before'][:, 0]
    for history, x in inputs.items():
        ix, iq = independent_event.assemble(ibase, history)
        assert q == iq == 28
        checks.exact(x, ix, history)
        expected = {
            'original': own if history == 'two' else independent_warm.replay(ix, w, 'LEARN', iq, 'original'),
            'replay': independent_event.replay(ix, w, iq),
            'dense': independent_warm.replay(ix, w, 'LEARN', iq, 'eviction_late_native', reference),
        }
        for rule in expected:
            raw, diag, timing = results[f'{history}_{rule}']
            checks.compare(raw | diag, expected[rule], (history, rule))
            if rule == 'replay':
                active = raw['reset_steps'][:, 25]
                expected_reset = base['selected'] >= 2 if history != 'two' else np.zeros(n, dtype=bool)
                checks.exact(active, expected_reset, (history, 'analytic retention'))
                per_batch = [int(active[i:i+32].sum()) for i in range(0, n, 32)]
                checks.exact(timing['replay_batches'], np.count_nonzero(per_batch), 'replay batch count')
                checks.exact(timing['replay_scratch_peak_bytes'], max(per_batch)*25*6*8, 'batch input scratch')
    audit = time.perf_counter()-tick
    shifted_fixture(w, checks)

    full, diag, _ = results['four_replay']
    episodes = np.flatnonzero(diag['replay_steps'])[:16]
    assert len(episodes) == 16
    latencies = []
    for i in episodes:
        raw, single_diag, timing = event_model.evaluate(inputs['four'][i:i+1], w, q, 'replay')
        checks.exact(raw['counts'][0], full['counts'][i], 'single-episode counts')
        checks.exact(single_diag['replay_resources'][0], diag['replay_resources'][i], 'single-episode replay events')
        checks.exact(timing['replay_batches'], 1, 'single-episode invocation')
        assert timing['replay_wall_seconds'] > 0.
        latencies.append(timing['replay_wall_seconds'])
    assert time.perf_counter()-start < 120 and peak_memory() < 1024**3
    return dict(examples=n, science_seconds=science, audit_seconds=audit,
                science_forecast_seconds=3*BLOCKS*science+30, audit_forecast_seconds=3*BLOCKS*audit+30,
                checks=checks.count, max_numeric_error=checks.maximum, verified_paths=12,
                single_episode_replay_seconds=latencies)


def main():
    assert not any((ROOT / name).exists() for name in ('PRECHECK.json', 'TIMING.json', 'FREEZE.json',
                                                       'RUN_STATE.json', 'evaluate'))
    start = time.perf_counter()
    candidates = []
    selected = None
    try:
        for n in (512, 256):
            result = verify_fixture(n, start)
            candidates.append(result)
            if result['science_forecast_seconds'] < CAP and result['audit_forecast_seconds'] < 300:
                selected = n
                break
    except Exception as error:
        attempt = 1+len(list(ROOT.glob('PRECHECK_FAILURE_*.json')))
        dump(f'PRECHECK_FAILURE_{attempt}.json', dict(status='FAIL', reason=str(error),
             exception=type(error).__name__, scientific_data_generated=False))
        raise
    dump('PRECHECK.json', dict(status='PASS', scientific_data_generated=False, onset_fixture='PASS',
         data_seed=DATA_SEED, weight_seed=WEIGHT_SEED, candidates=candidates,
         peak_working_set_bytes=peak_memory(), wall_seconds=time.perf_counter()-start))
    dump('TIMING.json', dict(status='PASS' if selected else 'INFEASIBLE', blocks=BLOCKS,
         examples_per_block=selected, candidates=candidates, scientific_data_generated=False))
    if selected is None:
        raise SystemExit('No allowed example count fits both time budgets')
    print(f'PRECHECK PASS: {candidates[-1]}', flush=True)


if __name__ == '__main__':
    main()
