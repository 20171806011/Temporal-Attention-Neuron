"""Choose the common update count from independent CPU timing fixtures only."""

import json
from pathlib import Path
import time

import numpy as np

import population
import scale_model
import task_data


ROOT = Path(__file__).resolve().parent
ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2', 'SCALE', 'SCALE_MEAN')
DELAYS = (0, 16)
DATA_SEED = 4099901401
WEIGHT_SEED = 4099901402
CAP_SECONDS = 840


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run_timing(start):
    base = task_data.make_base(64, DATA_SEED)
    rows = []
    for arm in ARMS:
        for delay in DELAYS:
            x, q = task_data.assemble(base, delay, 'nominal')
            weights = population.initialize(WEIGHT_SEED, 6)
            first = np.zeros_like(weights)
            second = np.zeros_like(weights)
            times = []
            for step in range(1, 13):
                assert time.perf_counter()-start < 120, '120 second fixture cap'
                tick = time.perf_counter()
                loss, grad, _ = scale_model.loss_grad(x, base['y'], weights, arm, q)
                assert np.isfinite(loss) and np.all(np.isfinite(grad))
                grad /= max(1., np.linalg.norm(grad))
                first = .9*first + .1*grad
                second = .999*second + .001*grad*grad
                mean = first/(1-.9**step)
                var = second/(1-.999**step)
                weights = np.clip(weights-.0025*mean/(np.sqrt(var)+1e-8), -2, 2)
                times.append(time.perf_counter()-tick)
            median = float(np.median(times[6:]))
            rows.append(dict(arm=arm, train_delay=delay, time_steps=x.shape[1], query_start=q,
                             seconds=times, last_six_median_seconds=median))
            print(f'TIMED {arm} D{delay}: {median:.6f}s/update', flush=True)
    unit_sum = sum(row['last_six_median_seconds'] for row in rows)
    forecasts = {str(updates): 2*3*updates*unit_sum+100 for updates in (600, 400)}
    selected = next((updates for updates in (600, 400) if forecasts[str(updates)] < CAP_SECONDS), None)
    return dict(status='PASS' if selected is not None else 'INFEASIBLE', updates=selected,
                forecast_seconds=forecasts[str(selected)] if selected is not None else None,
                candidate_forecast_seconds=forecasts, cap_seconds=CAP_SECONDS,
                formula='2 * 3 * updates * sum(last six per-cell update medians) + 100',
                fixture_rows=64, fixture_updates=len(rows)*12, fixture_data_seed=DATA_SEED,
                fixture_weight_seed=WEIGHT_SEED, scientific_seeds_used=False, rows=rows,
                wall_seconds=time.perf_counter()-start)


def main():
    assert not (ROOT / 'FREEZE.json').exists() and not (ROOT / 'RUN_STATE.json').exists()
    assert not (ROOT / 'TIMING.json').exists()
    assert json.loads((ROOT / 'PRECHECK.json').read_text(encoding='utf-8'))['status'] == 'PASS'
    start = time.perf_counter()
    try:
        result = run_timing(start)
    except Exception as error:
        failure = ROOT / f'TIMING_FAILURE_{len(list(ROOT.glob("TIMING_FAILURE_*.json")))+1}.json'
        write_json(failure, dict(status='FAIL', exception=type(error).__name__, reason=str(error),
                                 wall_seconds=time.perf_counter()-start))
        raise
    write_json(ROOT / 'TIMING.json', result)
    if result['status'] != 'PASS':
        raise SystemExit('Timing forecast exceeds 840 seconds even at 400 updates')
    print(f'TIMING PASS: {result["updates"]} updates, forecast {result["forecast_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
