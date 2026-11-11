"""Forecast runtime and storage using full-size independent fixture arms."""

from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

import continuation
import population
import probe
import task_data
from core import dump, memory
from preflight import ARMS, TAPS, configurations, donors, output_features


ROOT = Path(__file__).resolve().parent
FIT_SEED = 4099900401
TEST_SEED = 4099900402
WEIGHT_SEED = 4099900403
WALL_CAP = 840
DISK_CAP = int(1.5 * 1024**3)
RAM_CAP = 1024**3
ARCHIVE_ALLOWANCE = 64 * 1024
METADATA_ALLOWANCE = 64 * 1024**2


def joint_state(capture, tap):
    index = TAPS.index(tap)
    return np.concatenate([capture[key][:, index] for key in ('d', 'v', 'p', 'o')], axis=1)


def measure_arm(arm, folder, guard):
    start = time.perf_counter()
    folder.mkdir()
    archive_bound = 0

    def save(name, arrays):
        nonlocal archive_bound
        path = folder / (name + '.npz')
        np.savez_compressed(path, **arrays)
        bound = sum(np.asarray(value).nbytes for value in arrays.values()) + ARCHIVE_ALLOWANCE
        assert path.stat().st_size <= bound
        archive_bound += bound

    weights = population.initialize(WEIGHT_SEED)
    fit_base = task_data.make_base(128, FIT_SEED)
    fit_values, _ = task_data.assemble(fit_base, 'nominal')
    # Fixture-only state taps provide stand-ins for the previously locked probes.
    fit_capture = continuation.collect(fit_values, weights, arm)
    save('fit_data', fit_base)
    save('fit_baseline', {key: fit_capture[key] for key in ('raster', 'counts', 'resources')})
    output_probes = {}
    for kind in ('counts', 'raster'):
        output_probes[kind] = probe.fit(output_features(fit_capture, kind), fit_base['y'][:, None])
        save('fit_' + kind, output_probes[kind])
    state_probes = {
        tap: probe.fit(joint_state(fit_capture, tap), fit_base['y'][:, None]) for tap in TAPS
    }
    guard()

    test_base = task_data.make_base(256, TEST_SEED)
    test_values, _ = task_data.assemble(test_base, 'nominal')
    test_capture = continuation.collect(test_values, weights, arm)
    donor_indices = donors(test_base)
    save('test_data', test_base)
    save('test_baseline', test_capture)
    rows = []
    for tap, group, donor_name in configurations(arm):
        index = donor_indices[donor_name]
        state = continuation.splice(test_capture, tap, group, index)
        continuation_result = continuation.resume(test_values, weights, arm, tap, state)
        joint = np.concatenate([state[key] for key in ('d', 'v', 'p', 'o')], axis=1)
        score = probe.predict(state_probes[tap], joint)[:, 0]
        save(f'test_t{tap}_{group}_{donor_name}', dict(donor_index=index, state_score=score,
                                                       **continuation_result))
        prefix = test_capture['raster'][:, :max(0, tap - 15)]
        raster = np.concatenate((prefix, continuation_result['suffix_raster']), axis=1)
        capture = dict(raster=raster, counts=raster.sum(axis=1))
        row = dict(tap=tap, group=group, donor=donor_name)
        row['raster_change_rate'] = float(np.mean(np.any(raster != test_capture['raster'], axis=(1, 2))))
        row['count_change_rate'] = float(np.mean(np.any(capture['counts'] != test_capture['counts'], axis=1)))
        for kind, fitted in output_probes.items():
            prediction = probe.predict(fitted, output_features(capture, kind))[:, 0]
            row[kind] = probe.metrics(prediction, test_base['y'][index])
        rows.append(row)
        guard()
    dump(folder / 'test_rows.json', rows)
    expected = 30 if arm in ('FULL', 'NOATT') else 27
    assert len(rows) == expected
    written = sum(path.stat().st_size for path in folder.iterdir())
    return dict(arm=arm, wall_seconds=time.perf_counter() - start, configurations=expected,
                archived_bytes=written, archive_storage_bound_bytes=archive_bound)


def main():
    assert not (ROOT / 'FREEZE.json').exists(), 'frozen artifacts are read-only'
    assert not (ROOT / 'TIMING.json').exists(), 'retain the previous TIMING result before rerunning'
    start = time.perf_counter()
    workspace = ROOT.parents[1].resolve(strict=True)
    work = (workspace / 'work').resolve(strict=True)
    assert work.parent == workspace and work.is_dir()
    details = []
    free_disk = shutil.disk_usage(work).free

    def guard():
        if 2 * 9 * (time.perf_counter() - start) + 100 >= WALL_CAP:
            raise TimeoutError('fixture already reaches the scientific projection cap')
        if memory()['current_working_set_bytes'] > RAM_CAP:
            raise MemoryError('fixture exceeds 1 GiB working set')

    try:
        with tempfile.TemporaryDirectory(dir=work, prefix='readout-path-timing-') as directory:
            scratch = Path(directory).resolve(strict=True)
            assert scratch.parent == work and scratch.name.startswith('readout-path-timing-')
            for arm in ARMS:
                detail = measure_arm(arm, scratch / arm, guard)
                details.append(detail)
                print('TIMED', arm, f"{detail['wall_seconds']:.3f}s", flush=True)
        assert sum(detail['configurations'] for detail in details) == 114
        forecast = 2 * 9 * sum(detail['wall_seconds'] for detail in details) + 100
        storage_bound = 9 * sum(detail['archive_storage_bound_bytes'] for detail in details) + METADATA_ALLOWANCE
        assert forecast < WALL_CAP, 'scientific time forecast reaches cap'
        assert storage_bound < DISK_CAP, 'conservative archive estimate reaches cap'
        assert free_disk >= storage_bound, 'insufficient workspace free space for conservative archive estimate'
    except Exception as error:
        dump(ROOT / 'TIMING.json', dict(status='FAIL', reason=str(error), exception=type(error).__name__,
                                        arms=details, wall_seconds=time.perf_counter() - start))
        raise
    result = dict(status='PASS', arms=details, wall_seconds=time.perf_counter() - start,
                  forecast_seconds=forecast, formula='2 * 9 * sum(arm wall seconds) + 100',
                  cap_seconds=WALL_CAP, fixture_fit_rows=128, fixture_test_rows=256,
                  fixture_seed_fit=FIT_SEED, fixture_seed_test=TEST_SEED, fixture_seed_weights=WEIGHT_SEED,
                  projected_storage_bound_bytes=storage_bound, storage_cap_bytes=DISK_CAP,
                  metadata_allowance_bytes=METADATA_ALLOWANCE, free_bytes_before_fixture=free_disk,
                  **memory())
    dump(ROOT / 'TIMING.json', result)
    print('TIMING PASS', f'forecast {forecast:.1f}s', f'storage bound {storage_bound / 2**20:.1f} MiB', flush=True)


if __name__ == '__main__':
    main()
