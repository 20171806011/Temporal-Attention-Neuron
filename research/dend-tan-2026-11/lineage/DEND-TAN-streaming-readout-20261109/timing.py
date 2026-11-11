"""Project the scientific batch from four full-size independent fixture arms."""

from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

import continuation
import population
from core import dump, memory
from preflight import ARMS, fixture_input
from run import evaluate_model, fit_model, make_permutations


ROOT = Path(__file__).resolve().parent
FIT_ROWS, TEST_ROWS = 512, 1024
FIT_SEED, TEST_SEED = 4099900801, 4099900802
WEIGHT_SEED = 4099900803
FIT_PERM_SEED, TEST_PERM_SEED = 4099900804, 4099900805
WALL_CAP = 840
RAM_CAP = 1024**3
DISK_CAP = int(1.5 * 1024**3)
ARCHIVE_ALLOWANCE = 64 * 1024
METADATA_ALLOWANCE = 64 * 1024**2


def measure_arm(arm, folder, guard):
    start = time.perf_counter()
    folder.mkdir()
    archive_bound = 0

    def save(name, arrays, variable_pair_rows=0):
        nonlocal archive_bound
        path = folder / (name + '.npz')
        np.savez_compressed(path, **arrays)
        bound = sum(np.asarray(value).nbytes for value in arrays.values()) + ARCHIVE_ALLOWANCE
        assert path.stat().st_size <= bound
        # At most four opposing pairs arise per quartet. Reserve a further
        # row budget for matched indices and any pair scores in the archive.
        archive_bound += bound + variable_pair_rows * 16 * 8

    weights = population.initialize(WEIGHT_SEED)
    fit_base, fit_x = fixture_input(FIT_ROWS, FIT_SEED)
    fit_orders = make_permutations(FIT_ROWS, FIT_PERM_SEED)
    fit_native = continuation.collect(fit_x, weights, arm, taps=())
    save('fit_data', fit_base)
    save('fit_permutations', dict(permutations=fit_orders))
    save('fit_native', fit_native)
    fit = fit_model(fit_native['raster'], fit_base, fit_orders)
    save('fitted', fit)
    guard()

    test_base, test_x = fixture_input(TEST_ROWS, TEST_SEED)
    test_orders = make_permutations(TEST_ROWS, TEST_PERM_SEED)
    test_native = continuation.collect(test_x, weights, arm, taps=())
    save('test_data', test_base)
    save('test_permutations', dict(permutations=test_orders))
    save('test_native', test_native)
    scores, test_row = evaluate_model(test_native, test_base, test_orders, fit)
    elapsed = scores['timing_ns']
    assert elapsed.shape == (3, 64) and elapsed.dtype == np.int64 and np.all(elapsed > 0)
    save('test_scores', scores, variable_pair_rows=TEST_ROWS)
    dump(folder / 'test_row.json', test_row)
    guard()

    written = sum(path.stat().st_size for path in folder.iterdir())
    return dict(arm=arm, wall_seconds=time.perf_counter() - start, native_captures=2,
                readout_fits=11, timed_batches_per_rule=64,
                archived_bytes=written, archive_storage_bound_bytes=archive_bound)


def main():
    assert not (ROOT / 'FREEZE.json').exists(), 'frozen artifacts are read-only'
    assert not (ROOT / 'TIMING.json').exists(), 'retain the previous TIMING result before rerunning'
    start = time.perf_counter()
    details = []

    def guard():
        if 2 * 9 * (time.perf_counter() - start) + 100 >= WALL_CAP:
            raise TimeoutError('fixture already reaches the scientific projection cap')
        if memory()['current_working_set_bytes'] > RAM_CAP:
            raise MemoryError('fixture exceeds 1 GiB working set')

    try:
        workspace = ROOT.parents[1].resolve(strict=True)
        work = (workspace / 'work').resolve(strict=True)
        assert work.parent == workspace and work.is_dir()
        free_disk = shutil.disk_usage(work).free
        with tempfile.TemporaryDirectory(dir=work, prefix='streaming-readout-fixture-') as directory:
            scratch = Path(directory).resolve(strict=True)
            assert scratch.parent == work and scratch.name.startswith('streaming-readout-fixture-')
            for arm in ARMS:
                details.append(measure_arm(arm, scratch / arm, guard))
                print('TIMED', arm, f"{details[-1]['wall_seconds']:.3f}s", flush=True)
        assert sum(arm['readout_fits'] for arm in details) == 44
        forecast = 2 * 9 * sum(arm['wall_seconds'] for arm in details) + 100
        storage_bound = 9 * sum(arm['archive_storage_bound_bytes'] for arm in details) + METADATA_ALLOWANCE
        assert forecast < WALL_CAP, 'scientific time forecast reaches cap'
        assert storage_bound < DISK_CAP, 'conservative archive estimate reaches cap'
        assert free_disk >= storage_bound, 'insufficient workspace free space for conservative archive estimate'
    except Exception as error:
        dump(ROOT / 'TIMING.json', dict(status='FAIL', reason=str(error), exception=type(error).__name__,
                                        arms=details, wall_seconds=time.perf_counter() - start))
        raise

    result = dict(status='PASS', arms=details, wall_seconds=time.perf_counter() - start,
                  forecast_seconds=forecast, formula='2 * 9 * sum(arm wall seconds) + 100',
                  cap_seconds=WALL_CAP, fixture_fit_rows=FIT_ROWS, fixture_test_rows=TEST_ROWS,
                  fixture_seed_fit=FIT_SEED, fixture_seed_test=TEST_SEED, fixture_seed_weights=WEIGHT_SEED,
                  fixture_seed_fit_permutations=FIT_PERM_SEED, fixture_seed_test_permutations=TEST_PERM_SEED,
                  projected_storage_bound_bytes=storage_bound, storage_cap_bytes=DISK_CAP,
                  metadata_allowance_bytes=METADATA_ALLOWANCE, free_bytes_before_fixture=free_disk,
                  **memory())
    dump(ROOT / 'TIMING.json', result)
    print('TIMING PASS', f'forecast {forecast:.1f}s', f'storage bound {storage_bound / 2**20:.1f} MiB', flush=True)


if __name__ == '__main__':
    main()
