"""Budget forecast from independent, full-size fixture arms and scratch writes."""

from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

import capture
import population
import probe
import task_data
from core import CONDITIONS, R, dump, memory, native


ARMS = ("DEND", "FULL", "NOATT", "GAIN2")
FIT_SEED = 4099900201
TEST_SEED = 4099900202
FIT_PERMUTATION_SEED = 4099900203
TEST_PERMUTATION_SEED = 4099900204
WEIGHT_SEED = 4099900205
WALL_CAP = 840
DISK_CAP = int(1.5 * 1024**3)
RAM_CAP = 1024**3
ARCHIVE_ALLOWANCE = 64 * 1024
METADATA_ALLOWANCE = 64 * 1024**2


def measure_arm(arm, folder, guard):
    start = time.perf_counter()
    folder.mkdir()
    archive_bound = 0

    def save(name, arrays):
        nonlocal archive_bound
        path = folder / (name + ".npz")
        np.savez_compressed(path, **arrays)
        # Uncompressed payload plus generous NPY/ZIP headers and DEFLATE overhead.
        bound = sum(np.asarray(value).nbytes for value in arrays.values()) + ARCHIVE_ALLOWANCE
        assert path.stat().st_size <= bound
        archive_bound += bound

    weights = population.initialize(WEIGHT_SEED)
    fit_data = task_data.make_base(256, FIT_SEED)
    fit_data["permuted_y"] = probe.permutations(fit_data, FIT_PERMUTATION_SEED)
    save("fit", fit_data)
    fit_input, _ = task_data.assemble(fit_data, "nominal")
    fit_capture = capture.capture(fit_input, weights, arm)
    save("fit_capture", fit_capture)
    labels = np.column_stack((fit_data["y"], fit_data["permuted_y"]))
    configurations = capture.configurations(arm)
    rows = []
    for group, tap in configurations:
        fit = probe.fit(capture.features(fit_capture, group, tap), labels)
        ident = f"{group}_t{tap}"
        save(f"{ident}_fit", fit)
        rows.append(dict(group=group, time=tap, **probe.metrics(fit["fit_scores"][:, 0], labels[:, 0])))
        guard()
    dump(folder / "fit_rows.json", rows)

    test_data = task_data.make_base(512, TEST_SEED)
    test_data["permuted_y"] = probe.permutations(test_data, TEST_PERMUTATION_SEED)
    save("test", test_data)
    captures = []
    native_rows = []
    for condition in CONDITIONS:
        test_input, _ = task_data.assemble(test_data, condition)
        taps = capture.capture(test_input, weights, arm)
        captures.append(taps)
        save(f"test_capture_{condition}", taps)
        native_rows.append(dict(condition=condition, **native(taps["counts"], test_data["y"])))
        guard()
    rows = []
    for group, tap in configurations:
        ident = f"{group}_t{tap}"
        with np.load(folder / (ident + "_fit.npz"), allow_pickle=False) as stored:
            fit = dict(stored)
        scores = np.stack([probe.predict(fit, capture.features(taps, group, tap)) for taps in captures])
        save(f"{ident}_test", {"test_scores": scores})
        for condition, score in zip(CONDITIONS, scores):
            row = dict(condition=condition, group=group, time=tap,
                       **probe.metrics(score[:, 0], test_data["y"]))
            if condition == "nominal":
                row["fit_shuffle"] = [probe.metrics(score[:, j + 1], test_data["y"]) for j in range(20)]
                row["test_shuffle"] = [probe.metrics(score[:, 0], test_data["permuted_y"][:, j])
                                       for j in range(20)]
            rows.append(row)
        guard()
    dump(folder / "test_rows.json", rows)
    dump(folder / "native_rows.json", native_rows)
    bytes_written = sum(path.stat().st_size for path in folder.iterdir())
    return dict(arm=arm, wall_seconds=time.perf_counter() - start, configurations=len(configurations),
                archived_bytes=bytes_written, archive_storage_bound_bytes=archive_bound)


def main():
    assert not (R / "FREEZE.json").exists(), "frozen artifacts are read-only"
    start = time.perf_counter()
    workspace = R.parents[1].resolve(strict=True)
    work = (workspace / "work").resolve(strict=True)
    assert work.parent == workspace and work.is_dir()
    details = []
    free_disk = shutil.disk_usage(work).free

    def guard():
        if 2 * 9 * (time.perf_counter() - start) + 100 >= WALL_CAP:
            raise TimeoutError("fixture already reaches the scientific projection cap")
        if memory()["current_working_set_bytes"] > RAM_CAP:
            raise MemoryError("fixture exceeds 1 GiB working set")

    try:
        with tempfile.TemporaryDirectory(dir=work, prefix="frozen-probes-timing-") as directory:
            scratch = Path(directory).resolve(strict=True)
            assert scratch.parent == work and scratch.name.startswith("frozen-probes-timing-")
            for arm in ARMS:
                detail = measure_arm(arm, scratch / arm, guard)
                details.append(detail)
                print("TIMED", arm, f"{detail['wall_seconds']:.3f}s", flush=True)
        forecast = 2 * 9 * sum(detail["wall_seconds"] for detail in details) + 100
        storage_bound = 9 * sum(detail["archive_storage_bound_bytes"] for detail in details) + METADATA_ALLOWANCE
        assert forecast < WALL_CAP, "scientific time forecast reaches cap"
        assert storage_bound < DISK_CAP, "conservative archive estimate reaches cap"
        assert free_disk >= storage_bound, "insufficient workspace free space for conservative archive estimate"
    except Exception as error:
        dump(R / "TIMING.json", dict(status="FAIL", reason=str(error), exception=type(error).__name__,
                                    arms=details, wall_seconds=time.perf_counter() - start))
        raise
    result = dict(status="PASS", arms=details, wall_seconds=time.perf_counter() - start,
                  forecast_seconds=forecast, formula="2 * 9 * sum(arm wall seconds) + 100",
                  cap_seconds=WALL_CAP, fixture_fit_rows=256, fixture_test_rows=512,
                  fixture_seed_fit=FIT_SEED, fixture_seed_test=TEST_SEED,
                  fixture_seed_fit_permutation=FIT_PERMUTATION_SEED,
                  fixture_seed_test_permutation=TEST_PERMUTATION_SEED, fixture_seed_weights=WEIGHT_SEED,
                  projected_storage_bound_bytes=storage_bound, storage_cap_bytes=DISK_CAP,
                  metadata_allowance_bytes=METADATA_ALLOWANCE, free_bytes_before_fixture=free_disk,
                  **memory())
    dump(R / "TIMING.json", result)
    print("TIMING PASS", f"forecast {forecast:.1f}s", f"storage bound {storage_bound / 2**20:.1f} MiB",
          flush=True)


if __name__ == "__main__":
    main()
