"""Bounded, counterbalanced engineering study on the saved synthetic fixtures."""

import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from zipfile import ZIP_DEFLATED, ZIP_STORED

import numpy as np

import block
import block_audit
import compare
import shared_model


ROOT = Path(__file__).resolve().parent
BUDGET = ROOT.parent / 'DEND-TAN-shared-block-budget-20261111'
EDGE = ROOT.parent / 'DEND-TAN-shared-replay-fixture-20261111'
BASELINE = BUDGET / 'trials' / 'n256_r1'
PAUSED = ROOT.parent / 'DEND-TAN-evicted-key-confirm-20261111' / 'WORK_STATUS.json'
VARIANTS = {'A': (32, ZIP_DEFLATED), 'B': (256, ZIP_DEFLATED), 'C': (256, ZIP_STORED)}
ROUNDS = ('ABC', 'CBA')
EXAMPLES = 256
CAPS = {'science': 180., 'audit': 300.}
MEMORY_CAP = 1024**3
ARTIFACT_CAP = 3*1024**3//2
THREAD_KEYS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def peak_memory():
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
             'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
             'PagefileUsage', 'PeakPagefileUsage')]

    counter = Counters()
    counter.cb = ctypes.sizeof(counter)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(counter), counter.cb):
        raise OSError('PeakWorkingSetSize unavailable')
    return counter.PeakWorkingSetSize


def artifact_size(directory=ROOT):
    return sum(path.stat().st_size for path in directory.rglob('*') if path.is_file())


def processor_load():
    snapshot = dict(utc_start=utc(), status='unavailable', percentages=None)
    command = ('$ErrorActionPreference = "Stop"; '
               '$loads = @(Get-CimInstance -ClassName Win32_Processor | Select-Object -ExpandProperty LoadPercentage); '
               'ConvertTo-Json -InputObject $loads -Compress')
    try:
        result = subprocess.run(['powershell.exe', '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', command],
                                capture_output=True, text=True, check=True, timeout=5,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        loads = json.loads(result.stdout)
        if not isinstance(loads, list) or not loads:
            raise ValueError('Win32_Processor returned no processor records')
        if any(type(value) is not int or not 0 <= value <= 100 for value in loads):
            raise ValueError('Win32_Processor returned invalid LoadPercentage')
        snapshot.update(status='ok', percentages=loads)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        snapshot['error'] = f'{type(error).__name__}: {error}'
    snapshot['utc_finish'] = utc()
    return snapshot


def sealed_inputs():
    references = {}
    for folder, status, names in (
        (BUDGET, 'SEALED_BUDGET_ONLY', ['TIMING.json', 'TIMING_AUDIT.json', 'WORK_STATUS.json'] +
         [path.relative_to(BUDGET).as_posix() for path in sorted(BASELINE.iterdir()) if path.is_file()]),
        (EDGE, 'SEALED_FIXTURE_ONLY', ['fixture/edge.npz', 'fixture/weights.npz', 'fixture/edge_expectations.json'] +
         [f'evaluate/edge_{rule}_separate.npz' for rule in block.RULES]),
    ):
        manifest_path = folder / 'MANIFEST.json'
        manifest = read(manifest_path)
        assert manifest['status'] == status
        for name in names:
            digest = sha(folder / name)
            assert digest == manifest['files'][name]['sha256'], name
            references[f'{folder.name}/{name}'] = digest
        references[f'{folder.name}/MANIFEST.json'] = sha(manifest_path)
    original = read(BUDGET / 'MANIFEST.json')['files']
    for path in ROOT.glob('*.py'):
        if path.name in original and path.name not in {'block.py', 'block_audit.py', 'shared_model.py'}:
            assert sha(path) == original[path.name]['sha256'], path.name
    assert read(PAUSED)['status'] == 'STOPPED_BEFORE_FORMAL_BUDGET'
    references[PAUSED.parent.name + '/' + PAUSED.name] = sha(PAUSED)
    return references


def design():
    return dict(examples=EXAMPLES, edge_examples=12, edge_packet=8, edge_order=list(VARIANTS),
                variants={name: dict(packet_size=packet, zip_mode=mode) for name, (packet, mode) in VARIANTS.items()},
                rounds=[list(order) for order in ROUNDS], caps_seconds=CAPS,
                trial_admission_reserve_seconds=25., edge_admission_reserve_seconds=5.,
                memory_cap_bytes=MEMORY_CAP, artifact_cap_bytes=ARTIFACT_CAP)


def freeze():
    assert not any((ROOT/name).exists() for name in ('FREEZE.json', 'results', 'SUMMARY.json'))
    files = sorted(ROOT.glob('*.py')) + [ROOT/'PROTOCOL.md', ROOT/'REPRODUCE.md']
    write(ROOT/'FREEZE.json', dict(utc=utc(), scope='SYNTHETIC_ENGINEERING_ONLY', host=platform.node(),
          design=design(), references=sealed_inputs(), files={p.name: sha(p) for p in files},
          formal_training_fits=0, formal_evaluations=0))
    print('Frozen fixed A/B/C engineering design.', flush=True)


class Budget:
    def __init__(self):
        self.totals = dict(science=0., audit=0.)

    def run(self, record, stage, reserve, work):
        started = time.perf_counter()
        remaining = CAPS[stage] - self.totals[stage]
        if remaining < reserve:
            raise TimeoutError(f'{stage}: {remaining:.3f}s remains, less than admission reserve {reserve}s')

        def guard(disk=False):
            if time.perf_counter()-started > remaining:
                raise TimeoutError(f'{stage}: cumulative cap {CAPS[stage]}s')
            if peak_memory() > MEMORY_CAP:
                raise MemoryError('peak working set exceeds 1 GiB')
            if disk and artifact_size() > ARTIFACT_CAP:
                raise OSError('artifact size exceeds 1.5 GiB')

        record['stage'] = stage
        record[f'{stage}_utc_start'] = utc()
        record[f'{stage}_monotonic_start'] = started
        try:
            guard(disk=True)
            result = work(guard)
            guard(disk=True)
            return result
        finally:
            finished = time.perf_counter()
            elapsed = finished-started
            self.totals[stage] += elapsed
            record[f'{stage}_seconds'] = elapsed
            record[f'{stage}_utc_finish'] = utc()
            record[f'{stage}_monotonic_finish'] = finished


def write_edge(directory, x, weights, compression, guard):
    packets = []
    for left in range(0, len(x), 8):
        guard()
        cases, _ = shared_model.evaluate(x[left:left+8], weights, block.confirm_data.Q)
        packets.append(cases)
    for rule in block.RULES:
        arrays = {field: np.concatenate([cases[rule][field] for cases in packets], axis=0)
                  for field in packets[0][rule]}
        block._save_npz(directory/f'edge_{rule}.npz', arrays, compression)


def run_check(folder, baseline, guard):
    started = time.perf_counter()
    comparison = compare.compare_trial(folder, baseline, guard)
    write(folder/'BASELINE_COMPARISON.json', comparison)
    return time.perf_counter()-started


def audit_trial(directory, packet, compression, record, guard):
    start = time.perf_counter()
    block_audit.execute(directory, EXAMPLES, guard, packet_size=packet, compression=compression)
    record['independent_block_seconds'] = time.perf_counter()-start
    record['baseline_comparison_seconds'] = run_check(directory, BASELINE, guard)


def perform(budget, record, folder, science, audit, reserve):
    folder.mkdir()
    record['stage'] = 'pending'
    record['load_before'] = processor_load()
    record['utc_start'] = utc()
    record['monotonic_start'] = time.perf_counter()
    try:
        budget.run(record, 'science', reserve, science)
        budget.run(record, 'audit', reserve, audit)
        record['stage'] = 'complete'
    except Exception as error:
        record['error'] = f'{type(error).__name__}: {error}'
        (folder/'FAILURE.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        record['monotonic_finish'] = time.perf_counter()
        record['utc_finish'] = utc()
        record['load_after'] = processor_load()
        record['peak_working_set_bytes'] = peak_memory()
        record['trial_bytes'] = artifact_size(folder)
        write(folder/'TIMING.json', record)


def run():
    frozen = read(ROOT/'FREEZE.json')
    assert frozen['design'] == design() and frozen['host'] == platform.node()
    assert all(os.environ.get(key) == '1' for key in THREAD_KEYS)
    for name, digest in frozen['files'].items():
        assert sha(ROOT/name) == digest, name
    for name, digest in frozen['references'].items():
        assert sha(ROOT.parent/name) == digest, name
    results = ROOT/'results'
    results.mkdir()
    budget, trials, edges = Budget(), [], []
    wall_start = time.perf_counter()
    started_utc = utc()
    snapshot_seconds = 0.
    failure = None
    try:
        with np.load(EDGE/'fixture'/'edge.npz', allow_pickle=False) as saved:
            x = saved['x']
        with np.load(EDGE/'fixture'/'weights.npz', allow_pickle=False) as saved:
            weights = saved['edge']
        assert x.shape == (12, 36, 6) and weights.shape == (180,)
        for name, (_, compression) in VARIANTS.items():
            folder = results/f'edge_{name}'
            record = dict(variant=name, directory=folder.name)
            edges.append(record)

            def audit_edge(guard):
                comparison = compare.compare_edge(folder, EDGE, x, weights, guard, compression=compression)
                write(folder/'AUDIT.json', comparison)

            outer_start = time.perf_counter()
            perform(budget, record, folder, lambda guard: write_edge(folder, x, weights, compression, guard),
                    audit_edge, 5.)
            snapshot_seconds += time.perf_counter()-outer_start-(record['monotonic_finish']-record['monotonic_start'])
            print(f'Edge {name} verified.', flush=True)

        for repeat, order in enumerate(ROUNDS, 1):
            for name in order:
                packet, compression = VARIANTS[name]
                folder = results/f'r{repeat}_{name}'
                record = dict(repeat=repeat, variant=name, directory=folder.name, packet_size=packet, zip_mode=compression)
                trials.append(record)
                outer_start = time.perf_counter()
                perform(budget, record, folder,
                        lambda guard: block.execute(folder, EXAMPLES, guard, packet_size=packet, compression=compression),
                        lambda guard: audit_trial(folder, packet, compression, record, guard), 25.)
                snapshot_seconds += time.perf_counter()-outer_start-(record['monotonic_finish']-record['monotonic_start'])
                print(f'{folder.name}: science {record["science_seconds"]:.3f}s; audit {record["audit_seconds"]:.3f}s.', flush=True)
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
        (ROOT/'FAILURE.txt').write_text(traceback.format_exc(), encoding='utf-8')

    wall_seconds = time.perf_counter()-wall_start
    summary = dict(status='STOPPED' if failure else 'COMPLETE_ENGINEERING_ONLY', failure=failure, utc_start=started_utc,
                   utc_finish=utc(), wall_seconds=wall_seconds, outside_stage_seconds=wall_seconds-sum(budget.totals.values()),
                   completed_trial_observation_and_bookkeeping_seconds=snapshot_seconds,
                   cumulative_stage_seconds=budget.totals, peak_working_set_bytes=peak_memory(),
                   artifact_bytes=artifact_size(), edges=edges, trials=trials,
                   formal_training_fits=0, formal_evaluations=0, formal_data_generated=False)
    write(ROOT/'SUMMARY.json', summary)
    if failure:
        raise SystemExit(failure)


if __name__ == '__main__':
    {'freeze': freeze, 'run': run}[sys.argv[1]]()
