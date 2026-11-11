"""Freeze and evaluate the saved bilinear matcher checkpoints for complete-block budget feasibility."""

import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import subprocess
import time

import block
import block_audit


ROOT = Path(__file__).resolve().parent
BLOCKS = 12
SCIENCE_CAP = 180
AUDIT_CAP = 300
MEMORY_CAP = 1024**3
ARTIFACT_CAP = int(1.5*1024**3)
FIXED_GAP_SECONDS = 30.
LOAD_SNAPSHOT_LEAD_SECONDS = 6.
THREAD_KEYS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def dump(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                             encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_lock(name):
    for relative, digest in read(name)['files'].items():
        assert sha(ROOT / relative) == digest, relative


def peak_memory():
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
             'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
             'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    counter = Counters()
    counter.cb = ctypes.sizeof(counter)
    assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(counter), counter.cb)
    return counter.PeakWorkingSetSize


def artifact_size():
    return sum(path.stat().st_size for path in ROOT.rglob('*') if path.is_file())


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _processor_load():
    snapshot = dict(utc=_utc(), status='unavailable', percentages=None)
    command = ('$ErrorActionPreference = "Stop"; '
               '$loads = @(Get-CimInstance -ClassName Win32_Processor | '
               'Select-Object -ExpandProperty LoadPercentage); '
               'ConvertTo-Json -InputObject $loads -Compress')
    try:
        result = subprocess.run(['powershell.exe', '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', command],
                                capture_output=True, text=True, check=True, timeout=5,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        loads = json.loads(result.stdout)
        if not isinstance(loads, list) or not loads:
            raise ValueError('Win32_Processor returned no processor records')
        snapshot.update(status='ok', percentages=loads)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        snapshot['error'] = f'{type(error).__name__}: {error}'
    return snapshot


def source_provenance():
    lock = read('SOURCE_LOCK.json')
    assert len(lock['files']) == 19
    verify_lock('SOURCE_LOCK.json')
    previous = ROOT/lock['previous_directory']
    blocked = ROOT/lock['blocked_directory']
    assert sha(previous/'MANIFEST.json') == lock['previous_manifest_sha256']
    assert sha(blocked/'PREFLIGHT_MANIFEST.json') == lock['blocked_manifest_sha256']
    assert sha(blocked/'TIMING.json') == lock['blocked_timing_sha256']
    assert sha(blocked/'PRECHECK.json') == lock['blocked_precheck_sha256']
    assert all(not (blocked/name).exists() for name in ('FREEZE.json', 'train', 'evaluate'))
    assert read(lock['blocked_directory']+'/WORK_STATUS.json')['status'] == 'STOPPED_BEFORE_FORMAL_BUDGET'
    for name, digest in lock['files'].items():
        assert sha(previous/name) == digest, name
    before = read(lock['blocked_directory']+'/PRECHECK.json')
    estimate = BLOCKS*600/24*max(row['training_audit_seconds'] for row in before['train_fixtures'])
    assert estimate == lock['training_audit_forecast_seconds']
    return estimate


def freeze():
    assert not any((ROOT/name).exists() for name in ('FREEZE.json', 'RUN_STATE.json', 'TIMING.json', 'trials', 'train', 'evaluate'))
    estimate = source_provenance()
    for name in ('PROTOCOL.md', 'REPRODUCE.md', 'block.py', 'block_audit.py', 'audit_timing.py'):
        assert (ROOT/name).is_file(), name
    files = {path.name: sha(path) for path in sorted(ROOT.iterdir())
             if path.is_file() and path.suffix in ('.py', '.md', '.json')}
    dump('FREEZE.json', dict(utc=utc(), study='FULL_BLOCK_BUDGET_ONLY', fixture_data_generated=False,
         blocks=BLOCKS, fixture_data_seed=4099903301, native_init_seed=4099903302, candidate_sizes=[512, 256],
         repeats_per_candidate=2, fixed_gap_seconds=FIXED_GAP_SECONDS, science_cap_seconds=SCIENCE_CAP,
         audit_cap_seconds=AUDIT_CAP, memory_cap_bytes=MEMORY_CAP, artifact_cap_bytes=ARTIFACT_CAP,
         training_audit_forecast_seconds=estimate, files=files))
    print('FROZEN complete-block budget only, fixed two measurements per attempted tier', flush=True)


def stage_guard(started, spent, cap, disk=False):
    if time.perf_counter()-started+spent > cap:
        raise TimeoutError(f'{cap} second cumulative stage limit')
    if peak_memory() > MEMORY_CAP:
        raise MemoryError('1 GiB peak working-set limit')
    if disk and artifact_size() > ARTIFACT_CAP:
        raise OSError('1.5 GiB artifact limit')


def execute_trial(trial, n, totals):
    directory = ROOT/trial['directory']
    trial['stage'] = 'science'
    tick = time.perf_counter()
    try:
        block.execute(directory, n, lambda disk=False: stage_guard(tick, totals['science'], SCIENCE_CAP, disk))
        stage_guard(tick, totals['science'], SCIENCE_CAP, disk=True)
    finally:
        trial['science_seconds'] = time.perf_counter()-tick
        totals['science'] += trial['science_seconds']
    trial['stage'] = 'audit'
    tick = time.perf_counter()
    try:
        block_audit.execute(directory, n, lambda disk=False: stage_guard(tick, totals['audit'], AUDIT_CAP, disk))
        stage_guard(tick, totals['audit'], AUDIT_CAP, disk=True)
    finally:
        trial['audit_seconds'] = time.perf_counter()-tick
        totals['audit'] += trial['audit_seconds']
    assert read(trial['directory']+'/AUDIT.json')['status'] == 'PASS'
    trial['stage'] = 'complete'


def measure_candidate(candidate, estimate, totals, all_candidates):
    n = candidate['examples']
    trials = candidate['trials']
    previous_finish = None
    for number in (1, 2):
        if previous_finish is not None:
            delay = previous_finish+FIXED_GAP_SECONDS-LOAD_SNAPSHOT_LEAD_SECONDS-time.perf_counter()
            if delay > 0.:
                time.sleep(delay)
        before = _processor_load()
        if previous_finish is not None:
            delay = previous_finish+FIXED_GAP_SECONDS-time.perf_counter()
            if delay > 0.:
                time.sleep(delay)
        start_utc, start = utc(), time.perf_counter()
        trial = dict(trial=number, directory=f'trials/n{n}_r{number}', utc_start=start_utc,
                     monotonic_start_seconds=start, load_before=before,
                     gap_from_previous_seconds=None if previous_finish is None else start-previous_finish)
        trials.append(trial)
        try:
            execute_trial(trial, n, totals)
        finally:
            previous_finish = time.perf_counter()
            trial.update(utc_finish=utc(), monotonic_finish_seconds=previous_finish, load_after=_processor_load())
            directory = ROOT/trial['directory']
            trial['peak_working_set_bytes'] = peak_memory()
            trial['artifact_bytes'] = sum(path.stat().st_size for path in directory.rglob('*') if path.is_file())
            for field, name in (('block_sha256', 'BLOCK.json'), ('audit_sha256', 'AUDIT.json')):
                if (directory/name).is_file():
                    trial[field] = sha(directory/name)
            dump('PROGRESS.json', dict(scientific_data_generated=False, candidates=all_candidates))
        print(f"{n} examples, fixed trial {number}/2: science {trial['science_seconds']:.3f}s, "
              f"audit {trial['audit_seconds']:.3f}s", flush=True)
    science, audit = (max(row[f'{name}_seconds'] for row in trials) for name in ('science', 'audit'))
    science_forecast, audit_forecast = 36*science+30, 36*audit+30+estimate
    candidate.update(science_seconds=science, audit_seconds=audit, science_forecast_seconds=science_forecast,
                     audit_forecast_seconds=audit_forecast, within_budget=science_forecast < SCIENCE_CAP and audit_forecast < AUDIT_CAP)
    print(f'{n} forecast: science {science_forecast:.2f}s, audit {audit_forecast:.2f}s, '
          f"within budget={candidate['within_budget']}", flush=True)


def run():
    assert all(os.environ.get(key) == '1' for key in THREAD_KEYS)
    assert not any((ROOT/name).exists() for name in ('RUN_STATE.json', 'TIMING.json', 'trials', 'train', 'evaluate'))
    verify_lock('FREEZE.json')
    frozen = read('FREEZE.json')
    estimate = frozen['training_audit_forecast_seconds']
    started = time.perf_counter()
    candidates = []
    totals = dict(science=0., audit=0.)
    selected = None
    dump('RUN_STATE.json', dict(status='running_budget', utc=utc(), scientific_data_generated=False))
    try:
        (ROOT/'trials').mkdir()
        for n in (512, 256):
            candidate = dict(examples=n, fixed_gap_seconds=FIXED_GAP_SECONDS, trials=[])
            candidates.append(candidate)
            measure_candidate(candidate, estimate, totals, candidates)
            if candidate['within_budget']:
                selected = n
                break
        dump('TIMING.json', dict(status='FEASIBLE' if selected else 'INFEASIBLE',
             selected_examples_per_block=selected, blocks=BLOCKS, training_audit_forecast_seconds=estimate,
             candidates=candidates, scientific_data_generated=False, formal_training_fits=0, formal_evaluations=0,
             cumulative_science_seconds=totals['science'], cumulative_audit_seconds=totals['audit'],
             wall_seconds=time.perf_counter()-started, peak_working_set_bytes=peak_memory(), artifact_bytes=artifact_size()))
        dump('RUN_STATE.json', dict(status='complete_budget', utc=utc(), selected_examples_per_block=selected,
                                   scientific_data_generated=False))
    except Exception as exc:
        stopped = dict(status='stopped_budget_error', utc=utc(), exception=type(exc).__name__, reason=str(exc),
                       selected_examples_per_block=None, scientific_data_generated=False, candidates=candidates,
                       cumulative_science_seconds=totals['science'], cumulative_audit_seconds=totals['audit'])
        dump('TIMING.json', stopped)
        dump('RUN_STATE.json', stopped)
        raise


if __name__ == '__main__':
    {'freeze': freeze, 'measure': run}[sys.argv[1]]()
