"""Measure the fixed full-block budget preflight for the sealed stored executor."""

import os
from pathlib import Path
import platform
import sys
import time
import traceback
from zipfile import ZIP_STORED

import numpy as np

import block
import block_audit
import compare
import shared_model
from budget_io import ROOT, artifact_size, peak_memory, processor_load, read, sha, utc, write


OUTPUTS = ROOT.parent
OPTIMIZATION = OUTPUTS/'DEND-TAN-block-optimization-20261111'
PRIOR_BUDGET = OUTPUTS/'DEND-TAN-shared-block-budget-20261111'
EDGE = OUTPUTS/'DEND-TAN-shared-replay-fixture-20261111'
PAUSED = OUTPUTS/'DEND-TAN-evicted-key-confirm-20261111'
BASELINES = {512: PRIOR_BUDGET/'trials'/'n512_r1', 256: OPTIMIZATION/'results'/'r1_C'}
DESIGN = dict(packet_size=256, zip_mode=ZIP_STORED, examples=[512, 256], repeats=2, gap_seconds=30,
              blocks=12, science_cap_seconds=180, audit_cap_seconds=300, train_cap_seconds=600,
              memory_cap_bytes=1024**3, artifact_cap_bytes=3*1024**3//2)
THREAD_KEYS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
TRAIN_FIT_UPPER_BYTES = 2*1024**2
METADATA_RESERVE_BYTES = 64*1024**2
GAP_JITTER_LIMIT_SECONDS = .25
LOAD_SNAPSHOT_LEAD_SECONDS = 6.


def sealed_references():
    references = {}

    def require(folder, manifest_name, status, names):
        manifest_path = folder/manifest_name
        manifest = read(manifest_path)
        assert manifest['status'] == status, manifest_path
        references[manifest_path.relative_to(OUTPUTS).as_posix()] = sha(manifest_path)
        for name in names:
            path = folder/name
            digest = sha(path)
            assert manifest['files'][name]['sha256'] == digest, path
            references[path.relative_to(OUTPUTS).as_posix()] = digest

    study_entry_points = {'study.py', 'seal.py', 'verify_study.py', 'compare.py'}
    model_sources = sorted(path.name for path in OPTIMIZATION.glob('*.py') if path.name not in study_entry_points)
    old = [p.relative_to(PRIOR_BUDGET).as_posix() for p in BASELINES[512].iterdir() if p.is_file()]
    optimized = [p.relative_to(OPTIMIZATION).as_posix() for p in BASELINES[256].iterdir() if p.is_file()]
    require(PRIOR_BUDGET, 'MANIFEST.json', 'SEALED_BUDGET_ONLY', ['TIMING.json', 'WORK_STATUS.json'] + sorted(old))
    require(OPTIMIZATION, 'MANIFEST.json', 'SEALED_ENGINEERING_ONLY',
            ['VERIFY.json', 'WORK_STATUS.json'] + model_sources + sorted(optimized))
    require(EDGE, 'MANIFEST.json', 'SEALED_FIXTURE_ONLY',
            ['fixture/edge.npz', 'fixture/weights.npz', 'fixture/edge_expectations.json'] +
            [f'evaluate/edge_{rule}_separate.npz' for rule in block.RULES])
    require(PAUSED, 'PREFLIGHT_MANIFEST.json', 'SEALED_PREFLIGHT_ONLY',
            ['WORK_STATUS.json', 'PRECHECK.json', 'TIMING.json', 'trainer.py', 'learned_data.py'])
    assert {'block.py', 'block_audit.py', 'shared_model.py'} <= set(model_sources)
    for name in model_sources:
        assert sha(ROOT/name) == references[f'{OPTIMIZATION.name}/{name}'], name
    assert read(OPTIMIZATION/'VERIFY.json')['status'] == 'PASS'
    assert read(BASELINES[256]/'BASELINE_COMPARISON.json')['status'] == 'PASS'
    assert read(PRIOR_BUDGET/'WORK_STATUS.json')['budget_decision'] == 'INFEASIBLE'
    assert read(PAUSED/'WORK_STATUS.json')['status'] == 'STOPPED_BEFORE_FORMAL_BUDGET'
    return references


def freeze():
    assert not any((ROOT/name).exists() for name in ('FREEZE.json', 'TIMING.json', 'trials', 'edge', 'train', 'evaluate'))
    required = ('PROTOCOL.md', 'REPRODUCE.md', 'verify.py', 'seal.py', 'compare.py')
    assert all((ROOT/name).is_file() for name in required)
    references = sealed_references()
    original = read(PAUSED/'PRECHECK.json')['train_fixtures']
    assert len(original) == 2 and all(row['updates'] == 24 for row in original)
    estimate = DESIGN['blocks']*600/24*max(row['training_audit_seconds'] for row in original)
    assert estimate == read(PAUSED/'TIMING.json')['training_audit_forecast_seconds'] == 9.150149999186397
    sources = sorted(ROOT.glob('*.py')) + [ROOT/'PROTOCOL.md', ROOT/'REPRODUCE.md']
    write(ROOT/'FREEZE.json', dict(utc=utc(), host=platform.node(), study='STORED_FULL_BLOCK_BUDGET_ONLY',
          design=DESIGN, files={p.name: sha(p) for p in sources}, references=references,
          training_audit_forecast_seconds=estimate, training_fit_upper_bytes=TRAIN_FIT_UPPER_BYTES,
          metadata_reserve_bytes=METADATA_RESERVE_BYTES, gap_jitter_limit_seconds=GAP_JITTER_LIMIT_SECONDS,
          fixture_data_seed=4099903301, native_init_seed=4099903302,
          formal_training_fits=0, formal_evaluations=0, scientific_data_generated=False))
    print('Frozen packet-256 ZIP_STORED budget preflight.', flush=True)


class Ledger:
    def __init__(self):
        self.totals = dict(science=0., audit=0.)

    def stage(self, record, kind, work):
        limit = DESIGN[f'{kind}_cap_seconds']
        remaining = limit-self.totals[kind]
        started = time.perf_counter()
        record['stage'] = kind

        def guard(disk=False):
            if time.perf_counter()-started > remaining:
                raise TimeoutError(f'{kind} cumulative limit {limit}s')
            if peak_memory() > DESIGN['memory_cap_bytes']:
                raise MemoryError('1 GiB working set exceeded')
            if disk and artifact_size() > DESIGN['artifact_cap_bytes']:
                raise OSError('1.5 GiB study artifacts exceeded')

        try:
            guard(disk=True)
            result = work(guard)
            guard(disk=True)
            return result
        finally:
            record[f'{kind}_seconds'] = time.perf_counter()-started
            self.totals[kind] += record[f'{kind}_seconds']


def execute_edge(directory, guard):
    with np.load(EDGE/'fixture'/'edge.npz', allow_pickle=False) as archive:
        x = archive['x']
    with np.load(EDGE/'fixture'/'weights.npz', allow_pickle=False) as archive:
        w = archive['edge']
    assert x.shape == (12, 36, 6) and w.shape == (180,)
    packets = []
    for start in (0, 8):
        guard()
        cases, _ = shared_model.evaluate(x[start:start+8], w, block.confirm_data.Q)
        packets.append(cases)
    for rule in block.RULES:
        arrays = {name: np.concatenate([packet[rule][name] for packet in packets], axis=0)
                  for name in packets[0][rule]}
        block._save_npz(directory/f'edge_{rule}.npz', arrays, ZIP_STORED)
    return x, w


def measure_edge(ledger, record):
    directory = ROOT/record['directory']
    directory.mkdir()
    x, w = ledger.stage(record, 'science', lambda guard: execute_edge(directory, guard))

    def audit(guard):
        result = compare.compare_edge(directory, EDGE, x, w, guard, compression=ZIP_STORED)
        write(directory/'EDGE_COMPARISON.json', result)

    ledger.stage(record, 'audit', audit)
    record['comparison_sha256'] = sha(directory/'EDGE_COMPARISON.json')
    print('All 12 sealed edge cases verified.', flush=True)


def measure_trial(ledger, record, n):
    directory = ROOT/record['directory']
    directory.mkdir()

    def science(guard):
        block.execute(directory, n, guard, packet_size=256, compression=ZIP_STORED)

    def audit(guard):
        tick = time.perf_counter()
        block_audit.execute(directory, n, guard, packet_size=256, compression=ZIP_STORED)
        record['independent_block_seconds'] = time.perf_counter()-tick
        tick = time.perf_counter()
        result = compare.compare_trial(directory, BASELINES[n], guard, examples=n)
        write(directory/'BASELINE_COMPARISON.json', result)
        record['reference_compare_seconds'] = time.perf_counter()-tick

    ledger.stage(record, 'science', science)
    ledger.stage(record, 'audit', audit)


def execute(ledger, record, previous_finish, work):
    target = None if previous_finish is None else previous_finish+DESIGN['gap_seconds']
    if target is not None:
        time.sleep(max(0., target-LOAD_SNAPSHOT_LEAD_SECONDS-time.perf_counter()))
    record['load_before'] = processor_load()
    if target is not None:
        time.sleep(max(0., target-time.perf_counter()))
    start = time.perf_counter()
    gap = None if previous_finish is None else start-previous_finish
    record.update(utc_start=utc(), monotonic_start_seconds=start, gap_from_previous_seconds=gap,
                  target_monotonic_start_seconds=target, stage='pending')
    try:
        if gap is not None and not DESIGN['gap_seconds'] <= gap <= DESIGN['gap_seconds']+GAP_JITTER_LIMIT_SECONDS:
            raise TimeoutError(f'fixed 30s gap missed ({gap:.6f}s)')
        work()
        record['stage'] = 'complete'
    except Exception as error:
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        record.update(monotonic_finish_seconds=time.perf_counter(), utc_finish=utc())
        record['load_after'] = processor_load()
        directory = ROOT/record['directory']
        for name, file in (('block_sha256', 'BLOCK.json'), ('audit_sha256', 'AUDIT.json'),
                           ('comparison_sha256', 'BASELINE_COMPARISON.json')):
            if (directory/file).is_file():
                record[name] = sha(directory/file)
        record.update(peak_working_set_bytes=peak_memory(), artifact_bytes=artifact_size(directory))
        write(ROOT/'progress'/(Path(record['directory']).name+'.json'), record)


def predict(candidate, frozen):
    rows = candidate['trials']
    science, audit = (max(row[f'{kind}_seconds'] for row in rows) for kind in ('science', 'audit'))
    scale = 3*DESIGN['blocks']
    science_forecast = scale*science+DESIGN['gap_seconds']
    audit_forecast = scale*audit+DESIGN['gap_seconds']+frozen['training_audit_forecast_seconds']
    test_bytes = DESIGN['blocks']*max(row['artifact_bytes'] for row in rows)
    train_bytes = DESIGN['blocks']*frozen['training_fit_upper_bytes']
    forecast_bytes = test_bytes+train_bytes+frozen['metadata_reserve_bytes']
    memory = max(row['peak_working_set_bytes'] for row in rows)
    fits = (science_forecast < DESIGN['science_cap_seconds'] and audit_forecast < DESIGN['audit_cap_seconds'] and
            forecast_bytes <= DESIGN['artifact_cap_bytes'] and memory <= DESIGN['memory_cap_bytes'])
    candidate.update(science_seconds=science, audit_seconds=audit, science_forecast_seconds=science_forecast,
                     audit_forecast_seconds=audit_forecast, test_artifact_forecast_bytes=test_bytes,
                     training_artifact_forecast_bytes=train_bytes, formal_artifact_forecast_bytes=forecast_bytes,
                     peak_working_set_bytes=memory, within_budget=fits)
    print(f"{candidate['examples']} forecast: science {science_forecast:.2f}s, audit {audit_forecast:.2f}s, "
          f'formal artifacts {forecast_bytes/2**20:.2f} MiB, fits={fits}.', flush=True)


def measure():
    assert all(os.environ.get(key) == '1' for key in THREAD_KEYS)
    frozen = read(ROOT/'FREEZE.json')
    assert frozen['host'] == platform.node() and frozen['design'] == DESIGN
    for name, digest in frozen['files'].items():
        assert sha(ROOT/name) == digest, name
    for name, digest in frozen['references'].items():
        assert sha(OUTPUTS/name) == digest, name
    (ROOT/'progress').mkdir()
    (ROOT/'trials').mkdir()
    started = time.perf_counter()
    write(ROOT/'RUN_STARTED.json', dict(utc=utc(), scientific_data_generated=False))
    ledger = Ledger()
    edge, candidates, selected, failure = dict(directory='edge'), [], None, None
    try:
        execute(ledger, edge, None, lambda: measure_edge(ledger, edge))
        for n in DESIGN['examples']:
            candidate = dict(examples=n, fixed_gap_seconds=DESIGN['gap_seconds'], trials=[])
            candidates.append(candidate)
            previous_finish = None
            for repeat in (1, 2):
                trial = dict(trial=repeat, directory=f'trials/n{n}_r{repeat}')
                candidate['trials'].append(trial)
                execute(ledger, trial, previous_finish, lambda: measure_trial(ledger, trial, n))
                previous_finish = trial['monotonic_finish_seconds']
                print(f"{trial['directory']}: science {trial['science_seconds']:.3f}s, audit {trial['audit_seconds']:.3f}s.",
                      flush=True)
            predict(candidate, frozen)
            if candidate['within_budget']:
                selected = n
                break
    except Exception as error:
        failure = f'{type(error).__name__}: {error}'
        (ROOT/'FAILURE.txt').write_text(traceback.format_exc(), encoding='utf-8')
    status = 'STOPPED_ERROR' if failure else ('FEASIBLE' if selected else 'INFEASIBLE')
    wall = time.perf_counter()-started
    write(ROOT/'TIMING.json', dict(status=status, failure=failure, selected_examples_per_block=selected,
          blocks=DESIGN['blocks'], training_audit_forecast_seconds=frozen['training_audit_forecast_seconds'],
          edge=edge, candidates=candidates, scientific_data_generated=False, formal_training_fits=0,
          formal_evaluations=0, cumulative_science_seconds=ledger.totals['science'],
          cumulative_audit_seconds=ledger.totals['audit'], wall_seconds=wall,
          outside_stage_seconds=wall-sum(ledger.totals.values()), peak_working_set_bytes=peak_memory(),
          artifact_bytes=artifact_size()))
    if failure:
        raise SystemExit(failure)


if __name__ == '__main__':
    {'freeze': freeze, 'measure': measure}[sys.argv[1]]()
