"""Reconstruct every archived training update and certify the locked held-out weights."""

import hashlib
from pathlib import Path
import time

import numpy as np

from block_audit import Audit, read
import independent_confirm as training_reference
import independent_learned_data as train_data


ROOT = Path(__file__).resolve().parent
BLOCKS, UPDATES, TRAIN_N = 12, 600, 128
T, Q, ARM = 36, 28, 'LEARN'
TRAIN_CAP = 600
MEMORY_CAP, ARTIFACT_CAP = 1024**3, 3*1024**3//2
TOLERANCE = 5e-12


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


def execute(guard):
    """Return independently audited final weights in seed order and scalar diagnostics."""
    started, cpu_started = time.perf_counter(), time.process_time()
    audit = Audit(guard)
    directory = ROOT/'train'
    lock_path = directory/'TRAIN_LOCK.json'
    train_lock = read(lock_path)
    audit.check(train_lock['test_generated'] is False and train_lock['blocks'] == BLOCKS
                and train_lock['updates'] == BLOCKS*UPDATES, 'completed training lock predates test generation')
    expected_files = {'ROWS.json'} | {f'base_s{s}.npz' for s in range(BLOCKS)} | {f's{s}_{ARM}.npz' for s in range(BLOCKS)}
    audit.check(set(train_lock['files']) == expected_files, 'exact training archive lock set')
    audit.check({path.name for path in directory.iterdir() if path.is_file()} == expected_files | {'TRAIN_LOCK.json'},
                'exact complete training archive directory')
    for name, expected in train_lock['files'].items():
        audit.guard()
        audit.check(digest(directory/name) == expected, f'locked training archive {name}')
    rows = read(directory/'ROWS.json')
    index = {row['seed']: row for row in rows}
    audit.check(len(rows) == len(index) == BLOCKS and set(index) == set(range(BLOCKS)), 'all fresh training blocks')
    weights = []
    update_steps, final_steps = UPDATES*TRAIN_N*T, TRAIN_N*T
    train_audit_checks = first_last = post_update = 0
    for seed in range(BLOCKS):
        audit.guard()
        ident, row = f's{seed}_{ARM}', index[seed]
        saved = load(directory/f'base_s{seed}.npz')
        base = train_data.make_base(TRAIN_N, 4087010000+100*seed)
        audit.check(set(saved) == set(base), f'{ident}: training base fields')
        for key, value in base.items():
            audit.check(saved[key].dtype == value.dtype, f'{ident}: train data dtype {key}')
            audit.exact(saved[key], value, f'{ident}: independent train data {key}')
        x, q = train_data.assemble(base)
        audit.check(q == Q and train_data.check(base, x)['ceiling'] == 1., f'{ident}: nominal training data')
        archive = load(directory/f'{ident}.npz')
        audit.exact(archive['initial'], training_reference.initialize(4087030000+seed), f'{ident}: independently generated new weights')
        checked = training_reference.verify_training_archive(x, base['y'], q, archive)
        audit.guard()
        train_audit_checks += checked['checks']
        audit.max_error = max(audit.max_error, checked['max_numeric_error'])
        audit.check(np.isfinite(checked['max_numeric_error']) and checked['max_numeric_error'] <= TOLERANCE,
                    f'{ident}: archive verification respects {TOLERANCE} numeric tolerance')
        first_last += checked['endpoint_replays']
        post_update += 1
        audit.check(checked['updates'] == UPDATES, f'{ident}: all 600 updates')
        meta = dict(id=ident, seed=seed, arm=ARM, T=T, q=Q, examples=TRAIN_N, updates=UPDATES,
                    dense_forward_macs=156*(update_steps+final_steps), dense_backward_macs=240*update_steps)
        audit.check(set(row) == set(meta) | {'seconds', 'cpu_seconds', 'first_loss', 'last_loss', 'final_loss'},
                    f'{ident}: exact train row schema')
        for key, value in meta.items():
            audit.check(row[key] == value, f'{ident}: fixed training {key}')
        audit.check(0 < archive['step_seconds'].sum() <= row['seconds'] <= TRAIN_CAP and row['cpu_seconds'] > 0.,
                    f'{ident}: positive fit and step durations')
        audit.close(row['first_loss'], archive['history'][0, 0], f'{ident}: first objective', TOLERANCE)
        audit.close(row['last_loss'], archive['history'][-1, 0], f'{ident}: last update objective', TOLERANCE)
        audit.close(row['final_loss'], checked['final_loss'], f'{ident}: final parameter objective', TOLERANCE)
        weights.append(archive['theta'])
        print(f'Independently verified new LEARN training block {seed+1}/{BLOCKS}', flush=True)

    summary = read(ROOT/'TRAIN_SUMMARY.json')
    update_steps, final_steps = BLOCKS*update_steps, BLOCKS*final_steps
    expected = dict(blocks=BLOCKS, fits=BLOCKS, updates=BLOCKS*UPDATES, examples_per_fit=TRAIN_N,
                    update_sample_steps=update_steps, final_forward_sample_steps=final_steps,
                    dense_forward_macs=156*(update_steps+final_steps), dense_backward_macs=240*update_steps,
                    matcher_forward_macs=8*(update_steps+final_steps), matcher_backward_unit_steps=2*update_steps)
    audit.check(summary['status'] == 'COMPLETE' and summary['cap_seconds'] == TRAIN_CAP
                and 0 < summary['wall_seconds'] <= TRAIN_CAP and summary['cpu_seconds'] > 0., 'training duration and completion')
    audit.check(0 < summary['peak_working_set_bytes'] <= MEMORY_CAP, 'training peak memory cap')
    audit.check(0 < summary['artifact_bytes_at_train_end'] <= ARTIFACT_CAP, 'training artifact cap')
    for key, value in expected.items():
        audit.check(summary[key] == value, f'training ledger {key}')
    audit.check(first_last == 2*BLOCKS and post_update == BLOCKS, 'all first, last and final forward verifications')
    audit.check(audit.max_error <= TOLERANCE, 'aggregate training numerical tolerance')
    train_lock_sha256 = digest(lock_path)
    audit.guard()
    diagnostics = dict(status='PASS', checks=audit.checks+train_audit_checks, max_numeric_error=audit.max_error,
                       optimizer_updates=BLOCKS*UPDATES, training_checks=train_audit_checks,
                       first_last_training_replays=first_last, final_training_replays=post_update,
                       train_lock_sha256=train_lock_sha256, wall_seconds=time.perf_counter()-started,
                       cpu_seconds=time.process_time()-cpu_started)
    return weights, diagnostics
