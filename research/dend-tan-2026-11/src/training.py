"""New-fit archive and the test entry gate from the frozen confirmation recipe."""

import os
import time

import numpy as np

import learned_data
import population
import trainer
from budget_io import artifact_size, peak_memory, read, sha, utc, write


BLOCKS, UPDATES, TRAIN_N, T, Q = 12, 600, 128, 36, 28


def expected_archives(fits):
    return {'ROWS.json'} | {f'base_s{s}.npz' for s in range(fits)} | {f's{s}_LEARN.npz' for s in range(fits)}


def lock(root, fits, updates, guard):
    folder = root/'train'
    names = expected_archives(fits)
    assert {path.name for path in folder.iterdir()} == names
    files = {}
    for name in sorted(names):
        guard()
        files[name] = sha(folder/name)
    target = folder/'TRAIN_LOCK.json'
    write(target, dict(utc=utc(), test_generated=False, blocks=fits, updates=fits*updates, files=files))
    guard(disk=True)
    return sha(target)


def open_test(root, lock_digest, fits, updates, guard):
    """Validate all locked artifacts before permitting a caller to generate any test input."""
    marker = root/'TEST_START.json'
    assert not marker.exists() and not (root/'evaluate').exists()
    folder = root/'train'
    target = folder/'TRAIN_LOCK.json'
    assert sha(target) == lock_digest, 'training lock digest'
    sealed, summary = read(target), read(root/'TRAIN_SUMMARY.json')
    assert sealed['test_generated'] is False and sealed['blocks'] == fits and sealed['updates'] == fits*updates
    assert summary['status'] == 'COMPLETE' and summary['fits'] == fits and summary['updates'] == fits*updates
    names = expected_archives(fits)
    assert set(sealed['files']) == names and {path.name for path in folder.iterdir()} == names | {'TRAIN_LOCK.json'}
    for name, digest in sealed['files'].items():
        guard()
        assert sha(folder/name) == digest, name
    weights = []
    for seed in range(fits):
        with np.load(folder/f's{seed}_LEARN.npz', allow_pickle=False) as archive:
            theta = archive['theta']
            assert theta.shape == (180,) and np.isfinite(theta).all()
            assert archive['gradients'].shape == (updates, 180)
            weights.append(theta)
    guard(disk=True)
    write(marker, dict(status='READY', utc=utc(), blocks=fits, test_generated=False, train_lock_sha256=lock_digest))
    return weights


def objective(counts, y, w):
    z = .5*(counts[:, 1].astype(float)-counts[:, 0].astype(float))
    return float(np.mean(np.logaddexp(0., (1-2*y)*z))+.0001*np.mean(w*w))


def execute(root, guard, checkpoint):
    start, cpu = time.perf_counter(), time.process_time()
    assert not any((root/name).exists() for name in ('train', 'evaluate', 'TEST_START.json'))
    folder = root/'train'
    folder.mkdir()
    rows = []
    for seed in range(BLOCKS):
        checkpoint(seed, fits=len(rows))
        guard(disk=True)
        ident = f's{seed}_LEARN'
        base = learned_data.make_base(TRAIN_N, 4087010000+100*seed)
        np.savez_compressed(folder/f'base_s{seed}.npz', **base)
        x, q = learned_data.assemble(base)
        initial = np.r_[population.initialize(4087030000+seed, 6), np.zeros(4)]
        assert x.shape == (TRAIN_N, T, 6) and q == Q and initial.shape == (180,)
        tick, fit_cpu = time.perf_counter(), time.process_time()
        archive = trainer.fit_one(x, base['y'], q, initial, UPDATES, guard, folder/f'{ident}.npz')
        assert archive['gradients'].shape == archive['raw_gradients'].shape == (UPDATES, 180)
        assert archive['counts'].shape == (UPDATES, TRAIN_N, 2)
        sample_steps = UPDATES*TRAIN_N*T
        rows.append(dict(id=ident, seed=seed, arm='LEARN', T=T, q=q, examples=TRAIN_N, updates=UPDATES,
                         seconds=time.perf_counter()-tick, cpu_seconds=time.process_time()-fit_cpu,
                         first_loss=float(archive['history'][0, 0]), last_loss=float(archive['history'][-1, 0]),
                         final_loss=objective(archive['final_counts'], base['y'], archive['theta']),
                         dense_forward_macs=156*(sample_steps+TRAIN_N*T), dense_backward_macs=240*sample_steps))
        pending = folder/'ROWS.pending.json'
        write(pending, rows)
        os.replace(pending, folder/'ROWS.json')
        checkpoint(seed, fits=len(rows))
        guard(disk=True)
        print(f'Training block {seed+1}/{BLOCKS} archived, held-out data ungenerated.', flush=True)

    assert not (root/'evaluate').exists()
    lock_digest = lock(root, BLOCKS, UPDATES, guard)
    step_updates, step_final = BLOCKS*UPDATES*TRAIN_N*T, BLOCKS*TRAIN_N*T
    write(root/'TRAIN_SUMMARY.json', dict(status='COMPLETE', blocks=BLOCKS, fits=BLOCKS, updates=BLOCKS*UPDATES,
          examples_per_fit=TRAIN_N, update_sample_steps=step_updates, final_forward_sample_steps=step_final,
          dense_forward_macs=156*(step_updates+step_final), dense_backward_macs=240*step_updates,
          matcher_forward_macs=8*(step_updates+step_final), matcher_backward_unit_steps=2*step_updates,
          wall_seconds=time.perf_counter()-start, cpu_seconds=time.process_time()-cpu, cap_seconds=600,
          peak_working_set_bytes=peak_memory(), artifact_bytes_at_train_end=artifact_size(root)))
    guard(disk=True)
    return lock_digest
