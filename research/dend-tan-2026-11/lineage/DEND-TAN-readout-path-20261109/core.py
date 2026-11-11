"""Batch design, provenance helpers and resource accounting."""

import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np


R = Path(__file__).resolve().parent
TAPS = (9, 15, 19)
ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2')
DONORS = ('cue_flip', 'content_flip', 'both_flip')
ATTENTION_ARMS = ('FULL', 'NOATT')


def configurations(arm):
    yield 9, 'full', 'content_flip'
    groups = ('full', 'hidden', 'branch', 'soma_p', 'output')
    if arm in ATTENTION_ARMS:
        groups += ('history',)
    for tap, tap_groups in ((15, groups), (19, ('full', 'hidden', 'output'))):
        yield tap, 'full', 'self'
        for group in tap_groups:
            for donor in DONORS:
                yield tap, group, donor


def donors(base):
    quartet, selected, bit = (base[key] for key in ('quartet', 'selected', 'first_bit'))
    lookup = {key: index for index, key in enumerate(zip(quartet, selected, bit))}
    assert len(lookup) == len(quartet)
    indices = {'self': np.arange(len(quartet))}
    for name, flip_s, flip_b in zip(DONORS, (1, 0, 1), (0, 1, 1)):
        indices[name] = np.array([lookup[(q, s ^ flip_s, b ^ flip_b)]
                                  for q, s, b in zip(quartet, selected, bit)])
    return indices


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memory():
    class ProcessCounters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
            (key, ctypes.c_size_t) for key in (
                'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage',
                'PagefileUsage', 'PeakPagefileUsage')]
    counters = ProcessCounters()
    counters.cb = ctypes.sizeof(counters)
    assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(counters), counters.cb)
    return dict(current_working_set_bytes=counters.WorkingSetSize,
                peak_working_set_bytes=counters.PeakWorkingSetSize)


def check_lock(lock):
    for relative, digest in lock['files'].items():
        assert sha(R / relative) == digest, relative


class Budget:
    def __init__(self):
        check_lock(read(R / 'FREEZE.json'))
        assert not (R / 'RUN_STATE.json').exists(), 'one scientific run per batch'
        self.start = time.perf_counter()
        self.cpu_start = time.process_time()
        self.stage = 'setup'
        self.steps = self.baselines = self.patches = self.fits = 0
        self.mark('running')

    def mark(self, status, **other):
        dump(R / 'RUN_STATE.json', dict(status=status, stage=self.stage, utc=utc(),
                                       wall_seconds=time.perf_counter()-self.start, **other))

    def guard(self, disk=False):
        if time.perf_counter() - self.start > 840:
            raise TimeoutError('840 second scientific cap')
        if memory()['current_working_set_bytes'] > 1024**3:
            raise MemoryError('1 GiB working-set cap')
        if disk and artifact_size() > 1.5 * 1024**3:
            raise OSError('1.5 GiB artifact cap')

    def record_baseline(self, rows):
        self.baselines += 1
        self.steps += rows * 24

    def record_patch(self, rows, tap):
        self.patches += 1
        self.steps += rows * (23 - tap)

    def finish(self):
        self.guard(disk=True)
        assert (self.baselines, self.fits, self.patches, self.steps) == (72, 72, 1026, 2119680)
        dump(R / 'SCIENCE_SUMMARY.json', dict(
            status='COMPLETE', utc=utc(), models=36, neural_training_updates=0,
            probe_fits=self.fits, baseline_captures=self.baselines, patch_conditions=self.patches,
            forward_sample_steps=self.steps, dense_forward_macs=self.steps*156,
            wall_seconds=time.perf_counter()-self.start, cpu_seconds=time.process_time()-self.cpu_start,
            cap_seconds=840, artifact_bytes_at_science_end=artifact_size(), **memory()))
        self.mark('complete')


def artifact_size():
    return sum(path.stat().st_size for path in R.rglob('*') if path.is_file())
