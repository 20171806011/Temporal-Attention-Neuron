"""Frozen hold-path design, locks, and explicit replay resource ledger."""

import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


R = Path(__file__).resolve().parent
ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2')
ATTENTION_ARMS = ('FULL', 'NOATT')
STATE_TAPS = {0: (8,), 16: (8, 16, 24)}
CAPTURE_TAPS = {0: (8,), 16: (8, 16, 24, 27)}
REPLAY = (
    ('self_blank', 9, 25, None), ('mean_blank', 9, 25, None),
    ('flip_first_blank', 9, 25, 0), ('flip_second_blank', 9, 25, 1),
    ('mean_early', 9, 17, None), ('mean_late', 17, 25, None), ('mean_query', 28, 36, None),
)


def groups(arm):
    fields = ('branch', 'soma', 'previous', 'soma_p', 'output', 'other', 'full')
    return fields + (('history', 'branch_history') if arm in ATTENTION_ARMS else ())


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memory():
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    result = Counters()
    result.cb = ctypes.sizeof(result)
    assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(result), result.cb)
    return dict(current_working_set_bytes=result.WorkingSetSize, peak_working_set_bytes=result.PeakWorkingSetSize)


def artifact_size():
    return sum(path.stat().st_size for path in R.rglob('*') if path.is_file())


def verify_lock(name):
    for relative, digest in read(R / name)['files'].items():
        assert sha(R / relative) == digest, relative


class Budget:
    def __init__(self):
        verify_lock('FREEZE.json')
        assert not (R / 'RUN_STATE.json').exists(), 'one scientific run per batch'
        self.start = time.perf_counter()
        self.cpu_start = time.process_time()
        self.stage = 'setup'
        self.steps = self.clamped = self.diagnostic_rec_macs = 0
        self.baselines = self.state_cases = self.recurrence_cases = 0
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

    def record(self, n, baseline, cases):
        self.baselines += 1
        self.steps += n*baseline['T']
        if baseline['train_delay'] == 16:
            self.diagnostic_rec_macs += n*baseline['T']*72
        for row in cases:
            self.steps += n*row['continuation_steps']
            self.clamped += n*row['clamped_steps']
            self.state_cases += int(row['kind'] == 'state')
            self.recurrence_cases += int(row['kind'] == 'recurrence')

    def finish(self):
        self.guard(disk=True)
        assert (self.baselines, self.state_cases, self.recurrence_cases) == (24, 816, 84)
        assert (self.steps, self.clamped, self.diagnostic_rec_macs) == (4220928, 270336, 7962624)
        dump(R / 'SCIENCE_SUMMARY.json', dict(
            status='COMPLETE', utc=utc(), models=24, neural_training_updates=0,
            baseline_captures=self.baselines, state_cases=self.state_cases, recurrence_cases=self.recurrence_cases,
            forward_sample_steps=self.steps, clamped_sample_steps=self.clamped,
            dense_forward_macs=156*self.steps-72*self.clamped, diagnostic_rec_macs=self.diagnostic_rec_macs,
            wall_seconds=time.perf_counter()-self.start, cpu_seconds=time.process_time()-self.cpu_start,
            cap_seconds=840, artifact_bytes_at_science_end=artifact_size(), **memory()))
        self.mark('complete')
