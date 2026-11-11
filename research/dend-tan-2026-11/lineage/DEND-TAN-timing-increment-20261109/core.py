"""Paths, locks and CPU budget for the frozen timing experiment."""

import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


R = Path(__file__).resolve().parent
ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2')
ATTENTION_ARMS = ('FULL', 'NOATT')
TAPS = (9, 15, 19)  # Interface of the unchanged native collection module.


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
        self.baselines = self.steps = self.rank_fits = self.class_fits = 0
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
        self.steps += 24*rows

    def finish(self):
        self.guard(disk=True)
        assert (self.baselines, self.steps, self.rank_fits, self.class_fits) == (72, 1327104, 612, 72)
        dump(R / 'SCIENCE_SUMMARY.json', dict(
            status='COMPLETE', utc=utc(), models=36, neural_training_updates=0,
            baseline_captures=self.baselines, rank_fits=self.rank_fits, class_fits=self.class_fits,
            forward_sample_steps=self.steps, dense_forward_macs=156*self.steps,
            wall_seconds=time.perf_counter()-self.start, cpu_seconds=time.process_time()-self.cpu_start,
            cap_seconds=840, artifact_bytes_at_science_end=artifact_size(), **memory()))
        self.mark('complete')
