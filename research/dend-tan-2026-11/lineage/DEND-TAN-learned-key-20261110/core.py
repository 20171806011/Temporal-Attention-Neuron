"""Paths, locks, native cost ledger and CPU budget for the rotated-query learned-key task."""

import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


R = Path(__file__).resolve().parent
ARMS = ('MEAN', 'ORACLE', 'LEARN')
SWAPPABLE = ('LEARN', 'ORACLE')
CONDITIONS = ('nominal', 'no_cue', 'no_tags', 'no_contents')


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
        assert not any((R / name).exists() for name in ('RUN_STATE.json', 'train', 'evaluate')), \
            'one scientific run per batch'
        self.start = time.perf_counter()
        self.cpu_start = time.process_time()
        self.stage = 'setup'
        self.steps = self.train_steps = self.updates = self.evaluations = 0
        self.mark('running')

    def mark(self, status, **other):
        dump(R / 'RUN_STATE.json', dict(status=status, stage=self.stage, utc=utc(),
                                      wall_seconds=time.perf_counter()-self.start, **other))

    def guard(self, disk=False):
        if time.perf_counter() - self.start > 840:
            raise TimeoutError('840 second scientific cap')
        if memory()['peak_working_set_bytes'] > 1024**3:
            raise MemoryError('1 GiB working-set cap')
        if disk and artifact_size() > 1.5 * 1024**3:
            raise OSError('1.5 GiB artifact cap')

    def record_train(self, rows, steps):
        self.updates += 1
        self.steps += rows*steps
        self.train_steps += rows*steps

    def record_eval(self, rows, steps):
        self.evaluations += 1
        self.steps += rows*steps

    def finish(self, blocks, updates_per_fit):
        self.guard(disk=True)
        assert self.updates == 3*blocks*updates_per_fit and self.evaluations == 14*blocks
        dump(R / 'SCIENCE_SUMMARY.json', dict(
            status='COMPLETE', utc=utc(), blocks=blocks, fits=3*blocks, updates=self.updates,
            updates_per_fit=updates_per_fit, evaluations=self.evaluations,
            forward_sample_steps=self.steps, dense_forward_macs=156*self.steps,
            dense_backward_macs=240*self.train_steps,
            wall_seconds=time.perf_counter()-self.start, cpu_seconds=time.process_time()-self.cpu_start,
            cap_seconds=840, artifact_bytes_at_science_end=artifact_size(), **memory()))
        self.mark('complete')


def structural_cost(arm, steps):
    units = 12*steps
    return dict(parameters=180 if arm == 'LEARN' else 176, allocated_parameter_slots=180,
                fixed_match_constants=0 if arm == 'LEARN' else 4, spiking_cells=8, graded_branches=12,
                continuous_state_slots=48, history_slots=0, trace_slots=24, key_slots=4,
                persistent_array_bytes=448, dense_forward_macs=156*steps, graded_transfers=units,
                matcher_dense_macs=8*steps, matcher_logit_scale_multiplies=2*steps,
                softmax_exponentials=2*steps, softmax_divisions=2*steps,
                query_norm_square_multiplies=2*steps, query_norm_adds=steps, query_norm_sqrts=steps,
                value_tanh_evaluations=units, value_read_multiplies=2*units, value_read_adds=units,
                value_decay_multiplies=2*units, injection_multiplies=units)
