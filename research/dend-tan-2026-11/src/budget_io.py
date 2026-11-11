"""Resource and record helpers from the sealed block optimization study."""

import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent


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


