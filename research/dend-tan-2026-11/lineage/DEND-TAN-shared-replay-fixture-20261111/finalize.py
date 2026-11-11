"""Seal completed fixture evidence without claiming a scientific confirmation."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    with (ROOT/name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    freeze, summary, audit, timing, state = map(read, ('FREEZE.json', 'SUMMARY.json', 'AUDIT.json', 'TIMING.json', 'RUN_STATE.json'))
    assert freeze['study'] == 'FIXTURE_ONLY' and freeze['fixture_data_generated'] is False
    assert summary['status'] == 'COMPLETE_FIXTURE' and audit['status'] == 'PASS' and state['status'] == 'complete_fixture'
    assert summary['fits'] == summary['updates'] == 0 and summary['scientific_data_generated'] is False
    assert summary['checked_episodes'] == audit['checked_episodes'] == 2060
    assert summary['logical_paths'] == audit['logical_paths'] == 27
    assert len(timing['repeats']) == summary['timing_repeats'] == audit['timing_repeats'] == 3
    assert len(timing['single']) == summary['isolated_pairs'] == audit['isolated_pairs']
    assert summary['work'] == audit['work']
    assert 0 < summary['wall_seconds'] <= 90 and 0 < audit['wall_seconds'] <= 60
    assert summary['peak_working_set_bytes'] <= 768*1024**2 and audit['peak_working_set_bytes'] <= 768*1024**2
    assert len(list((ROOT/'evaluate').glob('*.npz'))) == 54
    assert all(not (ROOT/name).exists() for name in ('train', 'weights', 'FIXTURE_FAILURE.json', 'AUDIT_FAILURE.json'))
    for name, digest in freeze['files'].items():
        assert sha(ROOT/name) == digest, name
    for name in ('REPORT.md', 'NEXT_PLAN.md', 'FIXTURE_LOG.txt', 'AUDIT_LOG.txt'):
        assert (ROOT/name).stat().st_size > 0, name
    stamp = datetime.now(timezone.utc).isoformat()
    write('WORK_STATUS.json', dict(status='COMPLETE_FIXTURE_ONLY', utc=stamp, fits=0, formal_evaluations=0,
                                   equivalence_verified=True, scientific_data_generated=False))
    write('DELIVERY_CHECKS.json', dict(status='PASS', scope='engineering fixture only', utc=stamp,
                                      checks=audit['checks'], frozen_files=len(freeze['files'])))
    files = {p.relative_to(ROOT).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(ROOT.rglob('*')) if p.is_file()}
    total = sum(value['bytes'] for value in files.values())
    assert total <= 256*1024**2
    write('MANIFEST.json', dict(status='SEALED_FIXTURE_ONLY', utc=stamp, files=files,
                               total_bytes=total, excludes='MANIFEST.json itself'))
    print(json.dumps(dict(status='SEALED_FIXTURE_ONLY', files=len(files), mib=total/2**20)))


if __name__ == '__main__':
    main()
