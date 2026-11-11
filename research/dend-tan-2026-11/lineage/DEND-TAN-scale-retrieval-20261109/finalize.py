"""Verify delivery completeness and seal the frozen-weight D16 retrieval batch."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def write(name, value):
    with (ROOT / name).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (ROOT / 'MANIFEST.json').exists()
    frozen, science, audit, qa = map(read, ('FREEZE.json', 'SCIENCE_SUMMARY.json', 'AUDIT.json', 'VISUAL_QA.json'))
    n = frozen['examples_per_block']
    assert n in (128, 256) and frozen['blocks'] == 12
    expected = dict(blocks=12, examples_per_block=n, fits=0, updates=0, evaluations=252,
                    forward_sample_steps=252*n*36, dense_forward_macs=252*n*36*156)
    assert all(science[key] == value for key, value in expected.items())
    assert science['status'] == 'COMPLETE' and audit['status'] == qa['status'] == 'PASS'
    assert audit['evaluations'] == 252
    assert 0 < science['wall_seconds'] <= 840 and 0 < audit['wall_seconds'] <= 300
    assert 0 < science['peak_working_set_bytes'] <= 1024**3
    assert not (ROOT / 'SCIENCE_FAILURE.json').exists() and not (ROOT / 'AUDIT_FAILURE.json').exists()
    assert not (ROOT / 'train').exists()
    for relative, digest in frozen['files'].items():
        assert sha(ROOT / relative) == digest, relative
    rows = read('evaluate/ROWS.json')
    assert len(rows) == len({(row['seed'], row['arm'], row['condition']) for row in rows}) == 252
    assert len(list((ROOT / 'weights').glob('*.npy'))) == 36
    assert len(list((ROOT / 'evaluate').glob('*.npz'))) == 276
    assert sha(ROOT / 'RETRIEVAL_EFFECTS.png') == qa['images']['RETRIEVAL_EFFECTS.png']
    for name in ('REPORT.md', 'RESULTS.json', 'BLOCK_RESULTS.csv', 'PRIMARY_EFFECTS.csv', 'TRANSFER_ROWS.csv',
                 'SCIENCE_LOG.txt', 'AUDIT_LOG.txt', 'ANALYSIS_LOG.txt', 'RETRIEVAL_EFFECTS.svg'):
        assert (ROOT / name).stat().st_size > 0, name
    timestamp = datetime.now(timezone.utc).isoformat()
    write('WORK_STATUS.json', dict(status='COMPLETE_SCALE_RETRIEVAL', utc=timestamp, **expected))
    write('DELIVERY_CHECKS.json', dict(status='PASS', utc=timestamp, frozen_files=len(frozen['files']),
                                      verified_evaluations=252, visual_reviewed=1, audit_checks=audit['checks']))
    files = {path.relative_to(ROOT).as_posix(): dict(bytes=path.stat().st_size, sha256=sha(path))
             for path in sorted(ROOT.rglob('*')) if path.is_file()}
    total = sum(value['bytes'] for value in files.values())
    assert total < 1.5*1024**3
    write('MANIFEST.json', dict(utc=timestamp, files=files, total_bytes=total, excludes='MANIFEST.json itself'))
    print(json.dumps(dict(status='SEALED', files=len(files), mib=total/2**20)))


if __name__ == '__main__':
    main()
