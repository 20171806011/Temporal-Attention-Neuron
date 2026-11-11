"""Check delivery and seal the rotated-query learned-key study."""

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
    blocks, updates = frozen['blocks'], frozen['updates_per_fit']
    assert blocks in (8, 12) and updates == 600
    fits, evaluations = 3*blocks, 14*blocks
    train_steps, eval_steps = fits*updates*128*36, evaluations*256*36
    expected = dict(blocks=blocks, fits=fits, updates=fits*updates, updates_per_fit=updates,
                    evaluations=evaluations, forward_sample_steps=train_steps+eval_steps,
                    dense_forward_macs=156*(train_steps+eval_steps), dense_backward_macs=240*train_steps)
    assert all(science[key] == value for key, value in expected.items())
    assert science['status'] == 'COMPLETE' and audit['status'] == qa['status'] == 'PASS'
    assert audit['fits'] == fits and audit['evaluations'] == evaluations
    assert audit['optimizer_updates'] == fits*updates
    assert 0 < science['wall_seconds'] <= 840 and 0 < audit['wall_seconds'] <= 300
    assert 0 < science['peak_working_set_bytes'] <= 1024**3
    assert not (ROOT / 'SCIENCE_FAILURE.json').exists() and not (ROOT / 'AUDIT_FAILURE.json').exists()
    assert not list((ROOT / 'train').glob('*PARTIAL.npz'))
    verified = 0
    for lock, folder in (('FREEZE.json', ROOT), ('train/TRAIN_LOCK.json', ROOT/'train')):
        for relative, digest in read(lock)['files'].items():
            assert sha(folder / relative) == digest, relative
            verified += 1
    training, evaluation = map(read, ('train/ROWS.json', 'evaluate/ROWS.json'))
    assert len(training) == len({row['id'] for row in training}) == fits
    assert len(evaluation) == len({(row['id'], row['condition']) for row in evaluation}) == evaluations
    assert len(list((ROOT / 'train').glob('*.npz'))) == fits+blocks
    assert len(list((ROOT / 'evaluate').glob('*.npz'))) == evaluations+blocks
    assert len(read('RESULTS.json')['primary_effects']) == 2
    for filename, digest in qa['images'].items():
        path = ROOT / filename
        assert sha(path) == digest and path.with_suffix('.svg').stat().st_size > 0
    assert 'LEARNED_KEY_EFFECTS.png' in qa['images']
    for name in ('REPORT.md', 'BLOCK_RESULTS.csv', 'PRIMARY_EFFECTS.csv', 'TRAIN_ROWS.csv', 'EVAL_ROWS.csv',
                 'NEGATIVE_CONTROLS.csv', 'SCIENCE_LOG.txt', 'AUDIT_LOG.txt', 'ANALYSIS_LOG.txt'):
        assert (ROOT / name).stat().st_size > 0, name

    timestamp = datetime.now(timezone.utc).isoformat()
    write('WORK_STATUS.json', dict(status='COMPLETE_LEARNED_KEY', utc=timestamp, **expected))
    write('DELIVERY_CHECKS.json', dict(status='PASS', utc=timestamp, verified_lock_entries=verified,
                                      verified_evaluations=evaluations, primary_effects=2,
                                      visual_reviewed=len(qa['images']), audit_checks=audit['checks']))
    files = {path.relative_to(ROOT).as_posix(): dict(bytes=path.stat().st_size, sha256=sha(path))
             for path in sorted(ROOT.rglob('*')) if path.is_file()}
    total = sum(value['bytes'] for value in files.values())
    assert total < 1.5*1024**3
    write('MANIFEST.json', dict(utc=timestamp, files=files, total_bytes=total, excludes='MANIFEST.json itself'))
    print(json.dumps(dict(status='SEALED', files=len(files), mib=total/2**20, lock_entries=verified)))


if __name__ == '__main__':
    main()
