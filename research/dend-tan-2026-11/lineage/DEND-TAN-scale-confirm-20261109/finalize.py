"""Check and seal the matched-D16 batch without rerunning it."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LIMITS = {
    'no_cue': ('disagree_ba', 'disagree_coverage', .75),
    'no_candidates': ('balanced_accuracy', 'coverage', .5),
    'no_first': ('select_first_ba', 'select_first_coverage', .75),
    'no_second': ('select_second_ba', 'select_second_coverage', .75),
}


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
    frozen, science, audit = map(read, ('FREEZE.json', 'SCIENCE_SUMMARY.json', 'AUDIT.json'))
    blocks, updates = frozen['blocks'], frozen['updates_per_fit']
    assert blocks in (8, 12) and updates == 600
    fits, evaluations = 3*blocks, 15*blocks
    train_steps, eval_steps = fits*updates*64*36, evaluations*256*36
    expected = dict(blocks=blocks, fits=fits, updates=fits*updates, updates_per_fit=updates,
                    evaluations=evaluations, forward_sample_steps=train_steps+eval_steps,
                    dense_forward_macs=156*(train_steps+eval_steps), dense_backward_macs=240*train_steps)
    for key, value in expected.items():
        assert science[key] == value, key
    assert science['status'] == 'COMPLETE' and audit['status'] == 'PASS'
    assert audit['fits'] == fits and audit['evaluations'] == evaluations
    assert audit['optimizer_updates'] == fits*updates
    assert audit['mathematical_negative_control_conditions'] == 12*blocks
    assert audit['scale_trace_evaluations'] == 10*blocks
    assert 0 < science['wall_seconds'] < 840 and 0 < audit['wall_seconds'] < 300
    assert 0 < science['peak_working_set_bytes'] < 1024**3
    assert not list(ROOT.glob('*FAILURE*.json')) and not list((ROOT / 'train').glob('*PARTIAL.npz'))

    verified = 0
    for filename, folder in (('FREEZE.json', ROOT), ('train/TRAIN_LOCK.json', ROOT/'train')):
        for relative, digest in read(filename)['files'].items():
            assert sha(folder / relative) == digest, relative
            verified += 1
    training, rows = map(read, ('train/ROWS.json', 'evaluate/ROWS.json'))
    assert len(training) == len({row['id'] for row in training}) == fits
    assert len(rows) == len({(r['id'], r['test_delay'], r['condition']) for r in rows}) == evaluations
    assert len(list((ROOT / 'train').glob('*.npz'))) == fits+blocks
    assert len(list((ROOT / 'evaluate').glob('*.npz'))) == evaluations+blocks
    controls = [row for row in rows if row['condition'] != 'nominal']
    assert len(controls) == 12*blocks
    for row in controls:
        ba, coverage, ceiling = LIMITS[row['condition']]
        assert abs(row[ba]-.5*row[coverage]) < 1e-12
        assert row['balanced_accuracy'] <= ceiling

    qa = read('VISUAL_QA.json')
    assert qa['status'] == 'PASS' and qa['images']
    for name, digest in qa['images'].items():
        path = ROOT / name
        assert sha(path) == digest and path.with_suffix('.svg').stat().st_size > 0
    for name in ('REPORT.md', 'PROTOCOL.md', 'REPRODUCE.md', 'AUDIT_LOG.txt', 'SCIENCE_LOG.txt',
                 'RESULTS.json', 'BLOCK_RESULTS.csv', 'PRIMARY_EFFECTS.csv', 'NEGATIVE_CONTROLS.csv'):
        assert (ROOT / name).stat().st_size > 0, name

    timestamp = datetime.now(timezone.utc).isoformat()
    write('WORK_STATUS.json', dict(status='COMPLETE_SCALE_CONFIRMATION', utc=timestamp, **expected))
    write('DELIVERY_CHECKS.json', dict(status='PASS', utc=timestamp, verified_lock_entries=verified,
                                      deletion_controls=len(controls), primary_effects=2,
                                      visual_reviewed=len(qa['images'])))
    files = {path.relative_to(ROOT).as_posix(): dict(bytes=path.stat().st_size, sha256=sha(path))
             for path in sorted(ROOT.rglob('*')) if path.is_file()}
    total = sum(value['bytes'] for value in files.values())
    assert total < 1.5*1024**3
    write('MANIFEST.json', dict(utc=timestamp, files=files, total_bytes=total, excludes='MANIFEST.json itself'))
    print(json.dumps(dict(status='SEALED', files=len(files), mib=total/2**20, lock_entries=verified)))


if __name__ == '__main__':
    main()
