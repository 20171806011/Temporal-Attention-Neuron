"""Seal the completed study or its preserved stopped prefix."""

from study import ARTIFACT_CAP, PAUSED, ROOT, artifact_size, read, sha, utc, write


def main():
    frozen, summary, verified = (read(ROOT/name) for name in ('FREEZE.json', 'SUMMARY.json', 'VERIFY.json'))
    complete = summary['status'] == 'COMPLETE_ENGINEERING_ONLY'
    assert verified['status'] == ('PASS' if complete else 'STOPPED_VERIFIED')
    assert summary['formal_training_fits'] == summary['formal_evaluations'] == 0 and not summary['formal_data_generated']
    assert read(PAUSED)['status'] == 'STOPPED_BEFORE_FORMAL_BUDGET'
    assert sha(PAUSED) == frozen['references'][PAUSED.parent.name+'/'+PAUSED.name]
    assert all((ROOT/name).stat().st_size > 0 for name in ('REPORT.md', 'NEXT_PLAN.md', 'RUN_LOG.txt'))
    assert all(not (ROOT/name).exists() for name in ('train', 'evaluate', 'TEST_START.json', 'MANIFEST.json'))
    assert artifact_size() < ARTIFACT_CAP
    stamp = utc()
    write(ROOT/'WORK_STATUS.json', dict(status=summary['status'], utc=stamp, verifier=verified['status'],
          completed_edges=sum(row['stage']=='complete' for row in summary['edges']),
          completed_blocks=sum(row['stage']=='complete' for row in summary['trials']),
          formal_confirmation='STOPPED_BEFORE_FORMAL_BUDGET', formal_training_fits=0, formal_evaluations=0))
    files = {path.relative_to(ROOT).as_posix(): dict(bytes=path.stat().st_size, sha256=sha(path))
             for path in sorted(ROOT.rglob('*')) if path.is_file()}
    write(ROOT/'MANIFEST.json', dict(status='SEALED_ENGINEERING_ONLY' if complete else 'SEALED_STOPPED_ENGINEERING',
          utc=stamp, files=files, total_bytes=sum(record['bytes'] for record in files.values()),
          excludes='MANIFEST.json'))
    print(f'Sealed {len(files)} study files.')


if __name__ == '__main__':
    main()
