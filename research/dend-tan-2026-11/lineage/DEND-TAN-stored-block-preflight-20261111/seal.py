"""Seal a verified preflight and its report without starting a formal phase."""

from budget_io import ROOT, artifact_size, read, sha, utc, write


def main():
    frozen, timing, audit = (read(ROOT/name) for name in ('FREEZE.json', 'TIMING.json', 'VERIFY.json'))
    complete = timing['status'] in ('FEASIBLE', 'INFEASIBLE')
    assert audit['status'] == ('PASS' if complete else 'STOPPED_VERIFIED')
    assert timing['formal_training_fits'] == timing['formal_evaluations'] == 0 and not timing['scientific_data_generated']
    assert all((ROOT/name).stat().st_size > 0 for name in ('REPORT.md', 'NEXT_PLAN.md', 'RUN_LOG.txt'))
    assert all(not (ROOT/name).exists() for name in ('train', 'evaluate', 'TEST_START.json', 'MANIFEST.json'))
    assert artifact_size() <= frozen['design']['artifact_cap_bytes']
    stamp = utc()
    trials = [trial for candidate in timing['candidates'] for trial in candidate['trials']]
    write(ROOT/'WORK_STATUS.json', dict(status='COMPLETE_BUDGET_ONLY' if complete else 'STOPPED_BUDGET_ERROR',
          utc=stamp, decision=timing['status'], selected_examples_per_block=timing['selected_examples_per_block'],
          completed_trials=sum(trial.get('stage') == 'complete' for trial in trials),
          formal_confirmation='PAUSED', formal_training_fits=0, formal_evaluations=0))
    files = {path.relative_to(ROOT).as_posix(): dict(bytes=path.stat().st_size, sha256=sha(path))
             for path in sorted(ROOT.rglob('*')) if path.is_file()}
    write(ROOT/'MANIFEST.json', dict(status='SEALED_BUDGET_ONLY' if complete else 'SEALED_STOPPED_PREFLIGHT',
          utc=stamp, files=files, total_bytes=sum(record['bytes'] for record in files.values()),
          excludes='MANIFEST.json'))
    print(f'Sealed {len(files)} budget-only files.')


if __name__ == '__main__':
    main()
