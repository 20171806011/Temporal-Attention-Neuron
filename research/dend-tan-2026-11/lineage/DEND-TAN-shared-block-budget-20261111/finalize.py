"""Seal only the completed complete-block budget evidence."""

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
    frozen, timing, audit, state = map(read, ('FREEZE.json', 'TIMING.json', 'TIMING_AUDIT.json', 'RUN_STATE.json'))
    assert frozen['study'] == 'FULL_BLOCK_BUDGET_ONLY' and state['status'] == 'complete_budget'
    assert timing['status'] in ('FEASIBLE', 'INFEASIBLE') and audit['status'] == 'PASS'
    assert timing['formal_training_fits'] == timing['formal_evaluations'] == 0
    assert timing['scientific_data_generated'] is False
    assert all(not (ROOT/name).exists() for name in ('train', 'evaluate', 'weights', 'TEST_START.json',
                                                    'TRAIN_SUMMARY.json', 'SCIENCE_SUMMARY.json'))
    assert timing['selected_examples_per_block'] == state['selected_examples_per_block']
    trials = [row for candidate in timing['candidates'] for row in candidate['trials']]
    assert len(trials) in (2, 4)
    for trial in trials:
        folder = ROOT/trial['directory']
        assert sha(folder/'BLOCK.json') == trial['block_sha256']
        assert sha(folder/'AUDIT.json') == trial['audit_sha256']
        assert read(trial['directory']+'/AUDIT.json')['status'] == 'PASS'
    for name, digest in frozen['files'].items():
        assert sha(ROOT/name) == digest, name
    for name in ('REPORT.md', 'NEXT_PLAN.md', 'BUDGET_LOG.txt', 'TIMING_AUDIT_LOG.txt'):
        assert (ROOT/name).stat().st_size > 0
    stamp = datetime.now(timezone.utc).isoformat()
    write('WORK_STATUS.json', dict(status='COMPLETE_BUDGET_ONLY', utc=stamp,
          budget_decision=timing['status'], examples_per_block=timing['selected_examples_per_block'],
          formal_training_fits=0, formal_evaluations=0, scientific_data_generated=False))
    write('DELIVERY_CHECKS.json', dict(status='PASS', scope='budget measurement and equivalence only', utc=stamp,
                                      complete_trials=len(trials), frozen_files=len(frozen['files'])))
    files = {p.relative_to(ROOT).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(ROOT.rglob('*')) if p.is_file()}
    total = sum(value['bytes'] for value in files.values())
    assert total <= int(1.5*1024**3)
    write('MANIFEST.json', dict(status='SEALED_BUDGET_ONLY', utc=stamp, files=files,
                               total_bytes=total, excludes='MANIFEST.json itself'))
    print(json.dumps(dict(status='SEALED_BUDGET_ONLY', files=len(files), mib=total/2**20)))


if __name__ == '__main__':
    main()
