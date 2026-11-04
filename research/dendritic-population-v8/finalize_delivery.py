"""Finalize preservation metadata and manifest; no scientific execution.

Run only after reports and artifacts are complete. Re-running this changes no
scientific source or result but regenerates the two integrity metadata files.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

R = Path(__file__).resolve().parent
REPO = R.parents[1]
REL = R.relative_to(REPO).as_posix()
BASE = '0c74816da453003490f313c8ee7939e16840e4f2'
PRE = 'cd5cd4ef1cc65fd098b3a73d36d68e7a49307d17'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO, text=True).strip()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-commit', required=True)
    args = parser.parse_args()
    cp = git('rev-parse', args.checkpoint_commit)
    historical = []
    for phase in range(1, 8):
        directory = R.parent/f'dendritic-population-v{phase}'
        path = directory/'MANIFEST.json'
        record = json.loads(path.read_text())
        for name, digest in record['files'].items():
            assert sha(directory/name) == digest, (phase, name)
        historical.append(dict(phase=phase, manifest_sha256=sha(path), entries=len(record['files'])))
    freeze = json.loads((R/'FREEZE.json').read_text())
    for name, digest in freeze['sha256'].items():
        assert sha(R/name) == digest, name
    for name in ('verification', 'gradient_audit'):
        assert json.loads((R/'results'/f'{name}.json').read_text())['status'] == 'PASS', name
    before = git('ls-tree', '-r', '--name-only', PRE, '--', REL).splitlines()
    checkpoint = git('ls-tree', '-r', '--name-only', cp, '--', REL).splitlines()
    assert not any('/results/' in name for name in before)
    assert len([name for name in checkpoint if '/results/' in name and name.endswith('.npy')]) == 832
    assert not any('/test_s' in name or 'TEST_STARTED' in name for name in checkpoint)
    lock = subprocess.check_output(['git', 'show', cp+':'+REL+'/results/TRAIN_LOCK.json'], cwd=REPO)
    assert hashlib.sha256(lock).hexdigest() == sha(R/'results/TRAIN_LOCK.json')
    train = json.loads(lock)
    assert len(train['rows']) == 416 and not train['test_generated']
    for name, digest in train['files'].items():
        assert sha(R/'results'/name) == digest, name
        assert REL+'/results/'+name in checkpoint, name
    test = json.loads((R/'results/TEST_SUMMARY.json').read_text())
    assert len(test['rows']) == 416
    for name, digest in test['hashes'].items():
        assert sha(R/'results'/name) == digest, name
    changes = git('diff', '--name-only', BASE).splitlines()
    outside = [name for name in changes if not name.startswith(REL+'/')]
    assert not outside, outside
    main_ref = git('rev-parse', 'refs/remotes/origin/main')
    assert main_ref == '85213f0d417d54e4f48b301ed1c0e4200ef3c421'
    dump(R/'PRESERVATION.json', dict(status='PASS', baseline=BASE,
        historical_manifests=historical, unchanged_manifest_entries=sum(x['entries'] for x in historical),
        recorded_origin_main=main_ref,
        main_check_scope='Existing local remote-tracking reference; no network fetch, push or main modification in V8',
        changes_outside_v8=outside, predata_commit=PRE, checkpoint_commit=cp,
        commit_gate_checks=dict(predata_tree_has_no_results=True,
            checkpoint_tree_has_416_initial_and_416_final_weights=True,
            checkpoint_tree_has_all_locked_training_artifacts=True,
            checkpoint_tree_has_no_test_data_or_marker=True,
            committed_train_lock_matches_delivered_bytes=True,
            all_frozen_sources_unchanged=True, all_audits_passed=True)))
    files = {p.relative_to(R).as_posix(): sha(p) for p in sorted(R.rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts and p.name != 'MANIFEST.json'}
    dump(R/'MANIFEST.json', dict(algorithm='SHA-256',
        scope='All V8 delivery files except this manifest and Python bytecode caches; earlier frozen artifacts preserved.',
        file_count=len(files), files=files))
    print(json.dumps(dict(manifest_files=len(files), historical_entries=sum(x['entries'] for x in historical),
        source_freeze='PASS', commit_gates='PASS', audits='PASS', scientific_calls=0)))


if __name__ == '__main__':
    main()
