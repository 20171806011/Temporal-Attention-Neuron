"""Offline consistency checks of archived evidence; no training or simulation."""
from pathlib import Path
import hashlib
import json
import numpy as np

R = Path(__file__).resolve().parent
U = R / 'upstream_probe1'
checks = 0

def check(value):
    global checks
    assert value
    checks += 1

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for base in (R, U):
    for line in (base / 'MANIFEST.sha256').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        check(sha(base / name) == digest)
j = json.loads((R / 'results.json').read_text())
check(sha(R / 'FROZEN_PROTOCOL.md') == j['protocol_sha256'])
differences = []
for row in j['runs']:
    seed = row['seed']
    with np.load(U / 'run' / f'inputs_{seed}.npz', allow_pickle=False) as inputs, np.load(U / 'run' / f'traces_{seed}.npz', allow_pickle=False) as traces, np.load(R / f'predictions_{seed}.npz', allow_pickle=False) as saved:
        check(sha(U / 'run' / f'inputs_{seed}.npz') == row['input_sha256'])
        y = inputs['test_y']
        mask = ~inputs['test_dist']
        check(np.array_equal(saved['test_y'], y))
        check(np.array_equal(saved['test_clean'], mask))
        check(np.array_equal(saved['test_group'], inputs['test_group']))
        check(int(mask.sum()) == 1001)
        perm = np.random.RandomState(2026110351 + seed).permutation(len(inputs['train_y']))
        clean, mixed = [], []
        for k in range(3):
            c = perm[(inputs['train_y'][perm] == k) & ~inputs['train_dist'][perm]]
            m = perm[(inputs['train_y'][perm] == k) & inputs['train_dist'][perm]]
            clean.extend(c[:500]); mixed.extend(c[:250]); mixed.extend(m[:250])
        scores = {}
        for arm, selected in [('clean_only', clean), ('mixed', mixed)]:
            a = row['arms'][arm]
            ids = saved[arm + '_indices']
            check(np.array_equal(ids, np.sort(selected)))
            check(np.array_equal(ids, a['selected_indices']))
            check(np.array_equal(np.bincount(inputs['train_y'][ids], minlength=3), a['class_counts']))
            check(np.array_equal(np.bincount(inputs['train_dist'][ids].astype(int), minlength=2), a['condition_counts']))
            f = traces['source_softmax_features']
            w = np.asarray(a['weights']).reshape(3, f.shape[1]+1)
            z = ((f - a['mean']) / a['std']) @ w[:, 1:].T + w[:, 0]
            e = np.exp(z - z.max(axis=1, keepdims=True))
            prob = e / e.sum(axis=1, keepdims=True)
            check(np.allclose(prob, saved[arm + '_probability'], rtol=0, atol=1e-12))
            pred = saved[arm + '_prediction']
            check(np.array_equal(prob.argmax(1), pred))
            cm = np.bincount(3*y[mask]+pred[mask], minlength=9).reshape(3, 3)
            check(np.array_equal(cm, a['confusion']))
            recalls = cm.diagonal() / cm.sum(axis=1)
            check(np.allclose(recalls, a['recalls'], rtol=0, atol=1e-15))
            scores[arm] = recalls.mean()
            check(abs(scores[arm] - a['balanced3']) < 1e-15)
            check(a['optimizer']['success'])
        delta = scores['clean_only'] - scores['mixed']
        check(abs(delta - row['difference']) < 1e-15)
        differences.append(delta)
check(abs(np.mean(differences) - j['primary_difference']) < 1e-15)
for arm in ['clean_only', 'mixed']:
    check(abs(np.mean([r['arms'][arm]['balanced3'] for r in j['runs']]) - j['means'][arm]) < 1e-15)
b = np.load(R / 'conditional_bootstrap.npy', allow_pickle=False)
check(b.shape == (2000,))
check(np.isfinite(b).all())
check(np.allclose(np.percentile(b, [2.5, 97.5]), j['conditional_percentile95'], rtol=0, atol=1e-15))
print(json.dumps({'status': 'passed', 'checks': checks, 'scope': 'Current hashes and archived evidence consistency; no new fits, simulation, or data generation.'}, indent=2))
