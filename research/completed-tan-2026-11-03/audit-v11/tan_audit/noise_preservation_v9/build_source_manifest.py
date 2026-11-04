#!/usr/bin/env python3
"""Read-only source/table provenance audit. Does not execute model code."""
import ast
import csv
import hashlib
import io
import json
import os
from pathlib import Path

ROOT = Path(os.environ['TAN_REPO_ROOT']) if 'TAN_REPO_ROOT' in os.environ else Path(__file__).resolve().parents[2] / 'Temporal-Attention-Neuron'
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'snapshot/code/experiments/statistical_validation.py'

def digest(data):
    return hashlib.sha256(data).hexdigest()

source_bytes = SOURCE.read_bytes()
source = source_bytes.decode('utf-8')
tree = ast.parse(source, filename=str(SOURCE))
selected_names = ('LIFNeuron', 'TANNeuron', 'run_experiment_B')
nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in selected_names]
assert tuple(n.name for n in nodes) == selected_names
segments = [ast.get_source_segment(source, n) for n in nodes]
extracted = '# Exact selected source bodies; original file is never imported.\nimport numpy as np\n\n' + '\n\n\n'.join(segments) + '\n'
(OUT / 'taskb_original_extracted.py').write_text(extracted)
compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec')  # compile only, never exec

paths = [SOURCE, ROOT / 'snapshot/code/experiments/baseline_comparison.py', ROOT / 'snapshot/code/experiments/ablation.py']
table_audits = {}
for stem in ('stats_validation_metrics', 'stats_validation_summary', 'baselines_metrics', 'baselines_summary', 'ablation_metrics', 'ablation_summary'):
    left = ROOT / f'snapshot/data/experiment_results/{stem}.csv'
    right = ROOT / f'snapshot/results/tables/{stem}.csv'
    lb, rb = left.read_bytes(), right.read_bytes()
    rows = list(csv.DictReader(io.StringIO(lb.decode())))
    task_column = 'exp' if 'exp' in rows[0] else 'task'
    taskb = [r for r in rows if r.get(task_column) == 'B']
    table_audits[stem] = {
        'duplicate_bytes_equal': lb == rb,
        'task_column': task_column,
        'n_rows': len(rows), 'n_task_B_rows': len(taskb),
        'task_B_summary_rows': taskb if 'summary' in stem else None,
    }
    if stem == 'stats_validation_metrics':
        keys = [(r['model'], int(r['seed'])) for r in taskb]
        expected = {(model, seed) for model in ('LIF', 'TAN') for seed in range(50)}
        assert len(keys) == 100 and len(set(keys)) == 100 and set(keys) == expected
        table_audits[stem]['task_B_seed_model_coverage_exact_50_by_2'] = True
    paths += [left, right]

manifest = {
    'scope': 'Source/table reads and compilation only; no model execution or new statistics.',
    'repository': str(ROOT),
    'source_files': {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': digest(p.read_bytes())} for p in paths},
    'selected_source_nodes': [{'name': n.name, 'start_line': n.lineno, 'end_line': n.end_lineno,
                              'exact_segment_sha256': digest(s.encode()),
                              'ast_sha256': digest(ast.dump(n, include_attributes=False).encode())}
                             for n, s in zip(nodes, segments)],
    'extracted_path': str(OUT / 'taskb_original_extracted.py'),
    'extracted_sha256': digest(extracted.encode()),
    'runtime_dependencies_for_selected_nodes': ['numpy as np', 'Python builtins'],
    'omitted_top_level_side_effects': ['font configuration', 'outputs/stage2_part1 directory creation', 'logging FileHandler'],
    'table_audits': table_audits,
}
(OUT / 'NOISE_SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'status': 'source-only manifest complete', 'selected_nodes': selected_names,
                  'task_B_coverage': table_audits['stats_validation_metrics']['task_B_seed_model_coverage_exact_50_by_2'],
                  'duplicate_tables_equal': {k: v['duplicate_bytes_equal'] for k, v in table_audits.items()}}, indent=2))
