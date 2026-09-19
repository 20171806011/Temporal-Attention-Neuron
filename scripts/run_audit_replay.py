"""Replay TAN-II audit suite in isolated snapshot directory.

Executes:
1. s41cd: sprint4_1cd_ei_dynamics.py
2. s42a: code/experiments/sprint4_2/sprint4_2_geometry.py
3. s42_shortcuts: code/experiments/sprint4_2/sprint4_2_shortcut_audit.py
4. s42b: code/experiments/sprint4_2/sprint4_2_vector_qk.py
5. s42b_probe: code/experiments/sprint4_2/sprint4_2_routing_probe.py
6. s42c: code/experiments/sprint4_2/sprint4_2_binding_probe.py
7. s42d: code/experiments/sprint4_2/sprint4_2_hard_binding.py
8. s43a: code/experiments/sprint4_3/sprint4_3a_composition_probe.py
9. s42_theory: code/experiments/sprint4_2/sprint4_2_theory_audit.py (Expected exit code 1, E4(iii) failure)
"""
import os
import sys
import subprocess
import time
import json

def run_replay(review_dir):
    snapshot_dir = os.path.join(review_dir, 'snapshot')
    logs_dir = os.path.join(review_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    jobs = [
        {'script': 'sprint4_1cd_ei_dynamics.py', 'name': 's41cd', 'expected_exit': 0},
        {'script': 'code/experiments/sprint4_2/sprint4_2_geometry.py', 'name': 's42a', 'expected_exit': 0},
        {'script': 'code/experiments/sprint4_2/sprint4_2_shortcut_audit.py', 'name': 's42_shortcuts', 'expected_exit': 0},
        {'script': 'code/experiments/sprint4_2/sprint4_2_vector_qk.py', 'name': 's42b', 'expected_exit': 0},
        {'script': 'code/experiments/sprint4_2/sprint4_2_routing_probe.py', 'name': 's42b_probe', 'expected_exit': 0},
        {'script': 'code/experiments/sprint4_2/sprint4_2_binding_probe.py', 'name': 's42c', 'expected_exit': 0},
        {'script': 'code/experiments/sprint4_2/sprint4_2_hard_binding.py', 'name': 's42d', 'expected_exit': 0},
        {'script': 'code/experiments/sprint4_3/sprint4_3a_composition_probe.py', 'name': 's43a', 'expected_exit': 0},
    ]
    
    results = {}
    
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['MPLCONFIGDIR'] = os.path.join(review_dir, 'matplotlib_config')
    os.makedirs(env['MPLCONFIGDIR'], exist_ok=True)
    
    for job in jobs:
        name = job['name']
        script = job['script']
        exp_exit = job['expected_exit']
        outdir = os.path.join(review_dir, name)
        os.makedirs(outdir, exist_ok=True)
        log_file = os.path.join(logs_dir, f"{name}.log")
        
        cmd = [sys.executable, '-X', 'utf8', '-B', script, '--outdir', outdir]
        print(f"Running {name} ({script})...")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=snapshot_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        elapsed = time.time() - t0
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(proc.stdout)
            
        print(f"  Finished in {elapsed:.2f}s, exit code = {proc.returncode}")
        assert proc.returncode == exp_exit, f"Job {name} exited with {proc.returncode}, expected {exp_exit}"
        results[name] = {
            'script': script,
            'exit_code': proc.returncode,
            'runtime_seconds': elapsed,
            'log_file': log_file,
            'expected_exit': exp_exit,
            'status': 'PASS' if proc.returncode == exp_exit else 'FAIL'
        }
        
    # Replay theory audit separately
    print("Running s42_theory (sprint4_2_theory_audit.py, expected exit=1)...")
    theory_script = 'code/experiments/sprint4_2/sprint4_2_theory_audit.py'
    theory_log = os.path.join(logs_dir, 's42_theory.log')
    t0 = time.time()
    proc_theory = subprocess.run([sys.executable, '-X', 'utf8', '-B', theory_script],
                                 cwd=snapshot_dir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    elapsed_theory = time.time() - t0
    with open(theory_log, 'w', encoding='utf-8') as f:
        f.write(proc_theory.stdout)
        
    print(f"  Finished theory audit in {elapsed_theory:.2f}s, exit code = {proc_theory.returncode}")
    
    # Check that failures contain E4(iii)
    failures = [line for line in proc_theory.stdout.splitlines() if '[FAIL]' in line]
    print(f"  Failures found ({len(failures)}):")
    for fl in failures:
        print("    ", fl)
        
    assert proc_theory.returncode == 1, f"Expected returncode 1, got {proc_theory.returncode}"
    assert len(failures) == 1 and 'E4(iii)' in failures[0], f"Expected exactly E4(iii) failure, got {failures}"
    
    results['s42_theory'] = {
        'script': theory_script,
        'exit_code': proc_theory.returncode,
        'runtime_seconds': elapsed_theory,
        'log_file': theory_log,
        'expected_exit': 1,
        'failures': failures,
        'status': 'PASS (Expected exit 1 and E4(iii) confirmed)'
    }
    
    summary_file = os.path.join(review_dir, 'audit_replay_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("ALL AUDIT REPLAYS COMPLETED! Now verifying 11 replay JSON outputs...")
    diff_report = verify_replay_outputs(review_dir)
    return results, diff_report

def compare_bytes_and_json(b1: bytes, b2: bytes):
    byte_match = (b1 == b2)
    j1 = json.loads(b1.decode('utf-8'))
    j2 = json.loads(b2.decode('utf-8'))
    json_match = (j1 == j2)
    if not byte_match or not json_match:
        raise AssertionError(f"Replay output mismatch: byte_match={byte_match}, json_match={json_match}")
    return True

def verify_replay_outputs(review_dir):
    """Compare all 11 replay JSON outputs against frozen original JSON files."""
    import hashlib
    snapshot_dir = os.path.join(review_dir, 'snapshot')
    
    pairs = [
        ('s41cd/sprint4_1cd_summary.json', 'results/sprint4_1cd/sprint4_1cd_summary.json'),
        ('s42a/sprint4_2_summary.json', 'results/sprint4_2/sprint4_2_summary.json'),
        ('s42_shortcuts/sprint4_2_shortcut_summary.json', 'results/sprint4_2/sprint4_2_shortcut_summary.json'),
        ('s42b/sprint4_2b_summary.json', 'results/sprint4_2/sprint4_2b_summary.json'),
        ('s42b_probe/sprint4_2b_probe_summary.json', 'results/sprint4_2/sprint4_2b_probe_summary.json'),
        ('s42c/sprint4_2c_summary.json', 'results/sprint4_2/sprint4_2c_summary.json'),
        ('s42d/sprint4_2d_summary.json', 'results/sprint4_2/sprint4_2d_summary.json'),
        ('s43a/audit_summary.json', 'results/sprint4_3/audit_summary.json'),
        ('s43a/canonical_results.json', 'results/sprint4_3/canonical_results.json'),
        ('s43a/counterfactual_results.json', 'results/sprint4_3/counterfactual_results.json'),
        ('s43a/monte_carlo_results.json', 'results/sprint4_3/monte_carlo_results.json'),
    ]
    
    report = {
        'total_pairs_checked': len(pairs),
        'all_byte_for_byte_identical': True,
        'results': []
    }
    
    for rep_rel, orig_rel in pairs:
        rep_path = os.path.join(review_dir, rep_rel)
        orig_path = os.path.join(snapshot_dir, orig_rel)
        
        assert os.path.exists(rep_path), f"Missing replay file: {rep_path}"
        assert os.path.exists(orig_path), f"Missing original file: {orig_path}"
        
        with open(rep_path, 'rb') as fp1, open(orig_path, 'rb') as fp2:
            b1 = fp1.read()
            b2 = fp2.read()
            
        h1 = hashlib.sha256(b1).hexdigest()
        h2 = hashlib.sha256(b2).hexdigest()
        
        try:
            compare_bytes_and_json(b1, b2)
            byte_match = True
            json_match = True
        except AssertionError:
            byte_match = (b1 == b2)
            try:
                json_match = (json.loads(b1.decode('utf-8')) == json.loads(b2.decode('utf-8')))
            except Exception:
                json_match = False
            report['all_byte_for_byte_identical'] = False
            
        entry = {
            'replay_file': rep_rel.replace('\\', '/'),
            'original_file': orig_rel.replace('\\', '/'),
            'byte_match': byte_match,
            'sha256_replay': h1,
            'sha256_original': h2,
            'sha256_match': (h1 == h2),
            'json_structural_match': json_match,
            'size_bytes': len(b1)
        }
        report['results'].append(entry)
        print(f"  Verified {rep_rel:45s} == {orig_rel:45s} -> Byte-match: {byte_match}, SHA256: {h1[:16]}...")
        
    diff_report_path = os.path.join(review_dir, 'replay_diff_report.json')
    with open(diff_report_path, 'w', encoding='utf-8') as fp:
        json.dump(report, fp, indent=2, ensure_ascii=False)
        
    assert report['all_byte_for_byte_identical'] is True, "Replay outputs drifted from original outputs!"
    print(f"CONFIRMED: All {len(pairs)} replay JSON files are 100% BIT-FOR-BIT IDENTICAL to originals!")
    print(f"Saved diff report to {diff_report_path}")
    return report

def test_replay_error_fixture():
    """Negative testing: verifies that the actual compare_bytes_and_json function rejects perturbed data."""
    print("Running Replay Error Fixture Tests (testing that real check function compare_bytes_and_json rejects corrupted data)...")
    data_orig = b'{"test": 1.0}'
    data_perturbed_num = b'{"test": 1.0000001}'
    data_perturbed_format = b'{"test":  1.0}'
    
    # 1. Perturbed numerical value
    try:
        compare_bytes_and_json(data_orig, data_perturbed_num)
        raise RuntimeError("compare_bytes_and_json failed to reject numerical difference!")
    except AssertionError:
        print("  [PASS] Perturbed numerical value correctly rejected by compare_bytes_and_json")
        
    # 2. Perturbed formatting (bytes differ, json matches) -> should still fail because bit-for-bit is required
    try:
        compare_bytes_and_json(data_orig, data_perturbed_format)
        raise RuntimeError("compare_bytes_and_json failed to reject byte difference!")
    except AssertionError:
        print("  [PASS] Byte-level drift correctly rejected by compare_bytes_and_json")
        
    # 3. Unaltered identical data passes
    assert compare_bytes_and_json(data_orig, data_orig) is True
    print("ALL REPLAY ERROR FIXTURES PASSED (Real checker functions verified to reject corrupted data)!")

if __name__ == '__main__':
    rev = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # When called directly with --verify-only, just verify outputs without re-executing all 8 jobs
    import sys
    if '--verify-only' in sys.argv:
        verify_replay_outputs(rev)
        test_replay_error_fixture()
    else:
        run_replay(rev)
        test_replay_error_fixture()
