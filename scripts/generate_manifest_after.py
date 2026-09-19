import os
import sys
import json
import hashlib
import datetime

def verify_bidirectional_manifest(baseline_map, current_map):
    """Verify bidirectional exact match between baseline and current maps."""
    missing = [k for k in baseline_map if k not in current_map]
    untracked = [k for k in current_map if k not in baseline_map]
    mismatches = [k for k in baseline_map if k in current_map and baseline_map[k] != current_map[k]]
    
    if missing:
        raise AssertionError(f"Integrity check failed: {len(missing)} missing files: {missing}")
    if untracked:
        raise AssertionError(f"Integrity check failed: {len(untracked)} untracked files: {untracked}")
    if mismatches:
        raise AssertionError(f"Integrity check failed: {len(mismatches)} hash mismatches: {mismatches}")
    return True

def run():
    src_dir = r'C:\Users\李则徐\Downloads\TAN_Final_Submission\TAN_Final_Submission'
    rev_dir = r'C:\Users\李则徐\Downloads\TAN_Review\P0_20260917_181757'
    before_manifest_file = os.path.join(rev_dir, 'manifest_before.json')
    
    with open(before_manifest_file, 'r', encoding='utf-8') as f:
        before_manifest = json.load(f)
        
    before_hash_map = {x['relative_path'].replace(os.path.sep, '/'): x['current_sha256'] for x in before_manifest}
    
    print(f'Starting BIDIRECTIONAL integrity verification against baseline ({len(before_hash_map)} entries)...')
    
    # --- Direction 1: Scan src_dir and compare with baseline ---
    src_files_map = {}
    for root, dirs, files in os.walk(src_dir):
        if '__pycache__' in root:
            continue
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, src_dir).replace(os.path.sep, '/')
            with open(p, 'rb') as fp:
                h = hashlib.sha256(fp.read()).hexdigest()
            src_files_map[rel] = h
            
    print(f'Found {len(src_files_map)} non-cache files in original source directory.')
    
    # Call verify_bidirectional_manifest
    verify_bidirectional_manifest(before_hash_map, src_files_map)
    print(f'CONFIRMED (Bidirectional): Exactly {len(src_files_map)} files verified.')
    print('  - Missing files: 0')
    print('  - Untracked new files: 0')
    print('  - Modified files: 0')
    print('Original research archive is 100% BIT-FOR-BIT IDENTICAL and UNTOUCHED.')
    
    # --- Generate manifest_after.json for review workspace ---
    after_manifest = []
    for root, dirs, files in os.walk(rev_dir):
        if '__pycache__' in root or 'matplotlib_config' in root:
            continue
        for f in files:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, rev_dir).replace(os.path.sep, '/')
            if rel == 'manifest_after.json':
                continue
                
            with open(p, 'rb') as fp:
                data = fp.read()
                h = hashlib.sha256(data).hexdigest()
                
            mtime = os.path.getmtime(p)
            mtime_str = datetime.datetime.fromtimestamp(mtime).isoformat()
            
            status = 'DERIVED_REVIEW_ARTIFACT'
            notes = ''
            
            if rel.startswith('snapshot/'):
                sub_rel = rel[len('snapshot/'):]
                orig_h = before_hash_map.get(sub_rel)
                if orig_h == h:
                    status = 'FROZEN_SNAPSHOT_MATCH'
                    notes = 'Matches original frozen file bit-for-bit'
                else:
                    status = 'DERIVED_REVISED_COPY'
                    notes = 'Revised in derived snapshot copy during P0 errata resolution'
            elif rel.startswith('scripts/'):
                status = 'INDEPENDENT_VERIFICATION_SCRIPT'
                notes = 'Independent verification or audit script'
            elif rel.startswith('logs/'):
                status = 'EXECUTION_LOG'
                notes = 'Execution log from replay suite'
            elif any(rel.startswith(d + '/') for d in ['s41cd', 's42a', 's42_shortcuts', 's42b', 's42b_probe', 's42c', 's42d', 's43a']):
                status = 'REPLAY_OUTPUT_DATA'
                notes = 'Independent replay output data artifact'
            elif rel.endswith('.md') or rel.endswith('.csv') or rel.endswith('.json'):
                status = 'P0_DELIVERABLE'
                notes = 'Mandatory P0 review deliverable'
                
            after_manifest.append({
                'relative_path': rel,
                'size_bytes': len(data),
                'current_sha256': h,
                'observed_timestamp': mtime_str,
                'artifact_category': status,
                'notes': notes
            })
            
    after_manifest.sort(key=lambda x: x['relative_path'])
    after_file = os.path.join(rev_dir, 'manifest_after.json')
    with open(after_file, 'w', encoding='utf-8') as fp:
        json.dump(after_manifest, fp, indent=2, ensure_ascii=False)
        
    print(f'manifest_after.json generated successfully with {len(after_manifest)} entries.')
    return len(src_files_map), len(after_manifest)

def test_manifest_error_fixture():
    print('Running Manifest Error Fixture Tests (testing that verify_bidirectional_manifest rejects corrupted maps)...')
    baseline_fake = {'a.txt': 'hash1', 'b.txt': 'hash2'}
    src_fake_missing = {'a.txt': 'hash1'}
    src_fake_untracked = {'a.txt': 'hash1', 'b.txt': 'hash2', 'c.txt': 'hash3'}
    src_fake_mismatch = {'a.txt': 'hash1', 'b.txt': 'hash2_wrong'}
    
    # 1. Missing file -> verify_bidirectional_manifest must raise AssertionError
    try:
        verify_bidirectional_manifest(baseline_fake, src_fake_missing)
        raise RuntimeError("verify_bidirectional_manifest failed to reject missing file!")
    except AssertionError as e:
        assert "missing files" in str(e)
        print("  [PASS] Missing file correctly rejected by verify_bidirectional_manifest")
        
    # 2. Untracked file -> verify_bidirectional_manifest must raise AssertionError
    try:
        verify_bidirectional_manifest(baseline_fake, src_fake_untracked)
        raise RuntimeError("verify_bidirectional_manifest failed to reject untracked file!")
    except AssertionError as e:
        assert "untracked files" in str(e)
        print("  [PASS] Untracked file correctly rejected by verify_bidirectional_manifest")
        
    # 3. Hash mismatch -> verify_bidirectional_manifest must raise AssertionError
    try:
        verify_bidirectional_manifest(baseline_fake, src_fake_mismatch)
        raise RuntimeError("verify_bidirectional_manifest failed to reject hash mismatch!")
    except AssertionError as e:
        assert "hash mismatches" in str(e)
        print("  [PASS] Hash mismatch correctly rejected by verify_bidirectional_manifest")
        
    # 4. Exact match -> must pass
    assert verify_bidirectional_manifest(baseline_fake, baseline_fake) is True
    print('ALL MANIFEST ERROR FIXTURES PASSED (Real checker functions verified to reject corrupted data)!')

if __name__ == '__main__':
    run()
    test_manifest_error_fixture()
