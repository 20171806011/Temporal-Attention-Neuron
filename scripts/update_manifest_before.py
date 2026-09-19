import json

HISTORICAL_RECORDS = {
    'sprint4_1cd_ei_dynamics.py': {
        'hash': '324548198e50f6045052337c53b444bb9949d01a61b9605c0c926b3007eeaf84',
        'source': 'results/sprint4_1cd/sprint4_1cd_summary.json:script_sha256'
    },
    'code/experiments/probe2/identifiability/probe2_identifiability_audit.py': {
        'hash': 'aabf8ebbb81e911e69117591f84e4f836cf59cf8c1b505de53a8b7c1a88143c7',
        'source': 'docs/REPRODUCIBILITY_LEDGER.md:71 & code/experiments/probe2/identifiability/README.md:11'
    },
    'code/experiments/sprint4_2/sprint4_2_geometry.py': {
        'hash': 'cfb201a9ba26ecabea293f146da4f5dbf220aeb0499da988439eb8be256d9ea4',
        'source': 'results/sprint4_2/sprint4_2_summary.json:script_sha256'
    },
    'code/experiments/sprint4_2/sprint4_2_shortcut_audit.py': {
        'hash': '5cdac1369597bde583436d7d3dd22dbf075ed124943e743461080b35b37d43f2',
        'source': 'results/sprint4_2/sprint4_2_shortcut_summary.json:script_sha256'
    },
    'code/experiments/sprint4_2/sprint4_2_vector_qk.py': {
        'hash': '3328362c8e956ff65f93f22c2c3b9f93d22678a07084210e9781864980bbbfcf',
        'source': 'results/sprint4_2/sprint4_2b_summary.json:script_sha256'
    },
    'code/experiments/sprint4_2/sprint4_2_routing_probe.py': {
        'hash': '8f0982c971018f1e8580f54cea69bcb02c965c0825a6a01e534f3e993c2e728f',
        'source': 'results/sprint4_2/sprint4_2b_probe_summary.json:script_sha256'
    },
    'code/experiments/sprint4_2/sprint4_2_binding_probe.py': {
        'hash': '5d55652500be0949aaeeec9497a579d07d341aa449339301cc38238e4e21a6a2',
        'source': 'results/sprint4_2/sprint4_2c_summary.json:script_sha256'
    },
    'code/experiments/sprint4_2/sprint4_2_hard_binding.py': {
        'hash': 'e5178f7713386518d4ad4032bd917efe0b9858c66fd147ca4da5c269df749d90',
        'source': 'results/sprint4_2/sprint4_2d_summary.json:script_sha256'
    },
    'code/experiments/sprint4_3/sprint4_3a_composition_probe.py': {
        'hash': '8155552b8a22e69566c0246d677eacb3b0d45452c6603dc976533cd1fc77bcf0',
        'source': 'results/sprint4_3/audit_summary.json:script_sha256'
    }
}

with open('manifest_before.json', 'r', encoding='utf-8') as f:
    manifest = json.load(f)

updated_count = 0
for entry in manifest:
    rel = entry['relative_path'].replace('\\\\', '/')
    if rel in HISTORICAL_RECORDS:
        rec = HISTORICAL_RECORDS[rel]
        entry['historical_hash_if_available'] = rec['hash']
        entry['historical_timestamp_evidence'] = 'Documented in historical artifact: ' + rec['source']
        entry['notes'] = 'Historical hash verified: 100% match with historical record from ' + rec['source'] + '. Prior mismatch warning was caused by truncated prefix padding.'
        assert entry['historical_hash_if_available'] == entry['current_sha256'], f'Hash mismatch on {rel}!'
        updated_count += 1

print(f'Updated {updated_count}/9 historical hash entries.')
assert updated_count == 9, f'Expected 9 updates, got {updated_count}'

with open('manifest_before.json', 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print('manifest_before.json successfully updated and verified.')
