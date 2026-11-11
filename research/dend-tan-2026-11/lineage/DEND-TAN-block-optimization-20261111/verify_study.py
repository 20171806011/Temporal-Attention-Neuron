"""Verify the saved engineering study ledger without replaying any trajectories."""

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile


ROOT = Path(__file__).resolve().parent
BUDGET = 'DEND-TAN-shared-block-budget-20261111'
EDGE = 'DEND-TAN-shared-replay-fixture-20261111'
PAUSED = 'DEND-TAN-evicted-key-confirm-20261111/WORK_STATUS.json'
VARIANTS = {'A': (32, ZIP_DEFLATED), 'B': (256, ZIP_DEFLATED), 'C': (256, ZIP_STORED)}
ROUNDS = ('ABC', 'CBA')
CAPS = {'science': 180., 'audit': 300.}
MEMORY_CAP = 1024**3
ARTIFACT_CAP = 3*1024**3//2
RULES = ('original', 'replay', 'guarded')
PATHS = {f'{g}_{h}_{r}.npz' for g in ('orthogonal', 'near')
         for h in ('four', 'three0', 'three2', 'two') for r in RULES}
TOLERANCE = 5e-12
TIME_TOLERANCE = 1e-7


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, context):
    if not condition:
        raise ValueError(context)


def nonnegative(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, name)
    return value


def stamp(value):
    result = datetime.fromisoformat(value)
    require(result.utcoffset() is not None and result.utcoffset().total_seconds() == 0, 'UTC timestamp')
    return result


def equal_time(left, right, context):
    require(abs(left-right) <= TIME_TOLERANCE, context)


def validate_freeze(frozen):
    design = dict(examples=256, edge_examples=12, edge_packet=8, edge_order=list(VARIANTS),
                  variants={name: dict(packet_size=packet, zip_mode=mode)
                            for name, (packet, mode) in VARIANTS.items()},
                  rounds=[list(order) for order in ROUNDS], caps_seconds=CAPS,
                  trial_admission_reserve_seconds=25., edge_admission_reserve_seconds=5.,
                  memory_cap_bytes=MEMORY_CAP, artifact_cap_bytes=ARTIFACT_CAP)
    require(frozen['design'] == design and frozen['scope'] == 'SYNTHETIC_ENGINEERING_ONLY', 'fixed design')
    require(frozen['host'].upper() == 'REDACTED_HOST', 'fixed host')
    require(frozen['formal_training_fits'] == frozen['formal_evaluations'] == 0, 'freeze scope')
    names = {path.name for path in ROOT.glob('*.py')} | {'PROTOCOL.md', 'REPRODUCE.md'}
    names.remove(Path(__file__).name)
    require(set(frozen['files']) - {Path(__file__).name} == names, 'complete frozen source set')
    for name, digest in frozen['files'].items():
        require(sha(ROOT/name) == digest, f'frozen source {name}')
    expected_refs = {PAUSED}
    for folder, status, names in (
        (BUDGET, 'SEALED_BUDGET_ONLY', {'TIMING.json', 'TIMING_AUDIT.json', 'WORK_STATUS.json'} |
         {f'trials/n256_r1/{path.name}' for path in (ROOT.parent/BUDGET/'trials/n256_r1').iterdir() if path.is_file()}),
        (EDGE, 'SEALED_FIXTURE_ONLY', {'fixture/edge.npz', 'fixture/weights.npz', 'fixture/edge_expectations.json'} |
         {f'evaluate/edge_{rule}_separate.npz' for rule in RULES}),
    ):
        manifest_path = ROOT.parent/folder/'MANIFEST.json'
        manifest = read(manifest_path)
        require(manifest['status'] == status, f'reference status {folder}')
        expected_refs.add(f'{folder}/MANIFEST.json')
        for name in names:
            relative = f'{folder}/{name}'
            expected_refs.add(relative)
            require(frozen['references'][relative] == manifest['files'][name]['sha256'], f'sealed hash {relative}')
    require(set(frozen['references']) == expected_refs, 'complete frozen reference set')
    for relative, digest in frozen['references'].items():
        require(sha(ROOT.parent/relative) == digest, f'frozen reference {relative}')
    paused = read(ROOT.parent/PAUSED)
    require(paused['status'] == 'STOPPED_BEFORE_FORMAL_BUDGET' and paused['scientific_data_generated'] is False
            and all(paused[key] == 0 for key in ('training_fits', 'formal_updates', 'formal_evaluations')),
            'formal confirmation remains stopped')


def validate_archive(folder, names, mode):
    require({path.name for path in folder.glob('*.npz')} == names, f'{folder.name}: complete NPZ set')
    for name in names:
        with ZipFile(folder/name) as archive:
            members = archive.infolist()
            require(members and len({member.filename for member in members}) == len(members)
                    and all(member.compress_type == mode and member.filename.endswith('.npy') for member in members),
                    f'{folder.name}/{name}: archive mode')


def validate_pass(result, examples, paths):
    require(result['status'] == 'PASS' and result['examples'] == examples and result['paths'] == paths,
            'saved comparison or audit status')
    require(type(result['checks']) is int and result['checks'] > 0, 'positive audit check count')
    require(nonnegative(result['max_numeric_error'], 'audit error') <= TOLERANCE, 'continuous tolerance')
    require(nonnegative(result['wall_seconds'], 'audit wall time') > 0, 'positive audit duration')


def verify_completed(folder, record, old_block, old_audit):
    name = record['directory']
    packet, mode = VARIANTS[record['variant']]
    audit = read(folder/'AUDIT.json')
    require(audit['wall_seconds'] <= record['audit_seconds']+TIME_TOLERANCE, f'{name}: audit nested time')
    require(not (folder/'FAILURE.txt').exists(), f'{name}: completed trial without failure')
    if name.startswith('edge_'):
        validate_pass(audit, 12, 3)
        validate_archive(folder, {f'edge_{rule}.npz' for rule in RULES}, mode)
        require(audit['arrays'] > 0 and audit['logical_array_bytes'] > 0, f'{name}: edge array ledger')
        return [audit]

    block, comparison = (read(folder/file) for file in ('BLOCK.json', 'BASELINE_COMPARISON.json'))
    validate_pass(audit, 256, 24)
    validate_pass(comparison, 256, 24)
    require(audit['packet_size'] == packet and audit['compression'] == mode, f'{name}: audited config')
    validate_archive(folder, PATHS | {'base.npz'}, mode)
    require(block['examples'] == 256 and block['paths'] == 24, f'{name}: block dimensions')
    phases = ('input_seconds', 'shared_seconds', 'verify_score_seconds', 'archive_seconds', 'isolation_seconds', 'other_seconds')
    equal_time(sum(nonnegative(block[key], f'{name}/{key}') for key in phases), block['total_seconds'],
               f'{name}: science phase total')
    require(0 < block['total_seconds'] <= record['science_seconds']+TIME_TOLERANCE, f'{name}: block nested time')
    phases = ('archive_load_seconds', 'independent_replay_seconds', 'comparison_bookkeeping_seconds')
    equal_time(sum(nonnegative(audit[key], f'{name}/{key}') for key in phases), audit['wall_seconds'], f'{name}: audit phase total')
    independent = nonnegative(record['independent_block_seconds'], f'{name}: independent wall')
    baseline = nonnegative(record['baseline_comparison_seconds'], f'{name}: baseline comparison wall')
    require(audit['wall_seconds'] <= independent+TIME_TOLERANCE
            and comparison['wall_seconds'] <= baseline+TIME_TOLERANCE
            and independent+baseline <= record['audit_seconds']+TIME_TOLERANCE, f'{name}: audit components nested')
    work = block['work']
    expected_calls = 8*(256//packet)
    require(work == old_block['work'] | {'shared_calls': expected_calls}, f'{name}: logical science work')
    require(audit['work'] == old_audit['work'], f'{name}: logical audit work')
    require(comparison['shared_calls'] == expected_calls and comparison['baseline_shared_calls'] == 64
            and comparison['isolated_pairs'] == block['metrics']['isolated_records'] == 32, f'{name}: call counts')
    raw = block['metrics']['retained_raw_array_bytes']
    require(raw == audit['retained_raw_array_bytes'] == old_block['metrics']['retained_raw_array_bytes'], f'{name}: raw bytes')
    require(comparison['arrays'] > 0 and comparison['logical_array_bytes'] == raw+block['metrics']['base_array_bytes']
            +block['metrics']['weights_array_bytes'], f'{name}: logical bytes')
    archived = sum((folder/file).stat().st_size for file in PATHS | {'base.npz', 'weights.npy', 'ROWS.json', 'MICROBENCH.json'})
    require(archived == comparison['archived_bytes'] == block['metrics']['archived_bytes_before_block']
            and comparison['baseline_archived_bytes'] == old_block['metrics']['archived_bytes_before_block'],
            f'{name}: physical science archive bytes')
    require(nonnegative(block['metrics']['max_numeric_error'], f'{name}: production residual') <= TOLERANCE,
            f'{name}: production tolerance')
    return [audit, comparison]


def validate_timing(record, totals, previous_finish):
    name = record['directory']
    begin, end = (nonnegative(record[f'monotonic_{key}'], f'{name}: {key}') for key in ('start', 'finish'))
    require(previous_finish <= begin <= end, f'{name}: non-overlapping trial order')
    require(stamp(record['utc_start']) <= stamp(record['utc_finish']), f'{name}: outer UTC order')
    reserve = 5 if name.startswith('edge_') else 25
    cursor = begin
    for stage in CAPS:
        keys = [f'{stage}_{part}' for part in ('utc_start', 'utc_finish', 'monotonic_start', 'monotonic_finish', 'seconds')]
        present = [key in record for key in keys]
        if not any(present):
            continue
        require(all(present), f'{name}: full {stage} ledger')
        require(stage == 'science' or 'science_seconds' in record, f'{name}: science precedes audit')
        require(CAPS[stage]-totals[stage] >= reserve, f'{name}: {stage} admission reserve')
        start, finish = (nonnegative(record[key], f'{name}/{key}') for key in keys[2:4])
        duration = nonnegative(record[keys[4]], f'{name}/{keys[4]}')
        require(cursor <= start <= finish <= end, f'{name}: nested {stage} monotonic interval')
        require(stamp(record['utc_start']) <= stamp(record[keys[0]]) <= stamp(record[keys[1]]) <= stamp(record['utc_finish']),
                f'{name}: nested {stage} UTC interval')
        equal_time(finish-start, duration, f'{name}: {stage} monotonic duration')
        totals[stage] += duration
        cursor = finish
    for key in ('load_before', 'load_after'):
        snapshot = record[key]
        require(stamp(snapshot['utc_start']) <= stamp(snapshot['utc_finish']), f'{name}: {key} timestamps')
        if snapshot['status'] == 'ok':
            load = snapshot['percentages']
            require(isinstance(load, list) and load and all(type(v) is int and 0 <= v <= 100 for v in load), f'{name}: CPU snapshot')
        else:
            require(snapshot['status'] == 'unavailable' and snapshot['percentages'] is None and snapshot.get('error'),
                    f'{name}: missing snapshot reason')
    return end


def verify():
    frozen, summary = (read(ROOT/name) for name in ('FREEZE.json', 'SUMMARY.json'))
    validate_freeze(frozen)
    complete = summary['status'] == 'COMPLETE_ENGINEERING_ONLY'
    require(complete or summary['status'] == 'STOPPED', 'recognized study status')
    require(summary['formal_training_fits'] == summary['formal_evaluations'] == 0
            and summary['formal_data_generated'] is False, 'no formal run')
    require(all(not (ROOT/name).exists() for name in ('train', 'evaluate', 'TEST_START.json')), 'no formal run folders')
    require(stamp(frozen['utc']) <= stamp(summary['utc_start']) <= stamp(summary['utc_finish']), 'study UTC order')
    require((summary['failure'] is None and not (ROOT/'FAILURE.txt').exists()) if complete else
            (summary['failure'] and (ROOT/'FAILURE.txt').is_file()), 'study failure ledger')
    expected = [dict(variant=name, directory=f'edge_{name}') for name in VARIANTS]
    expected.extend(dict(repeat=repeat, variant=name, directory=f'r{repeat}_{name}',
                         packet_size=VARIANTS[name][0], zip_mode=VARIANTS[name][1])
                    for repeat, order in enumerate(ROUNDS, 1) for name in order)
    records = summary['edges']+summary['trials']
    require(len(summary['edges']) <= 3 and (not summary['trials'] or len(summary['edges']) == 3), 'edge/trial ordering')
    require(len(records) <= len(expected) and (not complete or len(records) == len(expected)), 'complete design or stopped prefix')
    folders = {path.name for path in (ROOT/'results').iterdir()}
    require(folders <= {record['directory'] for record in records}, 'no extra runs')
    reference = ROOT.parent/BUDGET/'trials/n256_r1'
    old_block, old_audit = (read(reference/name) for name in ('BLOCK.json', 'AUDIT.json'))
    totals = dict(science=0., audit=0.)
    last_finish = 0.
    checks, max_error, completed_edges, completed_trials = 0, 0., 0, 0
    recorded_bytes = 0
    for index, (record, condition) in enumerate(zip(records, expected)):
        name = condition['directory']
        require(all(record.get(key) == value for key, value in condition.items()), f'{name}: fixed run order and settings')
        finished = record.get('stage') == 'complete'
        require(finished or (not complete and index == len(records)-1), f'{name}: incomplete tail only')
        folder = ROOT/'results'/name
        timing = folder/'TIMING.json'
        if not timing.is_file():
            require(not finished and not any(key.endswith('_seconds') for key in record), f'{name}: no unrecorded stage duration')
            continue
        require(read(timing) == record, f'{name}: per-run timing equals summary')
        last_finish = validate_timing(record, totals, last_finish)
        saved_bytes = sum(path.stat().st_size for path in folder.rglob('*') if path.is_file() and path != timing)
        require(saved_bytes == record['trial_bytes'], f'{name}: preserved run bytes')
        recorded_bytes += saved_bytes + timing.stat().st_size
        peak = nonnegative(record['peak_working_set_bytes'], f'{name}: peak bytes')
        require(0 < peak <= summary['peak_working_set_bytes'], f'{name}: cumulative process peak')
        if finished:
            require('error' not in record and 'science_seconds' in record and 'audit_seconds' in record, f'{name}: full phases')
            require(peak <= MEMORY_CAP and all(totals[stage] <= CAPS[stage] for stage in CAPS), f'{name}: completed cap checks')
            for result in verify_completed(folder, record, old_block, old_audit):
                checks += result['checks']
                max_error = max(max_error, result['max_numeric_error'])
            completed_edges += name.startswith('edge_')
            completed_trials += name.startswith('r')
        else:
            require(record.get('error') == summary['failure'] and (folder/'FAILURE.txt').is_file(), f'{name}: saved interrupted phase')

    require(set(summary['cumulative_stage_seconds']) == set(CAPS), 'cumulative stage keys')
    for stage in CAPS:
        equal_time(totals[stage], summary['cumulative_stage_seconds'][stage], f'cumulative {stage} seconds')
    outside = nonnegative(summary['outside_stage_seconds'], 'outside-stage time')
    equal_time(sum(totals.values())+outside, summary['wall_seconds'], 'study wall and outside-stage accounting')
    observed = nonnegative(summary['completed_trial_observation_and_bookkeeping_seconds'], 'observation time')
    require(observed <= outside+TIME_TOLERANCE, 'observation time lies outside guarded stages')
    require(recorded_bytes <= summary['artifact_bytes'], 'study bytes include attempted run records')
    if complete:
        require((completed_edges, completed_trials) == (3, 6), 'all runs completed and verified')
        require(0 < summary['peak_working_set_bytes'] <= MEMORY_CAP and summary['artifact_bytes'] <= ARTIFACT_CAP,
                'study resource caps')
    return dict(status='PASS' if complete else 'STOPPED_VERIFIED', completed_edges=completed_edges,
                completed_trials=completed_trials, attempted_runs=len(records), numeric_checks=checks,
                max_numeric_error=max_error, cumulative_stage_seconds=totals,
                frozen_source_files=len(frozen['files']), frozen_reference_files=len(frozen['references']),
                verifier_in_source_freeze=Path(__file__).name in frozen['files'],
                formal_training_fits=0, formal_evaluations=0, formal_data_generated=False)


def main():
    started = time.perf_counter()
    result = verify()
    result['wall_seconds'] = time.perf_counter()-started
    result['audit_seconds_including_ledger_verification'] = result['cumulative_stage_seconds']['audit']+result['wall_seconds']
    if result['status'] == 'PASS':
        require(result['audit_seconds_including_ledger_verification'] < CAPS['audit'], 'complete combined audit cap')
    with (ROOT/'VERIFY.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
