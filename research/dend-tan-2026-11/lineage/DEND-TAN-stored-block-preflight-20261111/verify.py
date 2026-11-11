"""Audit the frozen budget ledger and saved evidence; never invoke a model."""

from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time
from zipfile import ZipFile, ZIP_STORED


ROOT = Path(__file__).resolve().parent
DESIGN = dict(packet_size=256, zip_mode=0, examples=[512, 256], repeats=2,
              gap_seconds=30, blocks=12, science_cap_seconds=180,
              audit_cap_seconds=300, train_cap_seconds=600,
              memory_cap_bytes=1073741824, artifact_cap_bytes=1610612736)
BASELINES = {512: 'DEND-TAN-shared-block-budget-20261111/trials/n512_r1',
             256: 'DEND-TAN-block-optimization-20261111/results/r1_C'}
EDGE = 'DEND-TAN-shared-replay-fixture-20261111'
PAUSED = 'DEND-TAN-evicted-key-confirm-20261111/WORK_STATUS.json'
RULES = ('original', 'replay', 'guarded')
CONDITIONS = {f'{g}_{h}_{r}' for g in ('orthogonal', 'near')
              for h in ('four', 'three0', 'three2', 'two') for r in RULES}
ARCHIVES = {name+'.npz' for name in CONDITIONS} | {'base.npz'}
SCIENCE_FILES = ARCHIVES | {'weights.npy', 'ROWS.json', 'MICROBENCH.json'}
TOL = 1e-6
CHECKS = 0


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, context):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise ValueError(context)


def number(value, context, positive=False):
    require(type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0), context)
    return value


def close(left, right, context):
    number(left, context)
    number(right, context)
    require(abs(left-right) <= TOL, context)


def stamp(value):
    result = datetime.fromisoformat(value)
    require(result.utcoffset() is not None and result.utcoffset().total_seconds() == 0, 'explicit UTC timestamp')
    return result


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def size(folder):
    return sum(path.stat().st_size for path in folder.rglob('*') if path.is_file())


def checked_path(base, name):
    path = (base/name).resolve()
    require(path.is_relative_to(base.resolve()) and path.is_file(), f'contained existing path: {name}')
    return path


def freeze(frozen):
    require(frozen['study'] == 'STORED_FULL_BLOCK_BUDGET_ONLY', 'fixed study scope')
    require(frozen['design'] == DESIGN, 'exact frozen design')
    require(frozen['gap_jitter_limit_seconds'] == .25 and frozen['fixture_data_seed'] == 4099903301
            and frozen['native_init_seed'] == 4099903302, 'fixed jitter bound and synthetic seeds')
    require(frozen['training_fit_upper_bytes'] == 2*1024**2
            and frozen['metadata_reserve_bytes'] == 64*1024**2, 'fixed formal artifact reserves')
    close(frozen['training_audit_forecast_seconds'], 9.150149999186397, 'fixed previous training audit forecast')
    files = frozen['files']
    require({'verify.py', 'preflight.py', 'block.py', 'block_audit.py', 'compare.py', 'PROTOCOL.md'} <= set(files),
            'all execution and verification entry points frozen')
    require({path.name for path in ROOT.glob('*.py')} <= set(files), 'no unfrozen local Python source')
    for name, digest in files.items():
        require(sha(checked_path(ROOT, name)) == digest, f'frozen source hash: {name}')
    references = frozen['references']
    required = {PAUSED}
    for relative in BASELINES.values():
        folder = ROOT.parent/relative
        required.update(f'{relative}/{path.name}' for path in folder.iterdir() if path.is_file())
        top = relative.split('/')[0]
        required.add(f'{top}/MANIFEST.json')
        manifest = read(ROOT.parent/top/'MANIFEST.json')
        for path in folder.iterdir():
            if path.is_file():
                sub = path.relative_to(ROOT.parent/top).as_posix()
                require(references.get(f'{relative}/{path.name}') == manifest['files'][sub]['sha256'],
                        f'sealed reference lineage: {relative}/{path.name}')
    required |= {f'{EDGE}/MANIFEST.json', f'{EDGE}/fixture/edge.npz', f'{EDGE}/fixture/weights.npz'}
    required |= {f'{EDGE}/evaluate/edge_{rule}_separate.npz' for rule in RULES}
    require(required <= set(references), 'complete fixed same-size and edge reference set')
    for relative, digest in references.items():
        require(sha(checked_path(ROOT.parent, relative)) == digest, f'frozen reference hash: {relative}')
    old = read(ROOT.parent/PAUSED)
    require(old['status'] == 'STOPPED_BEFORE_FORMAL_BUDGET' and old['scientific_data_generated'] is False
            and all(old[key] == 0 for key in ('training_fits', 'formal_updates', 'formal_evaluations')),
            'formal predecessor remains stopped')


def load(snapshot, context):
    begin, finish = stamp(snapshot['utc_start']), stamp(snapshot['utc_finish'])
    require(begin <= finish, f'{context}: load observation timestamps')
    if snapshot['status'] == 'ok':
        values = snapshot['percentages']
        require(isinstance(values, list) and bool(values)
                and all(type(value) is int and 0 <= value <= 100 for value in values), f'{context}: host load values')
    else:
        require(snapshot['status'] == 'unavailable' and snapshot['percentages'] is None
                and bool(snapshot.get('error')), f'{context}: failed load observation retained')
    return begin, finish


def envelope(record, previous, frozen_utc, totals, complete):
    context = record['directory']
    folder = ROOT/context
    require(folder.resolve().is_relative_to(ROOT.resolve()), f'{context}: contained run folder')
    pending = record['stage'] == 'pending'
    require((not complete and pending and not folder.exists()) or folder.is_dir(), f'{context}: run folder state')
    begin = number(record['monotonic_start_seconds'], f'{context}: start')
    finish = number(record['monotonic_finish_seconds'], f'{context}: finish')
    require(begin < finish, f'{context}: positive outer interval')
    utc_begin, utc_end = stamp(record['utc_start']), stamp(record['utc_finish'])
    before, after = load(record['load_before'], context), load(record['load_after'], context)
    require(frozen_utc <= before[0] <= before[1] <= utc_begin <= utc_end <= after[0] <= after[1],
            f'{context}: observations outside measured interval')
    if 'gap_from_previous_seconds' in record:
        gap = record['gap_from_previous_seconds']
        if previous is None:
            require(gap is None, f'{context}: first repetition has no scheduled gap')
            require(record['target_monotonic_start_seconds'] is None, f'{context}: no first-repetition deadline')
        else:
            close(gap, begin-previous['monotonic_finish_seconds'], f'{context}: measured gap reconciles')
            close(record['target_monotonic_start_seconds'], previous['monotonic_finish_seconds']+30,
                  f'{context}: frozen scheduled deadline')
            if pending:
                require(not 30. <= gap <= 30.25 and 'fixed 30s gap missed' in record.get('error', ''),
                        f'{context}: explicit missed-gap stop before any science')
            else:
                require(30. <= gap <= 30.25, f'{context}: frozen scheduled gap and wake jitter')
    elif previous is not None:
        require(False, f'{context}: missing scheduled gap record')
    total = 0.
    for kind in ('science', 'audit'):
        if f'{kind}_seconds' in record:
            duration = number(record[f'{kind}_seconds'], f'{context}: {kind} duration', True)
            totals[kind] += duration
            total += duration
        elif complete:
            require(False, f'{context}: missing complete {kind} duration')
    require(total <= finish-begin+TOL, f'{context}: stages fit outer interval')
    if pending:
        require(total == 0 and previous is not None, f'{context}: missed-gap stop performs no stages')
    actual_bytes = size(folder)
    require(record['artifact_bytes'] == actual_bytes, f'{context}: complete retained byte count')
    peak = number(record['peak_working_set_bytes'], f'{context}: process peak', True)
    if complete:
        require(peak <= DESIGN['memory_cap_bytes'] and actual_bytes <= DESIGN['artifact_cap_bytes'],
                f'{context}: completed resource caps')
    comparison_name = 'EDGE_COMPARISON.json' if context == 'edge' else 'BASELINE_COMPARISON.json'
    for file, field in (('BLOCK.json', 'block_sha256'), ('AUDIT.json', 'audit_sha256'),
                        (comparison_name, 'comparison_sha256')):
        require((folder/file).is_file() == (field in record), f'{context}: {file} hash presence')
        if field in record:
            require(sha(folder/file) == record[field], f'{context}: {file} hash')
    require(read(ROOT/'progress'/(Path(context).name+'.json')) == record, f'{context}: preserved progress equals final ledger')
    return folder


def passed(result, n, paths):
    require(result['status'] == 'PASS' and result['examples'] == n and result['paths'] == paths,
            'saved independent audit scope and status')
    require(type(result['checks']) is int and result['checks'] > 0, 'positive independent checks')
    require(number(result['max_numeric_error'], 'numeric residual') <= 5e-12, 'fixed continuous tolerance')
    number(result['wall_seconds'], 'independent audit wall time', True)


def archives(folder, names):
    require({path.name for path in folder.glob('*.npz')} == names, f'{folder.name}: full archive set')
    for name in names:
        with ZipFile(folder/name) as stream:
            members = stream.infolist()
            require(bool(members) and len({item.filename for item in members}) == len(members)
                    and all(item.compress_type == ZIP_STORED and item.filename.endswith('.npy') for item in members),
                    f'{folder.name}/{name}: uncompressed lossless archive')


def untimed(rows):
    return [{key: value for key, value in row.items() if not key.endswith('_seconds')} for row in rows]


def complete_block(folder, record, n, frozen):
    require({path.name for path in folder.iterdir()} == SCIENCE_FILES | {'BLOCK.json', 'AUDIT.json', 'BASELINE_COMPARISON.json'},
            f'{folder.name}: exactly complete block evidence')
    block, audit, comparison = (read(folder/name) for name in ('BLOCK.json', 'AUDIT.json', 'BASELINE_COMPARISON.json'))
    passed(audit, n, 24)
    passed(comparison, n, 24)
    require(block['examples'] == n and block['paths'] == 24 and audit['packet_size'] == 256
            and audit['compression'] == ZIP_STORED, 'complete dimensions and fixed runner')
    archives(folder, ARCHIVES)
    baseline = ROOT.parent/BASELINES[n]
    old_block, old_audit = (read(baseline/name) for name in ('BLOCK.json', 'AUDIT.json'))
    require(Path(comparison['baseline_directory']).resolve() == baseline.resolve(), 'correct same-size sealed baseline')
    expected_calls = 8*(n//256)
    require(block['work'] == old_block['work'] | {'shared_calls': expected_calls}, 'unchanged complete logical science work')
    require(audit['work'] == old_audit['work'], 'unchanged complete independent audit work')
    require(comparison['shared_calls'] == expected_calls
            and comparison['baseline_shared_calls'] == (128 if n == 512 else 8)
            and comparison['baseline_packet_size'] == (32 if n == 512 else 256), 'exact packet call counts')
    require(comparison['trace_examples_per_path'] == 8 and comparison['trace_time_steps'] == 36,
            'global first-eight traces for all paths')
    phases = ('input_seconds', 'shared_seconds', 'verify_score_seconds', 'archive_seconds', 'isolation_seconds', 'other_seconds')
    close(sum(number(block[key], key) for key in phases), block['total_seconds'], 'complete science phase ledger')
    require(0 < block['total_seconds'] <= record['science_seconds']+TOL, 'science duration contains all phases')
    independent = number(record['independent_block_seconds'], 'independent block seconds', True)
    reference = number(record['reference_compare_seconds'], 'reference comparison seconds', True)
    require(audit['wall_seconds'] <= independent+TOL and comparison['wall_seconds'] <= reference+TOL
            and independent+reference <= record['audit_seconds']+TOL, 'total audit includes both full audits')
    close(sum(number(audit[key], key) for key in ('archive_load_seconds', 'independent_replay_seconds',
                                                'comparison_bookkeeping_seconds')), audit['wall_seconds'], 'independent phase ledger')
    close(audit['prior_training_audit_forecast_seconds'], frozen['training_audit_forecast_seconds'], 'training audit estimate')
    rows, singles = read(folder/'ROWS.json'), read(folder/'MICROBENCH.json')
    require(len(rows) == 24 and {row['condition'] for row in rows} == CONDITIONS
            and all(row['examples'] == n for row in rows), 'all 24 complete sample paths scored')
    require(len(singles) == comparison['isolated_pairs'] == block['metrics']['isolated_records'] == 32,
            'all 32 isolated pairs retained')
    require(untimed(singles) == untimed(read(baseline/'MICROBENCH.json')), 'unchanged isolated selection, order and work')
    require({(row['geometry'], row['schedule']) for row in singles} ==
            {(geometry, schedule) for geometry in ('orthogonal', 'near') for schedule in 'ABCD'},
            'all geometry and schedule isolation cells')
    for index, row in enumerate(singles):
        require(row['order'] == (['replay', 'guarded'] if index % 2 == 0 else ['guarded', 'replay']), 'fixed alternating call order')
        require(0 <= row['sample'] < n, 'in-range isolated sample')
        for key, value in row.items():
            if key.endswith('_seconds'):
                number(value, 'isolated timing')
    raw = block['metrics']['retained_raw_array_bytes']
    require(raw == audit['retained_raw_array_bytes'] == old_block['metrics']['retained_raw_array_bytes'], 'complete logical raw bytes')
    require(comparison['logical_array_bytes'] == raw+block['metrics']['base_array_bytes']+block['metrics']['weights_array_bytes'],
            'comparison includes all raw arrays, inputs and weights')
    stored = sum((folder/name).stat().st_size for name in SCIENCE_FILES)
    require(stored == block['metrics']['archived_bytes_before_block'] == comparison['archived_bytes'], 'physical science archive bytes')
    require(block['work']['spot_calls'] == 24 and block['work']['isolated_calls'] == 64
            and audit['work']['isolated_pairs'] == 32, 'full production spot and isolation work')
    return dict(directory=record['directory'], examples=n, checks=audit['checks']+comparison['checks'],
                max_numeric_error=max(audit['max_numeric_error'], comparison['max_numeric_error']),
                work=block['work'], audit_work=audit['work'])


def candidate_forecast(candidate, frozen):
    trials = candidate['trials']
    require(len(trials) == 2 and all(row['stage'] == 'complete' for row in trials), 'forecast after two complete repetitions')
    forecast = {}
    for kind in ('science', 'audit'):
        worst = max(row[f'{kind}_seconds'] for row in trials)
        close(candidate[f'{kind}_seconds'], worst, f'worst {kind} duration')
        forecast[kind] = 36*worst+30+(frozen['training_audit_forecast_seconds'] if kind == 'audit' else 0.)
        close(candidate[f'{kind}_forecast_seconds'], forecast[kind], f'exact {kind} forecast')
    artifact = 12*max(row['artifact_bytes'] for row in trials)+12*frozen['training_fit_upper_bytes']+frozen['metadata_reserve_bytes']
    require(candidate['formal_artifact_forecast_bytes'] == artifact, 'formal artifact forecast includes fixed reserves')
    require(candidate['test_artifact_forecast_bytes'] == 12*max(row['artifact_bytes'] for row in trials)
            and candidate['training_artifact_forecast_bytes'] == 12*frozen['training_fit_upper_bytes'], 'artifact forecast components')
    require(candidate['peak_working_set_bytes'] == max(row['peak_working_set_bytes'] for row in trials), 'candidate peak working set')
    within = (forecast['science'] < 180 and forecast['audit'] < 300
              and max(row['peak_working_set_bytes'] for row in trials) <= DESIGN['memory_cap_bytes']
              and artifact <= DESIGN['artifact_cap_bytes'])
    require(candidate['within_budget'] is within, 'correct complete budget decision')
    return within


def verify():
    frozen, timing = (read(ROOT/name) for name in ('FREEZE.json', 'TIMING.json'))
    freeze(frozen)
    frozen_utc = stamp(frozen['utc'])
    stopped = timing['status'] == 'STOPPED_ERROR'
    require(stopped or timing['status'] in ('FEASIBLE', 'INFEASIBLE'), 'recognized budget status')
    require(timing['blocks'] == 12 and timing['scientific_data_generated'] is False
            and timing['formal_training_fits'] == timing['formal_evaluations'] == 0, 'fixture-only timing scope')
    close(timing['training_audit_forecast_seconds'], frozen['training_audit_forecast_seconds'], 'timing training estimate')
    require(all(not (ROOT/name).exists() for name in ('train', 'evaluate', 'weights', 'TEST_START.json',
                                                   'TRAIN_SUMMARY.json', 'SCIENCE_SUMMARY.json')), 'no formal execution artifacts')
    if stopped:
        require(bool(timing.get('failure')) and (ROOT/'FAILURE.txt').is_file(), 'explicit preserved stop reason')
    else:
        require(timing.get('failure') is None and not (ROOT/'FAILURE.txt').exists(), 'complete run without failure')
    totals = dict(science=0., audit=0.)
    edge = timing.get('edge')
    completed = []
    last_finish = 0.
    if edge is not None:
        finished = edge['stage'] == 'complete'
        folder = envelope(edge, None, frozen_utc, totals, finished)
        last_finish = edge['monotonic_finish_seconds']
        if finished:
            require({path.name for path in folder.iterdir()} == {f'edge_{rule}.npz' for rule in RULES} | {'EDGE_COMPARISON.json'},
                    'complete edge artifacts')
            archives(folder, {f'edge_{rule}.npz' for rule in RULES})
            comparison = read(folder/'EDGE_COMPARISON.json')
            passed(comparison, 12, 3)
            require(comparison['wall_seconds'] <= edge['audit_seconds']+TOL, 'edge audit timing boundary')
        else:
            require(stopped and not timing['candidates'], 'edge failure stops before block trials')
    else:
        require(stopped and not timing['candidates'], 'missing edge only before any execution')
    candidates = timing['candidates']
    require([item['examples'] for item in candidates] in ([], [512], [512, 256]), 'fixed tier order without retries')
    selected = None
    directories = set()
    for index, candidate in enumerate(candidates):
        n, trials = candidate['examples'], candidate['trials']
        require(edge is not None and edge['stage'] == 'complete', 'edge validation precedes all blocks')
        require(candidate['fixed_gap_seconds'] == 30 and len(trials) <= 2
                and [row['trial'] for row in trials] == list(range(1, len(trials)+1)), 'fixed repetition prefix')
        if index:
            require(candidate_forecast(candidates[index-1], frozen) is False, 'fallback only after previous tier fails')
        current = []
        for position, trial in enumerate(trials):
            require(trial['directory'] == f'trials/n{n}_r{position+1}', 'fixed block artifact path')
            directories.add(trial['directory'])
            finished = trial['stage'] == 'complete'
            require(finished or (stopped and index == len(candidates)-1 and position == len(trials)-1), 'partial final trial only')
            require(trial['monotonic_start_seconds'] >= last_finish, 'strict non-overlapping execution sequence')
            folder = envelope(trial, trials[position-1] if position else None, frozen_utc, totals, finished)
            last_finish = trial['monotonic_finish_seconds']
            if finished:
                current.append(complete_block(folder, trial, n, frozen))
        if len(current) == 2:
            require(current[0]['work'] == current[1]['work'] and current[0]['audit_work'] == current[1]['audit_work'],
                    'identical repeated fixture logical work')
            within = candidate_forecast(candidate, frozen)
            if within:
                require(index == len(candidates)-1, 'stop measuring after first feasible size')
                selected = n
        else:
            require(stopped and index == len(candidates)-1, 'incomplete candidate only in stopped tail')
        completed.extend(current)
    trials_root = ROOT/'trials'
    actual = {path.relative_to(ROOT).as_posix() for path in trials_root.iterdir()} if trials_root.exists() else set()
    pending_directories = {row['directory'] for candidate in candidates for row in candidate['trials'] if row['stage'] == 'pending'}
    require(actual == directories-pending_directories, 'no hidden or unrecorded trial folders')
    progress_names = {Path(name).name+'.json' for name in directories} | ({'edge.json'} if edge else set())
    require({path.name for path in (ROOT/'progress').iterdir()} == progress_names, 'exact progress record set')
    if not stopped:
        require(selected is not None or len(candidates) == 2, 'infeasible follows both complete tiers')
        require(timing['status'] == ('FEASIBLE' if selected else 'INFEASIBLE'), 'independently derived final feasibility')
        require(timing['selected_examples_per_block'] == selected, 'same final size decision')
    for kind in totals:
        close(timing[f'cumulative_{kind}_seconds'], totals[kind], f'all edge and block {kind} charged')
        if not stopped:
            require(totals[kind] <= DESIGN[f'{kind}_cap_seconds'], f'cumulative {kind} cap')
    artifact = size(ROOT)
    close(sum(totals.values())+number(timing['outside_stage_seconds'], 'outside-stage time'), timing['wall_seconds'],
          'full wall clock includes waiting and observation')
    require(number(timing['artifact_bytes'], 'recorded total artifact bytes') <= artifact, 'recorded bytes contained in retained outputs')
    if not stopped:
        require(artifact <= DESIGN['artifact_cap_bytes'] and 0 < timing['peak_working_set_bytes'] <= DESIGN['memory_cap_bytes'],
                'whole-run artifact and memory caps')
    return dict(status='STOPPED_VERIFIED' if stopped else 'PASS', timing_status=timing['status'], checks=CHECKS,
                selected_examples_per_block=selected, completed_trials=len(completed), trials=completed,
                cumulative_stage_seconds=totals, artifact_bytes_at_verification=artifact,
                frozen_source_files=len(frozen['files']), frozen_reference_files=len(frozen['references']),
                formal_training_fits=0, formal_evaluations=0, scientific_data_generated=False,
                scope='Saved hashes, frozen same-size references, complete 24-path and 32-pair ledgers, fixed gap, all-audit and artifact budgets; no model invocation.')


def main():
    target = ROOT/'VERIFY.json'
    if target.exists():
        raise FileExistsError(target)
    started = time.perf_counter()
    result = verify()
    result['wall_seconds'] = time.perf_counter()-started
    result['audit_seconds_including_verification'] = result['cumulative_stage_seconds']['audit']+result['wall_seconds']
    if result['status'] == 'PASS':
        require(result['audit_seconds_including_verification'] <= DESIGN['audit_cap_seconds'], 'combined audit cap including verification')
    result['checks'] = CHECKS
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
