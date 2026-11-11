"""Verify the fixed full-block budget decision from saved timing and audit records."""

import hashlib
import json
import math
import time
from datetime import datetime, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PREDECESSOR = ROOT.parent / 'DEND-TAN-evicted-key-confirm-20261111'
TRAINING_AUDIT_SECONDS = 9.150149999186397
BLOCKS = 12
SCIENCE_CAP = 180.
AUDIT_CAP = 300.
GAP = 30.
MEMORY_CAP = 2**30
ARTIFACT_CAP = int(1.5 * 2**30)
DATA_SEED = 4099903301
WEIGHT_SEED = 4099903302
GEOMETRIES = ('orthogonal', 'near')
HISTORIES = ('four', 'three0', 'three2', 'two')
RULES = ('original', 'replay', 'guarded')


class TimingAudit:
    def __init__(self):
        self.checks = 0
        self.start = time.perf_counter()

    def check(self, condition, message):
        self.checks += 1
        if not condition:
            raise ValueError(message)

    def close(self, actual, expected, message, tolerance=1e-6):
        self.check(finite(actual) and finite(expected) and abs(actual - expected) <= tolerance,
                   f'{message}: found {actual!r}, expected {expected!r}')


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


def parse_utc(audit, value, context):
    observed = datetime.fromisoformat(value)
    audit.check(observed.utcoffset() == timedelta(0), f'{context}: timestamp explicitly in UTC')
    return observed


def verify_load(audit, load, context):
    observed = parse_utc(audit, load['utc'], context)
    audit.check(load['status'] in ('ok', 'unavailable'), f'{context}: observed load status')
    values = load['percentages']
    if load['status'] == 'ok':
        audit.check(isinstance(values, list) and bool(values)
                    and all(finite(value) and 0. <= value <= 100. for value in values),
                    f'{context}: finite host load percentages')
    else:
        audit.check(values is None and bool(load.get('error')), f'{context}: unavailable load retained')
    return observed


def verify_sources(audit):
    frozen = read(ROOT / 'FREEZE.json')
    expected = dict(study='FULL_BLOCK_BUDGET_ONLY', fixture_data_generated=False, blocks=BLOCKS,
                    science_cap_seconds=SCIENCE_CAP, audit_cap_seconds=AUDIT_CAP, fixture_data_seed=DATA_SEED,
                    native_init_seed=WEIGHT_SEED, candidate_sizes=[512, 256], repeats_per_candidate=2,
                    fixed_gap_seconds=GAP, training_audit_forecast_seconds=TRAINING_AUDIT_SECONDS,
                    memory_cap_bytes=MEMORY_CAP, artifact_cap_bytes=ARTIFACT_CAP)
    for key, value in expected.items():
        audit.check(frozen[key] == value, f'frozen {key}')
    files = frozen['files']
    audit.check({'PROTOCOL.md', 'SOURCE_LOCK.json', 'audit_timing.py'} <= set(files), 'required frozen sources')
    for name, expected_digest in files.items():
        path = (ROOT / name).resolve()
        audit.check(path.is_relative_to(ROOT) and path.is_file(), f'frozen path inside study: {name}')
        audit.check(sha256(path) == expected_digest, f'frozen digest: {name}')

    lock = read(ROOT / 'SOURCE_LOCK.json')
    prior_fixture = ROOT.parent / 'DEND-TAN-shared-replay-fixture-20261111'
    audit.check((ROOT / lock['previous_directory']).resolve() == prior_fixture, 'fixed source fixture directory')
    audit.check((ROOT / lock['blocked_directory']).resolve() == PREDECESSOR, 'fixed stopped predecessor directory')
    audit.check(sha256(prior_fixture / 'MANIFEST.json') == lock['previous_manifest_sha256'], 'source fixture manifest seal')
    for name, field in (('PREFLIGHT_MANIFEST.json', 'blocked_manifest_sha256'),
                        ('TIMING.json', 'blocked_timing_sha256'), ('PRECHECK.json', 'blocked_precheck_sha256')):
        audit.check(sha256(PREDECESSOR / name) == lock[field], f'stopped predecessor {name} seal')
    blocked = read(PREDECESSOR / 'TIMING.json')
    blocked_precheck = read(PREDECESSOR / 'PRECHECK.json')
    stopped = read(PREDECESSOR / 'WORK_STATUS.json')
    manifest = read(PREDECESSOR / 'PREFLIGHT_MANIFEST.json')
    audit.check(sha256(PREDECESSOR / 'WORK_STATUS.json') == manifest['files']['WORK_STATUS.json']['sha256'],
                'stopped predecessor decision seal')
    audit.check(blocked['status'] == 'INFEASIBLE' and blocked['examples_per_block'] is None
                and blocked['scientific_data_generated'] is False, 'stopped predecessor remains infeasible')
    audit.check(stopped['status'] == 'STOPPED_BEFORE_FORMAL_BUDGET' and stopped['training_fits'] == 0
                and stopped['formal_updates'] == stopped['formal_evaluations'] == 0
                and stopped['scientific_data_generated'] is False, 'stopped predecessor remains unrun')
    audit.check(not any((PREDECESSOR / name).exists() for name in ('FREEZE.json', 'train', 'evaluate')),
                'stopped predecessor has no formal execution artifacts')
    durations = [row['training_audit_seconds'] for row in blocked_precheck['train_fixtures']]
    audit.check(len(durations) == 2 and all(finite(value) and value > 0 for value in durations),
                'both sealed training-audit fixture durations')
    derived = BLOCKS * 600 / 24 * max(durations)
    audit.close(derived, TRAINING_AUDIT_SECONDS, 'recomputed archived training-audit forecast', 1e-12)
    audit.close(blocked['training_audit_forecast_seconds'], TRAINING_AUDIT_SECONDS, 'sealed training audit forecast', 1e-12)
    audit.close(lock['training_audit_forecast_seconds'], TRAINING_AUDIT_SECONDS, 'copied training audit forecast', 1e-12)
    fixture_manifest = read(prior_fixture / 'MANIFEST.json')
    source_files = fixture_manifest['files']
    for name, expected_digest in lock['files'].items():
        audit.check(files[name] == expected_digest == source_files[name]['sha256'], f'source copy lineage: {name}')
    return frozen


def verify_trial_envelope(audit, examples, trial, previous, frozen_utc):
    number = trial['trial']
    context = f'n={examples} trial={number}'
    directory = f'trials/n{examples}_r{number}'
    audit.check(trial['directory'] == directory, f'{context}: fixed archive directory')
    stage = ROOT / directory
    start, finish = (trial[f'monotonic_{key}_seconds'] for key in ('start', 'finish'))
    audit.check(finite(start) and finite(finish) and 0. <= start < finish, f'{context}: bounded monotonic interval')
    utc_start = parse_utc(audit, trial['utc_start'], f'{context} start')
    utc_finish = parse_utc(audit, trial['utc_finish'], f'{context} finish')
    before = verify_load(audit, trial['load_before'], f'{context} before load')
    after = verify_load(audit, trial['load_after'], f'{context} after load')
    audit.check(frozen_utc <= before <= utc_start <= utc_finish <= after,
                f'{context}: UTC bounds follow freeze and load snapshots outside measured interval')
    gap = trial['gap_from_previous_seconds']
    if previous is None:
        audit.check(gap is None, f'{context}: first trial records no predecessor')
    else:
        audit.check(finite(gap) and gap >= GAP, f'{context}: fixed 30-second monotonic gap met')
        audit.close(gap, start - previous['monotonic_finish_seconds'], f'{context}: recorded gap reconciles')
        previous_utc_finish = parse_utc(audit, previous['utc_finish'], f'{context} previous finish')
        audit.check(previous_utc_finish <= utc_start, f'{context}: sequential UTC trial bounds')
    measured = [trial[key] for key in ('science_seconds', 'audit_seconds') if key in trial]
    audit.check(measured and all(finite(value) and value > 0. for value in measured), f'{context}: positive measured durations')
    audit.check(sum(measured) <= finish - start + 1e-6,
                f'{context}: stage durations fit monotonic trial bounds')
    actual_bytes = sum(path.stat().st_size for path in stage.rglob('*') if path.is_file())
    audit.check(trial['artifact_bytes'] == actual_bytes, f'{context}: retained artifact byte count')
    for name, field in (('BLOCK.json', 'block_sha256'), ('AUDIT.json', 'audit_sha256')):
        audit.check((stage / name).is_file() == (field in trial), f'{context}: {name} hash presence')
        if field in trial:
            audit.check(sha256(stage / name) == trial[field], f'{context}: saved {name} digest')
    return context, stage


def verify_trial(audit, examples, trial, previous, frozen_utc):
    context, stage = verify_trial_envelope(audit, examples, trial, previous, frozen_utc)
    audit.check(0 < trial['peak_working_set_bytes'] <= MEMORY_CAP, f'{context}: peak working-set cap')
    audit.check(0 < trial['artifact_bytes'] <= ARTIFACT_CAP, f'{context}: artifact cap')
    audit.check(trial['stage'] == 'complete', f'{context}: both recorded stages completed')
    audit.check('science_seconds' in trial and 'audit_seconds' in trial, f'{context}: both stage durations recorded')
    paths = {f'{geometry}_{history}_{rule}.npz' for geometry in GEOMETRIES for history in HISTORIES for rule in RULES}
    paths.update({'base.npz', 'weights.npy', 'ROWS.json', 'MICROBENCH.json', 'BLOCK.json', 'AUDIT.json'})
    audit.check({path.name for path in stage.iterdir()} == paths, f'{context}: complete raw and audited stage archives')

    block, result = (read(stage / name) for name in ('BLOCK.json', 'AUDIT.json'))
    audit.check(block['examples'] == result['examples'] == examples and block['paths'] == result['paths'] == 24,
                f'{context}: complete block and audit scopes')
    audit.check(result['status'] == 'PASS' and result['checks'] > 0., f'{context}: independent equivalence passed')
    audit.check(finite(result['max_numeric_error']) and 0. <= result['max_numeric_error'] <= 5e-12,
                f'{context}: bounded continuous error')
    phase_names = ('input_seconds', 'shared_seconds', 'verify_score_seconds', 'archive_seconds', 'isolation_seconds',
                   'other_seconds')
    phases = {key: block[key] for key in phase_names}
    audit.check(all(finite(value) and value >= 0. for value in phases.values()), f'{context}: bounded science phases')
    audit.close(block['total_seconds'], sum(phases.values()), f'{context}: complete science duration ledger')
    audit.check(0. < block['total_seconds'] <= trial['science_seconds'] + 1e-6,
                f'{context}: measured science duration contains saved phase ledger')
    if 'wall_seconds' in result:
        audit.check(finite(result['wall_seconds']) and 0. < result['wall_seconds'] <= trial['audit_seconds'] + 1e-6,
                    f'{context}: measured independent audit duration contains saved ledger')
    audit.check(isinstance(block['work'], dict) and isinstance(result['work'], dict) and isinstance(block['metrics'], dict),
                f'{context}: recorded block work and fixture metrics')
    audit.check(block['metrics']['retained_raw_array_bytes'] == result['retained_raw_array_bytes'],
                f'{context}: both stages agree on raw array bytes')
    archive_bytes = sum((stage / name).stat().st_size for name in paths - {'BLOCK.json', 'AUDIT.json'})
    audit.check(block['metrics']['archived_bytes_before_block'] == archive_bytes,
                f'{context}: saved archive byte accounting before block and audit summaries')
    audit.close(result['prior_training_audit_forecast_seconds'], TRAINING_AUDIT_SECONDS,
                f'{context}: independent stage training-audit forecast', 1e-12)
    return dict(trial=trial['trial'], directory=trial['directory'], science_phases=phases, work=block['work'],
                audit_work=result['work'], metrics=block['metrics'], independent_checks=result['checks'],
                max_numeric_error=result['max_numeric_error'], block_sha256=trial['block_sha256'],
                audit_sha256=trial['audit_sha256'])


def verify_budget(audit, frozen):
    timing = read(ROOT / 'TIMING.json')
    audit.check(timing['blocks'] == frozen['blocks'] and timing['scientific_data_generated'] is False
                and timing['formal_training_fits'] == timing['formal_evaluations'] == 0, 'fixture-only timing scope')
    audit.close(timing['training_audit_forecast_seconds'], TRAINING_AUDIT_SECONDS, 'fixed training audit forecast', 1e-12)
    candidates = timing['candidates']
    audit.check([item['examples'] for item in candidates] in ([512], [512, 256]), 'attempted 512 before optional 256')
    frozen_utc = parse_utc(audit, frozen['utc'], 'freeze')
    selected, verified, directories = None, [], set()
    previous_finish = None
    for index, item in enumerate(candidates):
        n, trials = item['examples'], item['trials']
        audit.check(item['fixed_gap_seconds'] == GAP and [row['trial'] for row in trials] == [1, 2],
                    f'n={n}: exactly two fixed-gap trials')
        audit.check(previous_finish is None or trials[0]['monotonic_start_seconds'] >= previous_finish,
                    f'n={n}: sequential candidate tiers')
        blocks = [verify_trial(audit, n, trial, trials[i - 1] if i else None, frozen_utc)
                  for i, trial in enumerate(trials)]
        for key in ('work', 'audit_work', 'independent_checks'):
            audit.check(blocks[0][key] == blocks[1][key], f'n={n}: repeated deterministic {key}')
        stable_metrics = [{key: value for key, value in row['metrics'].items() if key != 'archived_bytes_before_block'}
                          for row in blocks]
        audit.check(stable_metrics[0] == stable_metrics[1], f'n={n}: repeated deterministic metrics')
        directories.update(block['directory'] for block in blocks)
        previous_finish = trials[-1]['monotonic_finish_seconds']
        forecasts = {}
        for kind in ('science', 'audit'):
            worst = max(row[f'{kind}_seconds'] for row in trials)
            audit.close(item[f'{kind}_seconds'], worst, f'n={n}: {kind} uses maximum of both trials')
            forecast = 3 * BLOCKS * worst + GAP + (TRAINING_AUDIT_SECONDS if kind == 'audit' else 0.)
            audit.close(item[f'{kind}_forecast_seconds'], forecast, f'n={n}: {kind} budget arithmetic')
            forecasts[kind] = forecast
        within_budget = forecasts['science'] < SCIENCE_CAP and forecasts['audit'] < AUDIT_CAP
        audit.check(item['within_budget'] is within_budget, f'n={n}: correct tier decision')
        if within_budget:
            audit.check(index == len(candidates) - 1, 'stopped after first feasible tier')
            selected = n
        verified.append(dict(examples=n, within_budget=within_budget, science_forecast_seconds=forecasts['science'],
                             audit_forecast_seconds=forecasts['audit'], trials=blocks))
    audit.check(selected is not None or len(candidates) == 2, 'infeasible decision follows both allowed tiers')
    expected_status = 'FEASIBLE' if selected is not None else 'INFEASIBLE'
    audit.check(timing['status'] == expected_status and timing['selected_examples_per_block'] == selected,
                'complete timing decision agrees with independently recomputed tier selection')
    archived = {path.relative_to(ROOT).as_posix() for path in (ROOT / 'trials').iterdir()}
    audit.check(archived == directories, 'retained exactly the permitted trial directories')
    for kind, cap in (('science', SCIENCE_CAP), ('audit', AUDIT_CAP)):
        total = sum(row[f'{kind}_seconds'] for item in candidates for row in item['trials'])
        audit.close(timing[f'cumulative_{kind}_seconds'], total, f'cumulative actual {kind} ledger')
        audit.check(0. < total <= cap, f'cumulative actual {kind} under cap')
    audit.check(0 < timing['peak_working_set_bytes'] <= MEMORY_CAP, 'whole fixture working-set cap')
    audit.check(0 < timing['artifact_bytes'] <= ARTIFACT_CAP, 'whole fixture archived-byte cap')
    return expected_status, selected, verified


def verify_scope(audit, expected_status, selected):
    state = read(ROOT / 'RUN_STATE.json')
    audit.check(state['status'] == expected_status and state['scientific_data_generated'] is False
                and state['selected_examples_per_block'] == selected, 'run state agrees with audited fixture-only decision')
    formal_paths = ('train', 'evaluate', 'weights', 'TRAIN_SUMMARY.json', 'SCIENCE_SUMMARY.json', 'TEST_START.json')
    for path in formal_paths:
        audit.check(not (ROOT / path).exists(), f'no formal artifact: {path}')
    artifact_bytes = sum(path.stat().st_size for path in ROOT.rglob('*') if path.is_file())
    if expected_status == 'complete_budget':
        audit.check(0 < artifact_bytes <= ARTIFACT_CAP, 'total fixture artifacts within disk cap')
    return artifact_bytes


def verify_stopped(audit, frozen, timing):
    state = read(ROOT / 'RUN_STATE.json')
    audit.check(state == timing and timing['status'] == 'stopped_budget_error' and bool(timing['exception'])
                and bool(timing['reason']), 'same preserved exception in stopped timing and run state')
    candidates = timing['candidates']
    audit.check([row['examples'] for row in candidates] in ([], [512], [512, 256]), 'permitted stopped tier prefix')
    frozen_utc = parse_utc(audit, frozen['utc'], 'freeze')
    verified, recorded = [], []
    for index, candidate in enumerate(candidates):
        n, trials = candidate['examples'], candidate['trials']
        audit.check(candidate['fixed_gap_seconds'] == GAP and len(trials) <= 2
                    and [row['trial'] for row in trials] == list(range(1, len(trials) + 1)),
                    f'n={n}: permitted stopped trial prefix')
        if index < len(candidates) - 1:
            audit.check(len(trials) == 2 and candidate['within_budget'] is False,
                        f'n={n}: complete earlier tier required to fall back')
        blocks = []
        for position, trial in enumerate(trials):
            previous = trials[position - 1] if position else None
            recorded.append(trial)
            if trial['stage'] == 'complete':
                blocks.append(verify_trial(audit, n, trial, previous, frozen_utc))
                continue
            audit.check(index == len(candidates) - 1 and position == len(trials) - 1
                        and trial['stage'] in ('science', 'audit'), 'only last trial may be partial')
            verify_trial_envelope(audit, n, trial, previous, frozen_utc)
            blocks.append(dict(trial=trial['trial'], directory=trial['directory'], stopped_stage=trial['stage']))
        verified.append(dict(examples=n, trials=blocks))
    for kind in ('science', 'audit'):
        actual = sum(trial.get(f'{kind}_seconds', 0.) for trial in recorded)
        audit.close(timing[f'cumulative_{kind}_seconds'], actual, f'cumulative stopped {kind} timing')
    return verified


def main():
    target = ROOT / 'TIMING_AUDIT.json'
    if target.exists():
        raise FileExistsError(target)
    audit = TimingAudit()
    frozen = verify_sources(audit)
    timing = read(ROOT / 'TIMING.json')
    if timing['status'] == 'stopped_budget_error':
        status, selected = timing['status'], None
        candidates = verify_stopped(audit, frozen, timing)
        artifact_bytes = verify_scope(audit, status, selected)
        verification_status = 'STOPPED'
        fault = dict(exception=timing['exception'], reason=timing['reason'])
    else:
        status, selected, candidates = verify_budget(audit, frozen)
        artifact_bytes = verify_scope(audit, 'complete_budget', selected)
        verification_status, fault = 'PASS', None
    result = dict(status=verification_status, timing_status=status, selected_examples_per_block=selected, checks=audit.checks,
                  candidates=candidates, artifact_bytes_at_audit=artifact_bytes, wall_seconds=time.perf_counter() - audit.start,
                  scientific_data_generated=False, formal_training_fits=0, formal_evaluations=0,
                  scope='Saved stage archive hashes, frozen inputs and scope, exact two-trial budget rule, and no formal phase.')
    if fault is not None:
        result['fault'] = fault
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'Budget timing audit {verification_status} ({audit.checks} checks), decision {status}, selected tier {selected}.')


if __name__ == '__main__':
    main()
