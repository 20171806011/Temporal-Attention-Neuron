"""Independently audit saved D16 retrieval interventions and primary effects."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import independent
import independent_data as data


ROOT = Path(__file__).resolve().parent
ARMS = ('GAIN2', 'SCALE', 'SCALE_MEAN')
BLOCKS = 12
T, Q, ENTRY = 36, 28, 25
SOURCE_FILES = {'model.py', 'population.py', 'reference.py', 'original_tan.py',
                'task_data.py', 'independent_data.py', 'metrics.py', 'scale_model.py'}
MEMORY_CAP = 1024**3
ARTIFACT_CAP = 3*1024**3//2
TWIN_TOLERANCE = 5e-12


class ProcessCounters(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
        (field, ctypes.c_size_t) for field in (
            'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
            'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]


def peak_memory():
    counter = ProcessCounters()
    counter.cb = ctypes.sizeof(counter)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(counter), counter.cb):
        raise OSError('Could not read audit peak working set')
    return counter.PeakWorkingSetSize


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return dict(archive)


class Audit:
    def __init__(self):
        self.start = time.perf_counter()
        self.cpu_start = time.process_time()
        self.checks = 0
        self.stage = 'provenance'
        self.max_error = 0.
        self.crossover_error = 0.
        self.crossover_cases = 0
        self.crossover_rows = 0

    def guard(self):
        if time.perf_counter() - self.start > 300:
            raise TimeoutError(f'300 second independent audit cap during {self.stage}')
        if peak_memory() > MEMORY_CAP:
            raise MemoryError('1 GiB audit peak working-set limit')

    def check(self, condition, context):
        self.checks += 1
        if time.perf_counter() - self.start > 300:
            raise TimeoutError(f'300 second independent audit cap during {self.stage}')
        assert bool(condition), context

    def exact(self, actual, expected, context):
        self.check(np.array_equal(actual, expected), context)

    def close(self, actual, expected, context, tolerance=1e-8):
        actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.check(actual.shape == expected.shape, f'{context}: shape')
        self.check(np.isfinite(actual).all() and np.isfinite(expected).all(), f'{context}: finite')
        error = float(np.max(np.abs(actual-expected))) if actual.size else 0.
        self.max_error = max(self.max_error, error)
        self.check(error <= tolerance, f'{context}: error {error:.17g} > {tolerance}')
        return error

    def twin(self, actual, expected, context):
        error = self.close(actual, expected, context, TWIN_TOLERANCE)
        self.crossover_error = max(self.crossover_error, error)

    def elapsed(self):
        return dict(wall_seconds=time.perf_counter()-self.start, cpu_seconds=time.process_time()-self.cpu_start)


def provenance(audit):
    frozen = read(ROOT/'FREEZE.json')
    examples = frozen['examples_per_block']
    audit.check(frozen['scientific_data_generated'] is False and frozen['blocks'] == BLOCKS
                and frozen['evaluations'] == 21*BLOCKS and examples in (256, 128), 'frozen scope')
    audit.check({'PROTOCOL.md', 'REPRODUCE.md', 'SOURCE_LOCK.json', 'PRECHECK.json', 'TIMING.json',
                 'run.py', 'intervene.py', 'audit.py', 'independent.py', 'summarize.py', 'report.py'}
                <= set(frozen['files']), 'frozen scripts and prerequisites')
    audit.check(all(not name.startswith('evaluate/') for name in frozen['files']), 'freeze excludes scientific outputs')
    for name, expected in frozen['files'].items():
        audit.check(digest(ROOT/name) == expected, f'frozen file {name}')
    precheck, timing = read(ROOT/'PRECHECK.json'), read(ROOT/'TIMING.json')
    audit.check(precheck['status'] == timing['status'] == 'PASS', 'fixture and timing passed')
    audit.check(precheck.get('scientific_data_generated', False) is False
                and timing['scientific_data_generated'] is False, 'fixtures precede scientific data')
    audit.check(timing['blocks'] == BLOCKS and timing['examples_per_block'] == examples, 'timing selection')
    candidates = timing['candidates']
    audit.exact([row['examples'] for row in candidates], [256] if examples == 256 else [256, 128],
                'timing tried larger size first')
    for i, row in enumerate(candidates):
        for kind in ('science', 'audit'):
            seconds = row[f'{kind}_seconds']
            audit.check(seconds > 0, f'positive {kind} fixture duration')
            audit.close(row[f'{kind}_forecast_seconds'], 36*seconds+30, f'{kind} forecast')
        passed = row['science_forecast_seconds'] < 840 and row['audit_forecast_seconds'] < 300
        audit.check(passed == (i == len(candidates)-1), 'selected first size satisfying both caps')

    lock = read(ROOT/'SOURCE_LOCK.json')
    audit.check(lock['previous_directory'] == '../DEND-TAN-scale-confirm-20261109', 'exact prior source directory')
    previous = ROOT/lock['previous_directory']
    audit.check(digest(previous/'MANIFEST.json') == lock['previous_manifest_sha256'], 'prior manifest seal')
    manifest = read(previous/'MANIFEST.json')['files']
    for name in ('AUDIT.json', 'SCIENCE_SUMMARY.json'):
        audit.check(digest(previous/name) == manifest[name]['sha256'], f'sealed prior {name}')
    earlier_audit = read(previous/'AUDIT.json')
    earlier_science = read(previous/'SCIENCE_SUMMARY.json')
    audit.check(earlier_audit['status'] == 'PASS' and earlier_science['status'] == 'COMPLETE'
                and earlier_science['blocks'] == BLOCKS and earlier_science['fits'] == 36,
                'source is the complete audited 12-block confirmation')
    expected_ids = {f's{s}_{arm}' for s in range(BLOCKS) for arm in ARMS}
    weights = {}
    audit.check(set(lock['checkpoints']) == expected_ids, 'complete checkpoint coverage')
    audit.check(set(lock['files']) == SOURCE_FILES | {f'weights/{ident}.npy' for ident in expected_ids},
                'source lock file set')
    for name, sha in lock['files'].items():
        audit.check(frozen['files'][name] == sha, f'frozen source lock digest: {name}')
    for name in SOURCE_FILES:
        audit.check(digest(previous/name) == manifest[name]['sha256'] == lock['files'][name],
                    f'copied source origin: {name}')
    for ident in sorted(expected_ids):
        rec = lock['checkpoints'][ident]
        source, weight_file = f'train/{ident}_d16.npz', f'weights/{ident}.npy'
        audit.check(rec['source'] == source and rec['weight_file'] == weight_file, f'{ident}: source and extract paths')
        audit.check(digest(previous/source) == rec['source_sha256'] == manifest[source]['sha256'],
                    f'{ident}: sealed checkpoint archive')
        audit.check(rec['weight_sha256'] == lock['files'][weight_file], f'{ident}: checkpoint extract lock')
        with np.load(previous/source, allow_pickle=False) as archive:
            theta = archive['theta']
        weights[ident] = np.load(ROOT/weight_file, allow_pickle=False)
        audit.check(theta.shape == (176,) and theta.dtype == weights[ident].dtype == np.float64
                    and np.isfinite(theta).all(), f'{ident}: native weights')
        audit.exact(weights[ident], theta, f'{ident}: final checkpoint theta preserved')
    audit.guard()
    return examples, weights


def native_scores(counts, base):
    y = base['y']
    prediction = np.sign(counts[:, 1]-counts[:, 0]).astype(int)
    prediction = np.where(prediction == 0, -1, (prediction+1)//2)
    ties = prediction < 0
    good = prediction == y
    silence = np.all(counts == 0, axis=1)
    saturation = np.all(counts == 8, axis=1)
    recalls = [float(np.mean(good[y == label])) for label in (0, 1)]
    result = dict(accuracy=float(good.mean()), coverage=float(np.mean(~ties)), silence=float(silence.mean()),
                  saturated_tie=float(saturation.mean()), intermediate_tie=float(np.mean(ties & ~silence & ~saturation)),
                  recalls=recalls, balanced_accuracy=float(np.mean(recalls)),
                  prediction_counts=[int(np.count_nonzero(prediction == label)) for label in (-1, 0, 1)])
    disagree, second = base['first_bit'] != base['second_bit'], base['selected'] == 1
    for name, mask in (('agree', ~disagree), ('disagree', disagree), ('select_first', ~second), ('select_second', second)):
        result[f'{name}_ba'] = float(np.mean([good[mask & (y == label)].mean() for label in (0, 1)]))
        result[f'{name}_coverage'] = float(np.mean(~ties[mask]))
    rows = data.lookup(base)
    both = [good[i] and good[rows[octet, 1, first, second_bit]]
            for (octet, selection, first, second_bit), i in rows.items() if selection == 0 and first != second_bit]
    result['disagree_joint_correct'] = float(np.mean(both))
    return result


def cost(arm):
    trace_slots = 0 if arm == 'GAIN2' else 24
    units = T*trace_slots//2
    return dict(parameters=176, spiking_cells=8, graded_branches=12,
                continuous_state_slots=20+trace_slots, history_slots=0, trace_slots=trace_slots,
                persistent_array_bytes=8*(28+trace_slots), dense_forward_macs=156*T,
                graded_transfers=12*T, history_roll_writes=0, attention_unit_steps=0,
                softmax_exponentials=0, trace_tanh_evaluations=units, trace_logistic_evaluations=units,
                trace_retained_multiplies=2*units, trace_write_multiplies=units,
                trace_update_adds=2*units, trace_context_unit_steps=units)


def check_stimuli(audit, seed, n):
    path = ROOT/'evaluate'/f'test_s{seed}.npz'
    base = data.make_base(n, 4072020000+100*seed)
    saved = load(path)
    audit.check(set(saved) == set(base), f'{path.name}: base fields')
    for key, values in base.items():
        audit.exact(saved[key], values, f'{path.name}: {key}')
    x, q = data.assemble(base, 16)
    audit.check(x.shape == (n, T, 6) and q == Q and not x[:, 9:25].any(), f'{path.name}: stimulus timing')
    mapping = load(ROOT/'evaluate'/f'donors_s{seed}.npz')
    expected = independent.donors(base)
    audit.check(set(mapping) == set(expected), f'{path.name}: donor fields')
    opposite = base['first_bit'] != base['second_bit']
    audit.check(opposite.sum() == n//2, f'{path.name}: opposite-content count')
    row_by_key = data.lookup(base)
    twins = np.empty(n, dtype=int)
    for key, i in row_by_key.items():
        octet, cue, first, second = key
        twins[i] = row_by_key[octet, 1-cue, first, second]
    audit.exact(twins[twins], np.arange(n), f'{path.name}: cue twin involution')
    audit.exact(x[:, :ENTRY], x[twins, :ENTRY], f'{path.name}: cue twin common prefix')
    for kind, index in mapping.items():
        audit.exact(index, expected[kind], f'{path.name}: {kind} mapping')
        audit.exact(index[~opposite], np.flatnonzero(~opposite), f'{path.name}: {kind} self donors')
        audit.exact(x[:, ENTRY:], x[index, ENTRY:], f'{path.name}: {kind} common suffix')
        audit.exact(base['octet'], base['octet'][index], f'{path.name}: {kind} matched nuisance octet')
        audit.exact(base['selected'], base['selected'][index], f'{path.name}: {kind} common cue')
        flips_first = base['first_bit'][index] != base['first_bit']
        flips_second = base['second_bit'][index] != base['second_bit']
        target_first = base['selected'] == int(kind == 'unselected')
        audit.exact(flips_first, opposite & target_first, f'{path.name}: {kind} exact first-bit flips')
        audit.exact(flips_second, opposite & ~target_first, f'{path.name}: {kind} exact second-bit flips')
        audit.exact(base['y'][index] != base['y'], opposite if kind == 'selected' else np.zeros(n, dtype=bool),
                    f'{path.name}: {kind} donor label relation')
    audit.exact(mapping['selected'][twins][opposite], twins[mapping['unselected']][opposite],
                f'{path.name}: selected becomes unselected across cue')
    audit.exact(mapping['unselected'][twins][opposite], twins[mapping['selected']][opposite],
                f'{path.name}: unselected becomes selected across cue')
    return base, x, mapping, twins, opposite


def compare_raw(audit, raw, expected, context):
    audit.check(set(raw) == set(expected), f'{context}: exact raw fields')
    for key, values in raw.items():
        if key in ('counts', 'spikes', 'resources'):
            audit.exact(values, expected[key], f'{context}: {key}')
        else:
            audit.close(values, expected[key], f'{context}: {key}')
    audit.exact(raw['states'][..., 20:], raw['spikes'][..., :6], f'{context}: stored previous hidden spikes')
    audit.exact(raw['entry_native'][..., 20:], expected['entry_native'][..., 20:], f'{context}: entry spikes')
    audit.exact(raw['counts'][:8], raw['spikes'][:, Q:, 6:].sum(axis=1), f'{context}: archived query counts')
    audit.check(np.all((0 <= raw['counts']) & (raw['counts'] <= 8)), f'{context}: legal count range')


def check_traces(audit, raw, arm, condition, context):
    trace, alpha = raw['traces'], raw['weights_fast']
    audit.check(np.abs(trace).max() <= 1 and np.abs(raw['entry_traces']).max() <= 1,
                f'{context}: bounded signed traces')
    audit.check(np.min(alpha) > 0 and np.max(alpha) < 1, f'{context}: weight range')
    audit.close(trace[:, 9:25], trace[:, 8:24]*np.array([.85, .97]), f'{context}: blank pure decay')
    audit.exact(raw['injections'][:, 9:25], np.zeros((8, 16, 12)), f'{context}: blank zero injection')
    if arm == 'SCALE_MEAN' or condition == 'post_mean':
        uniform = alpha if arm == 'SCALE_MEAN' else alpha[:, ENTRY:]
        audit.exact(uniform, np.full_like(uniform, .5), f'{context}: exact uniform scale weights')


def check_scope(audit, raw, nominal, mapping, opposite, arm, condition, context):
    scope, _, kind = condition.partition('_')
    donor = mapping.get(kind, np.arange(len(opposite)))
    if condition == 'nominal':
        audit.exact(raw['entry_native'][:8], raw['states'][:, ENTRY-1], f'{context}: native entry tick')
        if arm != 'GAIN2':
            audit.exact(raw['entry_traces'][:8], raw['traces'][:, ENTRY-1], f'{context}: trace entry tick')
    if condition == 'identity':
        for name in nominal:
            audit.exact(raw[name], nominal[name], f'{context}: full identity {name}')
    for key in ('states', 'spikes', 'injections', 'traces', 'weights_fast'):
        if key in raw:
            audit.exact(raw[key][:, :ENTRY], nominal[key][:, :ENTRY], f'{context}: unchanged prefix {key}')
    native_donor = donor if scope in ('native', 'all') else slice(None)
    audit.exact(raw['entry_native'], nominal['entry_native'][native_donor], f'{context}: native entry scope')
    if arm != 'GAIN2':
        trace_donor = donor if scope in ('trace', 'all') else slice(None)
        audit.exact(raw['entry_traces'], nominal['entry_traces'][trace_donor], f'{context}: trace entry scope')
    if scope == 'all' or (arm == 'GAIN2' and scope == 'native'):
        audit.exact(raw['counts'], nominal['counts'][donor], f'{context}: exact donor continuation')
    if kind in mapping:
        for name in ('counts', 'resources'):
            audit.exact(raw[name][~opposite], nominal[name][~opposite], f'{context}: self-donor {name}')


def check_crossover(audit, left, right, opposite, twins, context):
    other_cue = twins[opposite]
    a, b = left['entry_native'][opposite], right['entry_native'][other_cue]
    audit.twin(a[:, :20], b[:, :20], f'{context}: cue twin graded entry')
    audit.exact(a[:, 20:], b[:, 20:], f'{context}: cue twin hidden spikes')
    if 'entry_traces' in left:
        audit.twin(left['entry_traces'][opposite], right['entry_traces'][other_cue], f'{context}: cue twin trace entry')
    audit.crossover_cases += 1
    audit.crossover_rows += len(other_cue)


def verify_row(audit, row, base, raw, ident, seed, arm, condition):
    context = f'{ident}/{condition}'
    identity = dict(id=ident, seed=seed, arm=arm, condition=condition, T=T, q=Q)
    scores = native_scores(raw['counts'], base)
    audit.check(set(row) == set(identity) | set(scores) | {'mean_resources', 'cost_per_episode'},
                f'{context}: exact score-row fields')
    for key, value in identity.items():
        audit.check(row[key] == value, f'{context}: {key}')
    for key, value in scores.items():
        audit.close(row[key], value, f'{context}: {key}', tolerance=1e-13)
    audit.close(row['mean_resources'], raw['resources'].mean(axis=0), f'{context}: mean event/delivery ledger')
    audit.check(row['cost_per_episode'] == cost(arm), f'{context}: per-episode structural cost')
    return scores


def reverse_rate(counts, base, opposite):
    y = base['y'][opposite]
    pair = counts[opposite]
    return float(np.mean(pair[np.arange(len(y)), 1-y] > pair[np.arange(len(y)), y]))


def evaluate(audit, n, weights):
    audit.stage = 'replay'
    rows = read(ROOT/'evaluate'/'ROWS.json')
    index = {(row['id'], row['condition']): row for row in rows}
    expected_rows = {(f's{s}_{arm}', condition) for s in range(BLOCKS) for arm in ARMS
                     for condition in independent.CONDITIONS[arm]}
    audit.check(len(rows) == len(index) == 21*BLOCKS and set(index) == expected_rows, 'complete evaluation rows')
    expected_files = {f'{ident}_{condition}.npz' for ident, condition in expected_rows}
    expected_files |= {f'{stem}_s{s}.npz' for stem in ('test', 'donors') for s in range(BLOCKS)}
    audit.check({p.name for p in (ROOT/'evaluate').glob('*.npz')} == expected_files, 'exact raw evaluation file set')
    totals = dict(intervention_payload_bytes=0, donor_index_bytes=0, input_array_bytes=0, retained_raw_array_bytes=0)
    primary = {'trace_transfer_effect': [], 'scale_nominal_minus_post_mean': []}
    donor_recoveries = identity_cases = clamp_cases = 0
    for seed in range(BLOCKS):
        base, x, mapping, twins, opposite = check_stimuli(audit, seed, n)
        totals['donor_index_bytes'] += sum(array.nbytes for array in mapping.values())
        totals['input_array_bytes'] += x.nbytes
        transfers = []
        for arm in ARMS:
            ident = f's{seed}_{arm}'
            raw_by_condition, scores = {}, {}
            for condition in independent.CONDITIONS[arm]:
                audit.guard()
                context = f'{ident}/{condition}'
                raw = load(ROOT/'evaluate'/f'{ident}_{condition}.npz')
                expected = independent.replay(x, weights[ident], arm, Q, condition, mapping)
                compare_raw(audit, raw, expected, context)
                raw_by_condition[condition] = raw
                check_scope(audit, raw, raw_by_condition['nominal'], mapping, opposite, arm, condition, context)
                if arm != 'GAIN2':
                    check_traces(audit, raw, arm, condition, context)
                scores[condition] = verify_row(audit, index[ident, condition], base, raw, ident, seed, arm, condition)
                scope, _, _ = condition.partition('_')
                payload_fields = set()
                if scope in ('identity', 'native', 'all'):
                    payload_fields.add('entry_native')
                if arm != 'GAIN2' and scope in ('identity', 'trace', 'all'):
                    payload_fields.add('entry_traces')
                totals['intervention_payload_bytes'] += sum(raw[key].nbytes for key in payload_fields)
                totals['retained_raw_array_bytes'] += sum(value.nbytes for value in raw.values())
                identity_cases += int(condition == 'identity')
                clamp_cases += int(condition == 'post_mean')
                donor_recoveries += int(scope == 'all' or (scope == 'native' and arm == 'GAIN2'))
            nominal = raw_by_condition['nominal']
            check_crossover(audit, nominal, nominal, opposite, twins, f'{ident}/nominal')
            families = ('native',) if arm == 'GAIN2' else ('native', 'trace', 'all')
            for family in families:
                left, right = (raw_by_condition[f'{family}_{kind}'] for kind in ('selected', 'unselected'))
                check_crossover(audit, left, right, opposite, twins, f'{ident}/{family}')
            if arm != 'GAIN2':
                reverse = [reverse_rate(raw_by_condition[f'trace_{kind}']['counts'], base, opposite)
                           for kind in ('selected', 'unselected')]
                transfers.append(reverse[0]-reverse[1])
            if arm == 'SCALE':
                primary['scale_nominal_minus_post_mean'].append(scores['nominal']['disagree_ba']
                                                                - scores['post_mean']['disagree_ba'])
        primary['trace_transfer_effect'].append(float(np.mean(transfers)))
        print(f'Independently replayed block {seed+1}/{BLOCKS}', flush=True)
    counts = dict(evaluations=len(rows), identity_cases=identity_cases, clamp_cases=clamp_cases,
                  donor_continuation_cases=donor_recoveries, crossover_cases=audit.crossover_cases,
                  crossover_rows=audit.crossover_rows, max_crossover_error=audit.crossover_error)
    audit.check((identity_cases, clamp_cases, donor_recoveries) == (36, 12, 72), 'complete intervention invariants')
    audit.check(audit.crossover_cases == 120 and audit.crossover_rows == 60*n, 'complete cue crossover invariants')
    return totals, counts, primary


def primary_inference(audit, effects):
    audit.stage = 'primary inference'
    rng = np.random.default_rng(4072900001)
    draws = rng.integers(0, BLOCKS, (50_000, BLOCKS))
    result = []
    for name, block_values in effects.items():
        values = np.asarray(block_values)
        audit.check(values.shape == (BLOCKS,) and np.isfinite(values).all(), f'{name}: complete primary values')
        bootstrap = np.sum(values[draws], axis=1)/BLOCKS
        low, high = np.quantile(bootstrap, [.0125, .9875])
        direction = 'positive' if low > 0 else 'negative' if high < 0 else 'undetermined'
        result.append(dict(effect=name, blocks=BLOCKS, values=block_values, mean=float(values.mean()),
                           minimum=float(values.min()), maximum=float(values.max()),
                           positive=int((values > 0).sum()), zero=int((values == 0).sum()),
                           negative=int((values < 0).sum()), ci97_5=[float(low), float(high)], direction=direction))
    return result


def main(audit):
    audit.check(all(os.environ.get(key) == '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')),
                'single-thread independent audit')
    n, weights = provenance(audit)
    totals, counts, primary = evaluate(audit, n, weights)
    inference = primary_inference(audit, primary)
    audit.stage = 'resource ledger'
    science = read(ROOT/'SCIENCE_SUMMARY.json')
    steps = 21*BLOCKS*n*T
    expected = dict(blocks=BLOCKS, examples_per_block=n, evaluations=21*BLOCKS, fits=0, updates=0,
                    forward_sample_steps=steps, dense_forward_macs=156*steps, **totals)
    audit.check(science['status'] == 'COMPLETE' and science['cap_seconds'] == 840
                and 0 < science['wall_seconds'] <= 840 and science['cpu_seconds'] > 0, 'science wall and CPU time')
    audit.check(0 < science['peak_working_set_bytes'] <= MEMORY_CAP, 'science memory cap')
    audit.check(0 < science['artifact_bytes_at_science_end'] <= ARTIFACT_CAP, 'science artifact cap')
    audit.check(sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) < ARTIFACT_CAP, 'current artifact cap')
    audit.check(read(ROOT/'RUN_STATE.json')['status'] == 'complete', 'science completion state')
    audit.check(read(ROOT/'ENVIRONMENT.json')['threads'] == {
        key: '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}, 'science thread scope')
    for key, value in expected.items():
        audit.check(science[key] == value, f'science ledger {key}')
    audit.guard()
    result = dict(status='PASS', checks=audit.checks, blocks=BLOCKS, examples_per_block=n, fits=0, updates=0,
                  forward_sample_steps=steps, dense_forward_macs=156*steps, max_numeric_error=audit.max_error,
                  cap_seconds=300, peak_working_set_bytes=peak_memory(), **counts, **audit.elapsed(), **totals,
                  primary_effects=inference,
                  scope='Independent checkpoint provenance, stimulus and donor regeneration, every intervention replay, '
                        'archived inputs to the continuation, cue crossover, exact discrete outputs and delivery counts, '
                        'scope and resource ledgers, and paired bootstrap of both primary effects.')
    (ROOT/'AUDIT.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    auditor = Audit()
    try:
        main(auditor)
    except Exception as exc:
        failure = dict(status='FAIL', stage=auditor.stage, checks=auditor.checks, reason=str(exc),
                       exception=type(exc).__name__, **auditor.elapsed())
        (ROOT/'AUDIT_FAILURE.json').write_text(json.dumps(failure, indent=2)+'\n', encoding='utf-8')
        raise
