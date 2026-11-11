"""Independently audit saved D16 prefix-state interventions and primary effects."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import independent
import independent_data as data
from reference import split


ROOT = Path(__file__).resolve().parent
ARMS = ('GAIN2', 'SCALE', 'SCALE_MEAN')
BLOCKS = 12
T, Q, ENTRY = 36, 28, 25
SOURCE_FILES = {'model.py', 'population.py', 'reference.py', 'original_tan.py',
                'task_data.py', 'independent_data.py', 'metrics.py', 'scale_model.py', 'intervene.py'}
MEMORY_CAP = 1024**3
ARTIFACT_CAP = 3*1024**3//2
ENTRY_DIFFERENCE = 5e-12


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

    def elapsed(self):
        return dict(wall_seconds=time.perf_counter()-self.start, cpu_seconds=time.process_time()-self.cpu_start)


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


def primary_inference(audit, effects):
    audit.stage = 'primary inference'
    rng = np.random.default_rng(4073900001)
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


def provenance(audit):
    freeze = read(ROOT/'FREEZE.json')
    n = freeze['examples_per_block']
    audit.check(freeze['scientific_data_generated'] is False and freeze['blocks'] == BLOCKS
                and freeze['evaluations'] == 7*BLOCKS and n in (256, 128), 'frozen scope')
    audit.check({'PROTOCOL.md', 'REPRODUCE.md', 'SOURCE_LOCK.json', 'PRECHECK.json', 'TIMING.json',
                 'run.py', 'prefix.py', 'audit.py', 'independent.py', 'summarize.py', 'report.py'}
                <= set(freeze['files']), 'frozen scripts and prerequisites')
    audit.check(all(not name.startswith('evaluate/') for name in freeze['files']), 'freeze excludes scientific outputs')
    for name, expected in freeze['files'].items():
        audit.check(digest(ROOT/name) == expected, f'frozen file {name}')
    precheck, timing = read(ROOT/'PRECHECK.json'), read(ROOT/'TIMING.json')
    audit.check(precheck['status'] == timing['status'] == 'PASS', 'fixture and timing passed')
    audit.check(precheck.get('scientific_data_generated', False) is False
                and timing['scientific_data_generated'] is False, 'fixtures precede scientific data')
    audit.check(timing['blocks'] == BLOCKS and timing['examples_per_block'] == n, 'timing selection')
    candidates = timing['candidates']
    audit.exact([row['examples'] for row in candidates], [256] if n == 256 else [256, 128],
                'timing tried larger size first')
    for i, row in enumerate(candidates):
        for kind in ('science', 'audit'):
            seconds = row[f'{kind}_seconds']
            audit.check(seconds > 0, f'positive {kind} fixture duration')
            audit.close(row[f'{kind}_forecast_seconds'], 36*seconds+30, f'{kind} forecast')
        passed = row['science_forecast_seconds'] < 840 and row['audit_forecast_seconds'] < 300
        audit.check(passed == (i == len(candidates)-1), 'selected first size satisfying both caps')

    lock = read(ROOT/'SOURCE_LOCK.json')
    audit.check(lock['previous_directory'] == '../DEND-TAN-scale-retrieval-20261109', 'exact prior source directory')
    previous = ROOT/lock['previous_directory']
    audit.check(digest(previous/'MANIFEST.json') == lock['previous_manifest_sha256'], 'prior manifest seal')
    manifest = read(previous/'MANIFEST.json')['files']
    for name in ('AUDIT.json', 'SCIENCE_SUMMARY.json'):
        audit.check(digest(previous/name) == manifest[name]['sha256'], f'sealed prior {name}')
    earlier_audit, earlier_science = read(previous/'AUDIT.json'), read(previous/'SCIENCE_SUMMARY.json')
    audit.check(earlier_audit['status'] == 'PASS' and earlier_audit['evaluations'] == 252
                and earlier_science['status'] == 'COMPLETE' and earlier_science['blocks'] == BLOCKS
                and earlier_science['fits'] == 0, 'source is the complete audited retrieval study')
    weight_files = {f'weights/s{s}_{arm}.npy' for s in range(BLOCKS) for arm in ARMS}
    audit.check(set(lock['files']) == SOURCE_FILES | weight_files, 'source lock file set')
    for name, sha in lock['files'].items():
        audit.check(freeze['files'][name] == sha == manifest[name]['sha256'] == digest(previous/name),
                    f'byte-identical sealed source: {name}')
    weights = {Path(name).stem: np.load(ROOT/name, allow_pickle=False) for name in weight_files}
    for ident, values in weights.items():
        audit.check(values.shape == (176,) and values.dtype == np.float64 and np.isfinite(values).all(),
                    f'{ident}: valid source weights')
    audit.guard()
    return n, weights


def check_stimuli(audit, seed, n):
    path = ROOT/'evaluate'/f'test_s{seed}.npz'
    base = data.make_base(n, 4073020000+100*seed)
    saved = load(path)
    audit.check(set(saved) == set(base), f'{path.name}: base fields')
    for key, values in base.items():
        audit.exact(saved[key], values, f'{path.name}: {key}')
    x, q = data.assemble(base, 16)
    audit.check(x.shape == (n, T, 6) and q == Q and not x[:, 9:ENTRY].any(), f'{path.name}: stimulus timing')
    audit.check(np.count_nonzero(base['first_bit'] != base['second_bit']) == n//2,
                f'{path.name}: opposite-content count')
    return base, x


def check_trace_rules(audit, raw, x, w, arm, condition, context):
    trace, alpha = raw['traces'], raw['weights_fast']
    audit.check(np.abs(trace).max() <= 1 and np.abs(raw['entry_traces']).max() <= 1,
                f'{context}: bounded signed traces')
    audit.close(trace[:, 9:ENTRY], trace[:, 8:ENTRY-1]*np.array([.85, .97]), f'{context}: blank pure decay')
    audit.exact(raw['injections'][:, 9:ENTRY], np.zeros((8, 16, 12)), f'{context}: blank zero injection')
    wi, wr = split(w)[:2]
    # Join actual archived previous-step values, including the transplanted trace at the boundary.
    previous_p = np.concatenate((np.zeros((8, 1, 6)), raw['states'][:, :-1, 20:]), axis=1)
    previous_t = np.concatenate((np.zeros((8, 1, 12, 2)), trace[:, :-1]), axis=1)
    previous_p[:, ENTRY] = raw['entry_native'][:8, 20:]
    previous_t[:, ENTRY] = raw['entry_traces'][:8]
    drive = np.tanh(x[:8] @ wi + previous_p @ wr)
    gate = np.minimum(1., np.abs(x[:8]).sum(axis=2, keepdims=True))
    difference = previous_t[..., 0]-previous_t[..., 1]
    fast_weight = 1/(1+np.exp(-2*drive*difference))
    if arm == 'SCALE_MEAN':
        fast_weight.fill(.5)
        audit.exact(alpha, fast_weight, f'{context}: exact uniform weights throughout')
    elif condition in ('prefix_mean', 'mean_native'):
        fast_weight[:, :ENTRY] = .5
        audit.exact(alpha[:, :ENTRY], fast_weight[:, :ENTRY], f'{context}: exact uniform prefix weights')
    audit.close(alpha, fast_weight, f'{context}: prespecified pre/post beta at previous-trace read')
    expected_injection = gate*(previous_t[..., 1]+fast_weight*difference)
    audit.close(raw['injections'], expected_injection, f'{context}: input-gated previous-trace injection')
    rate = .5*gate
    expected_write = (1-rate)[..., None]*np.array([.85, .97])*previous_t+(rate*drive)[..., None]
    audit.close(trace, expected_write, f'{context}: unchanged trace write including boundary')


def check_mix(audit, raw, native_source, trace_source, context):
    audit.exact(raw['entry_native'], native_source['entry_native'], f'{context}: native source')
    audit.exact(raw['entry_native'][:8], raw['states'][:, ENTRY-1], f'{context}: native boundary tick')
    for key in ('states', 'spikes', 'injections', 'traces', 'weights_fast'):
        if key in raw:
            audit.exact(raw[key][:, :ENTRY], native_source[key][:, :ENTRY], f'{context}: native prefix archive {key}')
    if 'entry_traces' in raw:
        audit.exact(raw['entry_traces'], trace_source['entry_traces'], f'{context}: trace source')
        audit.exact(raw['entry_traces'][:8], trace_source['traces'][:, ENTRY-1], f'{context}: trace boundary tick')


def entry_changes(dynamic, mean):
    native_d, native_m = dynamic['entry_native'], mean['entry_native']
    graded = np.abs(native_d[:, :20]-native_m[:, :20])
    spikes = native_d[:, 20:] != native_m[:, 20:]
    traces = np.abs(dynamic['entry_traces']-mean['entry_traces'])
    return dict(native_graded_changed=int(np.any(graded > ENTRY_DIFFERENCE, axis=1).sum()),
                native_spike_changed=int(np.any(spikes, axis=1).sum()),
                trace_fast_changed=int(np.any(traces[..., 0] > ENTRY_DIFFERENCE, axis=1).sum()),
                trace_slow_changed=int(np.any(traces[..., 1] > ENTRY_DIFFERENCE, axis=1).sum()),
                max_native_graded_difference=float(np.max(graded)), max_trace_difference=float(np.max(traces)))


def evaluate(audit, n, weights):
    audit.stage = 'replay'
    rows = read(ROOT/'evaluate'/'ROWS.json')
    index = {(row['id'], row['condition']): row for row in rows}
    expected_rows = {(f's{s}_{arm}', condition) for s in range(BLOCKS) for arm in ARMS
                     for condition in independent.CONDITIONS[arm]}
    audit.check(len(rows) == len(index) == 7*BLOCKS and set(index) == expected_rows, 'complete evaluation rows')
    expected_files = {f'{ident}_{condition}.npz' for ident, condition in expected_rows}
    expected_files |= {f'test_s{s}.npz' for s in range(BLOCKS)}
    audit.check({p.name for p in (ROOT/'evaluate').glob('*.npz')} == expected_files, 'exact raw evaluation file set')
    totals = dict(input_array_bytes=0, retained_raw_array_bytes=0, boundary_bank_bytes=0)
    primary = {'prefix_total_effect': [], 'trace_source_effect': []}
    boundary_by_block = []
    mixed_cases = sham_pairs = 0
    for seed in range(BLOCKS):
        base, x = check_stimuli(audit, seed, n)
        totals['input_array_bytes'] += x.nbytes
        for arm in ARMS:
            ident = f's{seed}_{arm}'
            raw_by_condition, scores = {}, {}
            for condition in independent.CONDITIONS[arm]:
                audit.guard()
                context = f'{ident}/{condition}'
                raw = load(ROOT/'evaluate'/f'{ident}_{condition}.npz')
                compare_raw(audit, raw, independent.replay(x, weights[ident], arm, Q, condition), context)
                if arm != 'GAIN2':
                    check_trace_rules(audit, raw, x, weights[ident], arm, condition, context)
                raw_by_condition[condition] = raw
                scores[condition] = verify_row(audit, index[ident, condition], base, raw, ident, seed, arm, condition)
                totals['retained_raw_array_bytes'] += sum(value.nbytes for value in raw.values())
            nominal = raw_by_condition['nominal']
            check_mix(audit, nominal, nominal, nominal, f'{ident}/nominal')
            totals['boundary_bank_bytes'] += sum(value.nbytes for key, value in nominal.items() if key.startswith('entry_'))
            if arm == 'GAIN2':
                continue
            mean = raw_by_condition['prefix_mean']
            check_mix(audit, mean, mean, mean, f'{ident}/prefix_mean')
            totals['boundary_bank_bytes'] += sum(value.nbytes for key, value in mean.items() if key.startswith('entry_'))
            if arm == 'SCALE_MEAN':
                for name in nominal:
                    audit.exact(nominal[name], mean[name], f'{ident}: exact sham {name}')
                sham_pairs += 1
                continue
            check_mix(audit, raw_by_condition['mean_native'], mean, nominal, f'{ident}/mean_native')
            check_mix(audit, raw_by_condition['mean_traces'], nominal, mean, f'{ident}/mean_traces')
            mixed_cases += 2
            dd, mm, md, dm = (scores[name]['disagree_ba'] for name in independent.CONDITIONS[arm])
            primary['prefix_total_effect'].append(dd-mm)
            primary['trace_source_effect'].append(((dd-dm)+(md-mm))/2)
            boundary_by_block.append(dict(seed=seed, **entry_changes(nominal, mean)))
        print(f'Independently replayed block {seed+1}/{BLOCKS}', flush=True)
    audit.check(mixed_cases == 2*BLOCKS and sham_pairs == BLOCKS, 'complete scope and sham controls')
    audit.check(totals['boundary_bank_bytes'] == BLOCKS*n*8*(26+4*50), 'complete boundary-state bank ledger')
    counts = dict(evaluations=len(rows), mixed_cases=mixed_cases, sham_pairs=sham_pairs)
    return totals, counts, primary, boundary_by_block


def main(audit):
    audit.check(all(os.environ.get(key) == '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')),
                'single-thread independent audit')
    n, weights = provenance(audit)
    totals, counts, primary, boundary_by_block = evaluate(audit, n, weights)
    inference = primary_inference(audit, primary)
    audit.stage = 'resource ledger'
    science = read(ROOT/'SCIENCE_SUMMARY.json')
    steps = BLOCKS*n*(5*ENTRY+7*(T-ENTRY))
    path_steps = 7*BLOCKS*n*T
    expected = dict(blocks=BLOCKS, examples_per_block=n, evaluations=7*BLOCKS, fits=0, updates=0,
                    forward_sample_steps=steps, path_sample_steps=path_steps, dense_forward_macs=156*steps, **totals)
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
                  forward_sample_steps=steps, path_sample_steps=path_steps, dense_forward_macs=156*steps,
                  max_numeric_error=audit.max_error, cap_seconds=300, peak_working_set_bytes=peak_memory(),
                  **counts, **audit.elapsed(), **totals, primary_effects=inference, boundary_by_block=boundary_by_block,
                  scope='Independent source provenance, stimulus regeneration, all native replays, exact crossed entry '
                        'origins and sham control, read/write timing and clamp, resource ledgers, and primary block bootstrap.')
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
