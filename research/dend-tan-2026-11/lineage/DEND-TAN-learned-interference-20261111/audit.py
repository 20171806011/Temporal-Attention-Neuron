"""Independent audit of frozen two-prefix interference."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import independent_capacity as capacity
import independent_interference as native
import independent_capacity_data as data


ROOT = Path(__file__).resolve().parent
ARMS = ('LEARN', 'ORACLE', 'MEAN')
T, Q, CUE, BLOCKS = 36, 28, 25, 12
SCIENCE_CAP = 180
SOURCE_FILES = {'capacity_data.py', 'capacity_metrics.py', 'capacity_model.py', 'independent_capacity_data.py',
                'independent_capacity.py', 'learned_model.py', 'population.py', 'reference.py', 'original_tan.py'}
MEMORY_CAP = 1024**3
ARTIFACT_CAP = 3*1024**3//2


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


def regenerated_base(audit, path, n, seed):
    actual = load(path)
    expected = data.make_base(n, seed)
    audit.check(set(actual) == set(expected), f'{path.name}: data fields')
    for key in expected:
        audit.exact(actual[key], expected[key], f'{path.name}: independent {key}')
    return expected


def primary_inference(audit, effects, blocks):
    audit.stage = 'primary inference'
    rng = np.random.default_rng(4079900001)
    draws = rng.integers(0, blocks, (50_000, blocks))
    result = []
    for name, block_values in effects.items():
        values = np.asarray(block_values)
        audit.check(values.shape == (blocks,) and np.isfinite(values).all(), f'{name}: complete primary values')
        bootstrap = np.sum(values[draws], axis=1)/blocks
        low, high = np.quantile(bootstrap, [.0125, .9875])
        direction = 'positive' if low > 0 else 'negative' if high < 0 else 'undetermined'
        result.append(dict(effect=name, blocks=blocks, values=block_values, mean=float(values.mean()),
                           sample_sd=float(values.std(ddof=1)), minimum=float(values.min()), maximum=float(values.max()),
                           positive=int((values > 0).sum()), zero=int((values == 0).sum()),
                           negative=int((values < 0).sum()), ci97_5=[float(low), float(high)], direction=direction))
    return result


def provenance(audit):
    frozen = read(ROOT/'FREEZE.json')
    examples = frozen['examples_per_block']
    audit.check(frozen['scientific_data_generated'] is False and frozen['blocks'] == BLOCKS
                and frozen['evaluations'] == 12*BLOCKS and examples in (512, 256), 'frozen scope')
    audit.check({'PROTOCOL.md', 'REPRODUCE.md', 'SOURCE_LOCK.json', 'PRECHECK.json', 'TIMING.json',
                 'run.py', 'interference_model.py', 'audit.py', 'independent_interference.py', 'independent_capacity_data.py', 'summarize.py', 'report.py'}
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
    audit.exact([row['examples'] for row in candidates], [512] if examples == 512 else [512, 256],
                'timing tried larger size first')
    for i, row in enumerate(candidates):
        for kind in ('science', 'audit'):
            seconds = row[f'{kind}_seconds']
            audit.check(seconds > 0, f'positive {kind} fixture duration')
            audit.close(row[f'{kind}_forecast_seconds'], 36*seconds+30, f'{kind} forecast')
        passed = row['science_forecast_seconds'] < SCIENCE_CAP and row['audit_forecast_seconds'] < 300
        audit.check(passed == (i == len(candidates)-1), 'selected first size satisfying both caps')

    lock = read(ROOT/'SOURCE_LOCK.json')
    audit.check(lock['previous_directory'] == '../DEND-TAN-learned-capacity-20261111', 'exact prior source directory')
    previous = ROOT/lock['previous_directory']
    audit.check(digest(previous/'MANIFEST.json') == lock['previous_manifest_sha256'], 'prior manifest seal')
    manifest = read(previous/'MANIFEST.json')['files']
    for name in ('AUDIT.json', 'SCIENCE_SUMMARY.json'):
        audit.check(digest(previous/name) == manifest[name]['sha256'], f'sealed prior {name}')
    earlier_audit = read(previous/'AUDIT.json')
    earlier_science = read(previous/'SCIENCE_SUMMARY.json')
    audit.check(earlier_audit['status'] == 'PASS' and earlier_science['status'] == 'COMPLETE'
                and earlier_science['blocks'] == BLOCKS and earlier_science['evaluations'] == 144
                and earlier_science['fits'] == 0, 'source is the complete audited 12-block capacity study')
    expected_ids = {f's{s}_{arm}' for s in range(BLOCKS) for arm in ARMS}
    weight_names = {f'weights/{ident}.npy' for ident in expected_ids}
    audit.check(set(lock['files']) == SOURCE_FILES | weight_names, 'complete copied sources and weights')
    for name, sha in lock['files'].items():
        audit.check(frozen['files'][name] == sha == manifest[name]['sha256'] == digest(previous/name),
                    f'byte-identical sealed source: {name}')
    weights = {}
    for ident in sorted(expected_ids):
        array = np.load(ROOT/f'weights/{ident}.npy', allow_pickle=False)
        audit.check(array.shape == (180,) and array.dtype == np.float64 and np.isfinite(array).all(),
                    f'{ident}: unchanged finite native weights')
        weights[ident] = array
    audit.guard()
    return examples, weights


def task_scores(raw, base, slots):
    counts, y, selected = raw['counts'], base['y'], base['selected']
    covered = counts[:, 0] != counts[:, 1]
    correct = covered & (np.argmax(counts, axis=1) == y)
    target_slot = 3-selected
    available = target_slot < slots
    cue_scores = raw['cue_scores'][:, 0]
    maxima = cue_scores.max(axis=1, keepdims=True)
    unique = (cue_scores == maxima).sum(axis=1) == 1
    routes = available & unique & (cue_scores.argmax(axis=1) == target_slot)
    target_weight = np.zeros(len(y))
    target_weight[available] = raw['cue_weights'][available, 0, target_slot[available]]

    def balanced(mask):
        return float(sum(correct[mask & (y == label)].mean() for label in (0, 1))/2)

    older = selected < 2
    scores = dict(balanced_accuracy=balanced(np.ones(len(y), dtype=bool)), older_ba=balanced(older),
                  recent_ba=balanced(~older), coverage=float(covered.mean()), older_coverage=float(covered[older].mean()),
                  recent_coverage=float(covered[~older].mean()), routing_accuracy=float(routes.mean()),
                  older_routing_accuracy=float(routes[older].mean()), recent_routing_accuracy=float(routes[~older].mean()),
                  target_weight=float(target_weight.mean()), older_target_weight=float(target_weight[older].mean()),
                  recent_target_weight=float(target_weight[~older].mean()), available_fraction=float(available.mean()))
    positions = [selected == position for position in range(4)]
    scores['ba_by_position'] = [balanced(mask) for mask in positions]
    for key, vector in (('coverage', covered), ('routing', routes), ('target_weight', target_weight)):
        scores[f'{key}_by_position'] = [float(vector[mask].mean()) for mask in positions]
    return scores


def cost(arm, condition):
    slots = 2
    hard_steps = 0
    matched_steps = 0
    return dict(parameters=180 if arm == 'LEARN' else 176, allocated_parameter_slots=180,
                fixed_match_constants=0 if arm == 'LEARN' else 4, spiking_cells=8, graded_branches=12,
                continuous_state_slots=20+14*slots, key_slots=2*slots, value_slots=12*slots,
                persistent_array_bytes=8*(28+14*slots), dense_forward_macs=156*T, graded_transfers=12*T,
                matcher_dense_macs=(4+2*slots)*T, matcher_logit_scale_multiplies=slots*T,
                softmax_exponentials=slots*T, softmax_divisions=slots*T, query_norm_square_multiplies=2*T,
                query_norm_adds=T, query_norm_sqrts=T, value_tanh_evaluations=12*T,
                value_decay_multiplies=12*slots*T, value_read_multiplies=12*slots*T,
                value_read_adds=12*(slots-1)*T, injection_multiplies=12*T,
                fifo_scalar_writes=14*slots*(2 if condition.startswith('late_') else 4),
                hard_equalities=slots*hard_steps, hard_tie_adds=(slots-1)*hard_steps,
                hard_divisions=slots*hard_steps, extra_value_read_multiplies=12*slots*matched_steps,
                extra_value_read_adds=12*(slots-1)*matched_steps, extra_injection_multiplies=12*matched_steps,
                normalization_norms=2*matched_steps, normalization_square_multiplies=24*matched_steps,
                normalization_sum_adds=22*matched_steps, normalization_sqrts=2*matched_steps,
                normalization_ratio_divisions_upper=matched_steps, normalization_scale_multiplies=12*matched_steps)


def compare_raw(audit, saved, expected, context):
    audit.check(set(saved) == set(expected), f'{context}: raw fields')
    discrete = {'counts', 'spikes', 'resources'}
    for key in sorted(discrete):
        audit.exact(saved[key], expected[key], f'{context}: exact {key}')
    for key in sorted(set(expected)-discrete):
        audit.close(saved[key], expected[key], f'{context}: {key}', 5e-12)
    counts = saved['counts']
    audit.check(np.all((counts >= 0) & (counts <= 8)) and np.array_equal(counts, np.rint(counts)),
                f'{context}: native count bounds')


def verify_oracle(audit, raw, base, slots, context):
    target = 3-base['selected']
    available = target < slots
    score = raw['cue_scores'][available]
    selected = target[available]
    indices = np.arange(len(selected))[:, None]
    ticks = np.arange(3)[None, :]
    target_scores = score[indices, ticks, selected[:, None]]
    competitors = score.copy()
    competitors[indices, ticks, selected[:, None]] = -np.inf
    minimum = float((target_scores-competitors.max(axis=-1)).min())
    audit.check(minimum >= 4*data.ORACLE_MARGIN_BOUND-1e-12, f'{context}: analytic fixed-oracle score margin')
    return minimum/4


def verify_archive(audit, raw, prefix_x, suffix_x, context):
    n = len(prefix_x)
    state, keys, values = raw['entry_native'], raw['entry_keys'], raw['entry_values']
    audit.check(state.shape == (n, 26) and keys.shape == (n, 2, 2) and values.shape == (n, 12, 2),
                f'{context}: complete boundary archive')
    audit.check(np.isin(state[:, 20:], (0., 1.)).all(), f'{context}: binary previous hidden pulse')
    audit.exact(state[:8, :20], raw['states'][:, CUE-1], f'{context}: boundary native trajectory')
    audit.exact(state[:8, 20:], raw['spikes'][:, CUE-1, :6], f'{context}: boundary previous pulse')
    audit.exact(keys[:8], raw['key_bank'][:, CUE-1], f'{context}: boundary keys')
    audit.check(np.abs(values).max() <= 1., f'{context}: bounded boundary values')

    actual_x = np.concatenate((prefix_x[:8, :CUE], suffix_x[:8, CUE:]), axis=1)
    observed, marker = actual_x[..., 2:4], actual_x[..., 5:6]
    read_gate = (1-marker)*np.minimum(1., np.linalg.norm(observed, axis=-1, keepdims=True))
    audit.exact(raw['write_gate'], marker, f'{context}: visible write marker')
    audit.close(raw['read_gate'], read_gate, f'{context}: visible read gate', 1e-12)
    audit.check(not marker[:, CUE:].any() and not read_gate[:, :CUE].any() and not read_gate[:, Q:].any(),
                f'{context}: causal write and read times')

    key_bank, value_bank = raw['key_bank'], raw['value_bank']
    previous_keys = np.concatenate((np.zeros_like(key_bank[:, :1]), key_bank[:, :-1]), axis=1)
    previous_values = np.concatenate((np.zeros_like(value_bank[:, :1]), value_bank[:, :-1]), axis=1)
    previous_values[:, CUE] = values[:8]
    key_shift = np.stack((observed, previous_keys[:, :, 0]), axis=2)
    retained = .97*previous_values
    value_shift = np.stack((raw['drive'], retained[..., 0]), axis=-1)
    audit.exact(key_bank, np.where(marker[..., None] == 1., key_shift, previous_keys), f'{context}: two-slot FIFO keys')
    audit.close(value_bank, np.where(marker[..., None] == 1., value_shift, retained),
                f'{context}: two-slot FIFO values and changed entry', 1e-12)

    scores, weights, injected = (raw[name] for name in native.CUE_ARRAYS)
    audit.check(scores.shape == weights.shape == (n, 3, 2) and injected.shape == (n, 3, 12),
                f'{context}: complete cue archive')
    exp_scores = np.exp(scores-scores.max(axis=-1, keepdims=True))
    audit.close(weights, exp_scores/exp_scores.sum(axis=-1, keepdims=True), f'{context}: recorded cue softmax', 1e-12)
    decay = .97**np.arange(3)
    gate = np.minimum(1., np.linalg.norm(suffix_x[:, CUE:Q, 2:4], axis=-1))
    cue_reads = 2*gate[..., None]*decay[None, :, None]*np.einsum('ntk,ndk->ntd', weights, values, optimize=False)
    audit.close(injected, cue_reads, f'{context}: all cue reads from actual entry values', 1e-12)
    for saved_name, trace_name in (('cue_scores', 'scores'), ('cue_weights', 'weights'), ('cue_read_vectors', 'injections')):
        audit.exact(raw[saved_name][:8], raw[trace_name][:, CUE:Q], f'{context}: cue trace {trace_name}')
    audit.exact(raw['counts'][:8], raw['spikes'][:, Q:, 6:].sum(axis=1), f'{context}: query count archive')


def verify_paths(audit, paths, full_x, late_x, weights, arm, context):
    full, late = paths['full_full'], paths['late_late']
    natural = {'full': full, 'late': late}
    for name, raw in paths.items():
        source_n, source_v = name.split('_')
        native_donor, value_donor = natural[source_n], natural[source_v]
        audit.exact(raw['entry_native'], native_donor['entry_native'], f'{context}/{name}: exact native donor')
        audit.exact(raw['entry_values'], value_donor['entry_values'], f'{context}/{name}: exact value donor')
        for field in ('entry_keys', 'cue_weights', 'cue_scores'):
            audit.exact(raw[field], full[field], f'{context}/{name}: shared {field}')
        for field in native.TRACES:
            audit.exact(raw[field][:, :CUE], native_donor[field][:, :CUE], f'{context}/{name}: native prefix {field}')
    for label, x, condition in (('full', full_x, 'full2'), ('late', late_x, 'base2')):
        raw = natural[label]
        audit.exact(raw['entry_values'][:8], raw['value_bank'][:, CUE-1], f'{context}/{label}: natural entry values')
        earlier = capacity.replay(x, weights, arm, Q, condition)
        for field in sorted(set(raw) & set(earlier)):
            if field in ('counts', 'spikes', 'resources'):
                audit.exact(raw[field], earlier[field], f'{context}/{label}: previous native {field}')
            else:
                audit.close(raw[field], earlier[field], f'{context}/{label}: previous trajectory {field}', 5e-12)


def verify_row(audit, row, raw, base, seed, arm, condition):
    ident = f's{seed}_{arm}'
    context = f'{ident}/{condition}'
    identity = dict(id=ident, seed=seed, arm=arm, condition=condition, T=T, q=Q, slots=2)
    scores = task_scores(raw, base, 2)
    audit.check(set(row) == set(identity) | set(scores) | {'mean_resources', 'cost_per_episode'},
                f'{context}: row schema')
    for key, value in identity.items():
        audit.check(row[key] == value, f'{context}: {key}')
    for key, value in scores.items():
        audit.close(row[key], value, f'{context}: native {key}', 1e-13)
    audit.close(row['mean_resources'], raw['resources'][:, :4].mean(axis=0), f'{context}: resource means', 1e-12)
    audit.check(row['cost_per_episode'] == cost(arm, condition), f'{context}: per-path structural costs')
    return scores


def evaluate(audit, n, weights):
    audit.stage = 'replay'
    rows = read(ROOT/'evaluate'/'ROWS.json')
    index = {(row['id'], row['condition']): row for row in rows}
    expected = {(f's{s}_{arm}', condition) for s in range(BLOCKS) for arm in ARMS for condition in native.CONDITIONS}
    audit.check(len(rows) == len(index) == 12*BLOCKS and set(index) == expected, 'complete 144 paired evaluation rows')
    files = {f'{ident}_{condition}.npz' for ident, condition in expected} | {f'test_s{s}.npz' for s in range(BLOCKS)}
    audit.check({p.name for p in (ROOT/'evaluate').glob('*.npz')} == files, 'exact evaluation archive set')
    effects = {'native_effect': [], 'value_effect': []}
    input_bytes = raw_bytes = duplicates = 0
    oracle_min = np.inf
    for seed in range(BLOCKS):
        base = regenerated_base(audit, ROOT/'evaluate'/f'test_s{seed}.npz', n, 4079020000+100*seed)
        full_x, q = data.assemble(base, 'full4')
        late_x, q_late = data.assemble(base, 'base2')
        audit.check(q == q_late == Q and data.check(base, full_x)['ceiling'] == 1., f'block {seed}: full task')
        audit.check(data.check(base, late_x, 'base2')['ceiling'] == .75, f'block {seed}: deletion ceiling')
        kept = [tick for tick in range(T) if tick not in (0, 2)]
        audit.exact(full_x[:, kept], late_x[:, kept], f'block {seed}: only early writes removed')
        audit.check(not late_x[:, (0, 2)].any(), f'block {seed}: no residual early write channels')
        left, right = data.pairs(base).T
        audit.exact(late_x[left], late_x[right], f'block {seed}: identical missing-label inputs')
        audit.exact(base['y'][left], 1-base['y'][right], f'block {seed}: opposite missing labels')
        input_bytes += full_x.nbytes+late_x.nbytes
        for arm in ARMS:
            ident = f's{seed}_{arm}'
            paths, scores = {}, {}
            for condition in native.CONDITIONS:
                audit.guard()
                context = f'{ident}/{condition}'
                raw = load(ROOT/'evaluate'/f'{ident}_{condition}.npz')
                replayed = native.replay(full_x, late_x, weights[ident], arm, q, condition)
                compare_raw(audit, raw, replayed, context)
                audit.exact(raw['entry_native'][:, 20:], replayed['entry_native'][:, 20:],
                            f'{context}: exact boundary pulse replay')
                verify_archive(audit, raw, full_x if condition.startswith('full_') else late_x, full_x, context)
                scores[condition] = verify_row(audit, index[ident, condition], raw, base, seed, arm, condition)
                if condition == 'late_late':
                    audit.exact(raw['counts'][left], raw['counts'][right], f'{context}: exact native counts on duplicates')
                    audit.check(scores[condition]['older_ba'] == .5*scores[condition]['older_coverage'],
                                f'{context}: missing-record BA equals half coverage')
                    duplicates += len(left)
                if arm == 'ORACLE':
                    oracle_min = min(oracle_min, verify_oracle(audit, raw, base, 2, context))
                paths[condition] = raw
                raw_bytes += sum(value.nbytes for value in raw.values())
            verify_paths(audit, paths, full_x, late_x, weights[ident], arm, ident)
            if arm == 'LEARN':
                ff, lf, fl, ll = (scores[condition]['recent_ba'] for condition in native.CONDITIONS)
                native_effect = ((lf-ff)+(ll-fl))/2.
                value_effect = ((fl-ff)+(ll-lf))/2.
                audit.check(native_effect+value_effect == ll-ff, f'{ident}: paired decomposition recovers natural difference')
                effects['native_effect'].append(native_effect)
                effects['value_effect'].append(value_effect)
        print(f'Independently replayed interference block {seed+1}/{BLOCKS}', flush=True)
    diagnostics = dict(duplicate_opposite_label_pairs=duplicates, oracle_margin_min=float(oracle_min),
                       oracle_margin_bound=float(data.ORACLE_MARGIN_BOUND))
    return effects, input_bytes, raw_bytes, diagnostics


def main(audit):
    audit.check(all(os.environ.get(key) == '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')),
                'single-thread independent audit')
    n, weights = provenance(audit)
    effects, input_bytes, raw_bytes, diagnostics = evaluate(audit, n, weights)
    inference = primary_inference(audit, effects, BLOCKS)
    audit.stage = 'resource ledger'
    science = read(ROOT/'SCIENCE_SUMMARY.json')
    steps = 3*BLOCKS*n*(2*25+4*11)
    expected = dict(blocks=BLOCKS, examples_per_block=n, evaluations=12*BLOCKS, fits=0, updates=0,
                    forward_sample_steps=steps, path_sample_steps=12*BLOCKS*n*T, dense_forward_macs=156*steps,
                    matcher_macs=8*steps, boundary_bank_bytes=3*BLOCKS*n*2*(26+4+24)*8,
                    input_array_bytes=input_bytes, retained_raw_array_bytes=raw_bytes)
    audit.check(input_bytes == 2*BLOCKS*n*T*6*8, 'independent input-byte ledger')
    audit.check(science['status'] == 'COMPLETE' and science['cap_seconds'] == SCIENCE_CAP
                and 0 < science['wall_seconds'] <= SCIENCE_CAP and science['cpu_seconds'] > 0, 'science wall and CPU time')
    audit.check(0 < science['peak_working_set_bytes'] <= MEMORY_CAP, 'science memory cap')
    audit.check(0 < science['artifact_bytes_at_science_end'] <= ARTIFACT_CAP, 'science artifact cap')
    audit.check(sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) < ARTIFACT_CAP, 'current artifact cap')
    audit.check(read(ROOT/'RUN_STATE.json')['status'] == 'complete', 'science completion state')
    audit.check(read(ROOT/'ENVIRONMENT.json')['threads'] == {
        key: '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}, 'science thread scope')
    for key, value in expected.items():
        audit.check(science[key] == value, f'science ledger {key}')
    audit.guard()
    result = dict(status='PASS', checks=audit.checks, **expected, max_numeric_error=audit.max_error,
                  cap_seconds=300, peak_working_set_bytes=peak_memory(), **audit.elapsed(), **diagnostics,
                  primary_effects=inference,
                  scope='Independent sealed-weight provenance, regenerated stimuli, all frozen native replays, '
                        'natural and mixed boundary sources, common cue and fixed keys, exact deleted-label counts, '
                        'original natural trajectories, resource ledgers and paired bootstrap effects.')
    with (ROOT/'AUDIT.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    assert not any((ROOT/name).exists() for name in ('AUDIT.json', 'AUDIT_FAILURE.json')), 'audit already attempted'
    auditor = Audit()
    try:
        main(auditor)
    except Exception as exc:
        failure = dict(status='FAIL', stage=auditor.stage, checks=auditor.checks, reason=str(exc),
                       exception=type(exc).__name__, **auditor.elapsed())
        with (ROOT/'AUDIT_FAILURE.json').open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(failure, indent=2, allow_nan=False)+'\n')
        raise


