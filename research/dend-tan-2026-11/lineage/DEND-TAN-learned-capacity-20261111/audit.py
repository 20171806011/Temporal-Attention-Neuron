"""Independent audit of frozen learned-key capacity and matched hard reads."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import independent_capacity as native
import independent_capacity_data as data


ROOT = Path(__file__).resolve().parent
ARMS = ('LEARN', 'ORACLE', 'MEAN')
MATCHED = 'full4_hard_matched'
T, Q, CUE, BLOCKS = 36, 28, 25, 12
SCIENCE_CAP = 180
SOURCE_FILES = {'population.py', 'model.py', 'metrics.py', 'reference.py', 'original_tan.py',
                'learned_model.py', 'learned_data.py', 'independent_learned.py', 'independent_learned_data.py'}
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
    rng = np.random.default_rng(4078900001)
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
                 'run.py', 'capacity_model.py', 'audit.py', 'independent_capacity.py', 'independent_capacity_data.py', 'summarize.py', 'report.py'}
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
    audit.check(lock['previous_directory'] == '../DEND-TAN-learned-key-20261110', 'exact prior source directory')
    previous = ROOT/lock['previous_directory']
    audit.check(digest(previous/'MANIFEST.json') == lock['previous_manifest_sha256'], 'prior manifest seal')
    manifest = read(previous/'MANIFEST.json')['files']
    for name in ('AUDIT.json', 'SCIENCE_SUMMARY.json'):
        audit.check(digest(previous/name) == manifest[name]['sha256'], f'sealed prior {name}')
    earlier_audit = read(previous/'AUDIT.json')
    earlier_science = read(previous/'SCIENCE_SUMMARY.json')
    audit.check(earlier_audit['status'] == 'PASS' and earlier_science['status'] == 'COMPLETE'
                and earlier_science['blocks'] == BLOCKS and earlier_science['fits'] == 36,
                'source is the complete audited 12-block learned-key study')
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
        source, weight_file = f'train/{ident}.npz', f'weights/{ident}.npy'
        audit.check(rec['source'] == source and rec['weight_file'] == weight_file, f'{ident}: source and extract paths')
        audit.check(digest(previous/source) == rec['source_sha256'] == manifest[source]['sha256'],
                    f'{ident}: sealed checkpoint archive')
        audit.check(rec['weight_sha256'] == lock['files'][weight_file], f'{ident}: checkpoint extract lock')
        with np.load(previous/source, allow_pickle=False) as archive:
            theta = archive['theta']
        weights[ident] = np.load(ROOT/weight_file, allow_pickle=False)
        audit.check(theta.shape == (180,) and theta.dtype == weights[ident].dtype == np.float64
                    and np.isfinite(theta).all(), f'{ident}: native weights')
        audit.exact(weights[ident], theta, f'{ident}: final checkpoint theta preserved')
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
    slots = native.slot_count(condition)
    hard_steps = T if condition in ('full4_hard', MATCHED) else 0
    matched_steps = T if condition == MATCHED else 0
    return dict(parameters=180 if arm == 'LEARN' else 176, allocated_parameter_slots=180,
                fixed_match_constants=0 if arm == 'LEARN' else 4, spiking_cells=8, graded_branches=12,
                continuous_state_slots=20+14*slots, key_slots=2*slots, value_slots=12*slots,
                persistent_array_bytes=8*(28+14*slots), dense_forward_macs=156*T, graded_transfers=12*T,
                matcher_dense_macs=(4+2*slots)*T, matcher_logit_scale_multiplies=slots*T,
                softmax_exponentials=slots*T, softmax_divisions=slots*T, query_norm_square_multiplies=2*T,
                query_norm_adds=T, query_norm_sqrts=T, value_tanh_evaluations=12*T,
                value_decay_multiplies=12*slots*T, value_read_multiplies=12*slots*T,
                value_read_adds=12*(slots-1)*T, injection_multiplies=12*T,
                fifo_scalar_writes=14*slots*(2 if condition == 'base2' else 4),
                hard_equalities=slots*hard_steps, hard_tie_adds=(slots-1)*hard_steps,
                hard_divisions=slots*hard_steps, extra_value_read_multiplies=12*slots*matched_steps,
                extra_value_read_adds=12*(slots-1)*matched_steps, extra_injection_multiplies=12*matched_steps,
                normalization_norms=2*matched_steps, normalization_square_multiplies=24*matched_steps,
                normalization_sum_adds=22*matched_steps, normalization_sqrts=2*matched_steps,
                normalization_ratio_divisions_upper=matched_steps, normalization_scale_multiplies=12*matched_steps)


def compare_raw(audit, saved, expected, context):
    audit.check(set(saved) == set(expected), f'{context}: raw fields')
    discrete = {'counts', 'spikes', 'resources', 'cue_fallback'}
    for key in sorted(discrete):
        audit.exact(saved[key], expected[key], f'{context}: exact {key}')
    for key in sorted(set(expected)-discrete):
        audit.close(saved[key], expected[key], f'{context}: {key}', 5e-12)
    counts = saved['counts']
    audit.check(np.all((counts >= 0) & (counts <= 8)) and np.array_equal(counts, np.rint(counts)),
                f'{context}: native count bounds')


def verify_archive(audit, raw, x, condition, context):
    slots, n = native.slot_count(condition), len(x)
    keys, values = raw['key_bank'], raw['value_bank']
    audit.check(keys.shape == (8, T, slots, 2) and values.shape == (8, T, 12, slots), f'{context}: complete bank archive')
    audit.check(np.abs(values).max() <= 1., f'{context}: bounded values')
    audit.check(raw['cue_read_vectors'].shape == (n, 3, 12) and raw['cue_fallback'].shape == (n, 3)
                and raw['cue_fallback'].dtype == np.bool_, f'{context}: complete cue vectors and fallback archive')
    audit.exact(raw['cue_read_vectors'][:8], raw['injections'][:, CUE:Q], f'{context}: cue injections agree with trace')
    audit.exact(raw['cue_scores'][:8], raw['scores'][:, CUE:Q], f'{context}: cue scores agree with trace')
    audit.exact(raw['cue_weights'][:8], raw['weights'][:, CUE:Q], f'{context}: cue weights agree with trace')
    audit.exact(raw['counts'][:8], raw['spikes'][:, Q:, 6:].sum(axis=1), f'{context}: query count archive')
    if condition != MATCHED:
        audit.check(not raw['cue_fallback'].any(), f'{context}: fallback reserved for matched read')

    marker, observed = x[:8, :, 5:6], x[:8, :, 2:4]
    read = (1-marker)*np.minimum(1., native.norm(observed)[..., None])
    audit.exact(raw['write_gate'], marker, f'{context}: visible write marker')
    audit.close(raw['read_gate'], read, f'{context}: visible read gate', 1e-12)
    audit.check(not marker[:, CUE:].any() and not read[:, :CUE].any() and not read[:, Q:].any(),
                f'{context}: causal write and read times')
    silent = read[..., 0] == 0.
    audit.exact(raw['injections'][silent], np.zeros_like(raw['injections'][silent]), f'{context}: no injection at zero gate')

    prev_keys = np.concatenate((np.zeros_like(keys[:, :1]), keys[:, :-1]), axis=1)
    prev_values = np.concatenate((np.zeros_like(values[:, :1]), values[:, :-1]), axis=1)
    shift_keys = np.concatenate((observed[..., None, :], prev_keys[:, :, :-1]), axis=2)
    retained = .97*prev_values
    shift_values = np.concatenate((raw['drive'][..., None], retained[..., :-1]), axis=-1)
    audit.exact(keys, np.where(marker[..., None] == 1., shift_keys, prev_keys), f'{context}: FIFO key order')
    audit.close(values, np.where(marker[..., None] == 1., shift_values, retained),
                f'{context}: FIFO value order and decay', 1e-12)

    scores = raw['cue_scores']
    exponentials = np.exp(scores-scores.max(axis=-1, keepdims=True))
    soft_weights = exponentials/exponentials.sum(axis=-1, keepdims=True)
    audit.close(raw['cue_soft_weights'], soft_weights, f'{context}: softmax of recorded scores', 1e-12)
    weights = native.hard_weights(scores) if condition in ('full4_hard', MATCHED) else soft_weights
    audit.close(raw['cue_weights'], weights, f'{context}: applied read weights', 1e-12)
    cue_values = prev_values[:, CUE:Q]
    direct = 2*read[:, CUE:Q]*np.einsum('ntk,ntdk->ntd', weights[:8], cue_values, optimize=False)
    if condition == MATCHED:
        target = 2*read[:, CUE:Q]*np.einsum('ntk,ntdk->ntd', soft_weights[:8], cue_values, optimize=False)
        direct, fallback = native.matched_injection(direct, target)
        audit.exact(raw['cue_fallback'][:8], fallback, f'{context}: fallback from trace banks')
    audit.close(raw['cue_read_vectors'][:8], direct, f'{context}: independently contracted trace-bank read', 1e-12)


def verify_matched(audit, raw, hard, soft, context):
    vectors = raw['cue_read_vectors']
    direction = hard['cue_read_vectors']
    target = soft['cue_read_vectors']
    dir_norm, target_norm, actual_norm = (np.linalg.norm(vector, axis=-1) for vector in (direction, target, vectors))
    norm_residual = audit.close(actual_norm, target_norm, f'{context}: per-read injection norm', 1e-12)
    zero_target = target_norm == 0.
    fallback = (dir_norm == 0.) & ~zero_target
    audit.exact(raw['cue_fallback'], fallback, f'{context}: zero direction with positive target')
    audit.exact(vectors[zero_target], np.zeros_like(vectors[zero_target]), f'{context}: zero-target injection')
    fallback_vectors = np.zeros_like(vectors[fallback])
    fallback_vectors[:, 0] = target_norm[fallback]
    audit.close(vectors[fallback], fallback_vectors, f'{context}: first-branch unit fallback', 1e-12)
    valid = ~fallback & ~zero_target
    audit.check(np.all(actual_norm[valid] > 0.), f'{context}: nonzero matched reads')
    dir_residual = audit.close(vectors[valid]/actual_norm[valid, None], direction[valid]/dir_norm[valid, None],
                               f'{context}: hard-read direction preserved', 1e-12)
    return norm_residual, dir_residual


def verify_paths(audit, paths, context):
    two, four = paths['full2'], paths['full4']
    for condition, raw in paths.items():
        if condition in ('base2', 'full4'):
            continue
        for key in ('states', 'spikes', 'candidates', 'currents', 'injections', 'drive'):
            audit.exact(raw[key][:, :CUE], four[key][:, :CUE], f'{context}/{condition}: identical prefix {key}')
        if condition == 'full2':
            audit.exact(two['key_bank'], four['key_bank'][..., :2, :], f'{context}: shared newest key slots')
            audit.exact(two['value_bank'], four['value_bank'][..., :2], f'{context}: shared newest value slots')
        else:
            for key in ('key_bank', 'value_bank', 'cue_scores', 'cue_soft_weights', 'read_gate', 'write_gate'):
                audit.exact(raw[key], four[key], f'{context}/{condition}: unchanged bank or score {key}')
    if MATCHED in paths:
        return verify_matched(audit, paths[MATCHED], paths['full4_hard'], four, context)
    return 0., 0.


def verify_row(audit, row, raw, base, seed, arm, condition):
    ident = f's{seed}_{arm}'
    context = f'{ident}/{condition}'
    slots = native.slot_count(condition)
    identity = dict(id=ident, seed=seed, arm=arm, condition=condition, T=T, q=Q, slots=slots)
    scores = task_scores(raw, base, slots)
    audit.check(set(row) == set(identity) | set(scores) | {'mean_resources', 'cost_per_episode', 'fallback_reads'},
                f'{context}: row schema')
    for key, value in identity.items():
        audit.check(row[key] == value, f'{context}: {key}')
    for key, value in scores.items():
        audit.close(row[key], value, f'{context}: {key}', 1e-13)
    audit.close(row['mean_resources'], raw['resources'][:, :4].mean(axis=0), f'{context}: event and fanout ledger', 1e-12)
    audit.check(row['cost_per_episode'] == cost(arm, condition), f'{context}: per-episode structural costs')
    audit.check(row['fallback_reads'] == np.count_nonzero(raw['cue_fallback']), f'{context}: fallback read count')
    return scores


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


def evaluate(audit, n, weights):
    audit.stage = 'replay'
    rows = read(ROOT/'evaluate'/'ROWS.json')
    index = {(row['id'], row['condition']): row for row in rows}
    expected = {(f's{s}_{arm}', condition) for s in range(BLOCKS) for arm in ARMS for condition in native.CONDITIONS[arm]}
    audit.check(len(rows) == len(index) == 12*BLOCKS and set(index) == expected, 'complete 144 paired evaluation rows')
    files = {f'{ident}_{condition}.npz' for ident, condition in expected} | {f'test_s{s}.npz' for s in range(BLOCKS)}
    audit.check({p.name for p in (ROOT/'evaluate').glob('*.npz')} == files, 'exact evaluation archive set')
    totals = dict(input_array_bytes=0, retained_raw_array_bytes=0, matched_cases=0,
                  matched_read_steps=0, fallback_reads=0)
    effects = {'capacity_gain': [], 'matched_hard_gain': []}
    norm_residual = direction_residual = 0.
    duplicates, oracle_min = 0, np.inf
    for seed in range(BLOCKS):
        base = regenerated_base(audit, ROOT/'evaluate'/f'test_s{seed}.npz', n, 4078020000+100*seed)
        x_full, q = data.assemble(base, 'full4')
        x_base, q_base = data.assemble(base, 'base2')
        audit.check(q == q_base == Q and data.check(base, x_full)['ceiling'] == 1., f'block {seed}: full task and margin')
        audit.check(data.check(base, x_base, 'base2')['ceiling'] == .75, f'block {seed}: deletion ceiling')
        left, right = data.pairs(base).T
        audit.exact(x_base[left], x_base[right], f'block {seed}: exact missing-label input pairs')
        audit.exact(base['y'][left], 1-base['y'][right], f'block {seed}: opposite missing labels')
        totals['input_array_bytes'] += x_full.nbytes+x_base.nbytes
        for arm in ARMS:
            ident = f's{seed}_{arm}'
            paths, scores = {}, {}
            for condition in native.CONDITIONS[arm]:
                audit.guard()
                context = f'{ident}/{condition}'
                x = x_base if condition == 'base2' else x_full
                raw = load(ROOT/'evaluate'/f'{ident}_{condition}.npz')
                compare_raw(audit, raw, native.replay(x, weights[ident], arm, q, condition), context)
                verify_archive(audit, raw, x, condition, context)
                scores[condition] = verify_row(audit, index[ident, condition], raw, base, seed, arm, condition)
                if condition == 'base2':
                    audit.exact(raw['counts'][left], raw['counts'][right], f'{context}: exact native counts on duplicates')
                    audit.check(scores[condition]['older_ba'] == .5*scores[condition]['older_coverage'],
                                f'{context}: missing-record BA equals half coverage')
                    duplicates += len(left)
                if arm == 'ORACLE':
                    oracle_min = min(oracle_min, verify_oracle(audit, raw, base, native.slot_count(condition), context))
                paths[condition] = raw
                totals['retained_raw_array_bytes'] += sum(value.nbytes for value in raw.values())
                totals['fallback_reads'] += int(np.count_nonzero(raw['cue_fallback']))
                if condition == MATCHED:
                    totals['matched_cases'] += 1
                    totals['matched_read_steps'] += raw['cue_fallback'].size
            norm, direction = verify_paths(audit, paths, ident)
            norm_residual = max(norm_residual, norm)
            direction_residual = max(direction_residual, direction)
            if arm == 'LEARN':
                effects['capacity_gain'].append(scores['full4']['older_ba']-scores['full2']['older_ba'])
                effects['matched_hard_gain'].append(scores[MATCHED]['older_ba']-scores['full4']['older_ba'])
        print(f'Independently replayed capacity block {seed+1}/{BLOCKS}', flush=True)
    audit.check(totals['matched_cases'] == BLOCKS and totals['matched_read_steps'] == BLOCKS*n*3,
                'complete matched read scope')
    diagnostics = dict(max_matched_norm_residual=norm_residual, max_matched_direction_residual=direction_residual,
                       duplicate_opposite_label_pairs=duplicates, oracle_margin_min=float(oracle_min),
                       oracle_margin_bound=float(data.ORACLE_MARGIN_BOUND))
    return effects, totals, diagnostics


def main(audit):
    audit.check(all(os.environ.get(key) == '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')),
                'single-thread independent audit')
    n, weights = provenance(audit)
    effects, totals, diagnostics = evaluate(audit, n, weights)
    inference = primary_inference(audit, effects, BLOCKS)
    audit.stage = 'resource ledger'
    science = read(ROOT/'SCIENCE_SUMMARY.json')
    steps = 12*BLOCKS*n*T
    expected = dict(blocks=BLOCKS, examples_per_block=n, evaluations=12*BLOCKS, fits=0, updates=0,
                    forward_sample_steps=steps, dense_forward_macs=156*steps, **totals)
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
                  scope='Independent checkpoint provenance, four-record stimuli, all frozen native evaluations, '
                        'FIFO states and slot scores, deleted-label duplicate inputs and counts, fixed oracle margin, '
                        'hard-read and norm matching, resource ledgers and paired primary bootstraps.')
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


