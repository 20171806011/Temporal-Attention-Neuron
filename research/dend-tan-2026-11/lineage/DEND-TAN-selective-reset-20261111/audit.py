"""Independent audit of frozen read-onset selective reset."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import independent_capacity as capacity
import independent_reset as native
import independent_capacity_data as data


ROOT = Path(__file__).resolve().parent
ARMS = ('LEARN', 'ORACLE', 'MEAN')
T, Q, CUE, BLOCKS = 36, 28, 25, 12
SCIENCE_CAP = 180
CONDITIONS = tuple(f'{prefix}_{rule}' for prefix in ('full', 'late') for rule in native.RULES)
EVALUATIONS = 18*BLOCKS
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
    rng = np.random.default_rng(4080900001)
    draws = rng.integers(0, blocks, (50_000, blocks))
    result = []
    thresholds = {'recent_gain': 0., 'older_change': -.02}
    for name, block_values in effects.items():
        values = np.asarray(block_values)
        audit.check(values.shape == (blocks,) and np.isfinite(values).all(), f'{name}: complete primary values')
        bootstrap = np.sum(values[draws], axis=1)/blocks
        low, high = np.quantile(bootstrap, [.0125, .9875])
        direction = 'positive' if low > 0 else 'negative' if high < 0 else 'undetermined'
        result.append(dict(effect=name, blocks=blocks, values=block_values, mean=float(values.mean()),
                           sample_sd=float(values.std(ddof=1)), minimum=float(values.min()), maximum=float(values.max()),
                           positive=int((values > 0).sum()), zero=int((values == 0).sum()),
                           negative=int((values < 0).sum()), ci97_5=[float(low), float(high)], direction=direction,
                           threshold=thresholds[name], passes=bool(low > thresholds[name])))
    return result


def provenance(audit):
    frozen = read(ROOT/'FREEZE.json')
    examples = frozen['examples_per_block']
    audit.check(frozen['scientific_data_generated'] is False and frozen['blocks'] == BLOCKS
                and frozen['evaluations'] == EVALUATIONS and examples in (512, 256), 'frozen scope')
    audit.check({'PROTOCOL.md', 'REPRODUCE.md', 'SOURCE_LOCK.json', 'PRECHECK.json', 'TIMING.json',
                 'run.py', 'reset_model.py', 'audit.py', 'independent_reset.py', 'independent_capacity_data.py', 'summarize.py', 'report.py'}
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
    audit.check(lock['previous_directory'] == '../DEND-TAN-learned-interference-20261111', 'exact prior source directory')
    previous = ROOT/lock['previous_directory']
    audit.check(digest(previous/'MANIFEST.json') == lock['previous_manifest_sha256'], 'prior manifest seal')
    manifest = read(previous/'MANIFEST.json')['files']
    for name in ('AUDIT.json', 'SCIENCE_SUMMARY.json'):
        audit.check(digest(previous/name) == manifest[name]['sha256'], f'sealed prior {name}')
    earlier_audit = read(previous/'AUDIT.json')
    earlier_science = read(previous/'SCIENCE_SUMMARY.json')
    audit.check(earlier_audit['status'] == 'PASS' and earlier_science['status'] == 'COMPLETE'
                and earlier_science['blocks'] == BLOCKS and earlier_science['evaluations'] == 144
                and earlier_science['fits'] == 0, 'source is the complete audited 12-block interference study')
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


def cost(arm, condition, reset_count, examples):
    slots = 2
    hard_steps = 0
    matched_steps = 0
    rule = condition.split('_', 1)[1]
    selective = rule == 'selective'
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
                normalization_ratio_divisions_upper=matched_steps, normalization_scale_multiplies=12*matched_steps,
                logical_latch_bits=0 if rule == 'original' else 1, prototype_onset_unit_steps=T,
                cosine_norms_per_episode=3 if selective else 0,
                cosine_denominator_multiplies_per_episode=2 if selective else 0,
                cosine_divisions_upper_per_episode=2 if selective else 0,
                reset_scalar_writes_mean=26*reset_count/examples)


def compare_raw(audit, saved, expected, context):
    audit.check(set(saved) == set(expected), f'{context}: raw fields')
    discrete = {'counts', 'spikes', 'resources', 'read_onset', 'reset_steps'}
    for key in sorted(discrete):
        audit.exact(saved[key], expected[key], f'{context}: exact {key}')
    for key in sorted(set(expected)-discrete):
        audit.close(saved[key], expected[key], f'{context}: {key}', 5e-12)
    counts = saved['counts']
    audit.check(np.all((counts >= 0) & (counts <= 8)) and np.array_equal(counts, np.rint(counts)),
                f'{context}: native count bounds')


def verify_gate_fixture(audit):
    """Check shifted and repeated reads with an exact threshold case."""
    x = np.zeros((1, T, 6))
    x[0, 1, (2, 5)] = 1.
    x[0, 4, (3, 5)] = 1.
    x[0, 17:19, 3] = 1.
    x[0, 21, 3] = -1.
    weights = np.zeros(180)
    for rule in native.RULES:
        raw = native.replay(x, weights, 'ORACLE', Q, rule)
        audit.exact(np.flatnonzero(raw['read_onset'][0]), (17, 21), f'fixture: shifted and repeated edges {rule}')
        expected = {'original': (), 'always': (17, 21), 'selective': (17,)}[rule]
        audit.exact(np.flatnonzero(raw['reset_steps'][0]), expected, f'fixture: visible trigger {rule}')
    x[0, 17:19, 2] = 1.
    weights[176:] = np.eye(2).ravel()
    raw = native.replay(x, weights, 'LEARN', Q, 'selective')
    audit.check(raw['match_confidence'][0, 17] == native.COSINE_THRESHOLD and raw['reset_steps'][0, 17],
                'fixture: cosine threshold equality triggers')
    raw = native.replay(x, weights, 'MEAN', Q, 'selective')
    audit.check(not raw['reset_steps'].any() and np.all(raw['match_confidence'] == -2.), 'fixture: zero projection')
    x[:, :17] = 0.
    raw = native.replay(x, weights, 'LEARN', Q, 'selective')
    audit.check(not raw['reset_steps'].any() and np.all(raw['match_confidence'] == -2.), 'fixture: empty key bank')


def verify_archive(audit, raw, x, rule, context):
    n = len(x)
    onset, reset, confidence = (raw[name] for name in ('read_onset', 'reset_steps', 'match_confidence'))
    audit.check(onset.shape == reset.shape == confidence.shape == (n, T)
                and onset.dtype == reset.dtype == np.bool_, f'{context}: complete edge and reset archive')
    observed, marker = x[..., 2:4], x[..., 5:6]
    gate = (1-marker)*np.minimum(1., np.linalg.norm(observed, axis=-1, keepdims=True))
    reading = gate[..., 0] > 0.
    prior_reading = np.concatenate((np.zeros_like(reading[:, :1]), reading[:, :-1]), axis=1)
    audit.exact(onset, reading & ~prior_reading, f'{context}: rising observed read edge')
    audit.check(np.all(onset.sum(axis=1) == 1) and onset[:, CUE].all(), f'{context}: one observed scientific cue')
    audit.check(not (reset & ~onset).any(), f'{context}: resets occur only at observed edges')
    if rule == 'original':
        audit.check(not reset.any(), f'{context}: original never resets')
    elif rule == 'always':
        audit.exact(reset, onset, f'{context}: every edge resets')
    else:
        audit.exact(reset, confidence >= native.COSINE_THRESHOLD, f'{context}: 45-degree decision')
    sentinel = ~onset if rule == 'selective' else np.ones_like(onset)
    audit.check(np.all(confidence[sentinel] == -2.), f'{context}: confidence sentinel')
    valid = confidence != -2.
    audit.check(np.all(np.abs(confidence[valid]) <= 1.+1e-13), f'{context}: bounded valid cosine')

    before, after = raw['cue_native_before'], raw['cue_native_after']
    audit.check(before.shape == after.shape == (n, 3, 26), f'{context}: complete cue native states')
    audit.exact(after, np.where(reset[:, CUE:Q, None], 0., before), f'{context}: reset acts on all 26 native states')
    audit.exact(before[:8, :, :20], raw['states'][:, CUE-1:Q-1], f'{context}: pre-reset native trajectory')
    audit.exact(before[:8, :, 20:], raw['spikes'][:, CUE-1:Q-1, :6], f'{context}: pre-reset previous pulse')

    keys, values = raw['key_bank'], raw['value_bank']
    audit.check(keys.shape == (8, T, 2, 2) and values.shape == (8, T, 12, 2), f'{context}: complete FIFO archive')
    audit.exact(raw['write_gate'], marker[:8], f'{context}: visible write marker')
    audit.close(raw['read_gate'], gate[:8], f'{context}: visible read gate', 1e-12)
    previous_keys = np.concatenate((np.zeros_like(keys[:, :1]), keys[:, :-1]), axis=1)
    previous_values = np.concatenate((np.zeros_like(values[:, :1]), values[:, :-1]), axis=1)
    key_shift = np.stack((observed[:8], previous_keys[:, :, 0]), axis=2)
    retained = .97*previous_values
    value_shift = np.stack((raw['drive'], retained[..., 0]), axis=-1)
    writing = marker[:8, ..., None] == 1.
    audit.exact(keys, np.where(writing, key_shift, previous_keys), f'{context}: unchanged FIFO keys')
    audit.close(values, np.where(writing, value_shift, retained), f'{context}: unchanged FIFO values', 1e-12)

    scores, weights, injected = (raw[name] for name in ('cue_scores', 'cue_weights', 'cue_read_vectors'))
    audit.check(scores.shape == weights.shape == (n, 3, 2) and injected.shape == (n, 3, 12),
                f'{context}: complete cue reads')
    exp_scores = np.exp(scores-scores.max(axis=-1, keepdims=True))
    audit.close(weights, exp_scores/exp_scores.sum(axis=-1, keepdims=True), f'{context}: recorded cue softmax', 1e-12)
    from_bank = 2*gate[:8, CUE:Q]*np.einsum('ntk,ntdk->ntd', weights[:8], previous_values[:, CUE:Q], optimize=False)
    audit.close(injected[:8], from_bank, f'{context}: archived-bank cue reads', 1e-12)
    for saved_name, trace_name in (('cue_scores', 'scores'), ('cue_weights', 'weights'), ('cue_read_vectors', 'injections')):
        audit.exact(raw[saved_name][:8], raw[trace_name][:, CUE:Q], f'{context}: cue trace {trace_name}')
    audit.exact(raw['counts'][:8], raw['spikes'][:, Q:, 6:].sum(axis=1), f'{context}: native query count archive')


def verify_paths(audit, paths, x, weights, arm, prefix, context):
    original = paths['original']
    for rule in ('always', 'selective'):
        raw = paths[rule]
        for key in ('key_bank', 'value_bank', 'scores', 'weights', 'read_gate', 'write_gate', 'injections',
                    'cue_scores', 'cue_weights', 'cue_read_vectors', 'read_onset'):
            audit.exact(raw[key], original[key], f'{context}/{rule}: unchanged {key}')
        for key in ('states', 'spikes', 'candidates', 'currents', 'drive'):
            audit.exact(raw[key][:, :CUE], original[key][:, :CUE], f'{context}/{rule}: unchanged prefix {key}')
        audit.exact(raw['cue_native_before'][:, 0], original['cue_native_before'][:, 0],
                    f'{context}/{rule}: identical native state before first edge')
    earlier = capacity.replay(x, weights, arm, Q, 'base2' if prefix == 'late' else 'full2')
    for field in sorted(set(original) & set(earlier)):
        if field in ('counts', 'spikes', 'resources'):
            audit.exact(original[field], earlier[field], f'{context}: previous native {field}')
        else:
            audit.close(original[field], earlier[field], f'{context}: previous trajectory {field}', 5e-12)
    if arm == 'MEAN':
        for field in sorted(original):
            audit.exact(paths['selective'][field], original[field], f'{context}: zero-matrix equivalence {field}')


def verify_oracle(audit, raw, base, context):
    retained = base['selected'] >= 2
    reset = raw['reset_steps'][:, CUE]
    cosine = raw['match_confidence'][:, CUE]
    audit.exact(reset, retained, f'{context}: fixed-oracle reset exactly retained labels')
    e = .12*np.sqrt(2.)
    recent_bound, older_bound = (1-e)/(1+e), e/(1-e)
    audit.check(cosine[retained].min() >= recent_bound-1e-12, f'{context}: retained key cosine bound')
    audit.check(cosine[~retained].max() <= older_bound+1e-12, f'{context}: missing key cosine bound')
    return float(cosine[retained].min()), float(cosine[~retained].max())


def verify_row(audit, row, raw, base, seed, arm, condition):
    ident = f's{seed}_{arm}'
    context = f'{ident}/{condition}'
    identity = dict(id=ident, seed=seed, arm=arm, condition=condition, T=T, q=Q, slots=2)
    scores = task_scores(raw, base, 2)
    n = len(base['y'])
    selected = base['selected']
    reset = raw['reset_steps'].any(axis=1)
    reset_count = int(np.count_nonzero(raw['reset_steps']))
    scores.update(reset_count=reset_count, onset_count=int(np.count_nonzero(raw['read_onset'])),
                  reset_fraction=float(reset.mean()), older_reset_fraction=float(reset[selected < 2].mean()),
                  recent_reset_fraction=float(reset[selected >= 2].mean()),
                  reset_by_position=[float(reset[selected == position].mean()) for position in range(4)])
    audit.check(set(row) == set(identity) | set(scores) | {'mean_resources', 'cost_per_episode'}, f'{context}: row schema')
    for key, value in identity.items():
        audit.check(row[key] == value, f'{context}: {key}')
    for key, value in scores.items():
        audit.close(row[key], value, f'{context}: native {key}', 1e-13)
    audit.close(row['mean_resources'], raw['resources'][:, :4].mean(axis=0), f'{context}: resource means', 1e-12)
    audit.check(row['cost_per_episode'] == cost(arm, condition, reset_count, n), f'{context}: per-path structural costs')
    return scores


def evaluate(audit, n, weights):
    audit.stage = 'replay'
    rows = read(ROOT/'evaluate'/'ROWS.json')
    index = {(row['id'], row['condition']): row for row in rows}
    expected = {(f's{s}_{arm}', condition) for s in range(BLOCKS) for arm in ARMS for condition in CONDITIONS}
    audit.check(len(rows) == len(index) == EVALUATIONS and set(index) == expected, 'complete 216 paired evaluation rows')
    files = {f'{ident}_{condition}.npz' for ident, condition in expected} | {f'test_s{s}.npz' for s in range(BLOCKS)}
    audit.check({p.name for p in (ROOT/'evaluate').glob('*.npz')} == files, 'exact evaluation archive set')
    effects = {'recent_gain': [], 'older_change': []}
    input_bytes = raw_bytes = duplicates = reset_count = selective_onsets = 0
    oracle_recent_min, oracle_older_max = np.inf, -np.inf
    for seed in range(BLOCKS):
        base = regenerated_base(audit, ROOT/'evaluate'/f'test_s{seed}.npz', n, 4080020000+100*seed)
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
            scores = {}
            for prefix, x in (('full', full_x), ('late', late_x)):
                paths = {}
                for rule in native.RULES:
                    audit.guard()
                    condition = f'{prefix}_{rule}'
                    context = f'{ident}/{condition}'
                    raw = load(ROOT/'evaluate'/f'{ident}_{condition}.npz')
                    replayed = native.replay(x, weights[ident], arm, q, rule)
                    compare_raw(audit, raw, replayed, context)
                    for field in ('cue_native_before', 'cue_native_after'):
                        audit.exact(raw[field][..., 20:], replayed[field][..., 20:], f'{context}: exact previous pulse {field}')
                    verify_archive(audit, raw, x, rule, context)
                    score = verify_row(audit, index[ident, condition], raw, base, seed, arm, condition)
                    if prefix == 'late':
                        audit.exact(raw['counts'][left], raw['counts'][right], f'{context}: exact native counts on duplicates')
                        audit.check(score['older_ba'] == .5*score['older_coverage'],
                                    f'{context}: missing-record BA equals half coverage')
                        duplicates += len(left)
                    if rule == 'selective':
                        selective_onsets += score['onset_count']
                        if arm == 'ORACLE':
                            min_recent, max_older = verify_oracle(audit, raw, base, context)
                            oracle_recent_min = min(oracle_recent_min, min_recent)
                            oracle_older_max = max(oracle_older_max, max_older)
                    reset_count += score['reset_count']
                    paths[rule] = raw
                    scores[condition] = score
                    raw_bytes += sum(value.nbytes for value in raw.values())
                verify_paths(audit, paths, x, weights[ident], arm, prefix, f'{ident}/{prefix}')
            if arm == 'LEARN':
                before, after = scores['full_original'], scores['full_selective']
                effects['recent_gain'].append(after['recent_ba']-before['recent_ba'])
                effects['older_change'].append(after['older_ba']-before['older_ba'])
        print(f'Independently replayed selective reset block {seed+1}/{BLOCKS}', flush=True)
    audit.check(selective_onsets == 6*BLOCKS*n, 'all selective onset opportunities')
    audit.check(duplicates == 9*BLOCKS*n//4, 'all opposite-label pair checks')
    ledgers = dict(input_array_bytes=input_bytes, retained_raw_array_bytes=raw_bytes, reset_count=reset_count,
                   selective_onsets=selective_onsets, reset_scalar_writes=26*reset_count)
    diagnostics = dict(duplicate_opposite_label_pairs=duplicates, oracle_recent_cosine_min=float(oracle_recent_min),
                       oracle_older_cosine_max=float(oracle_older_max))
    return effects, ledgers, diagnostics


def main(audit):
    audit.check(all(os.environ.get(key) == '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')),
                'single-thread independent audit')
    n, weights = provenance(audit)
    verify_gate_fixture(audit)
    effects, ledgers, diagnostics = evaluate(audit, n, weights)
    inference = primary_inference(audit, effects, BLOCKS)
    audit.stage = 'resource ledger'
    science = read(ROOT/'SCIENCE_SUMMARY.json')
    steps = EVALUATIONS*n*T
    expected = dict(blocks=BLOCKS, examples_per_block=n, evaluations=EVALUATIONS, fits=0, updates=0,
                    forward_sample_steps=steps, dense_forward_macs=156*steps, matcher_macs=8*steps, **ledgers)
    audit.check(ledgers['input_array_bytes'] == 2*BLOCKS*n*T*6*8, 'independent input-byte ledger')
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
                  primary_effects=inference, adoption_supported=all(effect['passes'] for effect in inference),
                  scope='Independent source provenance, stimuli and frozen native replays, observed read edges, '
                        'cosine gate and ORACLE bounds, pre-update native state reset and unchanged memory, '
                        'exact deleted-label counts, prior native trajectories, resource ledgers and paired bootstraps.')
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


