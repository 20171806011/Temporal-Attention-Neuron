"""Independent audit of frozen donor-content interventions."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import independent_reset as predecessor
import independent_warm as native
import independent_capacity_data as data
from independent_donor import build as donor_indices


ROOT = Path(__file__).resolve().parent
ARM = 'LEARN'
T, Q, CUE, BLOCKS = 36, 28, 25, 12
SCIENCE_CAP = 180
DONOR_CASES = {'full_own_natural': 'own', 'full_target_flip': 'target', 'full_other_flip': 'other',
               'full_deleted_flip': 'deleted'}
CONDITIONS = ('late_original', 'full_original', 'full_eviction_warm', *DONOR_CASES)
EVALUATIONS = len(CONDITIONS)*BLOCKS
SOURCE_FILES = {'capacity_data.py', 'capacity_metrics.py', 'capacity_model.py', 'independent_capacity_data.py',
                'independent_capacity.py', 'learned_model.py', 'population.py', 'reference.py', 'original_tan.py',
                'reset_model.py', 'independent_reset.py', 'warm_model.py', 'independent_warm.py'}
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
    rng = np.random.default_rng(4082900001)
    draws = rng.integers(0, blocks, (50_000, blocks))
    result = []
    thresholds = {'own_minus_target': 0., 'other_minus_target': 0.}
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
                 'run.py', 'warm_model.py', 'audit.py', 'independent_warm.py', 'independent_capacity_data.py',
                 'donor_map.py', 'independent_donor.py', 'summarize.py', 'report.py'}
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
    audit.check(lock['previous_directory'] == '../DEND-TAN-warm-reference-20261111', 'exact prior source directory')
    previous = ROOT/lock['previous_directory']
    audit.check(digest(previous/'MANIFEST.json') == lock['previous_manifest_sha256'], 'prior manifest seal')
    manifest = read(previous/'MANIFEST.json')['files']
    for name in ('AUDIT.json', 'SCIENCE_SUMMARY.json', 'NEXT_PLAN.md'):
        audit.check(digest(previous/name) == manifest[name]['sha256'], f'sealed prior {name}')
    earlier_audit = read(previous/'AUDIT.json')
    earlier_science = read(previous/'SCIENCE_SUMMARY.json')
    audit.check(earlier_audit['status'] == 'PASS' and earlier_science['status'] == 'COMPLETE'
                and earlier_science['blocks'] == BLOCKS and earlier_science['evaluations'] == 204
                and earlier_science['fits'] == 0, 'source is the complete audited 12-block warm-reference study')
    expected_ids = {f's{s}_{ARM}' for s in range(BLOCKS)}
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


def compare_raw(audit, saved, expected, context):
    audit.check(set(saved) == set(expected), f'{context}: raw fields')
    discrete = {'counts', 'resources', 'spikes', 'read_onset', 'reset_steps', 'eviction_seen', 'key_bank', 'write_gate'}
    for key in saved:
        audit.check(saved[key].shape == expected[key].shape and saved[key].dtype == expected[key].dtype,
                    f'{context}: {key} shape and dtype')
        if key in discrete:
            audit.exact(saved[key], expected[key], f'{context}: exact {key}')
        else:
            audit.close(saved[key], expected[key], f'{context}: {key}', 5e-12)
    for key in ('cue_native_before', 'cue_native_after'):
        audit.exact(saved[key][..., 20:], expected[key][..., 20:], f'{context}: exact prior pulse {key}')
    counts = saved['counts']
    audit.check(np.all((0 <= counts) & (counts <= 8)) and np.array_equal(counts, np.rint(counts)),
                f'{context}: native count bounds')


def expected_gates(x):
    observed, marker = x[..., 2:4], x[..., 5:6]
    gate = (1.-marker)*np.minimum(1., np.linalg.norm(observed, axis=-1, keepdims=True))
    reading = gate[..., 0] > 0.
    onset = reading & ~np.column_stack((np.zeros(len(x), dtype=bool), reading[:, :-1]))
    return gate, onset


def verify_archive(audit, raw, x, rule, eviction, reference, context):
    n = len(x)
    observed, marker = x[..., 2:4], x[..., 5:6]
    gate, expected_onset = expected_gates(x)
    onset, reset, confidence = (raw[key] for key in ('read_onset', 'reset_steps', 'match_confidence'))
    audit.exact(onset, expected_onset, f'{context}: rising observed read edge')
    audit.check(np.all(onset.sum(axis=1) == 1) and onset[:, CUE].all(), f'{context}: one scientific read')
    audit.exact(raw['eviction_seen'], eviction, f'{context}: exact write-history latch')
    if rule == 'original':
        eligible = np.zeros_like(onset)
    elif rule == 'match_zero':
        eligible = onset
    else:
        eligible = onset & eviction
    audit.check(np.all(confidence[~eligible] == native.NO_CONFIDENCE), f'{context}: confidence eligibility scope')
    valid = confidence != native.NO_CONFIDENCE
    audit.check(np.all(np.abs(confidence[valid]) <= 1.+1e-13), f'{context}: bounded valid cosine')
    audit.exact(reset, eligible & (confidence >= native.COSINE_THRESHOLD), f'{context}: exact applied trigger')

    before, after = raw['cue_native_before'], raw['cue_native_after']
    audit.check(before.shape == after.shape == (n, 3, 26), f'{context}: complete cue native states')
    if reference is None:
        reference = np.zeros((n, 26))
    reference = np.broadcast_to(reference, (n, 26))
    audit.exact(after, np.where(reset[:, CUE:Q, None], reference[:, None], before), f'{context}: exact replacement scope')
    audit.exact(before[:8, :, :20], raw['states'][:, CUE-1:Q-1], f'{context}: pre-replacement native trajectory')
    audit.exact(before[:8, :, 20:], raw['spikes'][:, CUE-1:Q-1, :6], f'{context}: pre-replacement previous pulse')

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


def cost(arm, condition, reset_count, eligible_count, n):
    _, rule = condition.split('_', 1)
    writes = 2 if condition.startswith('late_') else 4
    return dict(parameters=180 if arm == 'LEARN' else 176, allocated_parameter_slots=180,
                fixed_match_constants=0 if arm == 'LEARN' else 4, spiking_cells=8, graded_branches=12,
                continuous_state_slots=48, key_slots=4, value_slots=24, persistent_array_bytes=448,
                dense_forward_macs=156*T, graded_transfers=12*T, matcher_dense_macs=8*T,
                matcher_logit_scale_multiplies=2*T, softmax_exponentials=2*T, softmax_divisions=2*T,
                query_norm_square_multiplies=2*T, query_norm_adds=T, query_norm_sqrts=T,
                value_tanh_evaluations=12*T, value_decay_multiplies=24*T, value_read_multiplies=24*T,
                value_read_adds=12*T, injection_multiplies=12*T, fifo_scalar_writes=28*writes,
                hard_equalities=0, hard_tie_adds=0, hard_divisions=0, extra_value_read_multiplies=0,
                extra_value_read_adds=0, extra_injection_multiplies=0, normalization_norms=0,
                normalization_square_multiplies=0, normalization_sum_adds=0, normalization_sqrts=0,
                normalization_ratio_divisions_upper=0, normalization_scale_multiplies=0,
                logical_latch_bits=int(rule != 'original'), prototype_onset_unit_steps=T,
                cosine_norms_per_episode=3*eligible_count/n,
                cosine_denominator_multiplies_per_episode=2*eligible_count/n,
                cosine_divisions_upper_per_episode=2*eligible_count/n,
                reset_scalar_writes_mean=26*reset_count/n,
                eviction_latch_bits=int(rule.startswith('eviction_')), prototype_eviction_unit_steps=T,
                occupancy_checks_per_episode=writes, stored_reference_scalars=26 if rule == 'eviction_warm' else 0)


def check_warm_reference(audit, path, weights, arm, context):
    saved = np.load(path, allow_pickle=False)
    expected = native.warm_reference(weights)
    audit.check(saved.dtype == np.float64 and saved.shape == (26,), f'{context}: saved reference dtype and width')
    audit.close(saved, expected, f'{context}: weight-only warmed reference', 5e-12)
    earlier = predecessor.replay(np.zeros((1, native.WARM_STEPS+1, 6)), weights, arm, native.WARM_STEPS, 'original')
    warm_step = native.WARM_STEPS-1
    before_input = np.concatenate((earlier['states'][0, warm_step], earlier['spikes'][0, warm_step, :6]))
    audit.close(saved, before_input, f'{context}: predecessor zero-input warm replay', 5e-12)
    audit.exact(saved[20:], before_input[20:], f'{context}: exact warmed pulse state')
    return saved, expected


def verify_donors(audit, base, late, donors, context):
    """Check donor assignments and exact external-channel scope independently."""
    n = len(base['y'])
    row = np.arange(n, dtype=np.int64)
    position = base['selected']
    recent = position >= 2
    write_steps = np.asarray(data.WRITE_STEPS)
    for name, indices in donors.items():
        audit.check(indices.dtype == np.int64 and indices.shape == (n,), f'{context}/{name}: index dtype and shape')
        audit.exact(np.sort(indices), row, f'{context}/{name}: complete donor permutation')
        audit.exact(indices[indices], row, f'{context}/{name}: paired inverse')
        for field in ('group', 'selected'):
            audit.exact(base[field][indices], base[field], f'{context}/{name}: preserved {field}')
        flipping = np.zeros((n, 4), dtype=bool)
        if name == 'target':
            flipping[row, position] = True
        elif name == 'other':
            flipping[row[recent], 5-position[recent]] = True
        elif name == 'deleted':
            flipping[:, 0] = True
        else:
            audit.exact(indices, row, f'{context}: own identity')
        audit.exact(base['bits'][indices] != base['bits'], flipping, f'{context}/{name}: exact flipped bit')
        audit.exact(base['y'][indices], base['y'] ^ flipping[row, position], f'{context}/{name}: donor label relation')
        retained_flip = np.zeros((n, T), dtype=bool)
        retained_flip[:, write_steps[2:]] = flipping[:, 2:]
        external = late[indices]
        audit.exact(external[..., 2:], late[..., 2:], f'{context}/{name}: exact key, query, noise and write channels')
        expected_payload = np.where(retained_flip[..., None], late[..., 1::-1], late[..., :2])
        audit.exact(external[..., :2], expected_payload, f'{context}/{name}: exact retained payload swap only')
        audit.exact(np.any(external != late, axis=-1), retained_flip, f'{context}/{name}: exact change ticks')


def exact_subset(audit, actual, expected, mask, names, context):
    for key in names:
        field = actual[key]
        audit.exact(field[mask[:len(field)]], expected[key][mask[:len(field)]], f'{context}: {key}')


def verify_donor_paths(audit, paths, donors, base, context):
    original = paths['full_original']
    warm = paths['full_eviction_warm']
    own = paths['full_own_natural']
    boundary = paths['late_original']['cue_native_before'][:, 0]
    early = base['selected'] < 2
    audit.exact(boundary[donors['deleted']], boundary, f'{context}: removed bit does not alter native boundary')
    audit.exact(boundary[donors['target']][early], boundary[early], f'{context}: removed target does not alter early boundary')
    audit.exact(boundary[donors['other']][early], boundary[early], f'{context}: early other is own boundary')
    for condition in ('full_eviction_warm', *DONOR_CASES):
        raw = paths[condition]
        for key in ('key_bank', 'value_bank', 'scores', 'weights', 'read_gate', 'write_gate', 'injections',
                    'cue_scores', 'cue_weights', 'cue_read_vectors', 'read_onset', 'eviction_seen'):
            audit.exact(raw[key], original[key], f'{context}/{condition}: unchanged recipient {key}')
        for key in ('states', 'spikes', 'candidates', 'currents', 'drive'):
            audit.exact(raw[key][:, :CUE], original[key][:, :CUE], f'{context}/{condition}: unchanged recipient prefix {key}')
        audit.exact(raw['cue_native_before'][:, 0], original['cue_native_before'][:, 0],
                    f'{context}/{condition}: identical recipient boundary before intervention')
        for field in ('reset_steps', 'match_confidence'):
            audit.exact(raw[field], warm[field], f'{context}/{condition}: common trigger {field}')
        untriggered = ~raw['reset_steps'].any(axis=1)
        trajectory = set(raw)-{'match_confidence'}
        exact_subset(audit, raw, original, untriggered, sorted(trajectory), f'{context}/{condition}: untriggered parity')
    for field in own:
        audit.exact(paths['full_deleted_flip'][field], own[field], f'{context}: removed donor bit exact {field}')
    for condition in ('full_target_flip', 'full_other_flip'):
        exact_subset(audit, paths[condition], own, early, sorted(own), f'{context}/{condition}: early exact own parity')


def verify_row(audit, row, raw, base, seed, condition):
    ident = f's{seed}_{ARM}'
    context = f'{ident}/{condition}'
    identity = dict(id=ident, seed=seed, arm=ARM, condition=condition, T=T, q=Q, slots=2)
    scores = task_scores(raw, base, 2)
    selected, y = base['selected'], base['y']
    recent = selected >= 2
    counts = raw['counts']
    covered = counts[:, 0] != counts[:, 1]
    margin = (2*y.astype(np.int64)-1)*(counts[:, 1]-counts[:, 0])
    reset = raw['reset_steps'].any(axis=1)
    reset_count = int(np.count_nonzero(raw['reset_steps']))
    onset_count = int(np.count_nonzero(raw['read_onset']))
    cost_condition = 'full_eviction_late_native' if condition in DONOR_CASES else condition
    eligible_count = int(np.count_nonzero(raw['read_onset'] & raw['eviction_seen'])) if 'eviction_' in cost_condition else 0
    scores.update(reset_count=reset_count, onset_count=onset_count,
                  reset_fraction=float(reset.mean()), older_reset_fraction=float(reset[~recent].mean()),
                  recent_reset_fraction=float(reset[recent].mean()),
                  reset_by_position=[float(reset[selected == position].mean()) for position in range(4)],
                  eviction_fraction=float(raw['eviction_seen'].any(axis=1).mean()),
                  margin=float(margin.mean()), older_margin=float(margin[~recent].mean()),
                  recent_margin=float(margin[recent].mean()),
                  margin_by_position=[float(margin[selected == position].mean()) for position in range(4)],
                  recent_margin_by_label=[float(margin[recent & (y == label)].mean()) for label in (0, 1)],
                  recent_coverage_by_label=[float(covered[recent & (y == label)].mean()) for label in (0, 1)])
    audit.check(set(row) == set(identity) | set(scores) | {'mean_resources', 'cost_per_episode'}, f'{context}: row schema')
    for key, value in identity.items():
        audit.check(row[key] == value, f'{context}: {key}')
    for key, value in scores.items():
        audit.close(row[key], value, f'{context}: native {key}', 1e-13)
    audit.close(row['mean_resources'], raw['resources'][:, :4].mean(axis=0), f'{context}: resource means', 1e-12)
    audit.check(row['cost_per_episode'] == cost(ARM, cost_condition, reset_count, eligible_count, len(selected)),
                f'{context}: per-path structural costs')
    return scores, eligible_count, margin


def evaluate(audit, n, weights):
    audit.stage = 'replay'
    rows = read(ROOT/'evaluate'/'ROWS.json')
    index = {(row['id'], row['condition']): row for row in rows}
    expected = {(f's{s}_{ARM}', condition) for s in range(BLOCKS) for condition in CONDITIONS}
    audit.check(len(rows) == len(index) == EVALUATIONS and set(index) == expected, 'complete 84 paired evaluation rows')
    files = {f'{ident}_{condition}.npz' for ident, condition in expected}
    files |= {f'{archive}_s{s}.npz' for s in range(BLOCKS) for archive in ('test', 'donors')}
    audit.check({p.name for p in (ROOT/'evaluate').glob('*.npz')} == files, 'exact evaluation archive set')
    audit.check({p.name for p in (ROOT/'warm_reference').glob('*.npy')} == {f'{ident}.npy' for ident in weights},
                'exact warmed reference archive set')
    effects = {'own_minus_target': [], 'other_minus_target': []}
    input_bytes = raw_bytes = donor_bytes = reset_count = cosine_onsets = reference_bytes = duplicates = donor_episodes = 0
    recent_triggered = recent_target_changes = recent_other_changes = 0
    for seed in range(BLOCKS):
        ident = f's{seed}_{ARM}'
        base = regenerated_base(audit, ROOT/'evaluate'/f'test_s{seed}.npz', n, 4082020000+100*seed)
        full, q = data.assemble(base, 'full4')
        late, late_q = data.assemble(base, 'base2')
        audit.check(q == late_q == Q and data.check(base, full)['ceiling'] == 1., f'{ident}: full task')
        audit.check(data.check(base, late, 'base2')['ceiling'] == .75, f'{ident}: deletion ceiling')
        kept = [tick for tick in range(T) if tick not in (0, 2)]
        audit.exact(full[:, kept], late[:, kept], f'{ident}: only early writes removed')
        audit.check(not late[:, (0, 2)].any(), f'{ident}: no residual early write channels')
        input_bytes += full.nbytes+late.nbytes
        distinct_late = len({episode.tobytes() for episode in late})
        audit.check(distinct_late == n//4, f'{ident}: exact donor forward multiplicity')
        donor_episodes += distinct_late

        donors = load(ROOT/'evaluate'/f'donors_s{seed}.npz')
        expected_donors = donor_indices(base)
        audit.check(set(donors) == set(expected_donors), f'{ident}: donor index fields')
        for name in expected_donors:
            audit.exact(donors[name], expected_donors[name], f'{ident}: independent {name} donors')
        verify_donors(audit, base, late, donors, ident)
        donor_bytes += sum(array.nbytes for array in donors.values())

        w = weights[ident]
        saved_warm, replayed_warm = check_warm_reference(audit, ROOT/'warm_reference'/f'{ident}.npy', w, ARM, ident)
        reference_bytes += saved_warm.nbytes
        paths, margins = {}, {}
        recent = base['selected'] >= 2
        for condition in CONDITIONS:
            audit.guard()
            is_full = condition != 'late_original'
            x = full if is_full else late
            context = f'{ident}/{condition}'
            raw = load(ROOT/'evaluate'/f'{ident}_{condition}.npz')
            saved_ref = replayed_ref = None
            if condition in DONOR_CASES:
                rule = 'eviction_late_native'
                donor_name = DONOR_CASES[condition]
                saved_ref = saved_boundary[donors[donor_name]]
                replayed_ref = replayed_boundary[expected_donors[donor_name]]
            elif condition == 'full_eviction_warm':
                rule = 'eviction_warm'
                saved_ref, replayed_ref = saved_warm, replayed_warm
            else:
                rule = 'original'
            replayed = native.replay(x, w, ARM, q, rule, replayed_ref)
            compare_raw(audit, raw, replayed, context)
            eviction = np.broadcast_to((np.arange(T) > 4) & is_full, (n, T))
            verify_archive(audit, raw, x, rule, eviction, saved_ref, context)
            scores, opportunities, margin = verify_row(audit, index[ident, condition], raw, base, seed, condition)
            if condition == 'late_original':
                saved_boundary = raw['cue_native_before'][:, 0]
                replayed_boundary = replayed['cue_native_before'][:, 0]
                left, right = data.pairs(base).T
                audit.exact(base['y'][left], 1-base['y'][right], f'{context}: opposite labels')
                audit.exact(late[left], late[right], f'{context}: same deleted-label stimuli')
                audit.exact(saved_boundary[left], saved_boundary[right], f'{context}: same deleted-label boundary')
                audit.exact(replayed_boundary[left], replayed_boundary[right], f'{context}: independently same boundary')
                audit.exact(raw['counts'][left], raw['counts'][right], f'{context}: same deleted-label counts')
                audit.check(scores['older_ba'] == .5*scores['older_coverage'], f'{context}: missing BA equals half coverage')
                duplicates += len(left)
            reset_count += scores['reset_count']
            cosine_onsets += opportunities
            paths[condition], margins[condition] = raw, margin
            raw_bytes += sum(value.nbytes for value in raw.values())
        verify_donor_paths(audit, paths, donors, base, ident)
        triggered = paths['full_own_natural']['reset_steps'].any(axis=1)
        recent_triggered += int(np.count_nonzero(recent & triggered))
        recent_target_changes += int(np.count_nonzero(recent & np.any(saved_boundary[donors['target']] != saved_boundary, axis=1)))
        recent_other_changes += int(np.count_nonzero(recent & np.any(saved_boundary[donors['other']] != saved_boundary, axis=1)))
        target = margins['full_target_flip'][recent]
        effects['own_minus_target'].append(float((margins['full_own_natural'][recent]-target).mean()))
        effects['other_minus_target'].append(float((margins['full_other_flip'][recent]-target).mean()))
        print(f'Independently replayed donor-content block {seed+1}/{BLOCKS}', flush=True)

    audit.check(cosine_onsets == 5*BLOCKS*n, 'all cosine onset opportunities')
    audit.check(duplicates == BLOCKS*n//4, 'all opposite-label pair checks')
    audit.check(reference_bytes == BLOCKS*26*8 and donor_bytes == BLOCKS*4*n*8, 'independent reference and index bytes')
    ledgers = dict(input_array_bytes=input_bytes, retained_raw_array_bytes=raw_bytes, reset_count=reset_count,
                   cosine_onsets=cosine_onsets, reset_scalar_writes=26*reset_count, warm_reference_bytes=reference_bytes,
                   donor_index_bytes=donor_bytes, donor_forward_unique_episodes=donor_episodes)
    diagnostics = dict(duplicate_opposite_label_pairs=duplicates, recent_query_samples=BLOCKS*n//2,
                       recent_triggered_samples=recent_triggered, recent_target_boundary_changes=recent_target_changes,
                       recent_other_boundary_changes=recent_other_changes)
    return effects, ledgers, diagnostics


def main(audit):
    thread_keys = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
    audit.check(all(os.environ.get(key) == '1' for key in thread_keys), 'single-thread independent audit')
    n, weights = provenance(audit)
    effects, ledgers, diagnostics = evaluate(audit, n, weights)
    inference = primary_inference(audit, effects, BLOCKS)
    audit.stage = 'resource ledger'
    science = read(ROOT/'SCIENCE_SUMMARY.json')
    donor_steps = ledgers['donor_forward_unique_episodes']*T
    steps = (EVALUATIONS-BLOCKS)*n*T+donor_steps
    warm_count, warm_steps = len(weights), len(weights)*native.WARM_STEPS
    expected = dict(blocks=BLOCKS, examples_per_block=n, evaluations=EVALUATIONS, fits=0, updates=0,
                    forward_sample_steps=steps, dense_forward_macs=156*steps, matcher_macs=8*steps,
                    warm_reference_count=warm_count, warm_reference_steps=warm_steps,
                    warm_reference_native_macs=156*warm_steps, warm_reference_matcher_macs=8*warm_steps,
                    donor_forward_evaluations=BLOCKS, donor_forward_sample_steps=donor_steps,
                    mechanism_diagnostic_evaluations=4*BLOCKS, mechanism_diagnostic_sample_steps=4*BLOCKS*n*T,
                    **ledgers)
    audit.check(ledgers['input_array_bytes'] == 2*BLOCKS*n*T*6*8, 'independent input-byte ledger')
    audit.check(science['status'] == 'COMPLETE' and science['cap_seconds'] == SCIENCE_CAP
                and 0 < science['wall_seconds'] <= SCIENCE_CAP and science['cpu_seconds'] > 0, 'science wall and CPU time')
    audit.check(0 < science['peak_working_set_bytes'] <= MEMORY_CAP, 'science memory cap')
    audit.check(0 < science['artifact_bytes_at_science_end'] <= ARTIFACT_CAP, 'science artifact cap')
    audit.check(sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) < ARTIFACT_CAP, 'current artifact cap')
    completion = read(ROOT/'RUN_STATE.json')
    audit.check(completion['status'] == 'complete' and completion['evaluations'] == EVALUATIONS, 'science completion state')
    audit.check(read(ROOT/'ENVIRONMENT.json')['threads'] == {key: '1' for key in thread_keys}, 'science thread scope')
    for key, value in expected.items():
        audit.check(science[key] == value, f'science ledger {key}')
    audit.guard()
    result = dict(status='PASS', checks=audit.checks, **expected, max_numeric_error=audit.max_error,
                  cap_seconds=300, peak_working_set_bytes=peak_memory(), **audit.elapsed(), **diagnostics,
                  primary_effects=inference, mechanism_supported=all(effect['passes'] for effect in inference),
                  scope='Independent source provenance, fresh factorial stimuli, donor mappings and boundaries, '
                        'recipient native replay, common triggering and FIFO reads, exact deletion, early and '
                        'untriggered controls, cost ledger and two all-recent paired bootstrap effects.')
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


