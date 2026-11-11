"""Independent scientific audit of the continuous-key FIFO matcher."""

import ctypes
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import independent_learned as native
import independent_learned_data as data


ROOT = Path(__file__).resolve().parent
ARMS = ('MEAN', 'ORACLE', 'LEARN')
SWAP_ARMS = ('LEARN', 'ORACLE')
T, Q, CUE = 36, 28, 25
SOURCE_FILES = {'model.py', 'population.py', 'reference.py', 'original_tan.py', 'metrics.py'}
MEMORY_CAP = 1024**3
ARTIFACT_CAP = 3*1024**3//2
FIT_META = ('id', 'seed', 'arm', 'train_delay', 'h', 'lr', 'loss', 'updates')


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


def initialize(seed):
    rng = np.random.default_rng(seed)
    input_weights = rng.normal(0., .45, (6, 12))
    recurrent_weights = rng.normal(0., .25, (6, 12))
    output_weights = rng.normal(0., .5, (6, 2))
    return np.r_[input_weights.ravel(), recurrent_weights.ravel(), np.full(12, .15),
                 np.full(6, .05), output_weights.ravel(), np.full(2, .1), np.zeros(4)]


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
    rows = {key: i for i, key in enumerate(zip(base['octet'], base['selected'], base['first_bit'], base['second_bit']))}
    both = [good[i] and good[rows[octet, 1, first, second_bit]]
            for (octet, selection, first, second_bit), i in rows.items() if selection == 0 and first != second_bit]
    result['disagree_joint_correct'] = float(np.mean(both))
    return result


def compare_raw(audit, saved, expected, context):
    audit.check(set(saved) == set(expected), f'{context}: raw fields')
    for key in ('counts', 'spikes', 'resources'):
        audit.exact(saved[key], expected[key], f'{context}: exact {key}')
    for key in set(expected)-{'counts', 'spikes', 'resources'}:
        audit.close(saved[key], expected[key], f'{context}: {key}')
    audit.check(np.all((saved['counts'] >= 0) & (saved['counts'] <= 8))
                and np.array_equal(saved['counts'], np.rint(saved['counts'])), f'{context}: native count bounds')


def primary_inference(audit, effects, blocks):
    audit.stage = 'primary inference'
    rng = np.random.default_rng(4077900001)
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


def verify_hashes(audit, entries, folder, context):
    for name, sha in entries.items():
        audit.check(digest(folder/name) == sha, f'{context}: {name}')


def provenance(audit):
    freeze, timing = (read(ROOT/name) for name in ('FREEZE.json', 'TIMING.json'))
    precheck = read(ROOT/'PRECHECK.json')
    blocks, updates = freeze['blocks'], freeze['updates_per_fit']
    audit.check(freeze['scientific_data_generated'] is False and updates == timing['updates'] == 600,
                'fixed training scope before scientific data')
    audit.check(timing['status'] == precheck['status'] == 'PASS', 'fixture and timing passes')
    audit.check(timing.get('scientific_seeds_used', False) is False and precheck.get('scientific_seeds_used', False) is False,
                'non-scientific fixtures')
    rows = timing['rows']
    audit.check(len(rows) == 3 and {row['arm'] for row in rows} == set(ARMS), 'three timed training rules')
    medians = []
    for row in rows:
        samples = np.asarray(row['seconds'])
        audit.check(samples.shape == (12,) and np.isfinite(samples).all() and np.min(samples) > 0,
                    f'{row["arm"]}: twelve positive finite timing samples')
        median = float(np.median(samples[6:]))
        audit.close(row['last_six_median_seconds'], median, f'{row["arm"]}: last six fixture median', 1e-12)
        medians.append(median)
    science_forecasts, audit_forecasts = (timing[key] for key in (
        'candidate_forecast_seconds', 'candidate_audit_forecast_seconds'))
    audit.check(set(science_forecasts) == set(audit_forecasts) == {'12', '8'}, 'two candidate block sizes')
    components = timing['audit_components']
    names = ('ledger_seconds', 'evaluation_seconds', 'training_replay_seconds')
    for key in names:
        audit.check(np.isfinite(components[key]) and components[key] > 0, f'positive audit timing {key}')
    unit = 50*components['ledger_seconds']+sum(components[key] for key in names[1:])
    for b in (12, 8):
        audit.close(science_forecasts[str(b)], 2*b*updates*sum(medians)+100, f'{b}: science forecast', 1e-10)
        audit.close(audit_forecasts[str(b)], 3*b*unit+45, f'{b}: audit forecast', 1e-10)
    chosen = next((b for b in (12, 8) if science_forecasts[str(b)] < 840 and audit_forecasts[str(b)] < 300), None)
    audit.check(chosen == blocks == timing['blocks'] and blocks in (12, 8), 'first candidate within both caps')
    audit.check(timing['forecast_seconds'] == science_forecasts[str(blocks)]
                and timing['audit_forecast_seconds'] == audit_forecasts[str(blocks)], 'frozen selected forecasts')

    required = SOURCE_FILES | {'PROTOCOL.md', 'audit.py', 'independent_learned_data.py', 'independent_learned.py', 'learned_data.py', 'learned_model.py', 'run.py',
                               'TIMING.json', 'PRECHECK.json', 'SOURCE_LOCK.json'}
    audit.check(required <= set(freeze['files']), 'all source and protocol files frozen')
    audit.check(all(not name.startswith(('train/', 'evaluate/')) for name in freeze['files']), 'pre-data freeze scope')
    verify_hashes(audit, freeze['files'], ROOT, 'frozen file')
    source = read(ROOT/'SOURCE_LOCK.json')
    audit.check(source['previous_directory'] == '../DEND-TAN-key-binding-20261110', 'sealed key-binding source')
    previous = ROOT/source['previous_directory']
    audit.check(digest(previous/'MANIFEST.json') == source['previous_manifest_sha256'], 'previous manifest seal')
    manifest = read(previous/'MANIFEST.json')['files']
    audit.check(set(source['files']) == SOURCE_FILES, 'exact source lock files')
    for name, sha in source['files'].items():
        audit.check(manifest[name]['sha256'] == digest(previous/name) == freeze['files'][name] == sha,
                    f'unchanged native source: {name}')
    audit.check(digest(previous/'AUDIT.json') == manifest['AUDIT.json']['sha256']
                and read(previous/'AUDIT.json')['status'] == 'PASS', 'source batch independent audit seal')

    lock = read(ROOT/'train'/'TRAIN_LOCK.json')
    audit.check(lock['test_generated'] is False, 'test set generated after training lock')
    required = {f'base_s{s}.npz' for s in range(blocks)} | {'ROWS.json'}
    required |= {f's{s}_{arm}.npz' for s in range(blocks) for arm in ARMS}
    audit.check(set(lock['files']) == required, 'all training artifacts locked')
    verify_hashes(audit, lock['files'], ROOT/'train', 'locked training artifact')
    frozen_time, locked_time = (datetime.fromisoformat(item['utc']) for item in (freeze, lock))
    audit.check(frozen_time < locked_time, 'freeze predates locked training')
    for seed in range(blocks):
        audit.check((ROOT/'evaluate'/f'test_s{seed}.npz').stat().st_mtime >= locked_time.timestamp(),
                    f'test block {seed} timestamp no earlier than training lock')
    keys = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
    env = read(ROOT/'ENVIRONMENT.json')
    audit.check(all(env['threads'][key] == '1' and os.environ.get(key) == '1' for key in keys),
                'single-thread science and independent audit')
    audit.guard()
    return blocks, updates


def audit_fit(audit, row, x, y, w, updates):
    ident, arm = row['id'], row['arm']
    saved = load(ROOT/'train'/f'{ident}.npz')
    audit.check(set(saved) == {'initial', 'theta', 'gradients', 'counts', 'history'}, f'{ident}: training archive schema')
    audit.exact(saved['initial'], w, f'{ident}: independent paired initialization')
    gradients, counts, history = (saved[key] for key in ('gradients', 'counts', 'history'))
    audit.check(gradients.shape == (updates, 180) and counts.shape == (updates, 128, 2)
                and history.shape == (updates, 4), f'{ident}: all training arrays')
    audit.check(np.isfinite(gradients).all() and np.isfinite(history).all(), f'{ident}: finite training')
    audit.check(counts.dtype == np.uint8 and np.max(counts) <= 8, f'{ident}: eight-step training counts')
    if arm != 'LEARN':
        audit.exact(gradients[:, 176:], np.zeros((updates, 4)), f'{ident}: fixed matcher has zero tail gradients')
        audit.exact(saved['theta'][176:], np.zeros(4), f'{ident}: fixed matcher keeps all four padding slots at zero')
    m, v = np.zeros_like(w), np.zeros_like(w)
    for k, (g, native_counts, (loss, norm, task_norm, l2_norm)) in enumerate(zip(gradients, counts, history)):
        context = f'{ident}/{k+1}'
        audit.close(loss, count_objective(native_counts, y, w), f'{context}: native count CE + L2', 1e-10)
        audit.check(min(norm, task_norm, l2_norm) >= 0, f'{context}: nonnegative norms')
        audit.close(np.linalg.norm(g), min(1., norm), f'{context}: clipped norm', 1e-10)
        regularizer = .0002*w/180
        before_clip = g*max(1., norm)
        audit.close(np.linalg.norm(before_clip-regularizer), task_norm, f'{context}: task norm', 1e-10)
        audit.close(np.linalg.norm(regularizer), l2_norm, f'{context}: L2 norm', 1e-10)
        if k in (0, updates-1):
            replayed = native.replay(x, w, arm, Q)
            audit.exact(native_counts, replayed['counts'], f'{context}: independent training forward')
        m = .9*m+.1*g
        v = .999*v+.001*g*g
        corrected_m = m/(1-.9**(k+1))
        corrected_v = v/(1-.999**(k+1))
        w = np.clip(w-.0025*corrected_m/(np.sqrt(corrected_v)+1e-8), -2., 2.)
    audit.close(saved['theta'], w, f'{ident}: every Adam update reconstructed', 1e-10)
    return saved['theta']


def count_objective(counts, y, w):
    margin = .5*(counts[:, 1].astype(float)-counts[:, 0].astype(float))
    return float(np.mean(np.logaddexp(0., (1-2*y)*margin))+.0001*np.mean(w*w))


def training(audit, blocks, updates):
    audit.stage = 'training'
    rows = read(ROOT/'train'/'ROWS.json')
    index = {(row['seed'], row['arm']): row for row in rows}
    expected = {(s, arm) for s in range(blocks) for arm in ARMS}
    audit.check(len(rows) == len(index) == 3*blocks and set(index) == expected, 'complete paired fits')
    weights = {}
    steps_per_fit = updates*128*T
    for seed in range(blocks):
        base = regenerated_base(audit, ROOT/'train'/f'base_s{seed}.npz', 128, 4077010000+100*seed)
        x, q = data.assemble(base)
        audit.check(x.shape == (128, T, 6) and q == Q and data.check(base, x)['ceiling'] == 1., 'training timeline and task')
        w = initialize(4077030000+seed)
        for arm in ARMS:
            audit.guard()
            row = index[seed, arm]
            ident = f's{seed}_{arm}'
            meta = dict(id=ident, seed=seed, arm=arm, train_delay=16, h=6, lr=.0025, loss='CE', updates=updates, T=T, q=Q)
            for key, value in meta.items():
                audit.check(row[key] == value, f'{ident}: fixed training {key}')
            audit.check(0 < row['seconds'] < 840 and row['cost_per_episode'] == cost(arm), f'{ident}: fit elapsed/cost')
            audit.check(row['dense_forward_macs'] == 156*steps_per_fit and row['dense_backward_macs'] == 240*steps_per_fit,
                        f'{ident}: training dense MACs')
            audit.check(row['memory_backward_unit_steps'] == 12*steps_per_fit
                        and row['matcher_backward_unit_steps'] == (2*steps_per_fit if arm == 'LEARN' else 0),
                        f'{ident}: memory and matcher reverse steps')
            weights[ident] = audit_fit(audit, row, x, base['y'], w, updates)
        print(f'Independently audited training block {seed+1}/{blocks}', flush=True)
    stats = dict(optimizer_updates=3*blocks*updates, training_sample_steps=3*blocks*steps_per_fit,
                 first_last_training_replays=6*blocks)
    return rows, weights, stats


def binding_scores(counts, base):
    result = native_scores(counts, base)
    reverse = native_scores(counts[:, ::-1], base)
    result.update(counterfactual_joint=reverse['disagree_joint_correct'], counterfactual_disagree_ba=reverse['disagree_ba'])
    label, opposite = base['y'], base['first_bit'] != base['second_bit']
    predicted = np.where(counts[:, 1] == counts[:, 0], -1, np.argmax(counts, axis=1))
    for order in (0, 1):
        subgroup = opposite & (base['order'] == order)
        recalls = [np.mean(predicted[subgroup & (label == y)] == y) for y in (0, 1)]
        result[f'order{order}_disagree_ba'] = float(np.mean(recalls))
    return result


def verify_deletion(audit, base, x, counts, scores, condition, context):
    pairs = np.asarray(data.pairs(base, condition))
    expected = len(x)//2 if condition == 'no_contents' else len(x)//4
    audit.check(pairs.shape == (expected, 2), f'{context}: full opposite-label pairing')
    left, right = pairs.T
    audit.exact(x[left], x[right], f'{context}: exact duplicate input')
    audit.exact(base['y'][left], 1-base['y'][right], f'{context}: opposite labels')
    audit.exact(counts[left], counts[right], f'{context}: exact count equality on duplicates')
    audit.check(scores['balanced_accuracy'] <= data.CEILINGS[condition], f'{context}: whole-set BA ceiling')
    if condition == 'no_contents':
        audit.check(scores['balanced_accuracy'] == .5*scores['coverage'], f'{context}: whole-set BA/coverage identity')
    else:
        audit.check(scores['disagree_ba'] == .5*scores['disagree_coverage'], f'{context}: opposite BA/coverage identity')
    if condition == 'no_cue':
        audit.check(scores['disagree_joint_correct'] == 0., f'{context}: zero joint correctness without cue')
    return len(pairs)


def cost(arm):
    units = 12*T
    return dict(parameters=180 if arm == 'LEARN' else 176, allocated_parameter_slots=180,
                fixed_match_constants=0 if arm == 'LEARN' else 4, spiking_cells=8, graded_branches=12,
                continuous_state_slots=48, history_slots=0, trace_slots=24, key_slots=4, persistent_array_bytes=448,
                dense_forward_macs=156*T, graded_transfers=units, matcher_dense_macs=8*T,
                matcher_logit_scale_multiplies=2*T, softmax_exponentials=2*T, softmax_divisions=2*T,
                query_norm_square_multiplies=2*T, query_norm_adds=T, query_norm_sqrts=T,
                value_tanh_evaluations=units, value_read_multiplies=2*units, value_read_adds=units,
                value_decay_multiplies=2*units, injection_multiplies=units)


def routing_scores(raw, base):
    scores = binding_scores(raw['counts'], base)
    alpha = raw['cue_weights'][:, 0]
    early = base['selected'] == base['order']
    chosen = np.where(early, alpha[:, 1], alpha[:, 0])
    other = np.where(early, alpha[:, 0], alpha[:, 1])
    return dict(**scores, routing_accuracy=float(np.mean(chosen > other)),
                routing_tie_rate=float(np.mean(chosen == other)), routing_weight_mean=float(np.mean(chosen)))


def verify_memory(audit, raw, x, w, arm, condition, context):
    keys, values, alpha = raw['key_bank'], raw['value_bank'], raw['weights']
    old_keys = np.concatenate((np.zeros_like(keys[:, :1]), keys[:, :-1]), axis=1)
    old_values = np.concatenate((np.zeros_like(values[:, :1]), values[:, :-1]), axis=1)
    observed, marker = x[:8, :, 2:4], x[:8, :, 5:6]
    write = marker[..., 0] == 1.
    gate = (1-marker)*np.minimum(1., np.linalg.norm(observed, axis=-1, keepdims=True))
    audit.exact(raw['write_gate'], marker, f'{context}: observed write marker')
    audit.close(raw['read_gate'], gate, f'{context}: norm-based query gate', 1e-12)
    audit.check(np.abs(values).max() <= 1. and np.abs(raw['drive']).max() <= 1., f'{context}: bounded value slots')
    expected_keys, expected_values = old_keys.copy(), .97*old_values
    expected_keys[write, 0] = observed[write]
    expected_keys[write, 1] = old_keys[write, 0]
    expected_values[write, :, 0] = raw['drive'][write]
    expected_values[write, :, 1] = .97*old_values[write, :, 0]
    audit.exact(keys, expected_keys, f'{context}: FIFO key placement and stable key state')
    audit.close(values, expected_values, f'{context}: FIFO value placement and decay', 1e-12)
    matcher = w[176:].reshape(2, 2) if arm == 'LEARN' else np.array([[0., -1.], [1., 0.]]) if arm == 'ORACLE' else np.zeros((2, 2))
    expected_scores = 4*np.einsum('...a,ab,...kb->...k', observed, matcher, old_keys)
    audit.close(raw['scores'], expected_scores, f'{context}: pre-read oriented bilinear logits', 1e-12)
    log_normalizer = np.logaddexp(expected_scores[..., :1], expected_scores[..., 1:])
    expected_alpha = np.exp(expected_scores-log_normalizer)
    if condition == 'swap_read':
        expected_alpha = expected_alpha[..., ::-1]
    audit.close(alpha, expected_alpha, f'{context}: normalized read weights', 1e-12)
    audit.close(raw['injections'], 2*gate*np.einsum('...j,...dj->...d', alpha, old_values),
                f'{context}: read of old FIFO values', 1e-12)
    audit.exact(raw['injections'][gate[..., 0] == 0.], np.zeros_like(raw['injections'][gate[..., 0] == 0.]),
                f'{context}: no injection at zero read gate')
    cue_weights = raw['cue_weights']
    audit.check(cue_weights.shape == (len(x), 3, 2), f'{context}: all-example cue weights shape')
    audit.close(cue_weights.sum(axis=-1), np.ones((len(x), 3)), f'{context}: all-example cue normalization', 1e-12)
    audit.exact(cue_weights[:8], alpha[:, CUE:Q], f'{context}: full cue weights match archived paths')
    if arm == 'MEAN' or condition in ('no_cue', 'no_tags'):
        audit.exact(cue_weights, np.full_like(cue_weights, .5), f'{context}: required cue routing tie')
    if condition == 'no_tags':
        audit.exact(keys, np.zeros_like(keys), f'{context}: empty keys without observed write tags')
    audit.exact(raw['counts'][:8], raw['spikes'][:, Q:, 6:].sum(axis=1), f'{context}: first eight native query counts')


def verify_oracle(audit, base, raw, condition, context):
    early = base['selected'] == base['order']
    if condition == 'swap_read':
        early = ~early
    if condition in ('nominal', 'no_contents', 'swap_read'):
        attention = raw['cue_weights']
        selected = np.where(early[:, None], attention[..., 1], attention[..., 0])
        audit.check(np.min(selected) > .5, f'{context}: fixed rotation selects the specified slot every cue step')
        first_rows = np.arange(8)
        selected_slot = (base['selected'][:8] == base['order'][:8]).astype(int)
        cue_scores = raw['scores'][:, CUE]
        margin = cue_scores[first_rows, selected_slot]-cue_scores[first_rows, 1-selected_slot]
        expected = data.oracle_margins(base)[base['group'][:8], base['selected'][:8]]
        audit.close(margin, 4*expected, f'{context}: archived logits obey independent analytic rotation margin', 1e-12)


def verify_swap(audit, nominal, swap, context):
    for key in ('states', 'spikes', 'candidates', 'currents', 'injections', 'drive'):
        audit.exact(swap[key][:, :CUE], nominal[key][:, :CUE], f'{context}: same pre-cue {key}')
    for key in ('key_bank', 'value_bank', 'read_gate', 'write_gate', 'scores'):
        audit.exact(swap[key], nominal[key], f'{context}: unchanged {key} over full episode')
    audit.exact(swap['weights'], nominal['weights'][..., ::-1], f'{context}: exact reversed archived weights')
    audit.exact(swap['cue_weights'], nominal['cue_weights'][..., ::-1], f'{context}: exact all-example cue reversal')


def evaluation(audit, training_rows, weights, blocks):
    audit.stage = 'evaluation'
    rows = read(ROOT/'evaluate'/'ROWS.json')
    index = {(row['id'], row['condition']): row for row in rows}
    common = tuple(condition for condition in data.CONDITIONS if condition != 'swap_read')
    conditions = {arm: data.CONDITIONS if arm in SWAP_ARMS else common for arm in ARMS}
    expected = {(f's{s}_{arm}', condition) for s in range(blocks) for arm in ARMS for condition in conditions[arm]}
    audit.check(len(rows) == len(index) == 14*blocks and set(index) == expected, 'all paired evaluations')
    expected_files = {f'{ident}_{condition}.npz' for ident, condition in expected}
    expected_files |= {f'test_s{s}.npz' for s in range(blocks)}
    audit.check({p.name for p in (ROOT/'evaluate').glob('*.npz')} == expected_files, 'exact evaluation archive set')
    effects = {'learn_minus_mean': [], 'learn_swap_counterfactual_joint_gain': []}
    duplicate_pairs = deletions = swap_cases = oracle_cases = 0
    min_oracle_margin = np.inf
    for seed in range(blocks):
        base = regenerated_base(audit, ROOT/'evaluate'/f'test_s{seed}.npz', 256, 4077020000+100*seed)
        inputs = {}
        for condition in common:
            x, q = data.assemble(base, condition=condition)
            information = data.check(base, x, condition=condition)
            audit.check(x.shape == (256, T, 6) and q == Q and information['ceiling'] == data.CEILINGS[condition],
                        f'test {seed}/{condition}: timeline and enumerated input-equivalence ceiling')
            min_oracle_margin = min(min_oracle_margin, information['oracle_margin_min'])
            inputs[condition] = x
        at_nominal = {}
        for fit in (row for row in training_rows if row['seed'] == seed):
            ident, arm = fit['id'], fit['arm']
            for condition in conditions[arm]:
                audit.guard()
                context = f'{ident}_{condition}'
                row = index[ident, condition]
                x = inputs['nominal' if condition == 'swap_read' else condition]
                meta = {key: fit[key] for key in FIT_META}
                meta.update(condition=condition, test_delay=16, T=T, q=Q)
                for key, value in meta.items():
                    audit.check(row[key] == value, f'{context}: evaluation {key}')
                expected_raw = native.replay(x, weights[ident], arm, Q, swap_read=(condition == 'swap_read'))
                raw = load(ROOT/'evaluate'/f'{context}.npz')
                compare_raw(audit, raw, expected_raw, context)
                verify_memory(audit, raw, x, weights[ident], arm, condition, context)
                scores = routing_scores(raw, base)
                for key, value in scores.items():
                    audit.close(row[key], value, f'{context}: independent {key}', 1e-13)
                audit.close(row['mean_resources'], raw['resources'][:, :4].mean(0),
                            f'{context}: independent event and nonzero fanout totals')
                audit.check(row['cost_per_episode'] == cost(arm), f'{context}: online structural costs')
                if arm == 'ORACLE':
                    verify_oracle(audit, base, raw, condition, context)
                    oracle_cases += int(condition in ('nominal', 'no_contents', 'swap_read'))
                if condition in ('no_cue', 'no_tags', 'no_contents'):
                    duplicate_pairs += verify_deletion(audit, base, x, raw['counts'], scores, condition, context)
                    deletions += 1
                if condition == 'nominal':
                    at_nominal[arm] = scores
                    nominal = raw
                if condition == 'swap_read':
                    verify_swap(audit, nominal, raw, context)
                    if arm == 'LEARN':
                        effects['learn_swap_counterfactual_joint_gain'].append(scores['counterfactual_joint']
                                                                              - at_nominal[arm]['counterfactual_joint'])
                    swap_cases += 1
        effects['learn_minus_mean'].append(at_nominal['LEARN']['disagree_ba']-at_nominal['MEAN']['disagree_ba'])
        print(f'Independently replayed continuous-key block {seed+1}/{blocks}', flush=True)
    audit.check((duplicate_pairs, deletions, swap_cases, oracle_cases) == (768*blocks, 9*blocks, 2*blocks, 3*blocks),
                'complete duplicate-input, deletion, swap, and analytic oracle scope')
    stats = dict(evaluations=len(rows), evaluation_sample_steps=len(rows)*256*T,
                 duplicate_opposite_label_pairs=duplicate_pairs, deletion_cases=deletions,
                 swap_cases=swap_cases, oracle_cases=oracle_cases, oracle_margin_min=min_oracle_margin,
                 oracle_margin_bound=float(data.ORACLE_MARGIN_BOUND))
    return effects, stats


def main(audit):
    blocks, updates = provenance(audit)
    rows, weights, training_stats = training(audit, blocks, updates)
    effects, eval_stats = evaluation(audit, rows, weights, blocks)
    inference = primary_inference(audit, effects, blocks)
    audit.stage = 'resource ledger'
    science = read(ROOT/'SCIENCE_SUMMARY.json')
    train_steps, eval_steps = training_stats['training_sample_steps'], eval_stats['evaluation_sample_steps']
    expected = dict(blocks=blocks, fits=3*blocks, updates=3*blocks*updates, updates_per_fit=updates,
                    evaluations=14*blocks, forward_sample_steps=train_steps+eval_steps,
                    dense_forward_macs=156*(train_steps+eval_steps), dense_backward_macs=240*train_steps)
    audit.check(science['status'] == 'COMPLETE' and science['cap_seconds'] == 840
                and 0 < science['wall_seconds'] <= 840 and science['cpu_seconds'] > 0, 'science wall and CPU time')
    audit.check(0 < science['peak_working_set_bytes'] <= MEMORY_CAP, 'science memory cap')
    audit.check(0 < science['artifact_bytes_at_science_end'] <= ARTIFACT_CAP, 'science artifact cap')
    audit.check(sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) < ARTIFACT_CAP, 'current artifact cap')
    audit.check(read(ROOT/'RUN_STATE.json')['status'] == 'complete', 'science completion state')
    for key, value in expected.items():
        audit.check(science[key] == value, f'science ledger {key}')
    audit.guard()
    del eval_stats['evaluations']
    result = dict(status='PASS', checks=audit.checks, **expected, max_numeric_error=audit.max_error,
                  cap_seconds=300, peak_working_set_bytes=peak_memory(), **audit.elapsed(), **training_stats,
                  **eval_stats, primary_effects=inference,
                  scope='Independent stimulus generation, initial weights, all stored objectives and Adam updates, '
                        'first/last native forwards, all evaluation raw replays, resource and FIFO/matcher invariants, '
                        'exact deletion equivalences, swap prefix and bank identity, and paired bootstraps. '
                        'Scientific parameter gradients are not independently regenerated at each update.')
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
