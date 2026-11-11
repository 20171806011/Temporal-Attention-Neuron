"""Audit paired two-trace training, anchored gradients, and common-rule tests."""

import ctypes
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import independent_data as data
import independent_scale
import independent_training as proxy


ROOT = Path(__file__).resolve().parent
ARMS = ('DYN', 'DETACH', 'MEAN')
EVAL_RULES = {'mean': 'SCALE_MEAN', 'dynamic': 'SCALE'}
T, Q = 36, 28
SOURCE_FILES = {'model.py', 'population.py', 'reference.py', 'original_tan.py', 'task_data.py',
                'independent_data.py', 'metrics.py', 'scale_model.py', 'independent_scale.py'}
MEMORY_CAP = 1024**3
ARTIFACT_CAP = 3*1024**3//2
FIT_META = ('id', 'seed', 'arm', 'train_rule', 'detached', 'train_delay', 'h', 'lr', 'loss', 'updates')


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
                 np.full(6, .05), output_weights.ravel(), np.full(2, .1)]


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


def cost():
    trace_slots = 24
    units = T*trace_slots//2
    return dict(parameters=176, spiking_cells=8, graded_branches=12,
                continuous_state_slots=20+trace_slots, history_slots=0, trace_slots=trace_slots,
                persistent_array_bytes=8*(28+trace_slots), dense_forward_macs=156*T,
                graded_transfers=12*T, history_roll_writes=0, attention_unit_steps=0,
                softmax_exponentials=0, trace_tanh_evaluations=units, trace_logistic_evaluations=units,
                trace_retained_multiplies=2*units, trace_write_multiplies=units,
                trace_update_adds=2*units, trace_context_unit_steps=units)


def compare_raw(audit, saved, expected, context):
    audit.check(set(saved) == set(expected), f'{context}: raw fields')
    for key in ('counts', 'spikes', 'resources'):
        audit.exact(saved[key], expected[key], f'{context}: exact {key}')
    for key in set(expected)-{'counts', 'spikes', 'resources'}:
        audit.close(saved[key], expected[key], f'{context}: {key}')
    audit.check(np.all((saved['counts'] >= 0) & (saved['counts'] <= 8))
                and np.array_equal(saved['counts'], np.rint(saved['counts'])), f'{context}: native count bounds')


def scale_invariants(audit, x, saved, eval_rule, context):
    gate = saved['gates']
    memory = saved['traces']
    signed = saved['drive']
    weight = saved['weights_fast']
    previous = np.concatenate((np.zeros_like(memory[:, :1]), memory[:, :-1]), axis=1)
    audit.close(gate, np.minimum(1., np.abs(x[:8]).sum(axis=-1, keepdims=True)),
                f'{context}: current-input-only gate', 1e-13)
    audit.check(np.max(np.abs(memory)) <= 1+1e-12 and np.max(np.abs(signed)) <= 1+1e-12,
                f'{context}: bounded signed traces and drive')
    audit.check(np.all(np.abs(saved['injections']) <= gate+1e-12), f'{context}: gated injection bound')
    if eval_rule == 'mean':
        audit.exact(weight, np.full_like(weight, .5), f'{context}: exact matched mean weight')
    else:
        lower = .5*(1-np.tanh(2.))
        audit.check(np.min(weight) >= lower-1e-12 and np.max(weight) <= 1-lower+1e-12,
                    f'{context}: bounded logistic weight')
    read = weight*previous[..., 0]+(1-weight)*previous[..., 1]
    audit.close(saved['contexts'], read, f'{context}: previous-trace read', 1e-12)
    audit.close(saved['injections'], gate*read, f'{context}: read and injection relation', 1e-12)
    blank = gate[..., 0] == 0
    audit.exact(saved['injections'][blank], np.zeros_like(saved['injections'][blank]),
                f'{context}: exact blank injection zero')
    audit.close(memory[blank], previous[blank]*np.array([.85, .97]),
                f'{context}: pure blank decay', 1e-12)
    return int(np.sum(blank))


def primary_inference(audit, effects, blocks):
    audit.stage = 'primary inference'
    rng = np.random.default_rng(4074900001)
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
    names = ('ledger_seconds', 'evaluation_seconds', 'training_replay_seconds', 'direction_seconds')
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

    required = SOURCE_FILES | {'PROTOCOL.md', 'audit.py', 'independent_training.py', 'train_model.py', 'run.py',
                               'TIMING.json', 'PRECHECK.json', 'SOURCE_LOCK.json'}
    audit.check(required <= set(freeze['files']), 'all source and protocol files frozen')
    audit.check(all(not name.startswith(('train/', 'evaluate/')) for name in freeze['files']), 'pre-data freeze scope')
    verify_hashes(audit, freeze['files'], ROOT, 'frozen file')
    source = read(ROOT/'SOURCE_LOCK.json')
    audit.check(source['previous_directory'] == '../DEND-TAN-scale-confirm-20261109', 'sealed confirmation source')
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


def anchored_directions(audit, x, y, w, arm, gradients, directions, native_counts, context):
    anchors = proxy.anchor(x, w, arm, Q)
    audit.exact(np.sum([step['os'] for step in anchors[Q:]], axis=0), native_counts,
                f'{context}: independent proxy native anchor counts')
    audit.close(proxy.objective(x, y, w, arm, Q, anchors), proxy.count_objective(native_counts, y, w),
                f'{context}: anchored objective centered on native objective', 1e-12)
    eps = 1e-5
    max_error = 0.
    for index, direction in enumerate(directions):
        up = proxy.objective(x, y, w+eps*direction, arm, Q, anchors)
        down = proxy.objective(x, y, w-eps*direction, arm, Q, anchors)
        error = audit.close((up-down)/(2*eps), float(gradients @ direction),
                            f'{context}: first-step anchored direction {index}', 2e-7)
        max_error = max(max_error, error)
    return max_error


def audit_fit(audit, row, x, y, w, updates, directions):
    ident, arm = row['id'], row['arm']
    saved = load(ROOT/'train'/f'{ident}.npz')
    audit.check(set(saved) == {'initial', 'theta', 'gradients', 'counts', 'history'}, f'{ident}: training archive schema')
    audit.exact(saved['initial'], w, f'{ident}: independent paired initialization')
    gradients, counts, history = (saved[key] for key in ('gradients', 'counts', 'history'))
    audit.check(gradients.shape == (updates, 176) and counts.shape == (updates, 64, 2)
                and history.shape == (updates, 4), f'{ident}: all training arrays')
    audit.check(np.isfinite(gradients).all() and np.isfinite(history).all(), f'{ident}: finite training')
    audit.check(counts.dtype == np.uint8 and np.max(counts) <= 8, f'{ident}: eight-step training counts')
    m, v = np.zeros_like(w), np.zeros_like(w)
    max_direction_error = 0.
    for k, (g, native_counts, (loss, norm, task_norm, l2_norm)) in enumerate(zip(gradients, counts, history)):
        context = f'{ident}/{k+1}'
        audit.close(loss, proxy.count_objective(native_counts, y, w), f'{context}: native count CE + L2', 1e-10)
        audit.check(min(norm, task_norm, l2_norm) >= 0, f'{context}: nonnegative norms')
        audit.close(np.linalg.norm(g), min(1., norm), f'{context}: clipped norm', 1e-10)
        regularizer = .0002*w/176
        before_clip = g*max(1., norm)
        audit.close(np.linalg.norm(before_clip-regularizer), task_norm, f'{context}: task norm', 1e-10)
        audit.close(np.linalg.norm(regularizer), l2_norm, f'{context}: L2 norm', 1e-10)
        if k in (0, updates-1):
            native = independent_scale.replay(x, w, EVAL_RULES[row['train_rule']], Q)
            audit.exact(native_counts, native['counts'], f'{context}: independent training forward')
        if k == 0:
            max_direction_error = anchored_directions(audit, x, y, w, arm, before_clip, directions, native_counts, context)
        m = .9*m+.1*g
        v = .999*v+.001*g*g
        corrected_m = m/(1-.9**(k+1))
        corrected_v = v/(1-.999**(k+1))
        w = np.clip(w-.0025*corrected_m/(np.sqrt(corrected_v)+1e-8), -2., 2.)
    audit.close(saved['theta'], w, f'{ident}: every Adam update reconstructed', 1e-10)
    return saved['theta'], counts[0], gradients[0], max_direction_error


def training(audit, blocks, updates):
    audit.stage = 'training'
    rows = read(ROOT/'train'/'ROWS.json')
    index = {(row['seed'], row['arm']): row for row in rows}
    expected = {(s, arm) for s in range(blocks) for arm in ARMS}
    audit.check(len(rows) == len(index) == 3*blocks and set(index) == expected, 'complete paired fits')
    weights = {}
    max_direction_error = 0.
    initial_pair_distances = []
    steps_per_fit = updates*64*T
    for seed in range(blocks):
        base = regenerated_base(audit, ROOT/'train'/f'base_s{seed}.npz', 64, 4074010000+100*seed)
        x, q = data.assemble(base, 16)
        audit.check(x.shape == (64, T, 6) and q == Q and not x[:, 9:25].any()
                    and data.check(base, x, 16)['cue_and_content_required'], 'training timeline and information boundary')
        w = initialize(4074030000+seed)
        directions = np.random.default_rng(4074800000+seed).normal(size=(2, 176))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        first = {}
        for arm in ARMS:
            audit.guard()
            row = index[seed, arm]
            ident = f's{seed}_{arm}'
            meta = dict(id=ident, seed=seed, arm=arm, train_rule='mean' if arm == 'MEAN' else 'dynamic',
                        detached=arm == 'DETACH', train_delay=16, h=6, lr=.0025, loss='CE', updates=updates, T=T, q=Q)
            for key, value in meta.items():
                audit.check(row[key] == value, f'{ident}: fixed training {key}')
            audit.check(0 < row['seconds'] < 840 and row['cost_per_episode'] == cost(), f'{ident}: fit elapsed/cost')
            audit.check(row['dense_forward_macs'] == 156*steps_per_fit and row['dense_backward_macs'] == 240*steps_per_fit,
                        f'{ident}: training dense MACs')
            audit.check(row['memory_backward_unit_steps'] == 12*steps_per_fit
                        and row['compatibility_backward_unit_steps'] == (0 if arm == 'DETACH' else 12*steps_per_fit),
                        f'{ident}: trace and executed compatibility reverse steps')
            weights[ident], counts, gradient, error = audit_fit(audit, row, x, base['y'], w, updates, directions)
            first[arm] = (counts, gradient)
            max_direction_error = max(max_direction_error, error)
        audit.exact(first['DYN'][0], first['DETACH'][0], f'block {seed}: identical initial DYN/DETACH forward counts')
        initial_pair_distances.append(float(np.linalg.norm(first['DYN'][1]-first['DETACH'][1])))
        print(f'Independently audited training block {seed+1}/{blocks}', flush=True)
    stats = dict(optimizer_updates=3*blocks*updates, initial_forward_identity_blocks=blocks,
                 first_gradient_directions=6*blocks, max_first_gradient_direction_error=max_direction_error,
                 initial_dyn_detach_clipped_gradient_distances=initial_pair_distances,
                 training_sample_steps=3*blocks*steps_per_fit)
    return rows, weights, stats


def evaluation(audit, training_rows, weights, blocks):
    audit.stage = 'evaluation'
    rows = read(ROOT/'evaluate'/'ROWS.json')
    index = {(row['id'], row['eval_rule']): row for row in rows}
    expected = {(f's{s}_{arm}', rule) for s in range(blocks) for arm in ARMS for rule in EVAL_RULES}
    audit.check(len(rows) == len(index) == 6*blocks and set(index) == expected, 'all common-rule evaluations')
    expected_files = {f'{ident}_{rule}.npz' for ident, rule in expected}
    expected_files |= {f'test_s{s}.npz' for s in range(blocks)}
    audit.check({p.name for p in (ROOT/'evaluate').glob('*.npz')} == expected_files, 'exact evaluation archive set')
    effects = {'dyn_minus_mean_at_mean': [], 'dyn_minus_detach_at_mean': []}
    blank_steps = 0
    for seed in range(blocks):
        base = regenerated_base(audit, ROOT/'evaluate'/f'test_s{seed}.npz', 256, 4074020000+100*seed)
        x, q = data.assemble(base, 16)
        audit.check(x.shape == (256, T, 6) and q == Q and not x[:, 9:25].any()
                    and data.check(base, x, 16)['cue_and_content_required'], 'test timeline and information boundary')
        at_mean = {}
        for fit in (row for row in training_rows if row['seed'] == seed):
            ident, arm = fit['id'], fit['arm']
            for rule, eval_arm in EVAL_RULES.items():
                audit.guard()
                context = f'{ident}_{rule}'
                row = index[ident, rule]
                meta = {key: fit[key] for key in FIT_META}
                meta.update(eval_rule=rule, condition='nominal', test_delay=16, T=T, q=q)
                for key, value in meta.items():
                    audit.check(row[key] == value, f'{context}: evaluation {key}')
                native = independent_scale.replay(x, weights[ident], eval_arm, q)
                raw = load(ROOT/'evaluate'/f'{context}.npz')
                compare_raw(audit, raw, native, context)
                blank_steps += scale_invariants(audit, x, raw, rule, context)
                scores = native_scores(raw['counts'], base)
                for key, value in scores.items():
                    audit.close(row[key], value, f'{context}: independent {key}', 1e-13)
                audit.close(row['mean_resources'], raw['resources'][:, :4].mean(0),
                            f'{context}: independent event and nonzero fanout totals')
                audit.check(row['cost_per_episode'] == cost(), f'{context}: online structural costs')
                if rule == 'mean':
                    at_mean[arm] = scores['disagree_ba']
        effects['dyn_minus_mean_at_mean'].append(at_mean['DYN']-at_mean['MEAN'])
        effects['dyn_minus_detach_at_mean'].append(at_mean['DYN']-at_mean['DETACH'])
        print(f'Independently replayed test block {seed+1}/{blocks}', flush=True)
    stats = dict(evaluations=len(rows), evaluation_sample_steps=len(rows)*256*T,
                 audited_blank_archive_steps=blank_steps)
    return effects, stats


def main(audit):
    blocks, updates = provenance(audit)
    rows, weights, training_stats = training(audit, blocks, updates)
    effects, eval_stats = evaluation(audit, rows, weights, blocks)
    inference = primary_inference(audit, effects, blocks)
    audit.stage = 'resource ledger'
    science = read(ROOT/'SCIENCE_SUMMARY.json')
    train_steps = training_stats['training_sample_steps']
    eval_steps = eval_stats['evaluation_sample_steps']
    expected = dict(blocks=blocks, fits=3*blocks, updates=3*blocks*updates, updates_per_fit=updates,
                    evaluations=6*blocks, forward_sample_steps=train_steps+eval_steps,
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
    result = dict(status='PASS', checks=audit.checks, **expected, max_numeric_error=audit.max_error,
                  cap_seconds=300, peak_working_set_bytes=peak_memory(), **audit.elapsed(), **training_stats,
                  evaluation_sample_steps=eval_steps, audited_blank_archive_steps=eval_stats['audited_blank_archive_steps'],
                  primary_effects=inference,
                  scope='Independent stimulus and initialization regeneration, every stored objective, clipped norm and '
                        'Adam update, both boundary hard forwards, two initial anchored directional checks per fit, all '
                        'native evaluation replays and resources, and both paired primary bootstraps. Full gradients at '
                        'later updates are reconstructed from the stored clipped gradients rather than recomputed.')
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
