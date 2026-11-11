"""Independent audit of fixed-weight keyed direction/norm interventions."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

import independent_amplitude as native
import independent_binding_data as data


ROOT = Path(__file__).resolve().parent
ARMS = ('KEY', 'KEY_MEAN', 'GAIN2')
KEYED = ('KEY', 'KEY_MEAN')
MATCHED = ('swap_direction', 'swap_magnitude', 'mean_matched')
T, Q, CUE, BLOCKS = 36, 28, 25, 12
SOURCE_FILES = {'model.py', 'population.py', 'reference.py', 'original_tan.py', 'metrics.py',
                'binding_data.py', 'key_model.py', 'independent_binding_data.py', 'independent_key.py'}
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


def cost(arm, condition):
    slots = 24 if arm in KEYED else 0
    units, selectors = slots*T//2, slots*T//12
    normal_steps = T if condition in MATCHED else 0
    return dict(parameters=176, spiking_cells=8, graded_branches=12,
                continuous_state_slots=20+slots, history_slots=0, trace_slots=slots,
                persistent_array_bytes=8*(28+slots), dense_forward_macs=156*T, graded_transfers=12*T,
                key_selection_unit_steps=selectors, key_normalization_divisions_upper=selectors,
                memory_tanh_evaluations=units, memory_read_adds=units, injection_multiplies=units,
                memory_retained_multiplies=2*units, memory_write_multiplies=2*units,
                memory_update_adds=2*units, memory_read_multiplies=2*units,
                extra_selector_unit_steps=2*normal_steps, extra_source_read_gate_multiplies=normal_steps,
                extra_source_read_multiplies=24*normal_steps, extra_source_read_adds=12*normal_steps,
                extra_source_injection_multiplies=12*normal_steps, normalization_norms=2*normal_steps,
                normalization_square_multiplies=24*normal_steps, normalization_sum_adds=22*normal_steps,
                normalization_sqrts=2*normal_steps, normalization_ratio_divisions_upper=normal_steps,
                normalization_scale_multiplies=12*normal_steps)


def compare_raw(audit, saved, expected, context):
    audit.check(set(saved) == set(expected), f'{context}: raw fields')
    for key in ('counts', 'spikes', 'resources', 'direction_fallback'):
        audit.exact(saved[key], expected[key], f'{context}: exact {key}')
    for key in set(expected)-{'counts', 'spikes', 'resources', 'direction_fallback'}:
        audit.close(saved[key], expected[key], f'{context}: {key}')
    audit.check(np.all((saved['counts'] >= 0) & (saved['counts'] <= 8))
                and np.array_equal(saved['counts'], np.rint(saved['counts'])), f'{context}: native count bounds')


def primary_inference(audit, effects, blocks):
    audit.stage = 'primary inference'
    rng = np.random.default_rng(4076900001)
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
                and frozen['evaluations'] == 8*BLOCKS and examples in (256, 128), 'frozen scope')
    audit.check({'PROTOCOL.md', 'REPRODUCE.md', 'SOURCE_LOCK.json', 'PRECHECK.json', 'TIMING.json',
                 'run.py', 'amplitude_model.py', 'audit.py', 'independent_amplitude.py', 'summarize.py', 'report.py'}
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
    audit.check(lock['previous_directory'] == '../DEND-TAN-key-binding-20261110', 'exact prior source directory')
    previous = ROOT/lock['previous_directory']
    audit.check(digest(previous/'MANIFEST.json') == lock['previous_manifest_sha256'], 'prior manifest seal')
    manifest = read(previous/'MANIFEST.json')['files']
    for name in ('AUDIT.json', 'SCIENCE_SUMMARY.json'):
        audit.check(digest(previous/name) == manifest[name]['sha256'], f'sealed prior {name}')
    earlier_audit = read(previous/'AUDIT.json')
    earlier_science = read(previous/'SCIENCE_SUMMARY.json')
    audit.check(earlier_audit['status'] == 'PASS' and earlier_science['status'] == 'COMPLETE'
                and earlier_science['blocks'] == BLOCKS and earlier_science['fits'] == 36,
                'source is the complete audited 12-block key-binding study')
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
        audit.check(theta.shape == (176,) and theta.dtype == weights[ident].dtype == np.float64
                    and np.isfinite(theta).all(), f'{ident}: native weights')
        audit.exact(weights[ident], theta, f'{ident}: final checkpoint theta preserved')
    audit.guard()
    return examples, weights


def verify_archive(audit, raw, x, arm, condition, context):
    read_vectors, fallback = raw['read_vectors'], raw['direction_fallback']
    audit.check(read_vectors.shape == (len(x), 3, 12) and fallback.shape == (len(x), 3)
                and fallback.dtype == np.bool_, f'{context}: complete per-example read archive')
    audit.exact(read_vectors[:8], raw['injections'][:, CUE:Q], f'{context}: cue read archive agrees with trace')
    audit.exact(raw['counts'][:8], raw['spikes'][:, Q:, 6:].sum(axis=1), f'{context}: query count archive')
    if condition not in MATCHED:
        audit.check(not fallback.any(), f'{context}: no fallback on unrescaled path')
    if arm == 'GAIN2':
        audit.exact(raw['injections'], raw['currents'], f'{context}: strong gain injection')
        return
    memory, cue_memory = raw['memory'], raw['cue_memory']
    audit.check(cue_memory.shape == (len(x), 3, 12, 2) and np.abs(cue_memory).max() <= 1.
                and np.abs(memory).max() <= 1., f'{context}: complete bounded slot archive')
    audit.exact(cue_memory[:8], memory[:, CUE-1:Q-1], f'{context}: pre-read cue bank')
    audit.close(cue_memory[:, 1:], .97*cue_memory[:, :-1], f'{context}: all-example cue decay', 1e-12)
    previous = np.concatenate((np.zeros_like(memory[:, :1]), memory[:, :-1]), axis=1)
    writes, drive, gate = raw['writes'], raw['drive'], raw['read_gate']
    expected = .97*(1-writes[..., None, :])*previous+writes[..., None, :]*drive[..., None]
    audit.close(memory, expected, f'{context}: unchanged slot decay and write', 1e-12)
    audit.check(not writes[:, CUE:].any() and not gate[:, :CUE].any() and not gate[:, Q:].any(),
                f'{context}: writes before cue and reads within cue only')
    no_read = gate[..., 0] == 0.
    audit.exact(raw['injections'][no_read], np.zeros_like(raw['injections'][no_read]), f'{context}: no injection at zero read gate')
    cue_tags = x[:8, CUE:Q, 2:4]
    cue_keys = cue_tags/cue_tags.sum(axis=-1, keepdims=True)
    expected_selector = cue_keys
    if condition in ('swap_read', 'swap_direction'):
        expected_selector = 1.-cue_keys
    if arm == 'KEY_MEAN' or condition in ('mean_read', 'mean_matched'):
        expected_selector = np.full_like(cue_keys, .5)
    audit.exact(raw['selector'][:, CUE:Q], expected_selector, f'{context}: observed direction selector')


def source_vectors(x, bank):
    """Recover all direct read vectors from saved pre-read slots and observed addresses."""
    cue = x[:, CUE:Q]
    tags = np.maximum(0., cue[..., 2:4])
    mass = tags.sum(axis=-1, keepdims=True)
    keys = np.divide(tags, mass, out=np.zeros_like(tags), where=mass > 0.)
    gate = (1.-np.clip(cue[..., 5:6], 0., 1.))*np.minimum(1., mass)
    selected = 2*gate*np.einsum('...dj,...j->...d', bank, keys)
    other = 2*gate*np.einsum('...dj,...j->...d', bank, 1.-keys)
    mean = gate*bank.sum(axis=-1)
    return selected, other, mean


def verify_matched(audit, raw, source, target, context):
    """Check norm, orientation, and zero-direction behavior separately."""
    vector = raw['read_vectors']
    source_norm = np.linalg.norm(source, axis=-1)
    target_norm = np.linalg.norm(target, axis=-1)
    vector_norm = np.linalg.norm(vector, axis=-1)
    norm_residual = audit.close(vector_norm, target_norm, f'{context}: fixed per-read full-vector norm', 1e-12)
    zero_target = target_norm == 0.
    fallback = (source_norm == 0.) & ~zero_target
    audit.exact(raw['direction_fallback'], fallback, f'{context}: exactly zero direction and positive target norm')
    audit.exact(vector[zero_target], np.zeros_like(vector[zero_target]), f'{context}: zero target produces zero read')
    fallback_vectors = np.zeros_like(vector[fallback])
    fallback_vectors[:, 0] = target_norm[fallback]
    audit.close(vector[fallback], fallback_vectors, f'{context}: first-branch unit fallback', 1e-12)
    valid = ~zero_target & ~fallback
    audit.check(np.all(vector_norm[valid] > 0.), f'{context}: nonzero read for positive target norm')
    source_unit = source[valid]/source_norm[valid, None]
    vector_unit = vector[valid]/vector_norm[valid, None]
    direction_residual = audit.close(vector_unit, source_unit, f'{context}: preserves source vector direction', 1e-12)
    return norm_residual, direction_residual


def verify_key_paths(audit, x, paths, context):
    nominal = paths['nominal']
    selected, other, mean = source_vectors(x, nominal['cue_memory'])
    sources = {'nominal': selected, 'swap_read': other, 'mean_read': mean}
    targets = {'swap_direction': (other, selected), 'swap_magnitude': (selected, other),
               'mean_matched': (mean, selected)}
    direct_norm_reference = {'swap_direction': 'nominal', 'swap_magnitude': 'swap_read', 'mean_matched': 'nominal'}
    max_norm_residual = max_direction_residual = 0.
    for name, raw in paths.items():
        name_context = f'{context}/{name}'
        for key in ('states', 'spikes', 'candidates', 'currents', 'injections', 'drive'):
            audit.exact(raw[key][:, :CUE], nominal[key][:, :CUE], f'{name_context}: unchanged prefix {key}')
        for key in ('cue_memory', 'memory', 'keys', 'writes', 'read_gate'):
            audit.exact(raw[key], nominal[key], f'{name_context}: unchanged memory/input signal {key}')
        if name in sources:
            audit.close(raw['read_vectors'], sources[name], f'{name_context}: specified unrescaled read source', 1e-12)
        else:
            norm, direction = verify_matched(audit, raw, *targets[name], name_context)
            direct_norm = np.linalg.norm(paths[direct_norm_reference[name]]['read_vectors'], axis=-1)
            matched_norm = np.linalg.norm(raw['read_vectors'], axis=-1)
            equal = audit.close(matched_norm, direct_norm, f'{name_context}: norm of paired natural read', 1e-12)
            max_norm_residual = max(max_norm_residual, norm, equal)
            max_direction_residual = max(max_direction_residual, direction)
    return max_norm_residual, max_direction_residual


def verify_row(audit, row, raw, base, seed, arm, condition):
    ident = f's{seed}_{arm}'
    context = f'{ident}/{condition}'
    identity = dict(id=ident, seed=seed, arm=arm, condition=condition, T=T, q=Q)
    scores = binding_scores(raw['counts'], base)
    audit.check(set(row) == set(identity) | set(scores) | {'mean_resources', 'cost_per_episode', 'fallback_reads'},
                f'{context}: row schema')
    for key, value in identity.items():
        audit.check(row[key] == value, f'{context}: {key}')
    for key, value in scores.items():
        audit.close(row[key], value, f'{context}: native {key}', 1e-13)
    audit.close(row['mean_resources'], raw['resources'][:, :4].mean(axis=0), f'{context}: event and fanout ledger')
    audit.check(row['cost_per_episode'] == cost(arm, condition), f'{context}: per-episode structural costs')
    audit.check(row['fallback_reads'] == np.count_nonzero(raw['direction_fallback']), f'{context}: fallback read count')
    return scores


def evaluate(audit, n, weights):
    audit.stage = 'replay'
    rows = read(ROOT/'evaluate'/'ROWS.json')
    index = {(row['id'], row['condition']): row for row in rows}
    expected = {(f's{s}_{arm}', condition) for s in range(BLOCKS) for arm in ARMS for condition in native.CONDITIONS[arm]}
    audit.check(len(rows) == len(index) == 8*BLOCKS and set(index) == expected, 'complete 96 paired evaluation rows')
    files = {f'{ident}_{condition}.npz' for ident, condition in expected} | {f'test_s{s}.npz' for s in range(BLOCKS)}
    audit.check({p.name for p in (ROOT/'evaluate').glob('*.npz')} == files, 'exact evaluation archive set')
    totals = dict(input_array_bytes=0, retained_raw_array_bytes=0, matched_cases=0,
                  matched_read_steps=0, fallback_reads=0)
    effects = {'equal_norm_direction_effect': [], 'selected_direction_minus_matched_mean': []}
    max_norm_residual = max_direction_residual = 0.
    for seed in range(BLOCKS):
        base = regenerated_base(audit, ROOT/'evaluate'/f'test_s{seed}.npz', n, 4076020000+100*seed)
        x, q = data.assemble(base)
        audit.check(x.shape == (n, T, 6) and q == Q and data.check(base, x)['ceiling'] == 1.,
                    f'block {seed}: regenerated full binding task')
        totals['input_array_bytes'] += x.nbytes
        for arm in ARMS:
            ident = f's{seed}_{arm}'
            paths, scores = {}, {}
            for condition in native.CONDITIONS[arm]:
                audit.guard()
                context = f'{ident}/{condition}'
                raw = load(ROOT/'evaluate'/f'{ident}_{condition}.npz')
                compare_raw(audit, raw, native.replay(x, weights[ident], arm, q, condition), context)
                verify_archive(audit, raw, x, arm, condition, context)
                scores[condition] = verify_row(audit, index[ident, condition], raw, base, seed, arm, condition)
                paths[condition] = raw
                totals['retained_raw_array_bytes'] += sum(value.nbytes for value in raw.values())
                totals['fallback_reads'] += int(np.count_nonzero(raw['direction_fallback']))
                if condition in MATCHED:
                    totals['matched_cases'] += 1
                    totals['matched_read_steps'] += raw['direction_fallback'].size
            if arm == 'KEY_MEAN':
                raw = paths['nominal']
                _, _, mean = source_vectors(x, raw['cue_memory'])
                audit.close(raw['read_vectors'], mean, f'{ident}: all-example mean read source', 1e-12)
            if arm != 'KEY':
                continue
            norm, direction = verify_key_paths(audit, x, paths, ident)
            max_norm_residual = max(max_norm_residual, norm)
            max_direction_residual = max(max_direction_residual, direction)
            other = {condition: score['counterfactual_joint'] for condition, score in scores.items()}
            effects['equal_norm_direction_effect'].append(((other['swap_direction']-other['nominal'])
                                                           +(other['swap_read']-other['swap_magnitude']))/2.)
            effects['selected_direction_minus_matched_mean'].append(scores['nominal']['disagree_joint_correct']
                                                                    -scores['mean_matched']['disagree_joint_correct'])
        print(f'Independently replayed amplitude block {seed+1}/{BLOCKS}', flush=True)
    audit.check(totals['matched_cases'] == 36 and totals['matched_read_steps'] == 36*n*3, 'complete matched read scope')
    diagnostics = dict(max_matched_norm_residual=max_norm_residual, max_matched_direction_residual=max_direction_residual)
    return effects, totals, diagnostics


def main(audit):
    audit.check(all(os.environ.get(key) == '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')),
                'single-thread independent audit')
    n, weights = provenance(audit)
    effects, totals, diagnostics = evaluate(audit, n, weights)
    inference = primary_inference(audit, effects, BLOCKS)
    audit.stage = 'resource ledger'
    science = read(ROOT/'SCIENCE_SUMMARY.json')
    steps = 8*BLOCKS*n*T
    expected = dict(blocks=BLOCKS, examples_per_block=n, evaluations=8*BLOCKS, fits=0, updates=0,
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
    result = dict(status='PASS', checks=audit.checks, **expected, max_numeric_error=audit.max_error,
                  cap_seconds=300, peak_working_set_bytes=peak_memory(), **audit.elapsed(), **diagnostics,
                  primary_effects=inference,
                  scope='Independent checkpoint provenance, stimulus regeneration, all native evaluations, '
                        'unchanged slot and prefix states, independently derived norm and direction sources, '
                        'exact fallback flags and branch convention, raw scores, resources and both primary bootstraps.')
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
