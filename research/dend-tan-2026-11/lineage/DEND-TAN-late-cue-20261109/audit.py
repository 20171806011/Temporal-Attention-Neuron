"""Independent audit of late-cue datasets, native dynamics, Adam and metrics."""

from datetime import datetime
import hashlib
from itertools import product
import json
from pathlib import Path
import time

import numpy as np

import independent_data as data
import reference


ROOT = Path(__file__).resolve().parent
ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2')
TRAIN_DELAYS = (0, 16)
ATTENTION = ('FULL', 'NOATT')
SOURCE_FILES = {'model.py', 'population.py', 'reference.py', 'original_tan.py'}
IDENTITY = ('id', 'seed', 'arm', 'train_delay', 'h', 'lr', 'loss', 'updates')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path) as archive:
        return dict(archive)


class Audit:
    def __init__(self):
        self.start = time.perf_counter()
        self.cpu_start = time.process_time()
        self.checks = 0
        self.max_error = 0.
        self.stage = 'provenance'

    def check(self, condition, context):
        self.checks += 1
        if time.perf_counter() - self.start > 300:
            raise TimeoutError(f'300 second independent audit cap during {self.stage}')
        assert bool(condition), context

    def exact(self, actual, expected, context):
        self.check(np.array_equal(actual, expected), context)

    def close(self, actual, expected, context, tolerance=1e-8):
        actual = np.asarray(actual, dtype=float)
        expected = np.asarray(expected, dtype=float)
        self.check(actual.shape == expected.shape, f'{context}: shape')
        self.check(np.isfinite(actual).all() and np.isfinite(expected).all(), f'{context}: finite')
        error = float(np.max(np.abs(actual-expected))) if actual.size else 0.
        self.max_error = max(self.max_error, error)
        self.check(error <= tolerance, f'{context}: error {error:.17g} > {tolerance}')

    def elapsed(self):
        return dict(wall_seconds=time.perf_counter()-self.start, cpu_seconds=time.process_time()-self.cpu_start)


def verify_hashes(audit, entries, folder, context):
    for name, sha in entries.items():
        audit.check(digest(folder/name) == sha, f'{context}: {name}')


def provenance(audit):
    freeze = read(ROOT/'FREEZE.json')
    timing = read(ROOT/'TIMING.json')
    audit.check(freeze['scientific_data_generated'] is False, 'pre-data freeze')
    audit.check(timing['status'] == read(ROOT/'PRECHECK.json')['status'] == 'PASS', 'fixture passes')
    updates = freeze['updates_per_fit']
    audit.check(updates in (400, 600) and updates == timing['updates'], 'frozen timing-selected update count')
    audit.check(SOURCE_FILES | {'PROTOCOL.md', 'audit.py', 'independent_data.py', 'TIMING.json',
                                 'PRECHECK.json'} <= set(freeze['files']), 'complete frozen provenance')
    verify_hashes(audit, freeze['files'], ROOT, 'frozen source')
    source = read(ROOT/'SOURCE_LOCK.json')
    audit.check(source['previous_directory'] == '../DEND-TAN-streaming-readout-20261109',
                'expected sealed native source')
    previous = (ROOT/source['previous_directory']).resolve()
    manifest = previous/'MANIFEST.json'
    audit.check(digest(manifest) == source['previous_manifest_sha256'], 'previous source manifest')
    manifest_files = read(manifest)['files']
    source_hashes = source['files']
    audit.check(set(source_hashes) == SOURCE_FILES, 'exact unchanged-source set')
    for name, sha in source_hashes.items():
        audit.check(manifest_files[name]['sha256'] == digest(previous/name) == digest(ROOT/name) == sha,
                    f'unchanged native source: {name}')

    lock = read(ROOT/'train/TRAIN_LOCK.json')
    audit.check(lock['test_generated'] is False, 'test data held until training lock')
    required = {f'base_s{s}.npz' for s in range(3)} | {'ROWS.json'}
    required |= {f's{s}_{arm}_d{delay}.npz' for s, arm, delay in product(range(3), ARMS, TRAIN_DELAYS)}
    audit.check(required <= set(lock['files']), 'all training artifacts locked')
    verify_hashes(audit, lock['files'], ROOT/'train', 'locked training artifact')
    frozen_time = datetime.fromisoformat(freeze['utc'])
    locked_time = datetime.fromisoformat(lock['utc'])
    audit.check(frozen_time < locked_time, 'freeze predates locked training')
    for seed in range(3):
        audit.check((ROOT/'evaluate'/f'test_s{seed}.npz').stat().st_mtime >= locked_time.timestamp(),
                    'test data timestamp is no earlier than training lock')
    env = read(ROOT/'ENVIRONMENT.json')
    audit.check(all(env['threads'][name] == '1' for name in
                    ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')), 'single BLAS thread')
    return updates


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


def objective(counts, y, w):
    c = counts.astype(float)
    margin = .5*(c[:, 1]-c[:, 0])
    return float(np.mean(np.logaddexp(0, (1-2*y)*margin)) + .0001*np.mean(w**2))


def cost(arm, T):
    history = 60 if arm in ATTENTION else 0
    return dict(parameters=176, spiking_cells=8, graded_branches=12,
                continuous_state_slots=20+history, history_slots=history,
                persistent_array_bytes=8*(20+history+6+2), dense_forward_macs=156*T,
                graded_transfers=12*T, history_roll_writes=72*T if history else 0,
                attention_unit_steps=12*T if history else 0,
                softmax_exponentials=60*T if arm == 'FULL' else 0)


def verify_cost(audit, actual, arm, T, context):
    expected = cost(arm, T)
    audit.check(all(actual.get(key) == value for key, value in expected.items()), f'{context}: state/operation ledger')


def training(audit, updates):
    audit.stage = 'training'
    rows = read(ROOT/'train/ROWS.json')
    expected = set(product(range(3), ARMS, TRAIN_DELAYS))
    audit.check(len(rows) == 24 and {(r['seed'], r['arm'], r['train_delay']) for r in rows} == expected,
                '24 complete paired fits')
    total_steps = 0
    for seed in range(3):
        base = regenerated_base(audit, ROOT/'train'/f'base_s{seed}.npz', 64, 4041010000+100*seed)
        inputs = {delay: data.assemble(base, delay) for delay in TRAIN_DELAYS}
        for delay, (x, _) in inputs.items():
            audit.check(data.check(base, x, delay)['cue_and_content_required'], 'training information boundary')
        for row in (r for r in rows if r['seed'] == seed):
            arm, delay = row['arm'], row['train_delay']
            x, q = inputs[delay]
            T = x.shape[1]
            ident = f's{seed}_{arm}_d{delay}'
            audit.check(row['id'] == ident and row['T'] == T and row['q'] == q, 'training identity and timeline')
            audit.check(row['h'] == 6 and row['loss'] == 'CE' and row['lr'] == .0025 and row['updates'] == updates,
                        f'{ident}: fixed training recipe')
            audit.check(0 < row['seconds'] < 840, f'{ident}: fit elapsed time')
            saved = load(ROOT/'train'/f'{ident}.npz')
            w = initialize(4041030000+seed)
            audit.exact(saved['initial'], w, f'{ident}: independent paired initialization')
            gradients, counts, history = (saved[key] for key in ('gradients', 'counts', 'history'))
            audit.check(gradients.shape == (updates, 176) and counts.shape == (updates, 64, 2)
                        and history.shape == (updates, 4), f'{ident}: complete training arrays')
            audit.check(np.isfinite(gradients).all() and np.isfinite(history).all(), f'{ident}: finite training')
            audit.check(counts.dtype == np.uint8 and np.max(counts) <= 8, f'{ident}: eight-step training counts')
            m, v = np.zeros(176), np.zeros(176)
            for k in range(updates):
                g = gradients[k]
                loss, norm, task_norm, l2_norm = history[k]
                audit.close(loss, objective(counts[k], base['y'], w), f'{ident}/{k}: count CE + L2')
                audit.check(min(norm, task_norm, l2_norm) >= 0, f'{ident}/{k}: nonnegative norms')
                audit.close(np.linalg.norm(g), min(1., norm), f'{ident}/{k}: clipped norm')
                regularizer = .0002*w/176
                audit.close(np.linalg.norm(g*max(1., norm)-regularizer), task_norm, f'{ident}/{k}: task norm')
                audit.close(np.linalg.norm(regularizer), l2_norm, f'{ident}/{k}: L2 norm')
                if k in (0, updates-1):
                    native = reference.replay(x, w, arm, q)
                    audit.exact(counts[k], native['counts'], f'{ident}/{k}: independent training forward')
                m = .9*m + .1*g
                v = .999*v + .001*g*g
                corrected_m = m/(1.-.9**(k+1))
                corrected_v = v/(1.-.999**(k+1))
                w = np.clip(w-.0025*corrected_m/(np.sqrt(corrected_v)+1e-8), -2., 2.)
            audit.close(saved['theta'], w, f'{ident}: all Adam updates reconstructed', 1e-10)
            steps = updates*64*T
            audit.check(row['dense_forward_macs'] == 156*steps and row['dense_backward_macs'] == 240*steps,
                        f'{ident}: train dense MACs')
            audit.check(row['attention_backward_unit_steps'] == (12*steps if arm in ATTENTION else 0),
                        f'{ident}: backward attention unit steps')
            audit.check(row['history_gradient_accumulations'] == (12*updates*64*(5*T-10) if arm in ATTENTION else 0),
                        f'{ident}: history reverse accumulation count')
            verify_cost(audit, row['cost_per_episode'], arm, T, ident)
            total_steps += steps
        print(f'Independent late-cue training audit block {seed+1}/3 complete', flush=True)
    return rows, total_steps


def metrics(counts, base):
    y = base['y']
    d = counts[:, 1].astype(int)-counts[:, 0].astype(int)
    pred = np.where(d > 0, 1, np.where(d < 0, 0, -1))
    ties = d == 0
    silence = np.all(counts == 0, axis=1)
    saturation = np.all(counts == 8, axis=1)
    good = pred == y
    result = dict(accuracy=float(good.mean()), coverage=float(np.mean(~ties)), silence=float(silence.mean()),
                  saturated_tie=float(saturation.mean()), intermediate_tie=float(np.mean(ties & ~silence & ~saturation)),
                  recalls=[float(np.mean(good[y == label])) for label in (0, 1)],
                  prediction_counts=[int(np.count_nonzero(pred == label)) for label in (-1, 0, 1)])
    result['balanced_accuracy'] = float(np.mean(result['recalls']))
    disagree = base['first_bit'] != base['second_bit']
    second = base['selected'] == 1
    for name, mask in (('agree', ~disagree), ('disagree', disagree), ('select_first', ~second), ('select_second', second)):
        result[f'{name}_ba'] = float(np.mean([good[mask & (y == label)].mean() for label in (0, 1)]))
        result[f'{name}_coverage'] = float(np.mean(~ties[mask]))
    rows = data.lookup(base)
    both = [good[i] and good[rows[(octet, 1, first, second_bit)]]
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


def negative_control(audit, condition, base, counts, row, context):
    paired = data.pairs(base, condition)
    audit.exact(counts[paired[:, 0]], counts[paired[:, 1]], f'{context}: exact identical-input counts')
    ba, coverage = {
        'no_cue': ('disagree_ba', 'disagree_coverage'),
        'no_candidates': ('balanced_accuracy', 'coverage'),
        'no_first': ('select_first_ba', 'select_first_coverage'),
        'no_second': ('select_second_ba', 'select_second_coverage'),
    }[condition]
    audit.close(row[ba], row[coverage]/2, f'{context}: exact BA/coverage identity', 1e-12)
    audit.check(row['balanced_accuracy'] <= (.5 if condition == 'no_candidates' else .75),
                f'{context}: full information ceiling')
    if condition == 'no_cue':
        audit.check(row['disagree_joint_correct'] == 0, f'{context}: no joint cue-specific decisions without cue')
    return len(paired), abs(row[ba]-row[coverage]/2)


def evaluation(audit, training_rows):
    audit.stage = 'evaluation'
    all_rows = read(ROOT/'evaluate/ROWS.json')
    audit.check(len(all_rows) == 168, '168 final evaluation rows')
    counts = dict(evaluations=0, evaluation_sample_steps=0, source_injection_oracle_models=0,
                  mathematical_negative_control_conditions=0, duplicate_opposite_label_count_pairs=0,
                  independent_test_information_checks=0, max_negative_control_residual=0.)
    for seed in range(3):
        base = regenerated_base(audit, ROOT/'evaluate'/f'test_s{seed}.npz', 256, 4041020000+100*seed)
        settings = {(delay, 'nominal') for delay in data.DELAYS}
        settings.update(product(TRAIN_DELAYS, data.CONDITIONS[1:]))
        inputs = {setting: data.assemble(base, *setting) for setting in settings}
        for setting, (x, _) in inputs.items():
            audit.check(data.check(base, x, *setting)['balanced_labels'], f'test information controls {seed}/{setting}')
            counts['independent_test_information_checks'] += 1
        for fit in (row for row in training_rows if row['seed'] == seed):
            ident, arm, trained = fit['id'], fit['arm'], fit['train_delay']
            w = load(ROOT/'train'/f'{ident}.npz')['theta']
            rows = [row for row in all_rows if row['id'] == ident]
            needed = {(d, 'nominal') for d in data.DELAYS} | {(trained, c) for c in data.CONDITIONS[1:]}
            audit.check(len(rows) == 7 and {(r['test_delay'], r['condition']) for r in rows} == needed,
                        f'{ident}: all fixed evaluation cases')
            for row in rows:
                delay, condition = row['test_delay'], row['condition']
                context = f'{ident}_d{delay}_{condition}'
                audit.check(all(row[key] == fit[key] for key in IDENTITY), f'{context}: fitted identity')
                x, q = inputs[(delay, condition)]
                T = x.shape[1]
                audit.check(row['T'] == T and row['q'] == q, f'{context}: evaluation timeline')
                native = reference.replay(x, w, arm, q)
                saved = load(ROOT/'evaluate'/f'{context}.npz')
                compare_raw(audit, saved, native, context)
                for key, value in metrics(native['counts'], base).items():
                    audit.close(row[key], value, f'{context}: independent {key}')
                audit.close(row['mean_resources'], native['resources'][:, :4].mean(0), f'{context}: event/fanout totals')
                verify_cost(audit, row['cost_per_episode'], arm, T, context)
                if condition != 'nominal':
                    pairs, residual = negative_control(audit, condition, base, native['counts'], row, context)
                    counts['mathematical_negative_control_conditions'] += 1
                    counts['duplicate_opposite_label_count_pairs'] += pairs
                    counts['max_negative_control_residual'] = max(counts['max_negative_control_residual'], residual)
                if arm in ATTENTION and condition == 'nominal' and delay == trained:
                    scalar = reference.scalar_oracle(x[:2], w, arm, q)
                    for key in ('counts', 'spikes'):
                        audit.exact(saved[key][:2], scalar[key], f'{context}: original scalar TAN {key}')
                    for key in ('states', 'histories', 'injections'):
                        audit.close(saved[key][:2], scalar[key], f'{context}: original scalar TAN {key}')
                    counts['source_injection_oracle_models'] += 1
                counts['evaluations'] += 1
                counts['evaluation_sample_steps'] += len(x)*T
        print(f'Independent late-cue evaluation audit block {seed+1}/3 complete', flush=True)
    audit.check(counts['evaluations'] == 168 and counts['mathematical_negative_control_conditions'] == 96
                and counts['duplicate_opposite_label_count_pairs'] == 7680
                and counts['independent_test_information_checks'] == 33
                and counts['source_injection_oracle_models'] == 12, 'complete evaluation audit scope')
    return counts


def main(audit):
    updates = provenance(audit)
    rows, train_steps = training(audit, updates)
    evaluation_counts = evaluation(audit, rows)
    audit.stage = 'ledger'
    science = read(ROOT/'SCIENCE_SUMMARY.json')
    total_steps = train_steps+evaluation_counts['evaluation_sample_steps']
    expected = dict(fits=24, updates=24*updates, updates_per_fit=updates, evaluations=168,
                    forward_sample_steps=total_steps, dense_forward_macs=156*total_steps,
                    dense_backward_macs=240*train_steps)
    audit.check(science['status'] == 'COMPLETE' and 0 < science['wall_seconds'] <= 840, 'science finished within cap')
    audit.check(science['peak_working_set_bytes'] <= 1024**3, 'science memory cap')
    audit.check(train_steps == 43008*updates and evaluation_counts['evaluation_sample_steps'] == 1351680,
                'independent aggregate sequence lengths')
    for key, value in expected.items():
        audit.check(science[key] == value, f'science ledger {key}')
    result = dict(status='PASS', checks=audit.checks, fits=24, optimizer_updates=24*updates,
                  updates_per_fit=updates, source_injection_oracle_episodes=24,
                  forward_sample_steps=total_steps, dense_forward_macs=156*total_steps,
                  dense_backward_macs=240*train_steps, max_numeric_error=audit.max_error, cap_seconds=300,
                  **evaluation_counts, **audit.elapsed(),
                  scope='Independent late-cue generator and exact information boundaries, all native evaluation replays, '
                        'first and last training forwards, all clipped Adam updates and count CE/L2 objectives, '
                        'exact duplicate input negative controls, original TAN premembrane scalar source, provenance and cost. '
                        'Scientific parameter gradients are not independently regenerated at every update.')
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
