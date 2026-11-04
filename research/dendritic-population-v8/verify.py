"""Independent V8 data, native dynamics, optimizer and seed-unit audit.

No imports from the scientific model, trainer or evaluator. Native replay uses
separately written scalar equations. Saved gradients reconstruct Adam updates;
this does not independently rederive every surrogate gradient. Internal workflow
verification is not outside replication or peer review.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np
from scipy.stats import t as student_t
R=Path(__file__).resolve().parent
checks=0

def check(ok,why):
    global checks
    checks+=1
    if not ok:raise AssertionError(why)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_json(p):return json.loads(p.read_text())

def initialize(seed):
    rng=np.random.default_rng(seed)
    return np.concatenate([rng.normal(0,.45,72),rng.normal(0,.25,72),np.full(12,.15),np.full(6,.05),rng.normal(0,.5,12),np.full(2,.1)])

def regenerate_xor(n,seed):
    g=np.random.default_rng(seed);pair=np.repeat(np.arange(4),n//4);g.shuffle(pair)
    amps=g.uniform(.8,1.2,(n,2));shift=g.integers(0,3,(n,2));x=np.zeros((n,24,6))
    for i in range(n):
        for k in range(24):
            if 2+shift[i,0]<=k<6+shift[i,0]:x[i,k,pair[i]//2]=amps[i,0]
            if 10+shift[i,1]<=k<14+shift[i,1]:x[i,k,2+pair[i]%2]=amps[i,1]
    x[:,16:,4]=1;x[:,7:10,5]=g.uniform(0,.5,(n,3));x=np.maximum(x+g.normal(0,.03,x.shape),0)
    return x,(pair//2)^(pair%2),pair

def replay(x,t,arm,reset_at=None):
    n=len(x);wi=t[:72].reshape(6,12);wr=t[72:144].reshape(6,12).copy();db=t[144:156];vb=t[156:162];wo=t[162:174].reshape(6,2);ob=t[174:]
    state=np.zeros((n,20));old=np.zeros((n,6));ss=[];cs=[];zs=[]
    for step in range(24):
        if step==reset_at:state=np.zeros((n,20));old=np.zeros((n,6))
        cand=np.zeros((n,20));spike=np.zeros((n,20));nextstate=np.zeros_like(state);loc=np.zeros((n,12))
        for b in range(12):
            drive=np.zeros(n);recur=np.zeros(n)
            for a in range(6):drive+=x[:,step,a]*wi[a,b];recur+=old[:,a]*wr[a,b]
            cand[:,b]=.85*state[:,b]+.25*(drive+recur+db[b])
            if arm=='POINT':
                spike[:,b]=cand[:,b]>.5;loc[:,b]=spike[:,b];nextstate[:,b]=cand[:,b]*(1-spike[:,b])
            else:loc[:,b]=np.maximum(cand[:,b],0);nextstate[:,b]=cand[:,b]
        for h in range(6):
            k=12+h;cand[:,k]=.65*state[:,k]+.5*(loc[:,2*h]+loc[:,2*h+1])+vb[h]
            spike[:,k]=cand[:,k]>.5;nextstate[:,k]=cand[:,k]*(1-spike[:,k])
        for o in range(2):
            drive=np.zeros(n)
            for h in range(6):drive+=old[:,h]*wo[h,o]
            k=18+o;cand[:,k]=.65*state[:,k]+drive+ob[o];spike[:,k]=cand[:,k]>.5;nextstate[:,k]=cand[:,k]*(1-spike[:,k])
        old=spike[:,12:18];state=nextstate;ss.append(state);cs.append(cand);zs.append(spike)
    spikes=np.stack(zs,1);counts=spikes[:,16:,18:].sum(1)
    pred=np.where(counts[:,0]==counts[:,1],-1,counts.argmax(1))
    events=np.stack([spikes[:,:,:12].sum((1,2)),spikes[:,:,12:18].sum((1,2)),spikes[:,:,18:].sum((1,2))],1)
    return dict(states=np.stack(ss,1),candidates=np.stack(cs,1),spikes=spikes,counts=counts,prediction=pred,events=events)


def adam(initial, gradients, lr):
    """Reconstruct every parameter state from clipped gradients, no training."""
    check(gradients.shape == (200, 176), '200 by 176 saved clipped gradients')
    theta = initial.copy()
    m = np.zeros(176)
    v = np.zeros(176)
    weights = [theta.copy()]
    update_norms = []
    for k, g in enumerate(gradients):
        check(np.isfinite(g).all() and np.linalg.norm(g) <= 1 + 1e-12,
              'finite globally clipped gradient')
        m = .9 * m + .1 * g
        v = .999 * v + .001 * g * g
        new = np.clip(theta - lr * (m / (1 - .9 ** (k + 1))) /
                      (np.sqrt(v / (1 - .999 ** (k + 1))) + 1e-8), -2, 2)
        update_norms.append(float(np.linalg.norm(new-theta)))
        theta = new
        weights.append(theta.copy())
    return np.stack(weights), np.array(update_norms)


def loss_from_counts(counts, y, theta, kind):
    if kind == 'RATE':
        target = np.full_like(counts, .125)
        target[np.arange(len(y)), y] = .75
        value = np.mean((counts / 8 - target) ** 2)
    else:
        check(kind == 'CE', 'registered loss')
        logits = counts * .5
        logits -= logits.max(1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(1, keepdims=True)
        value = -np.log(probability[np.arange(len(y)), y]).mean()
    return float(value + .0001 * np.mean(theta * theta))


def count_metrics(counts, y):
    pred = np.where(counts[:, 0] == counts[:, 1], -1, counts.argmax(1))
    n = len(y)
    matrix = np.zeros((2, 3), dtype=int)
    for truth, guess in zip(y, pred):
        matrix[int(truth), int(guess) if guess >= 0 else 2] += 1
    accuracy = float(np.mean(pred == y))
    coverage = float(np.mean(pred >= 0))
    silent = np.all(counts == 0, axis=1)
    ties = pred < 0
    margins = counts[np.arange(n), y] - counts[np.arange(n), 1-y]
    histogram = [int(np.sum(margins == k)) for k in range(-8, 9)]
    return dict(accuracy=accuracy, coverage=coverage,
                abstentions=int(np.sum(pred < 0)),
                abstention_rate=float(np.mean(pred < 0)),
                silent_fraction=float(silent.mean()),
                nonsilent_tie_fraction=float(np.mean(ties & ~silent)),
                both_saturated_fraction=float(np.all(counts == 8, axis=1).mean()),
                class_recall=[float(matrix[k, k] / np.sum(y == k)) for k in (0, 1)],
                confusion=matrix.tolist(),
                mean_signed_count_margin=float(margins.mean()),
                median_signed_count_margin=float(np.median(margins)),
                signed_count_margin_histogram=histogram,
                failed=bool(accuracy < .9 or coverage < .9),
                fully_silent=bool(silent.all()),
                fully_abstaining=bool(ties.all()))


def independent_description(v, y):
    result = count_metrics(v['counts'], y)
    spikes = v['spikes']
    groups = {'branch_event_rate': spikes[:, :, :12],
              'hidden_event_rate': spikes[:, :, 12:18],
              'output_event_rate': spikes[:, :, 18:20],
              'readout_hidden_event_rate': spikes[:, 16:, 12:18],
              'readout_output_event_rate': spikes[:, 16:, 18:20]}
    result.update({key: float(value.mean()) for key, value in groups.items()})
    result['output_candidate_margin_quantiles'] = np.quantile(
        v['candidates'][:, 16:, 18:] - .5, [0, .05, .5, .95, 1]).tolist()
    return result


def verify_native(saved, actual, why):
    max_residual = 0.0
    for field in ('spikes', 'counts', 'prediction'):
        check(np.array_equal(saved[field], actual[field]), why + ' exact ' + field)
    check(np.all((actual['spikes'] == 0) | (actual['spikes'] == 1)),
          why + ' hard binary spikes')
    for field in ('states', 'candidates'):
        residual = float(np.max(abs(actual[field] - saved[field])))
        max_residual = max(max_residual, residual)
        check(residual < 1e-10, why + ' scalar replay ' + field)
    return max_residual


def paired_interval(delta, rng_seed, quantiles=(.025, .975)):
    delta = np.asarray(delta, dtype=float)
    check(delta.shape == (104,), '104 independently trained seed blocks')
    generator = np.random.default_rng(rng_seed)
    indices = generator.integers(0, 104, size=(20000, 104))
    means = delta[indices].mean(1)
    return float(delta.mean()), np.quantile(means, quantiles).tolist()


def wilson(k, n=104):
    check(isinstance(k, (int, np.integer)) and 0 <= k <= n,
          'integer event count')
    z = 1.959963984540054
    proportion = k / n
    den = 1 + z*z/n
    center = (proportion + z*z/(2*n)) / den
    half = z / den * np.sqrt(proportion*(1-proportion)/n + z*z/(4*n*n))
    return [max(0.0, float(center-half)), min(1.0, float(center+half))]


def compare(actual, expected, why, atol=1e-12):
    """Recursive comparison, exact categories and tightly bounded float roundoff."""
    if isinstance(expected, dict):
        check(isinstance(actual, dict) and actual.keys() == expected.keys(), why + ' keys')
        for key, value in expected.items():
            compare(actual[key], value, why + '.' + key, atol)
    elif isinstance(expected, list):
        check(isinstance(actual, list) and len(actual) == len(expected), why + ' list')
        for i, value in enumerate(expected):
            compare(actual[i], value, why + '[' + str(i) + ']', atol)
    elif isinstance(expected, bool):
        check(type(actual) is bool and actual == expected, why + ' boolean')
    elif isinstance(expected, (float, np.floating)):
        check(np.isfinite(actual) and abs(actual-expected) <= atol, why + ' numeric')
    else:
        check(actual == expected, why + ' exact')


def check_description(row, native, labels, why):
    description = independent_description(native, labels)
    compare({key: row[key] for key in description}, description, why, atol=1e-10)


def inspect_dataset(out, name, n, expected_seed):
    with np.load(out/name, allow_pickle=False) as file:
        check(set(file.files) == {'x', 'y', 'pair', 'seed'}, 'dataset fields ' + name)
        d = {key: file[key] for key in file.files}
    check(int(d['seed']) == expected_seed, 'prespecified dataset seed ' + name)
    x, y, pair = regenerate_xor(n, expected_seed)
    check(d['x'].shape == (n, 24, 6), 'input shape ' + name)
    check(np.array_equal(d['x'], x) and np.array_equal(d['y'], y)
          and np.array_equal(d['pair'], pair), 'separate delayed XOR generator ' + name)
    check(np.array_equal(np.bincount(pair, minlength=4), np.full(4, n//4)),
          'exact pair balance ' + name)
    check(np.array_equal(y, (pair//2) ^ (pair % 2)), 'XOR target ' + name)
    return d


def audit_training(out, lock, registry):
    rows = lock['rows']
    wanted = {(s, arm, recipe) for s in range(104) for arm in ('POINT',)
              for recipe in ('RATE_0.01', 'RATE_0.03', 'CE_0.01', 'CE_0.03')}
    keys = [(r['seed'], r['arm'], r['recipe']) for r in rows]
    check(len(rows) == len(set(keys)) == 416 and set(keys) == wanted,
          'exactly four hundred sixteen registered fixed-condition training runs')
    check(lock['history_columns'] == ['loss', 'raw_gradient_norm', 'update_norm',
          'preupdate_accuracy', 'preupdate_coverage', 'preupdate_silent_fraction',
          'preupdate_mean_signed_count_margin', 'preupdate_nonsilent_tie_fraction'],
          'training history schema')
    datasets = {f'train_s{s}.npz': inspect_dataset(out, f'train_s{s}.npz', 128,
                2080000000 + 10*s) for s in range(104)}
    checkpoint_updates = registry['checkpoint_updates']
    updates, checkpoints, exact_train_episodes, initial_objectives = 0, 0, 0, 0
    max_residual = 0.0
    for row in rows:
        s, arm, recipe = row['seed'], row['arm'], row['recipe']
        check(row['id'] == f's{s}_{arm}_{recipe}', 'unique fixed-condition training identity')
        expected_kind, rate_text = recipe.split('_')
        expected_lr = float(rate_text)
        check((row['loss'], row['lr'], row['steps']) == (expected_kind, expected_lr, 200),
              'fixed recipe and update budget ' + row['id'])
        check(row['initial_seed'] == 2080020000+s, 'paired initialization seed')
        check(row['data_file'] == f'train_s{s}.npz', 'matched training data per seed pair')
        d = datasets[row['data_file']]
        initial = initialize(2080020000+s)
        check(np.array_equal(initial, np.load(out/row['initial_file'], allow_pickle=False)),
              'exact initialization distribution and shared initial vector')
        check(sha(out/row['file']) == row['sha256'], 'final parameter hash')
        theta = np.load(out/row['file'], allow_pickle=False)
        check(theta.shape == (176,) and np.isfinite(theta).all(), 'finite final parameters')
        with np.load(out/row['trace'], allow_pickle=False) as tr:
            check(set(tr.files) == {'history', 'clipped_gradients', 'train_counts'},
                  'complete training trace fields')
            history, gradients, counts = tr['history'], tr['clipped_gradients'], tr['train_counts']
        check(history.shape == (200, 8) and np.isfinite(history).all(), 'complete finite history')
        check(counts.shape == (200, 128, 2) and np.isfinite(counts).all()
              and np.all((counts >= 0) & (counts <= 8)) and np.array_equal(counts, counts.astype(int)),
              'bounded integer preupdate native count traces')
        weights, update_norms = adam(initial, gradients, row['lr'])
        check(np.array_equal(weights[-1], theta), 'all Adam steps reconstruct exact final weights')
        check(np.allclose(update_norms, history[:, 2], atol=1e-12, rtol=0), 'all update norms')
        check(np.allclose(np.linalg.norm(gradients, axis=1), np.minimum(history[:, 1], 1),
                          atol=1e-12, rtol=0), 'raw norms and clipped gradient norms')
        with np.load(out/row['checkpoints'], allow_pickle=False) as cp:
            check(set(cp.files) == {'updates', 'theta'}, 'checkpoint schema')
            check(np.array_equal(cp['updates'], checkpoint_updates), 'fixed checkpoint updates')
            check(np.array_equal(cp['theta'], weights[checkpoint_updates]),
                  'all checkpoint weights exactly reconstruct')
        for k in range(200):
            cm = count_metrics(counts[k], d['y'])
            expected_history = [loss_from_counts(counts[k], d['y'], weights[k], row['loss']),
                                cm['accuracy'], cm['coverage'], cm['silent_fraction'],
                                cm['mean_signed_count_margin'], cm['nonsilent_tie_fraction']]
            check(np.allclose(history[k, [0, 3, 4, 5, 6, 7]], expected_history,
                              atol=1e-12, rtol=0), 'every count-derived loss/metric history')
        diagnostics = read_json(out/row['diagnostics'])
        check(len(diagnostics) == len(checkpoint_updates), 'nine checkpoint diagnostic rows')
        for j, k in enumerate(checkpoint_updates):
            check(diagnostics[j]['updates'] == k, 'checkpoint diagnostic update alignment')
            native = replay(d['x'], weights[k], arm)
            check_description(diagnostics[j], native, d['y'], 'checkpoint native metrics')
            if k < 200:
                check(np.array_equal(native['counts'], counts[k]),
                      'sampled native checkpoint counts match perupdate traces')
                expected_loss = loss_from_counts(native['counts'], d['y'], weights[k], row['loss'])
                check(abs(expected_loss-history[k, 0]) < 1e-12,
                      'sampled checkpoint independent native objective')
                if k == 0:
                    initial_objectives += 1
            else:
                with np.load(out/row['train_native'], allow_pickle=False) as saved:
                    check(set(saved.files) == {'counts', 'prediction', 'spikes', 'states', 'candidates'},
                          'complete final training trajectory')
                    max_residual = max(max_residual, verify_native(saved, native, 'final training'))
                check_description(row, native, d['y'], 'final training row')
                exact_train_episodes += len(d['y'])
            checkpoints += 1
        updates += 200
    return datasets, dict(trajectories=416, reconstructed_updates=updates,
        native_training_checkpoints=checkpoints, native_training_checkpoint_episodes=checkpoints*128,
        full_saved_training_trajectory_episodes=exact_train_episodes,
        independently_replayed_initial_objectives=initial_objectives,
        count_derived_history_rows=updates, max_training_state_candidate_residual=max_residual)


def interval_record(delta, analysis_seed, primary=False):
    values = np.asarray(delta, dtype=float)
    mean, ci = paired_interval(values, analysis_seed)
    result = dict(mean=mean, percentile95=ci, paired_values=values.tolist(),
                  n_pairs=104, unit='independently trained seed/data block paired across four POINT cells',
                  degenerate=bool(np.ptp(values) == 0), primary_accuracy=primary)
    if primary:
        # Derive the critical value independently from SciPy's t distribution.
        sd = float(np.std(values, ddof=1))
        se = sd / np.sqrt(104)
        critical = float(student_t.ppf(.975, 103))
        result.update(student_t95=[float(mean-critical*se), float(mean+critical*se)],
                      sample_sd=sd, standard_error=float(se), degrees_of_freedom=103,
                      t_critical=critical)
    return result


def factorial_effects(blocks):
    """Separate four-cell arithmetic, never registry matrix multiplication."""
    rate_low, rate_high, ce_low, ce_high = blocks
    loss_low, loss_high = ce_low-rate_low, ce_high-rate_high
    lr_rate, lr_ce = rate_high-rate_low, ce_high-ce_low
    return {'POINT_loss_lr_interaction': lr_ce-lr_rate,
            'POINT_loss_main': (loss_low+loss_high)/2,
            'POINT_lr_main': (lr_rate+lr_ce)/2,
            'POINT_loss_at_0.01': loss_low,
            'POINT_loss_at_0.03': loss_high,
            'POINT_lr_at_RATE': lr_rate,
            'POINT_lr_at_CE': lr_ce}


def primary_names():
    return ['POINT_loss_lr_interaction']


def inspect_contrasts():
    """Check frozen coefficients and primary distribution independently."""
    recipes = ('RATE_0.01', 'RATE_0.03', 'CE_0.01', 'CE_0.03')
    coefficients = factorial_effects(np.eye(4))
    expected = dict(columns=['POINT|'+recipe for recipe in recipes],
        contrasts=[dict(name=name, coefficients=value.tolist(),
                        primary_accuracy=name in primary_names())
                   for name, value in coefficients.items()],
        primary_accuracy_family_size=1, n_pairs=104,
        primary_method='two-sided paired-mean Student-t interval',
        primary_confidence=.95, primary_df=103,
        primary_t_critical=float(student_t.ppf(.975, 103)),
        bootstrap_resamples=20000, bootstrap_quantiles=[.025, .975],
        practical_threshold=-.05)
    compare(read_json(R/'CONTRASTS.json'), expected, 'factorial contrast registry', atol=1e-14)
    check(len(coefficients) == 7, 'seven preregistered coefficient vectors')
    check(all(np.sum(c) == 0 for c in coefficients.values()), 'all contrasts sum to zero')
    check(abs(student_t.cdf(expected['primary_t_critical'], 103)-.975) < 1e-12,
          'primary t critical value distribution check')


def independent_statistics(testrows, trainrows, analysis_seed):
    def values(recipe, field, rows):
        by_seed = {r['seed']: r[field] for r in rows
                   if r['arm'] == 'POINT' and r['recipe'] == recipe}
        check(set(by_seed) == set(range(104)), '104 complete seed units per cell')
        return np.asarray([by_seed[s] for s in range(104)], dtype=float)
    intervals, cells = {}, []
    recipes = ('RATE_0.01', 'RATE_0.03', 'CE_0.01', 'CE_0.03')
    for metric in ('accuracy', 'failed'):
        blocks = np.stack([values(recipe, metric, testrows) for recipe in recipes])
        for name, delta in factorial_effects(blocks).items():
            intervals[metric+'_'+name] = interval_record(
                delta, analysis_seed, primary=metric == 'accuracy' and name in primary_names())
    for recipe in recipes:
        test = {r['seed']: r for r in testrows if r['arm'] == 'POINT' and r['recipe'] == recipe}
        train = {r['seed']: r for r in trainrows if r['arm'] == 'POINT' and r['recipe'] == recipe}
        rates = {}
        for event in ('failed', 'fully_silent', 'fully_abstaining'):
            k = sum(int(test[s][event]) for s in range(104))
            ci = wilson(k)
            check(ci[0] < ci[1], 'Wilson interval remains nondegenerate at zero/all events')
            rates[event] = dict(count=k, fraction=k/104, wilson95=ci)
        transition = {}
        for train_pass in (False, True):
            for test_pass in (False, True):
                key = f'train_{train_pass}_test_{test_pass}'
                transition[key] = sum((not train[s]['failed']) == train_pass
                                      and (not test[s]['failed']) == test_pass for s in range(104))
        check(sum(transition.values()) == 104, 'complete train-to-test pass transitions')
        def mean(key, source=test):
            return float(np.mean([source[s][key] for s in range(104)]))
        cells.append(dict(arm='POINT', recipe=recipe, n=104,
            mean_accuracy=mean('accuracy'), mean_coverage=mean('coverage'),
            mean_silent_fraction=mean('silent_fraction'), mean_abstention_rate=mean('abstention_rate'),
            mean_nonsilent_tie_fraction=mean('nonsilent_tie_fraction'),
            mean_train_accuracy=mean('accuracy', train), run_events=rates, pass_transitions=transition))
        key = f'POINT_{recipe}_test_minus_train_accuracy'
        intervals[key] = interval_record(values(recipe, 'accuracy', testrows)
                                        - values(recipe, 'accuracy', trainrows), analysis_seed)
    primary = intervals['accuracy_POINT_loss_lr_interaction']
    lower, upper = primary['student_t95']
    sensitivity_lower, sensitivity_upper = primary['percentile95']
    interpretation = dict(primary_interval_method='two-sided paired-mean Student-t95',
        negative_direction_supported=bool(upper < 0),
        opposite_direction_supported=bool(lower > 0),
        negative_magnitude_beyond_5pp_supported=bool(upper < -.05),
        bootstrap_negative_direction=bool(sensitivity_upper < 0),
        bootstrap_opposite_direction_supported=bool(sensitivity_lower > 0),
        bootstrap_direction_disagrees=bool(((upper < 0) != (sensitivity_upper < 0))
                                            or ((lower > 0) != (sensitivity_lower > 0))),
        limits='Primary t interval governs interpretation; bootstrap is prespecified sensitivity only. No equivalence from zero crossing or degenerate intervals. New-data internal replication, not external independent replication.')
    return dict(primary_family=['accuracy_POINT_loss_lr_interaction'],
                intervals=intervals, cells=cells, primary_interpretation=interpretation)


def audit_tests(out, trainrows, registry):
    summary = read_json(out/'TEST_SUMMARY.json')
    check(summary['train_lock_sha256'] == sha(out/'TRAIN_LOCK.json'), 'test summary checkpoint lock')
    check(read_json(out/'TEST_STARTED.json')['train_lock_sha256'] == sha(out/'TRAIN_LOCK.json'),
          'test start checkpoint lock')
    for name, digest in summary['hashes'].items():
        check(sha(out/name) == digest, 'test artifact hash ' + name)
    rows = summary['rows']
    check(rows == read_json(out/'TEST_LOG.json'), 'final test log complete')
    check(len(rows) == 416 and len({r['id'] for r in rows}) == 416, 'four hundred sixteen unique test evaluations')
    check({r['id'] for r in rows} == {r['id'] for r in trainrows}, 'no missing or selected test run')
    datasets = {f'test_s{s}.npz': inspect_dataset(out, f'test_s{s}.npz', 512,
                2080010000 + 10*s) for s in range(104)}
    train_index = {r['id']: r for r in trainrows}
    max_residual = 0.0
    for row in rows:
        source = train_index[row['id']]
        for key in ('seed', 'arm', 'recipe'):
            check(row[key] == source[key], 'test condition identity ' + key)
        check(row['source_file'] == source['file'], 'exact final test weight provenance')
        check(row['data_file'] == f"test_s{row['seed']}.npz", 'matched held-out data per seed')
        d = datasets[row['data_file']]
        theta = np.load(out/source['file'], allow_pickle=False)
        native = replay(d['x'], theta, row['arm'])
        with np.load(out/row['file'], allow_pickle=False) as saved:
            check(set(saved.files) == {'counts', 'prediction', 'spikes', 'states', 'candidates'},
                  'complete held-out native trajectory')
            max_residual = max(max_residual, verify_native(saved, native, 'held-out test'))
        check_description(row, native, d['y'], 'held-out test metrics')
    expected_hashes = set(datasets) | {r['file'] for r in rows}
    check(set(summary['hashes']) == expected_hashes, 'exact test data/trajectory inventory')
    check(summary['unique_test_histories'] == 53248 and summary['model_test_episodes'] == 212992,
          'histories are distinct from repeated model evaluations')
    actual = summary['statistics']
    expected = independent_statistics(rows, trainrows, registry['analysis_seed'])
    compare({key: actual[key] for key in expected}, expected, 'all seed-unit statistics', atol=1e-13)
    check(len(actual['intervals']) == 18 and len(actual['cells']) == 4, 'registered comparisons and cells')
    check(sum('student_t95' in v for v in actual['intervals'].values()) == 1,
          'exactly one primary Student-t accuracy interval')
    return datasets, dict(test_models=416, test_model_episodes=212992,
        unique_test_histories=53248, max_test_state_candidate_residual=max_residual,
        paired_seed_intervals=18, student_t_primary_intervals=1, wilson_run_event_intervals=12)


def assert_disjoint(datasets):
    seen = set()
    for name, d in datasets.items():
        for x in d['x']:
            digest = hashlib.sha256(np.ascontiguousarray(x).tobytes()).digest()
            check(digest not in seen, 'no exact episode duplication across/within split ' + name)
            seen.add(digest)
    return len(seen)


def main():
    start = time.perf_counter()
    out = R/'results'
    freeze = read_json(R/'FREEZE.json')
    for name, digest in freeze['sha256'].items():
        check(sha(R/name) == digest, 'frozen source ' + name)
    for name, digest in freeze.get('dependencies', {}).items():
        check(sha(R.parent/name) == digest, 'frozen dependency ' + name)
    check(sha(R/'model.py') == sha(R.parent/'dendritic-population-v7/model.py'),
          'native model byte-identical to V7')
    check(sha(R/'data.py') == sha(R.parent/'dendritic-population-v7/data.py'),
          'task generator byte-identical to V7')
    inspect_contrasts()
    registry = read_json(R/'SEEDS.json')
    expected_pairs = [dict(seed=s, train_seed=2080000000+10*s,
                          test_seed=2080010000+10*s, initial_seed=2080020000+s) for s in range(104)]
    check(registry['n_pairs'] == 104, 'fixed sample count without optional stopping')
    check(registry['pairs'] == expected_pairs, 'prespecified 104 fresh paired seed namespaces')
    check(registry['analysis_seed'] == 2080030001, 'fixed analysis RNG seed')
    check(registry['checkpoint_updates'] == [0, 1, 5, 10, 20, 50, 100, 150, 200],
          'fixed nine diagnostic checkpoints')
    check(registry['sampled_updates'] == [0, 99, 199], 'fixed sampled derivative replay indices')
    all_seeds = [v for row in expected_pairs for key, v in row.items() if key != 'seed']
    check(len(set(all_seeds)) == 312, 'nonoverlapping training/test/initialization seed namespaces')
    lock = read_json(out/'TRAIN_LOCK.json')
    check(lock['freeze_sha256'] == sha(R/'FREEZE.json'), 'training freeze lock identity')
    check(lock['test_generated'] is False, 'training lock predates held-out generation')
    marker = read_json(out/'TRAIN_STARTED.json')
    check(marker['freeze_sha256'] == sha(R/'FREEZE.json')
          and marker['seed_registry_sha256'] == sha(R/'SEEDS.json'), 'training start immutable identities')
    for name, digest in lock['files'].items():
        check(sha(out/name) == digest, 'locked training artifact ' + name)
    check(lock['rows'] == read_json(out/'TRAIN_LOG.json'), 'all completed training rows locked')
    training_data, training = audit_training(out, lock, registry)
    test_data, tests = audit_tests(out, lock['rows'], registry)
    distinct = assert_disjoint({**training_data, **test_data})
    check(distinct == 66560, '104 independent 128-train/512-test split pairs')
    result = dict(status='PASS', checks=checks, training=training, tests=tests,
        distinct_training_test_histories=distinct, seconds=time.perf_counter()-start,
        scope='Separate delayed-XOR RNG, initialization, scalar native equations, metrics, factorial cell arithmetic, seed-unit bootstrap sensitivity, single primary Student-t interval and decision flags, and Wilson implementation. All saved-gradient Adam steps/checkpoints reconstruct; every stored count-derived loss/metric and all sampled checkpoint native metrics match. Full native spikes/counts/predictions replay for final train and test. Native trajectories are not re-evaluated at every unsampled training update. Surrogate gradients are checked separately. Internal workflow verification is not outside replication or peer review.')
    (out/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
