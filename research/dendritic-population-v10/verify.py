"""Independent V10 finite binding/size-study audit.

No scientific model, dataset, training or analysis imports. The checker derives
RNG draws, scalar native equations, metrics, seed intervals and all saved-gradient
Adam updates separately. It does not independently rederive all surrogate
gradients. This is internal workflow checking, not external replication.
"""
from pathlib import Path
import hashlib
import json
import time
import subprocess
import numpy as np

R = Path(__file__).resolve().parent
ARMS = ('DEND', 'POINT')
SIZES = (6, 4, 2, 1)
CHECKPOINTS = [0, 1, 2, 5, 10, 25, 50, 100, 200]
NATIVE_FIELDS = {'counts', 'prediction', 'spikes', 'states', 'candidates'}
checks = 0
audit_deadline = None


def check(ok, why):
    global checks
    checks += 1
    if audit_deadline is not None and time.perf_counter() > audit_deadline:
        raise TimeoutError('V10 independent audit exceeded frozen 300s cap; no automatic retry')
    if not ok:
        raise AssertionError(why)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def compare(actual, expected, why, atol=1e-12):
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
        check(np.isfinite(actual) and abs(actual - expected) <= atol, why + ' numeric')
    else:
        check(actual == expected, why + ' exact')


def parameter_count(h):
    return 6 * (2 * h) + h * (2 * h) + 2 * h + h + h * 2 + 2


def initialize(seed, h):
    rng = np.random.default_rng(seed)
    return np.concatenate((rng.normal(0, .45, 12*h),
        rng.normal(0, .25, 2*h*h), np.full(2*h, .15), np.full(h, .05),
        rng.normal(0, .5, 2*h), np.full(2, .1)))


def regenerate_binding(n, seed):
    """Independent per-episode/time construction, same prespecified draw order."""
    check(n % 16 == 0, 'integer replicates of all sixteen input patterns')
    rng = np.random.default_rng(seed)
    pattern = np.array([p for p in range(16) for _ in range(n//16)])
    rng.shuffle(pattern)
    high = rng.uniform(.8, 1.2, (n, 4))
    low = rng.uniform(0, .1, (n, 4))
    x = np.zeros((n, 24, 6))
    y = np.empty(n, dtype=int)
    for episode, p in enumerate(pattern):
        bit = [int(p) // (2**j) % 2 for j in range(4)]
        y[episode] = int((bit[0] == bit[1] == 1) or (bit[2] == bit[3] == 1))
        for j in range(4):
            for step in range(8, 16):
                x[episode, step, j] = high[episode, j] if bit[j] else low[episode, j]
        for step in range(16, 24):
            x[episode, step, 4] = 1
    nuisance = rng.uniform(0, .5, (n, 3))
    for step in range(7, 10):
        x[:, step, 5] = nuisance[:, step-7]
    noise = rng.normal(0, .03, x.shape)
    return np.maximum(x + noise, 0), y, pattern


def replay(x, theta, arm, h):
    """Scalar connection sums with vectorization only over independent episodes."""
    check(arm in ARMS and theta.shape == (parameter_count(h),), 'native model dimensions')
    bcount, width = 2*h, 3*h+2
    offset = 12*h
    wi = theta[:offset].reshape(6, bcount)
    wr = theta[offset:offset+2*h*h].reshape(h, bcount)
    offset += 2*h*h
    db = theta[offset:offset+bcount]
    offset += bcount
    vb = theta[offset:offset+h]
    offset += h
    wo = theta[offset:offset+2*h].reshape(h, 2)
    ob = theta[offset+2*h:]
    n = len(x)
    state, previous = np.zeros((n, width)), np.zeros((n, h))
    states, candidates, event_arrays = [], [], []
    for step in range(24):
        cand, spike = np.zeros((n, width)), np.zeros((n, width))
        next_state, local = np.zeros_like(state), np.zeros((n, bcount))
        for branch in range(bcount):
            drive, recurrent = np.zeros(n), np.zeros(n)
            for channel in range(6):
                drive += x[:, step, channel] * wi[channel, branch]
            for cell in range(h):
                recurrent += previous[:, cell] * wr[cell, branch]
            cand[:, branch] = .85*state[:, branch] + .25*(drive+recurrent+db[branch])
            if arm == 'POINT':
                spike[:, branch] = cand[:, branch] > .5
                local[:, branch] = spike[:, branch]
                next_state[:, branch] = cand[:, branch]*(1-spike[:, branch])
            else:
                local[:, branch] = np.maximum(cand[:, branch], 0)
                next_state[:, branch] = cand[:, branch]
        for cell in range(h):
            j = bcount+cell
            cand[:, j] = .65*state[:, j] + .5*(local[:, 2*cell]+local[:, 2*cell+1]) + vb[cell]
            spike[:, j] = cand[:, j] > .5
            next_state[:, j] = cand[:, j]*(1-spike[:, j])
        for output in range(2):
            drive = np.zeros(n)
            for cell in range(h):
                drive += previous[:, cell]*wo[cell, output]
            j = 3*h+output
            cand[:, j] = .65*state[:, j]+drive+ob[output]
            spike[:, j] = cand[:, j] > .5
            next_state[:, j] = cand[:, j]*(1-spike[:, j])
        previous = spike[:, bcount:3*h]
        state = next_state
        states.append(state)
        candidates.append(cand)
        event_arrays.append(spike)
    spikes = np.stack(event_arrays, 1)
    counts = spikes[:, 16:, 3*h:].sum(1)
    prediction = np.full(n, -1, dtype=int)
    prediction[counts[:, 0] > counts[:, 1]] = 0
    prediction[counts[:, 1] > counts[:, 0]] = 1
    return dict(states=np.stack(states, 1), candidates=np.stack(candidates, 1),
                spikes=spikes, counts=counts, prediction=prediction)


def adam(initial, gradients, lr):
    check(gradients.shape == (200, len(initial)), 'complete clipped-gradient dimensions')
    weights = [initial.copy()]
    theta = initial.copy()
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    norms = []
    for step, g in enumerate(gradients, start=1):
        check(np.isfinite(g).all() and np.linalg.norm(g) <= 1+1e-12, 'finite global gradient clip')
        m = .9*m + .1*g
        v = .999*v + .001*g*g
        new = np.clip(theta-lr*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8), -2, 2)
        norms.append(float(np.linalg.norm(new-theta)))
        theta = new
        weights.append(theta.copy())
    return np.stack(weights), np.array(norms)


def rate_objective(counts, labels, theta):
    # The two targets are .75 for the true class and .125 for the other.
    correct = counts[np.arange(len(labels)), labels]/8-.75
    other = counts[np.arange(len(labels)), 1-labels]/8-.125
    return float((np.sum(correct*correct)+np.sum(other*other))/(2*len(labels))
                 + .0001*np.mean(theta*theta))


def count_metrics(counts, labels):
    n = len(labels)
    pred = np.full(n, -1, dtype=int)
    pred[counts[:, 0] > counts[:, 1]] = 0
    pred[counts[:, 1] > counts[:, 0]] = 1
    matrix = np.zeros((2, 3), dtype=int)
    for truth, guess in zip(labels, pred):
        matrix[int(truth), int(guess) if guess >= 0 else 2] += 1
    recall = [float(matrix[k, k]/np.sum(labels == k)) for k in (0, 1)]
    accuracy = float(np.mean(pred == labels))
    balanced = float((recall[0]+recall[1])/2)
    coverage = float(np.mean(pred >= 0))
    silence = np.logical_and(counts[:, 0] == 0, counts[:, 1] == 0)
    ties = pred < 0
    margins = counts[np.arange(n), labels]-counts[np.arange(n), 1-labels]
    return dict(accuracy=accuracy, balanced_accuracy=balanced, coverage=coverage,
        abstentions=int(ties.sum()), abstention_rate=float(ties.mean()),
        silent_fraction=float(silence.mean()), nonsilent_tie_fraction=float((ties & ~silence).mean()),
        both_saturated_fraction=float(np.all(counts == 8, axis=1).mean()),
        class_recall=recall, confusion=matrix.tolist(),
        mean_signed_count_margin=float(margins.mean()), median_signed_count_margin=float(np.median(margins)),
        signed_count_margin_histogram=[int(np.sum(margins == k)) for k in range(-8, 9)],
        failed=bool(balanced < .9 or coverage < .9), fully_silent=bool(silence.all()),
        fully_abstaining=bool(ties.all()))


def independent_description(native, labels, pattern, h):
    result = count_metrics(native['counts'], labels)
    spikes, states, candidates = native['spikes'], native['states'], native['candidates']
    n = len(labels)
    group_bounds = [('branch', 0, 2*h), ('hidden', 2*h, 3*h), ('output', 3*h, 3*h+2)]
    events = {}
    for name, lo, hi in group_bounds:
        group = spikes[:, :, lo:hi]
        events[name] = float(group.sum((1, 2)).mean())
        result[name+'_event_rate'] = float(group.mean())
    variation = np.zeros(n)
    previous = np.zeros((n, 3*h+2))
    for step in range(24):
        variation += np.abs(states[:, step]-previous).sum(1)
        previous = states[:, step]
    prefix_predictions = np.full((n, 8), -1, dtype=int)
    for deadline in range(8):
        prefix = spikes[:, 16:17+deadline, 3*h:].sum(1)
        prefix_predictions[prefix[:, 0] > prefix[:, 1], deadline] = 0
        prefix_predictions[prefix[:, 1] > prefix[:, 0], deadline] = 1
    final = prefix_predictions[:, 7]
    settled, correct = np.zeros((n, 8), bool), np.zeros((n, 8), bool)
    for deadline in range(8):
        settled[:, deadline] = np.all(prefix_predictions[:, deadline:] == final[:, None], axis=1) & (final >= 0)
        correct[:, deadline] = settled[:, deadline] & (final == labels)
    latency = np.full(n, 9, dtype=int)
    for episode in range(n):
        possible = np.flatnonzero(correct[episode])
        if len(possible):
            latency[episode] = int(possible[0])+1
    result.update(spike_events_mean=events,
        total_spike_events_mean=float(sum(events.values())),
        state_l1_change_mean=float(variation.mean()),
        dense_event_delivery_proxy_mean=float(events['branch']+(2*h+2)*events['hidden']+events['output']),
        correct_settled_fraction_by_step=correct.mean(0).tolist(),
        unique_settled_fraction_by_step=settled.mean(0).tolist(),
        capped_correct_settled_latency_mean=float(latency.mean()), latency_failure_cap=9,
        latency_scope='Retrospective prefix-count settling within readout steps1..8; wrong/abstaining episodes receive9. No online stopping, no label-conditioned dynamics. Activity is not energy.',
        pattern_recall=[float(np.mean(native['prediction'][pattern == p] == labels[pattern == p])) for p in range(16)],
        readout_hidden_event_rate=float(spikes[:, 16:, 2*h:3*h].mean()),
        readout_output_event_rate=float(spikes[:, 16:, 3*h:].mean()),
        output_candidate_margin_quantiles=np.quantile(candidates[:, 16:, 3*h:]-.5, [0, .05, .5, .95, 1]).tolist())
    check(np.all(np.diff(correct.mean(0)) >= 0), 'settled-correct deadline curve monotone')
    check(float(correct[:, -1].mean()) == result['accuracy'], 'last settled-correct deadline equals unconditional accuracy')
    check(float(settled[:, -1].mean()) == result['coverage'], 'last unique-settled deadline equals coverage')
    return result


def verify_description(row, native, labels, pattern, h, why):
    expected = independent_description(native, labels, pattern, h)
    compare({key: row[key] for key in expected}, expected, why, atol=1e-10)


def verify_native(saved, expected, why):
    check(set(saved.files) == NATIVE_FIELDS, why+' native fields')
    residual = 0.0
    for key in ('spikes', 'counts', 'prediction'):
        check(np.array_equal(saved[key], expected[key]), why+' exact '+key)
    check(np.all((expected['spikes'] == 0) | (expected['spikes'] == 1)), why+' binary native spikes')
    for key in ('states', 'candidates'):
        check(saved[key].shape == expected[key].shape and np.isfinite(saved[key]).all(), why+' finite '+key)
        value = float(np.max(np.abs(saved[key]-expected[key])))
        residual = max(residual, value)
        check(value < 1e-10, why+' scalar replay '+key)
    return residual


def inspect_dataset(out, name, n, expected_seed):
    with np.load(out/name, allow_pickle=False) as stored:
        check(set(stored.files) == {'x', 'y', 'pattern', 'seed'}, name+' dataset fields')
        data = {key: stored[key] for key in stored.files}
    check(int(data['seed']) == expected_seed, name+' seed')
    x, labels, pattern = regenerate_binding(n, expected_seed)
    check(data['x'].shape == (n, 24, 6), name+' shape')
    check(np.array_equal(data['x'], x) and np.array_equal(data['y'], labels)
          and np.array_equal(data['pattern'], pattern), name+' independent exact regeneration')
    check(np.array_equal(np.bincount(pattern, minlength=16), np.full(16, n//16)), name+' exact pattern balance')
    check(np.array_equal(np.bincount(labels, minlength=2), np.array([9, 7])*(n//16)), name+' seven positive and nine negative patterns')
    check(np.all(labels[pattern == 15] == 1), name+' all-four pattern positive')
    check(np.isfinite(x).all() and np.min(x) >= 0, name+' finite nonnegative currents')
    return data


def audit_training(out, lock):
    rows = lock['rows']
    keys = [(r['seed'], r['arm'], r['h']) for r in rows]
    wanted = {(s, arm, h) for s in range(5) for arm in ARMS for h in SIZES}
    check(len(rows) == len(set(keys)) == 40 and set(keys) == wanted, 'all forty fixed training runs retained')
    check(lock['history_columns'] == ['loss', 'raw_gradient_norm', 'update_norm',
        'preupdate_accuracy', 'preupdate_coverage', 'preupdate_silent_fraction',
        'preupdate_mean_signed_count_margin', 'preupdate_nonsilent_tie_fraction',
        'preupdate_balanced_accuracy'], 'history schema')
    datasets = {f'train_s{s}.npz': inspect_dataset(out, f'train_s{s}.npz', 128, 2100000000+10*s) for s in range(5)}
    inventory = set(datasets) | {'TRAIN_STARTED.json', 'TRAIN_LOG.json'}
    for row in rows:
        inventory.update(row[k] for k in ('initial_file', 'file', 'trace', 'checkpoints', 'diagnostics', 'train_native'))
    check(set(lock['files']) == inventory, 'complete locked training artifact inventory')
    updates, checkpoints, final_episodes, initial_objectives = 0, 0, 0, 0
    max_residual = 0.0
    for row in rows:
        s, arm, h = row['seed'], row['arm'], row['h']
        run_id = f's{s}_{arm}_h{h}'
        check(row['id'] == run_id, 'run identity')
        check((row['loss'], row['lr'], row['steps']) == ('RATE', .01, 200), run_id+' fixed recipe')
        initial_seed = 2100020000+100*h+s
        check(row['initial_seed'] == initial_seed, run_id+' paired initialization seed')
        check(row['data_file'] == f'train_s{s}.npz', run_id+' shared training data')
        data = datasets[row['data_file']]
        initial = initialize(initial_seed, h)
        check(np.array_equal(initial, np.load(out/row['initial_file'], allow_pickle=False)), run_id+' initialization')
        check(sha(out/row['file']) == row['sha256'], run_id+' final parameter hash')
        theta = np.load(out/row['file'], allow_pickle=False)
        check(theta.shape == (parameter_count(h),) and np.isfinite(theta).all() and np.max(abs(theta)) <= 2, run_id+' final parameters')
        with np.load(out/row['trace'], allow_pickle=False) as trace:
            check(set(trace.files) == {'history', 'clipped_gradients', 'train_counts'}, run_id+' trace fields')
            history, gradients, counts = trace['history'], trace['clipped_gradients'], trace['train_counts']
        check(history.shape == (200, 9) and np.isfinite(history).all(), run_id+' finite full history')
        check(counts.shape == (200, 128, 2) and np.isfinite(counts).all() and np.all((counts >= 0) & (counts <= 8))
              and np.array_equal(counts, counts.astype(int)), run_id+' bounded integer count traces')
        weights, norms = adam(initial, gradients, .01)
        check(np.array_equal(weights[-1], theta), run_id+' all Adam steps exact')
        check(np.allclose(norms, history[:, 2], atol=1e-12, rtol=0), run_id+' every update norm')
        check(np.allclose(np.linalg.norm(gradients, axis=1), np.minimum(history[:, 1], 1), atol=1e-12, rtol=0), run_id+' clipped gradient norms')
        with np.load(out/row['checkpoints'], allow_pickle=False) as cp:
            check(set(cp.files) == {'updates', 'theta'}, run_id+' checkpoint fields')
            check(np.array_equal(cp['updates'], CHECKPOINTS), run_id+' checkpoint schedule')
            check(np.array_equal(cp['theta'], weights[CHECKPOINTS]), run_id+' all checkpoint weights')
        for update in range(200):
            metric = count_metrics(counts[update], data['y'])
            expected = [rate_objective(counts[update], data['y'], weights[update]),
                metric['accuracy'], metric['coverage'], metric['silent_fraction'],
                metric['mean_signed_count_margin'], metric['nonsilent_tie_fraction'], metric['balanced_accuracy']]
            check(np.allclose(history[update, [0, 3, 4, 5, 6, 7, 8]], expected, atol=1e-12, rtol=0), run_id+' all count-derived histories')
        diagnostics = read_json(out/row['diagnostics'])
        check(len(diagnostics) == len(CHECKPOINTS), run_id+' all nine checkpoint diagnostics')
        for j, update in enumerate(CHECKPOINTS):
            check(diagnostics[j]['updates'] == update, run_id+' diagnostic update identity')
            native = replay(data['x'], weights[update], arm, h)
            verify_description(diagnostics[j], native, data['y'], data['pattern'], h, run_id+' checkpoint metrics')
            if update < 200:
                check(np.array_equal(native['counts'], counts[update]), run_id+' native checkpoint count trace')
                check(abs(rate_objective(native['counts'], data['y'], weights[update])-history[update, 0]) < 1e-12,
                      run_id+' native checkpoint objective')
                initial_objectives += int(update == 0)
            else:
                with np.load(out/row['train_native'], allow_pickle=False) as saved:
                    max_residual = max(max_residual, verify_native(saved, native, run_id+' final train'))
                verify_description(row, native, data['y'], data['pattern'], h, run_id+' final training row')
                final_episodes += len(data['y'])
            checkpoints += 1
        updates += 200
    return datasets, dict(trajectories=40, reconstructed_updates=updates,
        native_training_checkpoints=checkpoints, native_training_checkpoint_episodes=checkpoints*128,
        full_saved_training_trajectory_episodes=final_episodes,
        independently_replayed_initial_objectives=initial_objectives,
        count_derived_history_rows=updates, max_training_state_candidate_residual=max_residual)


def bootstrap_record(values, seed):
    values = np.asarray(values, dtype=float)
    check(values.shape == (5,), 'five paired seed units')
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, 5, size=(20000, 5))
    bootstrap_means = values[indices].sum(1)/5
    return dict(mean=float(values.mean()), percentile95=np.quantile(bootstrap_means, [.025, .975]).tolist(),
        paired_values=values.tolist(), n_pairs=5, degenerate=bool(np.ptp(values) == 0),
        unit='paired training/data seed block')


def wilson(k):
    check(type(k) is int and 0 <= k <= 5, 'five-run event count')
    z, n = 1.959963984540054, 5
    p = k/n
    den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    half = z/den*np.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return [float(max(0, center-half)), float(min(1, center+half))]


def independent_statistics(rows, analysis_seed):
    cells, intervals = [], {}
    for h in SIZES:
        by_arm = {}
        for arm in ARMS:
            by_seed = {r['seed']: r for r in rows if r['arm'] == arm and r['h'] == h}
            check(set(by_seed) == set(range(5)), 'all five seed units in cell')
            ordered = [by_seed[s] for s in range(5)]
            def avg(key):
                return float(np.mean([r[key] for r in ordered]))
            failures = sum(int(r['failed']) for r in ordered)
            by_arm[arm] = np.array([r['balanced_accuracy'] for r in ordered])
            cells.append(dict(arm=arm, h=h, n=5, mean_accuracy=avg('accuracy'),
                mean_balanced_accuracy=avg('balanced_accuracy'), mean_coverage=avg('coverage'),
                failures=failures, failure_wilson95=wilson(failures),
                fully_silent=sum(int(r['fully_silent']) for r in ordered),
                fully_abstaining=sum(int(r['fully_abstaining']) for r in ordered),
                mean_total_spikes=avg('total_spike_events_mean'), mean_state_l1_change=avg('state_l1_change_mean'),
                mean_capped_correct_settled_latency=avg('capped_correct_settled_latency_mean'),
                correct_settled_fraction_by_step=np.mean([r['correct_settled_fraction_by_step'] for r in ordered], axis=0).tolist()))
        intervals[f'h{h}_DEND_minus_POINT_balanced_accuracy'] = bootstrap_record(by_arm['DEND']-by_arm['POINT'], analysis_seed)
    return dict(primary_family=list(intervals), cells=cells, intervals=intervals,
        uncertainty='Five paired seed blocks; four pointwise exploratory percentile intervals without familywise protection. Zero-crossing/degenerate intervals do not establish equivalence. No best-size selection or hyperparameter search.',
        interpretation='Learnt binding/size comparison with strong spiking branch baseline. No pooling intervention; this does not isolate causal necessity of local nonlinear partition, prove representation impossibility, or establish physical energy efficiency.')


def audit_tests(out, trainrows, registry):
    summary = read_json(out/'TEST_SUMMARY.json')
    check(summary['train_lock_sha256'] == sha(out/'TRAIN_LOCK.json'), 'test checkpoint lock')
    check(read_json(out/'TEST_STARTED.json')['train_lock_sha256'] == sha(out/'TRAIN_LOCK.json'), 'test-start checkpoint lock')
    for name, digest in summary['hashes'].items():
        check(sha(out/name) == digest, 'test artifact hash '+name)
    rows = summary['rows']
    check(rows == read_json(out/'TEST_LOG.json'), 'test log matches summary')
    check(len(rows) == len({r['id'] for r in rows}) == 40, 'forty unique held-out evaluations')
    check({r['id'] for r in rows} == {r['id'] for r in trainrows}, 'no selection of final test runs')
    datasets = {f'test_s{s}.npz': inspect_dataset(out, f'test_s{s}.npz', 512, 2100010000+10*s) for s in range(5)}
    train_index = {r['id']: r for r in trainrows}
    max_residual = 0.0
    for row in rows:
        source = train_index[row['id']]
        for key in ('seed', 'arm', 'h'):
            check(row[key] == source[key], 'test condition identity '+key)
        check(row['source_file'] == source['file'], 'exact frozen test parameter provenance')
        check(row['data_file'] == f"test_s{row['seed']}.npz", 'paired held-out data')
        data = datasets[row['data_file']]
        theta = np.load(out/source['file'], allow_pickle=False)
        native = replay(data['x'], theta, row['arm'], row['h'])
        with np.load(out/row['file'], allow_pickle=False) as saved:
            max_residual = max(max_residual, verify_native(saved, native, row['id']+' test'))
        verify_description(row, native, data['y'], data['pattern'], row['h'], row['id']+' held-out metrics')
    check(set(summary['hashes']) == set(datasets) | {r['file'] for r in rows}, 'complete held-out inventory')
    check(summary['unique_test_histories'] == 2560 and summary['model_test_episodes'] == 20480, 'distinct test histories and model episodes')
    expected = independent_statistics(rows, registry['analysis_seed'])
    compare({key: summary['statistics'][key] for key in expected}, expected, 'all seed statistics', atol=1e-13)
    check(len(summary['statistics']['intervals']) == 4 and len(summary['statistics']['cells']) == 8, 'four exploratory comparisons and eight cells')
    return datasets, dict(test_models=40, test_model_episodes=20480, unique_test_histories=2560,
        max_test_state_candidate_residual=max_residual, paired_seed_intervals=4, wilson_failure_intervals=8)


def assert_disjoint(datasets):
    seen = set()
    for name, data in datasets.items():
        for episode in data['x']:
            digest = hashlib.sha256(np.ascontiguousarray(episode).tobytes()).digest()
            check(digest not in seen, 'no repeated episode across or within splits '+name)
            seen.add(digest)
    return len(seen)


def check_git_provenance(out, freeze, lock, marker):
    """Validate committed blobs and commit ancestry, without invoking research code."""
    root = R.parent.parent
    protocol_commit = lock['protocol_commit']
    summary = read_json(out/'TEST_SUMMARY.json')
    test_marker = read_json(out/'TEST_STARTED.json')
    checkpoint_commit = summary['checkpoint_commit']
    check(marker['protocol_commit'] == protocol_commit, 'training records same protocol commit')
    check(test_marker['checkpoint_commit'] == checkpoint_commit, 'test records same checkpoint commit')
    for label, commit in [('protocol', protocol_commit), ('checkpoint', checkpoint_commit)]:
        check(isinstance(commit, str) and len(commit) == 40 and all(c in '0123456789abcdef' for c in commit), label+' full Git commit')
        kind = subprocess.check_output(['git', 'cat-file', '-t', commit], cwd=root, text=True).strip()
        check(kind == 'commit', label+' resolves to Git commit object')
    check(protocol_commit != checkpoint_commit, 'distinct pre-data and pre-test commits')
    ancestry = subprocess.run(['git', 'merge-base', '--is-ancestor', protocol_commit, checkpoint_commit], cwd=root, capture_output=True)
    check(ancestry.returncode == 0, 'protocol commit precedes checkpoint commit')
    relative = R.relative_to(root).as_posix()
    def tree(commit):
        raw = subprocess.check_output(['git', 'ls-tree', '-r', '--full-tree', commit, '--', relative], cwd=root, text=True)
        result = {}
        for line in raw.splitlines():
            prefix, path = line.split('\t', 1)
            mode, kind, digest = prefix.split()
            check(kind == 'blob', 'regular committed study artifact')
            result[path] = digest
        return result
    protocol_tree, checkpoint_tree = tree(protocol_commit), tree(checkpoint_commit)
    def matches(path, objects, why):
        content = path.read_bytes()
        digest = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        check(objects.get(path.relative_to(root).as_posix()) == digest, why+' committed byte identity')
    for name in ['FREEZE.json', *freeze['sha256']]:
        matches(R/name, protocol_tree, 'pre-data '+name)
    matches(out/'TRAIN_LOCK.json', checkpoint_tree, 'pre-test TRAIN_LOCK')
    for name in lock['files']:
        matches(out/name, checkpoint_tree, 'pre-test '+name)
    check(not any(path.startswith(relative+'/results/') for path in protocol_tree), 'no scientific results in protocol commit')
    check(relative+'/results/TEST_STARTED.json' not in checkpoint_tree, 'checkpoint commit precedes test start')
    return dict(protocol_commit=protocol_commit, checkpoint_commit=checkpoint_commit,
        committed_source_blobs=1+len(freeze['sha256']), committed_training_blobs=1+len(lock['files']),
        commit_ancestry='PASS')


def main():
    global audit_deadline
    started = time.perf_counter()
    audit_deadline = started+300
    out = R/'results'
    freeze = read_json(R/'FREEZE.json')
    for name, digest in freeze['sha256'].items():
        check(sha(R/name) == digest, 'frozen source '+name)
    for name, digest in freeze.get('dependencies', {}).items():
        check(sha(R.parent/name) == digest, 'frozen dependency '+name)
    registry = read_json(R/'SEEDS.json')
    expected_pairs = [dict(seed=s, train_seed=2100000000+10*s, test_seed=2100010000+10*s,
        initial_seeds={str(h): 2100020000+100*h+s for h in SIZES}) for s in range(5)]
    check(len(registry['pairs']) == 5 and registry['pairs'] == expected_pairs, 'fixed five independent seed blocks')
    seeds = [2100030001]
    for pair in expected_pairs:
        seeds.extend([pair['train_seed'], pair['test_seed'], *pair['initial_seeds'].values()])
    check(len(seeds) == len(set(seeds)) == 31, 'disjoint train/test/size initialization/analysis RNG seeds')
    check(registry['sizes'] == list(SIZES), 'all four fixed sizes')
    check((registry['training_updates'], registry['training_batch'], registry['test_episodes'],
        registry['loss'], registry['learning_rate']) == (200, 128, 512, 'RATE', .01), 'frozen training and test budgets')
    check(registry['analysis_seed'] == 2100030001, 'fixed analysis RNG')
    check(registry['checkpoint_updates'] == CHECKPOINTS, 'nine fixed diagnostic checkpoints')
    check(registry['sampled_updates'] == [0, 1, 49, 99, 199], 'fixed derivative replay indices')
    lock = read_json(out/'TRAIN_LOCK.json')
    check(lock['freeze_sha256'] == sha(R/'FREEZE.json'), 'training freeze identity')
    check(lock['test_generated'] is False, 'training lock before held-out generation')
    marker = read_json(out/'TRAIN_STARTED.json')
    check(marker['freeze_sha256'] == sha(R/'FREEZE.json') and marker['seed_registry_sha256'] == sha(R/'SEEDS.json'), 'training-start immutable identities')
    check(marker['protocol_commit'] == lock['protocol_commit'], 'consistent protocol commit identity')
    for name, digest in lock['files'].items():
        check(sha(out/name) == digest, 'locked training artifact '+name)
    check(lock['rows'] == read_json(out/'TRAIN_LOG.json'), 'all training rows locked')
    provenance = check_git_provenance(out, freeze, lock, marker)
    training_data, training = audit_training(out, lock)
    test_data, tests = audit_tests(out, lock['rows'], registry)
    distinct = assert_disjoint({**training_data, **test_data})
    check(distinct == 3200, 'five independent 128-train/512-test dataset pairs')
    result = dict(status='PASS', checks=checks, training=training, tests=tests, git_provenance=provenance,
        distinct_training_test_histories=distinct, seconds=time.perf_counter()-started,
        scope='Separate all-pattern binding RNG, initialization, scalar native equations, RATE objectives, metrics, pattern recalls, activity/state variation, retrospective settling curves, seed-paired bootstrap and Wilson intervals. All 8000 stored-gradient Adam updates and all checkpoints reconstruct. Full native spikes/counts/predictions replay for saved final train and test trajectories. Native trajectories are replayed at nine checkpoints per run, not every unsampled update. Surrogate derivatives are audited separately. No scientific implementation imports. Internal workflow verification is not outside replication or peer review.')
    (out/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
