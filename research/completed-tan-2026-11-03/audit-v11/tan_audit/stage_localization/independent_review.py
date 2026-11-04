"""Independent reconstruction of the frozen stage-localization audit.

Reads primary artifacts without importing or executing run_probe or TAN models.
Recreates all RNG streams, uses a separately written vector Euler solver, and
checks ridge coefficients against augmented least squares, then every prediction,
metric cell and declared bootstrap. This is verification, not a new experiment.
"""
from pathlib import Path
import collections
import hashlib
import itertools
import json
import math
import platform
import resource
import subprocess
import time
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent / 'Temporal-Attention-Neuron'
P1B = REPO / 'p1_experiments/p1b'
VIEWS = ['QUERY_ONLY', 'NATIVE_OUTPUT_Q', 'CUE_U_Q', 'CUE_U_A_Q', 'SHUFFLED_STATE_Q']
ARMS = ['FROZEN_NATIVE', *VIEWS, 'ZERO', 'CORRECT_HISTORY_ORACLE']
DELTAS = [0., 5., 10.]
SPLITS = ['train', 'validation', 'test']
PRIMARY = ['PREREGISTRATION.md', 'run_probe.py', 'INPUTS_SHA256', 'inputs.jsonl',
           'split_and_donor_ledger.json', 'native_observations.jsonl',
           'A_invisibility_checks.json', 'allowed_features.npz', 'frozen_models.json',
           'predictions.jsonl', 'test_history_losses.json', 'results.json', 'run.log']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def readjson(name):
    return json.loads((HERE / name).read_text())

def jsonlines(name):
    return [json.loads(line) for line in (HERE / name).read_text().splitlines()]

def close(a, b, *, tolerance=1e-12, label=''):
    x, y = np.asarray(a), np.asarray(b)
    assert x.shape == y.shape, (label, x.shape, y.shape)
    assert np.isfinite(x).all() and np.isfinite(y).all(), label
    error = float(np.max(np.abs(x - y))) if x.size else 0.
    assert error <= tolerance, (label, error, tolerance)
    return error

def recreate_inputs():
    """Literal independent implementation of documented generator call order."""
    split_rng = np.random.RandomState(2026110222)
    histories = []
    for count, seed in [(2, 2026110220), (3, 2026110221)]:
        rng = np.random.RandomState(seed)
        generated = []
        for index in range(1792):
            rejection_count = 0
            while True:
                keys_sorted = np.sort(rng.uniform(.3, 2.5, count))
                if (np.diff(keys_sorted) >= .05).all():
                    break
                rejection_count += 1
            active_keys = rng.permutation(keys_sorted)
            active_values = rng.uniform(-3., 3., count)
            positions = sorted(rng.choice(5, count, replace=False).tolist())
            keys = np.zeros(5)
            values = np.zeros(5)
            mask = np.zeros(5, dtype=bool)
            keys[positions], values[positions], mask[positions] = active_keys, active_values, True
            target_idx = rng.choice(positions, 2, replace=False).tolist()
            generated.append(dict(id=f'N{count}_{index}', N=count, trial_id=index,
                seed_metadata=seed+index, keys=keys.tolist(), values=values.tolist(),
                mask=mask.tolist(), positions=positions, queries=keys[target_idx].tolist(),
                targets=values[target_idx].tolist(), target_indices=target_idx,
                rejection_count=rejection_count, all_positive=bool((active_keys > 1.4).all())))
        permutation = split_rng.permutation(1792)
        assigned = np.empty(1792, dtype=object)
        assigned[permutation[:1024]] = 'train'
        assigned[permutation[1024:1280]] = 'validation'
        assigned[permutation[1280:]] = 'test'
        for h, split in zip(generated, assigned):
            h['split'] = split
        histories.extend(generated)
    return histories

def step(u, gate, refractory, content=0., novelty=0., query=0.):
    """Independent vectorization of registered old-A Euler update and reset."""
    next_gate = np.clip(gate + (-gate + novelty)/15.*.1, 0., 5.)
    is_refractory = refractory > 0.
    candidate = u + (-u + gate*content + query)/10.*.1
    spike = (~is_refractory) & (candidate >= 1.)
    next_u = np.where(is_refractory | spike, 0., candidate)
    next_r = np.where(is_refractory, np.maximum(0., refractory-.1),
                      np.where(spike, 2., 0.))
    return next_u, next_gate, next_r, spike, candidate

def query_trajectory(u, gate, refractory, queries):
    u, gate, refractory = u.copy(), gate.copy(), refractory.copy()
    raw_spikes = np.zeros((len(u), 100), dtype=np.int8)
    for tick in range(100):
        before = refractory.copy()
        u, gate, refractory, emitted, candidate = step(u, gate, refractory, query=queries)
        raw_spikes[:, tick] = emitted
    readout = np.where(before > 0., 0., candidate)
    return dict(readout=readout, raw_spikes=raw_spikes, r_before=before,
                r_after=refractory, post_u=u, final_gate=gate)

def reconstruct_native(histories):
    h = len(histories)
    keys = np.array([x['keys'] for x in histories])
    values = np.array([x['values'] for x in histories])
    mask = np.array([x['mask'] for x in histories])
    u = np.zeros(h)
    gate, refractory = u.copy(), u.copy()
    for tick in range(1000):
        slot, phase = divmod(tick, 200)
        active = mask[:, slot] & (phase < 100)
        content = np.where(active, keys[:, slot]+values[:, slot], 0.)
        novelty = np.where(active, np.maximum(0., keys[:, slot]-1.4), 0.)
        u, gate, refractory, _, _ = step(u, gate, refractory, content, novelty)
    assert (refractory == 0.).all()
    endpoints = [np.column_stack((u, gate, refractory))]
    for tick in range(100):
        u, gate, refractory, _, _ = step(u, gate, refractory)
        if tick in [49, 99]:
            endpoints.append(np.column_stack((u, gate, refractory)))
    # Shapes preserve input history, delay, then role order in primary ledger.
    cues = np.stack(endpoints, axis=1)
    observed = np.repeat(cues, 2, axis=1).reshape(-1, 3)
    queries = np.repeat(np.array([x['queries'] for x in histories])[:, None, :], 3, axis=1).ravel()
    targets = np.repeat(np.array([x['targets'] for x in histories])[:, None, :], 3, axis=1).ravel()
    native = query_trajectory(*observed.T, queries)
    return observed, queries, targets, native

def independent_design(raw, selected):
    means = np.mean(raw[selected], axis=0)
    scales = np.sqrt(np.mean((raw[selected]-means)**2, axis=0))
    scales[scales == 0.] = 1.
    normalized = (raw-means)/scales
    dimension = raw.shape[1]
    powers = [x for total in range(5) for x in itertools.product(range(5), repeat=dimension)
              if sum(x) == total]
    # Columnwise monomial construction avoids primary broadcast/prod routine.
    columns = []
    for power in powers:
        column = np.ones(len(raw))
        for j in range(dimension):
            column = column * normalized[:, j]**power[j]
        columns.append(column)
    polynomial = np.column_stack(columns)
    poly_means = polynomial[selected].mean(axis=0)
    poly_scales = np.sqrt(np.mean((polynomial[selected]-poly_means)**2, axis=0))
    poly_scales[poly_scales == 0.] = 1.
    poly_means[0], poly_scales[0] = 0., 1.
    design = (polynomial-poly_means)/poly_scales
    return design, dict(raw_mean=means, raw_scale=scales, exponents=powers,
        expanded_mean=poly_means, expanded_scale=poly_scales)

def metric_values(prediction, raw_prediction, truth, history_ids):
    difference = np.abs(prediction-truth)
    raw_difference = np.abs(raw_prediction-truth)
    changed = raw_prediction != prediction
    return dict(n_histories=len(set(history_ids)), n_pairs=len(truth),
        mean_ecomp=float(np.mean(np.max(difference, axis=1))),
        mae_per_role=np.mean(difference, axis=0).tolist(),
        rmse_per_role=np.sqrt(np.mean(difference**2, axis=0)).tolist(),
        max_error=float(np.max(difference)),
        strict_0_5_joint=float(np.mean((difference < .5).all(axis=1))),
        raw_mean_ecomp=float(np.max(raw_difference, axis=1).mean()),
        raw_max_error=float(np.max(raw_difference)),
        clip_count_per_role=changed.sum(axis=0).tolist(),
        clip_pair_count=int(changed.any(axis=1).sum()))

def main():
    start = time.perf_counter()
    before = {name: sha(HERE/name) for name in PRIMARY}
    # Source-only archives intentionally omit .git. Do not claim verification of
    # repository status when that provenance is unavailable in snapshot replay.
    git_available = (REPO/'.git').exists()
    git_before = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True) if git_available else None
    result = readjson('results.json')
    fit = readjson('frozen_models.json')
    for artifact, field in [('PREREGISTRATION.md', 'preregistration_sha256'),
                             ('run_probe.py', 'script_sha256'),
                             ('inputs.jsonl', 'inputs_sha256'),
                             ('frozen_models.json', 'frozen_models_sha256')]:
        assert before[artifact] == result[field], artifact
    model_path = P1B/'models_p1b_v2.py'
    assert sha(model_path) == result['model_source_sha256'] == '6b31fe0ef8b18acdd1c7364879c18cc70c587f088841fa3aacf1f67b6771191f'
    generator_path = REPO/'p1_experiments/task_generator.py'
    generator_hash = '674cf9f49f5cd7f4975b87d29cd413e801fe40bd10fc56b966e663c9e6fc989d'
    assert sha(generator_path) == generator_hash
    assert fit['input_hash'] == before['inputs.jsonl']
    assert fit['features_sha256'] == before['allowed_features.npz']
    assert fit['output_clip'] == [-3., 3.] and not fit['validation_or_test_used_to_fit']
    calibration = json.loads((P1B/'summary_p1b.json').read_text())['calibration']
    assert calibration == result['calibration']
    histories = recreate_inputs()
    assert histories == jsonlines('inputs.jsonl')
    history_index = {r['id']: i for i, r in enumerate(histories)}
    splits = {r['id']: r['split'] for r in histories}
    ledger = readjson('split_and_donor_ledger.json')
    assert ledger['splits'] == splits
    donors = {}
    shuffle_rng = np.random.RandomState(2026110223)
    subgroup_counts = {}
    for n in [2, 3]:
        for split, expected_count in zip(SPLITS, [1024, 256, 512]):
            ids = [r['id'] for r in histories if r['N'] == n and r['split'] == split]
            assert len(ids) == expected_count
            cycle = shuffle_rng.permutation(len(ids))
            for p, recipient in enumerate(cycle):
                donors[ids[recipient]] = ids[cycle[(p+1) % len(cycle)]]
            subgroup_counts[f'N{n}_{split}'] = sum(histories[history_index[x]]['all_positive'] for x in ids)
    assert donors == ledger['donors']
    assert all(x != y and splits[x] == splits[y] and x[:2] == y[:2] for x, y in donors.items())
    assert len(set(donors.values())) == len(histories) == 3584
    for n in [2, 3]:
        for split in SPLITS:
            ids = {x for x in splits if x.startswith(f'N{n}_') and splits[x] == split}
            visited = set()
            current = next(iter(ids))
            while current not in visited:
                visited.add(current)
                current = donors[current]
            assert visited == ids
    print('PASS inputs, RNG streams, all splits, six donor cycles', flush=True)

    observed, queries, targets, native = reconstruct_native(histories)
    records = jsonlines('native_observations.jsonl')
    assert len(records) == 21504
    expected_prediction = calibration['g_dec']*native['readout']+calibration['b_dec']
    max_native_errors = {}
    for field, expected in dict(u_cue=observed[:, 0], A_cue=observed[:, 1],
        r_cue=observed[:, 2], q=queries, target=targets, u_readout=native['readout'],
        native_prediction=expected_prediction, r_deadline_minus=native['r_before'],
        spike_count=native['raw_spikes'][:, :99].sum(axis=1), oracle_prediction=targets).items():
        max_native_errors[field] = close([r[field] for r in records], expected, label=field)
    for row, record in enumerate(records):
        hi, offset = divmod(row, 6)
        delay_i, role = divmod(offset, 2)
        source = histories[hi]
        for field in ['N', 'trial_id', 'split', 'all_positive']:
            assert record[field] == source[field]
        assert record['row_id'] == row and record['input_id'] == source['id']
        assert record['delay_ms'] == DELTAS[delay_i] and record['role'] == role
        positions = source['positions']
        best = min(positions, key=lambda i: abs(source['keys'][i]-record['q']))
        assert source['values'][best] == record['oracle_prediction']
    assert (observed[:, 2] == 0.).all()

    structural = readjson('A_invisibility_checks.json')
    assert structural['checkpoints'] == 96 and structural['intervention_windows'] == 192
    selected = {n: [i for i, x in enumerate(histories) if x['N'] == n and x['split'] == 'test'][:8]
                for n in [2, 3]}
    structural_rows = np.array([6*i+j for n in [2, 3] for i in selected[n] for j in range(6)])
    for r, case in zip(structural_rows, structural['cases'], strict=True):
        assert case['input_id'] == records[r]['input_id'] and case['role'] == records[r]['role']
        assert case['delay_ms'] == records[r]['delay_ms'] and case['interventions'] == [0., 5.]
        assert case['all_native_results_exactly_equal'] is True
        close(case['u_readout'], native['readout'][r], label='structural readout')
        close(case['raw_spikes'], native['raw_spikes'][r], tolerance=0., label='structural spikes')
    for value in [0., 5.]:
        intervened = query_trajectory(observed[structural_rows, 0],
            np.full(96, value), observed[structural_rows, 2], queries[structural_rows])
        for field in ['readout', 'raw_spikes', 'r_before', 'r_after', 'post_u']:
            close(intervened[field], native[field][structural_rows], tolerance=0., label=f'A intervention {field}')
    print('PASS every native query and cue state; 96 checkpoints/192 A interventions', flush=True)

    donor_rows = np.array([6*history_index[donors[r['input_id']]]+2*DELTAS.index(r['delay_ms']) for r in records])
    independently_built_features = dict(QUERY_ONLY=queries[:, None],
        NATIVE_OUTPUT_Q=np.column_stack((native['readout'], queries)),
        CUE_U_Q=np.column_stack((observed[:, 0], queries)),
        CUE_U_A_Q=np.column_stack((observed[:, :2], queries)),
        SHUFFLED_STATE_Q=np.column_stack((observed[donor_rows, :2], queries)))
    feature_archive = np.load(HERE/'allowed_features.npz')
    assert set(feature_archive.files) == set(VIEWS)
    feature_residuals = {name: close(feature_archive[name], independently_built_features[name],
                         label='features '+name) for name in VIEWS}
    train = np.repeat(np.array([h['split'] == 'train' for h in histories]), 6)
    assert train.sum() == 12288
    train_hash = hashlib.sha256(np.flatnonzero(train).astype('<i8').tobytes()).hexdigest()
    solver_checks = {}
    raw_predictions = dict(FROZEN_NATIVE=expected_prediction, ZERO=np.zeros(len(records)),
                           CORRECT_HISTORY_ORACLE=targets.copy())
    for view, expected_terms in zip(VIEWS, [5, 15, 15, 35, 35]):
        model = fit['models'][view]
        assert model['degree'] == 4 and model['ridge_lambda'] == .001
        assert model['n_terms'] == expected_terms and model['training_rows'] == 12288
        assert model['training_row_ids_sha256'] == train_hash
        X, transform = independent_design(independently_built_features[view], train)
        transform_errors = {key: close(value, model[key], label=view+' '+key) for key, value in transform.items()}
        Xtrain = X[train]
        ytrain = targets[train]
        # Solve augmented least squares: [X/sqrt(n); sqrt(lambda) I_no_intercept].
        penalty = np.eye(expected_terms)*math.sqrt(.001)
        penalty[0, 0] = 0.
        augmented_X = np.vstack((Xtrain/math.sqrt(len(Xtrain)), penalty))
        augmented_y = np.concatenate((ytrain/math.sqrt(len(Xtrain)), np.zeros(expected_terms)))
        beta, _, rank, singular_values = np.linalg.lstsq(augmented_X, augmented_y, rcond=None)
        assert rank == expected_terms
        frozen = np.array(model['coefficient'])
        coefficient_error = close(beta, frozen, tolerance=1e-8, label=view+' augmented least squares')
        gram = Xtrain.T@Xtrain/len(Xtrain)+penalty.T@penalty
        rhs = Xtrain.T@ytrain/len(Xtrain)
        normal_error = close(gram@frozen, rhs, tolerance=1e-10, label=view+' normal equation')
        predictions_saved_coefficient = X@frozen
        independent_prediction_error = close(X@beta, predictions_saved_coefficient,
                                             tolerance=1e-8, label=view+' predictions')
        raw_predictions[view] = predictions_saved_coefficient
        solver_checks[view] = dict(n_terms=expected_terms, training_rows=int(train.sum()),
            transform_max_residuals=transform_errors, augmented_least_squares_coefficient_error=coefficient_error,
            frozen_normal_equation_residual=normal_error, independently_refit_prediction_error=independent_prediction_error,
            augmented_condition=float(singular_values[0]/singular_values[-1]),
            regularized_gram_condition=float(np.linalg.cond(gram)))
    print('PASS five allowed feature views, training-only transforms, alternate ridge solutions', flush=True)

    pairs = jsonlines('predictions.jsonl')
    assert len(pairs) == 10752
    prediction_residuals = {}
    prediction_arrays, raw_arrays = {}, {}
    for arm in ARMS:
        raw_array = np.array([r['raw_predictions'][arm] for r in pairs])
        pred_array = np.array([r['predictions'][arm] for r in pairs])
        independently_predicted = raw_predictions[arm].reshape(-1, 2)
        prediction_residuals[arm] = close(raw_array, independently_predicted, tolerance=1e-8,
                                         label=arm+' every saved raw prediction')
        expected_clipped = np.clip(raw_array, -3., 3.) if arm in VIEWS else raw_array
        close(pred_array, expected_clipped, tolerance=0., label=arm+' exact clipping')
        prediction_arrays[arm], raw_arrays[arm] = pred_array, raw_array
    for i, pair in enumerate(pairs):
        source = histories[i//3]
        assert pair['input_id'] == source['id'] and pair['delay_ms'] == DELTAS[i % 3]
        for key in ['N', 'split', 'all_positive']:
            assert pair[key] == source[key]
        assert pair['target'] == source['targets'] and pair['queries'] == source['queries']
        assert pair['shuffled_state_donor'] == donors[source['id']]
        assert set(pair['predictions']) == set(pair['raw_predictions']) == set(ARMS)
    y_pairs = targets.reshape(-1, 2)
    pair_ids = np.array([r['input_id'] for r in pairs])
    pair_split = np.array([r['split'] for r in pairs])
    pair_n = np.array([r['N'] for r in pairs])
    pair_positive = np.array([r['all_positive'] for r in pairs])
    pair_delay = np.array([r['delay_ms'] for r in pairs])
    cells_by_key = {(r['split'], r['stratum'], r['N'], r['delay_ms'], r['arm']): r for r in result['cells']}
    assert len(cells_by_key) == len(result['cells']) == 576
    metric_max_residual = 0.
    for split, stratum, n, delay, arm in itertools.product(SPLITS, ['full', 'all_positive'],
            ['pooled', 2, 3], ['pooled', *DELTAS], ARMS):
        selection = pair_split == split
        if stratum == 'all_positive':
            selection &= pair_positive
        if n != 'pooled':
            selection &= pair_n == n
        if delay != 'pooled':
            selection &= pair_delay == delay
        checked = metric_values(prediction_arrays[arm][selection], raw_arrays[arm][selection],
                                 y_pairs[selection], pair_ids[selection])
        cell = cells_by_key[(split, stratum, n, delay, arm)]
        for key, value in checked.items():
            metric_max_residual = max(metric_max_residual, close(value, cell[key], label='metric '+str((split,stratum,n,delay,arm,key))))
    print('PASS all 10752 paired prediction rows, 172032 predictions and 576 metric cells', flush=True)

    test_history_rows = [i for i, h in enumerate(histories) if h['split'] == 'test']
    losses = {arm: np.max(np.abs(prediction_arrays[arm]-y_pairs), axis=1).reshape(-1, 3).mean(axis=1)
              for arm in ARMS}
    saved_history_losses = readjson('test_history_losses.json')
    assert len(saved_history_losses) == 1024
    history_loss_max_residual = 0.
    for index, row in zip(test_history_rows, saved_history_losses, strict=True):
        assert row['input_id'] == histories[index]['id']
        assert row['N'] == histories[index]['N'] and row['all_positive'] == histories[index]['all_positive']
        for arm in ARMS:
            history_loss_max_residual = max(history_loss_max_residual,
                close(row['losses'][arm], losses[arm][index], label='history loss'))
    comparisons = [('CUE_U_A_Q', 'FROZEN_NATIVE'), ('NATIVE_OUTPUT_Q', 'FROZEN_NATIVE'),
        ('CUE_U_Q', 'NATIVE_OUTPUT_Q'), ('CUE_U_A_Q', 'CUE_U_Q'),
        ('SHUFFLED_STATE_Q', 'CUE_U_A_Q'), ('QUERY_ONLY', 'CUE_U_A_Q')]
    bootstrap_rng = np.random.RandomState(2026110224)
    contrast_map = {(x['stratum'], x['left'], x['right']): x for x in result['paired_test_contrasts']}
    assert len(contrast_map) == len(result['paired_test_contrasts']) == 12
    bootstrap_max_residual = 0.
    for stratum in ['full', 'all_positive']:
        groups = {n: np.array([i for i in test_history_rows if histories[i]['N'] == n and
                  (stratum == 'full' or histories[i]['all_positive'])]) for n in [2, 3]}
        indices = {n: bootstrap_rng.randint(0, len(groups[n]), size=(2000, len(groups[n]))) for n in [2, 3]}
        count = sum(len(rows) for rows in groups.values())
        for left, right in comparisons:
            observed_difference = losses[left]-losses[right]
            draws = np.zeros(2000)
            mean_numerator = 0.
            for n in [2, 3]:
                d = observed_difference[groups[n]]
                draws += d[indices[n]].sum(axis=1)
                mean_numerator += d.sum()
            draws /= count
            contrast = contrast_map[(stratum, left, right)]
            assert contrast['n_histories'] == count and contrast['replicates'] == 2000
            assert contrast['counts_by_N'] == {str(n): len(groups[n]) for n in [2, 3]}
            for computed, stored in [(mean_numerator/count, contrast['mean_ecomp_difference']),
                    (np.quantile(draws, [.025, .975]), contrast['percentile95_interval'])]:
                bootstrap_max_residual = max(bootstrap_max_residual,
                    close(computed, stored, label='paired bootstrap'))
    print('PASS 1024 three-delay-averaged history losses and all 12 bootstrap contrasts', flush=True)

    positive_test_ids = [h['id'] for h in histories if h['split'] == 'test' and h['all_positive']]
    cross_stratum_donors = {str(n): sum(histories[history_index[x]]['N'] == n and
        not histories[history_index[donors[x]]]['all_positive'] for x in positive_test_ids) for n in [2, 3]}
    native_deadline_spikes = int(native['raw_spikes'][:, 99].sum())
    native_deadline_refractory = int((native['r_before'] > 0.).sum())
    after = {name: sha(HERE/name) for name in PRIMARY}
    assert before == after, 'Primary artifact was changed during independent review'
    if git_available:
        assert subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True) == git_before
    else:
        assert not (REPO/'.git').exists()
    assert sha(model_path) == result['model_source_sha256']
    assert sha(generator_path) == generator_hash
    checks = dict(status='PASS', primary_artifacts_unchanged=True,
        repository_status_verified=git_available,
        repository_status_unchanged=True if git_available else None,
        repository_status_availability='available' if git_available else 'unavailable: snapshot has no .git',
        repository_porcelain=git_before,
        primary_sha256=before, independent_script_sha256=sha(Path(__file__)),
        source_sha256={str(model_path): sha(model_path), str(generator_path): sha(generator_path)},
        independently_regenerated_histories=3584, query_rows_checked=21504,
        paired_prediction_rows_checked=10752, arms=8, raw_and_clipped_scalar_predictions_checked=344064,
        metric_cells_checked=576, paired_bootstrap_contrasts_checked=12,
        bootstrap_replicates_per_contrast=2000, positive_history_counts=subgroup_counts,
        test_positive_recipients_with_nonpositive_donors_by_N=cross_stratum_donors,
        native_max_residuals=max_native_errors, feature_max_residuals=feature_residuals,
        structural_checkpoints=96, structural_A_intervention_windows=192,
        native_deadline_spikes=native_deadline_spikes,
        native_deadline_refractory=native_deadline_refractory,
        cue_r_all_zero=True, solver_checks=solver_checks,
        saved_prediction_max_residuals=prediction_residuals,
        metric_max_residual=metric_max_residual, history_loss_max_residual=history_loss_max_residual,
        bootstrap_max_residual=bootstrap_max_residual,
        validation_or_test_used_for_refit=False,
        python=platform.python_version(), numpy=np.__version__,
        wall_seconds=time.perf_counter()-start,
        maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitations=['Manual source inspection complements output reconstruction; saved artifacts cannot alone prove historical execution ordering.',
            'Recipient bootstrap does not preserve donor-cycle dependence for shuffled-state contrasts.',
            'All-positive recipient shuffling is not restricted to all-positive donors.',
            'Intervals condition on the one frozen trained model and do not include training-sample variation.',
            'Predictive state content is not established query-specific key binding; no state-only or shuffled-query arm was run.',
            '15 versus 35 terms, regularization geometry, direct q and state access prevent unique causal attribution.',
            'No probe failure can establish absence of retained information.'])
    (HERE/'INDEPENDENT_REVIEW_CHECKS.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps({k: checks[k] for k in ['status', 'independently_regenerated_histories',
        'query_rows_checked', 'paired_prediction_rows_checked', 'metric_cells_checked',
        'paired_bootstrap_contrasts_checked', 'metric_max_residual', 'bootstrap_max_residual',
        'wall_seconds', 'maxrss_kib']}, indent=2), flush=True)

if __name__ == '__main__':
    main()
