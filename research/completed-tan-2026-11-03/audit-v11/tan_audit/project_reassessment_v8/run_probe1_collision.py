"""Frozen original-Probe1 swap experiment; no fitting or model search."""
from pathlib import Path
import csv, hashlib, itertools, json, platform, resource, sys, time
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent / 'Temporal-Attention-Neuron'
CODE = REPO / 'snapshot/code'
PLAN = 'd8f0b97300f7fd1d3e3b0c07cd5ca17c3cfb4b40b20efa25792e6ab3dd78df23'
SOURCES = {
    'snapshot/code/experiments/natural_collision.py': '8c7b4e75ec8284b671bca6280ee848498ddd9c6109a6dba31254f6d407ec4d7f',
    'snapshot/code/experiments/temporal_distractor.py': '6177df497f9735d9fdb434ff20e7751a8dbb4af7a32d836bb3c4898c06ad4f23',
}
ARMS = [('B1_lif', 'lif'), ('B2_buf', 'buf'), ('B3_noattn', 'noattn'), ('B4_tan', 'tan')]
LO = np.array([123, 273, 123, 273, 273], dtype=float) / 200
HI = np.array([127, 277, 127, 277, 277], dtype=float) / 200

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def instrument(X, model, suppress_reset_step=None):
    """Literal vector recurrence, with one explicitly named reset intervention.

    The event flag is always the strict threshold comparison. Only the reset
    assignment is skipped at the selected one-based step; all later resets run.
    """
    n, length = X.shape
    windows = np.lib.stride_tricks.sliding_window_view(
        np.concatenate((np.zeros((n, 4)), X), axis=1), 5, axis=1)
    out = {key: np.zeros((n, length)) for key in ['u', 'h', 'j']}
    if model in ('tan', 'noattn'):
        out['s'] = np.zeros((n, length))
    if model == 'tan':
        out['alpha'] = np.zeros((n, length, 5))
    spikes = np.zeros((n, length), dtype=bool)
    previous = np.zeros(n)
    for step in range(length):
        win = windows[:, step]
        if model == 'lif':
            current = win[:, -1]
        elif model == 'buf':
            current = win @ (np.arange(1, 6, dtype=float) / 15)
        else:
            surprise = np.maximum(0., win[:, -1] - win.mean(axis=1))
            out['s'][:, step] = surprise
            if model == 'noattn':
                context = win.mean(axis=1)
            else:
                exponential = np.exp(2. * surprise[:, None] * win)
                alpha = exponential / (exponential.sum(axis=1, keepdims=True) + 1e-9)
                out['alpha'][:, step] = alpha
                context = (alpha * win).sum(axis=1)
            current = np.tanh(surprise) * context
        candidate = .5 * previous + current
        fired = candidate > .5
        voltage = candidate if step + 1 == suppress_reset_step else np.where(fired, 0., candidate)
        out['u'][:, step] = candidate
        out['j'][:, step] = current
        out['h'][:, step] = voltage
        spikes[:, step] = fired
        previous = voltage
    out['spikes'] = spikes
    return out

def numbers(values):
    a = np.asarray(values, dtype=float)
    return dict(min=float(a.min()), max=float(a.max()), mean=float(a.mean()))

def main():
    started = time.perf_counter()
    assert sha(HERE / 'PREREGISTRATION.md') == PLAN
    for name, expected in SOURCES.items():
        assert sha(REPO / name) == expected, name
    cases = [('anchor', np.array([5, 11, 5, 11, 11], dtype=float) / 8)]
    cases.extend(('corner', np.where(np.array(bits), HI, LO)) for bits in itertools.product((0, 1), repeat=5))
    rng = np.random.RandomState(2026110225)
    cases.extend(('random_box', row) for row in rng.uniform(LO, HI, size=(512, 5)))
    inputs = []
    X = np.zeros((len(cases), 2, 14))
    for i, (kind, p) in enumerate(cases):
        assert np.all(p >= LO) and np.all(p <= HI)
        X[i, 0, 4:9] = p
        X[i, 1, 4:9] = p[[1, 0, 2, 3, 4]]
        inputs.append(dict(pair_id=i, kind=kind, parameters=p.tolist(),
            X=X[i].tolist(), labels=[0, 1], latent_classes=[[0, 1, 0, 1, 1], [1, 0, 0, 1, 1]],
            target_values=X[i, :, 4].tolist(), query=float(p[4])))
    assert len(inputs) == 545
    input_path = HERE / 'probe1_inputs.jsonl'
    input_path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in inputs))
    input_hash = sha(input_path)
    (HERE / 'PROBE1_INPUTS_SHA256').write_text(input_hash + '  probe1_inputs.jsonl\n')
    print('Frozen input ledger before model execution:', input_hash, flush=True)

    sys.path.insert(0, str(CODE))
    from experiments import natural_collision as nc
    from experiments import temporal_distractor as td
    assert (nc.W, nc.LAMBDA, nc.THETA, nc.W_Q, nc.W_K, nc.W_V, nc.BETA, nc.EPS) == (5, .5, .5, 2., 1., 1., 1., 0.)
    assert (td.L_BLK, td.Q_IDX, td.TGT_IDX) == (14, 8, 4)
    flat = X.reshape(-1, 14)
    windows = np.lib.stride_tricks.sliding_window_view(
        np.concatenate((np.zeros((len(flat), 4)), flat), axis=1), 5, axis=1).reshape(545, 2, 14, 5).copy()
    data = dict(inputs=X, windows=windows, labels=np.tile([0, 1], (545, 1)),
                parameters=np.array([x[1] for x in cases]))
    outputs = {}
    validation = {}
    for arm, model in ARMS:
        native = nc.simulate(flat, model, need_alpha=(model == 'tan'))
        reconstructed = instrument(flat, model)
        validation[arm] = {key: float(np.max(np.abs(value - reconstructed[key]))) for key, value in native.items()}
        native['spikes'] = native['u'] > nc.THETA
        native['features'] = td.resp_features(native)
        outputs[arm] = native
    intervention = instrument(flat, 'tan', suppress_reset_step=8)
    intervention['features'] = td.resp_features(intervention)
    outputs['B4_no_reset_t8'] = intervention
    for arm, output in outputs.items():
        for key, value in output.items():
            data[arm + '__' + key] = value.reshape((545, 2) + value.shape[1:])
    native = outputs['B4_tan']
    v = validation['intervention_scope'] = {
        'u_through_t8_exact': bool(np.array_equal(intervention['u'][:, :8], native['u'][:, :8])),
        'h_through_t7_exact': bool(np.array_equal(intervention['h'][:, :7], native['h'][:, :7])),
        't8_flag_unchanged': bool(np.array_equal(intervention['spikes'][:, 7], native['spikes'][:, 7])),
        't8_h_equals_candidate': bool(np.array_equal(intervention['h'][:, 7], intervention['u'][:, 7])),
        'j_s_alpha_exact_all_steps': all(np.array_equal(intervention[key], native[key]) for key in ['j', 's', 'alpha']),
        'other_steps_follow_native_reset_rule': all(np.array_equal(intervention['h'][:, step], np.where(intervention['spikes'][:, step], 0., intervention['u'][:, step])) for step in range(14) if step != 7),
    }
    oracle = np.where(windows[:, :, 8, 0] < .95, 0, np.where(windows[:, :, 8, 0] > 1.05, 1, -1))
    data['target_tap_oracle'] = oracle
    np.savez_compressed(HERE / 'probe1_trajectories.npz', **data)
    pair_rows = []
    arm_summaries = {}
    for arm in outputs:
        F = data[arm + '__features']
        gap = abs(F[:, 0] - F[:, 1])
        max_gap = gap.max(axis=1)
        literal = np.all(F[:, 0] == F[:, 1], axis=1)
        U, H, J, spikes = [data[arm + '__' + key] for key in ['u', 'h', 'j', 'spikes']]
        S = data.get(arm + '__s')
        for i, rec in enumerate(inputs):
            pair_rows.append(dict(pair_id=i, kind=rec['kind'], arm=arm, labels=[0, 1], target_values=rec['target_values'],
                query=rec['query'], features=F[i].tolist(), literal_equal_features=bool(literal[i]),
                max_abs_feature_gap=float(max_gap[i]), l2_feature_gap=float(np.linalg.norm(gap[i])),
                target_amplitude_gap=float(abs(X[i, 0, 4] - X[i, 1, 4])),
                window_t9_linf_gap=float(np.max(abs(windows[i, 0, 8] - windows[i, 1, 8]))),
                t8_u=U[i, :, 7].tolist(), t8_h=H[i, :, 7].tolist(), t8_spikes=spikes[i, :, 7].tolist(),
                t8_threshold_margin=(U[i, :, 7] - .5).tolist(), t9_u=U[i, :, 8].tolist(),
                t9_threshold_headroom=(.5 - U[i, :, 8]).tolist(), t9_surprise=S[i, :, 8].tolist() if S is not None else None,
                oracle_predictions=oracle[i].tolist(), both_oracle_correct=bool(np.array_equal(oracle[i], [0, 1]))))
        arm_summaries[arm] = dict(pairs=545, histories=1090,
            literal_equal_response_pairs=int(literal.sum()), near_equal_response_pairs_at_1e_13=int((max_gap <= 1e-13).sum()),
            response_linf_gap=numbers(max_gap), response_l2_gap=numbers(np.linalg.norm(gap, axis=1)),
            analog_feature_linf_gap=numbers(gap[:, :5].max(axis=1)),
            differing_spike_feature_pairs=int(np.any(F[:, 0, 5:] != F[:, 1, 5:], axis=1).sum()),
            query_spike_disagreement_pairs=int((spikes[:, 0, 8] != spikes[:, 1, 8]).sum()),
            query_u_abs_gap=numbers(abs(U[:, 0, 8] - U[:, 1, 8])),
            t8_spike_histories=int(spikes[:, :, 7].sum()), t8_zero_h_histories=int((H[:, :, 7] == 0).sum()),
            t8_current=numbers(J[:, :, 7]), t8_candidate_margin=numbers(U[:, :, 7] - .5),
            t9_positive_surprise_histories=int((S[:, :, 8] > 0).sum()) if S is not None else None,
            t9_surprise=numbers(S[:, :, 8]) if S is not None else None,
            t9_candidate=numbers(U[:, :, 8]), t9_threshold_headroom=numbers(.5 - U[:, :, 8]),
            anchor_features=F[0].tolist(), anchor_t8_u=U[0, :, 7].tolist(), anchor_t8_h=H[0, :, 7].tolist())
    (HERE / 'probe1_pair_ledger.jsonl').write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in pair_rows))

    tangent = data['B4_tan__features']
    expansion = np.column_stack([tangent[:, :, 0].reshape(-1) / 2 ** i for i in range(5)] + [np.zeros(1090)] * 5).reshape(545, 2, 10)
    checks = {
        'all_source_hashes_unchanged': all(sha(REPO / name) == expected for name, expected in SOURCES.items()),
        'all_native_reconstruction_residuals_below_1e_14': all(value < 1e-14 for arm in dict(ARMS) for value in validation[arm].values()),
        'all_intervention_scope_checks': all(v.values()),
        'B4_all_t8_reset': bool(np.all(data['B4_tan__h'][:, :, 7] == 0)),
        'B4_all_t9_positive_surprise': bool(np.all(data['B4_tan__s'][:, :, 8] > 0)),
        'B4_all_t9_strict_subthreshold': bool(np.all((data['B4_tan__u'][:, :, 8] > 0) & (data['B4_tan__u'][:, :, 8] < .5))),
        'B4_zero_gate_all_steps_after_query': bool(np.all(data['B4_tan__s'][:, :, 9:] == 0)),
        'B4_feature_expansion_exact': bool(np.array_equal(tangent, expansion)),
        'B4_ideal_collision_numeric_residual_below_1e_13': bool(np.max(abs(tangent[:, 0] - tangent[:, 1])) < 1e-13),
        'B2_anchor_query_formula': bool(np.max(abs(data['B2_buf__u'][0, :, 8] - [47 / 40, 9 / 8])) < 1e-14),
        'B2_box_query_gap_formula': bool(np.max(abs((data['B2_buf__u'][:, 0, 8] - data['B2_buf__u'][:, 1, 8]) - (X[:, 0, 5] - X[:, 0, 4]) / 15)) < 1e-14),
        'B1_anchor_response_equal': bool(np.array_equal(data['B1_lif__features'][0, 0], data['B1_lif__features'][0, 1])),
        'target_tap_oracle_correct_all_histories': bool(np.array_equal(oracle, data['labels'])),
        'full_windows_differ_at_t9_and_t10': bool(np.all(np.max(abs(windows[:, 0, 8:10] - windows[:, 1, 8:10]), axis=2) > 0)),
        'full_windows_equal_from_t11': bool(np.array_equal(windows[:, 0, 10:], windows[:, 1, 10:])),
        'B4_full_state_equal_from_t11_with_numeric_tolerance': bool(np.max(abs(data['B4_tan__h'][:, 0, 10:] - data['B4_tan__h'][:, 1, 10:])) < 1e-13),
    }
    result = dict(status='PASS' if all(checks.values()) else 'CHECK_FAILURE',
        protocol_sha256=PLAN, source_sha256=SOURCES, script_sha256=sha(Path(__file__)), inputs_sha256=input_hash,
        trajectories_sha256=sha(HERE / 'probe1_trajectories.npz'), pair_ledger_sha256=sha(HERE / 'probe1_pair_ledger.jsonl'),
        seed=2026110225, python=platform.python_version(), numpy=np.__version__,
        cases=dict(anchor=1, corners=32, random_box=512, total_pairs=545, histories_per_arm=1090, exclusions=0),
        model_trajectory_counts=dict(primary_native=4360, primary_single_reset_intervention=1090,
            native_vector_reconstruction_validation=4360, primary_euler_steps=76300, validation_euler_steps=61040),
        controls=dict(target_tap_oracle_correct=1090, target_tap_oracle_total=1090,
            target_amplitude_gap=numbers(abs(X[:, 0, 4] - X[:, 1, 4])),
            full_window_t9_linf_gap=numbers(np.max(abs(windows[:, 0, 8] - windows[:, 1, 8]), axis=1))),
        arms=arm_summaries, checks=checks, native_reconstruction=validation,
        wall_seconds=time.perf_counter() - started, maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        inference='Task-valid ideal response collision on a prescribed paired box; no population prevalence estimate from these cases, no learned decoding, no new historical label. Intervention is reset suppression at t8 only. Literal float equalities and residuals are separate from the ideal theorem.')
    write_json(HERE / 'probe1_results.json', result)
    print(json.dumps(dict(status=result['status'], arms=arm_summaries, checks=checks,
                         wall_seconds=result['wall_seconds']), indent=2), flush=True)
    assert all(checks.values()), 'Preserved failed checks; do not tune the frozen study.'

if __name__ == '__main__':
    main()
