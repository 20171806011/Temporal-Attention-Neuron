"""Non-scientific fixtures for late-cue data, native dynamics and CE gradients."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np

import independent_data
import independent_scale
import model
import population
import reference
import scale_model
import task_data


ROOT = Path(__file__).resolve().parent
ARMS = ('DEND', 'FULL', 'NOATT', 'GAIN2', 'SCALE', 'SCALE_MEAN')
ATTENTION = ('FULL', 'NOATT')
TRACES = ('SCALE', 'SCALE_MEAN')
DELAYS = (0, 16, 32)
CONDITIONS = ('nominal', 'no_cue', 'no_candidates', 'no_first', 'no_second')
BASE_SEED = 4099901301
WEIGHT_SEED = 4099901302
DIRECTION_SEED = 4099901303


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def equivalence_classes(x):
    groups = {}
    for i, row in enumerate(x):
        groups.setdefault(row.tobytes(), []).append(i)
    return groups.values()


def ceiling(x, y):
    """Exact optimal classification of identical inputs in the balanced fixture."""
    return sum(max(np.bincount(y[indices], minlength=2)) for indices in equivalence_classes(x))/len(y)


def deletion_masks(base):
    return {
        'no_cue': base['first_bit'] != base['second_bit'],
        'no_candidates': np.ones(len(base['y']), dtype=bool),
        'no_first': base['selected'] == 0,
        'no_second': base['selected'] == 1,
    }


def check_data(base, independent):
    assert set(base) == set(independent), 'independent data fields'
    for name in base:
        np.testing.assert_array_equal(base[name], independent[name], err_msg=name)
    y = base['y']
    masks = deletion_masks(base)
    datasets = {}
    for delay in DELAYS:
        cue = 9 + delay
        for condition in CONDITIONS:
            x, q = task_data.assemble(base, delay, condition)
            expected, expected_q = independent_data.assemble(independent, delay, condition)
            assert q == expected_q == 12 + delay and x.shape == (len(y), 20 + delay, 6)
            np.testing.assert_array_equal(x, expected)
            independent_data.check(independent, x, delay, condition)
            assert not np.any(x[:, 9:cue]), 'blank includes every channel'
            assert np.all(x[:, q:, 4] == 1) and not np.any(x[:, :q, 4])
            datasets[delay, condition] = (x, q)
            if condition == 'nominal':
                disagree = masks['no_cue']
                assert ceiling(x[:, :4], y) == .5, 'before the first candidate'
                assert ceiling(x[:, :8], y) == .75, 'only first candidate available'
                assert ceiling(x[:, :cue], y) == .75, 'both candidates without selection'
                assert ceiling(x[disagree, :cue], y[disagree]) == .5, 'conflicting candidates without selection'
                assert ceiling(x[:, :cue+1], y) == 1., 'both candidates and selection'
                assert ceiling(x[:, 9:], y) == .5, 'cue/future alone never reveals the chosen content'
            else:
                mask = masks[condition]
                assert ceiling(x[mask], y[mask]) == .5, f'{condition} masked ceiling'
                assert ceiling(x, y) == (.5 if condition == 'no_candidates' else .75), condition
                changed = datasets[delay, 'nominal'][0].copy()
                if condition == 'no_cue':
                    changed[:, cue, 2:4] = 0
                if condition in ('no_candidates', 'no_first'):
                    changed[:, 4, :2] = 0
                if condition in ('no_candidates', 'no_second'):
                    changed[:, 8, :2] = 0
                np.testing.assert_array_equal(x, changed)
    return datasets


def check_deletion_predictions(x, counts, y, mask):
    xs, cs, labels = x[mask], counts[mask], y[mask]
    for indices in equivalence_classes(xs):
        assert np.count_nonzero(labels[indices] == 0) == np.count_nonzero(labels[indices] == 1)
        np.testing.assert_array_equal(cs[indices], np.broadcast_to(cs[indices[0]], (len(indices), 2)))
    forced = np.argmax(cs, axis=1)
    decided = cs[:, 0] != cs[:, 1]
    correct = np.count_nonzero(decided & (forced == labels))
    assert 2*correct == np.count_nonzero(decided), 'strict BA equals half the coverage'


def check_attention():
    sequences = (np.zeros(24), np.ones(24), np.r_[np.zeros(8), 4., np.zeros(15)],
                 4*(np.arange(24) % 2), np.random.default_rng(DIRECTION_SEED).normal(0, 3, 24))
    fields = (('raw', 'S_raw'), ('S', 'S_t'), ('Q', 'Q'), ('K', 'K'), ('V', 'V'),
              ('C', 'C_t'), ('gate', 'gating'), ('A', 'A_t'))
    n_checks = 0
    for full in (True, False):
        for sequence in sequences:
            history = np.zeros((1, 1, 5))
            oracle = reference.source_attention(sequence, full)
            for t, current in enumerate(sequence):
                _, history, local = model.attention(np.array([[current]]), history, full)
                wanted = np.r_[np.zeros(max(0, 4-t)), sequence[max(0, t-4):t+1]]
                np.testing.assert_array_equal(history.ravel(), wanted)
                specific = (('score', 'attention_scores'), ('alpha', 'attention_weights')) if full else ()
                for actual_key, original_key in fields + specific:
                    np.testing.assert_allclose(np.asarray(local[actual_key]).ravel(),
                                               np.asarray(oracle[t][original_key]).ravel(), atol=1e-10, rtol=1e-10)
                    n_checks += 1
    return n_checks


def check_traces(actual, x, arm):
    traces = actual['traces']
    context = actual['contexts']
    a = actual['weights_fast']
    e = np.minimum(1., np.abs(x[:len(traces)]).sum(axis=2, keepdims=True))
    previous = np.concatenate((np.zeros_like(traces[:, :1]), traces[:, :-1]), axis=1)
    drive = actual['drive']
    np.testing.assert_array_equal(actual['gates'], e)
    assert np.max(np.abs(traces)) <= 1.+1e-12, 'trace bound'
    assert np.all(np.abs(actual['injections']) <= e+1e-12), 'injection bound'
    assert np.max(np.abs(drive)) <= 1.
    np.testing.assert_allclose(context, a*previous[..., 0]+(1.-a)*previous[..., 1],
                               atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual['injections'], e*context, atol=1e-12, rtol=0)
    np.testing.assert_allclose(traces, (1.-.5*e)[..., None]*np.array([.85, .97])*previous
                               + (.5*e*drive)[..., None], atol=1e-12, rtol=0)
    blank = e[..., 0] == 0.
    np.testing.assert_array_equal(actual['injections'][blank], np.zeros_like(actual['injections'][blank]))
    np.testing.assert_allclose(traces[blank], previous[blank]*np.array([.85, .97]), atol=1e-12, rtol=0)
    if arm == 'SCALE_MEAN':
        np.testing.assert_array_equal(a, np.full_like(a, .5))
    else:
        assert np.min(a) >= 1./(1.+np.exp(4.))-1e-12
        assert np.max(a) <= 1./(1.+np.exp(-4.))+1e-12


def run_checks():
    start = time.perf_counter()
    lock = json.loads((ROOT / 'SOURCE_LOCK.json').read_text(encoding='utf-8'))
    for name in ('model.py', 'population.py', 'reference.py', 'original_tan.py'):
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == lock['files'][name], name
    base = task_data.make_base(64, BASE_SEED)
    independent = independent_data.make_base(64, BASE_SEED)
    datasets = check_data(base, independent)
    attention_checks = check_attention()
    weights = population.initialize(WEIGHT_SEED, 6)
    rng = np.random.default_rng(DIRECTION_SEED)
    max_forward_error = max_fd_error = 0.
    fd_directions = replays = scalar_episodes = deletion_checks = trace_checks = 0
    task_norms = []

    def compare(actual, wanted, field):
        nonlocal max_forward_error
        assert actual.shape == wanted.shape, field
        if field in ('counts', 'spikes', 'resources'):
            np.testing.assert_array_equal(actual, wanted, err_msg=field)
        else:
            np.testing.assert_allclose(actual, wanted, atol=1e-10, rtol=1e-10, err_msg=field)
        max_forward_error = max(max_forward_error, float(np.max(np.abs(actual-wanted))))

    for delay in (0, 16):
        for arm in ARMS:
            for condition in CONDITIONS:
                assert time.perf_counter()-start < 120, '120 second fixture cap'
                x, q = datasets[delay, condition]
                actual = scale_model.evaluate(x, weights, arm, q)
                wanted = independent_scale.replay(x, weights, arm, q)
                assert set(actual) == set(wanted)
                for field in actual:
                    compare(actual[field], wanted[field], field)
                if arm in TRACES:
                    check_traces(actual, x, arm)
                    trace_checks += 1
                replays += 1
                if condition != 'nominal':
                    check_deletion_predictions(x, actual['counts'], base['y'], deletion_masks(base)[condition])
                    deletion_checks += 1
                elif arm in ATTENTION:
                    scalar = reference.scalar_oracle(x[:2], weights, arm, q)
                    for field in scalar:
                        compare(actual[field][:2], scalar[field], field)
                    scalar_episodes += 2

            # Whole task-shaped nominal fixture. The anchored proxy is the
            # differentiable extension, with hard decisions fixed at the anchor.
            x, q = datasets[delay, 'nominal']
            y = base['y']
            loss, grad, counts = scale_model.loss_grad(x, y, weights, arm, q)
            anchored_counts, anchor = scale_model.forward(x, weights, arm, q)
            np.testing.assert_array_equal(counts, anchored_counts)
            np.testing.assert_allclose(loss, scale_model.objective(x, y, weights, arm, q, anchor), atol=1e-12)
            reg = .0002*weights/len(weights)
            np.testing.assert_allclose(scale_model.loss_grad(x, y, weights, arm, q, True)[1], reg,
                                       atol=1e-12, rtol=0)
            task_norms.append(float(np.linalg.norm(grad-reg)))
            assert np.all(np.isfinite(grad)) and np.isfinite(loss)
            for _ in range(6):
                assert time.perf_counter()-start < 120, '120 second fixture cap'
                direction = rng.normal(size=len(weights))
                direction /= np.linalg.norm(direction)
                eps = 1e-6
                high = scale_model.objective(x, y, weights+eps*direction, arm, q, anchor)
                low = scale_model.objective(x, y, weights-eps*direction, arm, q, anchor)
                numeric = (high-low)/(2*eps)
                analytic = float(grad @ direction)
                error = abs(numeric-analytic)
                assert error < 2e-5 + 2e-4*max(abs(numeric), abs(analytic)), (arm, delay, error)
                max_fd_error = max(max_fd_error, error)
                fd_directions += 1
    assert max(task_norms) > 1e-6, 'fixture must exercise a nonzero task gradient'
    return dict(fixture_rows=64, fixture_data_seed=BASE_SEED, fixture_weight_seed=WEIGHT_SEED,
                fixture_direction_seed=DIRECTION_SEED, input_configurations=len(datasets),
                independent_replays=replays, scalar_tan_episodes=scalar_episodes,
                attention_local_checks=attention_checks, deletion_prediction_checks=deletion_checks,
                trace_invariant_checks=trace_checks, learned_parameters=len(weights),
                gradient_directions=fd_directions, max_forward_error=max_forward_error,
                max_fd_error=max_fd_error, max_task_gradient_norm=max(task_norms), scientific_seeds_used=False)


def main():
    assert not (ROOT / 'FREEZE.json').exists() and not (ROOT / 'RUN_STATE.json').exists()
    assert not (ROOT / 'PRECHECK.json').exists()
    start = time.perf_counter()
    try:
        result = run_checks()
    except Exception as error:
        failure = ROOT / f'PRECHECK_FAILURE_{len(list(ROOT.glob("PRECHECK_FAILURE_*.json")))+1}.json'
        write_json(failure, dict(status='FAIL', exception=type(error).__name__, reason=str(error),
                                 wall_seconds=time.perf_counter()-start))
        raise
    write_json(ROOT / 'PRECHECK.json', dict(status='PASS', wall_seconds=time.perf_counter()-start, **result))
    print('PRECHECK PASS', result, flush=True)


if __name__ == '__main__':
    main()
