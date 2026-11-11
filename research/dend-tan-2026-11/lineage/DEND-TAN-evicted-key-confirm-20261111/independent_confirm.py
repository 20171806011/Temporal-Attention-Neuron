"""Independent confirmation stimuli and reconstruction of recorded training updates."""

import numpy as np

import independent_learned as learner


HISTORIES = ('four', 'three0', 'three2', 'two')
SCHEDULES = ('A', 'B', 'C', 'D')
WRITE_TICKS = ((0, 3, 6, 11), (1, 4, 7, 13), (0, 2, 5, 10), (2, 5, 8, 12))
DISTRACTOR_TICKS = ((2, 8), (0, 10), (1, 8), (3, 6))
FIRST_READ = (22, 24, 20, 23)
T, Q = 36, 28


def assemble(base, history):
    """Generate the official evaluation input without the production assembler."""
    kept = {'four': (0, 1, 2, 3), 'three0': (0, 2, 3), 'three2': (1, 2, 3), 'two': (2, 3)}[history]
    x = np.zeros((len(base['y']), T, 6))
    for row, (group, selected, payload_bits) in enumerate(zip(base['group'], base['selected'], base['bits'])):
        schedule = group % len(SCHEDULES)
        order = base['write_order'][group]
        for position in kept:
            tick = WRITE_TICKS[schedule][position]
            x[row, tick, payload_bits[position]] = base['payload_amplitude'][group, position]
            x[row, tick, 2:4] = base['key_vectors'][group, order[position]]
            x[row, tick, 5] = 1.
        x[row, DISTRACTOR_TICKS[schedule], 4] = base['distractor'][group]
        first = FIRST_READ[schedule]
        x[row, first:first+3, 2:4] = base['query_vectors'][group, order[selected]]
        x[row, Q:, 4] = 1.
    return x, Q


def initialize(seed):
    """Use the sealed LEARN parameter ordering with independently drawn initial weights."""
    rng = np.random.default_rng(seed)
    input_weights = rng.normal(0., .45, (6, 12))
    recurrent_weights = rng.normal(0., .25, (6, 12))
    output_weights = rng.normal(0., .5, (6, 2))
    return np.r_[input_weights.ravel(), recurrent_weights.ravel(), np.full(12, .15),
                 np.full(6, .05), output_weights.ravel(), np.full(2, .1), np.zeros(4)]


def first_read_match(x, w):
    """Reconstruct the scores and softmax from the actual first observed read."""
    key = x[..., 2:4]
    read = (1.-x[..., 5])*np.minimum(1., np.linalg.norm(key, axis=-1))
    active = read > 0.
    assert np.all(active.any(axis=1))
    rows = np.arange(len(x))
    first = active.argmax(axis=1)
    latest = np.zeros((len(x), 2, 2))
    for row, stop in enumerate(first):
        writes = np.flatnonzero(x[row, :stop, 5] == 1.)[-2:][::-1]
        latest[row, :len(writes)] = key[row, writes]
    projected = np.einsum('ni,ij->nj', key[rows, first], w[176:].reshape(2, 2), optimize=False)
    scores = 4*np.einsum('ni,nki->nk', projected, latest, optimize=False)
    e = np.exp(scores-scores.max(axis=1, keepdims=True))
    return first, scores, e/e.sum(axis=1, keepdims=True)


def verify_training_archive(x, y, q, archive):
    """Replay recorded Adam updates and objectives, independently checking endpoint counts."""
    fields = {'initial', 'theta', 'raw_gradients', 'gradients', 'counts', 'history', 'step_seconds', 'final_counts'}
    assert set(archive) == fields, 'training archive fields'
    initial, theta = archive['initial'], archive['theta']
    raw, clipped, counts, history, seconds = (archive[key] for key in
        ('raw_gradients', 'gradients', 'counts', 'history', 'step_seconds'))
    updates = len(history)
    assert updates > 0 and x.shape == (len(y), T, 6) and q == Q, 'training input shape and count window'
    assert np.all((y == 0) | (y == 1)), 'binary training labels'
    assert initial.shape == theta.shape == (180,) and raw.shape == clipped.shape == (updates, 180), 'parameter shapes'
    assert counts.shape == (updates, len(y), 2) and history.shape == (updates, 4), 'training history shapes'
    assert counts.dtype == np.uint8 and np.all(counts <= T-q), 'eight-step native counts'
    float_arrays = (initial, theta, raw, clipped, history, seconds)
    assert all(a.dtype == np.float64 and np.isfinite(a).all() for a in float_arrays), 'finite float64 training archive'
    assert seconds.shape == (updates,) and np.all(seconds > 0.), 'positive recorded update durations'
    checks, max_error = 8, 0.

    def close(actual, expected, context, tolerance=1e-10):
        nonlocal checks, max_error
        error = float(np.max(np.abs(actual-expected)))
        assert error <= tolerance, f'{context}: {error:.17g} exceeds {tolerance}'
        max_error = max(error, max_error)
        checks += 1

    w = initial.copy()
    first_moment, second_moment = np.zeros_like(w), np.zeros_like(w)
    endpoint_replays = 0
    signs = 1-2*np.asarray(y, dtype=float)
    for k, (raw_step, g, c, (loss, norm, task_norm, regularizer_norm)) in enumerate(zip(raw, clipped, counts, history)):
        context = f'update {k+1}'
        margin = .5*(c[:, 1].astype(float)-c[:, 0].astype(float))
        objective = float(np.mean(np.logaddexp(0., signs*margin))+.0001*np.mean(w*w))
        regularizer = .0002*w/180
        assert min(norm, task_norm, regularizer_norm) >= 0., f'{context}: nonnegative gradient norms'
        assert np.array_equal(g, raw_step/max(1., norm)), f'{context}: exact global clipping'
        checks += 2
        close(loss, objective, f'{context}: count CE + L2')
        close(norm, np.linalg.norm(raw_step), f'{context}: raw gradient norm')
        close(np.linalg.norm(g), min(1., norm), f'{context}: clipped norm')
        close(task_norm, np.linalg.norm(raw_step-regularizer), f'{context}: task gradient norm')
        close(regularizer_norm, np.linalg.norm(regularizer), f'{context}: regularizer norm')
        if k in (0, updates-1):
            expected_counts = learner.replay(x, w, 'LEARN', q)['counts']
            assert np.array_equal(c, expected_counts), f'{context}: independent native counts'
            checks += 1
            endpoint_replays += 1
        first_moment = .9*first_moment+.1*g
        second_moment = .999*second_moment+.001*g*g
        corrected_m = first_moment/(1-.9**(k+1))
        corrected_v = second_moment/(1-.999**(k+1))
        w = np.clip(w-.0025*corrected_m/(np.sqrt(corrected_v)+1e-8), -2., 2.)
    close(theta, w, 'reconstructed final Adam parameters')
    assert np.array_equal(theta, w), 'bit-exact reconstructed final Adam parameters'
    final_counts = learner.replay(x, w, 'LEARN', q)['counts']
    saved_final = archive['final_counts']
    assert saved_final.dtype == np.uint8 and np.array_equal(saved_final, final_counts), 'independent post-update native counts'
    final_margin = .5*(final_counts[:, 1]-final_counts[:, 0])
    final_loss = float(np.logaddexp(0., signs*final_margin).mean()+.0001*np.mean(w*w))
    checks += 2
    return dict(checks=checks, max_numeric_error=max_error, updates=updates, endpoint_replays=endpoint_replays,
                final_loss=final_loss)
