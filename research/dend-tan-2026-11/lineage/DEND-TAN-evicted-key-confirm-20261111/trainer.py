"""Run the sealed nominal LEARN recipe and retain every update."""

from pathlib import Path
import time

import numpy as np

import learned_model


RATE = .0025


def fit_one(x, y, q, initial, updates, guard, destination=None):
    """Fit one LEARN block and return its archive, optionally saving it to disk."""
    w = initial.copy()
    m, v = np.zeros_like(w), np.zeros_like(w)
    raw_rows, gradient_rows, count_rows, history, seconds = [], [], [], [], []
    destination = Path(destination) if destination is not None else None

    def archive():
        return dict(initial=initial, theta=w,
                    raw_gradients=np.asarray(raw_rows, dtype=np.float64).reshape(-1, 180),
                    gradients=np.asarray(gradient_rows, dtype=np.float64).reshape(-1, 180),
                    counts=np.asarray(count_rows, dtype=np.uint8).reshape(-1, len(x), 2),
                    history=np.asarray(history, dtype=np.float64).reshape(-1, 4),
                    step_seconds=np.asarray(seconds, dtype=np.float64))

    try:
        for k in range(updates):
            guard()
            started = time.perf_counter()
            loss, gradient, counts = learned_model.loss_grad(x, y, w, 'LEARN', q)
            norm = float(np.linalg.norm(gradient))
            regularizer = .0002*w/180
            assert np.isfinite(loss) and np.isfinite(gradient).all() and np.isfinite(norm)
            assert np.isfinite(counts).all()
            diagnostics = [loss, norm, np.linalg.norm(gradient-regularizer), np.linalg.norm(regularizer)]
            raw_gradient = gradient.copy()
            gradient /= max(1, norm)
            m = .9*m + .1*gradient
            v = .999*v + .001*gradient*gradient
            corrected_m = m/(1-.9**(k+1))
            corrected_v = v/(1-.999**(k+1))
            updated = np.clip(w-RATE*corrected_m/(np.sqrt(corrected_v)+1e-8), -2, 2)
            assert np.isfinite(updated).all()
            raw_rows.append(raw_gradient)
            gradient_rows.append(gradient.copy())
            count_rows.append(counts.astype(np.uint8))
            history.append(diagnostics)
            seconds.append(time.perf_counter()-started)
            w = updated
        guard()
        final_counts, _ = learned_model.forward(x, w, 'LEARN', q)
        assert np.isfinite(final_counts).all()
    except Exception:
        if destination is not None:
            partial = destination.with_name(destination.stem+'_PARTIAL.npz')
            np.savez_compressed(partial, **archive())
        raise

    result = archive()
    result['final_counts'] = final_counts.astype(np.uint8)
    if destination is not None:
        np.savez_compressed(destination, **result)
    return result
