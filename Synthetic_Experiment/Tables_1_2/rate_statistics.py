"""Log-log OLS fits and two-stage hierarchical bootstrap of mean-MSE curves."""
from __future__ import annotations
import numpy as np

def loglog_fit(T: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return OLS slope, intercept, and R-squared on natural-log axes."""
    x = np.log(np.asarray(T, dtype=float))
    z = np.log(np.asarray(y, dtype=float))
    xc = x - x.mean()
    slope = float(xc @ z / (xc @ xc))
    intercept = float(z.mean() - slope * x.mean())
    fitted = intercept + slope * x
    residual_ss = float(np.sum((z - fitted) ** 2))
    total_ss = float(np.sum((z - z.mean()) ** 2))
    r_squared = 1.0 - residual_ss / total_ss if total_ss > 0 else float("nan")
    return slope, intercept, r_squared


def hierarchical_bootstrap_means(
    raw: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> np.ndarray:
    """Bootstrap mean-MSE curves from raw shape (n_T, systems, trials)."""
    n_t, n_systems, n_trials = raw.shape
    system_draws = rng.integers(0, n_systems, size=(n_bootstrap, n_systems))
    curves = np.zeros((n_bootstrap, n_t), dtype=float)

    # Repeated systems receive independent lower-level trial resamples.
    for t_index in range(n_t):
        for system_slot in range(n_systems):
            trial_draws = rng.integers(
                0, n_trials, size=(n_bootstrap, n_trials)
            )
            selected = raw[
                t_index,
                system_draws[:, system_slot, None],
                trial_draws,
            ]
            curves[:, t_index] += selected.mean(axis=1) / n_systems
    return curves


def bootstrap_slopes(T: np.ndarray, curves: np.ndarray) -> np.ndarray:
    x = np.log(np.asarray(T, dtype=float))
    xc = x - x.mean()
    return np.log(curves) @ xc / float(xc @ xc)
