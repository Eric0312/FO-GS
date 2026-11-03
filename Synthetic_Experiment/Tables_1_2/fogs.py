"""FO-GS estimation and fractional-order system simulation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
from scipy.signal import fftconvolve

def psi_sequence(alpha: float, T: int) -> np.ndarray:
    psi = np.empty(T + 1, dtype=float)
    psi[0] = 1.0
    for j in range(1, T + 1):
        psi[j] = psi[j - 1] * (j - 1 - alpha) / j
    return psi


def precompute_psi(alpha: float, T: int) -> np.ndarray:
    return psi_sequence(alpha, T)


def delta_alpha_prefix_1d(x_prefix: np.ndarray, alpha: float) -> np.ndarray:
    x_prefix = np.asarray(x_prefix, dtype=float).reshape(-1)
    T = x_prefix.shape[0] - 1
    psi = precompute_psi(alpha, T)
    return fftconvolve(x_prefix, psi, mode="full")[: T + 1]


def random_orthogonal_matrix(n: int, rng: np.random.Generator) -> np.ndarray:
    M = rng.normal(size=(n, n))
    Q, R = np.linalg.qr(M)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q @ np.diag(signs)
    return Q


def sample_stable_A(
    n: int,
    rng: np.random.Generator,
    eig_low: float = -0.5,
    eig_high: float = 0.5,
    symmetric: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    eigvals = rng.uniform(eig_low, eig_high, size=n)

    if symmetric:
        Q = random_orthogonal_matrix(n, rng)
        A = Q @ np.diag(eigvals) @ Q.T
    else:
        while True:
            V = rng.normal(size=(n, n))
            if np.linalg.matrix_rank(V) == n and np.linalg.cond(V) < 20:
                break
        A = V @ np.diag(eigvals) @ np.linalg.inv(V)

    return A, eigvals


def simulate_vector_fractional_system(
    A_star: np.ndarray,
    alpha_star_vec: np.ndarray,
    T: int,
    sigma_eta: float,
    rng: np.random.Generator,
    x0: Optional[np.ndarray] = None,
    x0_mean: float | np.ndarray = 0.0,
    x0_std: float = 1.0,
) -> np.ndarray:
    A_star = np.asarray(A_star, dtype=float)
    alpha_star_vec = np.asarray(alpha_star_vec, dtype=float).reshape(-1)
    n = A_star.shape[0]

    if A_star.shape != (n, n):
        raise ValueError("A_star must be square.")
    if alpha_star_vec.shape[0] != n:
        raise ValueError("alpha_star_vec must have length n.")

    x = np.zeros((T + 1, n), dtype=float)

    if x0 is None:
        x[0] = rng.normal(loc=x0_mean, scale=x0_std, size=n)
    else:
        x[0] = np.asarray(x0, dtype=float).reshape(n)

    psi_list = [precompute_psi(float(alpha_star_vec[i]), T) for i in range(n)]

    for t in range(T):
        eta = rng.normal(0.0, sigma_eta, size=n) if sigma_eta > 0 else np.zeros(n)
        mem = np.zeros(n, dtype=float)

        for i in range(n):
            for j in range(1, t + 2):
                mem[i] += psi_list[i][j] * x[t + 1 - j, i]

        x[t + 1] = A_star @ x[t] - mem + eta

    return x


def profiled_loss_and_rowhat_window(
    x_full: np.ndarray,
    row_idx: int,
    alpha: float,
    start: int,
    t_fit: int,
    ridge: float = 1e-8,
):
    """
    Row-wise profiled loss:
        y_i = [Delta^alpha x_{start+1}^{(i)}, ..., Delta^alpha x_{start+t_fit}^{(i)}]
        X   = [x_start, ..., x_{start+t_fit-1}]
    """
    assert start >= 0
    assert t_fit >= 1
    assert start + t_fit <= x_full.shape[0] - 1

    x_full = np.asarray(x_full, dtype=float)
    n = x_full.shape[1]

    x_row_prefix = x_full[: start + t_fit + 1, row_idx]
    y_row_full = delta_alpha_prefix_1d(x_row_prefix, alpha)
    y_row = y_row_full[start + 1 : start + t_fit + 1].reshape(1, t_fit)

    Xprev = x_full[start : start + t_fit].T
    G = Xprev @ Xprev.T + ridge * np.eye(n)
    rhs = y_row @ Xprev.T
    a_hat = np.linalg.lstsq(G, rhs.T, rcond=None)[0].T
    residual = y_row - a_hat @ Xprev
    loss = float(np.sum(residual * residual))

    return loss, a_hat.reshape(-1)


def build_alpha_grid(
    alpha_low: float,
    alpha_high: float,
    eps: Optional[float] = None,
    num: Optional[int] = None,
) -> np.ndarray:
    if eps is None and num is None:
        raise ValueError("At least one of eps or num must be provided.")
    if alpha_low > alpha_high:
        raise ValueError("alpha_low must be <= alpha_high.")

    if eps is not None:
        if eps <= 0:
            raise ValueError("eps must be positive.")
        grid = np.arange(alpha_low, alpha_high + 0.5 * eps, eps, dtype=float)
        if grid[-1] > alpha_high:
            grid[-1] = alpha_high
        if abs(grid[-1] - alpha_high) > 1e-12:
            grid = np.append(grid, alpha_high)
        return np.unique(grid)

    if num is None or num < 2:
        raise ValueError("num must be at least 2 when eps is not used.")
    return np.linspace(alpha_low, alpha_high, num, dtype=float)


def grid_search_alpha_row_window(
    x_full: np.ndarray,
    row_idx: int,
    start: int,
    t_fit: int,
    alpha_low: float,
    alpha_high: float,
    eps: Optional[float] = None,
    num: Optional[int] = None,
    ridge: float = 1e-8,
):
    alphas = build_alpha_grid(alpha_low=alpha_low, alpha_high=alpha_high, eps=eps, num=num)
    losses = np.zeros_like(alphas)

    best_loss = np.inf
    best_alpha = None
    best_row = None

    for k, alpha in enumerate(alphas):
        loss, a_hat = profiled_loss_and_rowhat_window(
            x_full=x_full,
            row_idx=row_idx,
            alpha=float(alpha),
            start=start,
            t_fit=t_fit,
            ridge=ridge,
        )
        losses[k] = loss

        if loss < best_loss:
            best_loss = loss
            best_alpha = float(alpha)
            best_row = a_hat

    return alphas, losses, best_alpha, best_row


def identify_vector_fractional_system(
    x_full: np.ndarray,
    start: int,
    t_fit: int,
    alpha_bounds: np.ndarray,
    eps_vec: Optional[np.ndarray] = None,
    num_vec: Optional[np.ndarray] = None,
    ridge: float = 1e-8,
):
    n = x_full.shape[1]
    alpha_bounds = np.asarray(alpha_bounds, dtype=float)
    if alpha_bounds.shape != (n, 2):
        raise ValueError(f"alpha_bounds must have shape ({n}, 2).")

    if eps_vec is not None:
        eps_vec = np.asarray(eps_vec, dtype=float).reshape(n)
    if num_vec is not None:
        num_vec = np.asarray(num_vec, dtype=int).reshape(n)

    alpha_hat = np.zeros(n, dtype=float)
    A_hat = np.zeros((n, n), dtype=float)
    details = []

    for i in range(n):
        eps_i = None if eps_vec is None else float(eps_vec[i])
        num_i = None if num_vec is None else int(num_vec[i])

        alphas, losses, best_alpha, best_row = grid_search_alpha_row_window(
            x_full=x_full,
            row_idx=i,
            start=start,
            t_fit=t_fit,
            alpha_low=float(alpha_bounds[i, 0]),
            alpha_high=float(alpha_bounds[i, 1]),
            eps=eps_i,
            num=num_i,
            ridge=ridge,
        )

        alpha_hat[i] = best_alpha
        A_hat[i, :] = best_row
        details.append(
            {
                "row_idx": i,
                "alpha_grid": alphas,
                "losses": losses,
                "alpha_hat": best_alpha,
                "a_hat": best_row.copy(),
                "best_loss": float(np.min(losses)),
            }
        )

    return {
        "alpha_hat": alpha_hat,
        "A_hat": A_hat,
        "details": details,
    }


def estimate_fosysid_single_trajectory(
    x: np.ndarray,
    alpha_bounds: np.ndarray,
    eps_vec: Optional[np.ndarray] = None,
    num_vec: Optional[np.ndarray] = None,
    ridge: float = 1e-8,
) -> Dict[str, Any]:
    """
    Method 1 with NO burn-in:
        start = 0, t_fit = T
    """
    x = np.asarray(x, dtype=float)
    T = x.shape[0] - 1
    return identify_vector_fractional_system(
        x_full=x,
        start=0,
        t_fit=T,
        alpha_bounds=alpha_bounds,
        eps_vec=eps_vec,
        num_vec=num_vec,
        ridge=ridge,
    )


def simulate_vector_fractional_system_cdq(
    A_star: np.ndarray,
    alpha_star_vec: np.ndarray,
    T: int,
    sigma_eta: float,
    rng: np.random.Generator,
    x0: np.ndarray | None = None,
    x0_mean: float | np.ndarray = 0.0,
    x0_std: float = 1.0,
    direct_threshold: int = 128,
) -> np.ndarray:
    """Simulate the original recurrence with causal FFT/CDQ convolution.

    For coordinate i the recurrence can be written

        x_i[t] = sum_{j=1}^t c_i[j] x_i[t-j]
                 + sum_{k != i} A[i,k] x_k[t-1] + eta_i[t-1],

    where c_i[1] = A[i,i] - psi_i[1] and c_i[j] = -psi_i[j]
    for j>=2.  Divide-and-conquer adds each already solved source-to-future
    contribution exactly once.  FFT changes only floating-point summation
    order, not the recurrence or random stream.
    """
    A_star = np.asarray(A_star, dtype=float)
    alpha_star_vec = np.asarray(alpha_star_vec, dtype=float).reshape(-1)
    T = int(T)
    n = A_star.shape[0]
    if T < 1:
        raise ValueError("T must be positive")
    if A_star.shape != (n, n):
        raise ValueError("A_star must be square")
    if alpha_star_vec.shape != (n,):
        raise ValueError("alpha_star_vec must have length n")
    if direct_threshold < 1:
        raise ValueError("direct_threshold must be positive")

    x = np.zeros((T + 1, n), dtype=float)
    if x0 is None:
        x[0] = rng.normal(loc=x0_mean, scale=x0_std, size=n)
    else:
        x[0] = np.asarray(x0, dtype=float).reshape(n)

    # NumPy draws the same scalar stream for size=(T,n) as T consecutive
    # size=n calls.  This preserves every established trial seed exactly.
    if sigma_eta > 0:
        x[1:] = rng.normal(0.0, sigma_eta, size=(T, n))

    kernels: list[np.ndarray] = []
    for coordinate, alpha in enumerate(alpha_star_vec):
        psi = np.empty(T + 1, dtype=float)
        psi[0] = 1.0
        for lag in range(1, T + 1):
            psi[lag] = psi[lag - 1] * (lag - 1 - alpha) / lag
        kernel = -psi
        kernel[0] = 0.0
        kernel[1] += A_star[coordinate, coordinate]
        kernels.append(kernel)

    off_diagonal_A = A_star.copy()
    np.fill_diagonal(off_diagonal_A, 0.0)

    def solve(left: int, right: int) -> None:
        if left == right:
            return
        middle = (left + right) // 2
        solve(left, middle)

        sources = x[left : middle + 1]
        needed_kernel = right - left + 1
        target_slice = slice(middle + 1 - left, right - left + 1)
        for coordinate in range(n):
            kernel = kernels[coordinate][:needed_kernel]
            if min(sources.shape[0], kernel.size) <= direct_threshold:
                convolution = np.convolve(sources[:, coordinate], kernel)
            else:
                convolution = fftconvolve(sources[:, coordinate], kernel)
            x[middle + 1 : right + 1, coordinate] += convolution[target_slice]

        # A's off-diagonal part has lag one, so only the split-boundary source
        # and target lie on opposite sides of this recursion node.
        x[middle + 1] += off_diagonal_A @ x[middle]
        solve(middle + 1, right)

    solve(0, T)
    if not np.all(np.isfinite(x)):
        raise FloatingPointError("Non-finite ultra-long trajectory")
    return x
