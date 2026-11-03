from __future__ import annotations

import argparse
import os
from pathlib import Path

# Small matrix solves run faster without nested BLAS threads.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
from fogs import (
    estimate_fosysid_single_trajectory,
    sample_stable_A,
    simulate_vector_fractional_system,
    simulate_vector_fractional_system_cdq,
)

HERE = Path(__file__).resolve().parent
SHORT_T = [100, 200, 300, 400]
LONG_T = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
# Original short experiment also had T=300; retain its RNG offset when absent.
SEED_OFFSETS = dict(zip([100, 200, 300, 400, 800, 1600, 3200, 6400, 12800, 25600], range(10)))
CONFIGS = {
    "small_n10": dict(n=10, seed=15063, bounds=(0.01, 0.20), eig=(0.1, 0.2)),
    "small_n20": dict(n=20, seed=16042, bounds=(0.01, 0.20), eig=(0.1, 0.2)),
    "large_n2": dict(n=2, seed=13429, bounds=(0.50, 0.99), eig=(-0.1, 0.1)),
}
FILES = {
    "table1_small_n10.npz": ("small_n10", SHORT_T, 20260802),
    "table1_small_n20.npz": ("small_n20", SHORT_T, 20260803),
    "table1_large_n2.npz": ("large_n2", SHORT_T, 20260804),
    "table2_small_n10.npz": ("small_n10", LONG_T, 20260803),
    "table2_small_n20.npz": ("small_n20", LONG_T, 20260804),
}


def sample_systems(config: dict, pairs: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preserve original interleaved A/alpha draws, including nonsymmetric A."""
    rng = np.random.default_rng(config["seed"])
    matrices, orders, eigenvalues = [], [], []
    for _ in range(pairs):
        matrix, eig = sample_stable_A(
            config["n"], rng, *config["eig"], symmetric=False
        )
        order = rng.uniform(*config["bounds"], size=config["n"])
        matrices.append(matrix)
        orders.append(order)
        eigenvalues.append(eig)
    return np.stack(matrices), np.stack(orders), np.stack(eigenvalues)


def run_trial(config: dict, matrix: np.ndarray, order: np.ndarray,
              horizon: int, pair: int, trial: int, simulator: str = "fast") -> dict:
    seed = 3123 + 100000 * SEED_OFFSETS[horizon] + 1000 * pair + trial
    simulate = (simulate_vector_fractional_system_cdq if simulator == "fast"
                else simulate_vector_fractional_system)
    trajectory = simulate(matrix, order, horizon, 0.1,
                          np.random.default_rng(seed), x0_std=4.0)
    n = config["n"]
    return estimate_fosysid_single_trajectory(
        trajectory, np.tile(config["bounds"], (n, 1)),
        num_vec=np.full(n, 20), ridge=1e-6,
    )


def atomic_save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def run_experiment(name: str, horizons: list[int], pairs: int, trials: int,
                   output: Path, simulator: str) -> None:
    config = CONFIGS[name]
    matrices, orders, eigenvalues = sample_systems(config, pairs)
    n = config["n"]
    shape = (len(horizons), pairs, trials)
    payload = dict(
        T_list=np.array(horizons), n=np.array(n), instance_seed=np.array(config["seed"]),
        N_pairs=np.array(pairs), N_trials=np.array(trials),
        pair_A_true=matrices, pair_alpha_true=orders, pair_eigvals_true=eigenvalues,
        alpha_sampling_bounds=np.array(config["bounds"]),
        fosysid_alpha_bounds=np.tile(config["bounds"], (n, 1)),
        fosysid_num_grid_vec=np.full(n, 20),
        A_eig_low=np.array(config["eig"][0]), A_eig_high=np.array(config["eig"][1]),
        sigma_eta=np.array(0.1), x0_std=np.array(4.0), fosysid_ridge=np.array(1e-6),
        legacy_seed_offsets=np.array([SEED_OFFSETS[t] for t in horizons]),
        simulator=np.array(simulator), completed=np.zeros(shape, dtype=bool),
        fosysid_alpha_hats=np.full(shape + (n,), np.nan),
        fosysid_A_hats=np.full(shape + (n, n), np.nan),
        fosysid_alpha_mse_trials=np.full(shape, np.nan),
        fosysid_A_mse_trials=np.full(shape, np.nan),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        with np.load(output, allow_pickle=False) as old:
            for key in payload:
                if key == "completed" or key.startswith("fosysid_") and key.endswith(("hats", "trials")):
                    if old[key].shape != payload[key].shape:
                        raise ValueError(f"Checkpoint shape mismatch: {output}: {key}")
                    payload[key] = old[key].copy()
                elif not np.array_equal(old[key], payload[key]):
                    raise ValueError(f"Checkpoint settings differ: {output}: {key}")
    for ti, horizon in enumerate(horizons):
        for pair in range(pairs):
            for trial in range(trials):
                if payload["completed"][ti, pair, trial]:
                    continue
                result = run_trial(config, matrices[pair], orders[pair], horizon, pair, trial, simulator)
                for parameter, truth in (("alpha", orders[pair]), ("A", matrices[pair])):
                    estimate = result[f"{parameter}_hat"]
                    if not np.all(np.isfinite(estimate)):
                        raise FloatingPointError("Non-finite parameter estimate")
                    payload[f"fosysid_{parameter}_hats"][ti, pair, trial] = estimate
                    payload[f"fosysid_{parameter}_mse_trials"][ti, pair, trial] = np.mean((estimate - truth) ** 2)
                payload["completed"][ti, pair, trial] = True
                atomic_save(output, payload)
        print(f"{name}: T={horizon} complete -> {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=["all", *CONFIGS], default="all")
    parser.add_argument("--table", choices=["all", "1", "2"], default="all")
    parser.add_argument("--output-dir", type=Path, default=HERE / "regenerated")
    parser.add_argument("--simulator", choices=["fast", "direct"], default="fast")
    parser.add_argument("--horizons", type=int, nargs="+", help="Optional subset for smoke runs")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.pairs <= 5 or not 1 <= args.trials <= 20:
        parser.error("Use 1..5 pairs and 1..20 trials")
    if args.table == "2" and args.experiment == "large_n2":
        parser.error("Table 2 only contains small-alpha systems")
    if args.horizons and (len(set(args.horizons)) != len(args.horizons) or
                          any(t not in SEED_OFFSETS for t in args.horizons)):
        parser.error("Horizons must be unique members of the documented seed map")
    for filename, (name, horizons, _) in FILES.items():
        if args.experiment not in ("all", name) or args.table not in ("all", filename[5]):
            continue
        selected = sorted(args.horizons) if args.horizons else horizons
        if any(t not in horizons for t in selected):
            parser.error(f"Horizons must be a subset of {horizons} for {filename}")
        run_experiment(name, selected, args.pairs, args.trials,
                       args.output_dir / filename, args.simulator)


if __name__ == "__main__":
    main()
