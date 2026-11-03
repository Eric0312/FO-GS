"""Recompute paper Tables 1 and 2 from raw FO-GS trial errors."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np
from experiments import FILES, HERE
from rate_statistics import loglog_fit, hierarchical_bootstrap_means, bootstrap_slopes


def analyze_file(path: Path, horizons: list[int], seed: int, bootstrap: int) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing simulation results: {path}. Run experiments.py first, "
            "or specify --data-dir with completed simulation results."
        )
    with np.load(path, allow_pickle=False) as data:
        all_t = data["T_list"]
        indices = [int(np.flatnonzero(all_t == t)[0]) for t in horizons if np.any(all_t == t)]
        if len(indices) != len(horizons):
            raise ValueError(f"Missing expected horizons in {path}")
        raw = data["fosysid_alpha_mse_trials"][indices]
        if raw.shape != (len(horizons), 5, 20):
            raise ValueError(f"Expected 5 systems x 20 trials at each horizon: {path}")
        if not np.all(np.isfinite(raw)) or np.any(raw < 0):
            raise ValueError(f"Invalid or incomplete trial errors: {path}")
        if "completed" in data and not data["completed"][indices].all():
            raise ValueError(f"Incomplete simulation: {path}")
    t = np.array(horizons, dtype=float)
    means = raw.mean(axis=(1, 2))
    if np.any(means <= 0):
        raise ValueError(f"Log-log fits require positive MSE: {path}")
    curves = hierarchical_bootstrap_means(raw, bootstrap, np.random.default_rng(seed))
    if np.any(curves <= 0):
        raise ValueError(f"Bootstrap produced zero MSE: {path}")
    windows = [("All horizons", np.ones(t.size, dtype=bool))]
    if path.name.startswith("table2"):
        windows.append(("t >= 3200", t >= 3200))
    rows = []
    for window, mask in windows:
        slope, _, r2 = loglog_fit(t[mask], means[mask])
        low, high = np.quantile(bootstrap_slopes(t[mask], curves[:, mask]), [0.025, 0.975])
        rows.append(dict(
            dataset=path.stem, fitting_window=window, T_fit=",".join(str(int(x)) for x in t[mask]),
            MSE_exponent=slope, CI95_low=float(low), CI95_high=float(high),
            CI95_minus=slope-float(low), CI95_plus=float(high)-slope,
            R_squared=r2, RMSE_exponent=slope/2, bootstrap=bootstrap, bootstrap_seed=seed,
        ))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_latex(path: Path, rows: list[dict], table: int) -> None:
    lines = [r"\begin{tabular}{llrr}", r"\hline"]
    lines.append(r"Regime & MSE exponent (95\% CI) & $R^2$ & RMSE rate \\" if table == 1
                 else r"Dimension & Fitting window & MSE exponent (95\% CI) & RMSE rate \\")
    lines.append(r"\hline")
    for row in rows:
        n = row["dataset"].rsplit("n", 1)[1]
        regime = ("Large" if "large" in row["dataset"] else "Small") + rf" $\alpha_\star$ ($n={n}$)"
        exponent = f"${row['MSE_exponent']:.3f}^{{+{row['CI95_plus']:.3f}}}_{{-{row['CI95_minus']:.3f}}}$"
        rate = f"$t^{{{row['RMSE_exponent']:.3f}}}$"
        cells = [regime, exponent, f"{row['R_squared']:.3f}", rate] if table == 1 else [
            f"$n={n}$", "All horizons" if row["fitting_window"] == "All horizons" else r"$t\geq 3200$", exponent, rate]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "regenerated")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap < 2:
        parser.error("--bootstrap must be at least 2")
    tables = {1: [], 2: []}
    for filename, (_, horizons, seed) in FILES.items():
        rows = analyze_file(args.data_dir / filename, horizons, seed, args.bootstrap)
        tables[int(filename[5])].extend(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for number, rows in tables.items():
        write_csv(args.output_dir / f"table_{number}.csv", rows)
        write_latex(args.output_dir / f"table_{number}.tex", rows, number)
        print(f"Table {number}")
        for row in rows:
            print(f"  {row['dataset']}, {row['fitting_window']}: "
                  f"MSE {row['MSE_exponent']:.6f} "
                  f"[{row['CI95_low']:.6f}, {row['CI95_high']:.6f}], "
                  f"R2={row['R_squared']:.6f}, RMSE={row['RMSE_exponent']:.6f}")
    print(f"Saved CSV and LaTeX tables to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
