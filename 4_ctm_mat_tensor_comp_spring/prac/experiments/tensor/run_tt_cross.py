"""Experiment for Task 13 (TT-cross)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.tensor._common import parse_args, write_csv  # noqa: E402
from matcomp.tensor_algorithms.tt_cross import tt_cross  # noqa: E402
from matcomp.utils.functional_tensor import evaluate_small  # noqa: E402
from matcomp.utils.plotting import apply_style, results_dir, save_fig  # noqa: E402
from matcomp.utils.tensor_test_objects import FunctionTensor  # noqa: E402

TASK = "task13_tt_cross"


def main() -> None:
    args = parse_args(TASK)
    apply_style()

    # 4-D analytic tensor 1 / (i + j + k + l + 1). Smooth, low TT rank in practice.
    target = FunctionTensor(
        func=lambda idx: 1.0 / (sum(idx) + 1.0),
        shape=(8, 8, 8, 8),
    )
    full = evaluate_small(target, max_total=10_000)

    rank_grid = [2, 3, 4, 5, 6]
    sweep_grid = [1, 3, 5, 8]
    rows: list[list[object]] = [["rank", "sweeps", "rel_err", "oracle_calls", "cache_hits"]]

    fig, ax = plt.subplots()
    for r in rank_grid:
        errs = []
        oracles = []
        for sw in sweep_grid:
            res = tt_cross(target, ranks=r, max_sweeps=sw, tol=None, random_seed=args.seed)
            err = float(np.linalg.norm(full - res.reconstruct_small(max_total=20_000)) / np.linalg.norm(full))
            errs.append(err)
            oracles.append(res.oracle_counts.total)
            rows.append([r, sw, f"{err:.6e}", res.oracle_counts.total, res.cache_hits])
        ax.semilogy(sweep_grid, np.maximum(errs, 1e-16), marker="o", label=f"rank={r}")
    ax.set_xlabel("max sweeps")
    ax.set_ylabel("relative Frobenius error")
    ax.set_title("TT-cross on 1/(i+j+k+l+1): error vs sweeps")
    ax.legend()
    save_fig(fig, "err_vs_sweeps", TASK)

    fig2, ax2 = plt.subplots()
    for r in rank_grid:
        oracles_plot = [int(row[3]) for row in rows[1:] if row[0] == r]
        errs_plot = [float(row[2]) for row in rows[1:] if row[0] == r]
        ax2.loglog(np.maximum(oracles_plot, 1), np.maximum(errs_plot, 1e-16), marker="o", label=f"rank={r}")
    ax2.set_xlabel("real oracle calls")
    ax2.set_ylabel("relative Frobenius error")
    ax2.set_title("TT-cross: oracle-calls / error trade-off")
    ax2.legend()
    save_fig(fig2, "oracle_vs_err", TASK)

    write_csv(rows, results_dir(args.out_dir) / f"{TASK}_table.csv")
    print(f"Task 13 artefacts written; rows: {len(rows) - 1}")


if __name__ == "__main__":
    main()
