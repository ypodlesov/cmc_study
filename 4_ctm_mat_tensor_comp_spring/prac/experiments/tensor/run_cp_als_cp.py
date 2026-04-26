"""Experiment for Task 8 (CP-ALS from a CP source)."""

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
from matcomp.tensor_algorithms.cp_als_cp import cp_als_from_cp  # noqa: E402
from matcomp.utils.plotting import apply_style, results_dir, save_fig  # noqa: E402
from matcomp.utils.seeding import make_rng  # noqa: E402
from matcomp.utils.tensor_low_rank import CPFactors  # noqa: E402
from matcomp.utils.tensor_test_objects import random_cp  # noqa: E402

TASK = "task8_cp_als_cp"


def main() -> None:
    args = parse_args(TASK)
    apply_style()
    rng = make_rng(args.seed)

    shape = (10, 12, 14)
    R_src = 8
    weights, factors = random_cp(shape, R_src, rng)
    src = CPFactors(weights=weights, factors=tuple(factors))

    target_ranks = [1, 2, 3, 4, 5, 6, 7, 8]
    rows: list[list[object]] = [["target_rank", "iters", "final_loss", "converged"]]
    fig, ax = plt.subplots()
    for r in target_ranks:
        res = cp_als_from_cp(
            src, target_rank=r, init="random", max_iter=120, tol=1e-12, random_seed=args.seed,
        )
        ax.semilogy(np.arange(1, res.iterations + 1), np.maximum(res.loss_history, 1e-16),
                    marker="o", label=f"R_target={r}")
        rows.append([r, res.iterations, f"{res.loss_history[-1]:.6e}", str(res.converged)])
    ax.set_xlabel("ALS sweep")
    ax.set_ylabel("relative Frobenius error")
    ax.set_title(f"CP-ALS from CP source (R_source={R_src}, shape={shape})")
    ax.legend(fontsize=8)
    save_fig(fig, "loss_vs_iter", TASK)
    write_csv(rows, results_dir(args.out_dir) / f"{TASK}_table.csv")
    print(f"Task 8 artefacts written; rows: {len(rows) - 1}")


if __name__ == "__main__":
    main()
