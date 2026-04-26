"""Experiment for Task 10 (CP via differentiable optimisation)."""

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
from matcomp.tensor_algorithms.cp_als_dense import cp_als  # noqa: E402
from matcomp.tensor_algorithms.cp_neural import fit_cp_neural  # noqa: E402
from matcomp.utils.plotting import apply_style, results_dir, save_fig  # noqa: E402
from matcomp.utils.seeding import make_rng  # noqa: E402
from matcomp.utils.tensor_linalg import cp_to_dense  # noqa: E402
from matcomp.utils.tensor_test_objects import CPSyntheticTensor, random_cp  # noqa: E402

TASK = "task10_cp_neural"


def main() -> None:
    args = parse_args(TASK)
    apply_style()
    rng = make_rng(args.seed)

    shape = (8, 9, 10)
    rank = 3
    weights, factors = random_cp(shape, rank, rng)
    X = cp_to_dense(weights, factors)
    ft = CPSyntheticTensor(weights, list(factors))
    norm = float(np.linalg.norm(X))

    rows: list[list[object]] = [["method", "step", "loss"]]

    # Adam dense
    res_adam = fit_cp_neural(
        X, rank=rank, mode="dense", optimizer="adam", lr=0.05,
        max_iter=2000, tol=None, random_seed=args.seed,
    )
    # ALS for comparison
    res_als = cp_als(X, rank=rank, init="svd", max_iter=200, tol=1e-12, random_seed=args.seed)

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(1, res_adam.iterations + 1), np.maximum(res_adam.loss_history, 1e-16),
                label="Adam (dense)")
    ax.semilogy(np.arange(1, res_als.iterations + 1), np.maximum(res_als.loss_history, 1e-16),
                label="ALS")
    ax.set_xlabel("step / sweep")
    ax.set_ylabel("loss")
    ax.set_title(f"CP-NN vs CP-ALS (rank={rank}, shape={shape})")
    ax.legend()
    save_fig(fig, "loss_curves", TASK)
    for k, v in enumerate(res_adam.loss_history):
        rows.append(["adam_dense", k, f"{v:.6e}"])
    for k, v in enumerate(res_als.loss_history):
        rows.append(["als", k, f"{v:.6e}"])

    # Batched mode through functional tensor: oracle calls vs batch_size.
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel("optimiser step")
    ax2.set_ylabel("MSE")
    for bs in [128, 512, 2048]:
        res = fit_cp_neural(
            ft, rank=rank, mode="batched", batch_size=bs, optimizer="adam", lr=0.05,
            max_iter=2000, tol=None, log_interval=20, random_seed=args.seed,
        )
        ax2.semilogy(
            np.arange(0, res.iterations) * 20,
            np.maximum(res.loss_history, 1e-16),
            label=f"batch_size={bs}",
        )
        rows.append(["batched", f"final@{bs}", f"{res.loss_history[-1]:.6e}"])
        if res.oracle_counts is not None:
            rows.append(["oracle_calls", f"@{bs}", str(res.oracle_counts.samples_calls)])
    ax2.set_title("Batched CP-NN: loss vs steps for varying batch size")
    ax2.legend()
    save_fig(fig2, "batched_loss", TASK)

    print(f"Task 10 final dense rel err: "
          f"{float(np.linalg.norm(X - res_adam.reconstruct_small()) / norm):.4e}")
    write_csv(rows, results_dir(args.out_dir) / f"{TASK}_table.csv")


if __name__ == "__main__":
    main()
