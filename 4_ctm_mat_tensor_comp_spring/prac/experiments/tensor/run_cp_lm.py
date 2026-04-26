"""Experiment for Task 11 (Levenberg–Marquardt for CP)."""

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
from matcomp.tensor_algorithms.cp_lm import cp_levenberg_marquardt  # noqa: E402
from matcomp.utils.plotting import apply_style, results_dir, save_fig  # noqa: E402
from matcomp.utils.seeding import make_rng  # noqa: E402
from matcomp.utils.tensor_linalg import cp_to_dense  # noqa: E402

TASK = "task11_cp_lm"


def _swamp_factors(rng: np.random.Generator, shape: tuple[int, ...], rank: int, eps: float) -> tuple[np.ndarray, list[np.ndarray]]:
    """CP factors with near-collinear columns — the classic ALS swamp scenario."""
    factors = []
    for s in shape:
        cols = rng.standard_normal((s, 1))
        # Build rank columns that are small perturbations of the same direction.
        F = np.tile(cols, (1, rank)) + eps * rng.standard_normal((s, rank))
        factors.append(F.astype(np.float64))
    weights = np.ones(rank, dtype=np.float64)
    return weights, factors


def main() -> None:
    args = parse_args(TASK)
    apply_style()
    rng = make_rng(args.seed)

    shape = (5, 6, 7)
    rank = 3
    weights, factors = _swamp_factors(rng, shape, rank, eps=1e-2)
    X = cp_to_dense(weights, factors)

    res_lm = cp_levenberg_marquardt(
        X, rank=rank, init="als", als_warmup=3, max_iter=80, tol=1e-14, random_seed=args.seed,
    )
    res_als = cp_als(X, rank=rank, init="svd", max_iter=200, tol=1e-14, random_seed=args.seed)

    fig, ax = plt.subplots()
    ax.semilogy(np.arange(len(res_lm.loss_history)), np.maximum(res_lm.loss_history, 1e-30), "o-", label="LM")
    ax.semilogy(np.arange(len(res_als.loss_history)), np.maximum(res_als.loss_history, 1e-30), "s-", label="ALS (rel err)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title(f"CP-LM vs CP-ALS on swamp tensor (shape={shape}, rank={rank})")
    ax.legend()
    save_fig(fig, "loss_curves", TASK)

    fig2, ax2 = plt.subplots()
    ax2.bar(["accepted", "rejected"], [res_lm.accepted_steps, res_lm.rejected_steps])
    ax2.set_ylabel("step count")
    ax2.set_title("LM step accept / reject counts")
    save_fig(fig2, "lm_accept_reject", TASK)

    rows: list[list[object]] = [["metric", "value"]]
    rows.append(["jacobian_shape", str(res_lm.jacobian_shape)])
    rows.append(["lm_accepted", res_lm.accepted_steps])
    rows.append(["lm_rejected", res_lm.rejected_steps])
    rows.append(["lm_final_loss", f"{res_lm.loss_history[-1]:.6e}"])
    rows.append(["als_final_rel_err", f"{res_als.loss_history[-1]:.6e}"])
    write_csv(rows, results_dir(args.out_dir) / f"{TASK}_table.csv")
    print(f"Task 11 artefacts written; LM accepted={res_lm.accepted_steps}, rejected={res_lm.rejected_steps}")


if __name__ == "__main__":
    main()
