"""Experiment for Task 9 (CP-ALS, dense)."""

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
from matcomp.utils.plotting import apply_style, results_dir, save_fig  # noqa: E402
from matcomp.utils.seeding import make_rng  # noqa: E402
from matcomp.utils.tensor_linalg import cp_to_dense  # noqa: E402
from matcomp.utils.tensor_test_objects import random_cp  # noqa: E402
from matcomp.utils.timing import timer  # noqa: E402

TASK = "task9_cp_als"


def main() -> None:
    args = parse_args(TASK)
    apply_style()
    rng = make_rng(args.seed)

    shape = (10, 11, 12)
    true_rank = 4
    weights, factors = random_cp(shape, true_rank, rng)
    X_signal = cp_to_dense(weights, factors)
    norm_signal = float(np.linalg.norm(X_signal))

    rows: list[list[object]] = [["matrix", "init", "rank", "rel_err", "iters", "time_s"]]

    # ---- error vs rank, both inits ----
    fig, ax = plt.subplots()
    ranks = [1, 2, 3, 4, 5, 6]
    for init in ("random", "svd"):
        errs = []
        for r in ranks:
            with timer() as t:
                res = cp_als(
                    X_signal, rank=r, init=init, max_iter=300, tol=1e-12, random_seed=args.seed,
                )
            err = float(np.linalg.norm(X_signal - res.reconstruct_small()) / norm_signal)
            errs.append(err)
            rows.append(["exact_cp", init, r, f"{err:.6e}", res.iterations, f"{t.seconds:.4f}"])
        ax.semilogy(ranks, np.maximum(errs, 1e-16), marker="o", label=f"init={init}")
    ax.set_xlabel("CP rank")
    ax.set_ylabel("relative Frobenius error")
    ax.set_title(f"CP-ALS error vs rank (true rank={true_rank}, shape={shape})")
    ax.legend()
    save_fig(fig, "rel_err_vs_rank", TASK)

    # ---- noisy signal: rank-true_rank fit accuracy vs noise level ----
    fig2, ax2 = plt.subplots()
    noise_levels = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    errs_noisy = []
    for nl in noise_levels:
        X = X_signal + nl * rng.standard_normal(shape).astype(np.float64)
        res = cp_als(X, rank=true_rank, init="svd", max_iter=300, tol=1e-12, random_seed=args.seed)
        err = float(np.linalg.norm(X_signal - res.reconstruct_small()) / norm_signal)
        errs_noisy.append(err)
        rows.append(["noisy", "svd", true_rank, f"{err:.6e}", res.iterations, ""])
    ax2.loglog(np.maximum(noise_levels, 1e-16), np.maximum(errs_noisy, 1e-16), "o-")
    ax2.set_xlabel("noise level")
    ax2.set_ylabel("rel. error to noise-free signal")
    ax2.set_title("CP-ALS robustness to additive noise")
    save_fig(fig2, "noise_robustness", TASK)

    write_csv(rows, results_dir(args.out_dir) / f"{TASK}_table.csv")
    print(f"Task 9 artefacts written; rows: {len(rows) - 1}")


if __name__ == "__main__":
    main()
