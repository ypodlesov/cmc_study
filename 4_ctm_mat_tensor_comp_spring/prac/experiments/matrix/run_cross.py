"""Experiment for Task 2 (cross / skeleton): error vs r on Hilbert / Cauchy / kernel."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.matrix._common import parse_args, write_csv  # noqa: E402
from matcomp.matrix_algorithms.cross import cross_rank_r  # noqa: E402
from matcomp.utils.functional_matrix import evaluate_small  # noqa: E402
from matcomp.utils.plotting import apply_style, results_dir, save_fig  # noqa: E402
from matcomp.utils.seeding import make_rng  # noqa: E402
from matcomp.utils.test_matrices import (  # noqa: E402
    CauchyMatrix,
    GaussianKernelMatrix,
    HilbertMatrix,
)

TASK = "task2_cross"


def main() -> None:
    args = parse_args(TASK)
    apply_style()
    rng = make_rng(args.seed)

    matrices = {
        "hilbert": HilbertMatrix(40, 40),
        "cauchy": CauchyMatrix(
            1.0 + rng.uniform(0.0, 1.0, size=40),
            0.5 + rng.uniform(0.0, 1.0, size=40),
        ),
        "gaussian_kernel": GaussianKernelMatrix(
            rng.uniform(0.0, 1.0, size=40),
            rng.uniform(0.0, 1.0, size=40),
            sigma=0.4,
        ),
    }
    inits = ["random", "maxvol", "greedy"]
    ranks = list(range(1, 13))

    rows: list[list[object]] = [["matrix", "init", "r", "rel_frob", "svd_optimal", "ratio", "oracle_calls"]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (name, fm) in zip(axes, matrices.items(), strict=True):
        A = evaluate_small(fm)
        norm_A = float(np.linalg.norm(A))
        s = np.linalg.svd(A, compute_uv=False)
        for init in inits:
            errs = []
            for r in ranks:
                res = cross_rank_r(fm, r=r, init=init, random_seed=args.seed, max_sweeps=8)  # type: ignore[arg-type]
                err = float(np.linalg.norm(A - res.reconstruct_small()) / norm_A)
                errs.append(err)
                opt = float(np.sqrt(np.sum(s[r:] ** 2)) / norm_A) if r < s.size else 0.0
                rows.append([name, init, r, f"{err:.6e}", f"{opt:.6e}", f"{err / max(opt, 1e-30):.4f}", res.oracle_counts.total])
            ax.semilogy(ranks, np.maximum(errs, 1e-16), marker="o", label=f"cross ({init})")
        opt_curve = [float(np.sqrt(np.sum(s[r:] ** 2)) / norm_A) for r in ranks]
        ax.semilogy(ranks, np.maximum(opt_curve, 1e-16), "k--", label="SVD optimum")
        ax.set_xlabel("rank r")
        ax.set_title(name)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("relative Frobenius error")
    fig.suptitle("Cross approximation error vs rank")
    save_fig(fig, "cross_error_vs_r", TASK)

    write_csv(rows, results_dir(args.out_dir) / f"{TASK}_table.csv")
    print("Cross experiment done. Plots and CSV written.")


if __name__ == "__main__":
    main()
