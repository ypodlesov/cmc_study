"""Experiment for Task 4 (adaptive cross with caching)."""

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
from matcomp.matrix_algorithms.adaptive_cross import adaptive_cross_cached  # noqa: E402
from matcomp.utils.functional_matrix import evaluate_small  # noqa: E402
from matcomp.utils.plotting import apply_style, results_dir, save_fig  # noqa: E402
from matcomp.utils.seeding import make_rng  # noqa: E402
from matcomp.utils.test_matrices import (  # noqa: E402
    CauchyMatrix,
    GaussianKernelMatrix,
    HilbertMatrix,
)

TASK = "task4_aca"


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

    rows: list[list[object]] = [
        [
            "matrix",
            "max_rank",
            "achieved_rank",
            "rel_frob",
            "cache_hits",
            "cache_misses",
            "real_oracle_calls",
        ]
    ]

    fig, ax = plt.subplots()
    max_ranks = list(range(1, 16))
    for name, fm in matrices.items():
        A = evaluate_small(fm)
        norm = float(np.linalg.norm(A))
        errs = []
        for max_r in max_ranks:
            res = adaptive_cross_cached(fm, max_rank=max_r, eps=1e-12, random_seed=args.seed)
            err = float(np.linalg.norm(A - res.reconstruct_small()) / norm)
            errs.append(err)
            rows.append(
                [
                    name,
                    max_r,
                    res.rank,
                    f"{err:.6e}",
                    res.cache_hits,
                    res.cache_misses,
                    res.oracle_counts.total,
                ]
            )
        ax.semilogy(max_ranks, np.maximum(errs, 1e-16), marker="o", label=name)
    ax.set_xlabel("max_rank")
    ax.set_ylabel("relative Frobenius error")
    ax.set_title("Adaptive cross approximation error vs max_rank")
    ax.legend()
    save_fig(fig, "aca_error_vs_max_rank", TASK)

    # Cache hit ratio plot ---------------------------------------------------
    fig2, ax2 = plt.subplots()
    for name, fm in matrices.items():
        ratios = []
        for max_r in max_ranks:
            res = adaptive_cross_cached(fm, max_rank=max_r, eps=1e-12, random_seed=args.seed)
            denom = res.cache_hits + res.cache_misses
            ratios.append(res.cache_hits / denom if denom else 0.0)
        ax2.plot(max_ranks, ratios, marker="o", label=name)
    ax2.set_xlabel("max_rank")
    ax2.set_ylabel("cache hit ratio")
    ax2.set_title("ACA cache effectiveness")
    ax2.legend()
    save_fig(fig2, "aca_cache_ratio", TASK)

    write_csv(rows, results_dir(args.out_dir) / f"{TASK}_table.csv")
    print("ACA experiment done.")


if __name__ == "__main__":
    main()
