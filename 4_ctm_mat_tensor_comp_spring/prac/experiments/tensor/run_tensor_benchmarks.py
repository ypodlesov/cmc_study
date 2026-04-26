"""Cross-method tensor benchmark.

Runs every tensor algorithm on a shared dense 3-D test set and a shared
4-D test set. Writes a unified CSV (``reports/results/tensor_benchmarks.csv``)
plus a Pareto plot (relative error vs factor-memory).
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.tensor._common import parse_args, write_csv  # noqa: E402
from matcomp.tensor_algorithms.cp_als_dense import cp_als  # noqa: E402
from matcomp.tensor_algorithms.cp_lm import cp_levenberg_marquardt  # noqa: E402
from matcomp.tensor_algorithms.cp_neural import fit_cp_neural  # noqa: E402
from matcomp.tensor_algorithms.st_hosvd import st_hosvd  # noqa: E402
from matcomp.tensor_algorithms.tt_cross import tt_cross  # noqa: E402
from matcomp.utils.functional_tensor import DenseTensor  # noqa: E402
from matcomp.utils.plotting import apply_style, results_dir, save_fig  # noqa: E402
from matcomp.utils.seeding import make_rng  # noqa: E402
from matcomp.utils.tensor_linalg import cp_to_dense  # noqa: E402
from matcomp.utils.tensor_low_rank import LowRankTensor  # noqa: E402
from matcomp.utils.tensor_metrics import compression_ratio_tensor, rel_frob_tensor  # noqa: E402
from matcomp.utils.tensor_test_objects import (  # noqa: E402
    GaussianKernel3DTensor,
    random_cp,
)

TASK = "tensor_benchmarks"


def _make_tensors(rng: np.random.Generator) -> dict[str, np.ndarray]:
    weights, factors = random_cp((8, 8, 8), 4, rng)
    cp_signal = cp_to_dense(weights, factors)
    return {
        "hilbert3d_8": np.array(
            [[[1.0 / (i + j + k + 1.0) for k in range(8)] for j in range(8)] for i in range(8)]
        ),
        "kernel3d_8": np.array(
            GaussianKernel3DTensor(
                rng.uniform(0.0, 1.0, size=8),
                rng.uniform(0.0, 1.0, size=8),
                rng.uniform(0.0, 1.0, size=8),
                sigma=0.5,
            ).block(
                (np.arange(8, dtype=np.intp), np.arange(8, dtype=np.intp), np.arange(8, dtype=np.intp))
            )
        ),
        "exact_cp4": cp_signal,
    }


def _run_method(name: str, fn, X: np.ndarray, rank: int) -> tuple[LowRankTensor, float]:
    start = perf_counter()
    res = fn(X, rank)
    elapsed = perf_counter() - start
    return res, elapsed


def main() -> None:
    args = parse_args(TASK)
    apply_style()
    rng = make_rng(args.seed)

    tensors = _make_tensors(rng)
    target_ranks = [2, 4, 6]

    rows: list[list[object]] = [
        ["method", "tensor", "rank_or_eps", "rel_frob", "time_s", "compression_ratio", "extra"],
    ]

    def cp_als_method(X: np.ndarray, rank: int) -> LowRankTensor:
        return cp_als(X, rank=rank, init="svd", max_iter=100, tol=1e-10, random_seed=args.seed)

    def cp_neural_method(X: np.ndarray, rank: int) -> LowRankTensor:
        return fit_cp_neural(
            X, rank=rank, mode="dense", optimizer="adam", lr=0.05,
            max_iter=500, tol=None, random_seed=args.seed,
        )

    def cp_lm_method(X: np.ndarray, rank: int) -> LowRankTensor:
        return cp_levenberg_marquardt(
            X, rank=rank, init="als", als_warmup=2, max_iter=30, tol=1e-12, random_seed=args.seed,
        )

    def st_hosvd_method(X: np.ndarray, rank: int) -> LowRankTensor:
        return st_hosvd(X, ranks=tuple([rank] * X.ndim), random_seed=args.seed)

    def tt_cross_method(X: np.ndarray, rank: int) -> LowRankTensor:
        ft = DenseTensor(X)
        return tt_cross(ft, ranks=rank, max_sweeps=5, tol=None, random_seed=args.seed)

    methods = {
        "cp_als": cp_als_method,
        "cp_neural": cp_neural_method,
        "cp_lm": cp_lm_method,
        "st_hosvd": st_hosvd_method,
        "tt_cross": tt_cross_method,
    }

    fig, ax = plt.subplots()
    for tname, X in tensors.items():
        for r in target_ranks:
            for mname, fn in methods.items():
                try:
                    res, elapsed = _run_method(mname, fn, X, r)
                except Exception as exc:
                    rows.append([mname, tname, r, "fail", "", "", str(exc)[:80]])
                    continue
                Xhat = res.reconstruct_small()
                err = rel_frob_tensor(X, Xhat)
                comp = compression_ratio_tensor(res)
                rows.append([mname, tname, r, f"{err:.6e}", f"{elapsed:.4f}", f"{comp:.3f}", ""])
                ax.scatter(comp, max(err, 1e-16), label=f"{mname} ({tname}, r={r})", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("compression ratio (dense / factor bytes)")
    ax.set_ylabel("relative Frobenius error")
    ax.set_title("Tensor algorithms: error vs compression Pareto")
    ax.legend(fontsize=6, loc="best", ncol=2)
    save_fig(fig, "pareto", TASK)

    write_csv(rows, results_dir(args.out_dir) / "tensor_benchmarks.csv")
    print(f"Tensor benchmark written; rows: {len(rows) - 1}")
    print()
    print("| method | tensor | r | rel_frob | time_s | compression |")
    print("|--------|--------|---|----------|--------|-------------|")
    for row in rows[1:]:
        if len(row) >= 6:
            print(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} |")


if __name__ == "__main__":
    main()
