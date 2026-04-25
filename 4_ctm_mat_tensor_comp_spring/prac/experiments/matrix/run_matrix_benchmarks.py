"""Experiment for Task 7 (cross-task SVD benchmark)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.matrix._common import parse_args, write_csv  # noqa: E402
from matcomp.matrix_algorithms.adaptive_cross import adaptive_cross_cached  # noqa: E402
from matcomp.matrix_algorithms.cross import cross_rank_r  # noqa: E402
from matcomp.matrix_algorithms.pivoted_qr import pivoted_qr_approx  # noqa: E402
from matcomp.matrix_algorithms.randomized_svd import randomized_svd  # noqa: E402
from matcomp.matrix_algorithms.svd_control import (  # noqa: E402
    benchmark_against_svd,
    benchmark_to_csv_rows,
    reference_svd_method,
)
from matcomp.utils.functional_matrix import FunctionalMatrix, evaluate_small  # noqa: E402
from matcomp.utils.low_rank import LowRankApprox  # noqa: E402
from matcomp.utils.plotting import apply_style, results_dir, save_fig  # noqa: E402
from matcomp.utils.seeding import make_rng  # noqa: E402
from matcomp.utils.test_matrices import (  # noqa: E402
    CauchyMatrix,
    GaussianKernelMatrix,
    HilbertMatrix,
)

TASK = "matrix_benchmarks"


def main() -> None:
    args = parse_args(TASK)
    apply_style()
    rng = make_rng(args.seed)

    matrices = {
        "hilbert30": HilbertMatrix(30, 30),
        "cauchy30": CauchyMatrix(
            1.0 + rng.uniform(0.0, 1.0, size=30),
            0.5 + rng.uniform(0.0, 1.0, size=30),
        ),
        "kernel30": GaussianKernelMatrix(
            rng.uniform(0.0, 1.0, size=30),
            rng.uniform(0.0, 1.0, size=30),
            sigma=0.4,
        ),
    }

    def cross_method(fm: FunctionalMatrix, r: int) -> LowRankApprox:
        return cross_rank_r(fm, r=r, init="maxvol", max_sweeps=8, random_seed=args.seed)

    def rsvd_method(fm: FunctionalMatrix, r: int) -> LowRankApprox:
        return randomized_svd(fm, r=r, oversampling=10, n_power_iter=2, random_seed=args.seed)

    def aca_method(fm: FunctionalMatrix, r: int) -> LowRankApprox:
        return adaptive_cross_cached(fm, max_rank=r, eps=1e-12, random_seed=args.seed)

    def pqr_method(fm: FunctionalMatrix, r: int) -> LowRankApprox:
        return pivoted_qr_approx(evaluate_small(fm), rank=r)

    methods = {
        "svd_ref": reference_svd_method,
        "cross_maxvol": cross_method,
        "rsvd_q2": rsvd_method,
        "aca_cached": aca_method,
        "pivoted_qr": pqr_method,
    }
    ranks = [2, 4, 6, 8, 10, 12, 15]

    rows = benchmark_against_svd(matrices, methods, ranks)
    write_csv(benchmark_to_csv_rows(rows), results_dir(args.out_dir) / "benchmarks.csv")

    # ---- Plot grid: one subplot per matrix --------------------------------
    fig, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4.2), sharey=True)
    for ax, (name, _fm) in zip(axes, matrices.items(), strict=True):
        for method_name in methods:
            xs = [r.r for r in rows if r.matrix == name and r.method == method_name]
            ys = [max(r.rel_frob, 1e-16) for r in rows if r.matrix == name and r.method == method_name]
            if not xs:
                continue
            ax.semilogy(xs, ys, marker="o", label=method_name)
        ax.set_xlabel("rank r")
        ax.set_title(name)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("relative Frobenius error")
    fig.suptitle("Matrix-task benchmark: error vs rank")
    save_fig(fig, "benchmark_error_vs_rank", "benchmarks")

    print(f"Benchmark CSV written to {results_dir(args.out_dir) / 'benchmarks.csv'}")
    print("Markdown summary:")
    print()
    # Pretty print top-of-table summary
    print("| method | matrix | r | rel_frob | ratio_frob | time_s | oracle_calls |")
    print("|--------|--------|---|----------|------------|--------|--------------|")
    for r in rows:
        print(
            f"| {r.method} | {r.matrix} | {r.r} | {r.rel_frob:.3e} | "
            f"{r.ratio_to_svd_frob:.3f} | {r.time_s:.4f} | {r.oracle_calls} |"
        )


if __name__ == "__main__":
    main()
