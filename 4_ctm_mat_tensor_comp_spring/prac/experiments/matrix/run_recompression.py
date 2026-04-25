"""Experiment for Task 5 (recompression)."""

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
from matcomp.matrix_algorithms.randomized_svd import randomized_svd  # noqa: E402
from matcomp.matrix_algorithms.recompression import recompress_low_rank  # noqa: E402
from matcomp.utils.functional_matrix import evaluate_small  # noqa: E402
from matcomp.utils.low_rank import UVFactors  # noqa: E402
from matcomp.utils.plotting import apply_style, results_dir, save_fig  # noqa: E402
from matcomp.utils.seeding import make_rng  # noqa: E402
from matcomp.utils.test_matrices import HilbertMatrix  # noqa: E402

TASK = "task5_recompression"


def main() -> None:
    args = parse_args(TASK)
    apply_style()
    rng = make_rng(args.seed)

    # ---- Synthetic redundant rank-(2r) → rank-r recovery ------------------
    m, n, true_r = 30, 25, 4
    U_true = rng.standard_normal((m, true_r))
    V_true = rng.standard_normal((n, true_r))
    A = U_true @ V_true.T
    norm_A = float(np.linalg.norm(A))

    rows: list[list[object]] = [
        [
            "input_format",
            "original_rank",
            "new_rank",
            "err_before",
            "err_after",
            "compression_ratio",
        ]
    ]

    # Three scenarios: redundant UV, redundant USV (Task 3), redundant CMR (Task 2).
    redundant_uv = UVFactors(
        U=np.column_stack([U_true, 1e-14 * rng.standard_normal((m, true_r))]),
        V=np.column_stack([V_true, 1e-14 * rng.standard_normal((n, true_r))]),
    )
    err_before = float(np.linalg.norm(A - redundant_uv.reconstruct_small()) / norm_A)
    res = recompress_low_rank(redundant_uv, eps=1e-10)
    err_after = float(np.linalg.norm(A - res.reconstruct_small()) / norm_A)
    rows.append(
        [
            "UVFactors",
            res.original_rank,
            res.new_rank,
            f"{err_before:.6e}",
            f"{err_after:.6e}",
            f"{redundant_uv.factors_memory() / res.factors.factors_memory():.4f}",
        ]
    )

    # USVFactors: take rSVD output at large r and recompress back to true_r.
    H = HilbertMatrix(40, 40)
    A_H = evaluate_small(H)
    norm_H = float(np.linalg.norm(A_H))
    rsvd_res = randomized_svd(H, r=15, oversampling=10, n_power_iter=2, random_seed=args.seed)
    err_before = float(np.linalg.norm(A_H - rsvd_res.reconstruct_small()) / norm_H)
    res2 = recompress_low_rank(rsvd_res.factors, eps=1e-6)
    err_after = float(np.linalg.norm(A_H - res2.reconstruct_small()) / norm_H)
    rows.append(
        [
            "USVFactors",
            rsvd_res.factors.rank,
            res2.new_rank,
            f"{err_before:.6e}",
            f"{err_after:.6e}",
            f"{rsvd_res.factors.factors_memory() / res2.factors.factors_memory():.4f}",
        ]
    )

    # CMRFactors: cross output recompressed.
    cross_res = cross_rank_r(H, r=15, init="maxvol", max_sweeps=8, random_seed=args.seed)
    err_before = float(np.linalg.norm(A_H - cross_res.reconstruct_small()) / norm_H)
    res3 = recompress_low_rank(cross_res.factors, eps=1e-6)
    err_after = float(np.linalg.norm(A_H - res3.reconstruct_small()) / norm_H)
    rows.append(
        [
            "CMRFactors",
            cross_res.factors.rank,
            res3.new_rank,
            f"{err_before:.6e}",
            f"{err_after:.6e}",
            f"{cross_res.factors.factors_memory() / res3.factors.factors_memory():.4f}",
        ]
    )

    write_csv(rows, results_dir(args.out_dir) / f"{TASK}_table.csv")

    # ---- Singular-value tail plot -----------------------------------------
    fig, ax = plt.subplots()
    s_H = np.linalg.svd(A_H, compute_uv=False)
    ax.semilogy(np.arange(1, s_H.size + 1), s_H, "o-", label="$\\sigma_k(A_H)$")
    ax.axvline(res2.new_rank, color="r", linestyle="--", label=f"recompressed rank = {res2.new_rank}")
    ax.set_xlabel("k")
    ax.set_ylabel("singular value")
    ax.set_title("Hilbert spectrum and chosen recompression cut-off")
    ax.legend()
    save_fig(fig, "recompression_cutoff", TASK)

    print("Recompression experiment done.")


if __name__ == "__main__":
    main()
