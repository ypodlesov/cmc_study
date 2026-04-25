"""Experiment for Task 3 (randomized SVD)."""

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
from matcomp.matrix_algorithms.randomized_svd import randomized_svd  # noqa: E402
from matcomp.utils.functional_matrix import evaluate_small  # noqa: E402
from matcomp.utils.plotting import apply_style, results_dir, save_fig  # noqa: E402
from matcomp.utils.seeding import make_rng  # noqa: E402
from matcomp.utils.test_matrices import GaussianKernelMatrix, HilbertMatrix  # noqa: E402

TASK = "task3_rsvd"


def main() -> None:
    args = parse_args(TASK)
    apply_style()
    rng = make_rng(args.seed)

    H = HilbertMatrix(60, 60)
    K = GaussianKernelMatrix(rng.uniform(0.0, 1.0, size=60), rng.uniform(0.0, 1.0, size=60), sigma=0.5)

    # ---- Effect of n_power_iter on Hilbert (slow decay) -------------------
    A_H = evaluate_small(H)
    s_H = np.linalg.svd(A_H, compute_uv=False)
    norm_H = float(np.linalg.norm(A_H))
    ranks = list(range(2, 21, 2))
    fig, ax = plt.subplots()
    for q in [0, 1, 2, 4]:
        errs = []
        for r in ranks:
            res = randomized_svd(H, r=r, oversampling=10, n_power_iter=q, random_seed=args.seed)
            errs.append(float(np.linalg.norm(A_H - res.reconstruct_small()) / norm_H))
        ax.semilogy(ranks, np.maximum(errs, 1e-16), marker="o", label=f"q={q}")
    opt = [float(np.sqrt(np.sum(s_H[r:] ** 2)) / norm_H) for r in ranks]
    ax.semilogy(ranks, np.maximum(opt, 1e-16), "k--", label="SVD optimum")
    ax.set_xlabel("rank r")
    ax.set_ylabel("relative Frobenius error")
    ax.set_title("rSVD on Hilbert: effect of power iterations q")
    ax.legend()
    save_fig(fig, "rsvd_hilbert_q", TASK)

    # ---- Effect of oversampling p on Gaussian kernel ----------------------
    A_K = evaluate_small(K)
    s_K = np.linalg.svd(A_K, compute_uv=False)
    norm_K = float(np.linalg.norm(A_K))
    fig2, ax2 = plt.subplots()
    for p in [0, 4, 10, 25]:
        errs = []
        for r in ranks:
            try:
                res = randomized_svd(K, r=r, oversampling=p, n_power_iter=0, random_seed=args.seed)
                errs.append(float(np.linalg.norm(A_K - res.reconstruct_small()) / norm_K))
            except ValueError:
                errs.append(np.nan)
        ax2.semilogy(ranks, np.maximum(errs, 1e-16), marker="o", label=f"p={p}")
    opt = [float(np.sqrt(np.sum(s_K[r:] ** 2)) / norm_K) for r in ranks]
    ax2.semilogy(ranks, np.maximum(opt, 1e-16), "k--", label="SVD optimum")
    ax2.set_xlabel("rank r")
    ax2.set_ylabel("relative Frobenius error")
    ax2.set_title("rSVD on Gaussian kernel: effect of oversampling p")
    ax2.legend()
    save_fig(fig2, "rsvd_kernel_p", TASK)

    # ---- CSV table ---------------------------------------------------------
    rows: list[list[object]] = [
        ["matrix", "r", "p", "q", "rel_frob", "svd_optimal", "matmat_calls", "rmatmat_calls"]
    ]
    for name, fm, A_full, s in [("hilbert", H, A_H, s_H), ("kernel", K, A_K, s_K)]:
        norm = float(np.linalg.norm(A_full))
        for r in [3, 5, 8]:
            for p in [4, 10]:
                for q in [0, 2]:
                    res = randomized_svd(fm, r=r, oversampling=p, n_power_iter=q, random_seed=args.seed)
                    err = float(np.linalg.norm(A_full - res.reconstruct_small()) / norm)
                    opt_v = float(np.sqrt(np.sum(s[r:] ** 2)) / norm)
                    rows.append(
                        [
                            name,
                            r,
                            p,
                            q,
                            f"{err:.6e}",
                            f"{opt_v:.6e}",
                            res.oracle_counts.matmat_calls,
                            res.oracle_counts.rmatmat_calls,
                        ]
                    )
    write_csv(rows, results_dir(args.out_dir) / f"{TASK}_table.csv")
    print("rSVD experiment done.")


if __name__ == "__main__":
    main()
