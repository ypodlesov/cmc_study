"""Experiment for Task 6 (pivoted QR)."""

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
from matcomp.matrix_algorithms.pivoted_qr import pivoted_qr_approx  # noqa: E402
from matcomp.utils.functional_matrix import evaluate_small  # noqa: E402
from matcomp.utils.plotting import apply_style, results_dir, save_fig  # noqa: E402
from matcomp.utils.seeding import make_rng  # noqa: E402
from matcomp.utils.test_matrices import (  # noqa: E402
    GaussianKernelMatrix,
    HilbertMatrix,
)

TASK = "task6_pivoted_qr"


def main() -> None:
    args = parse_args(TASK)
    apply_style()
    rng = make_rng(args.seed)

    matrices = {
        "hilbert": HilbertMatrix(30, 30),
        "kernel": GaussianKernelMatrix(rng.uniform(0, 1, size=30), rng.uniform(0, 1, size=30), sigma=0.4),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    rows: list[list[object]] = [["matrix", "k", "|R_kk|", "sigma_k"]]
    for ax, (name, fm) in zip(axes, matrices.items(), strict=True):
        A = evaluate_small(fm)
        res = pivoted_qr_approx(A)
        s = np.linalg.svd(A, compute_uv=False)
        diag = np.abs(np.diag(res.R))
        ax.semilogy(np.arange(1, diag.size + 1), np.maximum(diag, 1e-16), "o-", label="$|R_{kk}|$")
        ax.semilogy(np.arange(1, s.size + 1), np.maximum(s, 1e-16), "s--", label="$\\sigma_k$")
        ax.set_xlabel("k")
        ax.set_title(name)
        ax.legend()
        for k in range(diag.size):
            rows.append([name, k, f"{diag[k]:.6e}", f"{s[k]:.6e}"])
    axes[0].set_ylabel("magnitude")
    fig.suptitle("Pivoted QR: $|R_{kk}|$ vs singular values")
    save_fig(fig, "pivoted_qr_diag_vs_sigma", TASK)

    # ---- Approximation error vs SVD optimum -------------------------------
    fig2, ax2 = plt.subplots()
    rows_err: list[list[object]] = [["matrix", "r", "qr_err", "svd_optimal", "ratio"]]
    for name, fm in matrices.items():
        A = evaluate_small(fm)
        norm_A = float(np.linalg.norm(A))
        s = np.linalg.svd(A, compute_uv=False)
        ranks = list(range(1, min(A.shape)))
        qr_errs = []
        opts = []
        for r in ranks:
            res = pivoted_qr_approx(A, rank=r)
            err = float(np.linalg.norm(A - res.reconstruct_small()) / norm_A)
            qr_errs.append(err)
            opt = float(np.sqrt(np.sum(s[r:] ** 2)) / norm_A) if r < s.size else 0.0
            opts.append(opt)
            rows_err.append([name, r, f"{err:.6e}", f"{opt:.6e}", f"{err / max(opt, 1e-30):.4f}"])
        ax2.semilogy(ranks, np.maximum(qr_errs, 1e-16), marker="o", label=f"QR {name}")
        ax2.semilogy(ranks, np.maximum(opts, 1e-16), "--", color="grey", label=f"SVD opt {name}")
    ax2.set_xlabel("rank r")
    ax2.set_ylabel("relative Frobenius error")
    ax2.set_title("Pivoted QR vs SVD-optimal approximation")
    ax2.legend()
    save_fig(fig2, "pivoted_qr_vs_svd", TASK)

    write_csv(rows, results_dir(args.out_dir) / f"{TASK}_diag.csv")
    write_csv(rows_err, results_dir(args.out_dir) / f"{TASK}_error_vs_r.csv")
    print("Pivoted QR experiment done.")


if __name__ == "__main__":
    main()
