"""Experiment for Task 1 (Lanczos): convergence + orthogonality plots."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running without `pip install -e .` and with bare `python <path>`.
ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.matrix._common import parse_args, write_csv  # noqa: E402
from matcomp.matrix_algorithms.lanczos import lanczos  # noqa: E402
from matcomp.utils.plotting import (  # noqa: E402
    apply_style,
    figures_dir,
    results_dir,
    save_fig,
)
from matcomp.utils.seeding import make_rng  # noqa: E402

TASK = "task1_lanczos"


def main() -> None:
    args = parse_args(TASK)
    apply_style()
    rng = make_rng(args.seed)

    # ---- Setup matrices -----------------------------------------------------
    n = 60
    A_diag = np.diag(np.linspace(1.0, 10.0, n))
    B = rng.standard_normal((n, n))
    A_rand = (B + B.T) / 2

    eig_diag_max = float(np.linalg.eigvalsh(A_diag)[-1])
    eig_rand_max = float(np.linalg.eigvalsh(A_rand)[-1])

    k_grid = list(range(2, n + 1, 2))

    # ---- Convergence of max Ritz value -------------------------------------
    def collect(matvec, true_max: float, *, reorth: bool) -> list[float]:
        return [
            float(
                abs(
                    lanczos(matvec, n=n, k=k, reorthogonalize=reorth, random_seed=args.seed).ritz_values[-1]
                    - true_max
                )
            )
            for k in k_grid
        ]

    diag_err_with = collect(lambda x: A_diag @ x, eig_diag_max, reorth=True)
    diag_err_without = collect(lambda x: A_diag @ x, eig_diag_max, reorth=False)
    rand_err_with = collect(lambda x: A_rand @ x, eig_rand_max, reorth=True)
    rand_err_without = collect(lambda x: A_rand @ x, eig_rand_max, reorth=False)

    fig, ax = plt.subplots()
    ax.semilogy(k_grid, np.maximum(diag_err_with, 1e-16), "o-", label="diagonal, reorth")
    ax.semilogy(k_grid, np.maximum(diag_err_without, 1e-16), "s--", label="diagonal, no reorth")
    ax.semilogy(k_grid, np.maximum(rand_err_with, 1e-16), "^-", label="random sym, reorth")
    ax.semilogy(k_grid, np.maximum(rand_err_without, 1e-16), "v--", label="random sym, no reorth")
    ax.set_xlabel("Lanczos iterations k")
    ax.set_ylabel(r"$|\theta_{\max}(T_k) - \lambda_{\max}(A)|$")
    ax.set_title("Convergence of the largest Ritz value")
    ax.legend()
    save_fig(fig, "ritz_convergence", TASK)

    # ---- Orthogonality loss (only for the reorthogonalised run) ----------
    # The non-reorth branch does not store Q at all (by design — it would
    # nullify the memory savings), so we report the reorth curve only.
    ortho_with = []
    for k in k_grid:
        res_w = lanczos(lambda x: A_rand @ x, n=n, k=k, reorthogonalize=True, random_seed=args.seed)
        QtQ_w = res_w.Q[:, : res_w.iterations].T @ res_w.Q[:, : res_w.iterations]
        ortho_with.append(float(np.linalg.norm(QtQ_w - np.eye(QtQ_w.shape[0]))))
    fig2, ax2 = plt.subplots()
    ax2.semilogy(k_grid, np.maximum(ortho_with, 1e-16), "o-", color="tab:blue", label="reorthogonalised")
    ax2.set_xlabel("Lanczos iterations k")
    ax2.set_ylabel(r"$\|Q_k^\top Q_k - I\|_F$")
    ax2.set_title("Loss of orthogonality (reorthogonalised run)")
    ax2.legend()
    save_fig(fig2, "orthogonality_loss", TASK)

    # ---- CSV table of Ritz values vs eigh on the random symmetric ----------
    res_full = lanczos(
        lambda x: A_rand @ x,
        n=n,
        k=n,
        reorthogonalize=True,
        compute_ritz_vectors=False,
        random_seed=args.seed,
    )
    eig_true = np.linalg.eigvalsh(A_rand)
    rows: list[list[object]] = [["index", "eigh_value", "ritz_value", "abs_error"]]
    for idx, (e, r) in enumerate(zip(eig_true, res_full.ritz_values, strict=True)):
        rows.append([idx, f"{e:.6e}", f"{r:.6e}", f"{abs(e - r):.6e}"])
    write_csv(rows, results_dir(args.out_dir) / f"{TASK}_ritz_table.csv")

    print(f"Lanczos experiment artefacts written to {figures_dir(TASK, args.out_dir)} and {results_dir(args.out_dir)}")


if __name__ == "__main__":
    main()
