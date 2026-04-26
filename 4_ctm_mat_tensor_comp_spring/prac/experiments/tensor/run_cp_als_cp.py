"""Experiment for Task 8 (CP-ALS from a CP source).

Sweeps target rank ``R_target ∈ {1, ..., R_src}`` for two init
strategies (``random`` and ``source``), with multiple random restarts
per (R_target, init) so the report can record best and mean loss as
required by the PDF (Task 8: *"Compare several random initializations"*
and *"With multiple runs, the report must record the best and mean
result"*).

CSV columns
-----------
``R_target, init, n_runs, best_loss, mean_loss, std_loss, mean_iters,
mean_time_s, n_tol_converged, dense_check_for_best``

* ``best_loss`` / ``mean_loss`` / ``std_loss`` — relative Frobenius
  error (CP-vs-CP inner-product formula) across the ``n_runs`` restarts.
* ``mean_iters`` — mean number of completed ALS sweeps.
* ``mean_time_s`` — mean wall-clock seconds per run.
* ``n_tol_converged`` — restarts that hit the |Δloss| < tol plateau test
  (NOT a global-optimum certificate).
* ``dense_check_for_best`` — independent ``‖X - Ŷ‖_F / ‖X‖_F`` computed
  on the dense source for the best restart. Catches catastrophic
  cancellation in the CP-vs-CP loss formula at very small errors.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.tensor._common import parse_args, write_csv  # noqa: E402
from matcomp.tensor_algorithms.cp_als_cp import cp_als_from_cp  # noqa: E402
from matcomp.utils.plotting import apply_style, results_dir, save_fig  # noqa: E402
from matcomp.utils.seeding import make_rng  # noqa: E402
from matcomp.utils.tensor_linalg import cp_to_dense  # noqa: E402
from matcomp.utils.tensor_low_rank import CPFactors  # noqa: E402
from matcomp.utils.tensor_test_objects import random_cp  # noqa: E402

TASK = "task8_cp_als_cp"

N_REPLICATES = 10
MAX_ITER = 200
TOL = 1e-10


def _run_one(src: CPFactors, target_rank: int, init: str, seed: int):
    t0 = time.perf_counter()
    res = cp_als_from_cp(
        src,
        target_rank=target_rank,
        init=init,
        max_iter=MAX_ITER,
        tol=TOL,
        random_seed=seed,
    )
    return res, time.perf_counter() - t0


def main() -> None:
    args = parse_args(TASK)
    apply_style()
    rng = make_rng(args.seed)

    shape = (10, 12, 14)
    R_src = 8
    weights, factors = random_cp(shape, R_src, rng)
    src = CPFactors(weights=weights, factors=tuple(factors))
    src_dense = cp_to_dense(weights, factors)
    src_dense_norm = float(np.linalg.norm(src_dense))

    target_ranks = list(range(1, R_src + 1))
    inits = ("random", "source")

    rows: list[list[object]] = [[
        "R_target", "init", "n_runs",
        "best_loss", "mean_loss", "std_loss",
        "mean_iters", "mean_time_s",
        "n_tol_converged",
        "dense_check_for_best",
    ]]

    fig, axes = plt.subplots(1, len(inits), figsize=(11.5, 4.6), sharey=True)
    axes_iter = np.atleast_1d(axes)

    for ax, init in zip(axes_iter, inits, strict=True):
        for r in target_ranks:
            histories: list[np.ndarray] = []
            losses: list[float] = []
            iters_log: list[int] = []
            times_log: list[float] = []
            tol_conv = 0
            best_loss = float("inf")
            best_res = None
            for k in range(N_REPLICATES):
                # Distinct, deterministic seed per (R_target, init, replicate).
                run_seed = int(args.seed) * 1_000_000 + r * 1000 + k * 7 + (1 if init == "source" else 0)
                res, dt = _run_one(src, r, init, run_seed)
                histories.append(np.asarray(res.loss_history, dtype=float))
                losses.append(float(res.loss_history[-1]))
                iters_log.append(int(res.iterations))
                times_log.append(float(dt))
                if res.converged:
                    tol_conv += 1
                if losses[-1] < best_loss:
                    best_loss = losses[-1]
                    best_res = res

            assert best_res is not None
            mean_loss = float(np.mean(losses))
            std_loss = float(np.std(losses))
            mean_iters = float(np.mean(iters_log))
            mean_time = float(np.mean(times_log))
            dense_loss = float(
                np.linalg.norm(best_res.reconstruct_small() - src_dense)
            ) / src_dense_norm

            rows.append([
                r, init, N_REPLICATES,
                f"{best_loss:.6e}", f"{mean_loss:.6e}", f"{std_loss:.6e}",
                f"{mean_iters:.1f}", f"{mean_time:.4f}",
                tol_conv,
                f"{dense_loss:.6e}",
            ])

            # Plot mean curve and min/max envelope across replicates.
            max_len = max(h.size for h in histories)
            padded = np.full((len(histories), max_len), np.nan, dtype=float)
            for k, h in enumerate(histories):
                padded[k, : h.size] = np.maximum(h, 1e-16)
            mean_curve = np.nanmean(padded, axis=0)
            min_curve = np.nanmin(padded, axis=0)
            max_curve = np.nanmax(padded, axis=0)
            xs = np.arange(1, max_len + 1)
            line, = ax.semilogy(xs, mean_curve, label=f"R_t={r}")
            ax.fill_between(xs, min_curve, max_curve, alpha=0.18, color=line.get_color())

        ax.set_xlabel("ALS sweep")
        ax.set_title(f"init={init}")
        ax.legend(fontsize=7, ncol=2, loc="upper right")
    axes_iter[0].set_ylabel("relative Frobenius error")
    fig.suptitle(
        f"CP-ALS from CP source (R_src={R_src}, shape={shape}, "
        f"{N_REPLICATES} restarts, mean ± min/max)"
    )
    fig.tight_layout()
    save_fig(fig, "loss_vs_iter", TASK)
    write_csv(rows, results_dir(args.out_dir) / f"{TASK}_table.csv")
    print(
        f"Task 8 artefacts written; data rows: {len(rows) - 1} "
        f"({len(target_ranks)} ranks × {len(inits)} inits)"
    )


if __name__ == "__main__":
    main()
