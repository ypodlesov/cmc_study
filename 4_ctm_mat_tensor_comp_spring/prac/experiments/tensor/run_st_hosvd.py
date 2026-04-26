"""Experiment for Task 12 (ST-HOSVD)."""

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
from matcomp.tensor_algorithms.st_hosvd import st_hosvd  # noqa: E402
from matcomp.utils.functional_tensor import evaluate_small  # noqa: E402
from matcomp.utils.plotting import apply_style, results_dir, save_fig  # noqa: E402
from matcomp.utils.seeding import make_rng  # noqa: E402
from matcomp.utils.tensor_metrics import compression_ratio_tensor  # noqa: E402
from matcomp.utils.tensor_test_objects import (  # noqa: E402
    GaussianKernel3DTensor,
    Hilbert3DTensor,
)

TASK = "task12_st_hosvd"


def main() -> None:
    args = parse_args(TASK)
    apply_style()
    rng = make_rng(args.seed)

    targets = {
        "hilbert3d_12": evaluate_small(Hilbert3DTensor(12)),
        "kernel3d_12": evaluate_small(
            GaussianKernel3DTensor(
                rng.uniform(0.0, 1.0, size=12).astype(np.float64),
                rng.uniform(0.0, 1.0, size=12).astype(np.float64),
                rng.uniform(0.0, 1.0, size=12).astype(np.float64),
                sigma=0.5,
            )
        ),
    }
    eps_grid = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    rows: list[list[object]] = [["target", "eps", "ranks", "rel_error", "compression"]]

    fig, ax = plt.subplots()
    for name, X in targets.items():
        compressions = []
        errors = []
        for eps in eps_grid:
            res = st_hosvd(X, eps=eps, random_seed=args.seed)
            comp = compression_ratio_tensor(res.factors)
            compressions.append(comp)
            errors.append(res.rel_error)
            rows.append([name, eps, str(res.ranks), f"{res.rel_error:.6e}", f"{comp:.3f}"])
        ax.loglog(np.maximum(compressions, 1e-30), np.maximum(errors, 1e-16),
                  marker="o", label=name)
    ax.set_xlabel("compression ratio (= dense bytes / factor bytes)")
    ax.set_ylabel("relative Frobenius error")
    ax.set_title("ST-HOSVD: Pareto curve over eps")
    ax.legend()
    save_fig(fig, "pareto_eps", TASK)

    write_csv(rows, results_dir(args.out_dir) / f"{TASK}_table.csv")
    print(f"Task 12 artefacts written; rows: {len(rows) - 1}")


if __name__ == "__main__":
    main()
