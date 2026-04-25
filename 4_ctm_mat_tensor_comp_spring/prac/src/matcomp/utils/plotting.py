"""Shared Matplotlib style and figure-saving helpers.

Every experiment script imports :func:`apply_style` once and then calls
:func:`save_fig` to drop figures into ``reports/figures/<task_name>/``.
The style is intentionally minimal — readable on a printed page with no
colour-blindness issues.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Resolve the project's reports/ directory once; experiments resolve theirs
# relative to PROJECT_ROOT below so they work regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"


def apply_style() -> None:
    """Configure Matplotlib defaults for the project.

    Call once near the top of any experiment script.
    """
    mpl.rcParams.update(
        {
            "figure.figsize": (7.0, 4.5),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.alpha": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "legend.frameon": False,
            "image.cmap": "viridis",
        }
    )


def figures_dir(task: str, base: Path | None = None) -> Path:
    """Return the directory ``reports/figures/<task>/``, creating it if needed."""
    base_dir = (base or DEFAULT_REPORTS_DIR) / "figures" / task
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def results_dir(base: Path | None = None) -> Path:
    """Return the directory ``reports/results/``, creating it if needed."""
    base_dir = (base or DEFAULT_REPORTS_DIR) / "results"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def save_fig(fig: Figure, name: str, task: str, base: Path | None = None) -> Path:
    """Save ``fig`` to ``reports/figures/<task>/<name>.png`` and return the path."""
    out_dir = figures_dir(task, base)
    target = out_dir / f"{name}.png"
    fig.savefig(target)
    plt.close(fig)
    return target


__all__ = [
    "DEFAULT_REPORTS_DIR",
    "PROJECT_ROOT",
    "apply_style",
    "figures_dir",
    "results_dir",
    "save_fig",
]
