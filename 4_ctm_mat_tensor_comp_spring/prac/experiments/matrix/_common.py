"""Shared CLI scaffolding for experiments/matrix/run_*.py scripts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def parse_args(task_name: str, *, default_seed: int = 42) -> argparse.Namespace:
    """Standard CLI: --seed, --out-dir."""
    parser = argparse.ArgumentParser(description=f"Run experiments for {task_name}.")
    parser.add_argument("--seed", type=int, default=default_seed)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override the reports/ directory (defaults to project reports/).",
    )
    return parser.parse_args()


def write_csv(rows: list[list[Any]], out_path: Path) -> None:
    """Write a list of lists to ``out_path`` in CSV format (newline-safe)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
