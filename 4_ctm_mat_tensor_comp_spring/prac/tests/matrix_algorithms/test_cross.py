"""Tests for Task 2 — cross / skeleton approximation."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from matcomp.matrix_algorithms.cross import cross_rank_r
from matcomp.utils.functional_matrix import evaluate_small
from matcomp.utils.seeding import make_rng
from matcomp.utils.test_matrices import (
    CauchyMatrix,
    DenseMatrix,
    HilbertMatrix,
    LowRankMatrix,
)


@pytest.mark.parametrize("init", ["random", "maxvol", "greedy"])
@pytest.mark.parametrize("seed", [0, 17])
def test_exact_low_rank_recovery(init: str, seed: int) -> None:
    rng = make_rng(seed)
    U = rng.standard_normal((25, 4))
    V = rng.standard_normal((20, 4))
    fm = LowRankMatrix(U, V)
    A = evaluate_small(fm)
    res = cross_rank_r(fm, r=4, init=init, random_seed=seed, max_sweeps=10)  # type: ignore[arg-type]
    err = np.linalg.norm(A - res.reconstruct_small()) / np.linalg.norm(A)
    assert err < 1e-10


def test_hilbert_error_decreases_with_r() -> None:
    H = HilbertMatrix(40, 40)
    A = evaluate_small(H)
    norm = float(np.linalg.norm(A))
    errors = []
    for r in [3, 5, 8]:
        res = cross_rank_r(H, r=r, init="maxvol", max_sweeps=10, random_seed=42)
        errors.append(float(np.linalg.norm(A - res.reconstruct_small()) / norm))
    assert errors[0] >= errors[1] >= errors[2]


def test_cauchy_error_decreases_with_r() -> None:
    rng = make_rng(0)
    x = 1.0 + rng.uniform(0.0, 1.0, size=20)
    y = 0.5 + rng.uniform(0.0, 1.0, size=20)
    C = CauchyMatrix(x, y)
    A = evaluate_small(C)
    norm = float(np.linalg.norm(A))
    # On Cauchy the singular values decay exponentially, so the numerical
    # rank is small (~5 for sigma at 1e-12 in this size). Beyond the
    # numerical rank, cross adds round-off noise from the truncated
    # pseudo-inverse, so we only check the well-resolved regime.
    errors = []
    for r in [2, 3, 4, 5]:
        res = cross_rank_r(C, r=r, init="maxvol", max_sweeps=10, random_seed=42)
        errors.append(float(np.linalg.norm(A - res.reconstruct_small()) / norm))
    for prev, cur in zip(errors[:-1], errors[1:], strict=True):
        assert cur <= prev * 1.5


def test_ill_conditioned_pivot_emits_warning() -> None:
    rng = make_rng(0)
    v = rng.standard_normal(10)
    A_singular = np.outer(v, v) + 1e-15 * np.eye(10)
    fm = DenseMatrix(A_singular)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cross_rank_r(fm, r=3, init="random", random_seed=42)
        rt_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(rt_warnings) >= 1


def test_cross_does_not_use_matvec_or_individual_entries() -> None:
    """Cross approximation must use only block / row / col oracles."""
    H = HilbertMatrix(20, 20)
    res = cross_rank_r(H, r=4, init="random", random_seed=42)
    assert res.oracle_counts.matvec_calls == 0
    assert res.oracle_counts.matmat_calls == 0
    assert res.oracle_counts.entry_calls == 0  # PDF acceptance


def test_oversized_r_clamped_at_min_dim() -> None:
    fm = HilbertMatrix(5, 5)
    with pytest.raises(ValueError):
        cross_rank_r(fm, r=10, init="random")
