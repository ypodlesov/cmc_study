"""Tests for Task 12 — ST-HOSVD."""

from __future__ import annotations

import numpy as np
import pytest

from matcomp.tensor_algorithms.st_hosvd import st_hosvd
from matcomp.utils.seeding import make_rng
from matcomp.utils.tensor_linalg import tucker_to_dense
from matcomp.utils.tensor_test_objects import random_tucker


def test_factors_are_orthonormal() -> None:
    rng = make_rng(0)
    X = rng.standard_normal((6, 7, 8)).astype(np.float64)
    res = st_hosvd(X, ranks=(3, 4, 3))
    for U in res.factors.factors:
        np.testing.assert_allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-10)


def test_exact_recovery_at_true_multilinear_rank() -> None:
    rng = make_rng(0)
    core, factors = random_tucker((6, 7, 8), (2, 3, 2), rng)
    X = tucker_to_dense(core, factors)
    res = st_hosvd(X, ranks=(2, 3, 2))
    assert res.rel_error < 1e-12


def test_eps_path_meets_relative_error_budget() -> None:
    rng = make_rng(0)
    # Tensor with strongly decaying singular values: mode-n unfolding norm
    # decays like 2^{-k} on each mode.
    core = rng.standard_normal((4, 5, 4))
    factors = [np.linalg.qr(rng.standard_normal((s, r)))[0] for s, r in zip((10, 12, 8), (4, 5, 4), strict=True)]
    X = tucker_to_dense(core, factors)
    # Add a small tail to make the eps truncation meaningful.
    X = X + 1e-3 * rng.standard_normal(X.shape)
    res = st_hosvd(X, eps=0.05)
    # The Vannieuwenhoven bound gives rel_error <= eps when budgets are
    # split equally; we allow some slack for the per-mode budgeting.
    assert res.rel_error <= 0.06


def test_ranks_path_invalid_lengths_raise() -> None:
    rng = make_rng(0)
    X = rng.standard_normal((4, 5, 6)).astype(np.float64)
    with pytest.raises(ValueError):
        st_hosvd(X, ranks=(2, 3))
    with pytest.raises(ValueError):
        st_hosvd(X)  # neither ranks nor eps
    with pytest.raises(ValueError):
        st_hosvd(X, ranks=(1, 1, 1), eps=0.1)


def test_mode_order_invariance_within_tol() -> None:
    rng = make_rng(0)
    core, factors = random_tucker((5, 6, 4), (2, 3, 2), rng)
    X = tucker_to_dense(core, factors)
    r1 = st_hosvd(X, ranks=(2, 3, 2), mode_order=(0, 1, 2))
    r2 = st_hosvd(X, ranks=(2, 3, 2), mode_order=(2, 0, 1))
    # Different mode orders may produce different intermediate sizes but
    # the final reconstruction error must stay near machine precision.
    assert r1.rel_error < 1e-10
    assert r2.rel_error < 1e-10


def test_rank_clamped_to_unfolding_rank() -> None:
    """Asking for a rank above min(unfolding shape) is clamped silently."""
    rng = make_rng(0)
    X = rng.standard_normal((3, 4, 5)).astype(np.float64)
    res = st_hosvd(X, ranks=(10, 4, 5))  # mode 0 has only 3 rows
    assert res.ranks[0] == 3
