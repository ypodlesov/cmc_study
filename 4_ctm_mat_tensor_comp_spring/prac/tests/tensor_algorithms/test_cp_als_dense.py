"""Tests for Task 9 — CP-ALS on a dense tensor."""

from __future__ import annotations

import numpy as np
import pytest

from matcomp.tensor_algorithms.cp_als_dense import cp_als
from matcomp.utils.seeding import make_rng
from matcomp.utils.tensor_linalg import cp_to_dense
from matcomp.utils.tensor_test_objects import random_cp


def test_exact_recovery_on_random_cp() -> None:
    """Rank-3 synthetic CP must be recovered to ~machine precision."""
    rng = make_rng(0)
    weights, factors = random_cp((6, 7, 8), 3, rng)
    X = cp_to_dense(weights, factors)
    res = cp_als(X, rank=3, init="svd", max_iter=200, tol=1e-12, random_seed=0)
    err = np.linalg.norm(X - res.reconstruct_small()) / np.linalg.norm(X)
    assert err < 1e-8


def test_random_dense_loss_decreases() -> None:
    """Loss must be (approximately) non-increasing across sweeps."""
    rng = make_rng(0)
    X = rng.standard_normal((5, 6, 7)).astype(np.float64)
    res = cp_als(X, rank=4, init="random", max_iter=80, tol=1e-12, random_seed=0)
    # ALS is non-increasing in exact arithmetic; allow a tiny tolerance for round-off.
    diffs = np.diff(res.loss_history)
    assert np.all(diffs <= 1e-10), f"loss not monotone: {diffs}"


def test_noisy_signal_matches_signal_rank() -> None:
    """Rank-3 signal + small noise: rank-3 fit should resolve the signal."""
    rng = make_rng(0)
    weights, factors = random_cp((6, 7, 8), 3, rng)
    signal = cp_to_dense(weights, factors)
    noise = 1e-3 * rng.standard_normal(signal.shape)
    X = signal + noise
    res = cp_als(X, rank=3, init="svd", max_iter=300, tol=1e-12, random_seed=0)
    # The fit should not be much worse than the noise level scaled by the signal magnitude.
    err = np.linalg.norm(signal - res.reconstruct_small()) / np.linalg.norm(signal)
    assert err < 5e-3


@pytest.mark.parametrize("init", ["random", "svd"])
def test_invalid_rank_raises(init: str) -> None:
    rng = make_rng(0)
    X = rng.standard_normal((4, 5, 6)).astype(np.float64)
    with pytest.raises(ValueError):
        cp_als(X, rank=0, init=init)  # type: ignore[arg-type]


def test_normalize_keeps_unit_factor_columns() -> None:
    rng = make_rng(0)
    weights, factors = random_cp((6, 7, 8), 3, rng)
    X = cp_to_dense(weights, factors)
    res = cp_als(X, rank=3, init="svd", max_iter=50, tol=1e-12, normalize=True, random_seed=0)
    for fac in res.factors.factors:
        norms = np.linalg.norm(fac, axis=0)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-8)


def test_low_d2_works() -> None:
    """Order-d > 3: the implementation must not be hard-coded for 3-D."""
    rng = make_rng(0)
    weights, factors = random_cp((4, 5, 4, 3), rank=2, rng=rng)
    X = cp_to_dense(weights, factors)
    res = cp_als(X, rank=2, init="svd", max_iter=200, tol=1e-12, random_seed=0)
    err = np.linalg.norm(X - res.reconstruct_small()) / np.linalg.norm(X)
    assert err < 1e-8
