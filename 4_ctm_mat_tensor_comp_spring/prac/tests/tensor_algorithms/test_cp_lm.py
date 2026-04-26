"""Tests for Task 11 — Levenberg–Marquardt for CP."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from matcomp.tensor_algorithms.cp_lm import (
    _jacobian,
    _residual_vec,
    cp_levenberg_marquardt,
)
from matcomp.utils.seeding import make_rng
from matcomp.utils.tensor_linalg import cp_to_dense, pack_cp_theta
from matcomp.utils.tensor_test_objects import random_cp


def test_recovers_synthetic_cp() -> None:
    rng = make_rng(0)
    weights, factors = random_cp((4, 5, 6), 2, rng)
    X = cp_to_dense(weights, factors)
    res = cp_levenberg_marquardt(X, rank=2, init="als", als_warmup=3, max_iter=50, tol=1e-14, random_seed=0)
    err = np.linalg.norm(X - res.reconstruct_small()) / np.linalg.norm(X)
    assert err < 1e-8


def test_jacobian_matches_finite_differences() -> None:
    rng = make_rng(0)
    shape = (3, 4, 2)
    rank = 2
    weights = np.ones(rank)
    factors = [rng.standard_normal((s, rank)).astype(np.float64) for s in shape]
    X = rng.standard_normal(shape).astype(np.float64)
    theta = pack_cp_theta(weights, factors)
    J = _jacobian(theta, shape, rank)
    eps = 1e-6
    for p in [0, 1, rank, rank + 5, theta.size - 1]:
        th_plus = theta.copy()
        th_plus[p] += eps
        th_minus = theta.copy()
        th_minus[p] -= eps
        fd = (_residual_vec(th_plus, X, rank) - _residual_vec(th_minus, X, rank)) / (2 * eps)
        np.testing.assert_allclose(J[:, p], fd, atol=1e-6)


def test_jacobian_shape_matches_packed_theta() -> None:
    rng = make_rng(0)
    weights, factors = random_cp((4, 5, 6), 3, rng)
    X = cp_to_dense(weights, factors)
    res = cp_levenberg_marquardt(X, rank=3, init="random", max_iter=2, tol=None, random_seed=0)
    n_params = 3 + 3 * (4 + 5 + 6)  # weights + factor entries
    assert res.jacobian_shape == (X.size, n_params)


def test_loss_history_strictly_nonincreasing_after_acceptance() -> None:
    rng = make_rng(0)
    weights, factors = random_cp((4, 5, 6), 2, rng)
    X = cp_to_dense(weights, factors)
    res = cp_levenberg_marquardt(X, rank=2, init="als", als_warmup=3, max_iter=30, tol=1e-14, random_seed=0)
    diffs = np.diff(res.loss_history)
    assert np.all(diffs <= 1e-12), f"loss must be non-increasing: {diffs}"


def test_invalid_rank_raises() -> None:
    rng = make_rng(0)
    X = rng.standard_normal((3, 4, 5)).astype(np.float64)
    with pytest.raises(ValueError):
        cp_levenberg_marquardt(X, rank=0)


def test_singular_jacobian_falls_back_to_pinv_warns() -> None:
    """A nearly-singular system at tiny mu produces a pinv warning."""
    # Construct a CP with two near-collinear factor columns so JTJ is
    # rank deficient. Force mu_init very small to trigger the fallback.
    rng = make_rng(0)
    A = rng.standard_normal((4, 1))
    factors = [np.column_stack([A, A + 1e-14 * rng.standard_normal((4, 1))]) for _ in range(3)]
    weights = np.array([1.0, 1.0])
    X = cp_to_dense(weights, factors)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cp_levenberg_marquardt(
            X, rank=2, init="random", als_warmup=0, max_iter=3, mu_init=0.0, tol=None, random_seed=0,
        )
    rt_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(rt_warnings) >= 1
