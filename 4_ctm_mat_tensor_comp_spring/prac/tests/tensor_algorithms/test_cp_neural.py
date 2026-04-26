"""Tests for Task 10 — CP via differentiable optimisation."""

from __future__ import annotations

import numpy as np
import pytest

from matcomp.tensor_algorithms.cp_neural import CPModel, fit_cp_neural
from matcomp.utils.seeding import make_rng
from matcomp.utils.tensor_linalg import cp_to_dense
from matcomp.utils.tensor_test_objects import CPSyntheticTensor, random_cp


def test_grad_dense_matches_finite_differences() -> None:
    """Closed-form gradient must agree with central differences."""
    rng = make_rng(0)
    mdl = CPModel(shape=(3, 4, 5), rank=2, init="random", random_seed=0)
    X = rng.standard_normal((3, 4, 5)).astype(np.float64)
    g_w, g_factors = mdl.grad_dense(X)
    eps = 1e-5
    for n in range(3):
        f = mdl.factors[n].copy()
        f[0, 0] += eps
        mdl.factors[n] = f
        L_plus = mdl.loss_dense(X)
        f[0, 0] -= 2 * eps
        mdl.factors[n] = f
        L_minus = mdl.loss_dense(X)
        f[0, 0] += eps
        mdl.factors[n] = f
        fd = (L_plus - L_minus) / (2 * eps)
        np.testing.assert_allclose(g_factors[n][0, 0], fd, atol=1e-6)
    # Weight gradient
    w = mdl.weights.copy()
    w[0] += eps
    mdl.weights = w
    L_plus = mdl.loss_dense(X)
    w[0] -= 2 * eps
    mdl.weights = w
    L_minus = mdl.loss_dense(X)
    fd_w = (L_plus - L_minus) / (2 * eps)
    np.testing.assert_allclose(g_w[0], fd_w, atol=1e-6)


def test_dense_mode_recovers_synthetic_cp() -> None:
    rng = make_rng(0)
    weights, factors = random_cp((5, 6, 7), 3, rng)
    X = cp_to_dense(weights, factors)
    res = fit_cp_neural(
        X, rank=3, mode="dense", optimizer="adam", lr=0.05,
        max_iter=2000, tol=1e-12, random_seed=0,
    )
    err = np.linalg.norm(X - res.reconstruct_small()) / np.linalg.norm(X)
    assert err < 5e-3


def test_batched_mode_uses_only_samples_oracle() -> None:
    """PDF p. 9 acceptance: 'compute values at arbitrary indices without
    building the full Y'. The functional-tensor input must therefore be
    queried only via ``samples``."""
    rng = make_rng(0)
    weights, factors = random_cp((6, 7, 8), 3, rng)
    ft = CPSyntheticTensor(weights, list(factors))
    res = fit_cp_neural(
        ft, rank=3, mode="batched", batch_size=512, optimizer="adam", lr=0.05,
        max_iter=400, tol=None, log_interval=50, random_seed=0,
    )
    counts = res.oracle_counts
    assert counts is not None
    assert counts.entry_calls == 0
    assert counts.block_calls == 0
    assert counts.fiber_calls == 0
    assert counts.samples_calls > 0


def test_reproducible_with_seed() -> None:
    rng = make_rng(0)
    X = rng.standard_normal((4, 5, 6)).astype(np.float64)
    r1 = fit_cp_neural(X, rank=2, mode="dense", max_iter=50, tol=None, random_seed=42)
    r2 = fit_cp_neural(X, rank=2, mode="dense", max_iter=50, tol=None, random_seed=42)
    np.testing.assert_allclose(r1.loss_history, r2.loss_history)
    np.testing.assert_allclose(r1.factors.weights, r2.factors.weights)


def test_invalid_optimizer_raises() -> None:
    rng = make_rng(0)
    X = rng.standard_normal((4, 5, 6)).astype(np.float64)
    with pytest.raises(ValueError):
        fit_cp_neural(X, rank=2, optimizer="not_an_optimizer", max_iter=2)  # type: ignore[arg-type]


def test_grad_batch_at_full_grid_matches_grad_dense() -> None:
    """At full-batch (every multi-index drawn once) the batched gradient
    should agree with the dense closed form up to scaling by N."""
    rng = make_rng(0)
    mdl = CPModel(shape=(3, 4, 5), rank=2, init="random", random_seed=0)
    X = rng.standard_normal((3, 4, 5)).astype(np.float64)
    grid = np.indices((3, 4, 5)).reshape(3, -1).T.astype(np.intp)
    values = X[tuple(grid[:, k] for k in range(3))]
    g_w_b, g_facs_b = mdl.grad_batch(grid, values)
    g_w_d, g_facs_d = mdl.grad_dense(X)
    # Both gradients use 1/N scaling for the dense, 1/b scaling for the batch.
    # With b == N they coincide.
    np.testing.assert_allclose(g_w_b, g_w_d, atol=1e-10)
    for n in range(3):
        np.testing.assert_allclose(g_facs_b[n], g_facs_d[n], atol=1e-10)
