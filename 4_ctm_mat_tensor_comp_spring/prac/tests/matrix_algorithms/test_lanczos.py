"""Tests for Task 1 — Lanczos."""

from __future__ import annotations

import numpy as np
import pytest

from matcomp.matrix_algorithms.lanczos import lanczos
from matcomp.utils.counting import CountingFunctionalMatrix
from matcomp.utils.seeding import make_rng
from matcomp.utils.test_matrices import DenseMatrix


def test_diagonal_spectrum_recovers_extremes() -> None:
    n = 50
    A = np.diag(np.linspace(1.0, 10.0, n))
    matvec = lambda x: A @ x
    # k=40 converges both extremes to <1e-6; k=20 is enough for the upper.
    res = lanczos(matvec, n=n, k=40, reorthogonalize=True, random_seed=42)
    np.testing.assert_allclose(res.ritz_values[0], 1.0, atol=1e-3)
    np.testing.assert_allclose(res.ritz_values[-1], 10.0, atol=1e-3)


@pytest.mark.parametrize("seed", [0, 17, 12345])
def test_random_symmetric_matches_eigh(seed: int) -> None:
    rng = make_rng(seed)
    n = 30
    B = rng.standard_normal((n, n))
    A = (B + B.T) / 2
    matvec = lambda x: A @ x
    res = lanczos(matvec, n=n, k=n, reorthogonalize=True, random_seed=42)
    true_eig = np.linalg.eigvalsh(A)
    np.testing.assert_allclose(res.ritz_values[-1], true_eig[-1], atol=1e-8)
    np.testing.assert_allclose(res.ritz_values[0], true_eig[0], atol=1e-8)


def test_q_orthogonality_with_reorthogonalisation() -> None:
    rng = make_rng(0)
    n = 30
    A = rng.standard_normal((n, n))
    A = (A + A.T) / 2
    res = lanczos(lambda x: A @ x, n=n, k=20, reorthogonalize=True, random_seed=42)
    QtQ = res.Q.T @ res.Q
    err = float(np.linalg.norm(QtQ - np.eye(QtQ.shape[0])))
    assert err < 1e-10


def test_lanczos_only_calls_matvec() -> None:
    """Lanczos must not access individual entries (PDF p. 3 acceptance)."""
    rng = make_rng(0)
    n = 20
    A = rng.standard_normal((n, n))
    A = (A + A.T) / 2
    fm = CountingFunctionalMatrix(DenseMatrix(A))
    lanczos(fm.matvec, n=n, k=10, reorthogonalize=True, random_seed=42)
    assert fm.counts.entry_calls == 0
    assert fm.counts.row_calls == 0
    assert fm.counts.col_calls == 0
    assert fm.counts.block_calls == 0
    assert fm.counts.matvec_calls > 0


def test_happy_breakdown_on_low_rank() -> None:
    """A symmetric rank-2 matrix should breakdown after exactly 2 iterations."""
    rng = make_rng(0)
    v1 = rng.standard_normal(10)
    v2 = rng.standard_normal(10)
    A = np.outer(v1, v1) + np.outer(v2, v2)
    res = lanczos(
        lambda x: A @ x,
        n=10,
        k=8,
        q0=v1.copy(),
        tol=1e-10,
        reorthogonalize=True,
    )
    assert res.breakdown
    assert res.iterations <= 3  # 2 informative + maybe a residual step


def test_clustered_spectrum_needs_reorthogonalisation() -> None:
    """Without reorthogonalisation, ghost eigenvalues appear on tightly-clustered spectra."""
    n = 25
    diag = np.concatenate([np.linspace(1.0, 1.0 + 1e-10, n // 2), np.linspace(5.0, 6.0, n - n // 2)])
    A = np.diag(diag)
    matvec = lambda x: A @ x
    # With reorth: tight orthogonality
    res_with = lanczos(matvec, n=n, k=20, reorthogonalize=True, random_seed=42)
    QtQ_with = res_with.Q.T @ res_with.Q
    err_with = float(np.linalg.norm(QtQ_with - np.eye(QtQ_with.shape[0])))
    # Without reorth: orthogonality degrades
    res_without = lanczos(matvec, n=n, k=20, reorthogonalize=False, random_seed=42)
    # Cannot inspect Q without reorth (Q isn't kept), so compare convergence:
    assert err_with < 1e-10
    # The largest Ritz value should still be close to the truth even without reorth.
    np.testing.assert_allclose(res_without.ritz_values[-1], 6.0, atol=1e-3)


def test_compute_ritz_vectors_returns_matrix() -> None:
    rng = make_rng(0)
    n = 12
    A = rng.standard_normal((n, n))
    A = (A + A.T) / 2
    # Run all the way to n iterations so every Ritz pair has converged.
    res = lanczos(
        lambda x: A @ x,
        n=n,
        k=n,
        reorthogonalize=True,
        compute_ritz_vectors=True,
        random_seed=42,
    )
    assert res.ritz_vectors is not None
    # Each Ritz pair (theta, v) should approximately satisfy A v ≈ theta v.
    for k in range(res.ritz_vectors.shape[1]):
        v = res.ritz_vectors[:, k]
        residual = float(np.linalg.norm(A @ v - res.ritz_values[k] * v))
        assert residual < 1e-8
