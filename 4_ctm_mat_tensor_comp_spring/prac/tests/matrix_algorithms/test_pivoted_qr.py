"""Tests for Task 6 — pivoted (rank-revealing) QR."""

from __future__ import annotations

import numpy as np
import pytest

from matcomp.matrix_algorithms.pivoted_qr import pivoted_qr_approx
from matcomp.utils.seeding import make_rng


@pytest.mark.parametrize("seed", [0, 17, 12345])
def test_random_matrix_q_orthogonal_and_factors_match(seed: int) -> None:
    rng = make_rng(seed)
    A = rng.standard_normal((10, 8))
    res = pivoted_qr_approx(A)
    QtQ = res.Q.T @ res.Q
    np.testing.assert_allclose(QtQ, np.eye(QtQ.shape[0]), atol=1e-12)
    np.testing.assert_allclose(res.Q @ res.R, A[:, res.P], atol=1e-12)


def test_rank_deficient_matrix_recovers_true_rank() -> None:
    rng = make_rng(0)
    base = rng.standard_normal((10, 3))
    # Stack a rank-3 matrix with three exactly-zero columns and two
    # near-zero columns.  The numerical rank must come out as 3.
    A = np.column_stack(
        [
            base,
            np.zeros((10, 3)),
            1e-14 * rng.standard_normal((10, 2)),
        ]
    )
    res = pivoted_qr_approx(A, tol=1e-10)
    assert res.estimated_rank == 3


def test_ill_conditioned_pivot_order() -> None:
    """Three near-collinear directions; only one should land in the top pivots."""
    rng = make_rng(1)
    a = rng.standard_normal(20)
    a = a / np.linalg.norm(a)
    b = a + 1e-10 * rng.standard_normal(20)
    c = a + 1e-12 * rng.standard_normal(20)
    A = np.column_stack([a, b, c, rng.standard_normal((20, 3))])
    res = pivoted_qr_approx(A, tol=1e-8)
    # Numerical rank is at most 4 (three collinear cols collapse to one).
    assert res.estimated_rank <= 4
    diag = np.abs(np.diag(res.R))
    assert np.all(diag[1:] <= diag[:-1] + 1e-12)


def test_oracle_check_against_scipy() -> None:
    scipy_linalg = pytest.importorskip("scipy.linalg")
    rng = make_rng(42)
    A = rng.standard_normal((12, 9))
    res = pivoted_qr_approx(A)
    # SciPy returns Q, R, P with A[:, P] == Q R.
    Q_s, R_s, P_s = scipy_linalg.qr(A, mode="economic", pivoting=True)
    # Compare diagonal magnitudes — the absolute entries must agree
    # (signs depend on the Householder convention).
    np.testing.assert_allclose(
        np.abs(np.diag(res.R)),
        np.abs(np.diag(R_s)),
        rtol=1e-8,
        atol=1e-10,
    )
    # And both reconstruct A exactly.
    np.testing.assert_allclose(res.Q @ res.R, A[:, res.P], atol=1e-10)
    np.testing.assert_allclose(Q_s @ R_s, A[:, P_s], atol=1e-10)


def test_low_rank_approx_reconstruct() -> None:
    rng = make_rng(0)
    A = rng.standard_normal((12, 8))
    res = pivoted_qr_approx(A, rank=4)
    A_hat = res.reconstruct_small()
    err = np.linalg.norm(A - A_hat) / np.linalg.norm(A)
    s = np.linalg.svd(A, compute_uv=False)
    optimal = float(np.sqrt(np.sum(s[4:] ** 2)) / np.linalg.norm(A))
    # Pivoted QR is at most a small constant factor of the SVD optimum.
    assert err <= optimal * 5.0


def test_rank_argument_clamps() -> None:
    rng = make_rng(0)
    A = rng.standard_normal((6, 4))
    res = pivoted_qr_approx(A, rank=10)  # ask for too much
    assert res.estimated_rank == 4
