"""Tests for Task 3 — randomized SVD."""

from __future__ import annotations

import numpy as np
import pytest

from matcomp.matrix_algorithms.randomized_svd import randomized_svd
from matcomp.utils.functional_matrix import evaluate_small
from matcomp.utils.seeding import make_rng
from matcomp.utils.test_matrices import (
    GaussianKernelMatrix,
    HilbertMatrix,
    LowRankMatrix,
    LowRankPlusNoise,
)


@pytest.mark.parametrize("seed", [0, 17, 12345])
def test_exact_low_rank_recovery_with_oversampling(seed: int) -> None:
    rng = make_rng(seed)
    U = rng.standard_normal((30, 4))
    V = rng.standard_normal((25, 4))
    fm = LowRankMatrix(U, V)
    res = randomized_svd(fm, r=4, oversampling=10, random_seed=seed)
    A = evaluate_small(fm)
    err = np.linalg.norm(A - res.reconstruct_small()) / np.linalg.norm(A)
    assert err < 1e-12


def test_orthogonal_factors_to_explicit_tolerance() -> None:
    """U^T U ≈ I and V V^T ≈ I within 1e-10 (PDF p. 4 acceptance)."""
    rng = make_rng(0)
    U = rng.standard_normal((30, 5))
    V = rng.standard_normal((25, 5))
    fm = LowRankMatrix(U, V)
    res = randomized_svd(fm, r=5, oversampling=10, random_seed=42)
    np.testing.assert_allclose(
        res.factors.U.T @ res.factors.U,
        np.eye(5),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        res.factors.Vt @ res.factors.Vt.T,
        np.eye(5),
        atol=1e-10,
    )


def test_rapid_decay_no_power_iter_suffices() -> None:
    rng = make_rng(0)
    x = rng.uniform(0.0, 1.0, size=50)
    y = rng.uniform(0.0, 1.0, size=50)
    fm = GaussianKernelMatrix(x, y, sigma=0.4)
    A = evaluate_small(fm)
    s = np.linalg.svd(A, compute_uv=False)
    res = randomized_svd(fm, r=8, oversampling=10, n_power_iter=0, random_seed=42)
    err = np.linalg.norm(A - res.reconstruct_small()) / np.linalg.norm(A)
    optimal = float(np.sqrt(np.sum(s[8:] ** 2)) / np.linalg.norm(A))
    # Should be within a small factor of optimum.
    assert err <= optimal * 5.0


def test_slow_decay_benefits_from_power_iter() -> None:
    """On Hilbert, rsvd with q=2 should be ≤ rsvd with q=0."""
    H = HilbertMatrix(40, 40)
    A = evaluate_small(H)
    norm = float(np.linalg.norm(A))
    res_no_q = randomized_svd(H, r=5, oversampling=4, n_power_iter=0, random_seed=42)
    res_q2 = randomized_svd(H, r=5, oversampling=4, n_power_iter=2, random_seed=42)
    err_no_q = float(np.linalg.norm(A - res_no_q.reconstruct_small()) / norm)
    err_q2 = float(np.linalg.norm(A - res_q2.reconstruct_small()) / norm)
    assert err_q2 <= err_no_q + 1e-15


def test_noisy_low_rank_recovers_singular_values_within_noise() -> None:
    rng = make_rng(0)
    U = rng.standard_normal((40, 3))
    V = rng.standard_normal((40, 3))
    fm = LowRankPlusNoise(U, V, noise_level=1e-4, random_seed=99)
    res = randomized_svd(fm, r=3, oversampling=10, n_power_iter=2, random_seed=42)
    A = evaluate_small(fm)
    s = np.linalg.svd(A, compute_uv=False)[:3]
    np.testing.assert_allclose(np.sort(res.factors.S)[::-1], s, atol=1e-2)


def test_oracle_calls_bounded() -> None:
    H = HilbertMatrix(30, 30)
    res = randomized_svd(H, r=5, oversampling=10, n_power_iter=3, random_seed=42)
    total = res.oracle_counts.matmat_calls + res.oracle_counts.rmatmat_calls
    # Per the algorithm: 1 initial matmat + q*(rmatmat + matmat) + 1 final rmatmat.
    assert total <= 2 * (3 + 1)


def test_invalid_arguments_raise() -> None:
    fm = HilbertMatrix(5, 5)
    with pytest.raises(ValueError):
        randomized_svd(fm, r=0)
    with pytest.raises(ValueError):
        randomized_svd(fm, r=10)  # exceeds shape
    with pytest.raises(ValueError):
        randomized_svd(fm, r=2, oversampling=-1)
    with pytest.raises(ValueError):
        randomized_svd(fm, r=2, n_power_iter=-1)
