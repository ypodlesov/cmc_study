"""Tests for Task 4 — adaptive cross approximation with caching."""

from __future__ import annotations

import numpy as np
import pytest

from matcomp.matrix_algorithms.adaptive_cross import adaptive_cross_cached
from matcomp.utils.caching import CachedFunctionalMatrix
from matcomp.utils.counting import CountingFunctionalMatrix
from matcomp.utils.functional_matrix import evaluate_small
from matcomp.utils.seeding import make_rng
from matcomp.utils.test_matrices import (
    HilbertMatrix,
    LowRankMatrix,
    LowRankPlusNoise,
)


def test_cache_hits_after_repeated_entry_queries() -> None:
    """Five identical entry queries should produce one miss and four hits."""
    H = HilbertMatrix(8, 8)
    counted = CountingFunctionalMatrix(H)
    cached = CachedFunctionalMatrix(counted)
    for _ in range(5):
        cached.entry(2, 3)
    assert counted.counts.entry_calls == 1
    assert cached.cache_misses == 1
    assert cached.cache_hits == 4


def test_rank1_matrix_reconstructed_to_eps() -> None:
    rng = make_rng(0)
    u = rng.standard_normal((20, 1))
    v = rng.standard_normal((25, 1))
    fm = LowRankMatrix(u, v)
    res = adaptive_cross_cached(fm, eps=1e-10, rank1_only=True, random_seed=42)
    A = evaluate_small(fm)
    err = np.linalg.norm(A - res.reconstruct_small()) / np.linalg.norm(A)
    assert err < 1e-12
    assert res.rank == 1


def test_rank3_with_rank1_only_is_insufficient() -> None:
    rng = make_rng(0)
    U = rng.standard_normal((20, 3))
    V = rng.standard_normal((25, 3))
    fm = LowRankMatrix(U, V)
    res = adaptive_cross_cached(fm, rank1_only=True, random_seed=42)
    A = evaluate_small(fm)
    err = np.linalg.norm(A - res.reconstruct_small()) / np.linalg.norm(A)
    assert err > 1e-2  # clearly insufficient
    assert res.residual_estimate > 0.0


def test_full_aca_recovers_low_rank_matrix() -> None:
    rng = make_rng(0)
    U = rng.standard_normal((20, 3))
    V = rng.standard_normal((25, 3))
    fm = LowRankMatrix(U, V)
    res = adaptive_cross_cached(fm, max_rank=10, eps=1e-12, random_seed=42)
    A = evaluate_small(fm)
    err = np.linalg.norm(A - res.reconstruct_small()) / np.linalg.norm(A)
    assert err < 1e-10


@pytest.mark.parametrize("seed", [0, 17])
def test_noisy_matrix_bounded_residual(seed: int) -> None:
    rng = make_rng(seed)
    U = rng.standard_normal((20, 2))
    V = rng.standard_normal((20, 2))
    fm = LowRankPlusNoise(U, V, noise_level=1e-3, random_seed=seed)
    res = adaptive_cross_cached(fm, max_rank=10, eps=5e-3, random_seed=seed)
    A = evaluate_small(fm)
    err = np.linalg.norm(A - res.reconstruct_small()) / np.linalg.norm(A)
    # The signal Frobenius norm dominates; the absolute error should remain
    # roughly at or above the noise floor — not blow up.
    assert err < 1.0


def test_oracle_uses_only_row_col_entry() -> None:
    """ACA must not call matvec / rmatvec / block (PDF p. 4 acceptance)."""
    rng = make_rng(0)
    U = rng.standard_normal((10, 2))
    V = rng.standard_normal((12, 2))
    fm = LowRankMatrix(U, V)
    res = adaptive_cross_cached(fm, max_rank=4, eps=1e-12, random_seed=42)
    counts = res.oracle_counts
    assert counts.matvec_calls == 0
    assert counts.matmat_calls == 0
    assert counts.block_calls == 0
