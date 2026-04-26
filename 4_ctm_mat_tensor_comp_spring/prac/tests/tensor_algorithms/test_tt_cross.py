"""Tests for Task 13 — TT-cross."""

from __future__ import annotations

import numpy as np
import pytest

from matcomp.tensor_algorithms.tt_cross import tt_cross
from matcomp.utils.functional_tensor import DenseTensor
from matcomp.utils.seeding import make_rng
from matcomp.utils.tensor_linalg import tt_to_dense
from matcomp.utils.tensor_test_objects import (
    FunctionTensor,
    Hilbert3DTensor,
)


def test_recovers_exact_tt_with_sufficient_ranks_and_sweeps() -> None:
    rng = make_rng(0)
    cores = [
        rng.standard_normal((1, 5, 2)),
        rng.standard_normal((2, 4, 3)),
        rng.standard_normal((3, 6, 1)),
    ]
    X = tt_to_dense(cores)
    ft = DenseTensor(X)
    res = tt_cross(ft, ranks=(2, 3), max_sweeps=10, tol=None, random_seed=0)
    err = np.linalg.norm(X - res.reconstruct_small()) / np.linalg.norm(X)
    assert err < 1e-10


def test_does_not_materialise_for_smooth_tensor() -> None:
    """PDF p. 12 acceptance: 'The full tensor must not be built inside the
    algorithm for large tests'. We assert oracle calls are far below the
    total tensor size."""
    H = Hilbert3DTensor(8)  # 512 entries
    res = tt_cross(H, ranks=4, max_sweeps=4, tol=None, random_seed=0)
    assert res.oracle_counts.total < int(np.prod(H.shape))


def test_cached_oracle_reduces_real_calls() -> None:
    """Cache hits must outnumber misses on multiple sweeps over the same tensor."""
    H = Hilbert3DTensor(8)
    res = tt_cross(H, ranks=4, max_sweeps=4, tol=None, random_seed=0)
    assert res.cache_hits > res.cache_misses
    # And the real-oracle counter equals the number of misses.
    assert res.oracle_counts.total == res.cache_misses


def test_reconstructs_within_tol_for_kernel_tensor() -> None:
    rng = make_rng(0)
    # Smooth analytic 3-D tensor that admits a low TT rank.
    H = FunctionTensor(
        func=lambda idx: np.exp(-(idx[0] - idx[1]) ** 2 / 50.0 - (idx[1] - idx[2]) ** 2 / 50.0),
        shape=(8, 8, 8),
    )
    res = tt_cross(H, ranks=(6, 6), max_sweeps=6, tol=None, random_seed=0)
    # Compare on a random sample to keep the test independent of materialisation.
    sample_idx = rng.integers(0, 8, size=(50, 3)).astype(np.intp)
    truths = np.array([H.entry(tuple(int(v) for v in sample_idx[k])) for k in range(50)])
    preds = np.array([res.entry(tuple(int(v) for v in sample_idx[k])) for k in range(50)])
    err = np.linalg.norm(truths - preds) / np.linalg.norm(truths)
    # PDF Task 13 acceptance: "On an exact TT test, the error must be small
    # with sufficient ranks". For a smooth (non-exact-TT) target a 5%
    # tolerance at rank 6 with random initial right-indices is realistic.
    assert err < 5e-2


def test_invalid_ranks_raise() -> None:
    H = Hilbert3DTensor(6)
    with pytest.raises(ValueError):
        tt_cross(H, ranks=(2,), max_sweeps=1)  # too few interior ranks
    with pytest.raises(ValueError):
        tt_cross(H, ranks=(2, 0), max_sweeps=1)  # rank must be >= 1


def test_tt_factors_have_consistent_dimensions() -> None:
    H = Hilbert3DTensor(6)
    res = tt_cross(H, ranks=(3, 2), max_sweeps=2, tol=None, random_seed=0)
    cores = list(res.factors.cores)
    assert cores[0].shape[0] == 1
    assert cores[-1].shape[2] == 1
    for k in range(len(cores) - 1):
        assert cores[k].shape[2] == cores[k + 1].shape[0]
        assert cores[k].shape[1] == H.shape[k]
