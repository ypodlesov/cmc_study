"""Tests for Task 8 — CP-ALS with a CP source tensor."""

from __future__ import annotations

import numpy as np
import pytest

from matcomp.tensor_algorithms.cp_als_cp import cp_als_from_cp
from matcomp.utils.seeding import make_rng
from matcomp.utils.tensor_linalg import cp_to_dense
from matcomp.utils.tensor_low_rank import CPFactors
from matcomp.utils.tensor_test_objects import random_cp


def test_recovers_source_when_target_rank_matches() -> None:
    rng = make_rng(0)
    weights, factors = random_cp((6, 7, 8), 4, rng)
    src = CPFactors(weights=weights, factors=tuple(factors))
    res = cp_als_from_cp(src, target_rank=4, init="random", max_iter=200, tol=1e-12, random_seed=0)
    assert res.loss_history[-1] < 1e-8


def test_loss_decreases_each_sweep() -> None:
    rng = make_rng(1)
    weights, factors = random_cp((5, 6, 7), 6, rng)
    src = CPFactors(weights=weights, factors=tuple(factors))
    res = cp_als_from_cp(src, target_rank=3, init="random", max_iter=80, tol=1e-12, random_seed=0)
    diffs = np.diff(res.loss_history)
    assert np.all(diffs <= 1e-10), f"loss should be ~monotone: {diffs}"


def test_compression_to_smaller_rank_yields_positive_error() -> None:
    rng = make_rng(2)
    weights, factors = random_cp((5, 6, 7), 8, rng)
    src = CPFactors(weights=weights, factors=tuple(factors))
    res = cp_als_from_cp(src, target_rank=3, init="random", max_iter=100, tol=1e-12, random_seed=0)
    # Compressing rank 8 to rank 3 cannot be exact in general.
    assert res.loss_history[-1] > 1e-4


def test_invalid_target_rank_raises() -> None:
    rng = make_rng(0)
    weights, factors = random_cp((4, 5, 6), 3, rng)
    src = CPFactors(weights=weights, factors=tuple(factors))
    with pytest.raises(ValueError):
        cp_als_from_cp(src, target_rank=0)


def test_loss_consistent_with_dense_reference() -> None:
    """The CP-vs-CP loss must match the dense error to a tight tolerance."""
    rng = make_rng(3)
    weights, factors = random_cp((5, 6, 4), 5, rng)
    src = CPFactors(weights=weights, factors=tuple(factors))
    res = cp_als_from_cp(src, target_rank=3, init="random", max_iter=80, tol=1e-12, random_seed=0)
    src_dense = cp_to_dense(weights, factors)
    dense_err = np.linalg.norm(src_dense - res.reconstruct_small()) / np.linalg.norm(src_dense)
    np.testing.assert_allclose(res.loss_history[-1], dense_err, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("init", ["random", "source"])
def test_init_strategies_run(init: str) -> None:
    rng = make_rng(0)
    weights, factors = random_cp((4, 5, 6), 4, rng)
    src = CPFactors(weights=weights, factors=tuple(factors))
    res = cp_als_from_cp(src, target_rank=3, init=init, max_iter=20, tol=None, random_seed=0)  # type: ignore[arg-type]
    assert res.iterations >= 1
