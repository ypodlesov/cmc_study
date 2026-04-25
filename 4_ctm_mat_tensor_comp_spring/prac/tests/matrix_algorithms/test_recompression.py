"""Tests for Task 5 — recompression of low-rank decompositions."""

from __future__ import annotations

import numpy as np
import pytest

from matcomp.matrix_algorithms.recompression import recompress_low_rank
from matcomp.utils.low_rank import CMRFactors, USVFactors, UVFactors
from matcomp.utils.seeding import make_rng


@pytest.mark.parametrize("seed", [0, 17, 12345])
def test_redundant_uv_recovers_true_rank(seed: int) -> None:
    rng = make_rng(seed)
    m, n, true_r = 20, 18, 4
    U_true = rng.standard_normal((m, true_r))
    V_true = rng.standard_normal((n, true_r))
    A = U_true @ V_true.T
    # Build a deliberately redundant rank-(2 * true_r) UV pair.
    pad_U = 1e-14 * rng.standard_normal((m, 4))
    pad_V = 1e-14 * rng.standard_normal((n, 4))
    redundant = UVFactors(
        U=np.column_stack([U_true, pad_U]),
        V=np.column_stack([V_true, pad_V]),
    )
    res = recompress_low_rank(redundant, eps=1e-10)
    assert res.new_rank == true_r
    err = np.linalg.norm(A - res.reconstruct_small()) / np.linalg.norm(A)
    assert err < 1e-12


def test_usv_input_format() -> None:
    rng = make_rng(0)
    A = rng.standard_normal((15, 12))
    U_, S_, Vt_ = np.linalg.svd(A, full_matrices=False)
    res = recompress_low_rank(USVFactors(U_, S_, Vt_), rank=3)
    err = np.linalg.norm(A - res.reconstruct_small()) / np.linalg.norm(A)
    optimal = float(np.sqrt(np.sum(S_[3:] ** 2)) / np.linalg.norm(A))
    np.testing.assert_allclose(err, optimal, rtol=1e-12, atol=1e-12)
    assert res.new_rank == 3


def test_cmr_input_format() -> None:
    rng = make_rng(0)
    U = rng.standard_normal((20, 4))
    V = rng.standard_normal((20, 4))
    A = U @ V.T
    I = np.array([0, 5, 10, 15], dtype=np.intp)
    J = np.array([2, 7, 12, 17], dtype=np.intp)
    cmr = CMRFactors(C=A[:, J], M=np.linalg.pinv(A[np.ix_(I, J)]), R=A[I, :])
    res = recompress_low_rank(cmr, eps=1e-10)
    err = np.linalg.norm(A - res.reconstruct_small()) / np.linalg.norm(A)
    assert err < 1e-10


def test_eps_truncation_respects_threshold() -> None:
    """Build a deliberately heavy-tailed singular spectrum and verify truncation."""
    rng = make_rng(0)
    sigma = np.array([1.0, 1e-1, 1e-2, 1e-6, 1e-12])
    U = np.linalg.qr(rng.standard_normal((20, 5)))[0]
    Vt = np.linalg.qr(rng.standard_normal((15, 5)))[0].T
    res = recompress_low_rank(USVFactors(U, sigma, Vt), eps=1e-3)
    # We should have dropped the tail at 1e-6 and 1e-12.
    assert res.new_rank == 3


def test_compression_reduces_memory() -> None:
    rng = make_rng(0)
    m, n, true_r = 30, 28, 3
    U = rng.standard_normal((m, true_r))
    V = rng.standard_normal((n, true_r))
    redundant = UVFactors(
        U=np.column_stack([U, 1e-14 * rng.standard_normal((m, 5))]),
        V=np.column_stack([V, 1e-14 * rng.standard_normal((n, 5))]),
    )
    before = redundant.factors_memory()
    res = recompress_low_rank(redundant, eps=1e-10)
    after = res.factors_memory()
    assert after < before


def test_invalid_arguments_raise() -> None:
    fac = UVFactors(np.eye(3), np.eye(3))
    with pytest.raises(ValueError):
        recompress_low_rank(fac)  # neither eps nor rank
    with pytest.raises(ValueError):
        recompress_low_rank(fac, eps=1e-3, rank=2)  # both supplied
