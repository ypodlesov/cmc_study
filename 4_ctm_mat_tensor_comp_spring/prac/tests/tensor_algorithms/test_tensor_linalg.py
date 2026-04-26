"""Tests for the tensor linear-algebra primitives."""

from __future__ import annotations

import numpy as np
import pytest

from matcomp.utils.seeding import make_rng
from matcomp.utils.tensor_linalg import (
    cp_inner_product,
    cp_norm_sq,
    cp_to_dense,
    fold,
    khatri_rao,
    mode_dot,
    mttkrp,
    multi_mode_dot,
    multilinear_rank,
    pack_cp_theta,
    tt_entry,
    tt_to_dense,
    tucker_to_dense,
    unfold,
    unpack_cp_theta,
)


def test_unfold_fold_roundtrip() -> None:
    rng = make_rng(0)
    X = rng.standard_normal((4, 5, 6))
    for mode in range(X.ndim):
        Y = fold(unfold(X, mode), mode, X.shape)
        np.testing.assert_allclose(X, Y)


def test_unfold_shape_matches_kolda_dimension() -> None:
    X = np.zeros((3, 4, 2))
    assert unfold(X, 0).shape == (3, 8)
    assert unfold(X, 1).shape == (4, 6)
    assert unfold(X, 2).shape == (2, 12)


def test_mode_dot_against_einsum() -> None:
    rng = make_rng(0)
    X = rng.standard_normal((3, 4, 2))
    M = rng.standard_normal((5, 4))
    np.testing.assert_allclose(mode_dot(X, M, mode=1), np.einsum("ijk,pj->ipk", X, M))
    np.testing.assert_allclose(
        mode_dot(X, M, mode=1, transpose=False) - np.einsum("ijk,pj->ipk", X, M),
        0.0,
        atol=1e-12,
    )


def test_multi_mode_dot_with_default_modes() -> None:
    rng = make_rng(0)
    X = rng.standard_normal((3, 4, 2))
    F = [rng.standard_normal((6, 3)), rng.standard_normal((5, 4)), rng.standard_normal((7, 2))]
    Y = multi_mode_dot(X, F)
    np.testing.assert_allclose(Y, np.einsum("ijk,ai,bj,ck->abc", X, F[0], F[1], F[2]))


def test_khatri_rao_columnwise() -> None:
    rng = make_rng(0)
    A = rng.standard_normal((3, 2))
    B = rng.standard_normal((4, 2))
    K = khatri_rao([A, B])
    expected = np.zeros((12, 2))
    for r in range(2):
        expected[:, r] = np.outer(A[:, r], B[:, r]).ravel()
    np.testing.assert_allclose(K, expected)


def test_mttkrp_matches_einsum_3d() -> None:
    rng = make_rng(0)
    X = rng.standard_normal((4, 5, 6))
    factors = [rng.standard_normal((s, 3)) for s in X.shape]
    np.testing.assert_allclose(
        mttkrp(X, factors, 0), np.einsum("ijk,jr,kr->ir", X, factors[1], factors[2])
    )
    np.testing.assert_allclose(
        mttkrp(X, factors, 1), np.einsum("ijk,ir,kr->jr", X, factors[0], factors[2])
    )
    np.testing.assert_allclose(
        mttkrp(X, factors, 2), np.einsum("ijk,ir,jr->kr", X, factors[0], factors[1])
    )


def test_cp_norm_sq_matches_dense() -> None:
    rng = make_rng(0)
    weights = rng.standard_normal(3) ** 2 + 0.1
    factors = [rng.standard_normal((s, 3)) for s in (5, 6, 7)]
    Y = cp_to_dense(weights, factors)
    np.testing.assert_allclose(cp_norm_sq(weights, factors), float(np.sum(Y * Y)))


def test_cp_inner_product_matches_dense() -> None:
    rng = make_rng(1)
    wa = rng.standard_normal(3)
    Fa = [rng.standard_normal((s, 3)) for s in (4, 5, 6)]
    wb = rng.standard_normal(2)
    Fb = [rng.standard_normal((s, 2)) for s in (4, 5, 6)]
    Ya = cp_to_dense(wa, Fa)
    Yb = cp_to_dense(wb, Fb)
    np.testing.assert_allclose(
        cp_inner_product(wa, Fa, wb, Fb), float(np.sum(Ya * Yb))
    )


def test_tt_entry_matches_tt_to_dense() -> None:
    rng = make_rng(0)
    cores = [rng.standard_normal((1, 4, 2)), rng.standard_normal((2, 5, 3)), rng.standard_normal((3, 6, 1))]
    full = tt_to_dense(cores)
    for idx in [(0, 0, 0), (3, 4, 5), (1, 2, 3)]:
        np.testing.assert_allclose(tt_entry(cores, idx), full[idx])


def test_multilinear_rank_on_tucker_synthetic() -> None:
    rng = make_rng(0)
    core = rng.standard_normal((2, 3, 2))
    factors = [np.linalg.qr(rng.standard_normal((s, r)))[0] for s, r in zip((6, 7, 8), (2, 3, 2), strict=True)]
    Y = tucker_to_dense(core, factors)
    # Float64 round-off pushes "zero" singular values up to ~1e-15. Use a
    # relative tolerance so the count picks up only the meaningful ones.
    assert multilinear_rank(Y, eps=1e-10) == (2, 3, 2)


def test_pack_unpack_cp_theta_roundtrip() -> None:
    rng = make_rng(0)
    weights = rng.standard_normal(2)
    factors = [rng.standard_normal((s, 2)) for s in (3, 4, 5)]
    theta = pack_cp_theta(weights, factors)
    w2, f2 = unpack_cp_theta(theta, (3, 4, 5), 2)
    np.testing.assert_allclose(weights, w2)
    for f, ff in zip(factors, f2, strict=True):
        np.testing.assert_allclose(f, ff)


def test_invalid_inputs_raise() -> None:
    rng = make_rng(0)
    X = rng.standard_normal((3, 4, 2))
    with pytest.raises(ValueError):
        unfold(X, mode=5)
    with pytest.raises(ValueError):
        khatri_rao([], skip=None)
    with pytest.raises(ValueError):
        unpack_cp_theta(np.zeros(3), (3, 4), 2)
