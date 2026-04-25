"""Tests for Task 7 — SVD-based quality control harness."""

from __future__ import annotations

import numpy as np

from matcomp.matrix_algorithms.cross import cross_rank_r
from matcomp.matrix_algorithms.randomized_svd import randomized_svd
from matcomp.matrix_algorithms.svd_control import (
    benchmark_against_svd,
    benchmark_to_csv_rows,
    compare_eigenvalues_against_eigh,
    reference_svd_method,
)
from matcomp.utils.functional_matrix import FunctionalMatrix
from matcomp.utils.low_rank import LowRankApprox
from matcomp.utils.seeding import make_rng
from matcomp.utils.test_matrices import HilbertMatrix


def _cross(fm: FunctionalMatrix, r: int) -> LowRankApprox:
    return cross_rank_r(fm, r=r, init="maxvol", max_sweeps=5, random_seed=42)


def _rsvd(fm: FunctionalMatrix, r: int) -> LowRankApprox:
    return randomized_svd(fm, r=r, oversampling=10, n_power_iter=2, random_seed=42)


def test_svd_reference_produces_unit_ratio() -> None:
    H = HilbertMatrix(15, 15)
    rows = benchmark_against_svd(
        {"hilbert": H},
        {"svd": reference_svd_method},
        ranks=[2, 4, 6],
    )
    for row in rows:
        np.testing.assert_allclose(row.ratio_to_svd_frob, 1.0, rtol=1e-9)


def test_all_methods_produce_finite_results() -> None:
    H = HilbertMatrix(15, 15)
    rows = benchmark_against_svd(
        {"hilbert": H},
        {"svd": reference_svd_method, "cross": _cross, "rsvd": _rsvd},
        ranks=[3, 5],
    )
    assert len(rows) == 6
    for row in rows:
        assert np.isfinite(row.rel_frob)
        assert np.isfinite(row.rel_spec)
        assert row.compression_ratio >= 0.0
        assert row.time_s >= 0.0


def test_benchmark_to_csv_rows_layout() -> None:
    H = HilbertMatrix(12, 12)
    rows = benchmark_against_svd({"h": H}, {"svd": reference_svd_method}, ranks=[2])
    csv = benchmark_to_csv_rows(rows)
    # First row is the header.
    assert csv[0][0] == "method"
    # All rows must have equal length.
    assert {len(r) for r in csv} == {len(csv[0])}


def test_compare_eigenvalues_against_eigh() -> None:
    rng = make_rng(0)
    n = 20
    B = rng.standard_normal((n, n))
    A = (B + B.T) / 2
    rows = compare_eigenvalues_against_eigh(
        A, lambda x: A @ x, n=n, k_values=[5, 10, 15], random_seed=42
    )
    # Convergence: error should not grow as k increases.
    for prev, cur in zip(rows[:-1], rows[1:], strict=True):
        assert cur["err_max"] <= prev["err_max"] + 1e-12
