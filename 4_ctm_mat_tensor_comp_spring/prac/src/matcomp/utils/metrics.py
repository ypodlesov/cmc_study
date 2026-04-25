r"""Quality metrics shared across all matrix algorithms.

The practicum (PDF p. 2) defines four metrics that every implementation
must report when applicable:

* ``rel_frob`` — relative Frobenius-norm error :math:`\lVert A - \hat A
  \rVert_F / \lVert A \rVert_F` (main approximation metric).
* ``rel_spec`` — relative spectral-norm error :math:`\lVert A - \hat A
  \rVert_2 / \lVert A \rVert_2` (compared to the SVD tail).
* ``sample_rmse`` — root-mean-square error on a random sample of indices,
  used when :math:`A` is too large to materialise.
* ``compression_ratio`` — :math:`m\,n\,\mathrm{itemsize} / \text{factor
  storage}`, mandatory for "all low-rank methods".

This module provides each of them in a single place so the whole project
shares one definition.
"""

from __future__ import annotations

import numpy as np

from matcomp.utils.functional_matrix import FloatArray, FunctionalMatrix
from matcomp.utils.low_rank import LowRankApprox
from matcomp.utils.seeding import make_rng


def rel_frob(reference: FloatArray, approximation: FloatArray) -> float:
    r"""Relative Frobenius-norm error :math:`\lVert A - \hat A \rVert_F / \lVert A \rVert_F`.

    Parameters
    ----------
    reference
        The full matrix :math:`A`.
    approximation
        The approximation :math:`\hat A` (same shape).

    Returns
    -------
    float
        Relative Frobenius error. Returns ``0`` when both arrays are zero.
    """
    if reference.shape != approximation.shape:
        raise ValueError(
            f"shape mismatch: ref {reference.shape}, approx {approximation.shape}"
        )
    denom = float(np.linalg.norm(reference, ord="fro"))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(reference - approximation, ord="fro") / denom)


def rel_spec(reference: FloatArray, approximation: FloatArray) -> float:
    r"""Relative spectral-norm (operator-2) error :math:`\lVert A - \hat A \rVert_2 / \lVert A \rVert_2`."""
    if reference.shape != approximation.shape:
        raise ValueError(
            f"shape mismatch: ref {reference.shape}, approx {approximation.shape}"
        )
    denom = float(np.linalg.norm(reference, ord=2))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(reference - approximation, ord=2) / denom)


def sample_rmse(
    fm: FunctionalMatrix,
    approx: LowRankApprox,
    num_samples: int,
    random_seed: int | np.random.Generator | None = None,
) -> float:
    r"""Sampled root-mean-square error on random indices.

    The PDF (p. 1) prescribes this for functional matrices that are too
    large to materialise. We draw ``num_samples`` independent
    ``(i, j)`` pairs uniformly and compare the oracle value with the
    approximation.

    Parameters
    ----------
    fm
        The reference functional matrix.
    approx
        The low-rank approximation under test.
    num_samples
        Number of random index pairs to draw.
    random_seed
        Seed (or generator) for reproducibility.

    Returns
    -------
    float
        :math:`\sqrt{\frac{1}{N} \sum_{(i,j)\in\Omega}(A_{ij} - \hat A_{ij})^2}`.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if fm.shape != approx.shape:
        raise ValueError(
            f"shape mismatch: fm {fm.shape}, approx {approx.shape}"
        )
    rng = make_rng(random_seed)
    rows = rng.integers(0, fm.shape[0], size=num_samples)
    cols = rng.integers(0, fm.shape[1], size=num_samples)
    sq_err = 0.0
    for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
        a = float(fm.entry(i, j))
        b = float(approx.entry(i, j))
        sq_err += (a - b) ** 2
    return float(np.sqrt(sq_err / num_samples))


def best_rank_r_error_frob(reference: FloatArray, r: int) -> float:
    r"""Optimal Frobenius-norm error of the best rank-``r`` approximation.

    Computed via SVD on a small dense reference matrix. By Eckart–Young,
    this is :math:`\sqrt{\sum_{k > r} \sigma_k^2}`.
    """
    if r < 0:
        raise ValueError("r must be non-negative")
    s = np.linalg.svd(reference, compute_uv=False)
    if r >= s.size:
        return 0.0
    return float(np.sqrt(np.sum(s[r:] ** 2)))


def best_rank_r_error_spec(reference: FloatArray, r: int) -> float:
    r"""Optimal spectral-norm error of the best rank-``r`` approximation.

    By Eckart–Young, this is :math:`\sigma_{r+1}`.
    """
    if r < 0:
        raise ValueError("r must be non-negative")
    s = np.linalg.svd(reference, compute_uv=False)
    if r >= s.size:
        return 0.0
    return float(s[r])


def compression_ratio(approx: LowRankApprox, itemsize: int = 8) -> float:
    r"""Compression ratio :math:`m \cdot n \cdot \text{itemsize} / \text{factors\_memory}`.

    Parameters
    ----------
    approx
        Any object implementing :class:`LowRankApprox`.
    itemsize
        Bytes per dense entry (defaults to ``8`` for ``float64``).

    Returns
    -------
    float
        Ratio :math:`> 1` means the factor representation uses less memory
        than the dense form.
    """
    m, n = approx.shape
    factors_bytes = approx.factors_memory()
    if factors_bytes == 0:
        return float("inf")
    return float(m * n * itemsize) / float(factors_bytes)


__all__ = [
    "best_rank_r_error_frob",
    "best_rank_r_error_spec",
    "compression_ratio",
    "rel_frob",
    "rel_spec",
    "sample_rmse",
]
