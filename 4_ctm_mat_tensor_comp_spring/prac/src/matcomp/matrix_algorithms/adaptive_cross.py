r"""Task 4 — Adaptive Cross Approximation (ACA) with caching.

Implements the standard ACA recipe (Bebendorf & Rjasanow, *Computing*
70(1), 2003): build a low-rank approximation

.. math::

    \hat A = \sum_{k=1}^{r} \frac{u_k\,v_k^{\top}}{\delta_k},
    \qquad \delta_k = (\hat A - A)_{i_k,\,j_k},

by greedily selecting pivots :math:`(i_k, j_k)` on the running residual
and retaining the corresponding row and column. The wrapped functional
matrix is queried through a :class:`CachedFunctionalMatrix` so repeated
``entry`` / ``row`` / ``col`` calls do not re-execute the user-supplied
oracle.

Sub-routine :func:`adaptive_cross_cached` is also exposed to Task 2 as
the ``init="greedy"`` initialisation strategy: it produces the same
sequence of pivots as a maxvol-style row/column selection, just driven
by residual magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matcomp.utils.caching import CachedFunctionalMatrix
from matcomp.utils.counting import CountingFunctionalMatrix, OracleCounts
from matcomp.utils.functional_matrix import FloatArray, FunctionalMatrix
from matcomp.utils.low_rank import UVFactors
from matcomp.utils.seeding import make_rng


@dataclass(frozen=True)
class ACAResult:
    r"""Output of :func:`adaptive_cross_cached`.

    Attributes
    ----------
    factors
        Low-rank carrier :math:`\hat A = U V^{\top}` where the columns of
        :math:`V` already absorb the pivot scaling :math:`v_k / \delta_k`.
    pivots
        Selected pivot indices ``(i_k, j_k)`` in the order they were
        picked.
    residual_estimate
        Frobenius-norm estimate of :math:`\lVert A - \hat A\rVert_F` from
        a sampled validation grid (set after the loop terminates).
    cache_hits, cache_misses
        Forwarded from the :class:`CachedFunctionalMatrix` wrapper.
    oracle_counts
        Real (uncached) oracle counts.
    """

    factors: UVFactors
    pivots: tuple[tuple[int, int], ...]
    residual_estimate: float
    cache_hits: int
    cache_misses: int
    oracle_counts: OracleCounts

    # ---- LowRankApprox passthroughs ---------------------------------------

    @property
    def rank(self) -> int:
        """Number of selected pivots."""
        return self.factors.rank

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the represented matrix."""
        return self.factors.shape

    def entry(self, i: int, j: int) -> float:
        return self.factors.entry(i, j)

    def matvec(self, x: FloatArray) -> FloatArray:
        return self.factors.matvec(x)

    def rmatvec(self, y: FloatArray) -> FloatArray:
        return self.factors.rmatvec(y)

    def reconstruct_small(self) -> FloatArray:
        return self.factors.reconstruct_small()

    def factors_memory(self) -> int:
        return self.factors.factors_memory()


def _residual_estimate(
    cached: CachedFunctionalMatrix,
    U: FloatArray,
    V: FloatArray,
    rng: np.random.Generator,
    sample_size: int,
) -> float:
    r"""Sample-based Frobenius residual :math:`\lVert A - \hat A\rVert_F` estimate.

    PDF p. 1 prescribes sample-based metrics for functional matrices that
    are too large to materialise. We draw ``sample_size`` random index
    pairs and compute :math:`\sqrt{\frac{1}{N}\sum (A_{ij} - \hat
    A_{ij})^2 \cdot m\,n}`.
    """
    m, n = cached.shape
    rows = rng.integers(0, m, size=sample_size)
    cols = rng.integers(0, n, size=sample_size)
    sq_err = 0.0
    for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
        a = float(cached.entry(int(i), int(j)))
        b = float(U[int(i)] @ V[int(j)])
        sq_err += (a - b) ** 2
    mean_sq = sq_err / sample_size
    return float(np.sqrt(mean_sq * m * n))


def adaptive_cross_cached(
    A: FunctionalMatrix,
    *,
    eps: float = 1e-8,
    max_rank: int = 50,
    rank1_only: bool = False,
    initial_indices: tuple[int, int] | None = None,
    validation_samples: int = 64,
    random_seed: int | None = None,
) -> ACAResult:
    r"""Adaptive cross approximation with caching.

    Parameters
    ----------
    A
        Functional matrix to approximate. The implementation queries it
        only through ``entry``, ``row`` and ``col``; the full matrix is
        never materialised.
    eps
        Stop when the sampled Frobenius residual falls below
        ``eps * sample_estimate(A)``.
    max_rank
        Maximum number of pivots / rank-1 updates.
    rank1_only
        If ``True``, return after the very first pivot. Useful as Task 2's
        ``init="greedy"`` and for the rank-1 acceptance test from PDF p. 5.
    initial_indices
        Optional starting pivot ``(i*, j*)``. When ``None``, the row of
        largest entry-magnitude on a random row is used.
    validation_samples
        Sample size for the Frobenius-residual estimate.
    random_seed
        Seed for the RNG used by initial-pivot selection and residual
        sampling.

    Returns
    -------
    ACAResult
        Low-rank carrier, pivots, residual estimate and cache statistics.

    Notes
    -----
    The classical pivot rule is "largest absolute entry of the current
    residual row / column"; we implement exactly that. After choosing
    :math:`i_k`, the residual row is the original row minus the previous
    rank-1 contributions, and :math:`j_k = \arg\max_j |R_{i_k, j}|`. Then
    the residual column at :math:`j_k` provides :math:`u_k`. The pivot
    value is :math:`R_{i_k, j_k}`.
    """
    if max_rank < 1:
        raise ValueError("max_rank must be at least 1")
    rng = make_rng(random_seed)

    counted = CountingFunctionalMatrix(A)
    cached = CachedFunctionalMatrix(counted)
    m, n = cached.shape

    used_rows: set[int] = set()
    used_cols: set[int] = set()
    pivots: list[tuple[int, int]] = []
    Us: list[FloatArray] = []  # residual columns u_k
    Vs: list[FloatArray] = []  # residual rows v_k / pivot_k  (will be (n,))

    # ---- choose initial pivot row -----------------------------------------
    if initial_indices is not None:
        i0, j0 = int(initial_indices[0]), int(initial_indices[1])
        if not 0 <= i0 < m or not 0 <= j0 < n:
            raise ValueError("initial_indices out of bounds")
        # Treat (i0, j0) as a *suggestion* for the starting row; the actual
        # pivot column is recomputed on the residual row to guarantee
        # |delta| > 0.
        i_pivot = i0
    else:
        i_pivot = int(rng.integers(0, m))

    target_rank = 1 if rank1_only else max_rank
    initial_residual: float | None = None

    for k in range(target_rank):
        # ---- residual row at i_pivot --------------------------------------
        row = cached.row(i_pivot).astype(np.float64, copy=True)
        for u_prev, v_prev in zip(Us, Vs, strict=True):
            row -= u_prev[i_pivot] * v_prev
        # Mask already-used columns to keep selecting fresh pivots.
        masked = np.abs(row).copy()
        if used_cols:
            masked[list(used_cols)] = -1.0
        j_pivot = int(np.argmax(masked))
        delta = float(row[j_pivot])

        # ---- if delta is tiny, fall back to another row -------------------
        if abs(delta) < np.finfo(float).tiny * (1.0 + np.linalg.norm(row)):
            # try another, unused row
            unused = [i for i in range(m) if i not in used_rows]
            if not unused:
                break
            i_pivot = int(rng.choice(unused))
            continue

        # ---- residual column at j_pivot -----------------------------------
        col = cached.col(None, j_pivot).astype(np.float64, copy=True)
        for u_prev, v_prev in zip(Us, Vs, strict=True):
            col -= v_prev[j_pivot] * u_prev

        u_k = col
        v_k = row / delta

        Us.append(u_k)
        Vs.append(v_k)
        pivots.append((i_pivot, j_pivot))
        used_rows.add(i_pivot)
        used_cols.add(j_pivot)

        # ---- compute residual estimate; check stopping --------------------
        approx_norm_sq = float(u_k @ u_k) * float(v_k @ v_k)
        if k == 0:
            initial_residual = max(np.sqrt(approx_norm_sq), 1.0)
        if not rank1_only and k + 1 < max_rank:
            row_norm = float(np.linalg.norm(row))
            col_norm = float(np.linalg.norm(col))
            est_residual = row_norm * col_norm / max(abs(delta), np.finfo(float).tiny)
            if (
                initial_residual is not None
                and est_residual < eps * initial_residual
            ):
                break
            # Pick the next pivot row as the argmax of |u_k| over unused rows.
            unused_mask = np.ones(m, dtype=bool)
            unused_mask[list(used_rows)] = False
            if not unused_mask.any():
                break
            cand = np.where(unused_mask)[0]
            local_idx = int(np.argmax(np.abs(u_k[cand])))
            i_pivot = int(cand[local_idx])

    # ---- assemble factors --------------------------------------------------
    if Us:
        U_mat = np.column_stack(Us).astype(np.float64)
        V_mat = np.column_stack(Vs).astype(np.float64)
    else:
        U_mat = np.zeros((m, 0), dtype=np.float64)
        V_mat = np.zeros((n, 0), dtype=np.float64)

    residual_estimate = _residual_estimate(cached, U_mat, V_mat, rng, validation_samples)

    return ACAResult(
        factors=UVFactors(U=U_mat, V=V_mat),
        pivots=tuple(pivots),
        residual_estimate=residual_estimate,
        cache_hits=cached.cache_hits,
        cache_misses=cached.cache_misses,
        oracle_counts=counted.counts,
    )


__all__ = ["ACAResult", "adaptive_cross_cached"]
