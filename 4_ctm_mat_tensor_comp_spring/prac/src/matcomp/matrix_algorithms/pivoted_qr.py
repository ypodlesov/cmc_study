r"""Task 6 — Pivoted (rank-revealing) QR decomposition.

Implements column-pivoted Householder QR from scratch, following the
classical Chan (1987) / Gu–Eisenstat (1996) formulation:

.. math::

    A\,P = Q\,R,

where :math:`P` is a permutation that brings the columns of largest
remaining 2-norm to the front. The diagonal of :math:`R` is monotonically
non-increasing; truncating at the index where :math:`|R_{kk}|` falls below
``tol * |R_{00}|`` reveals the numerical rank.

The implementation uses Householder reflectors (numerically stable for
nearly-dependent columns) with downdated column-norm bookkeeping so each
reflector costs :math:`O(m\,n)` rather than the :math:`O(m\,n\,k)` of
naive recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matcomp.utils.functional_matrix import FloatArray, IntArray
from matcomp.utils.low_rank import LowRankApprox  # noqa: F401  (used in docstring)


@dataclass(frozen=True)
class PivotedQRResult:
    r"""Result of :func:`pivoted_qr_approx`.

    The factors satisfy :math:`A[:, P] = Q\,R`. The rank-:math:`r`
    approximation of the original matrix is therefore
    :math:`A \approx Q[:, :r]\,R[:r, :]\,P^{-1}`.

    Attributes
    ----------
    Q
        Orthonormal matrix of shape :math:`(m, k)` with
        :math:`k = \min(m, n)`.
    R
        Upper-triangular factor of shape :math:`(k, n)`.
    P
        Integer permutation array of length :math:`n`. ``A[:, P] == Q @ R``.
    estimated_rank
        Numerical rank inferred from the diagonal decay (or the user-supplied
        ``rank`` argument).
    """

    Q: FloatArray
    R: FloatArray
    P: IntArray
    estimated_rank: int

    @property
    def rank(self) -> int:
        """Alias for :attr:`estimated_rank` (satisfies :class:`LowRankApprox`)."""
        return int(self.estimated_rank)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the represented matrix."""
        return (int(self.Q.shape[0]), int(self.P.size))

    def _inverse_perm(self) -> IntArray:
        inv = np.empty_like(self.P)
        inv[self.P] = np.arange(self.P.size, dtype=self.P.dtype)
        return inv

    def entry(self, i: int, j: int) -> float:
        """Return :math:`\\hat A_{ij}` of the rank-truncated approximation."""
        r = int(self.estimated_rank)
        col_perm = int(self._inverse_perm()[j])
        return float(self.Q[i, :r] @ self.R[:r, col_perm])

    def matvec(self, x: FloatArray) -> FloatArray:
        r"""Return :math:`\\hat A\,x = Q[:, :r] R[:r, :] P^{-1} x`."""
        r = int(self.estimated_rank)
        return self.Q[:, :r] @ (self.R[:r, :] @ x[self._inverse_perm()])

    def rmatvec(self, y: FloatArray) -> FloatArray:
        r"""Return :math:`\\hat A^{\\top} y = (P^{-1})^{\\top} R^{\\top} Q^{\\top} y`."""
        r = int(self.estimated_rank)
        v = self.R[:r, :].T @ (self.Q[:, :r].T @ y)
        # Permute: out[P[k]] = v[k]
        out = np.empty_like(v)
        out[self.P] = v
        return out

    def reconstruct_small(self) -> FloatArray:
        """Materialise the rank-truncated approximation as a dense matrix."""
        r = int(self.estimated_rank)
        approx_perm = self.Q[:, :r] @ self.R[:r, :]
        out = np.empty_like(approx_perm)
        out[:, self.P] = approx_perm
        return out

    def factors_memory(self) -> int:
        """Sum of ``nbytes`` over ``Q``, ``R`` and ``P``."""
        return int(self.Q.nbytes + self.R.nbytes + self.P.nbytes)


def _householder_vector(x: FloatArray) -> tuple[FloatArray, float]:
    r"""Return ``(v, beta)`` such that :math:`(I - \\beta v v^{\\top}) x = \\pm \\lVert x \\rVert e_0`.

    Standard Householder construction with the LAPACK sign convention to
    avoid catastrophic cancellation.
    """
    sigma = float(x[1:] @ x[1:])
    v = x.astype(np.float64, copy=True)
    if sigma == 0.0 and x[0] >= 0.0:
        v[0] = 1.0
        return v, 0.0
    if sigma == 0.0 and x[0] < 0.0:
        v[0] = 1.0
        return v, -2.0
    mu = float(np.sqrt(x[0] * x[0] + sigma))
    if x[0] <= 0.0:
        v0 = x[0] - mu
    else:
        v0 = -sigma / (x[0] + mu)
    beta = 2.0 * v0 * v0 / (sigma + v0 * v0)
    v[0] = 1.0
    v[1:] = v[1:] / v0
    return v, beta


def pivoted_qr_approx(
    A: FloatArray,
    *,
    tol: float | None = None,
    rank: int | None = None,
) -> PivotedQRResult:
    r"""Column-pivoted Householder QR with rank revelation.

    Parameters
    ----------
    A
        Real matrix of shape :math:`(m, n)`.
    tol
        Relative tolerance used to estimate the numerical rank from
        :math:`|R_{kk}| / |R_{00}|`. Mutually exclusive with ``rank``. If
        both are ``None``, the estimated rank is :math:`\min(m, n)`.
    rank
        User-supplied target rank. Mutually exclusive with ``tol``.

    Returns
    -------
    PivotedQRResult
        Factors :math:`Q`, :math:`R`, permutation ``P`` and estimated rank.

    Notes
    -----
    The pivoting rule is greedy: at step :math:`k`, the column with the
    largest remaining 2-norm in :math:`A[k:, :]` is moved into position
    :math:`k`. To keep the cost at :math:`O(m\,n^{2})` rather than
    :math:`O(m\,n^{3})`, the per-column squared norms are *downdated*:
    after applying a Householder reflector to ``R``, every column's norm
    is reduced by the square of the new diagonal element. A periodic
    re-norm is performed when round-off makes the running estimate
    unreliable (see Drmač, 1996, for the discussion of the numerical
    pitfall).
    """
    if A.ndim != 2:
        raise ValueError("pivoted_qr_approx expects a 2-D matrix")
    if tol is not None and rank is not None:
        raise ValueError("pass at most one of tol or rank")
    m, n = A.shape
    k_max = min(m, n)
    R = A.astype(np.float64, copy=True)
    Q = np.eye(m, dtype=np.float64)
    P = np.arange(n, dtype=np.intp)

    # Initial column squared norms.
    col_sq_norms = np.einsum("ij,ij->j", R, R).astype(np.float64)
    initial_max = float(col_sq_norms.max() if col_sq_norms.size > 0 else 0.0)
    # Threshold below which we re-norm to clean up round-off.
    renorm_floor = 1e-14 * initial_max

    for step in range(k_max):
        # Pick the column with the largest remaining 2-norm.
        pivot = step + int(np.argmax(col_sq_norms[step:]))
        if col_sq_norms[pivot] <= 0.0:
            # All remaining columns are numerically zero — early exit.
            break
        if pivot != step:
            R[:, [step, pivot]] = R[:, [pivot, step]]
            P[[step, pivot]] = P[[pivot, step]]
            col_sq_norms[[step, pivot]] = col_sq_norms[[pivot, step]]

        # Householder reflector against the lower part of column `step`.
        v, beta = _householder_vector(R[step:, step])
        if beta != 0.0:
            R_block = R[step:, step:]
            w = R_block.T @ v
            R_block -= beta * np.outer(v, w)
            Q_block = Q[:, step:]
            wq = Q_block @ v
            Q_block -= beta * np.outer(wq, v)
        # Zero out the strict lower-triangular part of column `step` for cleanliness.
        R[step + 1 :, step] = 0.0

        # Downdate column norms (skip the just-processed column).
        if step + 1 < n:
            row = R[step, step + 1 :]
            col_sq_norms[step + 1 :] -= row * row
            # Re-norm when downdate has eaten too much of the original.
            mask = col_sq_norms[step + 1 :] < renorm_floor
            if np.any(mask):
                idx = np.where(mask)[0] + step + 1
                col_sq_norms[idx] = np.einsum(
                    "ij,ij->j", R[step + 1 :, idx], R[step + 1 :, idx]
                )

    R = R[:k_max, :]
    Q = Q[:, :k_max]

    diag = np.abs(np.diag(R))
    estimated_rank: int
    if rank is not None:
        estimated_rank = int(min(max(rank, 0), k_max))
    elif tol is not None and diag.size > 0 and diag[0] > 0.0:
        estimated_rank = int(np.sum(diag / diag[0] > tol))
    else:
        estimated_rank = int(k_max)

    return PivotedQRResult(Q=Q, R=R, P=P, estimated_rank=estimated_rank)


__all__ = ["PivotedQRResult", "pivoted_qr_approx"]
