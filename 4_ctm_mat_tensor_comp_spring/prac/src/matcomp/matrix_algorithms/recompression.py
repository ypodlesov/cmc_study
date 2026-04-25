r"""Task 5 — Recompression of an existing low-rank decomposition.

Given a low-rank carrier in any of the three formats supported by this
project (:class:`UVFactors`, :class:`USVFactors`, :class:`CMRFactors`),
return a smaller :class:`USVFactors` that approximates it within either a
relative tolerance ``eps`` (drop singular values below
:math:`\epsilon \cdot \sigma_0`) or a fixed target ``rank``.

Algorithm sketch (PDF p. 6):

1. Reduce the input to the canonical :math:`U V^{\top}` form (folding the
   middle factor for ``USV`` and ``CMR``).
2. ``QR(U) = Q_u R_u`` and ``QR(V) = Q_v R_v``.
3. ``SVD(R_u R_v^{\top}) = \tilde U \Sigma \tilde V^{\top}``.
4. Truncate by ``eps`` or ``rank``, return
   :math:`U_{\text{out}} = Q_u \tilde U`, :math:`S_{\text{out}} = \Sigma`,
   :math:`V^{\top}_{\text{out}} = \tilde V^{\top} Q_v^{\top}`.

The whole pipeline operates on factor-sized matrices only — the dense
:math:`A` is never built.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matcomp.utils.functional_matrix import FloatArray
from matcomp.utils.linalg import truncate_svd
from matcomp.utils.low_rank import (
    CMRFactors,
    Factors,
    USVFactors,
    UVFactors,
)


@dataclass(frozen=True)
class RecompressResult:
    r"""Output of :func:`recompress_low_rank`.

    Attributes
    ----------
    factors
        Recompressed approximation as :class:`USVFactors`. Implements
        :class:`LowRankApprox` directly.
    original_rank
        Storage rank of the input carrier (``len(S)`` for ``USVFactors``,
        the column count of ``U`` for ``UVFactors``, the dimension of
        ``M`` for ``CMRFactors``).
    new_rank
        Storage rank after truncation.
    """

    factors: USVFactors
    original_rank: int
    new_rank: int

    # ---- LowRankApprox passthroughs ---------------------------------------

    @property
    def rank(self) -> int:
        """Alias for :attr:`new_rank`."""
        return int(self.new_rank)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the represented matrix."""
        return self.factors.shape

    def entry(self, i: int, j: int) -> float:
        """Return the recompressed entry :math:`\\hat A_{ij}`."""
        return self.factors.entry(i, j)

    def matvec(self, x: FloatArray) -> FloatArray:
        """Return :math:`\\hat A\\,x`."""
        return self.factors.matvec(x)

    def rmatvec(self, y: FloatArray) -> FloatArray:
        """Return :math:`\\hat A^{\\top} y`."""
        return self.factors.rmatvec(y)

    def reconstruct_small(self) -> FloatArray:
        """Materialise the recompressed approximation."""
        return self.factors.reconstruct_small()

    def factors_memory(self) -> int:
        """``nbytes`` of the recompressed factors."""
        return self.factors.factors_memory()


def _to_uv(factors: Factors) -> tuple[FloatArray, FloatArray, int]:
    """Reduce any factor format to ``(U, V)`` such that ``A ≈ U V^T``.

    The third return value is the original storage rank, used for
    reporting.
    """
    match factors:
        case UVFactors(U=U, V=V):
            return np.ascontiguousarray(U, dtype=np.float64), np.ascontiguousarray(V, dtype=np.float64), int(U.shape[1])
        case USVFactors(U=U, S=S, Vt=Vt):
            U_out = U.astype(np.float64, copy=False) * S
            V_out = Vt.T.astype(np.float64, copy=True)
            return np.ascontiguousarray(U_out), np.ascontiguousarray(V_out), int(S.size)
        case CMRFactors(C=C, M=M, R=R):
            # Fold M into the left factor; this is mathematically equal to
            # (C M) R = U V^T with U = C M, V = R^T.
            U_out = (C @ M).astype(np.float64, copy=True)
            V_out = R.T.astype(np.float64, copy=True)
            return np.ascontiguousarray(U_out), np.ascontiguousarray(V_out), int(M.shape[0])


def recompress_low_rank(
    factors: Factors,
    *,
    eps: float | None = None,
    rank: int | None = None,
) -> RecompressResult:
    r"""Truncate a low-rank decomposition to a smaller rank.

    Parameters
    ----------
    factors
        Input carrier. Any of :class:`UVFactors`, :class:`USVFactors`, or
        :class:`CMRFactors`.
    eps
        Relative tolerance — singular values below
        :math:`\epsilon \cdot \sigma_0` are dropped. Mutually exclusive
        with ``rank``.
    rank
        Fixed target rank.

    Returns
    -------
    RecompressResult
        Truncated factors in :class:`USVFactors` form plus original / new
        storage ranks.
    """
    if (eps is None) == (rank is None):
        raise ValueError("recompress_low_rank: pass exactly one of eps or rank")

    U_in, V_in, original_rank = _to_uv(factors)

    # Standard recompression recipe:
    #   QR(U), QR(V), SVD of the small middle term.
    Q_u, R_u = np.linalg.qr(U_in, mode="reduced")
    Q_v, R_v = np.linalg.qr(V_in, mode="reduced")
    middle = R_u @ R_v.T
    U_mid, S_mid, Vt_mid = np.linalg.svd(middle, full_matrices=False)

    U_t, S_t, Vt_t, new_rank = truncate_svd(U_mid, S_mid, Vt_mid, eps=eps, rank=rank)

    U_out = Q_u @ U_t
    Vt_out = Vt_t @ Q_v.T

    out_factors = USVFactors(
        U=np.ascontiguousarray(U_out, dtype=np.float64),
        S=np.ascontiguousarray(S_t, dtype=np.float64),
        Vt=np.ascontiguousarray(Vt_out, dtype=np.float64),
    )
    return RecompressResult(factors=out_factors, original_rank=original_rank, new_rank=int(new_rank))


__all__ = ["RecompressResult", "recompress_low_rank"]
