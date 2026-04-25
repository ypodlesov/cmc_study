r"""Low-rank approximation carriers and the unifying protocol.

Every matrix algorithm in this project produces a low-rank object. To make
Task 7's benchmark harness (and Task 5's recompression) consume them
uniformly, we expose:

* :class:`LowRankApprox` — a :class:`typing.Protocol` describing the public
  surface of any rank-bounded approximation: ``rank``, ``shape``,
  ``entry(i, j)``, ``matvec``, ``rmatvec``, ``reconstruct_small`` and
  ``factors_memory``.
* :class:`UVFactors`, :class:`USVFactors`, :class:`CMRFactors` — three small
  frozen dataclasses representing the three factor formats used in the
  practicum:

  * :math:`A \approx U V^{\top}` (ACA / outer-product format),
  * :math:`A \approx U \,\mathrm{diag}(S)\, V^{\top}` (SVD-style),
  * :math:`A \approx C M R` with :math:`M = A[I, J]^{-1}` (cross / skeleton).

  The tagged union :data:`Factors` is what :func:`recompress_low_rank`
  consumes; mypy ``--strict`` checks every ``match`` is exhaustive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.floating]


@runtime_checkable
class LowRankApprox(Protocol):
    """Protocol for any rank-bounded approximation of a matrix."""

    rank: int
    shape: tuple[int, int]

    def entry(self, i: int, j: int) -> float:
        """Return :math:`\\hat A_{ij}`."""
        ...

    def matvec(self, x: FloatArray) -> FloatArray:
        """Return :math:`\\hat A\\,x`."""
        ...

    def rmatvec(self, y: FloatArray) -> FloatArray:
        """Return :math:`\\hat A^{\\top} y`."""
        ...

    def reconstruct_small(self) -> FloatArray:
        """Return the dense materialisation of :math:`\\hat A`."""
        ...

    def factors_memory(self) -> int:
        """Return the total ``nbytes`` over all stored factor arrays."""
        ...


@dataclass(frozen=True)
class UVFactors:
    r"""Outer-product format :math:`A \approx U V^{\top}`.

    Parameters
    ----------
    U
        Left factor of shape :math:`(m, r)`.
    V
        Right factor of shape :math:`(n, r)`. Note: ``V``, not ``V.T``.
    """

    U: FloatArray
    V: FloatArray

    @property
    def rank(self) -> int:
        """Numerical rank of the carrier (number of stored columns)."""
        return int(self.U.shape[1])

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the represented matrix."""
        return (int(self.U.shape[0]), int(self.V.shape[0]))

    def entry(self, i: int, j: int) -> float:
        """Return :math:`\\sum_k U_{ik} V_{jk}`."""
        return float(self.U[i] @ self.V[j])

    def matvec(self, x: FloatArray) -> FloatArray:
        r""":math:`U\,(V^{\top} x)`."""
        return self.U @ (self.V.T @ x)

    def rmatvec(self, y: FloatArray) -> FloatArray:
        r""":math:`V\,(U^{\top} y)`."""
        return self.V @ (self.U.T @ y)

    def reconstruct_small(self) -> FloatArray:
        r""":math:`U V^{\top}` as a dense matrix."""
        return self.U @ self.V.T

    def factors_memory(self) -> int:
        """Sum of ``nbytes`` over ``U`` and ``V``."""
        return int(self.U.nbytes + self.V.nbytes)


@dataclass(frozen=True)
class USVFactors:
    r"""SVD-style format :math:`A \approx U \,\mathrm{diag}(S)\, V^{\top}`.

    Parameters
    ----------
    U
        Left singular vectors, shape :math:`(m, r)`.
    S
        Singular values, shape :math:`(r,)`.
    Vt
        ``V.T`` — the *transposed* right singular vectors, shape :math:`(r, n)`.
    """

    U: FloatArray
    S: FloatArray
    Vt: FloatArray

    @property
    def rank(self) -> int:
        """Number of stored singular values."""
        return int(self.S.size)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the represented matrix."""
        return (int(self.U.shape[0]), int(self.Vt.shape[1]))

    def entry(self, i: int, j: int) -> float:
        r""":math:`\sum_k U_{ik} S_k V^{\top}_{kj}`."""
        return float(self.U[i] @ (self.S * self.Vt[:, j]))

    def matvec(self, x: FloatArray) -> FloatArray:
        r""":math:`U\,\mathrm{diag}(S)\,(V^{\top} x)`."""
        return self.U @ (self.S * (self.Vt @ x))

    def rmatvec(self, y: FloatArray) -> FloatArray:
        r""":math:`V\,\mathrm{diag}(S)\,(U^{\top} y)`."""
        return self.Vt.T @ (self.S * (self.U.T @ y))

    def reconstruct_small(self) -> FloatArray:
        r""":math:`U \mathrm{diag}(S) V^{\top}` as a dense matrix."""
        return (self.U * self.S) @ self.Vt

    def factors_memory(self) -> int:
        """Sum of ``nbytes`` over ``U``, ``S`` and ``Vt``."""
        return int(self.U.nbytes + self.S.nbytes + self.Vt.nbytes)


@dataclass(frozen=True)
class CMRFactors:
    r"""Cross / skeleton format :math:`A \approx C\,M\,R`.

    Parameters
    ----------
    C
        Selected columns of shape :math:`(m, r)` (i.e. ``A[:, J]``).
    M
        Inverse of the pivot block, shape :math:`(r, r)` (the practical form
        is :math:`\mathrm{pinv}(A[I, J])` to handle ill-conditioning).
    R
        Selected rows of shape :math:`(r, n)` (i.e. ``A[I, :]``).
    """

    C: FloatArray
    M: FloatArray
    R: FloatArray

    @property
    def rank(self) -> int:
        """Number of selected pivots."""
        return int(self.M.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the represented matrix."""
        return (int(self.C.shape[0]), int(self.R.shape[1]))

    def entry(self, i: int, j: int) -> float:
        r""":math:`C_{i,:}\,M\,R_{:,j}`."""
        return float(self.C[i] @ self.M @ self.R[:, j])

    def matvec(self, x: FloatArray) -> FloatArray:
        r""":math:`C\,(M\,(R\,x))`."""
        return self.C @ (self.M @ (self.R @ x))

    def rmatvec(self, y: FloatArray) -> FloatArray:
        r""":math:`R^{\top}\,(M^{\top}\,(C^{\top} y))`."""
        return self.R.T @ (self.M.T @ (self.C.T @ y))

    def reconstruct_small(self) -> FloatArray:
        r""":math:`C M R` as a dense matrix."""
        return self.C @ self.M @ self.R

    def factors_memory(self) -> int:
        """Sum of ``nbytes`` over ``C``, ``M`` and ``R``."""
        return int(self.C.nbytes + self.M.nbytes + self.R.nbytes)


# Tagged union used by Task 5 (recompression) and as the canonical low-rank
# carrier across Tasks 2/3/4.
Factors = UVFactors | USVFactors | CMRFactors


__all__ = [
    "CMRFactors",
    "Factors",
    "FloatArray",
    "LowRankApprox",
    "USVFactors",
    "UVFactors",
]
