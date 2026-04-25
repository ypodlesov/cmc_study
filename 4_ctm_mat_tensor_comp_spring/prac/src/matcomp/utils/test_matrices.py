r"""Concrete :class:`FunctionalMatrix` implementations used as test inputs.

The PDF (p. 2) requires every algorithm to be exercised on a minimum set of
test matrices: an exact low-rank matrix, a Hilbert matrix, a Cauchy / kernel
matrix and a low-rank matrix with added noise. This module provides those
classes plus a dense wrapper that adapts an existing ``ndarray`` to the
:class:`FunctionalMatrix` protocol.

Every concrete class overrides :meth:`block` and :meth:`matmat` with a
vectorised implementation, so block products in randomised SVD (Task 3) cost
one NumPy call rather than per-column loops.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from matcomp.utils.functional_matrix import (
    BaseFunctionalMatrix,
    FloatArray,
)
from matcomp.utils.seeding import make_rng


class DenseMatrix(BaseFunctionalMatrix):
    """Adapter that exposes a dense ``ndarray`` as a :class:`FunctionalMatrix`.

    Used as a baseline in tests, by Task 6 (which is dense by definition) and
    by Task 7 to wrap pre-materialised matrices uniformly.

    Parameters
    ----------
    array
        The 2-D array to wrap. The data is referenced, not copied.
    """

    def __init__(self, array: FloatArray) -> None:
        if array.ndim != 2:
            raise ValueError(f"DenseMatrix expected 2-D array, got ndim={array.ndim}")
        super().__init__(shape=(int(array.shape[0]), int(array.shape[1])), dtype=array.dtype)
        self._A = array

    def entry(self, i: int, j: int) -> float:
        return float(self._A[i, j])

    def block(
        self, indices_row: npt.ArrayLike, indices_col: npt.ArrayLike
    ) -> FloatArray:
        rows = np.asarray(indices_row, dtype=np.intp)
        cols = np.asarray(indices_col, dtype=np.intp)
        return self._A[np.ix_(rows, cols)]

    def row(self, i: int, indices_col: npt.ArrayLike | None = None) -> FloatArray:
        if indices_col is None:
            return self._A[i].copy()
        cols = np.asarray(indices_col, dtype=np.intp)
        return self._A[i, cols]

    def col(self, indices_row: npt.ArrayLike | None, j: int) -> FloatArray:
        if indices_row is None:
            return self._A[:, j].copy()
        rows = np.asarray(indices_row, dtype=np.intp)
        return self._A[rows, j]

    def matvec(self, x: FloatArray) -> FloatArray:
        return self._A @ x

    def rmatvec(self, y: FloatArray) -> FloatArray:
        return self._A.T @ y

    def matmat(self, x_block: FloatArray) -> FloatArray:
        return self._A @ x_block

    def rmatmat(self, y_block: FloatArray) -> FloatArray:
        return self._A.T @ y_block


class HilbertMatrix(BaseFunctionalMatrix):
    r"""The Hilbert matrix :math:`A_{ij} = 1 / (i + j + 1)`.

    Used as the canonical slow-singular-value-decay test in Tasks 2, 3 and 7.

    Parameters
    ----------
    m
        Number of rows.
    n
        Number of columns. Defaults to ``m`` (square).
    """

    def __init__(self, m: int, n: int | None = None) -> None:
        if n is None:
            n = m
        super().__init__(shape=(m, n), dtype=np.dtype(np.float64))

    def entry(self, i: int, j: int) -> float:
        return 1.0 / (i + j + 1)

    def block(
        self, indices_row: npt.ArrayLike, indices_col: npt.ArrayLike
    ) -> FloatArray:
        rows = np.asarray(indices_row, dtype=np.intp)
        cols = np.asarray(indices_col, dtype=np.intp)
        return 1.0 / (np.add.outer(rows, cols) + 1.0)

    def row(self, i: int, indices_col: npt.ArrayLike | None = None) -> FloatArray:
        cols = (
            np.arange(self.shape[1], dtype=np.intp)
            if indices_col is None
            else np.asarray(indices_col, dtype=np.intp)
        )
        return 1.0 / (i + cols + 1.0)

    def col(self, indices_row: npt.ArrayLike | None, j: int) -> FloatArray:
        rows = (
            np.arange(self.shape[0], dtype=np.intp)
            if indices_row is None
            else np.asarray(indices_row, dtype=np.intp)
        )
        return 1.0 / (rows + j + 1.0)


class CauchyMatrix(BaseFunctionalMatrix):
    r"""Cauchy matrix :math:`A_{ij} = 1 / (x_i + y_j)`.

    Mildly faster decay than Hilbert. Used in Task 2's error-vs-rank curves.

    Parameters
    ----------
    x
        Vector of node coordinates of length ``m``. ``x_i + y_j`` must be
        non-zero for all ``i, j``.
    y
        Vector of node coordinates of length ``n``.
    """

    def __init__(self, x: FloatArray, y: FloatArray) -> None:
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("CauchyMatrix requires 1-D x and y vectors")
        super().__init__(shape=(int(x.size), int(y.size)), dtype=np.dtype(np.float64))
        self._x = np.ascontiguousarray(x, dtype=np.float64)
        self._y = np.ascontiguousarray(y, dtype=np.float64)

    def entry(self, i: int, j: int) -> float:
        return float(1.0 / (self._x[i] + self._y[j]))

    def block(
        self, indices_row: npt.ArrayLike, indices_col: npt.ArrayLike
    ) -> FloatArray:
        rows = np.asarray(indices_row, dtype=np.intp)
        cols = np.asarray(indices_col, dtype=np.intp)
        return 1.0 / (np.add.outer(self._x[rows], self._y[cols]))

    def row(self, i: int, indices_col: npt.ArrayLike | None = None) -> FloatArray:
        cols = (
            np.arange(self.shape[1], dtype=np.intp)
            if indices_col is None
            else np.asarray(indices_col, dtype=np.intp)
        )
        return 1.0 / (self._x[i] + self._y[cols])

    def col(self, indices_row: npt.ArrayLike | None, j: int) -> FloatArray:
        rows = (
            np.arange(self.shape[0], dtype=np.intp)
            if indices_row is None
            else np.asarray(indices_row, dtype=np.intp)
        )
        return 1.0 / (self._x[rows] + self._y[j])


class GaussianKernelMatrix(BaseFunctionalMatrix):
    r"""Gaussian kernel matrix :math:`A_{ij} = \exp(-\lVert x_i - y_j\rVert^2 / \sigma^2)`.

    Has rapid singular-value decay and is the natural target for randomised
    SVD (Task 3). Inputs ``x`` and ``y`` may have ``d > 1`` columns; the
    pair-wise distance is computed in a vectorised manner.

    Parameters
    ----------
    x
        Source nodes, shape :math:`(m,)` or :math:`(m, d)`.
    y
        Target nodes, shape :math:`(n,)` or :math:`(n, d)`.
    sigma
        Bandwidth :math:`\sigma > 0`. Larger values yield smoother / lower
        rank.
    """

    def __init__(self, x: FloatArray, y: FloatArray, sigma: float) -> None:
        if sigma <= 0.0:
            raise ValueError("GaussianKernelMatrix requires sigma > 0")
        x_2d = np.ascontiguousarray(np.atleast_2d(x.T).T if x.ndim == 1 else x, dtype=np.float64)
        y_2d = np.ascontiguousarray(np.atleast_2d(y.T).T if y.ndim == 1 else y, dtype=np.float64)
        if x_2d.ndim != 2 or y_2d.ndim != 2:
            raise ValueError("x and y must be 1-D or 2-D")
        if x_2d.shape[1] != y_2d.shape[1]:
            raise ValueError(
                f"x and y dimension mismatch: {x_2d.shape[1]} vs {y_2d.shape[1]}"
            )
        super().__init__(
            shape=(int(x_2d.shape[0]), int(y_2d.shape[0])),
            dtype=np.dtype(np.float64),
        )
        self._x = x_2d
        self._y = y_2d
        self._sigma = float(sigma)

    @staticmethod
    def _sq_dists(a: FloatArray, b: FloatArray) -> FloatArray:
        # ||a_i - b_j||^2 = ||a_i||^2 + ||b_j||^2 - 2 a_i . b_j
        a_sq = np.einsum("ij,ij->i", a, a)
        b_sq = np.einsum("ij,ij->i", b, b)
        return np.maximum(a_sq[:, None] + b_sq[None, :] - 2.0 * (a @ b.T), 0.0)

    def entry(self, i: int, j: int) -> float:
        diff = self._x[i] - self._y[j]
        return float(np.exp(-float(diff @ diff) / (self._sigma * self._sigma)))

    def block(
        self, indices_row: npt.ArrayLike, indices_col: npt.ArrayLike
    ) -> FloatArray:
        rows = np.asarray(indices_row, dtype=np.intp)
        cols = np.asarray(indices_col, dtype=np.intp)
        d2 = self._sq_dists(self._x[rows], self._y[cols])
        return np.exp(-d2 / (self._sigma * self._sigma))

    def row(self, i: int, indices_col: npt.ArrayLike | None = None) -> FloatArray:
        cols = (
            np.arange(self.shape[1], dtype=np.intp)
            if indices_col is None
            else np.asarray(indices_col, dtype=np.intp)
        )
        return self.block(np.array([i]), cols)[0]

    def col(self, indices_row: npt.ArrayLike | None, j: int) -> FloatArray:
        rows = (
            np.arange(self.shape[0], dtype=np.intp)
            if indices_row is None
            else np.asarray(indices_row, dtype=np.intp)
        )
        return self.block(rows, np.array([j]))[:, 0]


class LowRankMatrix(BaseFunctionalMatrix):
    r"""Exact low-rank matrix :math:`A = U V^{\top}`.

    Used to verify *recovery* of an algorithm: a method that claims to
    produce a rank-``r`` approximation of an exact rank-``r`` matrix should
    return zero error (up to floating-point tolerance).

    Parameters
    ----------
    U
        Left factor of shape :math:`(m, r)`.
    V
        Right factor of shape :math:`(n, r)`. Note: ``V``, not ``V.T``.
    """

    def __init__(self, U: FloatArray, V: FloatArray) -> None:
        if U.ndim != 2 or V.ndim != 2:
            raise ValueError("LowRankMatrix expects 2-D U and V")
        if U.shape[1] != V.shape[1]:
            raise ValueError(
                f"LowRankMatrix: U has {U.shape[1]} cols, V has {V.shape[1]} cols"
            )
        super().__init__(shape=(int(U.shape[0]), int(V.shape[0])), dtype=np.dtype(np.float64))
        self._U = np.ascontiguousarray(U, dtype=np.float64)
        self._V = np.ascontiguousarray(V, dtype=np.float64)

    @property
    def rank(self) -> int:
        """Exact rank of the underlying factorisation."""
        return int(self._U.shape[1])

    def entry(self, i: int, j: int) -> float:
        return float(self._U[i] @ self._V[j])

    def block(
        self, indices_row: npt.ArrayLike, indices_col: npt.ArrayLike
    ) -> FloatArray:
        rows = np.asarray(indices_row, dtype=np.intp)
        cols = np.asarray(indices_col, dtype=np.intp)
        return self._U[rows] @ self._V[cols].T

    def row(self, i: int, indices_col: npt.ArrayLike | None = None) -> FloatArray:
        if indices_col is None:
            return self._U[i] @ self._V.T
        cols = np.asarray(indices_col, dtype=np.intp)
        return self._U[i] @ self._V[cols].T

    def col(self, indices_row: npt.ArrayLike | None, j: int) -> FloatArray:
        if indices_row is None:
            return self._U @ self._V[j]
        rows = np.asarray(indices_row, dtype=np.intp)
        return self._U[rows] @ self._V[j]

    def matvec(self, x: FloatArray) -> FloatArray:
        return self._U @ (self._V.T @ x)

    def rmatvec(self, y: FloatArray) -> FloatArray:
        return self._V @ (self._U.T @ y)

    def matmat(self, x_block: FloatArray) -> FloatArray:
        return self._U @ (self._V.T @ x_block)

    def rmatmat(self, y_block: FloatArray) -> FloatArray:
        return self._V @ (self._U.T @ y_block)


class LowRankPlusNoise(BaseFunctionalMatrix):
    r"""Low-rank matrix plus Gaussian noise: :math:`A = U V^{\top} + \sigma E`.

    The noise sample is fixed at construction so calls to :meth:`entry` are
    deterministic.

    Parameters
    ----------
    U, V
        Factors of the noise-free signal.
    noise_level
        Standard deviation of the Gaussian noise.
    random_seed
        Seed for the noise sample. ``None`` uses fresh entropy.
    """

    def __init__(
        self,
        U: FloatArray,
        V: FloatArray,
        noise_level: float,
        random_seed: int | None = None,
    ) -> None:
        if U.ndim != 2 or V.ndim != 2 or U.shape[1] != V.shape[1]:
            raise ValueError("LowRankPlusNoise: shape mismatch in U / V")
        if noise_level < 0.0:
            raise ValueError("noise_level must be non-negative")
        super().__init__(shape=(int(U.shape[0]), int(V.shape[0])), dtype=np.dtype(np.float64))
        rng = make_rng(random_seed)
        self._signal = LowRankMatrix(U, V)
        self._noise = noise_level * rng.standard_normal(self.shape).astype(np.float64)

    def entry(self, i: int, j: int) -> float:
        return self._signal.entry(i, j) + float(self._noise[i, j])

    def block(
        self, indices_row: npt.ArrayLike, indices_col: npt.ArrayLike
    ) -> FloatArray:
        rows = np.asarray(indices_row, dtype=np.intp)
        cols = np.asarray(indices_col, dtype=np.intp)
        return self._signal.block(rows, cols) + self._noise[np.ix_(rows, cols)]

    def row(self, i: int, indices_col: npt.ArrayLike | None = None) -> FloatArray:
        if indices_col is None:
            return self._signal.row(i) + self._noise[i]
        cols = np.asarray(indices_col, dtype=np.intp)
        return self._signal.row(i, cols) + self._noise[i, cols]

    def col(self, indices_row: npt.ArrayLike | None, j: int) -> FloatArray:
        if indices_row is None:
            return self._signal.col(None, j) + self._noise[:, j]
        rows = np.asarray(indices_row, dtype=np.intp)
        return self._signal.col(rows, j) + self._noise[rows, j]


__all__ = [
    "CauchyMatrix",
    "DenseMatrix",
    "GaussianKernelMatrix",
    "HilbertMatrix",
    "LowRankMatrix",
    "LowRankPlusNoise",
]
