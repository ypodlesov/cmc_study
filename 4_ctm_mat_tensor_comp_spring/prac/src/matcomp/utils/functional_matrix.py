"""The :class:`FunctionalMatrix` abstraction.

A *functional matrix* is one whose elements are produced by a function rather
than stored as a dense array. The practicum (PDF p. 2) defines the following
interface; tasks 2, 3, and 4 must work with this interface alone and never
materialise the full matrix.

Two layers are provided:

* :class:`FunctionalMatrix` — a :class:`typing.Protocol` describing the public
  contract. Algorithms type-annotate their inputs as
  :class:`FunctionalMatrix`.
* :class:`BaseFunctionalMatrix` — an abstract base class providing reasonable
  default implementations of the block products in terms of ``matvec`` and
  ``rmatvec``. Concrete classes inherit from it and override only what they
  can compute faster.

Concrete subclasses (Hilbert, Cauchy, Gaussian-kernel, exact low-rank, etc.)
live in :mod:`matcomp.utils.test_matrices`.
"""

from __future__ import annotations

import abc
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

# Public type aliases: every NDArray we pass around is float64 unless
# explicitly upcast. Index arrays use the integer alias.
FloatArray = npt.NDArray[np.floating]
ComplexArray = npt.NDArray[np.complexfloating]
IntArray = npt.NDArray[np.integer]

# Module-level dtype singleton used as the default for BaseFunctionalMatrix.
_DEFAULT_DTYPE: np.dtype[np.floating | np.complexfloating] = np.dtype(np.float64)


@runtime_checkable
class FunctionalMatrix(Protocol):
    """Public protocol of a black-box matrix.

    Attributes
    ----------
    shape
        ``(m, n)`` size of the conceptual matrix.
    dtype
        NumPy dtype of the entries (``np.float64`` or ``np.complex128``).

    Notes
    -----
    The protocol intentionally exposes both single-vector products
    (``matvec`` / ``rmatvec``) and block products (``matmat`` / ``rmatmat``).
    Block products are essential for randomized SVD: forming
    :math:`Y = A \\Omega` for a wide :math:`\\Omega` of shape
    :math:`(n, r + p)` should be one call, not :math:`r + p` per-column
    calls.
    """

    shape: tuple[int, int]
    dtype: np.dtype[np.floating | np.complexfloating]

    def entry(self, i: int, j: int) -> float:
        """Return :math:`A_{ij}`."""
        ...

    def block(
        self, indices_row: npt.ArrayLike, indices_col: npt.ArrayLike
    ) -> FloatArray:
        """Return ``A[np.ix_(indices_row, indices_col)]``."""
        ...

    def row(self, i: int, indices_col: npt.ArrayLike | None = None) -> FloatArray:
        """Return ``A[i, :]`` or ``A[i, indices_col]``."""
        ...

    def col(self, indices_row: npt.ArrayLike | None, j: int) -> FloatArray:
        """Return ``A[:, j]`` or ``A[indices_row, j]``."""
        ...

    def matvec(self, x: FloatArray) -> FloatArray:
        """Return :math:`A\\,x`."""
        ...

    def rmatvec(self, y: FloatArray) -> FloatArray:
        """Return :math:`A^{\\top} y`."""
        ...

    def matmat(self, x_block: FloatArray) -> FloatArray:
        """Return :math:`A\\,X` for a 2-D ``X``."""
        ...

    def rmatmat(self, y_block: FloatArray) -> FloatArray:
        """Return :math:`A^{\\top} Y` for a 2-D ``Y``."""
        ...


class BaseFunctionalMatrix(abc.ABC):
    """Abstract base class with default block-product fallbacks.

    Subclasses implement at minimum :meth:`entry`. Default implementations
    of the other six methods are provided in terms of ``entry``; they are
    correct but generally slow. Concrete classes typically override
    :meth:`row`, :meth:`col`, :meth:`block`, :meth:`matvec` and
    :meth:`matmat` with vectorised forms.

    Parameters
    ----------
    shape
        Tuple ``(m, n)``.
    dtype
        NumPy dtype of the entries. Defaults to :class:`numpy.float64`.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        dtype: np.dtype[np.floating | np.complexfloating] = _DEFAULT_DTYPE,
    ) -> None:
        self.shape = shape
        self.dtype = dtype

    @abc.abstractmethod
    def entry(self, i: int, j: int) -> float:  # pragma: no cover - abstract
        """Return :math:`A_{ij}`. Must be overridden."""
        raise NotImplementedError

    # ---- defaults ----------------------------------------------------------

    def block(
        self, indices_row: npt.ArrayLike, indices_col: npt.ArrayLike
    ) -> FloatArray:
        """Default: build the block by entry-wise calls."""
        rows = np.asarray(indices_row, dtype=np.intp)
        cols = np.asarray(indices_col, dtype=np.intp)
        out = np.empty((rows.size, cols.size), dtype=self.dtype)
        for a, i in enumerate(rows):
            for b, j in enumerate(cols):
                out[a, b] = self.entry(int(i), int(j))
        return out

    def row(self, i: int, indices_col: npt.ArrayLike | None = None) -> FloatArray:
        """Default: build a row by entry-wise calls."""
        if indices_col is None:
            cols = np.arange(self.shape[1], dtype=np.intp)
        else:
            cols = np.asarray(indices_col, dtype=np.intp)
        out = np.empty(cols.size, dtype=self.dtype)
        for b, j in enumerate(cols):
            out[b] = self.entry(int(i), int(j))
        return out

    def col(self, indices_row: npt.ArrayLike | None, j: int) -> FloatArray:
        """Default: build a column by entry-wise calls."""
        if indices_row is None:
            rows = np.arange(self.shape[0], dtype=np.intp)
        else:
            rows = np.asarray(indices_row, dtype=np.intp)
        out = np.empty(rows.size, dtype=self.dtype)
        for a, i in enumerate(rows):
            out[a] = self.entry(int(i), int(j))
        return out

    def matvec(self, x: FloatArray) -> FloatArray:
        r"""Default: :math:`A\,x` by computing every column on demand."""
        if x.shape != (self.shape[1],):
            raise ValueError(
                f"matvec expected x of shape ({self.shape[1]},), got {x.shape}"
            )
        out = np.zeros(self.shape[0], dtype=self.dtype)
        for j in range(self.shape[1]):
            out += self.col(None, j) * x[j]
        return out

    def rmatvec(self, y: FloatArray) -> FloatArray:
        r"""Default: :math:`A^{\top} y` by computing every row on demand."""
        if y.shape != (self.shape[0],):
            raise ValueError(
                f"rmatvec expected y of shape ({self.shape[0]},), got {y.shape}"
            )
        out = np.zeros(self.shape[1], dtype=self.dtype)
        for i in range(self.shape[0]):
            out += self.row(i) * y[i]
        return out

    def matmat(self, x_block: FloatArray) -> FloatArray:
        r"""Default: column-by-column reduction to :meth:`matvec`."""
        if x_block.ndim != 2 or x_block.shape[0] != self.shape[1]:
            raise ValueError(
                f"matmat expected X of shape ({self.shape[1]}, k), got {x_block.shape}"
            )
        cols = [self.matvec(np.ascontiguousarray(x_block[:, k])) for k in range(x_block.shape[1])]
        return np.column_stack(cols)

    def rmatmat(self, y_block: FloatArray) -> FloatArray:
        r"""Default: column-by-column reduction to :meth:`rmatvec`."""
        if y_block.ndim != 2 or y_block.shape[0] != self.shape[0]:
            raise ValueError(
                f"rmatmat expected Y of shape ({self.shape[0]}, k), got {y_block.shape}"
            )
        cols = [self.rmatvec(np.ascontiguousarray(y_block[:, k])) for k in range(y_block.shape[1])]
        return np.column_stack(cols)


def evaluate_small(fm: FunctionalMatrix, max_size: int = 1000) -> FloatArray:
    """Materialise a functional matrix as a dense ``ndarray``.

    Used by reference SVD checks and by Task 7's benchmark harness. Refuses to
    materialise matrices larger than ``max_size`` × ``max_size`` to keep
    accidental misuse cheap.

    Parameters
    ----------
    fm
        The functional matrix to materialise.
    max_size
        Maximum allowed value of ``max(m, n)``. Defaults to 1000 (PDF p. 2:
        "the full A may be built only for ``m, n`` not exceeding a fixed
        threshold, for example 1000").

    Returns
    -------
    numpy.ndarray
        Dense materialisation of the matrix.

    Raises
    ------
    ValueError
        If ``max(fm.shape) > max_size``.
    """
    m, n = fm.shape
    if max(m, n) > max_size:
        raise ValueError(
            f"refusing to materialise a {m}x{n} matrix; max_size={max_size}"
        )
    return fm.block(np.arange(m, dtype=np.intp), np.arange(n, dtype=np.intp))
