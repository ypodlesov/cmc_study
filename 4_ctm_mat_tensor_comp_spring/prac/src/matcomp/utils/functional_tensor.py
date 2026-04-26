"""The :class:`FunctionalTensor` abstraction.

A *functional tensor* is the d-dimensional generalisation of the
functional matrix used by Tasks 2/3/4: a tensor whose elements are
produced by a function rather than stored as a dense ``ndarray``. The
PDF (p. 2) defines the interface for the tensor side; Tasks 10 and 13
must work through it without ever materialising the full tensor.

Two layers are provided:

* :class:`FunctionalTensor` — a :class:`typing.Protocol` describing the
  public contract. Algorithms type-annotate their inputs as
  :class:`FunctionalTensor`.
* :class:`BaseFunctionalTensor` — an abstract base class providing
  reasonable default implementations of ``block``, ``fiber`` and
  ``samples`` in terms of ``entry``. Concrete classes inherit from it
  and override only what they can compute faster.

Concrete subclasses (CP / Tucker / Hilbert-3D / Gaussian-kernel-3D /
generic oracle) live in :mod:`matcomp.utils.tensor_test_objects`.
"""

from __future__ import annotations

import abc
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from matcomp.utils.functional_matrix import FloatArray, IntArray

IndexTuple = tuple[int, ...]

_DEFAULT_DTYPE: np.dtype[np.floating | np.complexfloating] = np.dtype(np.float64)


@runtime_checkable
class FunctionalTensor(Protocol):
    """Public protocol of a black-box d-mode tensor.

    Attributes
    ----------
    shape
        Tuple ``(n_1, ..., n_d)`` of mode sizes.
    ndim
        Number of modes (``len(shape)``).
    dtype
        NumPy dtype of the entries.

    Notes
    -----
    Four query methods are exposed:

    * ``entry(idx)`` — single element :math:`X[i_1, \\ldots, i_d]`.
    * ``block(indices)`` — :math:`X[\\mathrm{ix}\\_(I_1, \\ldots, I_d)]`.
    * ``fiber(mode, fixed_indices)`` — generalises matrix rows / columns.
      ``fixed_indices`` has length ``ndim`` with ``None`` in mode ``mode``.
    * ``samples(idx_array)`` — vectorised entry lookup. Input shape
      ``(N, d)``; output shape ``(N,)``. Used by Task 10 batched
      training and Task 13 cross.
    """

    shape: tuple[int, ...]
    ndim: int
    dtype: np.dtype[np.floating | np.complexfloating]

    def entry(self, idx: IndexTuple) -> float:
        """Return :math:`X[i_1, \\ldots, i_d]`."""
        ...

    def block(self, indices: tuple[npt.ArrayLike, ...]) -> FloatArray:
        """Return ``X[np.ix_(*indices)]``."""
        ...

    def fiber(self, mode: int, fixed_indices: tuple[int | None, ...]) -> FloatArray:
        """Return the mode-``mode`` fiber with the other modes fixed."""
        ...

    def samples(self, idx_array: IntArray) -> FloatArray:
        """Vectorised entry lookup.

        Parameters
        ----------
        idx_array
            Integer array of shape ``(N, ndim)``.

        Returns
        -------
        numpy.ndarray
            Length-``N`` array with ``out[k] = X[idx_array[k]]``.
        """
        ...


class BaseFunctionalTensor(abc.ABC):
    """Abstract base class with default fallbacks built from :meth:`entry`.

    Subclasses override :meth:`entry` at minimum; for any of ``block`` /
    ``fiber`` / ``samples`` they have a vectorised implementation, they
    should override that too — the default loops over ``entry`` and is
    only correct, not fast.

    Parameters
    ----------
    shape
        Tuple ``(n_1, ..., n_d)`` of mode sizes.
    dtype
        NumPy dtype of the entries. Defaults to :class:`numpy.float64`.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype[np.floating | np.complexfloating] = _DEFAULT_DTYPE,
    ) -> None:
        if not shape:
            raise ValueError("shape must have at least one mode")
        self.shape = tuple(int(s) for s in shape)
        self.ndim = len(self.shape)
        self.dtype = dtype

    @abc.abstractmethod
    def entry(self, idx: IndexTuple) -> float:  # pragma: no cover - abstract
        """Return :math:`X[i_1, \\ldots, i_d]`. Must be overridden."""
        raise NotImplementedError

    # ---- defaults ----------------------------------------------------------

    def block(self, indices: tuple[npt.ArrayLike, ...]) -> FloatArray:
        """Default: build the block by entry-wise calls."""
        if len(indices) != self.ndim:
            raise ValueError(f"expected {self.ndim} index arrays, got {len(indices)}")
        idx_arrays = [np.asarray(ix, dtype=np.intp) for ix in indices]
        out_shape = tuple(int(a.size) for a in idx_arrays)
        out = np.empty(out_shape, dtype=self.dtype)
        for flat in np.ndindex(*out_shape):
            tup = tuple(int(idx_arrays[k][flat[k]]) for k in range(self.ndim))
            out[flat] = self.entry(tup)
        return out

    def fiber(self, mode: int, fixed_indices: tuple[int | None, ...]) -> FloatArray:
        """Default: read the fiber entry-wise."""
        if not 0 <= mode < self.ndim:
            raise ValueError(f"mode {mode} out of range for ndim={self.ndim}")
        if len(fixed_indices) != self.ndim:
            raise ValueError(f"fixed_indices must have length ndim={self.ndim}")
        if fixed_indices[mode] is not None:
            raise ValueError(f"fixed_indices[{mode}] must be None for the fiber mode")
        n_mode = self.shape[mode]
        out = np.empty(n_mode, dtype=self.dtype)
        for k in range(n_mode):
            tup = tuple(
                int(k) if d == mode else int(fixed_indices[d])  # type: ignore[arg-type]
                for d in range(self.ndim)
            )
            out[k] = self.entry(tup)
        return out

    def samples(self, idx_array: IntArray) -> FloatArray:
        """Default: read each multi-index entry-wise."""
        idx = np.asarray(idx_array, dtype=np.intp)
        if idx.ndim != 2 or idx.shape[1] != self.ndim:
            raise ValueError(
                f"samples expected shape (N, {self.ndim}), got {idx.shape}"
            )
        out = np.empty(idx.shape[0], dtype=self.dtype)
        for k in range(idx.shape[0]):
            out[k] = self.entry(tuple(int(v) for v in idx[k]))
        return out


class DenseTensor(BaseFunctionalTensor):
    """Adapter exposing a dense ``ndarray`` as a :class:`FunctionalTensor`.

    Used as a baseline in tests and by Tasks 9 / 11 / 12 (which take
    dense ``ndarray`` directly) when they need to reuse oracle-mode code
    paths uniformly.

    Parameters
    ----------
    array
        The d-mode array to wrap. Data is referenced, not copied.
    """

    def __init__(self, array: FloatArray) -> None:
        if array.ndim < 1:
            raise ValueError(f"DenseTensor requires ndim >= 1, got {array.ndim}")
        super().__init__(shape=tuple(int(s) for s in array.shape), dtype=array.dtype)
        self._X = array

    def entry(self, idx: IndexTuple) -> float:
        return float(self._X[idx])

    def block(self, indices: tuple[npt.ArrayLike, ...]) -> FloatArray:
        if len(indices) != self.ndim:
            raise ValueError(f"expected {self.ndim} index arrays, got {len(indices)}")
        idx_arrays = [np.asarray(ix, dtype=np.intp) for ix in indices]
        return self._X[np.ix_(*idx_arrays)]

    def fiber(self, mode: int, fixed_indices: tuple[int | None, ...]) -> FloatArray:
        if not 0 <= mode < self.ndim:
            raise ValueError(f"mode {mode} out of range for ndim={self.ndim}")
        if len(fixed_indices) != self.ndim:
            raise ValueError(f"fixed_indices must have length ndim={self.ndim}")
        slicer = tuple(slice(None) if d == mode else int(fixed_indices[d])  # type: ignore[arg-type]
                       for d in range(self.ndim))
        return np.ascontiguousarray(self._X[slicer])

    def samples(self, idx_array: IntArray) -> FloatArray:
        idx = np.asarray(idx_array, dtype=np.intp)
        if idx.ndim != 2 or idx.shape[1] != self.ndim:
            raise ValueError(
                f"samples expected shape (N, {self.ndim}), got {idx.shape}"
            )
        return self._X[tuple(idx[:, k] for k in range(self.ndim))]


def evaluate_small(ft: FunctionalTensor, max_total: int = 1_000_000) -> FloatArray:
    """Materialise a functional tensor as a dense ``ndarray``.

    Used by ST-HOSVD and small reference paths. Refuses to materialise a
    tensor whose total number of elements exceeds ``max_total``, mirroring
    :func:`matcomp.utils.functional_matrix.evaluate_small` for matrices.

    Parameters
    ----------
    ft
        The functional tensor to materialise.
    max_total
        Maximum allowed value of ``prod(shape)``. Default 1e6 (≈8 MB at
        float64), which keeps small reference tests cheap.

    Returns
    -------
    numpy.ndarray
        Dense materialisation of the tensor.

    Raises
    ------
    ValueError
        If ``prod(ft.shape) > max_total``.
    """
    total = int(np.prod(ft.shape))
    if total > max_total:
        raise ValueError(
            f"refusing to materialise a tensor with {total} elements "
            f"(max_total={max_total})"
        )
    indices = tuple(np.arange(int(s), dtype=np.intp) for s in ft.shape)
    return ft.block(indices)


__all__ = [
    "BaseFunctionalTensor",
    "DenseTensor",
    "FunctionalTensor",
    "IndexTuple",
    "evaluate_small",
]
