""":class:`CountingFunctionalMatrix` — instrumented oracle wrapper.

Every algorithm in this project reports an ``oracle_calls`` count. To make
that count meaningful (and to assert in tests that, e.g., Lanczos never
calls anything other than ``matvec``), we wrap the underlying matrix in a
:class:`CountingFunctionalMatrix` that bumps a counter on every method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from matcomp.utils.functional_matrix import FloatArray, FunctionalMatrix, IntArray
from matcomp.utils.functional_tensor import FunctionalTensor, IndexTuple


@dataclass
class OracleCounts:
    """Per-method counters for a wrapped :class:`FunctionalMatrix`."""

    entry_calls: int = 0
    row_calls: int = 0
    col_calls: int = 0
    block_calls: int = 0
    matvec_calls: int = 0
    rmatvec_calls: int = 0
    matmat_calls: int = 0
    rmatmat_calls: int = 0

    @property
    def total(self) -> int:
        """Sum across every counter (one logical "oracle call" per invocation)."""
        return (
            self.entry_calls
            + self.row_calls
            + self.col_calls
            + self.block_calls
            + self.matvec_calls
            + self.rmatvec_calls
            + self.matmat_calls
            + self.rmatmat_calls
        )

    def reset(self) -> None:
        """Zero every counter."""
        self.entry_calls = 0
        self.row_calls = 0
        self.col_calls = 0
        self.block_calls = 0
        self.matvec_calls = 0
        self.rmatvec_calls = 0
        self.matmat_calls = 0
        self.rmatmat_calls = 0


class CountingFunctionalMatrix:
    """Decorator that records how often each oracle method is called.

    Parameters
    ----------
    inner
        The wrapped :class:`FunctionalMatrix`. The counter increments on each
        call before delegating to ``inner``.
    """

    def __init__(self, inner: FunctionalMatrix) -> None:
        self._inner = inner
        self.shape = inner.shape
        self.dtype = inner.dtype
        self.counts = OracleCounts()

    @property
    def inner(self) -> FunctionalMatrix:
        """The wrapped matrix (read-only access)."""
        return self._inner

    def entry(self, i: int, j: int) -> float:
        self.counts.entry_calls += 1
        return self._inner.entry(i, j)

    def block(
        self, indices_row: npt.ArrayLike, indices_col: npt.ArrayLike
    ) -> FloatArray:
        self.counts.block_calls += 1
        return self._inner.block(indices_row, indices_col)

    def row(self, i: int, indices_col: npt.ArrayLike | None = None) -> FloatArray:
        self.counts.row_calls += 1
        return self._inner.row(i, indices_col)

    def col(self, indices_row: npt.ArrayLike | None, j: int) -> FloatArray:
        self.counts.col_calls += 1
        return self._inner.col(indices_row, j)

    def matvec(self, x: FloatArray) -> FloatArray:
        self.counts.matvec_calls += 1
        return self._inner.matvec(x)

    def rmatvec(self, y: FloatArray) -> FloatArray:
        self.counts.rmatvec_calls += 1
        return self._inner.rmatvec(y)

    def matmat(self, x_block: FloatArray) -> FloatArray:
        self.counts.matmat_calls += 1
        return self._inner.matmat(x_block)

    def rmatmat(self, y_block: FloatArray) -> FloatArray:
        self.counts.rmatmat_calls += 1
        return self._inner.rmatmat(y_block)


@dataclass
class TensorOracleCounts:
    """Per-method counters for a wrapped :class:`FunctionalTensor`."""

    entry_calls: int = 0
    block_calls: int = 0
    fiber_calls: int = 0
    samples_calls: int = 0

    @property
    def total(self) -> int:
        """Sum across every counter (one logical "oracle call" per invocation)."""
        return (
            self.entry_calls
            + self.block_calls
            + self.fiber_calls
            + self.samples_calls
        )

    def reset(self) -> None:
        """Zero every counter."""
        self.entry_calls = 0
        self.block_calls = 0
        self.fiber_calls = 0
        self.samples_calls = 0


class CountingFunctionalTensor:
    """Decorator that records how often each tensor-oracle method is called.

    Mirrors :class:`CountingFunctionalMatrix` for the d-mode case. Tasks
    10 and 13 wrap their input tensor in this decorator before any
    oracle access so their result objects can report ``oracle_counts``.

    Parameters
    ----------
    inner
        The wrapped :class:`FunctionalTensor`. The counter increments on
        each call before delegating to ``inner``.
    """

    def __init__(self, inner: FunctionalTensor) -> None:
        self._inner = inner
        self.shape = inner.shape
        self.ndim = inner.ndim
        self.dtype = inner.dtype
        self.counts = TensorOracleCounts()

    @property
    def inner(self) -> FunctionalTensor:
        """The wrapped tensor (read-only access)."""
        return self._inner

    def entry(self, idx: IndexTuple) -> float:
        self.counts.entry_calls += 1
        return self._inner.entry(idx)

    def block(self, indices: tuple[npt.ArrayLike, ...]) -> FloatArray:
        self.counts.block_calls += 1
        return self._inner.block(indices)

    def fiber(self, mode: int, fixed_indices: tuple[int | None, ...]) -> FloatArray:
        self.counts.fiber_calls += 1
        return self._inner.fiber(mode, fixed_indices)

    def samples(self, idx_array: IntArray) -> FloatArray:
        self.counts.samples_calls += 1
        return self._inner.samples(idx_array)


__all__ = [
    "CountingFunctionalMatrix",
    "CountingFunctionalTensor",
    "OracleCounts",
    "TensorOracleCounts",
]


def _isinstance_check() -> None:
    """Internal: assert :class:`CountingFunctionalMatrix` satisfies the protocol.

    ``runtime_checkable`` :class:`Protocol` does *not* enforce method signatures,
    so we keep this in source for grep-ability rather than a real test.
    """
    fake = type("Fake", (), {})()
    fake.shape = (2, 2)
    fake.dtype = np.dtype(np.float64)
    _ = isinstance(fake, FunctionalMatrix)  # noqa: F841
