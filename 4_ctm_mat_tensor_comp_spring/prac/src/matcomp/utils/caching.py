""":class:`CachedFunctionalMatrix` — memoising oracle wrapper.

Adaptive cross approximation (Task 4) repeatedly queries the same rows,
columns and entries while building a low-rank decomposition. To avoid
re-executing the underlying oracle, this module wraps a
:class:`FunctionalMatrix` with caches and tracks ``cache_hits`` /
``cache_misses``.

Composition convention used in the project::

    raw = HilbertMatrix(...)
    counted = CountingFunctionalMatrix(raw)
    cached = CachedFunctionalMatrix(counted)

In that order, a cache hit does *not* increase the real oracle counter,
which satisfies the practicum's "Repeated requests must not increase the
counter of real computations" acceptance criterion (PDF p. 5).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from matcomp.utils.functional_matrix import FloatArray, FunctionalMatrix


class CachedFunctionalMatrix:
    """Wrap a :class:`FunctionalMatrix` with entry / row / column memoisation.

    Parameters
    ----------
    inner
        The wrapped matrix. Typically already wrapped in a
        :class:`CountingFunctionalMatrix`.
    """

    def __init__(self, inner: FunctionalMatrix) -> None:
        self._inner = inner
        self.shape = inner.shape
        self.dtype = inner.dtype
        self._entry_cache: dict[tuple[int, int], float] = {}
        self._row_cache: dict[int, FloatArray] = {}
        self._col_cache: dict[int, FloatArray] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    @property
    def inner(self) -> FunctionalMatrix:
        """The wrapped (uncached) matrix."""
        return self._inner

    def reset_cache(self) -> None:
        """Discard cached rows, columns and entries; reset hit / miss counters."""
        self._entry_cache.clear()
        self._row_cache.clear()
        self._col_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    # ---- core oracle methods ----------------------------------------------

    def entry(self, i: int, j: int) -> float:
        key = (i, j)
        if key in self._entry_cache:
            self.cache_hits += 1
            return self._entry_cache[key]
        # Try to satisfy from a cached row or column first.
        if i in self._row_cache:
            self.cache_hits += 1
            cached_value = float(self._row_cache[i][j])
            self._entry_cache[key] = cached_value
            return cached_value
        if j in self._col_cache:
            self.cache_hits += 1
            cached_value = float(self._col_cache[j][i])
            self._entry_cache[key] = cached_value
            return cached_value
        self.cache_misses += 1
        value = float(self._inner.entry(i, j))
        self._entry_cache[key] = value
        return value

    def row(self, i: int, indices_col: npt.ArrayLike | None = None) -> FloatArray:
        if indices_col is None:
            if i in self._row_cache:
                self.cache_hits += 1
                return self._row_cache[i]
            self.cache_misses += 1
            full = self._inner.row(i)
            self._row_cache[i] = full
            return full
        cols = np.asarray(indices_col, dtype=np.intp)
        if i in self._row_cache:
            self.cache_hits += 1
            return self._row_cache[i][cols]
        # Cache the full row for future hits — the practicum's wording (p. 5)
        # is that "repeated requests must not increase the counter of real
        # computations", so paying once for the full row is appropriate here.
        self.cache_misses += 1
        full = self._inner.row(i)
        self._row_cache[i] = full
        return full[cols]

    def col(self, indices_row: npt.ArrayLike | None, j: int) -> FloatArray:
        if indices_row is None:
            if j in self._col_cache:
                self.cache_hits += 1
                return self._col_cache[j]
            self.cache_misses += 1
            full = self._inner.col(None, j)
            self._col_cache[j] = full
            return full
        rows = np.asarray(indices_row, dtype=np.intp)
        if j in self._col_cache:
            self.cache_hits += 1
            return self._col_cache[j][rows]
        self.cache_misses += 1
        full = self._inner.col(None, j)
        self._col_cache[j] = full
        return full[rows]

    def block(
        self, indices_row: npt.ArrayLike, indices_col: npt.ArrayLike
    ) -> FloatArray:
        # Block queries bypass the cache: the assignment treats a single
        # block call as one oracle event, and per-element memoisation would
        # be an unnecessary storage explosion.
        self.cache_misses += 1
        return self._inner.block(indices_row, indices_col)

    def matvec(self, x: FloatArray) -> FloatArray:
        self.cache_misses += 1
        return self._inner.matvec(x)

    def rmatvec(self, y: FloatArray) -> FloatArray:
        self.cache_misses += 1
        return self._inner.rmatvec(y)

    def matmat(self, x_block: FloatArray) -> FloatArray:
        self.cache_misses += 1
        return self._inner.matmat(x_block)

    def rmatmat(self, y_block: FloatArray) -> FloatArray:
        self.cache_misses += 1
        return self._inner.rmatmat(y_block)


__all__ = ["CachedFunctionalMatrix"]
