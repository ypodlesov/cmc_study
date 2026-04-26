r"""Concrete :class:`FunctionalTensor` implementations used as test inputs.

Mirrors :mod:`matcomp.utils.test_matrices` for tensors. The PDF (p. 2)
prescribes a minimum exercise set: a tensor with known multilinear rank,
a kernel-style smooth tensor, and an analytical functional tensor. This
module provides those plus a generic oracle wrapper for Task 13.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from matcomp.utils.functional_matrix import FloatArray, IntArray
from matcomp.utils.functional_tensor import (
    BaseFunctionalTensor,
    IndexTuple,
)
from matcomp.utils.seeding import make_rng
from matcomp.utils.tensor_linalg import cp_to_dense, tucker_to_dense


class CPSyntheticTensor(BaseFunctionalTensor):
    r"""Exact CP tensor :math:`X = \sum_r \lambda_r \bigotimes_k A_k[:, r]`.

    Used to verify *recovery*: a method that claims to produce a rank-
    ``R`` CP approximation of an exact rank-``R`` CP tensor should return
    zero error (up to round-off).

    Parameters
    ----------
    weights
        Length-``R`` vector :math:`\lambda`.
    factors
        Tuple of ``d`` factor matrices, each of shape ``(n_k, R)``.
    """

    def __init__(self, weights: FloatArray, factors: Sequence[FloatArray]) -> None:
        if any(fac.ndim != 2 for fac in factors):
            raise ValueError("CPSyntheticTensor: factors must be 2-D")
        R = int(weights.size)
        if any(fac.shape[1] != R for fac in factors):
            raise ValueError("CPSyntheticTensor: factor column counts must match weights")
        shape = tuple(int(fac.shape[0]) for fac in factors)
        super().__init__(shape=shape, dtype=np.dtype(np.float64))
        self._weights = np.ascontiguousarray(weights, dtype=np.float64)
        self._factors = tuple(np.ascontiguousarray(f, dtype=np.float64) for f in factors)
        self._dense_cache: FloatArray | None = None

    @property
    def rank(self) -> int:
        """Exact CP rank."""
        return int(self._weights.size)

    def _dense(self) -> FloatArray:
        if self._dense_cache is None:
            self._dense_cache = cp_to_dense(self._weights, list(self._factors))
        return self._dense_cache

    def entry(self, idx: IndexTuple) -> float:
        accum = self._weights.copy()
        for k, fac in enumerate(self._factors):
            accum = accum * fac[int(idx[k]), :]
        return float(accum.sum())

    def block(self, indices: tuple[npt.ArrayLike, ...]) -> FloatArray:
        idx_arrays = [np.asarray(ix, dtype=np.intp) for ix in indices]
        return self._dense()[np.ix_(*idx_arrays)]

    def fiber(self, mode: int, fixed_indices: tuple[int | None, ...]) -> FloatArray:
        slicer = tuple(
            slice(None) if d == mode else int(fixed_indices[d])  # type: ignore[arg-type]
            for d in range(self.ndim)
        )
        return np.ascontiguousarray(self._dense()[slicer])

    def samples(self, idx_array: IntArray) -> FloatArray:
        idx = np.asarray(idx_array, dtype=np.intp)
        return self._dense()[tuple(idx[:, k] for k in range(self.ndim))]


class TuckerSyntheticTensor(BaseFunctionalTensor):
    r"""Exact Tucker tensor :math:`X = G \times_1 U_1 \cdots \times_d U_d`.

    Parameters
    ----------
    core
        Core tensor of multilinear shape :math:`(r_1, \ldots, r_d)`.
    factors
        Tuple of ``d`` factor matrices :math:`U_k` of shape
        :math:`(n_k, r_k)`.
    """

    def __init__(self, core: FloatArray, factors: Sequence[FloatArray]) -> None:
        if core.ndim != len(factors):
            raise ValueError("TuckerSyntheticTensor: core.ndim must match len(factors)")
        shape = tuple(int(fac.shape[0]) for fac in factors)
        super().__init__(shape=shape, dtype=np.dtype(np.float64))
        self._core = np.ascontiguousarray(core, dtype=np.float64)
        self._factors = tuple(np.ascontiguousarray(f, dtype=np.float64) for f in factors)
        self._dense_cache: FloatArray | None = None

    def _dense(self) -> FloatArray:
        if self._dense_cache is None:
            self._dense_cache = tucker_to_dense(self._core, list(self._factors))
        return self._dense_cache

    def entry(self, idx: IndexTuple) -> float:
        return float(self._dense()[idx])

    def block(self, indices: tuple[npt.ArrayLike, ...]) -> FloatArray:
        idx_arrays = [np.asarray(ix, dtype=np.intp) for ix in indices]
        return self._dense()[np.ix_(*idx_arrays)]


class LowRankPlusNoiseTensor(BaseFunctionalTensor):
    r"""CP tensor :math:`X_{CP}` plus i.i.d. Gaussian noise.

    The noise sample is fixed at construction so :meth:`entry` is
    deterministic.
    """

    def __init__(
        self,
        weights: FloatArray,
        factors: Sequence[FloatArray],
        noise_level: float,
        random_seed: int | None = None,
    ) -> None:
        if noise_level < 0.0:
            raise ValueError("noise_level must be non-negative")
        shape = tuple(int(fac.shape[0]) for fac in factors)
        super().__init__(shape=shape, dtype=np.dtype(np.float64))
        rng = make_rng(random_seed)
        self._signal = CPSyntheticTensor(weights, factors)
        self._noise = noise_level * rng.standard_normal(shape).astype(np.float64)

    def entry(self, idx: IndexTuple) -> float:
        return self._signal.entry(idx) + float(self._noise[idx])

    def block(self, indices: tuple[npt.ArrayLike, ...]) -> FloatArray:
        idx_arrays = [np.asarray(ix, dtype=np.intp) for ix in indices]
        return self._signal.block(indices) + self._noise[np.ix_(*idx_arrays)]

    def samples(self, idx_array: IntArray) -> FloatArray:
        idx = np.asarray(idx_array, dtype=np.intp)
        signal = self._signal.samples(idx)
        noise = self._noise[tuple(idx[:, k] for k in range(self.ndim))]
        return signal + noise


class Hilbert3DTensor(BaseFunctionalTensor):
    r"""3-D Hilbert tensor :math:`X[i, j, k] = 1 / (i + j + k + 1)`.

    Smooth, slow-decay analogue of the matrix Hilbert. Used as a TT-cross
    test target.
    """

    def __init__(self, n: int, m: int | None = None, p: int | None = None) -> None:
        if m is None:
            m = n
        if p is None:
            p = n
        super().__init__(shape=(int(n), int(m), int(p)), dtype=np.dtype(np.float64))

    def entry(self, idx: IndexTuple) -> float:
        i, j, k = (int(v) for v in idx)
        return 1.0 / (i + j + k + 1.0)

    def block(self, indices: tuple[npt.ArrayLike, ...]) -> FloatArray:
        I = np.asarray(indices[0], dtype=np.intp)
        J = np.asarray(indices[1], dtype=np.intp)
        K = np.asarray(indices[2], dtype=np.intp)
        return 1.0 / (I[:, None, None] + J[None, :, None] + K[None, None, :] + 1.0)

    def samples(self, idx_array: IntArray) -> FloatArray:
        idx = np.asarray(idx_array, dtype=np.intp)
        return 1.0 / (idx[:, 0] + idx[:, 1] + idx[:, 2] + 1.0)


class GaussianKernel3DTensor(BaseFunctionalTensor):
    r"""Gaussian kernel tensor :math:`X[i, j, k] = \exp(-(\|x_i - y_j\|^2 +
    \|y_j - z_k\|^2) / \sigma^2)`.

    Used to exercise CP-ALS and TT-cross on a smooth target with rapid
    multilinear-rank decay. ``x``, ``y``, ``z`` may be 1-D or 2-D
    coordinate arrays.
    """

    def __init__(
        self,
        x: FloatArray,
        y: FloatArray,
        z: FloatArray,
        sigma: float,
    ) -> None:
        if sigma <= 0.0:
            raise ValueError("GaussianKernel3DTensor requires sigma > 0")
        x_2d = np.atleast_2d(x).reshape(-1, 1) if x.ndim == 1 else np.ascontiguousarray(x, dtype=np.float64)
        y_2d = np.atleast_2d(y).reshape(-1, 1) if y.ndim == 1 else np.ascontiguousarray(y, dtype=np.float64)
        z_2d = np.atleast_2d(z).reshape(-1, 1) if z.ndim == 1 else np.ascontiguousarray(z, dtype=np.float64)
        if not (x_2d.shape[1] == y_2d.shape[1] == z_2d.shape[1]):
            raise ValueError("x, y, z must share the trailing dimension")
        super().__init__(
            shape=(int(x_2d.shape[0]), int(y_2d.shape[0]), int(z_2d.shape[0])),
            dtype=np.dtype(np.float64),
        )
        self._x = x_2d.astype(np.float64, copy=False)
        self._y = y_2d.astype(np.float64, copy=False)
        self._z = z_2d.astype(np.float64, copy=False)
        self._sigma = float(sigma)

    def _energy(self, i: int, j: int, k: int) -> float:
        dxy = self._x[i] - self._y[j]
        dyz = self._y[j] - self._z[k]
        return float(dxy @ dxy + dyz @ dyz)

    def entry(self, idx: IndexTuple) -> float:
        return float(np.exp(-self._energy(int(idx[0]), int(idx[1]), int(idx[2]))
                            / (self._sigma * self._sigma)))

    def block(self, indices: tuple[npt.ArrayLike, ...]) -> FloatArray:
        I = np.asarray(indices[0], dtype=np.intp)
        J = np.asarray(indices[1], dtype=np.intp)
        K = np.asarray(indices[2], dtype=np.intp)
        # ||x_i - y_j||^2 + ||y_j - z_k||^2 vectorised
        xi = self._x[I]
        yj = self._y[J]
        zk = self._z[K]
        d_xy = ((xi[:, None, :] - yj[None, :, :]) ** 2).sum(axis=-1)  # (|I|, |J|)
        d_yz = ((yj[:, None, :] - zk[None, :, :]) ** 2).sum(axis=-1)  # (|J|, |K|)
        energy = d_xy[:, :, None] + d_yz[None, :, :]
        return np.exp(-energy / (self._sigma * self._sigma))

    def samples(self, idx_array: IntArray) -> FloatArray:
        idx = np.asarray(idx_array, dtype=np.intp)
        out = np.empty(idx.shape[0], dtype=np.float64)
        for n in range(idx.shape[0]):
            out[n] = self.entry((int(idx[n, 0]), int(idx[n, 1]), int(idx[n, 2])))
        return out


class FunctionTensor(BaseFunctionalTensor):
    r"""Generic oracle wrapper around a Python callable ``func(idx) -> float``.

    Used to feed Task 13 (TT-cross) an arbitrary functional tensor — the
    user supplies ``func`` and the shape, and TT-cross sees the standard
    oracle interface.

    Parameters
    ----------
    func
        Callable mapping a tuple of indices to a scalar.
    shape
        Tensor shape ``(n_1, ..., n_d)``.
    """

    def __init__(self, func: Callable[[IndexTuple], float], shape: tuple[int, ...]) -> None:
        super().__init__(shape=shape, dtype=np.dtype(np.float64))
        self._func = func

    def entry(self, idx: IndexTuple) -> float:
        return float(self._func(tuple(int(v) for v in idx)))


def random_cp(
    shape: tuple[int, ...],
    rank: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, list[FloatArray]]:
    """Random unit-weight CP factorisation with i.i.d. Gaussian factors."""
    weights: FloatArray = np.ones(rank, dtype=np.float64)
    factors: list[FloatArray] = [
        rng.standard_normal((int(n), rank)).astype(np.float64) for n in shape
    ]
    return weights, factors


def random_tucker(
    shape: tuple[int, ...],
    ranks: tuple[int, ...],
    rng: np.random.Generator,
) -> tuple[FloatArray, list[FloatArray]]:
    """Random Tucker factorisation with orthonormal factor matrices."""
    if len(shape) != len(ranks):
        raise ValueError("shape and ranks must have the same length")
    core: FloatArray = rng.standard_normal(ranks).astype(np.float64)
    factors: list[FloatArray] = []
    for n, r in zip(shape, ranks, strict=True):
        Q, _ = np.linalg.qr(rng.standard_normal((int(n), int(r))).astype(np.float64), mode="reduced")
        factors.append(np.ascontiguousarray(Q, dtype=np.float64))
    return core, factors


__all__ = [
    "CPSyntheticTensor",
    "FunctionTensor",
    "GaussianKernel3DTensor",
    "Hilbert3DTensor",
    "LowRankPlusNoiseTensor",
    "TuckerSyntheticTensor",
    "random_cp",
    "random_tucker",
]
