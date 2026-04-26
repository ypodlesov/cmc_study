r"""Low-rank tensor carriers and the unifying protocol.

Mirrors :mod:`matcomp.utils.low_rank` but for d-mode tensors. Three
formats cover Tasks 8–13:

* :class:`CPFactors` — :math:`X \approx \sum_r \lambda_r \bigotimes_k
  A_k[:, r]` (canonical / CP). Tasks 8, 9, 10, 11.
* :class:`TuckerFactors` — :math:`X \approx G \times_1 U_1 \cdots
  \times_d U_d`. Task 12 (ST-HOSVD).
* :class:`TTFactors` — :math:`X \approx \sum_{\alpha} G_1[:, i_1,
  \alpha_1] G_2[\alpha_1, i_2, \alpha_2] \cdots G_d[\alpha_{d-1}, i_d,
  :]`. Task 13 (TT-cross).

The :class:`LowRankTensor` :class:`typing.Protocol` is what cross-task
benchmarks consume.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from matcomp.utils.functional_matrix import FloatArray
from matcomp.utils.tensor_linalg import (
    cp_to_dense,
    tt_entry,
    tt_to_dense,
    tucker_to_dense,
)


@runtime_checkable
class LowRankTensor(Protocol):
    """Protocol for any rank-bounded approximation of a d-mode tensor."""

    shape: tuple[int, ...]
    ndim: int

    def entry(self, idx: tuple[int, ...]) -> float:
        """Return :math:`\\hat X[i_1, \\ldots, i_d]`."""
        ...

    def reconstruct_small(self) -> FloatArray:
        """Return the dense materialisation."""
        ...

    def factors_memory(self) -> int:
        """Return the total ``nbytes`` over all stored factor arrays."""
        ...


@dataclass(frozen=True)
class CPFactors:
    r"""Canonical-polyadic carrier of a tensor.

    Represents :math:`X \approx \sum_r \lambda_r \bigotimes_k A_k[:, r]`.

    Parameters
    ----------
    weights
        Length-``R`` vector :math:`\lambda`.
    factors
        Tuple of ``d`` factor matrices :math:`A_k` of shape
        :math:`(n_k, R)`.
    """

    weights: FloatArray
    factors: tuple[FloatArray, ...]

    def __post_init__(self) -> None:
        if any(fac.ndim != 2 for fac in self.factors):
            raise ValueError("CPFactors: every factor must be 2-D")
        R = int(self.weights.size)
        if any(fac.shape[1] != R for fac in self.factors):
            raise ValueError("CPFactors: factor column counts must equal len(weights)")

    @property
    def rank(self) -> int:
        """CP rank :math:`R`."""
        return int(self.weights.size)

    @property
    def shape(self) -> tuple[int, ...]:
        """Tensor mode sizes :math:`(n_1, \\ldots, n_d)`."""
        return tuple(int(fac.shape[0]) for fac in self.factors)

    @property
    def ndim(self) -> int:
        """Number of modes :math:`d`."""
        return len(self.factors)

    def entry(self, idx: tuple[int, ...]) -> float:
        r""":math:`\sum_r \lambda_r \prod_k A_k[i_k, r]`."""
        if len(idx) != self.ndim:
            raise ValueError(f"CPFactors.entry: expected {self.ndim} indices")
        accum = np.array(self.weights, dtype=np.float64, copy=True)
        for k, fac in enumerate(self.factors):
            accum = accum * fac[int(idx[k]), :]
        return float(accum.sum())

    def reconstruct_small(self) -> FloatArray:
        r"""Materialise the CP tensor; intended for small tests."""
        return cp_to_dense(self.weights, list(self.factors))

    def factors_memory(self) -> int:
        """Total ``nbytes`` over weights and factor matrices."""
        return int(self.weights.nbytes) + int(sum(fac.nbytes for fac in self.factors))


@dataclass(frozen=True)
class TuckerFactors:
    r"""Tucker carrier :math:`X \approx G \times_1 U_1 \cdots \times_d U_d`.

    Parameters
    ----------
    core
        Core tensor of multilinear shape :math:`(r_1, \ldots, r_d)`.
    factors
        Tuple of ``d`` factor matrices :math:`U_k` of shape
        :math:`(n_k, r_k)`.
    """

    core: FloatArray
    factors: tuple[FloatArray, ...]

    def __post_init__(self) -> None:
        if self.core.ndim != len(self.factors):
            raise ValueError(
                f"TuckerFactors: core.ndim={self.core.ndim} must match len(factors)={len(self.factors)}"
            )
        for k, fac in enumerate(self.factors):
            if fac.ndim != 2 or fac.shape[1] != self.core.shape[k]:
                raise ValueError(
                    f"TuckerFactors: factor {k} shape {fac.shape} incompatible with "
                    f"core mode {k} of size {self.core.shape[k]}"
                )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(fac.shape[0]) for fac in self.factors)

    @property
    def ndim(self) -> int:
        return len(self.factors)

    @property
    def multilinear_rank(self) -> tuple[int, ...]:
        """Tucker multilinear rank :math:`(r_1, \\ldots, r_d)`."""
        return tuple(int(s) for s in self.core.shape)

    def entry(self, idx: tuple[int, ...]) -> float:
        if len(idx) != self.ndim:
            raise ValueError(f"TuckerFactors.entry: expected {self.ndim} indices")
        # Use einsum-free contraction: pick rows of each factor at i_k, then
        # contract against the core via tensordot.
        out = self.core
        for k, fac in enumerate(self.factors):
            out = np.tensordot(out, fac[int(idx[k]), :], axes=([0], [0]))
        return float(out)

    def reconstruct_small(self) -> FloatArray:
        return tucker_to_dense(self.core, list(self.factors))

    def factors_memory(self) -> int:
        return int(self.core.nbytes) + int(sum(fac.nbytes for fac in self.factors))


@dataclass(frozen=True)
class TTFactors:
    r"""Tensor-train carrier with cores :math:`G_k` of shape
    :math:`(r_{k-1}, n_k, r_k)`, :math:`r_0 = r_d = 1`.

    Parameters
    ----------
    cores
        Tuple of ``d`` core tensors.
    """

    cores: tuple[FloatArray, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.cores:
            raise ValueError("TTFactors: at least one core required")
        if self.cores[0].shape[0] != 1:
            raise ValueError(f"TTFactors: first core must have r0=1, got {self.cores[0].shape[0]}")
        if self.cores[-1].shape[2] != 1:
            raise ValueError(f"TTFactors: last core must have r_d=1, got {self.cores[-1].shape[2]}")
        for k in range(len(self.cores) - 1):
            if self.cores[k].shape[2] != self.cores[k + 1].shape[0]:
                raise ValueError(
                    f"TTFactors: core {k} right rank {self.cores[k].shape[2]} "
                    f"!= core {k + 1} left rank {self.cores[k + 1].shape[0]}"
                )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(G.shape[1]) for G in self.cores)

    @property
    def ndim(self) -> int:
        return len(self.cores)

    @property
    def ranks(self) -> tuple[int, ...]:
        """TT ranks :math:`(r_0, r_1, \\ldots, r_d)`. ``r_0 = r_d = 1``."""
        out = [int(self.cores[0].shape[0])]
        for G in self.cores:
            out.append(int(G.shape[2]))
        return tuple(out)

    def entry(self, idx: tuple[int, ...]) -> float:
        return tt_entry(list(self.cores), idx)

    def reconstruct_small(self) -> FloatArray:
        return tt_to_dense(list(self.cores))

    def storage(self) -> int:
        """Total ``nbytes`` over all TT cores (PDF Task 13 wording)."""
        return int(sum(G.nbytes for G in self.cores))

    def factors_memory(self) -> int:
        """Alias for :meth:`storage`."""
        return self.storage()


Tensors = CPFactors | TuckerFactors | TTFactors


def cp_from_factor_list(
    weights: FloatArray, factors: Sequence[FloatArray]
) -> CPFactors:
    """Convenience constructor from a list of factors."""
    return CPFactors(weights=np.ascontiguousarray(weights, dtype=np.float64),
                     factors=tuple(np.ascontiguousarray(f, dtype=np.float64) for f in factors))


def tucker_from_factor_list(
    core: FloatArray, factors: Sequence[FloatArray]
) -> TuckerFactors:
    """Convenience constructor from a list of factors."""
    return TuckerFactors(
        core=np.ascontiguousarray(core, dtype=np.float64),
        factors=tuple(np.ascontiguousarray(f, dtype=np.float64) for f in factors),
    )


def tt_from_core_list(cores: Sequence[FloatArray]) -> TTFactors:
    """Convenience constructor from a list of cores."""
    return TTFactors(
        cores=tuple(np.ascontiguousarray(G, dtype=np.float64) for G in cores)
    )


__all__ = [
    "CPFactors",
    "LowRankTensor",
    "TTFactors",
    "Tensors",
    "TuckerFactors",
    "cp_from_factor_list",
    "tt_from_core_list",
    "tucker_from_factor_list",
]
