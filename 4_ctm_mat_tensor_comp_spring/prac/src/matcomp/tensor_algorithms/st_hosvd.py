r"""Task 12 — Sequentially Truncated HOSVD.

Implements the ST-HOSVD of Vannieuwenhoven, Vandebril and Meerbergen
(*SIAM J. Sci. Comput.* 34(2), 2012, pp. A1027–A1052). Given a dense
tensor :math:`X` of shape :math:`(n_1, \ldots, n_d)`, ST-HOSVD returns
a Tucker approximation

.. math::

    X \;\approx\; G \times_1 U_1 \times_2 \cdots \times_d U_d,

where each :math:`U_k` is :math:`n_k \times r_k` orthonormal and
:math:`G \in \mathbb{R}^{r_1 \times \cdots \times r_d}` is the core.
The "sequential" part: after truncating mode :math:`k`, the tensor is
*compressed along mode k* (multiplied by :math:`U_k^\top`) before
moving on to mode :math:`k + 1`. Each subsequent SVD operates on a
smaller unfolding, so ST-HOSVD is strictly cheaper than the classical
all-modes-from-X HOSVD of De Lathauwer–De Moor–Vandewalle (2000).

Two stopping rules:

* ``ranks=(r_1, ..., r_d)`` — fixed multilinear rank.
* ``eps`` — relative Frobenius error budget. The budget is split
  equally across modes: each mode is truncated so that its truncation
  energy does not exceed :math:`\epsilon^2 / d \cdot \lVert X\rVert^2`
  (Vannieuwenhoven §3.2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from matcomp.matrix_algorithms.randomized_svd import randomized_svd
from matcomp.utils.functional_matrix import FloatArray
from matcomp.utils.seeding import make_rng
from matcomp.utils.tensor_linalg import (
    mode_dot,
    multilinear_rank,
    tucker_to_dense,
    unfold,
)
from matcomp.utils.tensor_low_rank import TuckerFactors
from matcomp.utils.test_matrices import DenseMatrix


@dataclass(frozen=True)
class StHosvdResult:
    r"""Output of :func:`st_hosvd`.

    Attributes
    ----------
    factors
        Tucker carrier with core and orthonormal factor matrices.
    ranks
        Final multilinear rank :math:`(r_1, \ldots, r_d)`. Equal to the
        ``ranks`` argument when supplied; auto-selected when ``eps`` is
        used.
    rel_error
        Relative Frobenius error :math:`\lVert X - \hat X\rVert_F /
        \lVert X\rVert_F` measured directly on the dense input.
    mode_order
        Order in which the modes were processed (defaults to
        ``(0, 1, ..., d - 1)``).
    """

    factors: TuckerFactors
    ranks: tuple[int, ...]
    rel_error: float
    mode_order: tuple[int, ...]

    # ---- LowRankTensor passthroughs ---------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.factors.shape

    @property
    def ndim(self) -> int:
        return self.factors.ndim

    def entry(self, idx: tuple[int, ...]) -> float:
        return self.factors.entry(idx)

    def reconstruct_small(self) -> FloatArray:
        return self.factors.reconstruct_small()

    def factors_memory(self) -> int:
        return self.factors.factors_memory()


def _truncated_svd_left(
    M: FloatArray,
    *,
    rank: int | None,
    eps_budget_sq: float | None,
    randomized: bool,
    oversample: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, int]:
    r"""Return the leading left singular vectors of ``M`` and the kept rank.

    Truncation rule: ``rank`` if supplied, else greedy by squared-tail
    budget — keep the smallest ``r`` such that
    :math:`\sum_{j > r} \sigma_j^{2} \le \mathrm{eps\_budget\_sq}`.
    """
    m, n = M.shape
    if rank is not None:
        r = int(min(max(rank, 0), min(m, n)))
        if randomized and r > 0 and r < min(m, n):
            wrap = DenseMatrix(np.ascontiguousarray(M, dtype=np.float64))
            extra = max(0, min(oversample, min(m, n) - r))
            res = randomized_svd(
                wrap,
                r=r,
                oversampling=extra,
                random_seed=int(rng.integers(0, 2**31 - 1)),
            )
            return np.ascontiguousarray(res.factors.U, dtype=np.float64), r
        U_full, _, _ = np.linalg.svd(M, full_matrices=False)
        return np.ascontiguousarray(U_full[:, :r], dtype=np.float64), r
    # eps path
    assert eps_budget_sq is not None
    U_full, S_full, _ = np.linalg.svd(M, full_matrices=False)
    # Pick smallest r with sum_{j > r} sigma_j^2 <= budget.
    tail_sq = np.cumsum(S_full[::-1] ** 2)[::-1]  # tail_sq[r] = sigma_r^2 + ...
    # tail beyond rank r is tail_sq[r], so we want smallest r with tail_sq[r] <= budget.
    # tail_sq[len(S_full)] would be 0 — extend the array.
    extended = np.concatenate([tail_sq, np.array([0.0])])
    candidates = np.where(extended <= eps_budget_sq)[0]
    r = int(candidates[0]) if candidates.size > 0 else int(S_full.size)
    return np.ascontiguousarray(U_full[:, :r], dtype=np.float64), r


def st_hosvd(
    X: FloatArray,
    *,
    ranks: tuple[int, ...] | None = None,
    eps: float | None = None,
    mode_order: Sequence[int] | None = None,
    randomized: bool = False,
    oversample: int = 10,
    random_seed: int | None = None,
) -> StHosvdResult:
    r"""Sequentially-truncated HOSVD of a dense tensor.

    Parameters
    ----------
    X
        Dense input tensor.
    ranks
        Target multilinear rank :math:`(r_1, \ldots, r_d)`. Mutually
        exclusive with ``eps``.
    eps
        Relative Frobenius error budget. Modes are truncated greedily
        with an equal share of :math:`\epsilon^2 \lVert X\rVert^2 / d`.
    mode_order
        Order in which to process modes. Defaults to
        ``(0, 1, ..., d - 1)``.
    randomized
        If ``True``, replace the per-mode :func:`numpy.linalg.svd` with
        :func:`matcomp.matrix_algorithms.randomized_svd.randomized_svd`.
        Useful when the unfoldings are tall.
    oversample
        Oversampling parameter for the randomized SVD path.
    random_seed
        Seed for the randomized path's Gaussian sketch.

    Returns
    -------
    StHosvdResult
    """
    if X.ndim < 2:
        raise ValueError("st_hosvd requires a tensor with at least 2 modes")
    if (ranks is None) == (eps is None):
        raise ValueError("st_hosvd: pass exactly one of ranks or eps")
    rng = make_rng(random_seed)

    d = X.ndim
    if mode_order is None:
        mode_order_t = tuple(range(d))
    else:
        mode_order_t = tuple(int(k) for k in mode_order)
        if sorted(mode_order_t) != list(range(d)):
            raise ValueError("mode_order must be a permutation of range(ndim)")

    eps_budget_sq: float | None
    ranks_t: tuple[int, ...] | None
    if ranks is not None:
        if len(ranks) != d:
            raise ValueError("ranks must have length equal to X.ndim")
        ranks_t = tuple(int(r) for r in ranks)
        eps_budget_sq = None
    else:
        assert eps is not None
        ranks_t = None
        norm_sq = float(np.sum(X * X))
        # Equal-share budget per mode (the ST-HOSVD bound is additive in
        # squared truncation tails, see §3.2 of Vannieuwenhoven et al.).
        eps_budget_sq = float(eps) ** 2 * norm_sq / float(d)

    factors_unordered: list[FloatArray | None] = [None] * d
    chosen_ranks: list[int | None] = [None] * d
    G: FloatArray = X.astype(np.float64, copy=True)

    for mode in mode_order_t:
        Mk = unfold(G, mode)
        target_rank = ranks_t[mode] if ranks_t is not None else None
        U_k, r_k = _truncated_svd_left(
            Mk,
            rank=target_rank,
            eps_budget_sq=eps_budget_sq,
            randomized=randomized,
            oversample=oversample,
            rng=rng,
        )
        if r_k == 0:
            # Degenerate truncation; fall back to retaining one component
            # to keep the Tucker shape well-defined.
            r_k = 1
            U_k = np.ascontiguousarray(np.eye(Mk.shape[0], 1), dtype=np.float64)
        factors_unordered[mode] = U_k
        chosen_ranks[mode] = r_k
        # Compress G along this mode by U_k^T.
        G = mode_dot(G, U_k, mode, transpose=True)

    # All factors are filled at this point; help mypy narrow the Optional.
    assert all(f is not None for f in factors_unordered)
    assert all(r is not None for r in chosen_ranks)
    factors_final: tuple[FloatArray, ...] = tuple(
        np.ascontiguousarray(factors_unordered[k], dtype=np.float64) for k in range(d)
    )
    ranks_final: tuple[int, ...] = tuple(int(r) for r in chosen_ranks if r is not None)

    tucker = TuckerFactors(core=np.ascontiguousarray(G, dtype=np.float64), factors=factors_final)
    rel_err = float(np.linalg.norm(X - tucker_to_dense(tucker.core, list(tucker.factors))))
    norm_X = float(np.linalg.norm(X))
    rel_err = rel_err / norm_X if norm_X > 0.0 else 0.0
    return StHosvdResult(
        factors=tucker,
        ranks=ranks_final,
        rel_error=rel_err,
        mode_order=mode_order_t,
    )


def reconstruct(result: StHosvdResult) -> FloatArray:
    """Convenience: return the dense Tucker reconstruction."""
    return result.reconstruct_small()


def multilinear_rank_of(X: FloatArray, eps: float = 0.0) -> tuple[int, ...]:
    """Re-export of :func:`matcomp.utils.tensor_linalg.multilinear_rank`."""
    return multilinear_rank(X, eps=eps)


__all__ = [
    "StHosvdResult",
    "multilinear_rank_of",
    "reconstruct",
    "st_hosvd",
]
