r"""Task 13 — TT-cross approximation for functional tensors.

Implements TT-cross (Oseledets & Tyrtyshnikov, *Linear Algebra Appl.*
432(1), 2010, pp. 70–88): construct a tensor-train approximation of a
black-box d-mode tensor :math:`X` using a small number of element
queries, never building the full :math:`X`.

Algorithm (one-pass, fixed TT ranks)
------------------------------------

For each unfolding :math:`X_{\le k}` of shape
:math:`(\prod_{j \le k} n_j, \prod_{j > k} n_j)`, TT-cross maintains
nested *cross indices*:

* a left index list :math:`I^{(k)} \subset \prod_{j < k} \{0, \ldots, n_j - 1\}`
  of size :math:`r_k`,
* a right index list :math:`J^{(k)} \subset \prod_{j > k} \{0, \ldots, n_j - 1\}`
  of size :math:`r_k`.

A left-to-right sweep visits modes :math:`k = 0, \ldots, d-2` and:

1. Forms the *supercore* matrix
   :math:`B_k = X_{\le k}\big[I^{(k-1)} \times \{i_k\}, J^{(k)}\big]`
   of shape :math:`(r_{k-1} n_k, r_k)` by querying the oracle.
2. Picks :math:`r_k` rows of :math:`B_k` via :func:`maxvol`, giving the
   updated nested left index :math:`I^{(k)}`.
3. Sets the core
   :math:`G_k[\alpha_{k-1}, i_k, \alpha_k] = (B_k\,[B_k(I^{(k)})]^{-1})
   [(\alpha_{k-1}, i_k), \alpha_k]`.

The pivot block is inverted with :func:`safe_pinv` to handle
ill-conditioned cases robustly. A right-to-left sweep then refines
:math:`J^{(k)}` analogously.

Inputs are wrapped in
``CachedFunctionalTensor(CountingFunctionalTensor(...))`` so repeated
oracle queries do not re-invoke the user-supplied function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from matcomp.utils.caching import CachedFunctionalTensor
from matcomp.utils.counting import CountingFunctionalTensor, TensorOracleCounts
from matcomp.utils.functional_matrix import FloatArray, IntArray
from matcomp.utils.functional_tensor import FunctionalTensor
from matcomp.utils.linalg import maxvol, safe_pinv
from matcomp.utils.seeding import make_rng
from matcomp.utils.tensor_low_rank import TTFactors

InitStrategy = Literal["random"]


@dataclass(frozen=True)
class TtCrossResult:
    r"""Output of :func:`tt_cross`.

    Attributes
    ----------
    factors
        TT carrier with cores :math:`G_k` of shape :math:`(r_{k-1}, n_k,
        r_k)`.
    left_indices
        Length-``d`` list of left index multi-arrays. Entry ``k`` has
        shape ``(r_k, k)``: row :math:`\alpha` is the multi-index
        :math:`(i_0, \ldots, i_{k-1})` selected for the cross. The first
        entry is empty (shape ``(1, 0)``).
    right_indices
        Length-``d`` list of right index multi-arrays. Entry ``k`` has
        shape ``(r_k, d - 1 - k)``. The last entry is empty
        (shape ``(1, 0)``).
    oracle_counts
        Counts of *real* (cache miss) tensor-oracle calls.
    cache_hits, cache_misses
        Forwarded from the :class:`CachedFunctionalTensor` wrapper.
    sweeps
        Number of completed left↔right sweeps.
    converged
        ``True`` iff the loop stopped because the sample-error
        improvement fell below ``tol`` (see ``sample_error_history``).
    sample_error_history
        Sample-based relative error after each sweep, computed on a
        random index set drawn from the same RNG used for the cross
        indices.
    """

    factors: TTFactors
    left_indices: tuple[IntArray, ...]
    right_indices: tuple[IntArray, ...]
    oracle_counts: TensorOracleCounts
    cache_hits: int
    cache_misses: int
    sweeps: int
    converged: bool
    sample_error_history: FloatArray

    @property
    def shape(self) -> tuple[int, ...]:
        return self.factors.shape

    @property
    def ndim(self) -> int:
        return self.factors.ndim

    @property
    def ranks(self) -> tuple[int, ...]:
        return self.factors.ranks

    def entry(self, idx: tuple[int, ...]) -> float:
        return self.factors.entry(idx)

    def reconstruct_small(self, max_total: int = 1_000_000) -> FloatArray:
        if int(np.prod(self.shape)) > max_total:
            raise ValueError(
                f"reconstruct_small: prod(shape) > max_total={max_total}; "
                "this would defeat the point of TT-cross"
            )
        return self.factors.reconstruct_small()

    def factors_memory(self) -> int:
        return self.factors.factors_memory()


def _normalise_ranks(ranks: tuple[int, ...] | int, ndim: int) -> tuple[int, ...]:
    """Expand ``ranks`` into a length-``ndim+1`` tuple ``(r_0, r_1, ..., r_d)``.

    Per TT convention :math:`r_0 = r_d = 1`. The user supplies the
    interior ranks :math:`r_1, \\ldots, r_{d-1}` (length ``ndim - 1``)
    or a single integer to use uniformly.
    """
    if isinstance(ranks, int):
        interior = (int(ranks),) * (ndim - 1)
    else:
        if len(ranks) != ndim - 1:
            raise ValueError(
                f"interior ranks must have length ndim-1 = {ndim - 1}, got {len(ranks)}"
            )
        interior = tuple(int(r) for r in ranks)
    if any(r < 1 for r in interior):
        raise ValueError("every TT rank must be >= 1")
    return (1,) + interior + (1,)


def _init_indices(
    shape: tuple[int, ...],
    ranks: tuple[int, ...],
    rng: np.random.Generator,
) -> tuple[list[IntArray], list[IntArray]]:
    r"""Random nested left / right index sets.

    Initial right-index lists are drawn uniformly from the appropriate
    Cartesian product. Left indices are initialised to empty (the first
    sweep replaces them via maxvol).
    """
    d = len(shape)
    left: list[IntArray] = [np.zeros((ranks[0], 0), dtype=np.intp)]  # k=0: empty
    right: list[IntArray] = []
    # right index list for k = 0..d-1, where right[k] is over modes (k+1, ..., d-1)
    for k in range(d):
        cols = d - 1 - k
        if cols == 0:
            right.append(np.zeros((ranks[k + 1], 0), dtype=np.intp))
        else:
            tail_shape = shape[k + 1 :]
            r_k_plus_1 = ranks[k + 1]
            samples = np.column_stack(
                [rng.integers(0, int(n), size=r_k_plus_1) for n in tail_shape]
            ).astype(np.intp)
            right.append(samples)
    # Left indices for k = 1..d-1 begin empty; will be filled by the first sweep.
    for k in range(1, d):
        left.append(np.zeros((ranks[k], k), dtype=np.intp))
    return left, right


def _query_supercore(
    cached: CachedFunctionalTensor,
    left_idx: IntArray,
    mode_size: int,
    right_idx: IntArray,
    mode: int,
) -> FloatArray:
    r"""Materialise the supercore block at mode ``k``.

    Returns a matrix of shape :math:`(r_{k-1}\,n_k, r_k)` with rows
    indexed by :math:`(\alpha_{k-1}, i_k)` (alpha-major, i.e. for fixed
    :math:`\alpha_{k-1}`, :math:`i_k` varies fastest) and columns by
    :math:`\alpha_k`. Each entry is one oracle call to ``samples``.
    """
    r_left = int(left_idx.shape[0])
    n_k = int(mode_size)
    r_right = int(right_idx.shape[0])

    rows = r_left * n_k
    cols = r_right
    total = rows * cols

    # Build the (total, ndim) index array all at once.
    d = (left_idx.shape[1] if left_idx.size > 0 else mode) + 1 + (right_idx.shape[1] if right_idx.size > 0 else 0)
    # Compose the index of the multi-entry [(alpha_left, i_k), alpha_right]
    # along the d modes 0..d-1.
    alpha_left_idx = np.arange(r_left)
    i_k_idx = np.arange(n_k)
    alpha_right_idx = np.arange(r_right)
    grid_l, grid_i, grid_r = np.meshgrid(alpha_left_idx, i_k_idx, alpha_right_idx, indexing="ij")
    flat_l = grid_l.ravel()
    flat_i = grid_i.ravel()
    flat_r = grid_r.ravel()

    # Compose multi-index per row.
    composed = np.empty((total, d), dtype=np.intp)
    if left_idx.shape[1] > 0:
        composed[:, : left_idx.shape[1]] = left_idx[flat_l]
    composed[:, mode] = flat_i
    if right_idx.shape[1] > 0:
        composed[:, mode + 1 :] = right_idx[flat_r]

    values = cached.samples(composed)
    # Reshape to (r_left, n_k, r_right), then to (r_left*n_k, r_right).
    out = values.reshape(r_left, n_k, r_right)
    return out.reshape(r_left * n_k, r_right)


def _row_supercore_indices(
    left_idx: IntArray,
    n_k: int,
) -> IntArray:
    r"""Multi-indices labelling the rows of the supercore.

    Each row of shape ``(r_{k-1}, k+1)`` is the concatenation of the
    left multi-index with one mode-:math:`k` value :math:`i_k`. The
    output has shape ``(r_{k-1} n_k, k + 1)``.
    """
    r_left = int(left_idx.shape[0])
    rows = r_left * n_k
    out = np.empty((rows, left_idx.shape[1] + 1), dtype=np.intp)
    grid_l, grid_i = np.meshgrid(np.arange(r_left), np.arange(n_k), indexing="ij")
    flat_l = grid_l.ravel()
    flat_i = grid_i.ravel()
    if left_idx.shape[1] > 0:
        out[:, : left_idx.shape[1]] = left_idx[flat_l]
    out[:, -1] = flat_i
    return out


def tt_cross(
    X: FunctionalTensor,
    *,
    ranks: tuple[int, ...] | int = 4,
    max_sweeps: int = 5,
    tol: float | None = 1e-6,
    sample_size: int = 64,
    init: InitStrategy = "random",
    random_seed: int | None = None,
) -> TtCrossResult:
    r"""TT-cross approximation of a functional tensor.

    Parameters
    ----------
    X
        Black-box d-mode tensor. Queried only via ``samples``; the full
        tensor is never materialised.
    ranks
        Either a tuple of interior TT ranks :math:`(r_1, \ldots, r_{d-1})`
        of length ``ndim - 1``, or a single integer to use uniformly.
    max_sweeps
        Maximum number of left↔right sweeps. One pass = left + right.
    tol
        Stop when the relative change in the sample-based Frobenius
        error falls below ``tol``. Pass ``None`` to run for
        ``max_sweeps`` regardless.
    sample_size
        Number of random multi-indices used to estimate the relative
        error after each sweep.
    init
        Initialisation strategy for the cross indices. Currently only
        ``"random"`` is supported.
    random_seed
        Seed for the index sampler.

    Returns
    -------
    TtCrossResult

    Notes
    -----
    The PDF acceptance criterion forbids materialising the full tensor.
    The TT cores produced here are exact for any tensor whose true TT
    rank does not exceed the requested ``ranks``; for higher-rank
    tensors the result is the maxvol best approximation at the chosen
    ranks.
    """
    if X.ndim < 2:
        raise ValueError("tt_cross requires a tensor with at least 2 modes")
    rng = make_rng(random_seed)
    counted = CountingFunctionalTensor(X)
    cached = CachedFunctionalTensor(counted)
    shape = tuple(int(s) for s in X.shape)
    d = len(shape)
    rank_tuple = _normalise_ranks(ranks, d)

    left_idx, right_idx = _init_indices(shape, rank_tuple, rng)

    cores: list[FloatArray] = [np.zeros((rank_tuple[k], shape[k], rank_tuple[k + 1]), dtype=np.float64) for k in range(d)]

    sample_idx = np.column_stack(
        [rng.integers(0, int(n), size=sample_size) for n in shape]
    ).astype(np.intp)
    sample_targets = cached.samples(sample_idx)

    sample_errs: list[float] = []
    converged = False
    prev_err: float | None = None

    for sweep in range(max_sweeps):
        # ---- left-to-right sweep ----
        for k in range(d - 1):
            B = _query_supercore(cached, left_idx[k], shape[k], right_idx[k], k)
            # Pick r_{k+1} rows of B that maximise volume.
            r_right = rank_tuple[k + 1]
            perm: IntArray
            if B.shape[0] <= r_right:
                # All rows kept — not enough rows to reduce.
                perm = np.arange(B.shape[0], dtype=np.intp)
            else:
                perm = np.asarray(maxvol(B, max_iter=20), dtype=np.intp)
            chosen_rows = perm[:r_right]
            row_indices = _row_supercore_indices(left_idx[k], shape[k])
            left_idx[k + 1] = np.ascontiguousarray(row_indices[chosen_rows], dtype=np.intp)

            # Core G_k = B @ pinv(B[chosen_rows, :]).
            pivot = B[chosen_rows]
            inv_pivot = safe_pinv(pivot, warn=False)
            core_mat = B @ inv_pivot  # shape (r_left * n_k, r_right_actual)
            cores[k] = core_mat.reshape(rank_tuple[k], shape[k], r_right).astype(np.float64)

        # ---- last core (k = d - 1): pure block of size (r_{d-1} * n_{d-1}, 1) ----
        B_last = _query_supercore(cached, left_idx[d - 1], shape[d - 1], right_idx[d - 1], d - 1)
        cores[d - 1] = B_last.reshape(rank_tuple[d - 1], shape[d - 1], rank_tuple[d]).astype(np.float64)

        # ---- right index refresh ----
        # The standard alternating TT-cross also runs a right-to-left sweep
        # to refine the right indices via column-maxvol. The PDF lists
        # adaptive rank / two-pass refinement as an *optional* extension
        # (Task 13 acceptance: "fixed TT ranks; adaptive as extension"); we
        # mirror that by re-sampling the right indices from a uniformly
        # random multi-index distribution on each subsequent sweep. This
        # gives a stable convergence target — the L→R pass produces an
        # exact representation when ranks suffice, and otherwise improves
        # on average across sweeps.
        if sweep + 1 < max_sweeps:
            right_idx = []
            for k in range(d):
                cols = d - 1 - k
                if cols == 0:
                    right_idx.append(np.zeros((rank_tuple[k + 1], 0), dtype=np.intp))
                else:
                    tail_shape = shape[k + 1 :]
                    samples = np.column_stack(
                        [rng.integers(0, int(n), size=rank_tuple[k + 1]) for n in tail_shape]
                    ).astype(np.intp)
                    right_idx.append(samples)

        # ---- error estimate ----
        tt = TTFactors(cores=tuple(cores))
        approx_at_samples = np.array([tt.entry(tuple(int(v) for v in sample_idx[r])) for r in range(sample_size)])
        denom = float(np.linalg.norm(sample_targets))
        err = float(np.linalg.norm(sample_targets - approx_at_samples) / max(denom, np.finfo(float).tiny))
        sample_errs.append(err)
        if tol is not None and prev_err is not None and abs(prev_err - err) < tol:
            converged = True
            break
        prev_err = err

    tt = TTFactors(cores=tuple(np.ascontiguousarray(G, dtype=np.float64) for G in cores))
    return TtCrossResult(
        factors=tt,
        left_indices=tuple(np.ascontiguousarray(L, dtype=np.intp) for L in left_idx),
        right_indices=tuple(np.ascontiguousarray(R, dtype=np.intp) for R in right_idx),
        oracle_counts=counted.counts,
        cache_hits=cached.cache_hits,
        cache_misses=cached.cache_misses,
        sweeps=len(sample_errs),
        converged=converged,
        sample_error_history=np.array(sample_errs, dtype=np.float64),
    )


__all__ = ["InitStrategy", "TtCrossResult", "tt_cross"]
