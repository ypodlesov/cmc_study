r"""Task 8 — CP-ALS with the source tensor in canonical (CP) form.

Specialisation of the dense CP-ALS for the case where the target tensor
is itself a CP decomposition

.. math::

    X \;=\; \sum_{r=1}^{R_{\text{src}}}\, \lambda^{(s)}_r\,
    \bigotimes_k A^{(s)}_k[:, r],

and we want to compress it to a smaller CP rank :math:`R_{\text{tgt}} <
R_{\text{src}}`. The point of the task (PDF p. 8) is to do everything
without densifying :math:`X`. This module never materialises the source
tensor: every quantity is computed via factor inner products / CP-vs-CP
MTTKRP.

Concretely, the ALS update for the target factor :math:`B_n` solves

.. math::

    B_n \, V \;=\; M, \qquad
    V \;=\; \bigodot_{k \neq n} B_k^{\top} B_k, \quad
    M \;=\; \sum_{r} \lambda^{(s)}_r\, A^{(s)}_n[:, r] \,
    \prod_{k \neq n} \big(A^{(s)}_k[:, r]^{\top} B_k\big)^{\top}_{:, r}.

The right-hand side reads off the source factors directly — there is no
:math:`X_{(n)}` to unfold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from matcomp.utils.functional_matrix import FloatArray
from matcomp.utils.linalg import safe_pinv
from matcomp.utils.seeding import make_rng
from matcomp.utils.tensor_linalg import (
    cp_factor_grams,
    cp_inner_product,
    cp_norm_sq,
)
from matcomp.utils.tensor_low_rank import CPFactors

InitStrategy = Literal["random", "source"]


@dataclass(frozen=True)
class CpAlsCpResult:
    r"""Output of :func:`cp_als_from_cp`.

    Attributes
    ----------
    factors
        Compressed CP carrier of rank :math:`R_{\text{target}}`.
    loss_history
        Length-``iterations`` array of relative Frobenius errors after
        each completed sweep, computed via CP inner products only.
    iterations
        Number of completed sweeps.
    converged
        ``True`` iff the loop stopped because the loss change fell
        below ``tol``.
    """

    factors: CPFactors
    loss_history: FloatArray
    iterations: int
    converged: bool

    # ---- LowRankTensor passthroughs ---------------------------------------

    @property
    def rank(self) -> int:
        return self.factors.rank

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


def _init_target(
    source: CPFactors,
    target_rank: int,
    init: InitStrategy,
    rng: np.random.Generator,
) -> tuple[FloatArray, list[FloatArray]]:
    """Initial weights / factors for the target CP."""
    shape = source.shape
    weights: FloatArray = np.ones(target_rank, dtype=np.float64)
    factors: list[FloatArray] = []
    if init == "random":
        factors = [rng.standard_normal((int(n), target_rank)).astype(np.float64) for n in shape]
        return weights, factors
    # init == "source": warm-start with the leading R_target columns of the
    # source factors (after sorting by source weight magnitude). Padding with
    # random columns when target_rank > source_rank is supported.
    src_w = np.asarray(source.weights, dtype=np.float64)
    order = np.argsort(-np.abs(src_w))
    for fac in source.factors:
        cols = fac[:, order]
        if cols.shape[1] >= target_rank:
            factors.append(np.ascontiguousarray(cols[:, :target_rank], dtype=np.float64))
        else:
            extra = rng.standard_normal((cols.shape[0], target_rank - cols.shape[1])).astype(np.float64)
            factors.append(np.ascontiguousarray(np.concatenate([cols, extra], axis=1), dtype=np.float64))
    return weights, factors


def _mttkrp_cp_cp(
    source_weights: FloatArray,
    source_factors: Sequence[FloatArray],
    target_factors: Sequence[FloatArray],
    mode: int,
) -> FloatArray:
    r"""CP-vs-CP MTTKRP without densifying either tensor.

    Returns a matrix of shape :math:`(n_{mode}, R_{\text{tgt}})`. For each
    target column :math:`s`,

    .. math::

        M[i, s] \;=\; \sum_{r=1}^{R_{\text{src}}}\, \lambda^{(s)}_r\,
        A^{(s)}_n[i, r] \, \prod_{k \neq n}
        \big(A^{(s)}_k[:, r]^{\top} B_k[:, s]\big).

    Cost is :math:`O(n_n \cdot R_{\text{src}} \cdot R_{\text{tgt}}) +
    O(\sum_{k \neq n} n_k \cdot R_{\text{src}} \cdot R_{\text{tgt}})`,
    independent of :math:`\prod n_k`.
    """
    R_src = int(source_weights.size)
    R_tgt = int(target_factors[0].shape[1])
    # Cross-Gram per mode: shape (R_src, R_tgt).
    cross = np.ones((R_src, R_tgt), dtype=np.float64)
    for k, (A_k, B_k) in enumerate(zip(source_factors, target_factors, strict=True)):
        if k == mode:
            continue
        cross = cross * (A_k.T @ B_k)
    # Multiply by source weights along the source axis.
    weighted = source_weights[:, None] * cross  # (R_src, R_tgt)
    # Combine with the source factor in the kept mode.
    return source_factors[mode] @ weighted


def _cp_cp_loss(
    source_weights: FloatArray,
    source_factors: Sequence[FloatArray],
    target_weights: FloatArray,
    target_factors: Sequence[FloatArray],
) -> float:
    r"""Relative error :math:`\lVert X_S - X_T\rVert_F / \lVert X_S\rVert_F`
    via CP inner products only.
    """
    norm_s_sq = cp_norm_sq(source_weights, list(source_factors))
    if norm_s_sq == 0.0:
        return 0.0
    norm_t_sq = cp_norm_sq(target_weights, list(target_factors))
    inner = cp_inner_product(
        source_weights, list(source_factors),
        target_weights, list(target_factors),
    )
    err_sq = max(norm_s_sq - 2.0 * inner + norm_t_sq, 0.0)
    return float(np.sqrt(err_sq / norm_s_sq))


def _normalize_columns(
    factors: list[FloatArray],
) -> tuple[FloatArray, list[FloatArray]]:
    """Same routine as Task 9, kept local for visibility."""
    rank = int(factors[0].shape[1])
    weights = np.ones(rank, dtype=np.float64)
    out: list[FloatArray] = []
    for fac in factors:
        norms = np.linalg.norm(fac, axis=0)
        safe_norms = np.where(norms > 0.0, norms, 1.0)
        weights = weights * np.where(norms > 0.0, norms, 0.0)
        out.append(fac / safe_norms[None, :])
    return weights, out


def cp_als_from_cp(
    source: CPFactors,
    target_rank: int,
    *,
    init: InitStrategy = "random",
    max_iter: int = 200,
    tol: float | None = 1e-8,
    reg: float = 0.0,
    normalize: bool = True,
    random_seed: int | None = None,
) -> CpAlsCpResult:
    r"""ALS compression of a CP tensor to a smaller rank.

    Parameters
    ----------
    source
        Source CP carrier of rank :math:`R_{\text{src}}`.
    target_rank
        Target CP rank :math:`R_{\text{tgt}}` (typically
        :math:`< R_{\text{src}}`).
    init
        Initialisation strategy: ``"random"`` (Gaussian), or
        ``"source"`` (leading-:math:`R_{\text{tgt}}` columns of the
        source factors, sorted by weight magnitude).
    max_iter
        Maximum ALS sweeps.
    tol
        Stop when the relative change in the loss falls below this. Pass
        ``None`` to run for ``max_iter`` sweeps.
    reg
        Tikhonov regularisation added to the normal equations.
    normalize
        If ``True``, normalise factor columns after every sweep.
    random_seed
        Seed for the initial factors and any random padding.

    Returns
    -------
    CpAlsCpResult
        Compressed CP factors plus the loss history (each entry computed
        by :func:`cp_cp_relative_error` — never densifies either tensor).
    """
    if target_rank < 1:
        raise ValueError("target_rank must be at least 1")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    rng = make_rng(random_seed)

    weights, factors = _init_target(source, target_rank, init, rng)
    losses: list[float] = []
    converged = False
    prev_loss: float | None = None

    for _ in range(max_iter):
        for mode in range(source.ndim):
            if mode == 0 and normalize:
                factors[0] = factors[0] * weights[None, :]
                weights = np.ones(target_rank, dtype=np.float64)
            # Normal-equation matrix V = ⊙_{k != mode} (B_k^T B_k) + reg I
            grams = cp_factor_grams(factors)
            V = np.ones((target_rank, target_rank), dtype=np.float64)
            for k, G in enumerate(grams):
                if k == mode:
                    continue
                V = V * G
            if reg > 0.0:
                V = V + reg * np.eye(target_rank, dtype=np.float64)
            rhs = _mttkrp_cp_cp(source.weights, source.factors, factors, mode)
            try:
                L = np.linalg.cholesky(V)
                factors[mode] = np.linalg.solve(L.T, np.linalg.solve(L, rhs.T)).T
            except np.linalg.LinAlgError:
                factors[mode] = rhs @ safe_pinv(V, warn=False)
        if normalize:
            weights, factors = _normalize_columns(factors)
        loss = _cp_cp_loss(source.weights, source.factors, weights, factors)
        losses.append(loss)
        if tol is not None and prev_loss is not None and abs(prev_loss - loss) < tol:
            converged = True
            break
        prev_loss = loss

    cp = CPFactors(
        weights=np.ascontiguousarray(weights, dtype=np.float64),
        factors=tuple(np.ascontiguousarray(f, dtype=np.float64) for f in factors),
    )
    return CpAlsCpResult(
        factors=cp,
        loss_history=np.array(losses, dtype=np.float64),
        iterations=len(losses),
        converged=converged,
    )


__all__ = ["CpAlsCpResult", "InitStrategy", "cp_als_from_cp"]
