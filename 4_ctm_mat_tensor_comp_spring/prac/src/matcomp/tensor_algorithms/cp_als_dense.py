r"""Task 9 — Classical CP-ALS for a dense tensor.

Implements alternating least squares for the canonical / CP
decomposition

.. math::

    X \;\approx\; \sum_{r=1}^{R}\, \lambda_r \, A_1[:, r] \circ \cdots
    \circ A_d[:, r],

following Kolda & Bader (*SIAM Review* 51(3), 2009, §3) and Tomasi &
Bro (*Comp. Stat. Data Anal.* 50(7), 2006). Each ALS sweep updates one
factor matrix at a time by solving the normal equations

.. math::

    A_n \, \Big(\bigodot_{k \neq n} A_k^{\top} A_k\Big)
    \;=\; \mathrm{MTTKRP}_n(X, \{A_k\}),

where :math:`\bigodot` denotes the element-wise (Hadamard) product of
the per-mode Gram matrices and :math:`\mathrm{MTTKRP}_n` is the
matricised tensor times Khatri–Rao product. Initialisation is either
random or by the leading-:math:`R` left singular vectors of each mode-n
unfolding (computed via :func:`randomized_svd`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from matcomp.matrix_algorithms.randomized_svd import randomized_svd
from matcomp.utils.functional_matrix import FloatArray
from matcomp.utils.linalg import safe_pinv
from matcomp.utils.seeding import make_rng
from matcomp.utils.tensor_linalg import (
    cp_factor_grams,
    cp_norm_sq,
    mttkrp,
    unfold,
)
from matcomp.utils.tensor_low_rank import CPFactors
from matcomp.utils.test_matrices import DenseMatrix

InitStrategy = Literal["random", "svd"]


@dataclass(frozen=True)
class CpAlsResult:
    r"""Output of :func:`cp_als`.

    Attributes
    ----------
    factors
        :class:`CPFactors` carrier with the requested rank ``R``.
    loss_history
        Length-``iterations`` array of relative Frobenius errors after
        each completed sweep.
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
        """CP rank :math:`R`."""
        return self.factors.rank

    @property
    def shape(self) -> tuple[int, ...]:
        """Tensor shape."""
        return self.factors.shape

    @property
    def ndim(self) -> int:
        """Number of modes."""
        return self.factors.ndim

    def entry(self, idx: tuple[int, ...]) -> float:
        return self.factors.entry(idx)

    def reconstruct_small(self) -> FloatArray:
        return self.factors.reconstruct_small()

    def factors_memory(self) -> int:
        return self.factors.factors_memory()


def _init_factors(
    X: FloatArray,
    rank: int,
    init: InitStrategy,
    rng: np.random.Generator,
) -> list[FloatArray]:
    """Initial factor matrices.

    ``"random"`` draws Gaussian entries; ``"svd"`` uses the leading-``R``
    left singular vectors of each mode-n unfolding (PDF p. 9 acceptance
    criterion: SVD init helps on ill-conditioned tensors). When the
    unfolding has fewer than ``R`` rows or columns, the SVD result is
    padded with random columns.
    """
    if init == "random":
        return [rng.standard_normal((int(s), rank)).astype(np.float64) for s in X.shape]
    factors: list[FloatArray] = []
    for k in range(X.ndim):
        Xk = unfold(X, k)
        m, n = Xk.shape
        if min(m, n) < rank:
            # SVD can return at most min(m, n) singular vectors; pad the
            # remainder with Gaussians so the column count matches rank.
            U_full, _, _ = np.linalg.svd(Xk, full_matrices=False)
            keep = U_full.shape[1]
            extra = rng.standard_normal((m, rank - keep)).astype(np.float64)
            factors.append(np.concatenate([U_full[:, :keep], extra], axis=1))
        else:
            wrap = DenseMatrix(np.ascontiguousarray(Xk, dtype=np.float64))
            r = randomized_svd(wrap, r=rank, oversampling=min(10, m - rank), random_seed=int(rng.integers(0, 2**31 - 1)))
            factors.append(np.ascontiguousarray(r.factors.U, dtype=np.float64))
    return factors


def _als_step(
    X: FloatArray,
    factors: list[FloatArray],
    mode: int,
    reg: float,
) -> FloatArray:
    r"""Solve the CP-ALS normal equation for a single factor.

    Returns the updated :math:`A_n` such that

    .. math::

        A_n \, V \;=\; \mathrm{MTTKRP}_n(X, \{A_k\}),
        \qquad V = \bigodot_{k \neq n} A_k^{\top} A_k + \mathrm{reg}\,I.

    Parameters
    ----------
    X
        Dense tensor.
    factors
        Current factors. ``factors[mode]`` is the target.
    mode
        Mode being updated.
    reg
        Tikhonov regularisation added to the diagonal of :math:`V`.

    Returns
    -------
    numpy.ndarray
        Updated factor of shape ``(X.shape[mode], R)``.
    """
    rank = int(factors[0].shape[1])
    grams = cp_factor_grams(factors)
    V = np.ones((rank, rank), dtype=np.float64)
    for k, G in enumerate(grams):
        if k == mode:
            continue
        V = V * G
    if reg > 0.0:
        V = V + reg * np.eye(rank, dtype=np.float64)
    rhs = mttkrp(X, factors, mode)
    # Solve A_n V = rhs. Use Cholesky if SPD, fall back to safe_pinv otherwise.
    try:
        L = np.linalg.cholesky(V)
        new_factor = np.linalg.solve(L.T, np.linalg.solve(L, rhs.T)).T
    except np.linalg.LinAlgError:
        new_factor = rhs @ safe_pinv(V, warn=False)
    return np.ascontiguousarray(new_factor, dtype=np.float64)


def _normalize_columns(
    factors: list[FloatArray],
) -> tuple[FloatArray, list[FloatArray]]:
    r"""Normalise factor columns to unit norm and return the absorbed weights.

    ``λ_r = ∏_k \lVert A_k[:, r] \rVert_2``; each factor column is then
    divided by its norm. PDF acceptance: "Factors after iterations must
    be normalised or their scale must be controlled" (p. 9).
    """
    rank = int(factors[0].shape[1])
    weights = np.ones(rank, dtype=np.float64)
    out: list[FloatArray] = []
    for fac in factors:
        norms = np.linalg.norm(fac, axis=0)
        # Keep zero columns as-is to avoid div-by-zero; their weight stays 0.
        safe_norms = np.where(norms > 0.0, norms, 1.0)
        weights = weights * np.where(norms > 0.0, norms, 0.0)
        out.append(fac / safe_norms[None, :])
    return weights, out


def _loss(X: FloatArray, weights: FloatArray, factors: Sequence[FloatArray]) -> float:
    r"""Relative Frobenius error :math:`\lVert X - X_{CP}\rVert_F /
    \lVert X\rVert_F` without densifying CP.
    """
    norm_X_sq = float(np.sum(X * X))
    if norm_X_sq == 0.0:
        return 0.0
    norm_cp_sq = cp_norm_sq(weights, list(factors))
    M0 = mttkrp(X, list(factors), 0)
    inner_per_r = np.sum(M0 * factors[0], axis=0)
    inner = float(weights @ inner_per_r)
    err_sq = max(norm_X_sq - 2.0 * inner + norm_cp_sq, 0.0)
    return float(np.sqrt(err_sq / norm_X_sq))


def cp_als(
    X: FloatArray,
    rank: int,
    *,
    init: InitStrategy = "svd",
    max_iter: int = 500,
    tol: float = 1e-7,
    reg: float = 0.0,
    normalize: bool = True,
    random_seed: int | None = None,
) -> CpAlsResult:
    r"""Classical CP-ALS for a dense tensor.

    Parameters
    ----------
    X
        Dense tensor of shape :math:`(n_1, \ldots, n_d)`.
    rank
        Target CP rank :math:`R \ge 1`.
    init
        ``"random"`` (Gaussian) or ``"svd"`` (leading singular vectors of
        each mode-n unfolding).
    max_iter
        Maximum number of ALS sweeps.
    tol
        Stop when the relative change in the loss falls below this.
    reg
        Tikhonov regularisation added to the normal equations.
    normalize
        If ``True``, normalise factor columns to unit norm and absorb the
        scale into the returned weights after every sweep.
    random_seed
        Seed for the initial factors.

    Returns
    -------
    CpAlsResult
        CP factors, loss history, iteration count and a converged flag.
    """
    if X.ndim < 2:
        raise ValueError("cp_als requires a tensor with at least 2 modes")
    if rank < 1:
        raise ValueError("rank must be at least 1")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    rng = make_rng(random_seed)

    factors: list[FloatArray] = _init_factors(X, rank, init, rng)
    weights: FloatArray = np.ones(rank, dtype=np.float64)
    losses: list[float] = []
    converged = False
    prev_loss: float | None = None

    for _ in range(max_iter):
        for mode in range(X.ndim):
            # Re-fold the absorbed weights into mode 0 before every sweep
            # so the normal equations work in unweighted form.
            if mode == 0 and normalize:
                factors[0] = factors[0] * weights[None, :]
                weights = np.ones(rank, dtype=np.float64)
            factors[mode] = _als_step(X, factors, mode, reg)
        if normalize:
            weights, factors = _normalize_columns(factors)
        loss = _loss(X, weights, factors)
        losses.append(loss)
        if prev_loss is not None and abs(prev_loss - loss) < tol:
            converged = True
            break
        prev_loss = loss

    cp = CPFactors(
        weights=np.ascontiguousarray(weights, dtype=np.float64),
        factors=tuple(np.ascontiguousarray(f, dtype=np.float64) for f in factors),
    )
    return CpAlsResult(
        factors=cp,
        loss_history=np.array(losses, dtype=np.float64),
        iterations=len(losses),
        converged=converged,
    )


__all__ = ["CpAlsResult", "InitStrategy", "_als_step", "_loss", "_normalize_columns", "cp_als"]
