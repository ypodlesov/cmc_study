r"""Task 11 — Levenberg–Marquardt for CP decomposition.

Treats CP fitting as nonlinear least squares: with parameter vector
:math:`\theta = (\lambda; B_1; \ldots; B_d)` packed by
:func:`pack_cp_theta`, we minimise

.. math::

    \mathcal{L}(\theta) \;=\; \tfrac{1}{2}\,\lVert r(\theta)\rVert_2^{2},
    \qquad r(\theta) = \mathrm{vec}\big(X - X_{CP}(\theta)\big).

At each iteration LM solves the regularised normal equations
:math:`(J^{\top}J + \mu I)\,\delta = J^{\top} r` with the Jacobian of
:math:`r` taken in closed form (every column corresponds to a single
parameter — for :math:`B_n[i, r]` the column is
:math:`-\lambda_r \cdot \mathbf{e}_i^{(n)} \otimes \prod_{k \neq n}
B_k[:, r]`; for :math:`\lambda_r` it is the ranked outer product). The
step is accepted when it decreases the loss; otherwise :math:`\mu` is
multiplied by ``mu_inc``. Accepted steps multiply :math:`\mu` by
``mu_dec`` (Marquardt's update rule).

Following Sorber, Van Barel and De Lathauwer (*SIAM J. Optim.* 23(2),
2013), an ALS warm-up is run first because LM converges much faster
once the swamp is escaped.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np

from matcomp.tensor_algorithms.cp_als_dense import _als_step, _normalize_columns
from matcomp.utils.functional_matrix import FloatArray
from matcomp.utils.linalg import safe_pinv
from matcomp.utils.seeding import make_rng
from matcomp.utils.tensor_linalg import (
    khatri_rao,
    pack_cp_theta,
    unpack_cp_theta,
)
from matcomp.utils.tensor_low_rank import CPFactors

InitStrategy = Literal["random", "svd", "als"]


@dataclass(frozen=True)
class CpLmResult:
    r"""Output of :func:`cp_levenberg_marquardt`.

    Attributes
    ----------
    factors
        Final CP carrier.
    loss_history
        Length-``iterations`` array of squared-residual values
        :math:`\tfrac{1}{2}\lVert r(\theta)\rVert^{2}` at the end of
        each *accepted* step (rejected steps are not recorded).
    mu_history
        Damping history matching ``loss_history``.
    iterations
        Total number of *accepted* steps.
    accepted_steps
        Same as ``iterations`` (kept for clarity in reports).
    rejected_steps
        Number of rejected proposals (dampening was increased).
    jacobian_shape
        Shape of the Jacobian matrix.
    converged
        ``True`` iff the loop stopped because the loss change fell
        below ``tol``.
    """

    factors: CPFactors
    loss_history: FloatArray
    mu_history: FloatArray
    iterations: int
    accepted_steps: int
    rejected_steps: int
    jacobian_shape: tuple[int, int]
    converged: bool

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


def _residual_vec(theta: FloatArray, X: FloatArray, rank: int) -> FloatArray:
    r"""Vectorised residual :math:`r(\theta) = \mathrm{vec}(X) -
    \mathrm{vec}(X_{CP}(\theta))`.

    Uses the mode-0 unfolding of :math:`X_{CP} = \sum_r \lambda_r
    B_0[:, r]\,(\bigodot_{k > 0} B_k[:, r]^{\top})` for assembly,
    yielding the same flattening order as ``X.ravel()`` (numpy C-order).
    """
    weights, factors = unpack_cp_theta(theta, X.shape, rank)
    KR = khatri_rao(factors, skip=0, reverse=False)  # (n_1*...*n_{d-1}, R)
    cp_unfold = (factors[0] * weights[None, :]) @ KR.T  # (n_0, n_1*...*n_{d-1})
    return X.ravel() - cp_unfold.ravel()


def _jacobian(
    theta: FloatArray, shape: tuple[int, ...], rank: int
) -> FloatArray:
    r"""Closed-form Jacobian of :math:`-r(\theta)` w.r.t. :math:`\theta`.

    The rows of :math:`J` are indexed by the flattened entries of
    :math:`X` (C-order); columns by the flattened parameters in the
    layout produced by :func:`pack_cp_theta`.

    Block structure (each column of the ``r``-th rank-1 component):

    * weight column :math:`\partial r / \partial \lambda_r =
      -\bigotimes_k B_k[:, r]`.
    * factor column :math:`\partial r / \partial B_n[i, r] =
      -\lambda_r\,\mathbf{e}_i^{(n)} \otimes \bigotimes_{k \neq n} B_k[:, r]`.

    For implementation simplicity we use the flat layout that matches
    :func:`numpy.reshape` C-order; the ordering of factor blocks matches
    :func:`pack_cp_theta`.
    """
    weights, factors = unpack_cp_theta(theta, shape, rank)
    d = len(shape)
    N = int(np.prod(shape))
    n_params = rank + sum(int(s) * rank for s in shape)
    J = np.zeros((N, n_params), dtype=np.float64)

    # Pre-compute per-rank unfolded outer products via repeated mode unfolding.
    # rank_outer[r] = ⊗_{k=0..d-1} B_k[:, r] flattened in C-order: shape (N,).
    # This equals reshape(reduce-broadcast over modes, -1).
    rank_outer = np.empty((rank, N), dtype=np.float64)
    for r in range(rank):
        out = np.array(1.0, dtype=np.float64)
        for k, B_k in enumerate(factors):
            shp = [1] * d
            shp[k] = int(shape[k])
            out = out * B_k[:, r].reshape(tuple(shp))
        rank_outer[r] = out.ravel()

    # Weight columns: ∂r/∂λ_r = -rank_outer[r].
    J[:, :rank] = -rank_outer.T  # (N, rank)

    # Factor columns. For B_n[i, r]: gradient is -λ_r times rank_outer with
    # the n-th index pinned to i and B_n[i, r] removed (i.e. divided out).
    cursor = rank
    for n in range(d):
        n_n = int(shape[n])
        # We need an N-array equal to (rank_outer[r] / B_n[i_n, r]) when i_n == i, else 0.
        # Reshape rank_outer to expose the mode-n axis explicitly.
        block = np.zeros((N, n_n * rank), dtype=np.float64)
        for r in range(rank):
            grid = rank_outer[r].reshape(shape)
            # Slice along mode n: for each i, the contribution is the slab
            # divided by B_n[i, r] (or recomputed when B_n[i, r] == 0).
            B_col = factors[n][:, r]
            for i in range(n_n):
                slab = np.take(grid, i, axis=n)  # shape: shape without mode n
                if abs(B_col[i]) > 0.0:
                    contrib = (slab / B_col[i]).ravel()
                else:
                    # Recompute the slab as ⊗_{k != n} B_k[:, r] flattened,
                    # then place it at the mode-n slice == i (mode-n indices
                    # are dropped; broadcast the slab back into the full
                    # tensor with a delta at i).
                    contrib_slab = np.array(1.0, dtype=np.float64)
                    sub_shape = [int(shape[k]) for k in range(d) if k != n]
                    for kk_axis, kk in enumerate([k for k in range(d) if k != n]):
                        shp = [1] * len(sub_shape)
                        shp[kk_axis] = int(shape[kk])
                        contrib_slab = contrib_slab * factors[kk][:, r].reshape(tuple(shp))
                    contrib = contrib_slab.ravel()
                # Build the full N-vector with the slab placed at i along mode n.
                # Reconstruct full-size indexing: insert axis of size n_n filled with delta at i.
                slab_reshaped = contrib.reshape([s for k, s in enumerate(shape) if k != n])
                full_grid = np.zeros(shape, dtype=np.float64)
                idxer: list[slice | int] = [slice(None)] * d
                idxer[n] = int(i)
                full_grid[tuple(idxer)] = slab_reshaped
                full = full_grid.ravel()
                # Column index inside this factor block: B_n is stored in
                # row-major (n_n, rank) order; index = i * rank + r.
                block[:, i * rank + r] = -weights[r] * full
        J[:, cursor : cursor + n_n * rank] = block
        cursor += n_n * rank

    return J


def _lm_step(
    J: FloatArray,
    r: FloatArray,
    mu: float,
) -> FloatArray:
    r"""Solve :math:`(J^{\top} J + \mu I)\,\delta = -J^{\top} r`.

    The Gauss–Newton descent direction for :math:`\mathcal{L} =
    \tfrac{1}{2}\lVert r(\theta)\rVert^{2}` is :math:`\delta =
    -(J^{\top} J + \mu I)^{-1}\,J^{\top} r`. Cholesky is tried first;
    on failure (the regularised matrix is poorly conditioned, e.g.
    when :math:`J` is rank deficient and :math:`\mu` is tiny), we fall
    back to :func:`safe_pinv` and emit a :class:`RuntimeWarning`.
    """
    JT_J = J.T @ J
    n = JT_J.shape[0]
    A = JT_J + mu * np.eye(n, dtype=np.float64)
    rhs = -J.T @ r
    try:
        L = np.linalg.cholesky(A)
        return np.linalg.solve(L.T, np.linalg.solve(L, rhs))
    except np.linalg.LinAlgError:
        warnings.warn(
            "cp_lm: Cholesky failed on (J^T J + mu I); falling back to safe_pinv. "
            "Increase mu_init or warm up with more ALS sweeps for better conditioning.",
            RuntimeWarning,
            stacklevel=2,
        )
        return safe_pinv(A, warn=False) @ rhs


def cp_levenberg_marquardt(
    X: FloatArray,
    rank: int,
    *,
    init: InitStrategy = "als",
    als_warmup: int = 5,
    max_iter: int = 200,
    mu_init: float = 1e-3,
    mu_inc: float = 10.0,
    mu_dec: float = 0.3,
    mu_min: float = 1e-12,
    mu_max: float = 1e12,
    tol: float | None = 1e-8,
    random_seed: int | None = None,
) -> CpLmResult:
    r"""CP fitting by the Levenberg–Marquardt algorithm.

    Parameters
    ----------
    X
        Dense input tensor.
    rank
        CP rank.
    init
        ``"random"`` (Gaussian), ``"svd"`` (leading singular vectors via
        :mod:`matcomp.tensor_algorithms.cp_als_dense._init_factors`), or
        ``"als"`` (warm-start with ``als_warmup`` ALS sweeps).
    als_warmup
        Number of ALS sweeps to run before LM. Ignored when
        ``init != "als"``.
    max_iter
        Maximum LM iterations (counts both accepted and rejected steps).
    mu_init, mu_inc, mu_dec, mu_min, mu_max
        Damping hyperparameters. ``mu`` is increased by ``mu_inc`` after
        a rejection and decreased by ``mu_dec`` after acceptance. The
        damping is clamped to ``[mu_min, mu_max]``.
    tol
        Stop when the relative change in the loss after acceptance falls
        below this.
    random_seed
        Seed for any randomised init.
    """
    if X.ndim < 2:
        raise ValueError("cp_levenberg_marquardt requires ndim >= 2")
    if rank < 1:
        raise ValueError("rank must be at least 1")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    rng = make_rng(random_seed)

    # ---- initialisation ------------------------------------------------------
    from matcomp.tensor_algorithms.cp_als_dense import _init_factors

    factors_list: list[FloatArray]
    if init == "random":
        factors_list = [
            rng.standard_normal((int(s), rank)).astype(np.float64) for s in X.shape
        ]
    elif init == "svd":
        factors_list = _init_factors(X, rank, "svd", rng)
    elif init == "als":
        factors_list = _init_factors(X, rank, "svd", rng)
        for _ in range(als_warmup):
            for mode in range(X.ndim):
                factors_list[mode] = _als_step(X, factors_list, mode, reg=0.0)
    else:
        raise ValueError(f"unknown init {init!r}")

    weights, factors_list = _normalize_columns(factors_list)
    theta = pack_cp_theta(weights, factors_list)
    n_params = theta.size

    # ---- LM loop -------------------------------------------------------------
    r_vec = _residual_vec(theta, X, rank)
    loss = 0.5 * float(r_vec @ r_vec)
    mu = float(mu_init)
    losses = [loss]
    mus = [mu]
    accepted = 0
    rejected = 0
    converged = False
    prev_loss = loss

    jac_shape = (X.size, n_params)

    for _ in range(max_iter):
        J = _jacobian(theta, X.shape, rank)
        delta = _lm_step(J, r_vec, mu)
        theta_new = theta + delta
        r_new = _residual_vec(theta_new, X, rank)
        loss_new = 0.5 * float(r_new @ r_new)
        if loss_new < loss:
            theta = theta_new
            r_vec = r_new
            loss = loss_new
            mu = max(mu * mu_dec, mu_min)
            accepted += 1
            losses.append(loss)
            mus.append(mu)
            if tol is not None and abs(prev_loss - loss) < tol:
                converged = True
                break
            prev_loss = loss
        else:
            mu = min(mu * mu_inc, mu_max)
            rejected += 1
            if mu >= mu_max:
                # Cannot make progress; stop.
                break

    # ---- assemble factors ---------------------------------------------------
    weights, factors_list = unpack_cp_theta(theta, X.shape, rank)
    cp = CPFactors(
        weights=np.ascontiguousarray(weights, dtype=np.float64),
        factors=tuple(np.ascontiguousarray(f, dtype=np.float64) for f in factors_list),
    )
    return CpLmResult(
        factors=cp,
        loss_history=np.array(losses, dtype=np.float64),
        mu_history=np.array(mus, dtype=np.float64),
        iterations=accepted,
        accepted_steps=accepted,
        rejected_steps=rejected,
        jacobian_shape=jac_shape,
        converged=converged,
    )


__all__ = ["CpLmResult", "InitStrategy", "cp_levenberg_marquardt"]
