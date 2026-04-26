r"""Task 10 — CP via differentiable optimisation.

Treats the CP decomposition as a differentiable optimisation problem:
factor matrices :math:`B_k` and the weight vector :math:`\lambda` are
trainable parameters; the reconstruction error

.. math::

    \mathcal{L}(\theta) \;=\;
    \sum_{(i_1, \ldots, i_d) \in \Omega}
    \Big(X[i_1, \ldots, i_d] - \sum_{r} \lambda_r \prod_k B_k[i_k, r]\Big)^{2}

is minimised by gradient methods. Two evaluation modes:

* ``"dense"`` — :math:`\Omega` is the full index grid, :math:`X` is a
  dense ``ndarray``. We compute the gradient analytically via
  :func:`mttkrp` (this is the same closed-form ALS uses, just with a
  step that is *not* the Newton step).
* ``"batched"`` — :math:`\Omega` is a random sample drawn each step;
  :math:`X` is a :class:`FunctionalTensor` queried via ``samples``.
  The gradient on the mini-batch is :math:`O(b\,d\,R)` per step,
  independent of :math:`\prod n_k`.

Optimisers: ``"adam"`` (default) and ``"sgd"`` are hand-rolled in NumPy
— there is no autograd dependency. The PDF describes PyTorch / JAX as
"convenient", not required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from matcomp.utils.counting import CountingFunctionalTensor, TensorOracleCounts
from matcomp.utils.functional_matrix import FloatArray, IntArray
from matcomp.utils.functional_tensor import FunctionalTensor
from matcomp.utils.seeding import make_rng
from matcomp.utils.tensor_linalg import (
    cp_norm_sq,
    mttkrp,
)
from matcomp.utils.tensor_low_rank import CPFactors

InitStrategy = Literal["random", "svd"]
OptimizerName = Literal["adam", "sgd"]
RunMode = Literal["dense", "batched"]


@dataclass(frozen=True)
class CpNeuralResult:
    r"""Output of :func:`fit_cp_neural`.

    Attributes
    ----------
    factors
        Final CP carrier.
    loss_history
        Length-``iterations`` array of loss values (per epoch in
        ``"dense"`` mode, per mini-batch in ``"batched"`` mode).
    iterations
        Number of completed gradient steps.
    converged
        ``True`` iff the loop stopped because the loss change fell
        below ``tol``.
    oracle_counts
        Counts of the wrapped functional-tensor oracle calls. ``None``
        when the input was a dense ``ndarray``.
    """

    factors: CPFactors
    loss_history: FloatArray
    iterations: int
    converged: bool
    oracle_counts: TensorOracleCounts | None

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


class CPModel:
    r"""Trainable CP factorisation with closed-form gradients.

    Holds :math:`\lambda` and ``d`` factor matrices :math:`B_k` as
    NumPy arrays. Provides ``predict``, ``loss_dense`` /
    ``loss_batch`` and ``grad_dense`` / ``grad_batch`` so callers can
    plug in any first-order optimiser (Adam / SGD are bundled).

    Parameters
    ----------
    shape
        Tensor shape.
    rank
        CP rank :math:`R`.
    init
        ``"random"`` (Gaussian, scaled by :math:`1 / \sqrt{n_k}`) or
        ``"svd"`` (leading singular vectors of mode-n unfoldings of a
        provided dense tensor — the reference is required as ``X_ref``
        for the SVD path).
    X_ref
        Optional reference dense tensor. Required for ``init="svd"``.
    random_seed
        Seed for the initial parameters.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        rank: int,
        *,
        init: InitStrategy = "random",
        X_ref: FloatArray | None = None,
        random_seed: int | None = None,
    ) -> None:
        if rank < 1:
            raise ValueError("rank must be at least 1")
        rng = make_rng(random_seed)
        self.shape = tuple(int(n) for n in shape)
        self.rank = int(rank)
        if init == "random":
            self.factors: list[FloatArray] = [
                (rng.standard_normal((int(n), self.rank)) / np.sqrt(int(n))).astype(np.float64)
                for n in self.shape
            ]
        elif init == "svd":
            if X_ref is None:
                raise ValueError("init='svd' requires X_ref")
            from matcomp.tensor_algorithms.cp_als_dense import _init_factors

            self.factors = _init_factors(X_ref, self.rank, "svd", rng)
        else:
            raise ValueError(f"unknown init {init!r}")
        self.weights: FloatArray = np.ones(self.rank, dtype=np.float64)

    # ---- forward ---------------------------------------------------------

    def predict(self, idx: IntArray) -> FloatArray:
        r"""Vectorised prediction at the given multi-indices.

        Parameters
        ----------
        idx
            Integer array of shape ``(N, ndim)``.

        Returns
        -------
        numpy.ndarray
            Length-``N`` prediction :math:`\hat X[\mathrm{idx}_n] =
            \sum_r \lambda_r \prod_k B_k[\mathrm{idx}_n[k], r]`.
        """
        idx = np.asarray(idx, dtype=np.intp)
        # Hadamard-reduce along modes; final dot with weights.
        prod = np.ones((idx.shape[0], self.rank), dtype=np.float64)
        for k, B in enumerate(self.factors):
            prod = prod * B[idx[:, k]]
        return prod @ self.weights

    # ---- loss + gradient -------------------------------------------------

    def loss_dense(self, X: FloatArray) -> float:
        r"""MSE :math:`\frac{1}{N} \sum (X - \hat X)^2` over the full grid."""
        if X.shape != self.shape:
            raise ValueError(f"loss_dense: shape mismatch X {X.shape} vs model {self.shape}")
        # Use ||X - X_CP||^2 expansion: O(prod(n_k) R) instead of materialising X_CP.
        norm_X_sq = float(np.sum(X * X))
        norm_cp_sq = cp_norm_sq(self.weights, self.factors)
        M0 = mttkrp(X, self.factors, 0)
        inner_per_r = np.sum(M0 * self.factors[0], axis=0)
        inner = float(self.weights @ inner_per_r)
        err_sq = max(norm_X_sq - 2.0 * inner + norm_cp_sq, 0.0)
        return float(err_sq / float(np.prod(self.shape)))

    def grad_dense(
        self, X: FloatArray
    ) -> tuple[FloatArray, list[FloatArray]]:
        r"""Closed-form gradients of :meth:`loss_dense`.

        Per-factor (derived from
        :math:`\partial \lVert X - X_{CP}\rVert^{2} / \partial B_n[i, r]
        = -2\,\lambda_r\,\big(M_n - B_n\,\mathrm{diag}(\lambda)\,V_n\big)[i, r]`):

        .. math::

            \nabla_{B_n}\mathcal{L} \;=\; \frac{2}{N}\,
            \big(B_n\,\mathrm{diag}(\lambda)\,V_n - M_n\big)\,
            \mathrm{diag}(\lambda),
            \qquad
            V_n = \bigodot_{k \neq n} (B_k^{\top} B_k),
            \quad
            M_n = \mathrm{MTTKRP}_n(X, \{B_k\}).

        Weight gradient:

        .. math::

            \nabla_{\lambda}\mathcal{L} = \tfrac{2}{N}\,(W\,\lambda - p),
            \qquad W = \bigodot_k (B_k^{\top} B_k),\quad
            p_r = \sum_i M_0[i, r]\,B_0[i, r].
        """
        if X.shape != self.shape:
            raise ValueError("grad_dense: shape mismatch")
        N = float(np.prod(self.shape))
        grams = [B.T @ B for B in self.factors]
        scale = 2.0 / N
        weights = self.weights

        grad_factors: list[FloatArray] = []
        for n in range(len(self.shape)):
            V_n = np.ones((self.rank, self.rank), dtype=np.float64)
            for k, G in enumerate(grams):
                if k == n:
                    continue
                V_n = V_n * G
            M_n = mttkrp(X, self.factors, n)
            # B_n diag(λ) V_n diag(λ): apply λ on the right of B_n, then V_n,
            # then λ again element-wise.
            term1 = ((self.factors[n] * weights[None, :]) @ V_n) * weights[None, :]
            term2 = M_n * weights[None, :]
            grad = scale * (term1 - term2)
            grad_factors.append(np.ascontiguousarray(grad, dtype=np.float64))

        W = np.ones((self.rank, self.rank), dtype=np.float64)
        for G in grams:
            W = W * G
        M_0 = mttkrp(X, self.factors, 0)
        p = np.sum(M_0 * self.factors[0], axis=0)
        grad_weights = scale * (W @ weights - p)
        return np.ascontiguousarray(grad_weights, dtype=np.float64), grad_factors

    def loss_batch(self, idx: IntArray, values: FloatArray) -> float:
        """Mean squared error on a sampled mini-batch."""
        preds = self.predict(idx)
        diff = preds - values
        return float(np.mean(diff * diff))

    def grad_batch(
        self, idx: IntArray, values: FloatArray
    ) -> tuple[FloatArray, list[FloatArray]]:
        r"""Mini-batch gradient of :meth:`loss_batch`.

        For each rank :math:`r`, ``per_rank[n, r] = \prod_k B_k[idx_n[k], r]``.
        Then the residual ``e_n = sum_r λ_r per_rank[n, r] - values[n]``
        feeds into per-factor and per-weight gradients in :math:`O(b d R)`.
        """
        idx = np.asarray(idx, dtype=np.intp)
        b = idx.shape[0]
        per_rank = np.ones((b, self.rank), dtype=np.float64)
        # Cache the per-mode rows for reuse in the gradients.
        rows = [self.factors[k][idx[:, k]] for k in range(len(self.shape))]  # each (b, R)
        for r in rows:
            per_rank = per_rank * r
        weights = self.weights
        residuals = (per_rank @ weights) - values  # shape (b,)

        scale = 2.0 / float(b)
        grad_factors: list[FloatArray] = []
        # ∂L/∂B_n[i, r] = scale * sum_{n: idx_n[mode] == i} residual_n * λ_r * ∏_{k ≠ n} B_k[idx_n[k], r]
        for n in range(len(self.shape)):
            # ∏_{k ≠ n} term per row = per_rank / rows[n] (with safety for zeros).
            denom = rows[n]
            with np.errstate(divide="ignore", invalid="ignore"):
                others = np.where(np.abs(denom) > 0.0, per_rank / denom, 0.0)
            # If denom has zeros, recompute that row's "others" by hand: product over k != n.
            zero_mask = np.any(np.abs(denom) <= 0.0, axis=1)
            if np.any(zero_mask):
                idx_zero = np.where(zero_mask)[0]
                for nn in idx_zero:
                    prod = np.ones(self.rank, dtype=np.float64)
                    for k in range(len(self.shape)):
                        if k == n:
                            continue
                        prod = prod * self.factors[k][idx[nn, k]]
                    others[nn] = prod
            others_weighted = others * weights[None, :]  # (b, R)
            grad_n = np.zeros_like(self.factors[n])
            # Scatter-add residual_n * scale * λ * others to row idx[n, mode_n].
            np.add.at(
                grad_n,
                idx[:, n],
                scale * residuals[:, None] * others_weighted,
            )
            grad_factors.append(grad_n)

        # Weight gradient: ∂L/∂λ_r = scale * sum_n residual_n * per_rank[n, r]
        grad_weights = scale * (per_rank.T @ residuals)
        return np.ascontiguousarray(grad_weights, dtype=np.float64), grad_factors

    # ---- carrier ---------------------------------------------------------

    def to_cp_factors(self) -> CPFactors:
        """Return the current parameters as an immutable :class:`CPFactors`."""
        return CPFactors(
            weights=np.ascontiguousarray(self.weights.copy(), dtype=np.float64),
            factors=tuple(np.ascontiguousarray(B.copy(), dtype=np.float64) for B in self.factors),
        )


def _adam_step(
    theta_grad: list[FloatArray],
    state_m: list[FloatArray],
    state_v: list[FloatArray],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    step: int,
) -> list[FloatArray]:
    r"""Adam (Kingma & Ba, 2015) update for a list of parameter tensors.

    Mutates ``state_m`` and ``state_v`` in-place; returns the proposed
    deltas (pre-bias-correction is folded in).
    """
    deltas: list[FloatArray] = []
    for g, m, v in zip(theta_grad, state_m, state_v, strict=True):
        m *= beta1
        m += (1.0 - beta1) * g
        v *= beta2
        v += (1.0 - beta2) * (g * g)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        deltas.append(lr * m_hat / (np.sqrt(v_hat) + eps))
    return deltas


def fit_cp_neural(
    target: FunctionalTensor | FloatArray,
    rank: int,
    *,
    mode: RunMode = "dense",
    batch_size: int = 4096,
    max_iter: int = 5000,
    optimizer: OptimizerName = "adam",
    lr: float = 1e-2,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    init: InitStrategy = "random",
    tol: float | None = 1e-6,
    log_interval: int = 1,
    random_seed: int | None = None,
) -> CpNeuralResult:
    r"""Fit a CP model by gradient descent.

    Parameters
    ----------
    target
        Either a dense ``ndarray`` (forces ``mode="dense"`` semantics)
        or a :class:`FunctionalTensor` (works in either mode; in
        ``"batched"`` mode the oracle is queried only via ``samples``).
    rank
        CP rank.
    mode
        ``"dense"`` (full-grid loss each step) or ``"batched"`` (random
        index batches).
    batch_size
        Mini-batch size in ``"batched"`` mode. Ignored otherwise.
    max_iter
        Maximum optimisation steps.
    optimizer
        ``"adam"`` (default) or ``"sgd"``.
    lr, beta1, beta2, eps
        Optimiser hyper-parameters.
    init
        Initialisation passed to :class:`CPModel`. ``"svd"`` requires a
        dense ``target`` (or a :class:`FunctionalTensor` we will
        materialise via ``samples`` over the full index grid — only
        attempted for tensors whose ``prod(shape)`` fits in memory).
    tol
        Stop when the moving-window relative change in the loss falls
        below this. Pass ``None`` to disable early stopping.
    log_interval
        Number of optimiser steps per recorded loss value.
    random_seed
        Seed for the initial parameters and (in batched mode) the index
        sampler.

    Returns
    -------
    CpNeuralResult
    """
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if log_interval < 1:
        raise ValueError("log_interval must be at least 1")
    rng = make_rng(random_seed)

    counted: CountingFunctionalTensor | None = None
    if isinstance(target, np.ndarray):
        shape = tuple(int(s) for s in target.shape)
        ft: FunctionalTensor | None = None
        X_dense: FloatArray | None = target
    else:
        counted = CountingFunctionalTensor(target)
        ft = counted
        shape = tuple(int(s) for s in target.shape)
        X_dense = None

    X_ref_for_init: FloatArray | None = X_dense
    if init == "svd" and X_ref_for_init is None:
        # Materialise from the oracle via ``samples`` over the full grid;
        # this still goes through the wrapped tensor so oracle_counts is
        # accurate.
        if int(np.prod(shape)) > 1_000_000:
            raise ValueError(
                "init='svd' on a functional tensor requires prod(shape) <= 1e6"
            )
        grid = np.indices(shape).reshape(len(shape), -1).T.astype(np.intp)
        assert ft is not None
        X_ref_for_init = ft.samples(grid).reshape(shape)

    model = CPModel(
        shape=shape,
        rank=rank,
        init=init,
        X_ref=X_ref_for_init,
        random_seed=int(rng.integers(0, 2**31 - 1)),
    )

    # Optimiser state.
    state_m: list[FloatArray] = [np.zeros_like(model.weights)] + [np.zeros_like(B) for B in model.factors]
    state_v: list[FloatArray] = [np.zeros_like(model.weights)] + [np.zeros_like(B) for B in model.factors]

    losses: list[float] = []
    converged = False
    prev_loss: float | None = None

    for step in range(1, max_iter + 1):
        if mode == "dense":
            if X_dense is None:
                # Functional input in dense mode: we need the full tensor each
                # step. Allowed but expensive — re-sample the grid.
                if int(np.prod(shape)) > 1_000_000:
                    raise ValueError(
                        "mode='dense' on a functional tensor requires prod(shape) <= 1e6"
                    )
                grid = np.indices(shape).reshape(len(shape), -1).T.astype(np.intp)
                assert ft is not None
                X_step = ft.samples(grid).reshape(shape)
            else:
                X_step = X_dense
            grad_w, grad_factors = model.grad_dense(X_step)
            loss = model.loss_dense(X_step)
        else:
            # batched
            if ft is None:
                # Dense ``ndarray`` input in batched mode — sample directly.
                idx = np.column_stack(
                    [rng.integers(0, int(n), size=batch_size) for n in shape]
                ).astype(np.intp)
                values = X_dense[tuple(idx[:, k] for k in range(len(shape)))]  # type: ignore[index]
            else:
                idx = np.column_stack(
                    [rng.integers(0, int(n), size=batch_size) for n in shape]
                ).astype(np.intp)
                values = ft.samples(idx)
            grad_w, grad_factors = model.grad_batch(idx, values)
            loss = model.loss_batch(idx, values)

        grads: list[FloatArray] = [grad_w] + grad_factors
        if optimizer == "adam":
            deltas = _adam_step(grads, state_m, state_v, lr=lr, beta1=beta1, beta2=beta2, eps=eps, step=step)
        elif optimizer == "sgd":
            deltas = [lr * g for g in grads]
        else:
            raise ValueError(f"unknown optimizer {optimizer!r}")

        model.weights = model.weights - deltas[0]
        for k in range(len(model.factors)):
            model.factors[k] = model.factors[k] - deltas[k + 1]

        if step % log_interval == 0:
            losses.append(loss)
            if tol is not None and prev_loss is not None and abs(prev_loss - loss) < tol:
                converged = True
                break
            prev_loss = loss

    return CpNeuralResult(
        factors=model.to_cp_factors(),
        loss_history=np.array(losses, dtype=np.float64),
        iterations=len(losses),
        converged=converged,
        oracle_counts=counted.counts if counted is not None else None,
    )


__all__ = ["CPModel", "CpNeuralResult", "InitStrategy", "OptimizerName", "RunMode", "fit_cp_neural"]
