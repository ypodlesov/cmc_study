r"""Quality metrics for tensor algorithms.

Mirrors :mod:`matcomp.utils.metrics` for the d-mode case. Provides:

* ``rel_frob_tensor`` — relative Frobenius-norm error for two dense
  tensors.
* ``cp_relative_error`` — :math:`\lVert X - X_{CP}\rVert_F / \lVert X
  \rVert_F` computed *without* densifying the CP tensor (used for
  loss tracking inside CP-ALS).
* ``cp_cp_relative_error`` — Task 8 stop criterion comparing two CP
  tensors only via factor inner products.
* ``sample_rmse_tensor`` — sample-based metric for functional tensors
  too large to materialise (PDF p. 1).
* ``compression_ratio_tensor`` — :math:`\prod n_k \cdot \mathrm{itemsize}
  \,/\,\mathrm{factors\_memory}`.
* ``multilinear_rank_truncation_error`` — Eckart–Young analogue per
  mode (lower bound on the ST-HOSVD error).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from matcomp.utils.functional_matrix import FloatArray
from matcomp.utils.functional_tensor import FunctionalTensor
from matcomp.utils.seeding import make_rng
from matcomp.utils.tensor_linalg import (
    cp_inner_product,
    cp_norm_sq,
    mttkrp,
    unfold,
)
from matcomp.utils.tensor_low_rank import LowRankTensor


def rel_frob_tensor(reference: FloatArray, approximation: FloatArray) -> float:
    r"""Relative Frobenius-norm error :math:`\lVert X - \hat X\rVert_F /
    \lVert X\rVert_F` for two dense tensors of identical shape.
    """
    if reference.shape != approximation.shape:
        raise ValueError(
            f"shape mismatch: ref {reference.shape}, approx {approximation.shape}"
        )
    denom = float(np.linalg.norm(reference))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(reference - approximation) / denom)


def cp_relative_error(
    weights: FloatArray,
    factors: Sequence[FloatArray],
    X: FloatArray,
) -> float:
    r"""Relative error :math:`\lVert X - X_{CP}\rVert_F / \lVert X\rVert_F`
    computed without materialising the CP tensor.

    Uses
    :math:`\lVert X - X_{CP}\rVert^{2} = \lVert X\rVert^{2}
    - 2\,\langle X, X_{CP}\rangle + \lVert X_{CP}\rVert^{2}` with
    :math:`\langle X, X_{CP}\rangle = \sum_r \lambda_r \cdot
    \mathbf{1}^{\top} (\mathrm{MTTKRP}_0(X, \{A_k\}) \odot A_0)[:, r]`.
    """
    if any(fac.shape[1] != weights.size for fac in factors):
        raise ValueError("cp_relative_error: factor column counts must match len(weights)")
    norm_X_sq = float(np.sum(X * X))
    norm_cp_sq = cp_norm_sq(weights, list(factors))
    # ⟨X, X_CP⟩ = sum_r weights[r] * (MTTKRP(X, factors, mode=0) ⊙ factors[0])[:, r]
    M0 = mttkrp(X, list(factors), 0)
    inner_per_r = np.sum(M0 * factors[0], axis=0)
    inner = float(weights @ inner_per_r)
    err_sq = max(norm_X_sq - 2.0 * inner + norm_cp_sq, 0.0)
    if norm_X_sq == 0.0:
        return 0.0
    return float(np.sqrt(err_sq / norm_X_sq))


def cp_cp_relative_error(
    target_weights: FloatArray,
    target_factors: Sequence[FloatArray],
    source_weights: FloatArray,
    source_factors: Sequence[FloatArray],
) -> float:
    r"""Relative error of one CP tensor approximating another.

    .. math::

        \frac{\lVert X_{S} - X_{T}\rVert_F}{\lVert X_{S}\rVert_F}.

    Both tensors stay in CP form; no densification occurs.
    """
    norm_s_sq = cp_norm_sq(source_weights, list(source_factors))
    norm_t_sq = cp_norm_sq(target_weights, list(target_factors))
    inner = cp_inner_product(
        source_weights, list(source_factors), target_weights, list(target_factors)
    )
    err_sq = max(norm_s_sq - 2.0 * inner + norm_t_sq, 0.0)
    if norm_s_sq == 0.0:
        return 0.0
    return float(np.sqrt(err_sq / norm_s_sq))


def sample_rmse_tensor(
    ft: FunctionalTensor,
    approx: LowRankTensor,
    num_samples: int,
    random_seed: int | np.random.Generator | None = None,
) -> float:
    r"""Sampled root-mean-square error on random multi-indices.

    Mirrors :func:`matcomp.utils.metrics.sample_rmse` for tensors. Used
    in TT-cross / CP-NN where the full tensor is intentionally never
    built.
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if tuple(ft.shape) != tuple(approx.shape):
        raise ValueError(
            f"shape mismatch: ft {tuple(ft.shape)}, approx {tuple(approx.shape)}"
        )
    rng = make_rng(random_seed)
    idx = np.column_stack(
        [rng.integers(0, int(n), size=num_samples) for n in ft.shape]
    ).astype(np.intp)
    ref_vals = ft.samples(idx)
    sq = 0.0
    for k in range(num_samples):
        tup = tuple(int(v) for v in idx[k])
        sq += (float(ref_vals[k]) - float(approx.entry(tup))) ** 2
    return float(np.sqrt(sq / num_samples))


def compression_ratio_tensor(approx: LowRankTensor, itemsize: int = 8) -> float:
    r"""Compression ratio :math:`\prod n_k \cdot \mathrm{itemsize} /
    \mathrm{factors\_memory}`. ``> 1`` means the carrier saves memory.
    """
    total = int(np.prod(approx.shape))
    factors_bytes = approx.factors_memory()
    if factors_bytes == 0:
        return float("inf")
    return float(total * itemsize) / float(factors_bytes)


def multilinear_rank_truncation_error(
    X: FloatArray, ranks: tuple[int, ...]
) -> float:
    r"""Lower bound on the ST-HOSVD relative Frobenius error.

    The Tucker truncation error obeys :math:`\lVert X -
    \mathrm{HOSVD}_r(X)\rVert_F^{2} \le \sum_k \sum_{j > r_k}
    \sigma_j^{2}(X_{(k)})`. We return the right-hand side normalised by
    :math:`\lVert X\rVert_F`.
    """
    if len(ranks) != X.ndim:
        raise ValueError(
            f"ranks length {len(ranks)} must equal X.ndim {X.ndim}"
        )
    bound_sq = 0.0
    for k, r in enumerate(ranks):
        s = np.linalg.svd(unfold(X, k), compute_uv=False)
        if r < s.size:
            bound_sq += float(np.sum(s[r:] ** 2))
    norm = float(np.linalg.norm(X))
    if norm == 0.0:
        return 0.0
    return float(np.sqrt(bound_sq) / norm)


__all__ = [
    "compression_ratio_tensor",
    "cp_cp_relative_error",
    "cp_relative_error",
    "multilinear_rank_truncation_error",
    "rel_frob_tensor",
    "sample_rmse_tensor",
]
