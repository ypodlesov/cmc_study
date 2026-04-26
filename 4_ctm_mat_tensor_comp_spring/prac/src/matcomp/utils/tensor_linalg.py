r"""NumPy primitives for tensor algorithms.

Every tensor task imports its low-level primitives from this single
module. The implementations are pure NumPy — no dependence on tensor
libraries — so the project keeps its minimal dependency footprint
(numpy + matplotlib).

Notation follows Kolda & Bader (*SIAM Review* 51(3), 2009): mode-``n``
unfolding, mode-``n`` product :math:`X \times_n M`, Khatri–Rao product
:math:`A \odot B`, and the matrix
:math:`\mathrm{MTTKRP}_n(X, \{A_k\}) = X_{(n)} \cdot
(A_d \odot \cdots \odot A_{n+1} \odot A_{n-1} \odot \cdots \odot A_1)`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from matcomp.utils.functional_matrix import FloatArray


def unfold(X: FloatArray, mode: int) -> FloatArray:
    r"""Mode-``n`` matricisation :math:`X_{(n)}`.

    Reorders ``X`` so mode ``n`` becomes the row index and the remaining
    modes are stacked into the columns in *Kolda* order: the column of
    multi-index :math:`(i_{n+1}, \ldots, i_d, i_1, \ldots, i_{n-1})` —
    i.e. modes after ``n`` first, then modes before, each ranging slow
    to fast. Equivalent to ``np.reshape(np.moveaxis(X, n, 0), (X.shape[n], -1))``.

    Parameters
    ----------
    X
        d-mode tensor.
    mode
        Mode along which to unfold. Must satisfy ``0 <= mode < X.ndim``.

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(X.shape[mode], prod(X.shape) // X.shape[mode])``.
    """
    if not 0 <= mode < X.ndim:
        raise ValueError(f"mode {mode} out of range for ndim={X.ndim}")
    return np.reshape(np.moveaxis(X, mode, 0), (X.shape[mode], -1))


def fold(M: FloatArray, mode: int, shape: tuple[int, ...]) -> FloatArray:
    """Inverse of :func:`unfold`.

    Reshapes ``M`` (the mode-``n`` matricisation) back into a tensor of
    the given ``shape``.

    Parameters
    ----------
    M
        Matrix of shape ``(shape[mode], prod(shape) // shape[mode])``.
    mode
        Mode along which the matrix was unfolded.
    shape
        Target tensor shape.

    Returns
    -------
    numpy.ndarray
        Tensor of shape ``shape``.
    """
    if not 0 <= mode < len(shape):
        raise ValueError(f"mode {mode} out of range for ndim={len(shape)}")
    full_shape = (shape[mode],) + tuple(s for d, s in enumerate(shape) if d != mode)
    return np.moveaxis(np.reshape(M, full_shape), 0, mode)


def mode_dot(X: FloatArray, M: FloatArray, mode: int, *, transpose: bool = False) -> FloatArray:
    r"""Mode-``n`` product :math:`X \times_n M`.

    Replaces mode ``mode`` of ``X`` with ``M @ unfold(X, mode)`` (or with
    ``M.T @ unfold(X, mode)`` if ``transpose=True``).

    Parameters
    ----------
    X
        d-mode tensor.
    M
        Factor matrix. Without transpose, of shape ``(p, X.shape[mode])``;
        with transpose, of shape ``(X.shape[mode], p)``.
    mode
        Mode index.
    transpose
        Use ``M.T`` instead of ``M``.

    Returns
    -------
    numpy.ndarray
        Tensor with mode ``mode`` resized to ``p``.
    """
    if not 0 <= mode < X.ndim:
        raise ValueError(f"mode {mode} out of range for ndim={X.ndim}")
    if M.ndim != 2:
        raise ValueError("mode_dot expects a 2-D matrix M")
    op = M.T if transpose else M
    if op.shape[1] != X.shape[mode]:
        raise ValueError(
            f"mode_dot: M cols ({op.shape[1]}) must match X.shape[{mode}] "
            f"= {X.shape[mode]}"
        )
    new_shape = list(X.shape)
    new_shape[mode] = op.shape[0]
    return fold(op @ unfold(X, mode), mode, tuple(new_shape))


def multi_mode_dot(
    X: FloatArray,
    factors: Sequence[FloatArray],
    modes: Sequence[int] | None = None,
    *,
    transpose: bool = False,
) -> FloatArray:
    """Apply :func:`mode_dot` along multiple modes in sequence.

    If ``modes`` is ``None``, defaults to ``range(len(factors))`` — i.e.
    ``factors[k]`` is applied along mode ``k``.

    Parameters
    ----------
    X
        d-mode tensor.
    factors
        Sequence of factor matrices.
    modes
        Sequence of mode indices in the order of application. ``None``
        applies them in mode order.
    transpose
        Pass through to :func:`mode_dot`.

    Returns
    -------
    numpy.ndarray
        Tensor with the selected modes contracted by the corresponding
        factors.
    """
    if modes is None:
        modes = list(range(len(factors)))
    if len(modes) != len(factors):
        raise ValueError("len(modes) must equal len(factors)")
    out = X
    for fac, mode in zip(factors, modes, strict=True):
        out = mode_dot(out, fac, mode, transpose=transpose)
    return out


def khatri_rao(
    matrices: Sequence[FloatArray],
    *,
    skip: int | None = None,
    reverse: bool = False,
) -> FloatArray:
    r"""Column-wise Khatri–Rao product of a sequence of matrices.

    Each input has shape :math:`(n_k, R)`; the result has shape
    :math:`(\prod n_k, R)`.

    The default ``reverse=False`` matches the mode order produced by
    :func:`unfold`. Concretely, for ``khatri_rao([A_1, A_2], reverse=False)``
    the result row index walks :math:`(i_1, i_2)` with :math:`i_2`
    varying fastest — which matches the C-order (numpy) unfolding
    convention used throughout this module. The Kolda & Bader survey
    uses the opposite ordering; pass ``reverse=True`` to obtain it.

    Parameters
    ----------
    matrices
        Sequence of factor matrices, all with the same column count.
    skip
        If not ``None``, exclude this index. Used by MTTKRP.
    reverse
        Iterate matrices in reverse mode order (Kolda convention).

    Returns
    -------
    numpy.ndarray
        Khatri–Rao product, shape ``(prod(n_k for k != skip), R)``.
    """
    selected = [m for k, m in enumerate(matrices) if k != skip]
    if not selected:
        raise ValueError("khatri_rao: no matrices selected after skip")
    R = selected[0].shape[1]
    if any(m.shape[1] != R for m in selected):
        raise ValueError("khatri_rao: all matrices must share the column count")
    iterator = list(reversed(selected)) if reverse else list(selected)
    out = iterator[0]
    for mat in iterator[1:]:
        n_out, _ = out.shape
        n_mat, _ = mat.shape
        # Column-wise Kron: row index walks (i_out, i_mat) with i_mat fastest.
        out = (out[:, None, :] * mat[None, :, :]).reshape(n_out * n_mat, R)
    return out


def mttkrp(X: FloatArray, factors: Sequence[FloatArray], mode: int) -> FloatArray:
    r"""Matricised Tensor Times Khatri–Rao Product.

    Computes :math:`X_{(n)}\,(A_0 \odot \cdots \odot A_{n-1} \odot
    A_{n+1} \odot \cdots \odot A_{d-1})` for the CP-ALS update, using
    the C-order (numpy) unfolding convention so the indices match
    :func:`unfold`.

    Parameters
    ----------
    X
        d-mode tensor.
    factors
        ``d`` factor matrices ``A_k`` of shape ``(X.shape[k], R)``.
    mode
        Mode being updated; ``factors[mode]`` is the *target* and is
        excluded from the Khatri–Rao product.

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(X.shape[mode], R)``.
    """
    if len(factors) != X.ndim:
        raise ValueError("mttkrp: need one factor per mode")
    kr = khatri_rao(factors, skip=mode, reverse=False)
    return unfold(X, mode) @ kr


def cp_factor_grams(factors: Sequence[FloatArray]) -> list[FloatArray]:
    r"""Per-factor Gram matrices :math:`G_k = A_k^{\top} A_k`.

    Used by ALS normal equations and by :func:`cp_inner_product`.
    """
    return [fac.T @ fac for fac in factors]


def cp_inner_product(
    weights_a: FloatArray,
    factors_a: Sequence[FloatArray],
    weights_b: FloatArray,
    factors_b: Sequence[FloatArray],
) -> float:
    r"""Inner product of two CP tensors via factor cross-Grams.

    .. math::

        \langle X_A, X_B\rangle = \sum_{r, s} \lambda^A_r \lambda^B_s
        \prod_{k} \big(A_k[:, r]^{\top} B_k[:, s]\big).

    Never densifies either CP tensor.
    """
    if len(factors_a) != len(factors_b):
        raise ValueError("cp_inner_product: factor counts disagree")
    cross = np.ones((int(weights_a.size), int(weights_b.size)), dtype=np.float64)
    for A_k, B_k in zip(factors_a, factors_b, strict=True):
        if A_k.shape[0] != B_k.shape[0]:
            raise ValueError("cp_inner_product: incompatible mode sizes")
        cross *= A_k.T @ B_k
    return float(weights_a @ (cross @ weights_b))


def cp_norm_sq(weights: FloatArray, factors: Sequence[FloatArray]) -> float:
    r"""Frobenius norm squared :math:`\lVert X_{CP}\rVert_F^{2}`.

    Equal to ``cp_inner_product(weights, factors, weights, factors)`` but
    spelled out for clarity.
    """
    return cp_inner_product(weights, factors, weights, factors)


def cp_to_dense(weights: FloatArray, factors: Sequence[FloatArray]) -> FloatArray:
    r"""Materialise a CP tensor :math:`X = \sum_r \lambda_r \bigotimes_k A_k[:, r]`.

    Used only for small tests; the algorithms themselves never call this.
    """
    if any(fac.shape[1] != weights.size for fac in factors):
        raise ValueError("cp_to_dense: factor column counts must match len(weights)")
    shape = tuple(int(fac.shape[0]) for fac in factors)
    R = int(weights.size)
    out = np.zeros(shape, dtype=np.float64)
    # Sum of rank-1 outer products. R is small in practice, so the
    # explicit loop is fine and easy to read.
    for r in range(R):
        rank1 = np.array(weights[r], dtype=np.float64)
        for k, fac in enumerate(factors):
            shp = [1] * len(factors)
            shp[k] = fac.shape[0]
            rank1 = rank1 * fac[:, r].reshape(tuple(shp))
        out += rank1
    return out


def tucker_to_dense(core: FloatArray, factors: Sequence[FloatArray]) -> FloatArray:
    r"""Materialise a Tucker tensor :math:`X = G \times_1 U_1 \cdots \times_d U_d`."""
    if len(factors) != core.ndim:
        raise ValueError("tucker_to_dense: factor count must equal core.ndim")
    return multi_mode_dot(core, list(factors))


def tt_to_dense(cores: Sequence[FloatArray]) -> FloatArray:
    r"""Materialise a TT tensor from its cores.

    Each core has shape :math:`(r_{k-1}, n_k, r_k)` with :math:`r_0 =
    r_d = 1`. Used only for small tests; never inside Task 13.
    """
    if not cores:
        raise ValueError("tt_to_dense: empty core list")
    G_first = cores[0]
    if G_first.shape[0] != 1:
        raise ValueError(f"tt_to_dense: first core must have r0=1, got {G_first.shape[0]}")
    G_last = cores[-1]
    if G_last.shape[2] != 1:
        raise ValueError(f"tt_to_dense: last core must have r_d=1, got {G_last.shape[2]}")
    shape = tuple(int(G.shape[1]) for G in cores)
    # Unfold sequentially: result shape (n_0, n_1 * ... * n_{d-1}).
    out = cores[0].reshape(cores[0].shape[1], cores[0].shape[2])  # (n_0, r_1)
    for k in range(1, len(cores)):
        G = cores[k]  # (r_{k-1}, n_k, r_k)
        rk_1 = G.shape[0]
        n_k = G.shape[1]
        rk = G.shape[2]
        out = out.reshape(-1, rk_1) @ G.reshape(rk_1, n_k * rk)
        out = out.reshape(-1, rk)
    return out.reshape(shape)


def tt_entry(cores: Sequence[FloatArray], idx: tuple[int, ...]) -> float:
    r"""Single entry :math:`X[i_1, \ldots, i_d]` of a TT tensor."""
    if len(idx) != len(cores):
        raise ValueError("tt_entry: index length must match number of cores")
    out = np.array([[1.0]])
    for k, G in enumerate(cores):
        slab = G[:, int(idx[k]), :]
        out = out @ slab
    return float(out[0, 0])


def multilinear_rank(X: FloatArray, eps: float = 0.0) -> tuple[int, ...]:
    r"""Multilinear rank :math:`(r_1, \ldots, r_d)` of a tensor.

    Each :math:`r_k` is the numerical rank of :math:`X_{(k)}`. With
    ``eps > 0``, the rank is the count of singular values exceeding
    :math:`\epsilon \cdot \sigma_0` of that unfolding.
    """
    ranks: list[int] = []
    for k in range(X.ndim):
        s = np.linalg.svd(unfold(X, k), compute_uv=False)
        if s.size == 0:
            ranks.append(0)
            continue
        if eps <= 0.0:
            ranks.append(int(np.sum(s > 0.0)))
        else:
            threshold = float(s[0]) * float(eps)
            ranks.append(int(np.sum(s > threshold)))
    return tuple(ranks)


def pack_cp_theta(weights: FloatArray, factors: Sequence[FloatArray]) -> FloatArray:
    r"""Flatten weights + factors into a single 1-D parameter vector :math:`\theta`.

    Layout: ``[weights, factors[0].ravel(), factors[1].ravel(), ...]``.
    Shared by Task 10 (CP via differentiable optimization) and Task 11
    (CP-LM) so both modules accept a uniform parameter vector.
    """
    return np.concatenate(
        [np.asarray(weights, dtype=np.float64).ravel()]
        + [np.asarray(fac, dtype=np.float64).ravel() for fac in factors]
    )


def unpack_cp_theta(
    theta: FloatArray,
    shape: tuple[int, ...],
    rank: int,
) -> tuple[FloatArray, list[FloatArray]]:
    """Inverse of :func:`pack_cp_theta`.

    Parameters
    ----------
    theta
        Flat 1-D parameter vector.
    shape
        Tensor mode sizes ``(n_1, ..., n_d)``.
    rank
        CP rank.

    Returns
    -------
    weights, factors
        Length-``rank`` weight vector and ``d`` factor matrices of
        shape ``(n_k, rank)``.
    """
    expected = rank + sum(int(n) * rank for n in shape)
    if theta.size != expected:
        raise ValueError(
            f"unpack_cp_theta: expected size {expected}, got {theta.size}"
        )
    weights = np.ascontiguousarray(theta[:rank].astype(np.float64, copy=True))
    factors: list[FloatArray] = []
    cursor = rank
    for n in shape:
        flat = theta[cursor : cursor + int(n) * rank]
        factors.append(np.ascontiguousarray(flat.reshape(int(n), rank).astype(np.float64, copy=True)))
        cursor += int(n) * rank
    return weights, factors


__all__ = [
    "cp_factor_grams",
    "cp_inner_product",
    "cp_norm_sq",
    "cp_to_dense",
    "fold",
    "khatri_rao",
    "mode_dot",
    "mttkrp",
    "multi_mode_dot",
    "multilinear_rank",
    "pack_cp_theta",
    "tt_entry",
    "tt_to_dense",
    "tucker_to_dense",
    "unfold",
    "unpack_cp_theta",
]
