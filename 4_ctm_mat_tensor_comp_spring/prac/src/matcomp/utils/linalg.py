"""Numerical primitives reused across multiple algorithms.

This module is intentionally small: it contains those utilities that
appeared in two or more algorithms, were non-trivial to keep correct, and
benefit from a single tested implementation.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt

from matcomp.utils.functional_matrix import FloatArray

IntArray = npt.NDArray[np.integer]


def modified_gram_schmidt(Q: FloatArray, v: FloatArray) -> tuple[FloatArray, FloatArray]:
    r"""Project ``v`` against an orthonormal basis ``Q``.

    Modified Gram–Schmidt is numerically more stable than the classical
    variant for re-orthogonalisation in Lanczos and randomised SVD.

    Parameters
    ----------
    Q
        ``(n, k)`` matrix whose columns form an orthonormal basis.
    v
        ``(n,)`` or ``(n, b)`` vector(s) to orthonormalise.

    Returns
    -------
    v_orth : numpy.ndarray
        ``v`` after subtracting components in ``range(Q)``.
    coefs : numpy.ndarray
        Coefficients :math:`Q^{\top} v` *as accumulated* during the
        projection (so ``v == Q @ coefs + v_orth`` holds up to round-off).
    """
    if Q.ndim != 2:
        raise ValueError("Q must be 2-D")
    if v.ndim not in (1, 2):
        raise ValueError("v must be 1-D or 2-D")
    n_basis = int(Q.shape[1])
    work = v.astype(np.float64, copy=True)
    coefs: FloatArray
    if v.ndim == 1:
        coefs = np.zeros(n_basis, dtype=np.float64)
        for k in range(n_basis):
            c_scalar = float(Q[:, k] @ work)
            coefs[k] = c_scalar
            work -= c_scalar * Q[:, k]
    else:
        coefs = np.zeros((n_basis, v.shape[1]), dtype=np.float64)
        for k in range(n_basis):
            c_vec = Q[:, k] @ work
            coefs[k] = c_vec
            work -= np.outer(Q[:, k], c_vec)
    return work, coefs


def safe_pinv(M: FloatArray, rcond: float = 1e-12, *, warn: bool = True) -> FloatArray:
    r"""Pseudo-inverse with a regularised condition-number check.

    Wraps :func:`numpy.linalg.pinv` but emits a :class:`RuntimeWarning` if
    the smallest non-zero singular value falls below ``rcond *
    sigma_max``. Tasks 2 and 4 require this when the pivot block
    :math:`A[I, J]` becomes ill-conditioned (PDF p. 4 acceptance: "If
    ``A[I, J]`` is poorly conditioned, use ``pinv`` or regularization and
    issue a warning").

    Parameters
    ----------
    M
        Matrix to invert.
    rcond
        Truncation threshold passed to :func:`numpy.linalg.pinv`.
    warn
        If ``True``, emit a ``RuntimeWarning`` when conditioning is poor.

    Returns
    -------
    numpy.ndarray
        Moore–Penrose pseudo-inverse :math:`M^{+}`.
    """
    if M.ndim != 2:
        raise ValueError("safe_pinv expects a 2-D matrix")
    s = np.linalg.svd(M, compute_uv=False)
    if warn and s.size > 0 and s[0] > 0.0:
        cond = float(s[0] / max(s[-1], np.finfo(float).tiny))
        if cond > 1.0 / rcond:
            warnings.warn(
                f"safe_pinv: matrix is ill-conditioned (cond={cond:.3e}); "
                f"applying truncated pseudo-inverse with rcond={rcond}",
                RuntimeWarning,
                stacklevel=2,
            )
    return np.linalg.pinv(M, rcond=rcond)


def truncate_svd(
    U: FloatArray,
    S: FloatArray,
    Vt: FloatArray,
    *,
    eps: float | None = None,
    rank: int | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, int]:
    r"""Truncate an SVD by either an absolute rank or a relative tolerance.

    Exactly one of ``eps`` or ``rank`` must be supplied.

    * ``rank=k`` keeps the top ``k`` singular components.
    * ``eps=ε`` drops every singular value with :math:`\sigma_k < \epsilon
      \cdot \sigma_0`. The kept rank is therefore at most ``len(S)``.

    Parameters
    ----------
    U, S, Vt
        Output of an SVD: ``U`` of shape ``(m, k)``, ``S`` of length ``k``,
        ``Vt`` of shape ``(k, n)``.
    eps
        Relative tolerance for the singular-value tail.
    rank
        Fixed target rank.

    Returns
    -------
    U_r, S_r, Vt_r, r : tuple
        Truncated factors and the chosen rank ``r``.
    """
    if (eps is None) == (rank is None):
        raise ValueError("truncate_svd: pass exactly one of eps or rank")
    if S.size == 0:
        return U, S, Vt, 0
    if rank is not None:
        r = int(min(max(rank, 0), int(S.size)))
    else:
        assert eps is not None  # narrowing for mypy
        sigma_max = float(S[0])
        threshold = eps * sigma_max
        r = int(np.sum(S >= threshold))
    return U[:, :r], S[:r], Vt[:r, :], r


def maxvol(B: FloatArray, *, max_iter: int = 50, tol: float = 1.05) -> IntArray:
    r"""Find a row subset of ``B`` with quasi-maximal :math:`|\det|`.

    Standard maxvol heuristic of Goreinov, Tyrtyshnikov and Zamarashkin
    (*Linear Algebra Appl.* 261, 1997). ``B`` has shape :math:`(N, r)`
    with :math:`N \ge r`. The returned permutation has length :math:`N`;
    the first :math:`r` entries are the chosen rows.

    Parameters
    ----------
    B
        Input matrix of shape :math:`(N, r)`.
    max_iter
        Maximum number of swap iterations.
    tol
        Stopping criterion: stop when no candidate row in the residual
        :math:`Z = B\,A^{-1}` (with :math:`A = B[I, :]`) exceeds ``tol``
        in absolute value. The canonical default 1.05 is from the
        original paper.

    Returns
    -------
    numpy.ndarray
        Permutation of ``range(N)``; the first ``r`` entries are the
        selected rows.

    Notes
    -----
    Used by Task 2 (cross approximation, ``init="maxvol"``) and Task 13
    (TT-cross). Both share this single tested implementation.
    """
    N, r = B.shape
    if N < r:
        raise ValueError(f"maxvol requires N >= r, got N={N}, r={r}")
    perm = np.arange(N, dtype=np.intp)
    # Initial pivot via Gaussian elimination with partial pivoting.
    work = B.astype(np.float64, copy=True)
    for k in range(r):
        idx = k + int(np.argmax(np.abs(work[k:, k])))
        if idx != k:
            perm[[k, idx]] = perm[[idx, k]]
            work[[k, idx]] = work[[idx, k]]
        if abs(work[k, k]) < np.finfo(float).tiny:
            break
        work[k + 1 :, k:] -= np.outer(work[k + 1 :, k] / work[k, k], work[k, k:])

    # Maxvol refinement: solve B = Z @ B[I, :], swap until no |Z[i,j]| > tol.
    for _ in range(max_iter):
        B_perm = B[perm]
        sub = B_perm[:r]
        try:
            Z = np.linalg.solve(sub.T, B_perm.T).T  # B_perm @ inv(sub), shape (N, r)
        except np.linalg.LinAlgError:
            break
        flat_idx = int(np.argmax(np.abs(Z)))
        i_max, j_max = divmod(flat_idx, r)
        if abs(Z[i_max, j_max]) <= tol:
            break
        perm[[j_max, i_max]] = perm[[i_max, j_max]]
    return perm


__all__ = ["IntArray", "maxvol", "modified_gram_schmidt", "safe_pinv", "truncate_svd"]
