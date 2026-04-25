r"""Task 1 — Lanczos iteration for symmetric / self-adjoint operators.

Implements the textbook three-term Lanczos recurrence (Saad,
*Numerical Methods for Large Eigenvalue Problems*, Algorithm 6.5) using
only the user-supplied ``matvec`` callable. The algorithm produces an
orthonormal basis :math:`Q_k` of a Krylov subspace and a tridiagonal
matrix :math:`T_k` whose eigenvalues (the *Ritz values*) approximate the
extreme eigenvalues of :math:`A`.

Public surface
--------------
:func:`lanczos`
    Run :math:`k` iterations from a starting vector ``q0``.

The implementation never accesses individual matrix entries — only
``matvec(x)`` calls. Optional full reorthogonalisation re-projects each
new vector against the entire previous basis to preserve orthogonality
in the presence of round-off.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from matcomp.utils.functional_matrix import FloatArray
from matcomp.utils.seeding import make_rng


@dataclass(frozen=True)
class LanczosResult:
    r"""Output of :func:`lanczos`.

    Attributes
    ----------
    alpha
        Diagonal entries of :math:`T_k`, length ``iterations``.
    beta
        Sub-diagonal of :math:`T_k`, length ``iterations - 1``.
    Q
        Orthonormal basis of shape :math:`(n, \text{iterations} + 1)`. The
        last column is the residual direction ``q_{k+1}`` produced by the
        recurrence (zero if a happy breakdown was hit).
    ritz_values
        Eigenvalues of :math:`T_k` in ascending order.
    ritz_vectors
        Optional ``(n, iterations)`` matrix of eigenvectors of :math:`A`
        approximated via :math:`Q_k\,Y` for ``Y`` the eigenvectors of
        :math:`T_k`. ``None`` unless ``compute_ritz_vectors=True``.
    iterations
        Number of completed iterations (≤ requested ``k``; smaller after
        a happy breakdown).
    breakdown
        ``True`` iff a happy breakdown stopped the iteration early.
    matvec_calls
        How many times the user-supplied ``matvec`` was invoked.
    """

    alpha: FloatArray
    beta: FloatArray
    Q: FloatArray
    ritz_values: FloatArray
    ritz_vectors: FloatArray | None
    iterations: int
    breakdown: bool
    matvec_calls: int


def _solve_tridiag_eig(
    alpha: FloatArray, beta: FloatArray, *, eigvecs: bool
) -> tuple[FloatArray, FloatArray | None]:
    r"""Eigendecomposition of a real symmetric tridiagonal matrix.

    NumPy does not expose a banded LAPACK driver; we materialise the
    (small) tridiagonal and call :func:`numpy.linalg.eigh`. ``alpha`` and
    ``beta`` are 1-D arrays.
    """
    k = int(alpha.size)
    if k == 0:
        return np.zeros(0), (np.zeros((0, 0)) if eigvecs else None)
    T = np.diag(alpha)
    if k > 1:
        T += np.diag(beta, k=1) + np.diag(beta, k=-1)
    if eigvecs:
        vals, vecs = np.linalg.eigh(T)
        return vals, vecs
    vals = np.linalg.eigvalsh(T)
    return vals, None


def lanczos(
    matvec: Callable[[FloatArray], FloatArray],
    n: int,
    k: int,
    *,
    q0: FloatArray | None = None,
    tol: float = 1e-12,
    reorthogonalize: bool = False,
    compute_ritz_vectors: bool = False,
    random_seed: int | None = None,
) -> LanczosResult:
    r"""Run :math:`k` steps of the symmetric Lanczos iteration.

    Parameters
    ----------
    matvec
        Callable returning :math:`A\,x` for a length-``n`` real vector.
    n
        Dimension of :math:`A`.
    k
        Number of Lanczos steps. Must satisfy :math:`1 \le k \le n`.
    q0
        Optional initial vector. If ``None``, a random unit vector is
        generated from ``random_seed``.
    tol
        Threshold on :math:`\beta_j`: if :math:`\beta_j < \text{tol}`, a
        happy breakdown is reported and the iteration stops.
    reorthogonalize
        If ``True``, perform full reorthogonalisation of each new
        :math:`q_{j+1}` against the entire previous basis. Slower but
        protects against ghost eigenvalues caused by loss of
        orthogonality.
    compute_ritz_vectors
        If ``True``, also return the Ritz vectors :math:`Q_k Y` where
        :math:`Y` are the eigenvectors of :math:`T_k`. Requires the basis
        :math:`Q_k`, so it forces the basis to be stored regardless of
        ``reorthogonalize``.
    random_seed
        Seed for generating ``q0`` when not supplied.

    Returns
    -------
    LanczosResult
        See the dataclass for the field layout.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if k > n:
        raise ValueError(f"k={k} exceeds dimension n={n}")

    rng = make_rng(random_seed)
    q = q0.astype(np.float64, copy=True) if q0 is not None else rng.standard_normal(n)
    if q.shape != (n,):
        raise ValueError(f"q0 must have shape ({n},), got {q.shape}")
    norm_q = float(np.linalg.norm(q))
    if norm_q == 0.0:
        raise ValueError("q0 must be non-zero")
    q = q / norm_q

    keep_basis = reorthogonalize or compute_ritz_vectors
    Q = np.zeros((n, k + 1), dtype=np.float64) if keep_basis else np.zeros((n, 2), dtype=np.float64)
    Q[:, 0] = q

    alpha = np.zeros(k, dtype=np.float64)
    beta = np.zeros(max(k - 1, 0), dtype=np.float64)

    matvec_calls = 0
    completed = 0
    breakdown = False
    q_prev = np.zeros(n, dtype=np.float64)
    beta_prev = 0.0

    for j in range(k):
        Aq = matvec(q)
        matvec_calls += 1
        a = float(q @ Aq)
        alpha[j] = a
        # Three-term recurrence:  r = A q_j - alpha_j q_j - beta_{j-1} q_{j-1}.
        r = Aq - a * q - beta_prev * q_prev

        if reorthogonalize and j > 0:
            # Full reorthogonalisation against the existing basis Q[:, :j+1].
            for _ in range(2):  # twice-is-enough rule of thumb
                proj = Q[:, : j + 1].T @ r
                r = r - Q[:, : j + 1] @ proj

        b = float(np.linalg.norm(r))
        completed = j + 1

        if j < k - 1:
            beta[j] = b
            if b < tol:
                breakdown = True
                if keep_basis:
                    Q[:, j + 1] = 0.0
                break
            q_next = r / b
            if keep_basis:
                Q[:, j + 1] = q_next
            else:
                Q[:, 1] = q_next
            q_prev = q
            q = q_next
            beta_prev = b
        # Last iteration — record the residual direction even though
        # it is not used internally.
        elif keep_basis:
            if b > 0.0:
                Q[:, j + 1] = r / b
            else:
                Q[:, j + 1] = 0.0

    alpha_eff = alpha[:completed]
    beta_eff = beta[: max(completed - 1, 0)]
    if keep_basis:
        Q_eff = Q[:, : completed + 1]
    else:
        # Without reorthogonalisation we did not store the basis; expose
        # only the two most recently held columns. Tests that need the
        # full basis must pass ``reorthogonalize=True`` or
        # ``compute_ritz_vectors=True``.
        Q_eff = Q[:, : min(completed + 1, 2)]

    ritz_values, ritz_eigvecs_T = _solve_tridiag_eig(alpha_eff, beta_eff, eigvecs=compute_ritz_vectors)
    ritz_vectors: FloatArray | None
    if compute_ritz_vectors and ritz_eigvecs_T is not None:
        ritz_vectors = Q_eff[:, :completed] @ ritz_eigvecs_T
    else:
        ritz_vectors = None

    return LanczosResult(
        alpha=alpha_eff,
        beta=beta_eff,
        Q=Q_eff,
        ritz_values=ritz_values,
        ritz_vectors=ritz_vectors,
        iterations=completed,
        breakdown=breakdown,
        matvec_calls=matvec_calls,
    )


__all__ = ["LanczosResult", "lanczos"]
