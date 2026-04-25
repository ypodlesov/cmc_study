r"""Task 2 — Cross / skeleton approximation for arbitrary rank :math:`r`.

Implements the ``A ≈ C M R`` factorisation with three initialisation
strategies:

* ``"random"`` — sample the index sets :math:`I` and :math:`J` uniformly
  at random.
* ``"greedy"`` — defer to :func:`adaptive_cross_cached` (Task 4) for
  rank-1 ACA pivots; this matches the PDF's "simplified greedy residual
  search" wording (p. 3).
* ``"maxvol"`` — start from a random or greedy guess and refine
  :math:`I`, :math:`J` by sweeping :math:`\max\,|\det A[I, J]|` in
  successive maxvol steps (Goreinov–Tyrtyshnikov–Zamarashkin, 1997).

The functional matrix is queried only through ``block`` / ``row`` /
``col``; the full :math:`A` is never materialised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from matcomp.matrix_algorithms.adaptive_cross import adaptive_cross_cached
from matcomp.utils.counting import CountingFunctionalMatrix, OracleCounts
from matcomp.utils.functional_matrix import FloatArray, FunctionalMatrix, IntArray
from matcomp.utils.linalg import safe_pinv
from matcomp.utils.low_rank import CMRFactors
from matcomp.utils.seeding import make_rng

InitStrategy = Literal["random", "maxvol", "greedy"]


@dataclass(frozen=True)
class CrossResult:
    r"""Result of :func:`cross_rank_r`.

    Attributes
    ----------
    factors
        Cross-format carrier :math:`A \approx C\,M\,R` with
        :math:`M = \mathrm{pinv}(A[I, J])`.
    I
        Selected row indices.
    J
        Selected column indices.
    oracle_counts
        Counts of each oracle method invoked while building the
        approximation.
    sweeps
        How many maxvol refinement sweeps were actually performed (0 for
        ``init="random"`` or ``init="greedy"``).
    """

    factors: CMRFactors
    I: IntArray
    J: IntArray
    oracle_counts: OracleCounts
    sweeps: int

    # ---- LowRankApprox passthroughs ---------------------------------------

    @property
    def rank(self) -> int:
        """Number of selected pivots."""
        return self.factors.rank

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the represented matrix."""
        return self.factors.shape

    def entry(self, i: int, j: int) -> float:
        return self.factors.entry(i, j)

    def matvec(self, x: FloatArray) -> FloatArray:
        return self.factors.matvec(x)

    def rmatvec(self, y: FloatArray) -> FloatArray:
        return self.factors.rmatvec(y)

    def reconstruct_small(self) -> FloatArray:
        return self.factors.reconstruct_small()

    def factors_memory(self) -> int:
        return self.factors.factors_memory()


def _maxvol(B: FloatArray, max_iter: int = 50, tol: float = 1.05) -> IntArray:
    r"""Find a row subset of ``B`` with quasi-maximal :math:`|\det|`.

    Implements the standard maxvol heuristic (Goreinov et al.). ``B`` has
    shape :math:`(N, r)` with :math:`N \ge r`. The returned permutation
    has length :math:`N`; the first :math:`r` entries are the chosen rows.

    Parameters
    ----------
    B
        Input matrix of shape :math:`(N, r)`.
    max_iter
        Maximum number of swaps.
    tol
        Stop when no candidate improves :math:`|\det A[I]|` by more than
        ``tol``. The default (1.05) is the canonical choice from the
        original paper.
    """
    N, r = B.shape
    if N < r:
        raise ValueError("maxvol requires N >= r")
    perm = np.arange(N, dtype=np.intp)
    # Initial pivot: pick rows by Householder column-pivoted QR.
    # We re-use np.linalg.qr by transposing: pivot of B^T gives row pivots of B.
    Q, _ = np.linalg.qr(B, mode="reduced")
    # Use simple Gaussian elimination with partial pivoting on B to
    # discover an initial row set.
    work = B.astype(np.float64, copy=True)
    for k in range(r):
        idx = k + int(np.argmax(np.abs(work[k:, k])))
        if idx != k:
            perm[[k, idx]] = perm[[idx, k]]
            work[[k, idx]] = work[[idx, k]]
        if abs(work[k, k]) < np.finfo(float).tiny:
            break
        work[k + 1 :, k:] -= np.outer(work[k + 1 :, k] / work[k, k], work[k, k:])

    # Maxvol refinement: solve B = Z * B[I,:], then swap until no |Z[i,j]| > tol.
    for _ in range(max_iter):
        B_perm = B[perm]
        sub = B_perm[:r]
        try:
            Z = np.linalg.solve(sub.T, B_perm.T).T  # B_perm @ inv(sub) shape (N,r)
        except np.linalg.LinAlgError:
            break
        flat_idx = int(np.argmax(np.abs(Z)))
        i_max, j_max = divmod(flat_idx, r)
        if abs(Z[i_max, j_max]) <= tol:
            break
        # swap row j_max (inside the chosen set) with row i_max (outside)
        perm[[j_max, i_max]] = perm[[i_max, j_max]]
    return perm


def cross_rank_r(
    A: FunctionalMatrix,
    r: int,
    *,
    init: InitStrategy = "maxvol",
    max_sweeps: int = 5,
    tol: float | None = None,
    pinv_rcond: float = 1e-12,
    random_seed: int | None = None,
) -> CrossResult:
    r"""Build a rank-``r`` cross / skeleton approximation of ``A``.

    Parameters
    ----------
    A
        Functional matrix to approximate.
    r
        Target rank :math:`r \ge 1`. Must satisfy
        :math:`r \le \min(m, n)`.
    init
        Initialisation strategy for the row / column index sets.
    max_sweeps
        Number of maxvol refinement sweeps when ``init="maxvol"``. Each
        sweep alternates row and column maxvol on
        :math:`A[:, J]` / :math:`A[I, :]^{\top}`. Ignored for other
        strategies.
    tol
        Optional convergence tolerance: stop refining when the
        Frobenius-norm difference between successive approximations
        falls below ``tol``.
    pinv_rcond
        ``rcond`` passed to :func:`safe_pinv` when forming
        :math:`M = \mathrm{pinv}(A[I, J])`.
    random_seed
        Seed for the RNG used by ``init="random"`` and ``init="greedy"``.

    Returns
    -------
    CrossResult
        Cross factors, selected indices and oracle counts.
    """
    if r < 1:
        raise ValueError("r must be at least 1")
    rng = make_rng(random_seed)
    counted = CountingFunctionalMatrix(A)
    m, n = counted.shape
    if r > min(m, n):
        raise ValueError(f"r={r} exceeds min(shape)={min(m, n)}")

    # ---- choose initial I, J ----------------------------------------------
    if init == "random":
        I = rng.choice(m, size=r, replace=False)
        J = rng.choice(n, size=r, replace=False)
    elif init == "greedy":
        # Reuse Task 4's ACA: harvest the first r pivots.
        seed = int(rng.integers(0, 2**31 - 1))
        aca = adaptive_cross_cached(counted, max_rank=r, eps=0.0, random_seed=seed)
        if not aca.pivots:
            I = rng.choice(m, size=r, replace=False)
            J = rng.choice(n, size=r, replace=False)
        else:
            I = np.array([p[0] for p in aca.pivots], dtype=np.intp)
            J = np.array([p[1] for p in aca.pivots], dtype=np.intp)
            # Fill if ACA stopped early.
            while I.size < r:
                extra_i = int(rng.integers(0, m))
                if extra_i not in I:
                    I = np.append(I, extra_i)
            while J.size < r:
                extra_j = int(rng.integers(0, n))
                if extra_j not in J:
                    J = np.append(J, extra_j)
            I = I[:r]
            J = J[:r]
    elif init == "maxvol":
        # Start with a random column set, then alternate row / column maxvol.
        J = rng.choice(n, size=r, replace=False)
        I = rng.choice(m, size=r, replace=False)
    else:
        raise ValueError(f"unknown init strategy: {init!r}")

    sweeps = 0
    last_residual: float | None = None
    if init == "maxvol":
        for _ in range(max_sweeps):
            # row maxvol on A[:, J]
            C = counted.block(np.arange(m, dtype=np.intp), J)
            row_perm = _maxvol(C)
            I = np.asarray(row_perm[:r], dtype=np.intp)
            # column maxvol on A[I, :]^T
            R = counted.block(I, np.arange(n, dtype=np.intp))
            col_perm = _maxvol(R.T)
            J = np.asarray(col_perm[:r], dtype=np.intp)
            sweeps += 1

            if tol is not None:
                # Materialise current C, M, R for the residual estimate on
                # a small grid (this is *not* free, but the PDF allows
                # sample-based residuals).
                C = counted.block(np.arange(m, dtype=np.intp), J)
                M = safe_pinv(counted.block(I, J), rcond=pinv_rcond, warn=False)
                R = counted.block(I, np.arange(n, dtype=np.intp))
                approx = C @ M @ R
                # Sampled residual on 64 random points
                ii = rng.integers(0, m, size=64)
                jj = rng.integers(0, n, size=64)
                refs = np.array([counted.entry(int(a), int(b)) for a, b in zip(ii, jj, strict=True)])
                got = approx[ii, jj]
                ref_norm = float(np.linalg.norm(refs))
                resid = float(np.linalg.norm(refs - got) / max(ref_norm, 1e-30))
                if last_residual is not None and abs(last_residual - resid) < tol:
                    break
                last_residual = resid

    # ---- form C, M, R -----------------------------------------------------
    I = np.asarray(I, dtype=np.intp)
    J = np.asarray(J, dtype=np.intp)
    C = counted.block(np.arange(m, dtype=np.intp), J)
    R = counted.block(I, np.arange(n, dtype=np.intp))
    pivot = counted.block(I, J)
    M = safe_pinv(pivot, rcond=pinv_rcond, warn=True)

    factors = CMRFactors(
        C=np.ascontiguousarray(C, dtype=np.float64),
        M=np.ascontiguousarray(M, dtype=np.float64),
        R=np.ascontiguousarray(R, dtype=np.float64),
    )
    return CrossResult(factors=factors, I=I, J=J, oracle_counts=counted.counts, sweeps=sweeps)


__all__ = ["CrossResult", "InitStrategy", "cross_rank_r"]
