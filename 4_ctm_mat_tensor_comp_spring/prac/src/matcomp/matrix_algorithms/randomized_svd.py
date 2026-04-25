r"""Task 3 — Randomized SVD for functional matrices.

Implements the randomized SVD of Halko, Martinsson and Tropp
(*SIAM Review* 53(2), 2011) using only the block-product methods of a
:class:`FunctionalMatrix`. The full :math:`A` is therefore never built.

Algorithm
---------

Given target rank :math:`r`, oversampling :math:`p \ge 0` and number of
power iterations :math:`q \ge 0`:

1. Draw a Gaussian sketch :math:`\Omega \in \mathbb{R}^{n \times (r+p)}`.
2. Form :math:`Y = A\,\Omega` via ``A.matmat(Ω)``.
3. Optional power iterations: for :math:`s = 1, \ldots, q`,

   .. math::

       Y \leftarrow A\,(A^{\top} Y),
       \qquad Y \leftarrow \mathrm{QR}(Y).

   Power iterations sharpen the basis when the singular spectrum decays
   slowly.
4. Orthonormal basis :math:`Q = \mathrm{QR}(Y)`.
5. Form :math:`B = Q^{\top} A` via ``A.rmatmat(Q).T``, compute
   :math:`B = \tilde U_B\,\Sigma\,V^{\top}`, and return
   :math:`U = Q\,\tilde U_B[:, :r]`, :math:`S = \Sigma[:r]`,
   :math:`V^{\top} = V^{\top}[:r, :]`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from matcomp.utils.counting import CountingFunctionalMatrix, OracleCounts
from matcomp.utils.functional_matrix import FloatArray, FunctionalMatrix
from matcomp.utils.low_rank import USVFactors
from matcomp.utils.seeding import make_rng


@dataclass(frozen=True)
class RSVDResult:
    r"""Output of :func:`randomized_svd`.

    Attributes
    ----------
    factors
        :class:`USVFactors` carrier with the requested rank ``r``.
    oracle_counts
        Counts of each oracle method invoked by the algorithm. ``matmat``
        and ``rmatmat`` are the only methods touched.
    """

    factors: USVFactors
    oracle_counts: OracleCounts

    # ---- LowRankApprox passthroughs ---------------------------------------

    @property
    def rank(self) -> int:
        return self.factors.rank

    @property
    def shape(self) -> tuple[int, int]:
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


def randomized_svd(
    A: FunctionalMatrix,
    r: int,
    *,
    oversampling: int = 10,
    n_power_iter: int = 0,
    random_seed: int | None = None,
) -> RSVDResult:
    r"""Halko–Martinsson–Tropp randomized SVD.

    Parameters
    ----------
    A
        Functional matrix. Only ``A.matmat`` and ``A.rmatmat`` are used,
        each at most ``n_power_iter + 1`` times.
    r
        Target rank.
    oversampling
        Extra random columns (``p``); the working sketch has width
        ``r + p``. Typical default is 10.
    n_power_iter
        Number of subspace power iterations (``q`` in the paper). Each
        iteration sharpens the spectral content of the basis at the cost
        of one ``A.matmat`` and one ``A.rmatmat`` call.
    random_seed
        Seed for the Gaussian sketch.

    Returns
    -------
    RSVDResult
        ``USVFactors`` of rank ``r`` plus oracle counts.
    """
    if r < 1:
        raise ValueError("r must be at least 1")
    if oversampling < 0:
        raise ValueError("oversampling must be non-negative")
    if n_power_iter < 0:
        raise ValueError("n_power_iter must be non-negative")

    counted = CountingFunctionalMatrix(A)
    m, n = counted.shape
    if r > min(m, n):
        raise ValueError(f"r={r} exceeds min(shape)={min(m, n)}")

    rng = make_rng(random_seed)
    sketch_width = min(r + oversampling, n)
    Omega = rng.standard_normal((n, sketch_width)).astype(np.float64)
    Y = counted.matmat(Omega)

    # Power iterations with QR-based stabilisation between every multiplication.
    for _ in range(n_power_iter):
        Q, _ = np.linalg.qr(Y, mode="reduced")
        Z = counted.rmatmat(Q)
        Q, _ = np.linalg.qr(Z, mode="reduced")
        Y = counted.matmat(Q)

    Q, _ = np.linalg.qr(Y, mode="reduced")
    # B = Q^T A — we use rmatmat(Q) which returns A^T Q (shape n x k); transpose to get B.
    B = counted.rmatmat(Q).T  # shape (k, n)
    U_b, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_b

    factors = USVFactors(
        U=np.ascontiguousarray(U[:, :r], dtype=np.float64),
        S=np.ascontiguousarray(S[:r], dtype=np.float64),
        Vt=np.ascontiguousarray(Vt[:r, :], dtype=np.float64),
    )
    return RSVDResult(factors=factors, oracle_counts=counted.counts)


__all__ = ["RSVDResult", "randomized_svd"]
