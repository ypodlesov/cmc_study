Task 1 — Lanczos iteration
============================

Goal
----

Approximate the extreme eigenvalues of a symmetric matrix
:math:`A \in \mathbb{R}^{n\times n}` (or a self-adjoint linear operator) using
only a user-supplied ``matvec`` callable.

Mathematics
-----------

Starting from a unit vector :math:`q_1`, build an orthonormal Krylov basis
:math:`Q_k = (q_1, q_2, \ldots, q_k)` such that

.. math::

   A\,Q_k \;=\; Q_k\,T_k \;+\; \beta_k\,q_{k+1}\,e_k^\top,

where :math:`T_k` is symmetric tridiagonal with diagonal
:math:`\alpha_j = q_j^\top A q_j` and sub-diagonal
:math:`\beta_j = \lVert r_{j+1} \rVert_2`. The eigenvalues
:math:`\theta_1 \le \cdots \le \theta_k` of :math:`T_k` are the *Ritz values*;
they converge monotonically to the extreme eigenvalues of :math:`A`.

Algorithm (Saad, Alg. 6.5)
--------------------------

.. code-block:: text

    q_1 ← q_0 / ||q_0||
    β_0 ← 0; q_0 ← 0
    for j = 1 .. k:
        u ← A q_j
        α_j ← q_j^T u
        r ← u − α_j q_j − β_{j−1} q_{j−1}
        if reorthogonalise:
            r ← r − Q_j (Q_j^T r)        (twice for stability)
        β_j ← ||r||
        if β_j < tol:
            return (happy breakdown)
        q_{j+1} ← r / β_j

API
---

.. autofunction:: matcomp.matrix_algorithms.lanczos.lanczos
.. autoclass:: matcomp.matrix_algorithms.lanczos.LanczosResult
   :members:

Results
-------

.. image:: ../../reports/figures/task1_lanczos/ritz_convergence.png
   :alt: Ritz value convergence
   :width: 600px

.. image:: ../../reports/figures/task1_lanczos/orthogonality_loss.png
   :alt: Loss of orthogonality
   :width: 600px

The full Ritz-vs-eigh comparison table is at
``reports/results/task1_lanczos_ritz_table.csv``.

Failure cases
-------------

* **Ghost eigenvalues** appear when orthogonality is lost. Without
  reorthogonalisation, repeated Ritz copies of an already-converged
  eigenvalue accumulate. Full reorthogonalisation costs :math:`O(n k^2)`
  flops but eliminates the issue.
* **Tightly clustered spectra** slow convergence: separating the largest
  from the second-largest eigenvalue requires :math:`k`-many iterations
  proportional to :math:`1 / \log(\lambda_1 / \lambda_2)`.
* **Invariant subspaces:** if :math:`q_0` lies inside an invariant
  subspace of dimension :math:`r < n`, a happy breakdown occurs at step
  :math:`r`. The recovered Ritz pairs are exact for that subspace.
