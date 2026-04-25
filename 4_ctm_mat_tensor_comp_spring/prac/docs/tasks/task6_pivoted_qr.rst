Task 6 — Pivoted (rank-revealing) QR
=====================================

Goal
----

Compute :math:`A\,P = Q\,R` with column permutation :math:`P` chosen so
that the diagonal :math:`|R_{kk}|` is monotonically non-increasing. The
estimated numerical rank is the largest :math:`r` with
:math:`|R_{rr}| / |R_{11}| > \text{tol}`. The rank-:math:`r` approximation
is :math:`Q[:, :r]\,R[:r, :]\,P^{-1}`.

Algorithm (Chan, 1987 / Gu & Eisenstat, 1996)
---------------------------------------------

At step :math:`k`, pick the column with the largest remaining 2-norm
in :math:`A[k:, :]` and swap it into position :math:`k`. Apply a
Householder reflector to zero out the lower entries of the new column.
The per-column squared norms are *downdated* so each step costs
:math:`O(m\,n)` rather than the :math:`O(m\,n\,k)` of naive
recomputation. A periodic re-norm guards against round-off in the
downdate.

API
---

.. autofunction:: matcomp.matrix_algorithms.pivoted_qr.pivoted_qr_approx
.. autoclass:: matcomp.matrix_algorithms.pivoted_qr.PivotedQRResult
   :members:

Results
-------

.. image:: ../../reports/figures/task6_pivoted_qr/pivoted_qr_diag_vs_sigma.png
   :alt: |R_kk| vs sigma_k
   :width: 800px

.. image:: ../../reports/figures/task6_pivoted_qr/pivoted_qr_vs_svd.png
   :alt: QR error vs SVD optimum
   :width: 600px

Failure cases
-------------

* **Strong rank-revealing failure.** Standard column-pivoted QR can fail
  to reveal the rank in adversarial Kahan-style examples. Strong RRQR
  (Gu–Eisenstat) supplies an explicit guarantee but at higher cost; that
  variant is left as an optional extension.
* **Sign / column-tie ambiguities.** When two columns have equal residual
  norm, the choice of pivot is arbitrary. The deterministic test against
  ``scipy.linalg.qr(pivoting=True)`` therefore compares
  :math:`|R_{kk}|` rather than the permutations themselves.
