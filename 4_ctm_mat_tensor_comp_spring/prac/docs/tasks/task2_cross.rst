Task 2 — Cross / skeleton approximation
========================================

Goal
----

Approximate a functional matrix :math:`A` by a low-rank product of selected
columns and rows:

.. math::

   A \;\approx\; C\,M\,R, \qquad
   C = A[:, J],\ \ R = A[I, :],\ \ M = A[I, J]^{+}.

The full :math:`A` is never materialised — only the chosen rows, columns,
and the pivot block.

Initialisation strategies
-------------------------

* ``"random"`` — uniform random row / column subsets.
* ``"greedy"`` — defer to Task 4 (rank-1 ACA pivots; PDF p. 3 "simplified
  greedy residual search").
* ``"maxvol"`` — start from a guess, then alternately maximise
  :math:`|\det A[I, J]|` over rows of :math:`A[:, J]` and over columns of
  :math:`A[I, :]^{\top}`. Goreinov–Tyrtyshnikov–Zamarashkin (1997) prove this
  converges to a "good" cross with bounded error to the best rank-:math:`r`
  approximation.

API
---

.. autofunction:: matcomp.matrix_algorithms.cross.cross_rank_r
.. autoclass:: matcomp.matrix_algorithms.cross.CrossResult
   :members:

Results
-------

.. image:: ../../reports/figures/task2_cross/cross_error_vs_r.png
   :alt: Cross error vs rank
   :width: 800px

The full numerical table is at ``reports/results/task2_cross_table.csv``.

Failure cases
-------------

* **Ill-conditioned pivot block.** When :math:`A[I, J]` becomes
  numerically singular (typical past the numerical rank), the pseudo-inverse
  is regularised via :func:`matcomp.utils.linalg.safe_pinv` and a
  ``RuntimeWarning`` is emitted (PDF p. 4 acceptance).
* **Random initialisation can stall.** Especially on Hilbert / Cauchy
  matrices, a random pivot can land on rows of small norm. ``"maxvol"`` is
  the recommended default.
* **Beyond numerical rank.** Cross approximation accumulates round-off
  noise from the truncated pseudo-inverse for :math:`r >`-rank. Use Task 5
  recompression to truncate back.
