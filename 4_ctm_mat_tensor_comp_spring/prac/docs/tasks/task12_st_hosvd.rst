Task 12 — ST-HOSVD
====================

Goal
----

Sequentially-truncated higher-order SVD (Vannieuwenhoven, Vandebril and
Meerbergen, *SIAM J. Sci. Comput.* 34(2), 2012). Returns a Tucker
approximation

.. math::

   X \;\approx\; G \times_1 U_1 \times_2 \cdots \times_d U_d,

where each :math:`U_k \in \mathbb{R}^{n_k \times r_k}` is orthonormal
and :math:`G \in \mathbb{R}^{r_1 \times \cdots \times r_d}` is the
core. Strictly cheaper than classical HOSVD because each SVD operates
on the *currently compressed* tensor.

Mathematics
-----------

For mode :math:`k = 0, \ldots, d-1`, take the rank-:math:`r_k` truncated
SVD of the unfolding :math:`G_{(k)}`, set :math:`U_k` to its left
singular vectors, and compress :math:`G \leftarrow G \times_k
U_k^{\top}`. With the eps stopping rule, the per-mode budget is
:math:`\epsilon^2 \lVert X\rVert_F^{2} / d` so the cumulative tail
remains below :math:`\epsilon \lVert X\rVert_F`
(Vannieuwenhoven §3.2).

Algorithm
---------

.. code-block:: text

    G ← X
    for k in mode_order:
        M = unfold(G, k)
        U_k, r_k = truncated_svd_left(M, ranks=ranks[k] or eps_budget_sq)
        G ← G ×_k U_k^T              # compress along this mode

API
---

.. autofunction:: matcomp.tensor_algorithms.st_hosvd.st_hosvd
.. autoclass:: matcomp.tensor_algorithms.st_hosvd.StHosvdResult
   :members:

Results
-------

.. image:: ../../reports/figures/task12_st_hosvd/pareto_eps.png
   :alt: ST-HOSVD compression-vs-error Pareto over eps
   :width: 600px

The full table is at ``reports/results/task12_st_hosvd_table.csv``.

Failure cases
-------------

* **Non-uniform decay across modes.** The equal-share budget
  :math:`\epsilon^2 / d` is conservative: when one mode has much
  faster spectral decay than others, the algorithm wastes budget on it
  and overshoots the global eps. The optional ``mode_order`` lets the
  user feed the slow-decay modes first.
* **Tall mode-n unfoldings.** For very tall :math:`X_{(k)}` the dense
  SVD path becomes the bottleneck; pass ``randomized=True`` to use
  :func:`matcomp.matrix_algorithms.randomized_svd.randomized_svd`
  instead.
* **Multilinear rank above unfolding rank.** Asking for
  ``ranks=(10, 4, 5)`` on a :math:`3 \times 4 \times 5` tensor clamps
  the first mode to 3; the test ``test_rank_clamped_to_unfolding_rank``
  exercises this.
