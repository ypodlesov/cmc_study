Task 13 — TT-cross
====================

Goal
----

Construct a tensor-train approximation of a black-box d-mode functional
tensor :math:`X` using a small number of element queries — never build
the full :math:`X`. Reference: Oseledets & Tyrtyshnikov,
*Linear Algebra Appl.* 432(1), 2010, pp. 70–88.

Mathematics
-----------

The TT format represents :math:`X` by ``d`` cores
:math:`G_k \in \mathbb{R}^{r_{k-1} \times n_k \times r_k}` with
:math:`r_0 = r_d = 1`:

.. math::

   X[i_1, \ldots, i_d]
   \;=\; G_1[\,:\,, i_1, \,:\,]\,G_2[\,:\,, i_2, \,:\,] \cdots G_d[\,:\,, i_d, \,:\,].

TT-cross builds the cores via *nested cross indices*: at each boundary
between mode :math:`k - 1` and mode :math:`k`, a left-index list
:math:`I^{(k)} \subset \prod_{j < k}\{0, \ldots, n_j - 1\}` of size
:math:`r_k` and a right-index list :math:`J^{(k)} \subset
\prod_{j > k}\{0, \ldots, n_j - 1\}` of size :math:`r_k` are maintained.
The supercore matrix :math:`B_k = X\big[I^{(k-1)} \times \{i_k\},
J^{(k)}\big]` of shape :math:`(r_{k-1} n_k, r_k)` is queried; rows are
selected by :func:`matcomp.utils.linalg.maxvol`; the core is

.. math::

   G_k \;=\; B_k\,\big[B_k(\text{chosen rows})\big]^{-1}.

Algorithm
---------

.. code-block:: text

    init random nested right_idx; left_idx empty
    for sweep = 1 .. max_sweeps:
        for k = 0 .. d - 2:
            B = oracle.samples(left_idx[k] ⊗ {i_k} ⊗ right_idx[k])
            chosen_rows = maxvol(B, max_iter=20)
            left_idx[k+1] = compose(left_idx[k], chosen_rows)
            G_k = B @ pinv(B[chosen_rows, :])
        G_{d-1} = oracle samples for last block
        randomise right_idx for the next sweep
        sample-error stopping check

The wrapper composition is ``Cached(Counting(X))`` — repeated queries
do not bump the real-oracle counter (PDF p. 5 acceptance:
"repeated requests must not increase the counter of real
computations").

API
---

.. autofunction:: matcomp.tensor_algorithms.tt_cross.tt_cross
.. autoclass:: matcomp.tensor_algorithms.tt_cross.TtCrossResult
   :members:

Results
-------

.. image:: ../../reports/figures/task13_tt_cross/err_vs_sweeps.png
   :alt: TT-cross relative error vs sweeps
   :width: 600px

.. image:: ../../reports/figures/task13_tt_cross/oracle_vs_err.png
   :alt: TT-cross oracle calls vs error
   :width: 600px

Failure cases
-------------

* **Insufficient ranks.** When the requested TT ranks are smaller than
  the true numerical TT rank, the error plateaus at the best
  rank-:math:`r` reachable approximation. Adaptive rank growth is an
  optional extension (PDF p. 12).
* **Unlucky random init.** The very first L→R sweep may produce poor
  cores if the initial random right-index lists do not span the
  important right subspaces. Subsequent sweeps re-randomise the right
  indices; ``max_sweeps >= 5`` is the recommended default for
  reasonable error.
* **Pivot-block ill-conditioning.** :func:`safe_pinv` rescues
  near-singular pivot blocks but can lose precision. For functional
  tensors with very tight low-rank structure (e.g. exact TT format),
  raising ``max_sweeps`` lets the maxvol heuristic refine the index
  choices and recover machine-precision accuracy.
