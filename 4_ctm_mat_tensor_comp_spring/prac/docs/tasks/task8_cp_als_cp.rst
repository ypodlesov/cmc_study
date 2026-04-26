Task 8 — CP-ALS from a CP source
==================================

Goal
----

Given a tensor already in canonical (CP) form
:math:`X = \sum_r \lambda^{(s)}_r \bigotimes_k A^{(s)}_k[:, r]` of source rank
:math:`R_{\text{src}}`, find a CP approximation of a smaller target rank
:math:`R_{\text{tgt}} < R_{\text{src}}` *without* densifying :math:`X`.

Mathematics
-----------

The ALS update for the target factor :math:`B_n` solves

.. math::

   B_n\,V \;=\; M, \qquad
   V \;=\; \bigodot_{k \neq n} B_k^{\top} B_k, \quad
   M \;=\; \mathrm{MTTKRP}_n(X, \{B_k\}).

When :math:`X` is itself in CP form, the right-hand side is

.. math::

   M[i, s] \;=\; \sum_{r=1}^{R_{\text{src}}}\, \lambda^{(s)}_r\,
   A^{(s)}_n[i, r] \, \prod_{k \neq n}
   \big(A^{(s)}_k[:, r]^{\top} B_k[:, s]\big),

i.e. it reads off the source factors directly with no :math:`X_{(n)}` ever
constructed. Loss tracking uses the CP–CP inner-product identity

.. math::

   \lVert X_{S} - X_{T}\rVert_F^{2}
   = \lVert X_{S}\rVert_F^{2} - 2\,\langle X_{S}, X_{T}\rangle
     + \lVert X_{T}\rVert_F^{2},

with each term computed via factor Hadamard Grams.

Algorithm
---------

.. code-block:: text

    initialise (λ, B_1, ..., B_d) — random or "source"-warm-started
    repeat:
        for n = 0 .. d - 1:
            V_n = ⊙_{k ≠ n} (B_k^T B_k) + reg I
            M_n = mttkrp_cp_cp(λ_S, A_S, B, n)        # never densifies X
            B_n = solve(V_n, M_n)
        normalise factor columns; absorb into λ
        loss = cp_cp_relative_error(λ_S, A_S, λ, B)   # CP-vs-CP only
        stop when |loss_{k} - loss_{k-1}| < tol

API
---

.. autofunction:: matcomp.tensor_algorithms.cp_als_cp.cp_als_from_cp
.. autoclass:: matcomp.tensor_algorithms.cp_als_cp.CpAlsCpResult
   :members:

Results
-------

.. image:: ../../reports/figures/task8_cp_als_cp/loss_vs_iter.png
   :alt: CP-ALS-from-CP loss vs sweep
   :width: 600px

The full sweep table is written to
``reports/results/task8_cp_als_cp_table.csv``.

Failure cases
-------------

* When :math:`R_{\text{tgt}}` is much smaller than the underlying
  numerical CP rank, ALS may stall in a "swamp" where the loss decreases
  by only a tiny amount each sweep. The CP-LM module (Task 11) is the
  recommended escape; it accepts a CP warm-start.
* The "source"-warm-start picks the strongest source columns by
  :math:`|\lambda^{(s)}_r|`. If the source weights are uninformative
  (e.g. all unity), this collapses to a near-random init.
