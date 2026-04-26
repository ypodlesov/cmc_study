Task 9 — CP-ALS for a dense tensor
====================================

Goal
----

Compute a CP / canonical-polyadic decomposition of a dense d-mode tensor
:math:`X` of shape :math:`(n_1, \ldots, n_d)` at a target rank
:math:`R \ge 1`.

Mathematics
-----------

Find :math:`(\lambda, A_1, \ldots, A_d)` minimising

.. math::

   \mathcal{L} \;=\; \big\lVert X - \sum_{r=1}^{R}\, \lambda_r\, A_1[:, r]
   \circ \cdots \circ A_d[:, r]\big\rVert_F^{2}.

Alternating least squares fixes all factors except one and solves the
normal equation

.. math::

   A_n\, V_n \;=\; \mathrm{MTTKRP}_n(X, \{A_k\}),
   \qquad V_n = \bigodot_{k \neq n} (A_k^{\top} A_k) + \mathrm{reg}\,I.

After every full sweep the columns of each :math:`A_k` are renormalised
to unit :math:`\ell_2` norm and the absorbed scale is folded into
:math:`\lambda` (PDF p. 9 acceptance).

Algorithm
---------

.. code-block:: text

    initialise factors (random Gaussian or leading-R left singular
        vectors of each X_(k) via randomized_svd)
    repeat:
        for n = 0 .. d - 1:
            V_n = ⊙_{k ≠ n} (A_k^T A_k) + reg I
            A_n = solve(V_n, MTTKRP(X, factors, n))
        normalise factor columns; absorb into λ
        loss = ||X - X_CP||_F / ||X||_F   (computed via inner products,
            never materialising X_CP)
        stop when |Δloss| < tol

API
---

.. autofunction:: matcomp.tensor_algorithms.cp_als_dense.cp_als
.. autoclass:: matcomp.tensor_algorithms.cp_als_dense.CpAlsResult
   :members:

Results
-------

.. image:: ../../reports/figures/task9_cp_als/rel_err_vs_rank.png
   :alt: CP-ALS rel error vs rank
   :width: 600px

.. image:: ../../reports/figures/task9_cp_als/noise_robustness.png
   :alt: CP-ALS robustness to noise
   :width: 600px

Failure cases
-------------

* **Local minima / swamps.** ALS can stall on tensors with
  highly-correlated factor columns. SVD initialisation usually beats
  random; for stubborn cases see Task 11 (Levenberg–Marquardt).
* **Overestimated rank.** When :math:`R` exceeds the true CP rank,
  factor columns can become near-collinear and :math:`V_n` is
  ill-conditioned. The default ``reg=0.0`` falls back to
  :func:`safe_pinv`; setting ``reg=1e-8`` removes the warning.
* **Underestimated rank.** The error plateaus at the best rank-:math:`R`
  approximation error — there is no recovery without raising :math:`R`.
