Task 11 — Levenberg–Marquardt for CP
======================================

Goal
----

Treat CP fitting as nonlinear least squares and solve it by the
Levenberg–Marquardt method. Comparison target: Task 9 (ALS) and Task 10
(Adam).

Mathematics
-----------

Parameter vector :math:`\theta` packs all weights and factor entries:

.. math::

   \theta \;=\; \big(\lambda;\, \mathrm{vec}(B_1);\, \ldots;\, \mathrm{vec}(B_d)\big),
   \qquad r(\theta) \;=\; \mathrm{vec}\big(X - X_{CP}(\theta)\big),
   \qquad \mathcal{L}(\theta) = \tfrac{1}{2}\,\lVert r(\theta)\rVert_2^{2}.

The Jacobian :math:`J = \partial r / \partial \theta` is sparse but
written down in closed form: column :math:`\partial r / \partial B_n[i, r]`
is :math:`-\lambda_r\,\mathbf{e}_i^{(n)} \otimes \prod_{k \neq n} B_k[:, r]`.

LM step solves the regularised normal equation

.. math::

   (J^{\top}J + \mu\,I)\,\delta \;=\; -J^{\top} r,

with damping :math:`\mu` increased on rejected steps and decreased on
accepted ones (Marquardt update).

Algorithm
---------

.. code-block:: text

    initialise θ via ALS warm-up (default als_warmup = 5 sweeps)
    μ ← μ_init
    repeat for max_iter:
        J = jacobian(θ);  r = residual(θ)
        δ = solve((J^T J + μ I), -J^T r)
        if 0.5 ||r(θ + δ)||² < 0.5 ||r(θ)||²:
            θ ← θ + δ;  μ ← max(μ * mu_dec, μ_min);  accepted += 1
        else:
            μ ← min(μ * mu_inc, μ_max);  rejected += 1

API
---

.. autofunction:: matcomp.tensor_algorithms.cp_lm.cp_levenberg_marquardt
.. autoclass:: matcomp.tensor_algorithms.cp_lm.CpLmResult
   :members:

Results
-------

.. image:: ../../reports/figures/task11_cp_lm/loss_curves.png
   :alt: LM vs ALS on the swamp tensor
   :width: 600px

.. image:: ../../reports/figures/task11_cp_lm/lm_accept_reject.png
   :alt: LM accept / reject step counts
   :width: 600px

The reported Jacobian dimensions and step counts land in
``reports/results/task11_cp_lm_table.csv``.

Failure cases
-------------

* **Rank-deficient Jacobian.** When two factor columns coincide, the
  block of :math:`J` for that rank component is rank deficient. With
  ``mu_init`` very small, :math:`J^{\top}J + \mu I` is poorly
  conditioned and Cholesky fails — :func:`safe_pinv` is the fallback,
  emitting a :class:`RuntimeWarning`. The remedy is to increase
  ``mu_init`` (e.g. 1e-3 → 1.0) or to add an explicit ``reg`` to the
  ALS warm-up.
* **Memory.** The dense Jacobian is :math:`O(N \cdot p)` where
  :math:`N = \prod n_k` and :math:`p = R\,(\sum_k n_k + 1)`. For large
  tensors prefer Task 10 (CP-NN) which never builds a Jacobian.
* **Oscillation.** Setting ``mu_inc`` too large can cause damping
  blow-up. The default 10× / 0.3× pair from Sorber et al. usually
  works.
