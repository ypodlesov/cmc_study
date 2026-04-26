Task 10 — CP via differentiable optimisation
==============================================

Goal
----

Minimise the CP reconstruction loss by gradient descent on the factor
matrices (and the rank weights). Two evaluation modes:

* **dense** — :math:`X` is a full ``ndarray`` and the loss is the
  per-element MSE.
* **batched** — :math:`X` is a :class:`FunctionalTensor` and the loss is
  estimated on a random multi-index batch each step.

Mathematics
-----------

Loss:

.. math::

   \mathcal{L}(\theta) \;=\; \sum_{(i_1, \ldots, i_d) \in \Omega}
   \big(X[i_1, \ldots, i_d] - \sum_r \lambda_r \prod_k B_k[i_k, r]\big)^{2}.

Closed-form gradients (no autograd needed):

.. math::

   \nabla_{B_n}\mathcal{L} \;=\; \frac{2}{|\Omega|}\,
   \big(B_n\,\mathrm{diag}(\lambda)\,V_n - M_n\big)\,\mathrm{diag}(\lambda),
   \qquad V_n = \bigodot_{k \neq n} (B_k^{\top} B_k),

.. math::

   \nabla_{\lambda}\mathcal{L} = \frac{2}{|\Omega|}\,(W\,\lambda - p),
   \qquad W = \bigodot_k (B_k^{\top} B_k),\quad
   p_r = \sum_i M_0[i, r]\,B_0[i, r].

Adam (Kingma & Ba 2015) and SGD are bundled. The implementation is pure
NumPy: the project's runtime dependency surface stays at
``numpy + matplotlib``.

Algorithm
---------

.. code-block:: text

    model = CPModel(shape, rank, init=...)        # random or SVD-warm-start
    state_m, state_v = zeros_like(theta)          # Adam moments
    for step = 1 .. max_iter:
        if dense:
            grad = model.grad_dense(X)
        else:
            idx = random multi-indices(batch_size)
            values = X.samples(idx)               # only oracle interaction
            grad = model.grad_batch(idx, values)
        delta = adam_step(grad, state_m, state_v, lr, β1, β2, eps, step)
        theta -= delta

API
---

.. autoclass:: matcomp.tensor_algorithms.cp_neural.CPModel
   :members:
.. autofunction:: matcomp.tensor_algorithms.cp_neural.fit_cp_neural
.. autoclass:: matcomp.tensor_algorithms.cp_neural.CpNeuralResult
   :members:

Results
-------

.. image:: ../../reports/figures/task10_cp_neural/loss_curves.png
   :alt: CP-NN vs CP-ALS loss curves
   :width: 600px

.. image:: ../../reports/figures/task10_cp_neural/batched_loss.png
   :alt: Batched CP-NN loss vs steps for varying batch size
   :width: 600px

Failure cases
-------------

* **Sluggish convergence on stiff problems** — Adam's first-order step
  is much slower than ALS's Newton step on well-conditioned tensors.
  Use Task 9 (ALS) when applicable; reserve this task for
  functional-tensor inputs where ALS would require densifying.
* **Permutation / scaling ambiguity** — the result factors agree with
  the truth only up to column permutation and per-mode rescaling that
  preserves the rank-1 outer products. Tests assert the *reconstruction*
  error, not factor identity.
* **Batch-size choice** — too small and the gradient noise dominates;
  too large and the oracle savings disappear. The experiment script
  sweeps ``batch_size`` so the report can pick a working value.

Cross-link
----------

For a second-order alternative, see :doc:`task11_cp_lm`.
