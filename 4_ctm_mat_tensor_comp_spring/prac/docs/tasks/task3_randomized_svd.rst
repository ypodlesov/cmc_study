Task 3 — Randomized SVD
========================

Goal
----

Compute an approximate rank-:math:`r` SVD :math:`A \approx U\,\Sigma\,V^{\top}`
of a functional matrix using only block products through ``matmat`` /
``rmatmat`` (PDF p. 4).

Algorithm (Halko–Martinsson–Tropp, 2011)
----------------------------------------

.. math::

    \Omega &\sim \mathcal{N}(0, 1)^{n\times (r+p)}\\
    Y &\leftarrow A\,\Omega\\
    \text{for } s = 1, \ldots, q: \quad
       Q &\leftarrow \mathrm{QR}(Y), \quad Y \leftarrow A\,(A^{\top} Q)\\
    Q &\leftarrow \mathrm{QR}(Y)\\
    B &\leftarrow Q^{\top} A\\
    \tilde U \Sigma V^{\top} &\leftarrow \mathrm{SVD}(B)\\
    U &\leftarrow Q \tilde U[:, :r]

Power iterations sharpen the basis when the singular spectrum decays
slowly. The cost is :math:`q + 1` ``matmat`` calls and :math:`q + 1`
``rmatmat`` calls in total.

API
---

.. autofunction:: matcomp.matrix_algorithms.randomized_svd.randomized_svd
.. autoclass:: matcomp.matrix_algorithms.randomized_svd.RSVDResult
   :members:

Results
-------

.. image:: ../../reports/figures/task3_rsvd/rsvd_hilbert_q.png
   :alt: rSVD on Hilbert with varying q
   :width: 600px

.. image:: ../../reports/figures/task3_rsvd/rsvd_kernel_p.png
   :alt: rSVD on kernel with varying p
   :width: 600px

Failure cases
-------------

* **Insufficient oversampling :math:`p`.** With :math:`p = 0` the random
  sketch may align poorly with the dominant singular subspace, especially
  for slow-decay spectra. Increase :math:`p` or :math:`q`.
* **Slow spectral decay.** Without power iterations (:math:`q = 0`), the
  basis includes too much weight from the spectral tail. One or two
  iterations close the gap to the SVD optimum.
