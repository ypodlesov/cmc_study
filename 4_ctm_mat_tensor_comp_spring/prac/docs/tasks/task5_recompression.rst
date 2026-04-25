Task 5 — Recompression of a low-rank decomposition
====================================================

Goal
----

Given an existing low-rank carrier (in any of the three formats
:class:`matcomp.utils.low_rank.UVFactors`,
:class:`matcomp.utils.low_rank.USVFactors`,
:class:`matcomp.utils.low_rank.CMRFactors`), return a smaller-rank
:class:`USVFactors` approximation within either a relative tolerance
``eps`` or a fixed target ``rank``.

Algorithm
---------

After reducing every input format to :math:`U V^{\top}` (folding ``S`` for
``USV`` or ``C M`` for ``CMR``):

.. math::

   U &= Q_u R_u, \qquad V = Q_v R_v\\
   R_u R_v^{\top} &= \tilde U \,\Sigma\, \tilde V^{\top}\\
   U_{\text{new}} &= Q_u \tilde U_{:,:r}, \qquad
   V^{\top}_{\text{new}} = \tilde V^{\top}_{:r,:} Q_v^{\top}

The truncation rank :math:`r` is chosen by either ``eps`` (drop singular
values below :math:`\epsilon \cdot \sigma_0`) or the user-supplied
``rank``.

API
---

.. autofunction:: matcomp.matrix_algorithms.recompression.recompress_low_rank
.. autoclass:: matcomp.matrix_algorithms.recompression.RecompressResult
   :members:

Results
-------

.. image:: ../../reports/figures/task5_recompression/recompression_cutoff.png
   :alt: Recompression cutoff vs spectrum
   :width: 600px

The before / after comparison is in
``reports/results/task5_recompression_table.csv``.

Failure cases
-------------

* **Aggressive ``eps``** (large) discards informative singular values
  and *increases* the approximation error. The PDF (p. 6) requires the
  error not to worsen by more than ``eps`` in absolute terms; this
  manifests as a corresponding ``ratio_to_svd`` jump in the benchmark.
* **Heavy-tailed spectrum + relative truncation** with small ``eps`` can
  retain numerically meaningless singular components; combining
  recompression with an absolute floor is sometimes preferable.
