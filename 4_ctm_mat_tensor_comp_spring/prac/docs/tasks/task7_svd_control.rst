Task 7 — Quality control through SVD
======================================

Goal
----

Run every method from Tasks 1–6 on a fixed set of test matrices and
ranks, and compare each method's error against the optimal SVD tail.
Produces the unified comparison table required by the PDF
pre-submission checklist (p. 13).

Reference
---------

For a small dense :math:`A` (size capped at :math:`m, n \le 1000`):

* The Frobenius-optimal rank-:math:`r` approximation has error
  :math:`\sqrt{\sum_{k > r} \sigma_k^2}` (Eckart–Young).
* The spectral-optimal rank-:math:`r` approximation has error
  :math:`\sigma_{r+1}`.
* For Task 1 (Lanczos), the reference is :func:`numpy.linalg.eigh`.

The harness records, for every (method, matrix, rank) triple, the
relative error and the ratio
:math:`\text{error}_\text{method} / \text{error}_\text{svd best}`.

API
---

.. autofunction:: matcomp.matrix_algorithms.svd_control.benchmark_against_svd
.. autoclass:: matcomp.matrix_algorithms.svd_control.BenchmarkRow
   :members:
.. autofunction:: matcomp.matrix_algorithms.svd_control.compare_eigenvalues_against_eigh
.. autofunction:: matcomp.matrix_algorithms.svd_control.reference_svd_method

SVD-isolation enforcement
-------------------------

Each non-reference method is invoked with a
:class:`matcomp.utils.counting.CountingFunctionalMatrix` wrapper. The
harness records ``oracle_calls`` for every run and surfaces it in the
output table. Methods that do not actually use the oracle interface (or
that use it more than :math:`m \cdot n` times — i.e. equivalent to
materialising the matrix) become visible as outliers.

Results
-------

.. image:: ../../reports/figures/benchmarks/benchmark_error_vs_rank.png
   :alt: Cross-method benchmark
   :width: 800px

The full table is at ``reports/results/benchmarks.csv``.

Failure cases
-------------

* **Ratio explodes near machine precision.** When the SVD tail itself is
  near :math:`10^{-16}`, the ratio :math:`\text{err}/\text{tail}` becomes
  meaningless. Reading the absolute error column is more informative
  past the numerical rank.
* **Small problems only.** The harness materialises every input. For
  large functional matrices, replace the reference SVD with
  :func:`matcomp.utils.metrics.sample_rmse` and stop emitting
  ``ratio_to_svd_*`` columns.
