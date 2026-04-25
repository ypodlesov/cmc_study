Task 4 — Adaptive cross approximation with caching
====================================================

Goal
----

Build a low-rank approximation by iteratively choosing pivots on the
running residual:

.. math::

   \hat A \;=\; \sum_{k=1}^{r} \frac{u_k\,v_k^{\top}}{\delta_k},

where :math:`(i_k, j_k)` is the next pivot, :math:`u_k` is the
corresponding residual column, :math:`v_k` is the residual row, and
:math:`\delta_k = (\hat A_{k-1} - A)_{i_k, j_k}` is the pivot value
(Bebendorf & Rjasanow, 2003).

Caching
-------

The wrapped functional matrix is decorated with
:class:`matcomp.utils.caching.CachedFunctionalMatrix`, so repeated row /
column / entry queries do not re-execute the user-supplied oracle. The
counting wrapper (:class:`matcomp.utils.counting.CountingFunctionalMatrix`)
sits *underneath* the cache, so cache hits do not increment the real
oracle counter — this matches the PDF's "Repeated requests must not
increase the counter of real computations" acceptance line (p. 5).

API
---

.. autofunction:: matcomp.matrix_algorithms.adaptive_cross.adaptive_cross_cached
.. autoclass:: matcomp.matrix_algorithms.adaptive_cross.ACAResult
   :members:

Results
-------

.. image:: ../../reports/figures/task4_aca/aca_error_vs_max_rank.png
   :alt: ACA error vs max_rank
   :width: 600px

.. image:: ../../reports/figures/task4_aca/aca_cache_ratio.png
   :alt: Cache hit ratio
   :width: 600px

Failure cases
-------------

* **Poor initial pivot.** A randomly chosen starting row may sit in a
  region of small absolute values, producing a tiny first pivot
  :math:`\delta_1` and a misleading early residual estimate.
* **Greedy is myopic.** ACA selects the locally best pivot, not the
  globally best one. For higher accuracy at fixed rank, use Task 2 with
  ``init="maxvol"``.
