"""Shared utilities used by every matrix and tensor algorithm in this project.

Matrix-side (Tasks 1–7):

* :mod:`matcomp.utils.functional_matrix` — the central :class:`FunctionalMatrix`
  protocol and concrete black-box matrix subclasses.
* :mod:`matcomp.utils.low_rank` — :class:`LowRankApprox` protocol used by Task 7
  to consume every algorithm's result uniformly, plus the tagged union of
  factor formats (``UVFactors``, ``USVFactors``, ``CMRFactors``).
* :mod:`matcomp.utils.test_matrices` — Hilbert / Cauchy / Gaussian-kernel /
  exact low-rank / low-rank+noise matrices.
* :mod:`matcomp.utils.metrics` — relative Frobenius / spectral / sample-RMSE
  errors and the compression-ratio metric.

Tensor-side (Tasks 8–13):

* :mod:`matcomp.utils.functional_tensor` — :class:`FunctionalTensor` protocol
  and the d-mode :class:`DenseTensor` adapter.
* :mod:`matcomp.utils.tensor_linalg` — unfold / fold / mode-n product /
  Khatri–Rao / MTTKRP / CP inner product / TT contraction / multilinear
  rank.
* :mod:`matcomp.utils.tensor_low_rank` — :class:`LowRankTensor` protocol
  and the carriers ``CPFactors``, ``TuckerFactors``, ``TTFactors``.
* :mod:`matcomp.utils.tensor_test_objects` — CP / Tucker / Hilbert-3D /
  Gaussian-kernel-3D / function-oracle test tensors.
* :mod:`matcomp.utils.tensor_metrics` — tensor-side relative error,
  sample RMSE and compression metrics.

Cross-cutting:

* :mod:`matcomp.utils.counting` — :class:`CountingFunctionalMatrix` and
  :class:`CountingFunctionalTensor` wrappers.
* :mod:`matcomp.utils.caching` — :class:`CachedFunctionalMatrix` and
  :class:`CachedFunctionalTensor` wrappers.
* :mod:`matcomp.utils.linalg` — modified Gram–Schmidt, ``safe_pinv``,
  truncated-SVD selector, ``maxvol``.
* :mod:`matcomp.utils.plotting` — shared Matplotlib style and ``save_fig``.
* :mod:`matcomp.utils.seeding` — :func:`make_rng` for reproducible RNGs.
* :mod:`matcomp.utils.timing` — perf-counter helpers.
"""

from __future__ import annotations
