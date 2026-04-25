"""Shared utilities used by every matrix algorithm in this project.

The submodules are deliberately small and orthogonal:

* :mod:`matcomp.utils.functional_matrix` — the central :class:`FunctionalMatrix`
  protocol and concrete black-box matrix subclasses.
* :mod:`matcomp.utils.low_rank` — :class:`LowRankApprox` protocol used by Task 7
  to consume every algorithm's result uniformly, plus the tagged union of
  factor formats (``UVFactors``, ``USVFactors``, ``CMRFactors``).
* :mod:`matcomp.utils.test_matrices` — Hilbert / Cauchy / Gaussian-kernel /
  exact low-rank / low-rank+noise matrices.
* :mod:`matcomp.utils.metrics` — relative Frobenius / spectral / sample-RMSE
  errors and the compression-ratio metric.
* :mod:`matcomp.utils.counting` — :class:`CountingFunctionalMatrix` wrapper.
* :mod:`matcomp.utils.caching` — :class:`CachedFunctionalMatrix` wrapper.
* :mod:`matcomp.utils.linalg` — modified Gram–Schmidt, ``safe_pinv``,
  truncated-SVD selector.
* :mod:`matcomp.utils.plotting` — shared Matplotlib style and ``save_fig``.
* :mod:`matcomp.utils.seeding` — :func:`make_rng` for reproducible RNGs.
* :mod:`matcomp.utils.timing` — perf-counter helpers.
"""

from __future__ import annotations
