"""Reproducible random-number generators.

All randomised algorithms in :mod:`matcomp` accept a ``random_seed`` argument
and obtain their generator through :func:`make_rng`. This keeps tests and
experiments reproducible while still allowing nested calls to produce
independent randomness when needed.
"""

from __future__ import annotations

import numpy as np


def make_rng(random_seed: int | np.random.Generator | None) -> np.random.Generator:
    """Return a :class:`numpy.random.Generator` from a flexible input.

    Parameters
    ----------
    random_seed
        Either an integer seed, an existing :class:`numpy.random.Generator`
        (returned as-is so callers can spawn child streams), or ``None`` to
        get a fresh entropy-seeded generator.

    Returns
    -------
    numpy.random.Generator
        A NumPy generator suitable for ``rng.standard_normal``,
        ``rng.integers``, etc.
    """
    if isinstance(random_seed, np.random.Generator):
        return random_seed
    return np.random.default_rng(random_seed)
