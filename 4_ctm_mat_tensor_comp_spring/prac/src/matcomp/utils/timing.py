"""Lightweight wall-clock timing helpers.

The matrix tasks need to report a runtime alongside accuracy and oracle
calls. Rather than reach for a profiling library, we expose a single
context-manager that captures elapsed seconds.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter


@dataclass
class Elapsed:
    """Mutable container holding the elapsed-time measurement.

    Attributes
    ----------
    seconds
        Wall-clock seconds between context entry and exit. Set after the
        :func:`timer` block exits.
    """

    seconds: float = 0.0


@contextmanager
def timer() -> Iterator[Elapsed]:
    """Measure wall-clock time of the ``with`` body.

    Examples
    --------
    >>> with timer() as t:
    ...     pass
    >>> t.seconds >= 0.0
    True
    """
    elapsed = Elapsed()
    start = perf_counter()
    try:
        yield elapsed
    finally:
        elapsed.seconds = perf_counter() - start
