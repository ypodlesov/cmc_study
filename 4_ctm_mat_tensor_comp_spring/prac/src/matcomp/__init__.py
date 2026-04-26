"""matcomp — matrix and tensor computation algorithms.

This package implements the practicum *Practical Tasks in Matrix and Tensor
Algorithms*. Tasks 1–7 (matrix) and 8–13 (tensor) live in two parallel
sub-packages with shared scaffolding under :mod:`matcomp.utils`:

* :mod:`matcomp.utils` — shared scaffolding
  (functional-matrix / functional-tensor protocols, low-rank carriers,
  oracle counters, error metrics, plotting, RNG helper).
* :mod:`matcomp.matrix_algorithms` — one module per matrix task (1–7).
* :mod:`matcomp.tensor_algorithms` — one module per tensor task (8–13).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
