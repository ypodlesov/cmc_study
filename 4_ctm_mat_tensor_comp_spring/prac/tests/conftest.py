"""Shared pytest fixtures for the matcomp test suite.

Each task's tests build their own task-specific inputs, but anything reused
across tasks (RNG, low-rank synthetic matrices, Hilbert / Cauchy /
Gaussian-kernel test matrices) lives here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow `import matcomp` without an installed package — useful for `pytest`
# in a clean checkout before `pip install -e .`.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from matcomp.utils.seeding import make_rng  # noqa: E402
from matcomp.utils.test_matrices import (  # noqa: E402
    CauchyMatrix,
    GaussianKernelMatrix,
    HilbertMatrix,
    LowRankMatrix,
    LowRankPlusNoise,
)

SEEDS = [0, 17, 12345]


@pytest.fixture(params=SEEDS, ids=[f"seed={s}" for s in SEEDS])
def rng(request: pytest.FixtureRequest) -> np.random.Generator:
    """Reproducible NumPy generator parametrised over a small seed set."""
    return make_rng(request.param)


@pytest.fixture
def hilbert_small() -> HilbertMatrix:
    """20×20 Hilbert matrix (slowly decaying spectrum)."""
    return HilbertMatrix(20, 20)


@pytest.fixture
def cauchy_small(rng: np.random.Generator) -> CauchyMatrix:
    """20×24 Cauchy matrix with random positive nodes."""
    x = 1.0 + rng.uniform(0.0, 1.0, size=20)
    y = 0.5 + rng.uniform(0.0, 1.0, size=24)
    return CauchyMatrix(x, y)


@pytest.fixture
def kernel_small(rng: np.random.Generator) -> GaussianKernelMatrix:
    """16×16 Gaussian-kernel matrix on random 1-D nodes."""
    x = rng.uniform(0.0, 1.0, size=16)
    y = rng.uniform(0.0, 1.0, size=16)
    return GaussianKernelMatrix(x, y, sigma=0.4)


@pytest.fixture
def low_rank_3(rng: np.random.Generator) -> LowRankMatrix:
    """Exact rank-3 matrix of size 18×22."""
    U = rng.standard_normal((18, 3))
    V = rng.standard_normal((22, 3))
    return LowRankMatrix(U, V)


@pytest.fixture
def noisy_low_rank(rng: np.random.Generator) -> LowRankPlusNoise:
    """Rank-3 signal of size 18×22 plus 1e-3 Gaussian noise."""
    U = rng.standard_normal((18, 3))
    V = rng.standard_normal((22, 3))
    return LowRankPlusNoise(U, V, noise_level=1e-3, random_seed=int(rng.integers(0, 2**32 - 1)))
