"""Shared fixtures for tensor-side tests.

The matrix-side ``tests/conftest.py`` already exports the seed-parametrised
``rng`` fixture; pytest auto-discovers it from the parent directory, so we
only add tensor-specific fixtures here.
"""

from __future__ import annotations

import numpy as np
import pytest

from matcomp.utils.tensor_low_rank import CPFactors, TuckerFactors
from matcomp.utils.tensor_test_objects import (
    CPSyntheticTensor,
    FunctionTensor,
    GaussianKernel3DTensor,
    Hilbert3DTensor,
    LowRankPlusNoiseTensor,
    TuckerSyntheticTensor,
    random_cp,
    random_tucker,
)


@pytest.fixture
def dense_3d_random(rng: np.random.Generator) -> np.ndarray:
    """6×7×8 Gaussian tensor."""
    return rng.standard_normal((6, 7, 8)).astype(np.float64)


@pytest.fixture
def dense_4d_small(rng: np.random.Generator) -> np.ndarray:
    """4×5×4×3 Gaussian tensor — used by TT-cross / ST-HOSVD."""
    return rng.standard_normal((4, 5, 4, 3)).astype(np.float64)


@pytest.fixture
def cp_synthetic_factors(rng: np.random.Generator) -> CPFactors:
    """Exact rank-3 CP factorisation over (6, 7, 8)."""
    weights, factors = random_cp((6, 7, 8), rank=3, rng=rng)
    return CPFactors(weights=weights, factors=tuple(factors))


@pytest.fixture
def cp_synthetic_tensor(cp_synthetic_factors: CPFactors) -> CPSyntheticTensor:
    """Functional-tensor wrapper around the same rank-3 CP."""
    return CPSyntheticTensor(
        weights=cp_synthetic_factors.weights, factors=list(cp_synthetic_factors.factors)
    )


@pytest.fixture
def tucker_synthetic_factors(rng: np.random.Generator) -> TuckerFactors:
    """Exact Tucker factorisation with multilinear rank (2, 3, 2)."""
    core, factors = random_tucker((6, 7, 8), (2, 3, 2), rng)
    return TuckerFactors(core=core, factors=tuple(factors))


@pytest.fixture
def tucker_synthetic_tensor(
    tucker_synthetic_factors: TuckerFactors,
) -> TuckerSyntheticTensor:
    """Functional-tensor wrapper around the rank-(2,3,2) Tucker."""
    return TuckerSyntheticTensor(
        core=tucker_synthetic_factors.core,
        factors=list(tucker_synthetic_factors.factors),
    )


@pytest.fixture
def kernel_3d_tensor(rng: np.random.Generator) -> GaussianKernel3DTensor:
    """8×8×8 Gaussian-kernel tensor (smooth, rapid multilinear-rank decay)."""
    x = rng.uniform(0.0, 1.0, size=8).astype(np.float64)
    y = rng.uniform(0.0, 1.0, size=8).astype(np.float64)
    z = rng.uniform(0.0, 1.0, size=8).astype(np.float64)
    return GaussianKernel3DTensor(x, y, z, sigma=0.5)


@pytest.fixture
def low_rank_plus_noise_3d(rng: np.random.Generator) -> LowRankPlusNoiseTensor:
    """Rank-3 CP signal of size (6, 7, 8) plus 1e-3 Gaussian noise."""
    weights, factors = random_cp((6, 7, 8), rank=3, rng=rng)
    return LowRankPlusNoiseTensor(
        weights, factors, noise_level=1e-3,
        random_seed=int(rng.integers(0, 2**32 - 1)),
    )


@pytest.fixture
def hilbert_3d() -> Hilbert3DTensor:
    """8×8×8 Hilbert tensor 1/(i+j+k+1) — slow multilinear-rank decay."""
    return Hilbert3DTensor(8)


@pytest.fixture
def function_tensor_oracle() -> FunctionTensor:
    """Generic functional-tensor oracle wrapping a closed-form rule."""
    return FunctionTensor(
        func=lambda idx: 1.0 / (idx[0] + idx[1] + idx[2] + 1.0),
        shape=(6, 6, 6),
    )
