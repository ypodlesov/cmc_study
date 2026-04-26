# matcomp — Matrix and tensor algorithms practicum (CMC)

Reference implementation of matrix tasks **1–7** and tensor tasks
**8–13** from the practicum *Practical Tasks in Matrix and Tensor
Algorithms* (see `../practice_task_settings.pdf`).

The implementation prioritises:

- **Clarity** — one file per task, shared scaffolding under `src/matcomp/utils/`.
- **Type safety** — `mypy --strict` passes.
- **Reproducibility** — every randomised routine takes `random_seed`.
- **Visibility** — every algorithm has an experiment script that produces
  plots and CSV summaries under `reports/`.

## Quick start

```bash
cd prac
make venv                          # create .venv/
source .venv/bin/activate
make install                       # editable install with dev + docs extras
make test                          # run the unit tests
make figures                       # produce all task plots in reports/figures/
make bench                         # produce the cross-task benchmark (Task 7)
```

## Repository tour

| Path | Purpose |
|------|---------|
| `src/matcomp/utils/` | Shared scaffolding: `FunctionalMatrix` / `FunctionalTensor`, `LowRankApprox` / `LowRankTensor`, oracle counting/caching wrappers, tensor primitives (unfold/Khatri–Rao/MTTKRP/CP–CP inner product/TT contraction), error metrics, plotting style, RNG helper. |
| `src/matcomp/matrix_algorithms/` | One module per matrix task (1–7). |
| `src/matcomp/tensor_algorithms/` | One module per tensor task (8–13). |
| `tests/matrix_algorithms/` | One pytest file per matrix task; shared fixtures in `tests/conftest.py`. |
| `tests/tensor_algorithms/` | One pytest file per tensor task plus `test_tensor_linalg.py`; shared tensor fixtures in `tests/tensor_algorithms/conftest.py`. |
| `experiments/matrix/` | Runnable scripts that produce the plots and CSV tables for matrix tasks. |
| `experiments/tensor/` | Runnable scripts that produce the plots and CSV tables for tensor tasks (incl. `run_tensor_benchmarks.py`). |
| `reports/figures/` | Output PNGs (organised by task). |
| `reports/results/` | Output CSV / JSON tables. |
| `docs/` | Sphinx sources (numpydoc + MathJax). |

## Task index

| # | Task | Module | Public function | Experiment | Tests |
|---|------|--------|-----------------|------------|-------|
| 1 | Lanczos for symmetric / self-adjoint operators | `matrix_algorithms/lanczos.py` | `lanczos(matvec, n, k, …)` | `run_lanczos.py` | `test_lanczos.py` |
| 2 | Cross / skeleton approximation, arbitrary rank `r` | `matrix_algorithms/cross.py` | `cross_rank_r(A, r, …)` | `run_cross.py` | `test_cross.py` |
| 3 | Randomized SVD for functional matrices | `matrix_algorithms/randomized_svd.py` | `randomized_svd(A, r, …)` | `run_randomized_svd.py` | `test_randomized_svd.py` |
| 4 | Adaptive cross with caching | `matrix_algorithms/adaptive_cross.py` | `adaptive_cross_cached(A, …)` | `run_adaptive_cross.py` | `test_adaptive_cross.py` |
| 5 | Recompression of low-rank decompositions | `matrix_algorithms/recompression.py` | `recompress_low_rank(factors, …)` | `run_recompression.py` | `test_recompression.py` |
| 6 | Pivoted (rank-revealing) QR | `matrix_algorithms/pivoted_qr.py` | `pivoted_qr_approx(A, …)` | `run_pivoted_qr.py` | `test_pivoted_qr.py` |
| 7 | Quality control through SVD (cross-task benchmark) | `matrix_algorithms/svd_control.py` | `benchmark_against_svd(matrices, methods, ranks)` | `run_matrix_benchmarks.py` | `test_svd_control.py` |
| 8 | CP-ALS from a CP source | `tensor_algorithms/cp_als_cp.py` | `cp_als_from_cp(source, target_rank, …)` | `run_cp_als_cp.py` | `test_cp_als_cp.py` |
| 9 | CP-ALS, dense | `tensor_algorithms/cp_als_dense.py` | `cp_als(X, rank, …)` | `run_cp_als_dense.py` | `test_cp_als_dense.py` |
| 10 | CP via differentiable optimisation | `tensor_algorithms/cp_neural.py` | `fit_cp_neural(target, rank, …)`, `CPModel` | `run_cp_neural.py` | `test_cp_neural.py` |
| 11 | CP via Levenberg–Marquardt | `tensor_algorithms/cp_lm.py` | `cp_levenberg_marquardt(X, rank, …)` | `run_cp_lm.py` | `test_cp_lm.py` |
| 12 | ST-HOSVD (Tucker) | `tensor_algorithms/st_hosvd.py` | `st_hosvd(X, ranks=…\| eps=…)` | `run_st_hosvd.py` | `test_st_hosvd.py` |
| 13 | TT-cross for functional tensors | `tensor_algorithms/tt_cross.py` | `tt_cross(X, ranks=…)` | `run_tt_cross.py` | `test_tt_cross.py` |

The Sphinx site (`make docs`) renders one page per task with the full math,
algorithm pseudocode, API, results, and a *failure cases* discussion.

### Task 1 — Lanczos

Iterative method for extreme eigenvalues of a symmetric or self-adjoint
operator. Works through a `matvec` callable; the algorithm never requests
individual matrix entries. Optional full reorthogonalisation; happy-breakdown
detection. Reports Ritz values, the tridiagonal `T_k`, and the orthonormal
basis when reorthogonalised.

### Task 2 — Cross / skeleton approximation

Low-rank approximation `Â = C M R` with `C = A[:, J]`, `R = A[I, :]`,
`M = pinv(A[I, J])`. Uses the `FunctionalMatrix` interface so the full matrix
is never materialised. Three initialisation strategies: `random`, `maxvol`,
and `greedy` (which delegates to Task 4's rank-1 ACA pivot).

### Task 3 — Randomized SVD

Halko–Martinsson–Tropp randomized SVD. Uses block products
`A.matmat(Ω)` / `A.rmatmat(Q)` so the cost is one block call per power
iteration regardless of `r`. Produces `U, S, V^T`.

### Task 4 — Adaptive cross approximation with caching

Iterative low-rank construction `Â = Σ_k u_k v_k^T / pivot_k`. Wraps the
oracle in a `CachedFunctionalMatrix` so repeated row / column queries cost
nothing. Reports cache-hit ratio and the residual decay.

### Task 5 — Recompression

Takes an existing low-rank factorisation in any of three formats —
`U V^T`, `U S V^T`, `C M R` — and returns the truncated `U S V^T` form.
Implementation uses two QRs and a small SVD per the standard recipe; the
target rank is set by `eps` (relative singular-value threshold) or a fixed
`rank`.

### Task 6 — Pivoted QR

From-scratch column-pivoted Householder QR with greedy norm-based pivot
selection (Chan, 1987 / Gu–Eisenstat, 1996). Reveals the numerical rank from
the diagonal of `R` and produces the rank-`r` approximation
`Q[:, :r] R[:r, :] P^{-1}`.

### Task 7 — Quality control through SVD

Single benchmark harness that runs every method on a fixed set of matrices
and ranks, computes the optimal SVD tail as a reference, and reports the
ratio `error_method / error_svd_best`. Also produces the unified
`reports/results/benchmarks.csv` table and the comparison plots.

## Running individual tasks

Each matrix task has a script in `experiments/matrix/`:

```bash
PYTHONPATH=src python experiments/matrix/run_lanczos.py        --seed 42
PYTHONPATH=src python experiments/matrix/run_cross.py          --seed 42
PYTHONPATH=src python experiments/matrix/run_randomized_svd.py --seed 42
PYTHONPATH=src python experiments/matrix/run_adaptive_cross.py --seed 42
PYTHONPATH=src python experiments/matrix/run_recompression.py  --seed 42
PYTHONPATH=src python experiments/matrix/run_pivoted_qr.py     --seed 42
```

Each tensor task has a script in `experiments/tensor/`:

```bash
PYTHONPATH=src python experiments/tensor/run_cp_als_cp.py      --seed 42
PYTHONPATH=src python experiments/tensor/run_cp_als_dense.py   --seed 42
PYTHONPATH=src python experiments/tensor/run_cp_neural.py      --seed 42
PYTHONPATH=src python experiments/tensor/run_cp_lm.py          --seed 42
PYTHONPATH=src python experiments/tensor/run_st_hosvd.py       --seed 42
PYTHONPATH=src python experiments/tensor/run_tt_cross.py       --seed 42
```

The cross-task benchmarks:

```bash
PYTHONPATH=src python experiments/matrix/run_matrix_benchmarks.py --seed 42  # Task 7
PYTHONPATH=src python experiments/tensor/run_tensor_benchmarks.py --seed 42  # CP/Tucker/TT compared
```

Outputs land under `reports/figures/<task_name>/` and `reports/results/`.

## Testing & quality gates

| Command | Purpose |
|---------|---------|
| `make test`      | Run pytest. Each task has a dedicated test module covering an analytical, a random-synthetic, and a noisy / ill-conditioned case. |
| `make typecheck` | `mypy --strict` over `src/`. |
| `make lint`      | `ruff check` over `src/`, `tests/`, `experiments/`. |
| `make format`    | `ruff format` (idempotent reformat). |

All tests are reproducible: every random routine accepts `random_seed`,
propagated via `numpy.random.Generator`.

## Documentation

```bash
make docs
$BROWSER docs/_build/html/index.html
```

The Sphinx site uses `numpydoc` for API rendering and `sphinx.ext.mathjax`
for inline math.

## Project layout

```
prac/
├── pyproject.toml
├── requirements.txt
├── README.md
├── Makefile
├── src/matcomp/
│   ├── utils/                  # shared scaffolding
│   │   ├── functional_matrix.py
│   │   ├── low_rank.py
│   │   ├── test_matrices.py
│   │   ├── metrics.py
│   │   ├── counting.py
│   │   ├── caching.py
│   │   ├── linalg.py
│   │   ├── plotting.py
│   │   ├── seeding.py
│   │   └── timing.py
│   └── matrix_algorithms/      # tasks 1–7, one file each
│       ├── lanczos.py
│       ├── cross.py
│       ├── randomized_svd.py
│       ├── adaptive_cross.py
│       ├── recompression.py
│       ├── pivoted_qr.py
│       └── svd_control.py
├── tests/                      # pytest files
├── experiments/matrix/         # CLI scripts producing plots + CSV
├── reports/                    # output figures + tables
└── docs/                       # Sphinx sources
```

## Glossary (PDF p. 14)

| Term | Meaning |
|------|---------|
| Functional matrix | Matrix whose elements are computed by a function; the full array is not stored. |
| Low-rank approximation | Approximation of an object by a sum of a small number of simple components. |
| Cross approximation | Approximation through informative rows / columns. |
| Recompression | Reduction of redundant rank in an already-constructed decomposition with error control. |
| Oracle calls | Number of real calls to the function that computes elements of a matrix. |

## References

- Course assignment: `../practice_task_settings.pdf`.
- Saad, Y. *Numerical Methods for Large Eigenvalue Problems* — Lanczos.
- Halko, N., Martinsson, P. G., Tropp, J. A. (SIAM Review 53(2), 2011) — randomized SVD.
- Bebendorf, M., Rjasanow, S. (*Computing* 70, 2003) — adaptive cross approximation.
- Goreinov, S. A., Tyrtyshnikov, E. E., Zamarashkin, N. L. (*Linear Algebra Appl.*
  261, 1997) — pseudo-skeleton / cross theory.
- Chan, T. F. (*Linear Algebra Appl.* 88–89, 1987) and Gu, M., Eisenstat, S. C.
  (*SIAM J. Sci. Comput.* 17(4), 1996) — rank-revealing QR.
