//! Sparse linear algebra primitives for the Biot simulator.
//!
//! Contains everything needed to solve the linear system produced by
//! [`crate::biot::assemble`]:
//!
//! * [`CsrBuilder`] — an insertion-friendly accumulator that turns a stream of
//!   `(row, col, value)` triples into a canonical [`CsrMatrix`].
//! * [`CsrMatrix`] — a compressed sparse row matrix with a cached diagonal-slot
//!   table. Columns of every row are sorted, which keeps ILU(0) factorisation
//!   cheap.
//! * [`Ilu0`] — ILU(0) factorisation sharing the sparsity pattern of its input
//!   matrix, used as the preconditioner in [`bicgstab`].
//! * [`bicgstab`] — preconditioned BiCGStab iterative solver. Chosen over CG
//!   because the Biot saddle-point system is indefinite.
//!
//! None of these types allocate after `finalize`/`new`, so they can be reused
//! across time steps once constructed.

use std::collections::BTreeMap;

/// Accumulator for sparse matrix entries, one `BTreeMap` per row.
///
/// Entries added via [`CsrBuilder::add`] are summed when the same `(i, j)`
/// index is passed multiple times — this matches the way `biot.cpp` writes to
/// the diagonal entry from several stencil contributions.
///
/// Calling [`CsrBuilder::finalize`] consumes the builder and returns a
/// [`CsrMatrix`] whose columns are sorted inside each row (a property that the
/// [`Ilu0`] factorisation below relies on).
pub struct CsrBuilder {
    rows: Vec<BTreeMap<usize, f64>>,
    n: usize,
}

impl CsrBuilder {
    /// Create an empty builder for an `n × n` matrix.
    ///
    /// Rows are initialised as empty; call [`CsrBuilder::add`] to populate
    /// them in any order.
    pub fn new(n: usize) -> Self {
        Self { rows: (0..n).map(|_| BTreeMap::new()).collect(), n }
    }

    /// Accumulate `v` into entry `(i, j)`.
    ///
    /// If the entry already exists the values are summed (i.e. behaves like
    /// `A[i, j] += v`). In debug builds the indices are bounds-checked against
    /// the matrix size.
    pub fn add(&mut self, i: usize, j: usize, v: f64) {
        debug_assert!(i < self.n && j < self.n);
        *self.rows[i].entry(j).or_insert(0.0) += v;
    }

    /// Consume the builder and produce the canonical [`CsrMatrix`].
    ///
    /// Also walks the non-zero pattern once to cache `diag[i]` — the slot
    /// inside `a`/`ja` that holds the diagonal entry for row `i`, or
    /// `usize::MAX` if row `i` has no explicit diagonal (which should not
    /// happen for the matrices assembled in this project, and will cause
    /// [`Ilu0::new`] to panic on unwrap-like accesses if it does).
    pub fn finalize(self) -> CsrMatrix {
        let n = self.n;
        let mut ia = Vec::with_capacity(n + 1);
        ia.push(0usize);
        let mut ja = Vec::new();
        let mut a = Vec::new();
        for row in &self.rows {
            for (&col, &val) in row.iter() {
                ja.push(col);
                a.push(val);
            }
            ia.push(ja.len());
        }
        let mut diag = vec![usize::MAX; n];
        for i in 0..n {
            for k in ia[i]..ia[i + 1] {
                if ja[k] == i {
                    diag[i] = k;
                    break;
                }
            }
        }
        CsrMatrix { n, ia, ja, a, diag }
    }
}

/// Square matrix in Compressed Sparse Row format.
///
/// # Layout
///
/// * `ia[i] .. ia[i+1]` is the half-open slot range for row `i`;
/// * `ja[k]` and `a[k]` are the column index and value stored at slot `k`;
/// * `diag[i]` is the slot inside row `i` that contains the diagonal entry
///   `(i, i)`.
///
/// Columns within each row are sorted in ascending order, which lets the
/// ILU(0) factorisation iterate lower entries before the diagonal and upper
/// entries after it without any extra bookkeeping.
pub struct CsrMatrix {
    /// Matrix dimension (both rows and columns).
    pub n: usize,
    /// Row-pointer array of length `n + 1`. `ia[0] == 0`, `ia[n] == a.len()`.
    pub ia: Vec<usize>,
    /// Column indices, one per non-zero, sorted within each row.
    pub ja: Vec<usize>,
    /// Non-zero values, aligned with `ja`.
    pub a: Vec<f64>,
    /// Slot in `a`/`ja` that holds the diagonal of each row.
    pub diag: Vec<usize>,
}

impl CsrMatrix {
    /// Compute `y ← A · x`.
    ///
    /// `x` and `y` must have length `self.n`; `y` is fully overwritten.
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        for i in 0..self.n {
            let mut s = 0.0;
            for k in self.ia[i]..self.ia[i + 1] {
                s += self.a[k] * x[self.ja[k]];
            }
            y[i] = s;
        }
    }
}

/// ILU(0) factorisation that reuses the sparsity pattern of its input.
///
/// An exact factorisation `A = L · U` would generally need extra non-zero
/// slots ("fill-in"). ILU(0) instead restricts both `L` and `U` to the
/// sparsity pattern of `A`, dropping any fill. The resulting factors form an
/// approximate inverse that is cheap to apply and, for the mildly indefinite
/// Biot system used here, keeps [`bicgstab`] to a few tens of iterations per
/// step.
///
/// The two triangular factors are stored together in a single array `lu`:
///
/// * slots with `ja[k] < i` hold `L[i, ja[k]]` (unit diagonal implicit);
/// * the slot `diag[i]` holds `U[i, i]`;
/// * slots with `ja[k] > i` hold `U[i, ja[k]]`.
pub struct Ilu0 {
    /// Matrix dimension.
    pub n: usize,
    /// Row pointers (shared with the source matrix).
    pub ia: Vec<usize>,
    /// Column indices (shared with the source matrix).
    pub ja: Vec<usize>,
    /// Combined `L`/`U` values.
    pub lu: Vec<f64>,
    /// Slot in `lu` that holds `U[i, i]`.
    pub diag: Vec<usize>,
}

impl Ilu0 {
    /// Factor `mat` using ILU(0).
    ///
    /// The factorisation is performed row by row. For every non-zero `(i, k)`
    /// with `k < i`, `L[i, k]` is divided by `U[k, k]` and then the update
    /// `A[i, j] -= L[i, k] · U[k, j]` is applied for every `j > k` that is
    /// already present in row `i` (no fill-in).
    ///
    /// # Robustness
    ///
    /// Saddle-point systems occasionally yield near-zero pivots mid-way
    /// through the elimination. Rather than fail catastrophically, the routine
    /// clamps any pivot whose absolute value is below `1e-14` to `±1e-14`
    /// (sign-preserving). In practice this never triggers for the matrices
    /// built by [`crate::biot::assemble`].
    pub fn new(mat: &CsrMatrix) -> Self {
        let n = mat.n;
        let ia = mat.ia.clone();
        let ja = mat.ja.clone();
        let mut lu = mat.a.clone();
        let diag = mat.diag.clone();

        // iw[col] = slot-in-row-i if col is present in row i, otherwise usize::MAX.
        let mut iw = vec![usize::MAX; n];

        for i in 0..n {
            for s in ia[i]..ia[i + 1] {
                iw[ja[s]] = s;
            }
            for k_slot in ia[i]..ia[i + 1] {
                let k = ja[k_slot];
                if k >= i {
                    break; // row sorted: remaining entries are upper part
                }
                let ukk = lu[diag[k]];
                let pivot = if ukk.abs() < 1e-14 {
                    if ukk >= 0.0 { 1e-14 } else { -1e-14 }
                } else {
                    ukk
                };
                let lik = lu[k_slot] / pivot;
                lu[k_slot] = lik;
                for j_slot in (diag[k] + 1)..ia[k + 1] {
                    let j = ja[j_slot];
                    let s = iw[j];
                    if s != usize::MAX {
                        lu[s] -= lik * lu[j_slot];
                    }
                }
            }
            for s in ia[i]..ia[i + 1] {
                iw[ja[s]] = usize::MAX;
            }
            let d = lu[diag[i]];
            if d.abs() < 1e-14 {
                lu[diag[i]] = if d >= 0.0 { 1e-14 } else { -1e-14 };
            }
        }

        Ilu0 { n, ia, ja, lu, diag }
    }

    /// Solve `(L · U) x = b`.
    ///
    /// `x` may alias `b` on entry. The routine is a plain forward / backward
    /// sweep over the triangular factors:
    ///
    /// 1. forward substitution: `L y = b` using the lower part of `lu`;
    /// 2. backward substitution: `U x = y` using the upper part of `lu`
    ///    (and the explicit diagonal).
    ///
    /// Both `b` and `x` must have length `self.n`.
    pub fn solve(&self, b: &[f64], x: &mut [f64]) {
        for i in 0..self.n {
            let mut s = b[i];
            for k in self.ia[i]..self.diag[i] {
                s -= self.lu[k] * x[self.ja[k]];
            }
            x[i] = s;
        }
        for i in (0..self.n).rev() {
            let mut s = x[i];
            for k in (self.diag[i] + 1)..self.ia[i + 1] {
                s -= self.lu[k] * x[self.ja[k]];
            }
            x[i] = s / self.lu[self.diag[i]];
        }
    }
}

/// Inner product of two equal-length vectors.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// BLAS-style `y ← y + α · x`.
fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for i in 0..y.len() {
        y[i] += alpha * x[i];
    }
}

/// Euclidean (ℓ²) norm of `v`.
fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Outcome of a [`bicgstab`] run.
///
/// `residual` is the final relative residual `‖r‖ / max(‖b‖, 1)`. `converged`
/// is `true` iff that ratio fell below the requested tolerance before `maxit`
/// iterations or a breakdown.
pub struct BiCGStabReport {
    /// Number of BiCGStab iterations performed.
    pub iters: usize,
    /// Final relative residual norm.
    pub residual: f64,
    /// Whether the tolerance was reached.
    pub converged: bool,
}

/// Solve `A x = b` with ILU(0)-preconditioned BiCGStab.
///
/// # Parameters
///
/// * `a` — system matrix.
/// * `m` — ILU(0) preconditioner, typically [`Ilu0::new(&a)`](Ilu0::new).
/// * `b` — right-hand side (length `a.n`).
/// * `x` — on entry the initial guess (e.g. the previous step's solution);
///   on return the computed solution. Reusing the previous step warm-starts
///   the solver and typically halves the iteration count after the first
///   time step.
/// * `tol` — target for `‖r‖ / max(‖b‖, 1)`.
/// * `maxit` — hard iteration cap.
///
/// # Algorithm
///
/// Implementation follows the classical right-preconditioned BiCGStab
/// (van der Vorst, 1992). Breakdowns in `ρ` or `ω` short-circuit the loop and
/// return `converged = false`.
pub fn bicgstab(
    a: &CsrMatrix,
    m: &Ilu0,
    b: &[f64],
    x: &mut [f64],
    tol: f64,
    maxit: usize,
) -> BiCGStabReport {
    let n = a.n;
    let bnorm = norm2(b).max(1.0);

    let mut r = vec![0.0; n];
    a.matvec(x, &mut r);
    for i in 0..n {
        r[i] = b[i] - r[i];
    }

    let r_hat = r.clone();
    let mut p = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut s = vec![0.0; n];
    let mut t = vec![0.0; n];

    let mut rho_old: f64 = 1.0;
    let mut alpha: f64 = 1.0;
    let mut omega_old: f64 = 1.0;

    let mut res_norm = norm2(&r);
    if res_norm / bnorm <= tol {
        return BiCGStabReport { iters: 0, residual: res_norm / bnorm, converged: true };
    }

    for iter in 0..maxit {
        let rho = dot(&r_hat, &r);
        if rho.abs() < 1e-30 {
            return BiCGStabReport { iters: iter, residual: res_norm / bnorm, converged: false };
        }
        let beta = (rho / rho_old) * (alpha / omega_old);
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega_old * v[i]);
        }
        m.solve(&p, &mut y);
        a.matvec(&y, &mut v);
        let rhat_v = dot(&r_hat, &v);
        if rhat_v.abs() < 1e-30 {
            return BiCGStabReport { iters: iter, residual: res_norm / bnorm, converged: false };
        }
        alpha = rho / rhat_v;
        for i in 0..n {
            s[i] = r[i] - alpha * v[i];
        }
        let s_norm = norm2(&s);
        if s_norm / bnorm <= tol {
            axpy(alpha, &y, x);
            return BiCGStabReport { iters: iter + 1, residual: s_norm / bnorm, converged: true };
        }
        m.solve(&s, &mut z);
        a.matvec(&z, &mut t);
        let tt = dot(&t, &t);
        let omega = if tt > 0.0 { dot(&t, &s) / tt } else { 0.0 };
        for i in 0..n {
            x[i] += alpha * y[i] + omega * z[i];
            r[i] = s[i] - omega * t[i];
        }
        res_norm = norm2(&r);
        if res_norm / bnorm <= tol {
            return BiCGStabReport { iters: iter + 1, residual: res_norm / bnorm, converged: true };
        }
        if omega.abs() < 1e-30 {
            return BiCGStabReport { iters: iter + 1, residual: res_norm / bnorm, converged: false };
        }
        rho_old = rho;
        omega_old = omega;
    }
    BiCGStabReport { iters: maxit, residual: res_norm / bnorm, converged: false }
}
