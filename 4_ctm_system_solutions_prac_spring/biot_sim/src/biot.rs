//! Assembly of the linear system for Biot poroelasticity.
//!
//! # Governing equations
//!
//! The continuous problem consists of a quasi-static elasticity equation
//! coupled to a parabolic pressure equation:
//!
//! ```text
//!   -μ · Δu - (μ + λ) · ∇(div u) + α · ∇p = 0                (momentum)
//!   ζ · ∂t p + α · ∂t(div u) - κ · Δp    = q                  (storage)
//! ```
//!
//! # Time discretisation
//!
//! A backward-Euler step is used. To keep all matrix rows comparably scaled,
//! the pressure row is multiplied through by `dt`:
//!
//! ```text
//!   (ζ + WI·dt) · p^{n+1} + α · div(u^{n+1}) - κ·dt · Δp^{n+1}
//!       = ζ · p^n + α · div(u^n) + WI·dt · p_bhp
//! ```
//!
//! The momentum equation is quasi-static: `u^{n+1}` is fully determined by
//! `p^{n+1}` at each step, but because `u` and `p` are solved together the
//! coupling is implicit.
//!
//! # Spatial discretisation
//!
//! Marker-and-cell (MAC) staggered grid identical to the one in
//! `mipt-solvers/biot.cpp`:
//!
//! ```text
//!   *---v(i, j+1)---*
//!   |               |
//! u(i,j)  p(i,j)  u(i+1,j)
//!   |               |
//!   *---v(i,  j )---*
//! ```
//!
//! so that `u` lives on vertical faces, `v` on horizontal faces, and `p` at
//! cell centres. The unknown layout in the global vector is
//! `[u (N+1)·M | v N·(M+1) | p N·M]` — see [`Grid`] for the accessors.
//!
//! # Boundary conditions
//!
//! Fixed displacement on all four boundaries (Dirichlet `u = v = 0`) and
//! no-flow on all four boundaries for pressure. This matches `biot.cpp`.

use crate::solver::{CsrBuilder, CsrMatrix};

/// Geometry of the MAC grid and the global unknown layout.
///
/// The domain is the unit square `[0, 1] × [0, 1]`, discretised into `N × M`
/// cells. Cell `(i, j)` is centred at `((i + 0.5)·hx, (j + 0.5)·hy)`.
pub struct Grid {
    /// Number of cells along Ox.
    pub n: usize,
    /// Number of cells along Oy.
    pub m: usize,
    /// Cell size in x, `1 / N`.
    pub hx: f64,
    /// Cell size in y, `1 / M`.
    pub hy: f64,
    /// Number of `u` unknowns `(N+1) · M`.
    pub nu: usize,
    /// Number of `v` unknowns `N · (M+1)`.
    pub nv: usize,
    /// Number of `p` unknowns `N · M`.
    pub np: usize,
    /// Offset of the `u` block inside the global vector (0).
    pub su: usize,
    /// Offset of the `v` block inside the global vector (`nu`).
    pub sv: usize,
    /// Offset of the `p` block inside the global vector (`nu + nv`).
    pub sp: usize,
    /// Total size of the linear system, `nu + nv + np`.
    pub ntot: usize,
}

impl Grid {
    /// Build an `N × M` unit-square grid.
    pub fn new(n: usize, m: usize) -> Self {
        let nu = (n + 1) * m;
        let nv = n * (m + 1);
        let np = n * m;
        Self {
            n, m,
            hx: 1.0 / n as f64,
            hy: 1.0 / m as f64,
            nu, nv, np,
            su: 0,
            sv: nu,
            sp: nu + nv,
            ntot: nu + nv + np,
        }
    }

    /// Global index of horizontal-face displacement `u(i, j)`, with
    /// `0 ≤ i ≤ N` and `0 ≤ j < M`.
    #[inline] pub fn iu(&self, i: usize, j: usize) -> usize { self.su + i * self.m + j }

    /// Global index of vertical-face displacement `v(i, j)`, with
    /// `0 ≤ i < N` and `0 ≤ j ≤ M`.
    #[inline] pub fn iv(&self, i: usize, j: usize) -> usize { self.sv + i * (self.m + 1) + j }

    /// Global index of cell-centred pressure `p(i, j)`, with
    /// `0 ≤ i < N` and `0 ≤ j < M`.
    #[inline] pub fn ip(&self, i: usize, j: usize) -> usize { self.sp + i * self.m + j }
}

/// Physical coefficients of the Biot system.
///
/// See the README for a conceptual description of each field. Defaults
/// chosen in `default_config` (see `main.rs`) give a visually clear transient;
/// every field can be overridden from the command line.
#[derive(Clone, Copy, Debug)]
pub struct PhysParams {
    /// First Lamé parameter (shear modulus) — controls resistance to shape
    /// change of the solid skeleton.
    pub mu: f64,
    /// Second Lamé parameter — controls resistance to volumetric strain.
    pub lambda: f64,
    /// Biot–Willis coefficient `0 < α ≤ 1`, strength of the two-way coupling
    /// between pressure and skeleton strain.
    pub alpha: f64,
    /// Specific storage coefficient (`ζ = 1/M`, inverse Biot modulus) —
    /// compressibility of the pore fluid.
    pub zeta: f64,
    /// Mobility `κ = k / ν` — permeability divided by fluid viscosity.
    pub kappa: f64,
    /// Wellbore radius used in Peaceman's well index.
    pub rw: f64,
}

/// A well located at a specific cell with a prescribed bottom-hole pressure.
///
/// The well contribution to cell `(i, j)` is `WI · (p_cell − p_bhp)`, with
/// Peaceman's well index
/// `WI = 2π·κ / ln(0.14·√(hx² + hy²) / rw) / (hx · hy)`.
///
/// `p_bhp > p_cell` makes the cell an injector, `p_bhp < p_cell` a producer.
#[derive(Clone, Copy, Debug)]
pub struct Well {
    /// Cell column index (`0 ≤ i < N`).
    pub i: usize,
    /// Cell row index (`0 ≤ j < M`).
    pub j: usize,
    /// Bottom-hole pressure target imposed by the Peaceman well model.
    pub p_bhp: f64,
}

/// Build the coupled Biot matrix and right-hand side for one time step.
///
/// The matrix implements the backward-Euler discretisation described at the
/// top of this module; the right-hand side carries the history terms that
/// depend on the previous step:
///
/// ```text
///   b[p(i,j)] = ζ · p_old(i,j) + α · div(u_old)(i,j) + Σ_wells WI·dt·p_bhp
///   b[u(i,j)] = 0        (quasi-static elasticity, no body force)
///   b[v(i,j)] = 0
/// ```
///
/// # Parameters
///
/// * `g` — grid geometry and unknown layout.
/// * `ph` — physical coefficients.
/// * `dt` — time step `Δt`.
/// * `wells` — wells active during this step; duplicates at the same cell add
///   up, so two injectors in the same cell act as one with summed strength.
/// * `u_old`, `v_old`, `p_old` — solution at the previous step. For the first
///   step pass zero-filled vectors.
///
/// # Returns
///
/// A `(CsrMatrix, Vec<f64>)` pair ready to be fed to
/// [`crate::solver::bicgstab`].
pub fn assemble(
    g: &Grid,
    ph: &PhysParams,
    dt: f64,
    wells: &[Well],
    u_old: &[f64],
    v_old: &[f64],
    p_old: &[f64],
) -> (CsrMatrix, Vec<f64>) {
    let (n, m) = (g.n, g.m);
    let (hx, hy) = (g.hx, g.hy);
    let hx2 = hx * hx;
    let hy2 = hy * hy;
    let hxy = hx * hy;
    let (mu, lambda, alpha, zeta, kappa, rw) =
        (ph.mu, ph.lambda, ph.alpha, ph.zeta, ph.kappa, ph.rw);

    let mut builder = CsrBuilder::new(g.ntot);
    let mut b = vec![0.0f64; g.ntot];

    // --- u rows: elasticity x-momentum ---
    for i in 0..=n {
        for j in 0..m {
            let row = g.iu(i, j);
            if i == 0 || i == n {
                builder.add(row, g.iu(i, j), 1.0); // Dirichlet u = 0
            } else {
                builder.add(row, g.iu(i - 1, j), -(2.0 * mu + lambda) / hx2);
                builder.add(row, g.iu(i, j),      (2.0 * mu + lambda) / hx2);
                builder.add(row, g.iu(i + 1, j), -(2.0 * mu + lambda) / hx2);
                builder.add(row, g.iu(i, j),      (2.0 * mu + lambda) / hx2);
                if j == 0 {
                    builder.add(row, g.iu(i, j), mu / (0.5 * hy2));
                } else {
                    builder.add(row, g.iu(i, j - 1), -mu / hy2);
                    builder.add(row, g.iu(i, j),      mu / hy2);
                }
                if j == m - 1 {
                    builder.add(row, g.iu(i, j), mu / (0.5 * hy2));
                } else {
                    builder.add(row, g.iu(i, j + 1), -mu / hy2);
                    builder.add(row, g.iu(i, j),      mu / hy2);
                }
                builder.add(row, g.iv(i,     j),      (mu + lambda) / hxy);
                builder.add(row, g.iv(i,     j + 1), -(mu + lambda) / hxy);
                builder.add(row, g.iv(i - 1, j),     -(mu + lambda) / hxy);
                builder.add(row, g.iv(i - 1, j + 1),  (mu + lambda) / hxy);
                builder.add(row, g.ip(i - 1, j), -alpha / hx);
                builder.add(row, g.ip(i,     j),  alpha / hx);
            }
        }
    }

    // --- v rows: elasticity y-momentum ---
    for i in 0..n {
        for j in 0..=m {
            let row = g.iv(i, j);
            if j == 0 || j == m {
                builder.add(row, g.iv(i, j), 1.0); // Dirichlet v = 0
            } else {
                builder.add(row, g.iv(i, j - 1), -(2.0 * mu + lambda) / hy2);
                builder.add(row, g.iv(i, j),      (2.0 * mu + lambda) / hy2);
                builder.add(row, g.iv(i, j + 1), -(2.0 * mu + lambda) / hy2);
                builder.add(row, g.iv(i, j),      (2.0 * mu + lambda) / hy2);
                if i == 0 {
                    builder.add(row, g.iv(i, j), mu / (0.5 * hx2));
                } else {
                    builder.add(row, g.iv(i - 1, j), -mu / hx2);
                    builder.add(row, g.iv(i, j),      mu / hx2);
                }
                if i == n - 1 {
                    builder.add(row, g.iv(i, j), mu / (0.5 * hx2));
                } else {
                    builder.add(row, g.iv(i + 1, j), -mu / hx2);
                    builder.add(row, g.iv(i, j),      mu / hx2);
                }
                builder.add(row, g.iu(i,     j),      (mu + lambda) / hxy);
                builder.add(row, g.iu(i,     j - 1), -(mu + lambda) / hxy);
                builder.add(row, g.iu(i + 1, j),     -(mu + lambda) / hxy);
                builder.add(row, g.iu(i + 1, j - 1),  (mu + lambda) / hxy);
                builder.add(row, g.ip(i, j),     alpha / hy);
                builder.add(row, g.ip(i, j - 1), -alpha / hy);
            }
        }
    }

    // --- p rows: fluid pressure with Peaceman well model ---
    let wi = 2.0 * std::f64::consts::PI * kappa
        / ((0.14 / rw) * (hx2 + hy2).sqrt()).ln()
        / hxy;

    for i in 0..n {
        for j in 0..m {
            let row = g.ip(i, j);

            builder.add(row, g.ip(i, j), zeta);
            b[row] += zeta * p_old[i * m + j];

            for w in wells {
                if w.i == i && w.j == j {
                    builder.add(row, g.ip(i, j), wi * dt);
                    b[row] += wi * dt * w.p_bhp;
                }
            }

            if i > 0 {
                builder.add(row, g.ip(i - 1, j), -kappa * dt / hx2);
                builder.add(row, g.ip(i, j),      kappa * dt / hx2);
            }
            if i < n - 1 {
                builder.add(row, g.ip(i + 1, j), -kappa * dt / hx2);
                builder.add(row, g.ip(i, j),      kappa * dt / hx2);
            }
            if j > 0 {
                builder.add(row, g.ip(i, j - 1), -kappa * dt / hy2);
                builder.add(row, g.ip(i, j),      kappa * dt / hy2);
            }
            if j < m - 1 {
                builder.add(row, g.ip(i, j + 1), -kappa * dt / hy2);
                builder.add(row, g.ip(i, j),      kappa * dt / hy2);
            }

            builder.add(row, g.iu(i,     j), -alpha / hx);
            builder.add(row, g.iu(i + 1, j),  alpha / hx);
            builder.add(row, g.iv(i, j),     -alpha / hy);
            builder.add(row, g.iv(i, j + 1),  alpha / hy);
            let div_old = (u_old[(i + 1) * m + j] - u_old[i * m + j]) / hx
                + (v_old[i * (m + 1) + j + 1] - v_old[i * (m + 1) + j]) / hy;
            b[row] += alpha * div_old;
        }
    }

    (builder.finalize(), b)
}
