//! Biot poroelasticity simulator — solves the task on slide 38 of
//! `presentation.pdf`.
//!
//! The entry point wraps the one-shot system assembler from
//! `mipt-solvers/biot.cpp` in a full simulator:
//!
//! 1. a backward-Euler time-step loop,
//! 2. a linear solve per step ([`solver::bicgstab`] + [`solver::Ilu0`]),
//! 3. a right-hand side that is refreshed from the previous step
//!    ([`biot::assemble`]),
//! 4. a VTK frame emitted every step ([`vtk::write_frame`]) plus a `.pvd`
//!    collection for ParaView ([`vtk::write_pvd`]),
//! 5. a configurable set of injector / producer wells ([`biot::Well`]).
//!
//! The command-line interface accepts up to ten positional arguments — see
//! [`parse_args`] for the order. Omitted arguments keep the defaults from
//! [`default_config`].

mod biot;
mod solver;
mod vtk;

use std::path::PathBuf;

use biot::{assemble, Grid, PhysParams, Well};
use solver::{bicgstab, CsrMatrix, Ilu0};

/// Everything the simulator needs to run one scenario.
///
/// All fields are owned so that [`run`] can be called repeatedly with
/// different configurations (e.g. for a parameter sweep) without touching
/// global state.
#[derive(Clone)]
struct Config {
    /// Number of cells along Ox.
    n: usize,
    /// Number of cells along Oy.
    m: usize,
    /// Physical coefficients of the Biot system.
    phys: PhysParams,
    /// Time step `Δt`.
    dt: f64,
    /// Total simulated time; the number of steps is `round(t_total / dt)`.
    t_total: f64,
    /// Wells active throughout the simulation.
    wells: Vec<Well>,
    /// Directory for the VTK time-series output.
    out_dir: PathBuf,
    /// Relative-residual tolerance for the linear solver.
    tol: f64,
    /// Hard iteration cap for the linear solver.
    maxit: usize,
}

/// Build the default configuration used when no CLI arguments are passed.
///
/// The defaults are tuned for a **visually clear animation**:
///
/// * `kappa = 0.01` — 100× slower diffusion than `biot.cpp`'s default so the
///   pressure front has to propagate through the medium instead of
///   saturating on step 1.
/// * `dt = 0.005`, `t_total = 1.0` — 200 smooth frames.
/// * Three wells (two injectors, one producer) to break the symmetry.
///
/// Any field can be overridden from the command line via [`parse_args`].
fn default_config() -> Config {
    let n = 40;
    let m = 40;
    let wells = vec![
        Well { i: n / 4,       j: m / 2, p_bhp: 100.0 }, // primary injector
        Well { i: 3 * n / 4,   j: m / 2, p_bhp: 0.0   }, // producer
        Well { i: n / 2,       j: m / 4, p_bhp: 50.0  }, // secondary injector
    ];

    Config {
        n, m,
        phys: PhysParams {
            mu: 1.0,
            lambda: 1.0,
            alpha: 1.0,
            zeta: 1.0,
            kappa: 0.01,
            rw: 1e-4,
        },
        dt: 0.005,
        t_total: 1.0,
        wells,
        out_dir: PathBuf::from("out"),
        tol: 1e-8,
        maxit: 2000,
    }
}

/// Split the global solution vector into its `u`, `v`, `p` blocks.
///
/// Returns owning `Vec`s (cheap for this project — the whole solution is at
/// most a few tens of thousands of floats) so the caller can keep them across
/// the next step without worrying about lifetimes.
fn split_solution(g: &Grid, x: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let u = x[g.su..g.su + g.nu].to_vec();
    let v = x[g.sv..g.sv + g.nv].to_vec();
    let p = x[g.sp..g.sp + g.np].to_vec();
    (u, v, p)
}

/// Main driver: run a complete simulation and write every frame to disk.
///
/// # Flow
///
/// 1. Initialise `u = v = p = 0` and write the `t = 0` frame.
/// 2. For each step `1..=n_steps`:
///    * assemble the Biot system for the current step
///      ([`biot::assemble`]),
///    * factor the matrix with ILU(0) and solve with BiCGStab
///      ([`solver::Ilu0`], [`solver::bicgstab`]) — the previous step's
///      solution serves as warm-start,
///    * split the solution into blocks and emit a `.vti` frame.
/// 3. Write `out.pvd` so ParaView can load the whole series at once.
///
/// Returns any I/O error from directory creation or from writing a frame.
fn run(cfg: &Config) -> std::io::Result<()> {
    std::fs::create_dir_all(&cfg.out_dir)?;
    let g = Grid::new(cfg.n, cfg.m);
    println!(
        "Grid {}x{} — unknowns: u={}, v={}, p={}, total={}",
        g.n, g.m, g.nu, g.nv, g.np, g.ntot
    );
    println!(
        "dt = {}, t_total = {}, steps = {}",
        cfg.dt,
        cfg.t_total,
        (cfg.t_total / cfg.dt).round() as usize
    );
    for (k, w) in cfg.wells.iter().enumerate() {
        println!("well[{}] cell=({},{}) p_bhp={}", k, w.i, w.j, w.p_bhp);
    }

    let mut u = vec![0.0; g.nu];
    let mut v = vec![0.0; g.nv];
    let mut p = vec![0.0; g.np];

    // Previous solution vector — reused as an initial guess for the next solve.
    let mut x = vec![0.0; g.ntot];

    let ext = vtk::frame_extension();
    let mut frames: Vec<(f64, String)> = Vec::new();
    let fname = format!("frame_{:04}.{}", 0, ext);
    vtk::write_frame(&cfg.out_dir.join(&fname), &g, 0.0, &u, &v, &p)?;
    frames.push((0.0, fname));

    let n_steps = (cfg.t_total / cfg.dt).round() as usize;
    for step in 1..=n_steps {
        let t = step as f64 * cfg.dt;
        let (mat, rhs): (CsrMatrix, Vec<f64>) =
            assemble(&g, &cfg.phys, cfg.dt, &cfg.wells, &u, &v, &p);
        let ilu = Ilu0::new(&mat);
        let report = bicgstab(&mat, &ilu, &rhs, &mut x, cfg.tol, cfg.maxit);
        println!(
            "step {:4} t={:7.4}  iters={:4}  residual={:.2e}  converged={}",
            step, t, report.iters, report.residual, report.converged
        );

        let (nu, nv, np) = split_solution(&g, &x);
        u = nu;
        v = nv;
        p = np;

        let fname = format!("frame_{:04}.{}", step, ext);
        vtk::write_frame(&cfg.out_dir.join(&fname), &g, t, &u, &v, &p)?;
        frames.push((t, fname));
    }

    vtk::write_pvd(&cfg.out_dir.join("out.pvd"), &frames)?;
    println!(
        "Wrote {} frames and out.pvd under {}",
        frames.len(),
        cfg.out_dir.display()
    );
    Ok(())
}

/// Apply CLI overrides to `cfg`.
///
/// Accepts up to ten positional arguments in this order:
///
/// ```text
///   biot_sim  N  M  dt  t_total  mu  lambda  alpha  zeta  kappa  rw
/// ```
///
/// Every argument is optional; passing `biot_sim` with no arguments keeps the
/// [`default_config`] values.
///
/// Invalid numeric arguments are silently ignored (the default stays in
/// place). If `N` is supplied without `M`, `M` is set equal to `N`.
fn parse_args(cfg: &mut Config) {
    let args: Vec<String> = std::env::args().collect();
    let get = |idx: usize| args.get(idx).and_then(|s| s.parse::<f64>().ok());
    if let Some(n) = args.get(1).and_then(|s| s.parse::<usize>().ok()) { cfg.n = n; cfg.m = n; }
    if let Some(m) = args.get(2).and_then(|s| s.parse::<usize>().ok()) { cfg.m = m; }
    if let Some(dt) = get(3) { cfg.dt = dt; }
    if let Some(tt) = get(4) { cfg.t_total = tt; }
    if let Some(v)  = get(5) { cfg.phys.mu = v; }
    if let Some(v)  = get(6) { cfg.phys.lambda = v; }
    if let Some(v)  = get(7) { cfg.phys.alpha = v; }
    if let Some(v)  = get(8) { cfg.phys.zeta = v; }
    if let Some(v)  = get(9) { cfg.phys.kappa = v; }
    if let Some(v)  = get(10) { cfg.phys.rw = v; }
}

/// Program entry point.
///
/// Builds the default configuration, applies command-line overrides,
/// refreshes the default wells for the (possibly new) grid size, and runs
/// the simulation. Propagates a non-zero exit code on I/O error.
fn main() {
    let mut cfg = default_config();
    parse_args(&mut cfg);
    cfg.wells = vec![
        Well { i: cfg.n / 4,     j: cfg.m / 2, p_bhp: 100.0 },
        Well { i: 3 * cfg.n / 4, j: cfg.m / 2, p_bhp: 0.0   },
        Well { i: cfg.n / 2,     j: cfg.m / 4, p_bhp: 50.0  },
    ];
    if let Err(e) = run(&cfg) {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}
