# biot_sim — Biot poroelasticity simulator (slide 38)

Rust implementation of the task on slide 38 of `presentation.pdf`.
Starts from the single-solve example `mipt-solvers/biot.cpp` and extends it to:

1. a backward-Euler **time-step loop**,
2. a **linear solve** at each step (BiCGStab + ILU(0)),
3. a **right-hand side that depends on the previous step** (`zeta*p^n + alpha*div(u^n)`),
4. a **VTK frame per step** plus a `.pvd` collection for ParaView,
5. an adjustable set of **wells** (injection + production).

## Layout

```
src/solver.rs  — CsrBuilder / CsrMatrix / Ilu0 / bicgstab
src/biot.rs    — Biot staggered-grid stencil assembly, time-dependent RHS
src/vtk.rs     — legacy VTK writer + .pvd collection
src/main.rs    — configuration, time loop, wells
```

## Build and run

```
cargo build --release
./target/release/biot_sim                          # defaults: 40x40, dt=0.01, t=0.5
./target/release/biot_sim 60 60 0.005 1.0          # 60x60, dt=0.005, t_total=1.0
```

CLI arguments (all optional, positional):
```
N  M  dt  t_total  mu  lambda  alpha  zeta  kappa  rw
```

## Output

Frames are written under `out/` as XML ImageData `frame_0000.vti …
frame_NNNN.vti` plus `out.pvd`. Open `out/out.pvd` in ParaView to get the full
time series at once — the `DISPLACEMENT` vector field is what visualises the
deformation of the porous medium. (VTK's `.pvd` reader only accepts XML-format
files like `.vti`/`.vtu`/`.vts`; legacy `.vtk` files cannot be referenced.)

### Making the animation visible in ParaView

With 200 frames the pressure and displacement fields change smoothly, but
ParaView auto-rescales the colour bar per frame by default, so the pattern
looks static. To actually see the evolution:

1. Load `out/out.pvd`, hit **Apply**.
2. Select the pipeline item, set **Coloring** to `PRESSURE`.
3. Open the colour-map editor and press **Rescale to data range over all
   timesteps** (the icon with the clock). Afterwards disable **Rescale on
   visibility change** so the bar stays fixed during playback.
4. To see the deformation of the porous medium, apply the
   **Filters → Alphabetical → Warp By Vector** filter with `DISPLACEMENT` as
   the vector and a scale factor of 1–10 (displacement is in the same metres
   as the domain, so raw amplitudes are small). Colour the warped dataset by
   `PRESSURE` for the classic poroelastic picture.
5. Press the ▶ play button — you should see the pressure front spreading from
   the injectors toward the producer while the grid warps accordingly.

## Fields written to every frame

Each `.vti` file carries three point-data arrays sampled at cell centres.

### `PRESSURE` — scalar, pore-fluid pressure `p`

Solution of the coupled pressure equation in physical units (same as `p_bhp`
on the wells). Starts at zero everywhere, builds up around injectors, drops
below ambient around producers.

Default run: global minimum and maximum evolve roughly as

| step | t     | `p_min` | `p_max` |
|------|-------|---------|---------|
| 0    | 0.000 | 0       | 0       |
| 1    | 0.005 | ~0.001  | ~7      |
| 10   | 0.050 | ~0.007  | ~30     |
| 50   | 0.250 | ~0.04   | ~40     |
| 200  | 1.000 | ~0.19   | ~46     |

The `p_max` cell is the primary injector `(N/4, M/2)` with `p_bhp = 100` —
it never reaches 100 because of the surrounding no-flow boundary and the
finite `WI·dt` "stiffness" of the Peaceman well model. `p_min` sits near the
producer, which also starts slightly above its `p_bhp = 0` because the
neighbouring injectors bleed pressure into it.

### `DISPLACEMENT` — vector, solid-skeleton displacement `u = (u_x, u_y, 0)`

Cell-centred average of the two face-staggered dofs `u`, `v`. Units match the
domain size (the unit square), so raw magnitudes are small — **apply
`Warp By Vector` with a scale factor of 1–10** to see the deformation.

Expected behaviour under defaults:

* Zero on every boundary (Dirichlet `u = v = 0`).
* Vectors point **outward from each injector** (pressure pushes the matrix
  away) and **inward toward the producer** (depressurisation lets the matrix
  relax back).
* Magnitude grows monotonically from 0. Over the default 200 steps
  `|u|_max` rises from ~1.2 · 10⁻² at step 1 to ~2.6 · 10⁻¹ at step 200.
* The peak is located between the two strong wells (injector at
  `(N/4, M/2)` and producer at `(3N/4, M/2)`), where the pressure gradient
  driving the skeleton is steepest.

### `DIV_U` — scalar, volumetric strain `∇·u`

Discrete divergence of the displacement using the same staggered stencil as
the pressure equation. Physically this is the **fractional change in pore
volume** per cell.

Expected behaviour under defaults:

* **Positive near injectors** (cells dilate as fluid is pushed in).
* **Negative near the producer** (cells compact as fluid is withdrawn).
* The magnitude is sharply peaked at the well cells and decays within a few
  cells. Over the default run `max |∇·u|` climbs from ~2.4 at step 1 to
  ~14.9 near the end.
* Loosely coupled to `PRESSURE` via the storage equation
  `ζ·∂t p + α·∂t(∇·u) = κ·Δp + q` — any pressure pattern you see should
  have a corresponding `DIV_U` pattern of the same sign.

## Change time step / total time

Slide 38 asks us to pick our own `dt` and `t_total`. The defaults
(`dt = 0.005`, `t_total = 1.0`, 200 steps, `kappa = 0.01`) produce a smooth
transient: pressure diffuses slowly enough that the front spreads across
several cells over the simulation, and the displacement magnitude grows
monotonically from 0 to ~0.26. With the original `kappa = 1.0` the pressure
nearly reaches steady state on the very first step, which is why the frames
appeared static.

Raise `t_total` or `kappa` to see the late-time steady state; lower `dt` for
finer resolution.

## Wells

Three wells are configured in `main.rs`:
- injector at `(N/4, M/2)` with `p_bhp = 100`,
- producer at `(3N/4, M/2)` with `p_bhp = 0`,
- secondary injector at `(N/2, M/4)` with `p_bhp = 50`.

Edit the `Well { ... }` entries in `main.rs` to experiment with different
layouts (item 5 on slide 38).
