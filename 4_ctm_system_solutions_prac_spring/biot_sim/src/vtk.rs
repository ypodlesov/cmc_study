//! Per-step and time-series VTK output for ParaView.
//!
//! Two formats are produced:
//!
//! * One `frame_NNNN.vti` per time step — VTK XML ImageData with point data
//!   `PRESSURE`, `DISPLACEMENT`, and `DIV_U` sampled at cell centres.
//! * A single `out.pvd` Collection file referencing every frame with its
//!   timestamp. This is what ParaView opens to obtain a playable time series.
//!
//! Note that `vtkPVDReader` can only reference XML VTK formats (`.vti`,
//! `.vtu`, `.vts`, `.vtp`, `.vtr`). Legacy ASCII `.vtk` files cannot be
//! listed inside a `.pvd`, which is why this module emits `.vti`.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::biot::Grid;

/// File extension used for a single frame (`"vti"`).
///
/// Exposed as a function so [`crate::main`] can format file names without
/// hard-coding the extension in multiple places.
pub fn frame_extension() -> &'static str {
    "vti"
}

/// Write one VTK ImageData frame to `path`.
///
/// # Parameters
///
/// * `path` — full output path for the `.vti` file.
/// * `g` — grid geometry (determines `ORIGIN`, `SPACING`, and array sizes).
/// * `time` — simulated time, stored in a `FieldData` array named `TIME` so
///   it is preserved inside the frame even if the `.pvd` collection is lost.
/// * `u`, `v`, `p` — face-staggered and cell-centred solution fields.
///
/// The three point-data arrays written are:
///
/// * `PRESSURE` — scalar, one value per cell.
/// * `DISPLACEMENT` — three-component vector; `ux` and `uy` are the face
///   values averaged to the cell centre, `uz = 0`.
/// * `DIV_U` — `∂u/∂x + ∂v/∂y` using the same staggered finite-difference
///   stencil as the pressure equation.
///
/// Node ordering follows VTK's convention (x varies fastest, then y, then z).
pub fn write_frame(
    path: &Path,
    g: &Grid,
    time: f64,
    u: &[f64],
    v: &[f64],
    p: &[f64],
) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    let nx = g.n as i64;
    let ny = g.m as i64;
    let ox = 0.5 * g.hx;
    let oy = 0.5 * g.hy;

    writeln!(w, r#"<?xml version="1.0"?>"#)?;
    writeln!(
        w,
        r#"<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian" header_type="UInt32">"#
    )?;
    writeln!(
        w,
        r#"  <FieldData>
    <DataArray type="Float64" Name="TIME" NumberOfTuples="1" format="ascii">
      {:.8}
    </DataArray>
  </FieldData>"#,
        time
    )?;
    writeln!(
        w,
        r#"  <ImageData WholeExtent="0 {nx_m1} 0 {ny_m1} 0 0" Origin="{ox:.8} {oy:.8} 0" Spacing="{hx:.8} {hy:.8} 1">"#,
        nx_m1 = nx - 1,
        ny_m1 = ny - 1,
        ox = ox,
        oy = oy,
        hx = g.hx,
        hy = g.hy,
    )?;
    writeln!(
        w,
        r#"    <Piece Extent="0 {nx_m1} 0 {ny_m1} 0 0">"#,
        nx_m1 = nx - 1,
        ny_m1 = ny - 1,
    )?;
    writeln!(
        w,
        r#"      <PointData Scalars="PRESSURE" Vectors="DISPLACEMENT">"#
    )?;

    writeln!(
        w,
        r#"        <DataArray type="Float64" Name="PRESSURE" NumberOfComponents="1" format="ascii">"#
    )?;
    for j in 0..g.m {
        for i in 0..g.n {
            write!(w, "{:.8} ", p[i * g.m + j])?;
        }
        writeln!(w)?;
    }
    writeln!(w, "        </DataArray>")?;

    writeln!(
        w,
        r#"        <DataArray type="Float64" Name="DISPLACEMENT" NumberOfComponents="3" format="ascii">"#
    )?;
    for j in 0..g.m {
        for i in 0..g.n {
            let ux = 0.5 * (u[(i + 1) * g.m + j] + u[i * g.m + j]);
            let uy = 0.5 * (v[i * (g.m + 1) + j + 1] + v[i * (g.m + 1) + j]);
            writeln!(w, "{:.8} {:.8} 0.0", ux, uy)?;
        }
    }
    writeln!(w, "        </DataArray>")?;

    writeln!(
        w,
        r#"        <DataArray type="Float64" Name="DIV_U" NumberOfComponents="1" format="ascii">"#
    )?;
    for j in 0..g.m {
        for i in 0..g.n {
            let d = (u[(i + 1) * g.m + j] - u[i * g.m + j]) / g.hx
                + (v[i * (g.m + 1) + j + 1] - v[i * (g.m + 1) + j]) / g.hy;
            write!(w, "{:.8} ", d)?;
        }
        writeln!(w)?;
    }
    writeln!(w, "        </DataArray>")?;

    writeln!(w, "      </PointData>")?;
    writeln!(w, "    </Piece>")?;
    writeln!(w, "  </ImageData>")?;
    writeln!(w, "</VTKFile>")?;

    w.flush()?;
    Ok(())
}

/// Write the ParaView `.pvd` Collection file that indexes every frame.
///
/// `frames` is the ordered list of `(time, file_name)` entries used to
/// populate the `<DataSet>` children. File names are recorded as-is, so they
/// must be valid relative to the location of the `.pvd` file (passing bare
/// `"frame_0001.vti"` assumes the `.pvd` sits next to the frames).
pub fn write_pvd(path: &Path, frames: &[(f64, String)]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut w = BufWriter::new(file);
    writeln!(w, r#"<?xml version="1.0"?>"#)?;
    writeln!(
        w,
        r#"<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">"#
    )?;
    writeln!(w, "  <Collection>")?;
    for (t, fname) in frames {
        writeln!(
            w,
            r#"    <DataSet timestep="{:.6}" group="" part="0" file="{}"/>"#,
            t, fname
        )?;
    }
    writeln!(w, "  </Collection>")?;
    writeln!(w, "</VTKFile>")?;
    w.flush()?;
    Ok(())
}
