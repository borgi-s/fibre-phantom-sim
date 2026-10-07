//! Rust port of the per-slice disc rasteriser in `phantom.voxelize_config`.
//!
//! Same arithmetic, in the same order, as the NumPy version, so the output is
//! bit-identical (checked in `tests/test_voxelize_rust.py`). Slices along the
//! fibre axis are independent, so they run in parallel with rayon.

use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Rasterise fibre discs into a (Z, ny, nx) float32 attenuation volume.
///
/// configuration: (Z, 2, N) fibre centres in um (index 0 = x, 1 = y); radii: (N,) in um.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn voxelize_slices<'py>(
    py: Python<'py>,
    configuration: PyReadonlyArray3<'py, f64>,
    radii: PyReadonlyArray1<'py, f64>,
    ny: usize,
    nx: usize,
    voxel_um: f64,
    mu_fibre: f64,
    mu_matrix: f64,
    supersample: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let cfg = configuration.as_array();
    let radii = radii.as_array();
    let (nz, two, n) = cfg.dim();
    if two != 2 || radii.len() != n {
        return Err(PyValueError::new_err(
            "configuration must be (Z, 2, N) and radii (N,)",
        ));
    }
    let ss = supersample.max(1);
    // subpixel offsets in (-0.5, 0.5), as (np.arange(ss) + 0.5) / ss - 0.5
    let sub: Vec<f64> = (0..ss).map(|i| (i as f64 + 0.5) / ss as f64 - 0.5).collect();
    let n_sub = (ss * ss) as f64;
    let delta = mu_fibre - mu_matrix;
    let cx0 = (nx as f64 - 1.0) / 2.0;
    let cy0 = (ny as f64 - 1.0) / 2.0;
    let mut buf = vec![mu_matrix as f32; nz * ny * nx];

    py.allow_threads(|| {
        buf.par_chunks_mut(ny * nx).enumerate().for_each(|(zi, slice)| {
            for k in 0..n {
                let cx = cfg[[zi, 0, k]] / voxel_um + cx0;
                let cy = cfg[[zi, 1, k]] / voxel_um + cy0;
                let r = radii[k] / voxel_um;
                let x0 = ((cx - r - 1.0).floor() as i64).max(0);
                let x1 = ((cx + r + 1.0).ceil() as i64 + 1).min(nx as i64);
                let y0 = ((cy - r - 1.0).floor() as i64).max(0);
                let y1 = ((cy + r + 1.0).ceil() as i64 + 1).min(ny as i64);
                if x1 <= x0 || y1 <= y0 {
                    continue;
                }
                let r2 = r * r;
                for yi in y0..y1 {
                    let gy = yi as f64;
                    let row = &mut slice[yi as usize * nx..(yi as usize + 1) * nx];
                    for xi in x0..x1 {
                        let gx = xi as f64;
                        let mut hits = 0u32;
                        for &dy in &sub {
                            let ey = gy + dy - cy;
                            for &dx in &sub {
                                let ex = gx + dx - cx;
                                if ex * ex + ey * ey <= r2 {
                                    hits += 1;
                                }
                            }
                        }
                        // coverage-weighted mix, then np.maximum (discs only touch)
                        let patch = (mu_matrix + (hits as f64 / n_sub) * delta) as f32;
                        let v = &mut row[xi as usize];
                        if patch > *v {
                            *v = patch;
                        }
                    }
                }
            }
        });
    });

    let vol = Array3::from_shape_vec((nz, ny, nx), buf).expect("buffer size matches shape");
    Ok(vol.into_pyarray_bound(py))
}

#[pymodule]
fn phantom_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(voxelize_slices, m)?)?;
    Ok(())
}
