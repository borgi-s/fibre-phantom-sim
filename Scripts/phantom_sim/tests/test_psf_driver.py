"""Torch-free tests for the PSF/MTF driver's pure helpers.

The forward projection and reconstruction in main() need the cluster; here we pin the sphere
phantom, the FWHM estimate, the MTF, and the metrics schema.
"""

import numpy as np

from phantom_sim.run_plenoptic_psf import (
    make_sphere_phantom,
    axis_profile,
    fwhm,
    mtf,
    psf_entry,
    assemble_psf_metrics,
)


def test_sphere_phantom_center_and_background():
    v = make_sphere_phantom(20, 20, 20, radius_vox=3, mu_fibre=1.0, mu_matrix=0.0)
    assert v.shape == (20, 20, 20)
    assert v[10, 10, 10] == 1.0  # centre voxel is fibre
    assert v[0, 0, 0] == 0.0  # corner is background
    # roughly a sphere of radius 3 voxels
    assert 0.5 < v.sum() / ((4.0 / 3.0) * np.pi * 3**3) < 1.5


def test_axis_profile_runs_through_the_centre():
    v = np.zeros((5, 7, 9), dtype=np.float32)
    v[:, 3, 4] = 1.0  # a line along axis 0 at the transverse centre
    p = axis_profile(v, axis=0)
    assert p.shape == (5,)
    assert np.allclose(p, 1.0)


def test_fwhm_of_a_gaussian():
    x = np.arange(201)
    sigma = 10.0
    g = np.exp(-0.5 * ((x - 100) / sigma) ** 2)
    assert abs(fwhm(g, spacing=1.0) - 2.3548 * sigma) < 0.5


def test_mtf_dc_is_one_and_falls_off():
    x = np.arange(201)
    g = np.exp(-0.5 * ((x - 100) / 8.0) ** 2)
    freqs, m = mtf(g, spacing=1.0)
    assert abs(m[0] - 1.0) < 1e-9
    assert m[1] < m[0]
    assert (m >= 0).all()
    assert freqs[0] == 0.0


def test_assemble_psf_metrics_schema():
    e = psf_entry(
        "z", 30.0, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.1]), np.array([1.0, 0.5])
    )
    m = assemble_psf_metrics([e], {"voxel_um": 2.0})
    assert m["psf"][0]["axis"] == "z"
    assert m["psf"][0]["fwhm_um"] == 30.0
    assert m["meta"]["voxel_um"] == 2.0
