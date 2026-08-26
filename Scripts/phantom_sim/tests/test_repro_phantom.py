"""This-box tests for the straight-fibre phantom used by the PlenoXFiber reproduction.

Straight (unidirectional) fibres are packed as a 2D cross-section and extruded along the
beam axis (dim 0), so every z-slice must be identical and the mu scale must match the
reference (per-voxel absorbance, matrix ~3.5e-4, fibre ~6.2e-4).
"""
import numpy as np

from phantom_sim.repro_plenoxfiber import make_straight_fibre_phantom


def test_straight_phantom_is_extruded_and_scaled():
    vol, radii = make_straight_fibre_phantom(
        nz=12, ny=80, nx=80, voxel_um=2.0, domain_radius_um=40.0, fvf=0.3,
        r_mean_um=3.5, mu_fibre=6.2e-4, mu_matrix=3.5e-4, iters=20, seed=0)
    assert vol.shape == (12, 80, 80)
    assert vol.dtype == np.float32
    # extruded: every slice identical to the first
    assert np.allclose(vol[0], vol[-1])
    assert np.allclose(vol[5], vol[0])
    # two-level attenuation at the reference scale
    assert vol.min() == np.float32(3.5e-4)
    assert vol.max() == np.float32(6.2e-4)
    # some fibre voxels present
    assert (vol > 4.0e-4).mean() > 0.02
