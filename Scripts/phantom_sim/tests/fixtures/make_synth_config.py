"""Generate a tiny synthetic fibre configuration for torch-free downstream tests."""
import numpy as np


def straight_bundle(Z=8, n_side=3, spacing_um=20.0, radius_um=3.5):
    """N = n_side**2 straight vertical fibres on a square grid, constant over Z."""
    xs = (np.arange(n_side) - (n_side - 1) / 2.0) * spacing_um
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    centres = np.stack([gx.ravel(), gy.ravel()], axis=0)          # (2, N)
    N = centres.shape[1]
    configuration = np.repeat(centres[None, :, :], Z, axis=0)      # (Z, 2, N)
    radii = np.full(N, radius_um, dtype=np.float64)
    return configuration.astype(np.float64), radii


def tilted_single(Z=8, axial_um=2.0, tilt_x_um_per_slice=0.5, radius_um=3.5):
    """One fibre tilted in x: centre moves tilt_x_um_per_slice per Z slice."""
    x = np.arange(Z) * tilt_x_um_per_slice
    y = np.zeros(Z)
    configuration = np.stack([x, y], axis=0).T[:, :, None]        # (Z, 2, 1)
    return configuration.astype(np.float64), np.array([radius_um])
