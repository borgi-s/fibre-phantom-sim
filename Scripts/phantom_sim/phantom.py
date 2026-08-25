"""Torch-free fibre phantom voxelisation from a fibre-pack configuration."""
import numpy as np


def voxelize_config(configuration, radii, transverse_shape, voxel_um,
                    axis=0, mu_fibre=0.30, mu_matrix=0.28):
    """Rasterise per-slice fibre discs into an attenuation volume.

    configuration: (Z, 2, N) fibre centres in um, index 0 = x, index 1 = y.
    radii: (N,) fibre radii in um.
    transverse_shape: (ny, nx) transverse voxel grid.
    Fibre axis is built as array dim 0 then moved to `axis`.
    """
    configuration = np.asarray(configuration, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    Z, _, N = configuration.shape
    ny, nx = transverse_shape
    vol = np.full((Z, ny, nx), mu_matrix, dtype=np.float32)
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx0 = (nx - 1) / 2.0
    cy0 = (ny - 1) / 2.0
    for zi in range(Z):
        for k in range(N):
            cx = configuration[zi, 0, k] / voxel_um + cx0
            cy = configuration[zi, 1, k] / voxel_um + cy0
            r = radii[k] / voxel_um
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            vol[zi][mask] = mu_fibre
    if axis != 0:
        vol = np.moveaxis(vol, 0, axis)
    return np.ascontiguousarray(vol, dtype=np.float32)


def bundle_orientations(configuration, axial_um, axis=0):
    """Per-fibre unit direction in array coordinates.

    Base array order is (along, y, x). The along-axis step is `axial_um` per slice;
    transverse steps come from consecutive-slice centre differences.
    """
    configuration = np.asarray(configuration, dtype=np.float64)
    Z, _, N = configuration.shape
    if Z >= 2:
        d = configuration[1:] - configuration[:-1]      # (Z-1, 2, N)
        dx = d[:, 0, :].mean(axis=0)
        dy = d[:, 1, :].mean(axis=0)
    else:
        dx = np.zeros(N)
        dy = np.zeros(N)
    base = np.stack([np.full(N, axial_um), dy, dx], axis=1)   # (N,3) order (along,y,x)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    # permute components to match np.moveaxis(vol, 0, axis)
    perm = np.moveaxis(np.arange(3), 0, axis)
    return base[:, perm]


def ground_truth_table(configuration, radii, axial_um, axis=0):
    configuration = np.asarray(configuration, dtype=np.float64)
    centre_um = configuration.mean(axis=0).T          # (N, 2) mean (x, y)
    return {
        "centre_um": centre_um,
        "orientation": bundle_orientations(configuration, axial_um, axis),
        "radius_um": np.asarray(radii, dtype=np.float64),
    }
