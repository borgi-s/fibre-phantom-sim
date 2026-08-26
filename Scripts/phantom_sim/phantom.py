"""Torch-free fibre phantom voxelisation from a fibre-pack configuration."""
import numpy as np


def voxelize_config(configuration, radii, transverse_shape, voxel_um,
                    axis=0, mu_fibre=0.30, mu_matrix=0.28, supersample=4):
    """Rasterise per-slice fibre discs into an attenuation volume.

    configuration: (Z, 2, N) fibre centres in um, index 0 = x, index 1 = y.
    radii: (N,) fibre radii in um.
    transverse_shape: (ny, nx) transverse voxel grid.
    Fibre axis is built as array dim 0 then moved to `axis`.

    `supersample` sets anti-aliasing: each voxel's attenuation is
    `mu_matrix + coverage * (mu_fibre - mu_matrix)`, where `coverage` is the fraction of
    the voxel area inside the disc, estimated on a supersample x supersample subgrid. This
    is physically correct partial-volume mixing and, for the small ~3-voxel fibres here,
    renders as smooth circles rather than the blocky squares/plus-signs a hard mask
    (supersample=1) produces. Fibres are packed non-overlapping, so where discs touch the
    larger coverage wins (`np.maximum`). Only a small bounding box per fibre is touched, so
    anti-aliasing does not scale with the full transverse grid.
    """
    configuration = np.asarray(configuration, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    Z, _, N = configuration.shape
    ny, nx = transverse_shape
    vol = np.full((Z, ny, nx), mu_matrix, dtype=np.float32)
    cx0 = (nx - 1) / 2.0
    cy0 = (ny - 1) / 2.0
    ss = max(1, int(supersample))
    sub = (np.arange(ss) + 0.5) / ss - 0.5          # subpixel offsets in (-0.5, 0.5)
    delta = mu_fibre - mu_matrix
    for zi in range(Z):
        for k in range(N):
            cx = configuration[zi, 0, k] / voxel_um + cx0
            cy = configuration[zi, 1, k] / voxel_um + cy0
            r = radii[k] / voxel_um
            x0 = max(0, int(np.floor(cx - r - 1)))
            x1 = min(nx, int(np.ceil(cx + r + 1)) + 1)
            y0 = max(0, int(np.floor(cy - r - 1)))
            y1 = min(ny, int(np.ceil(cy + r + 1)) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            xs = np.arange(x0, x1)
            ys = np.arange(y0, y1)
            gx, gy = np.meshgrid(xs, ys)            # (h, w) voxel centres
            r2 = r * r
            cover = np.zeros(gx.shape, dtype=np.float64)
            for dy in sub:
                for dx in sub:
                    cover += (gx + dx - cx) ** 2 + (gy + dy - cy) ** 2 <= r2
            cover /= ss * ss
            patch = mu_matrix + cover * delta
            region = vol[zi, y0:y1, x0:x1]
            np.maximum(region, patch.astype(np.float32), out=region)
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
    order = [n for n in range(3) if n != 0]
    order.insert(axis, 0)
    return base[:, order]


def ground_truth_table(configuration, radii, axial_um, axis=0):
    configuration = np.asarray(configuration, dtype=np.float64)
    centre_um = configuration.mean(axis=0).T          # (N, 2) mean (x, y)
    return {
        "centre_um": centre_um,
        "orientation": bundle_orientations(configuration, axial_um, axis),
        "radius_um": np.asarray(radii, dtype=np.float64),
    }


def pack_bundle(domain_radius_um, fvf, r_mean_um, r_sigma_um=0.0,
                n_slices=64, misalignment="none", iters=200, seed=0):
    """Run vendored fibre-pack and return the packing as numpy arrays.

    Uses torch internally (fibre-pack is torch-based). CPU is sufficient.

    fvf: fibre volume fraction as a 0-1 FRACTION (public interface). The
    vendored `from_fvf` expects a PERCENT convention (e.g. 40 for 40%),
    so it is scaled by 100 internally before being passed through.

    misalignment: a vendored fibre-pack preset STRING ('none', 'very low',
    'moderate', 'high', 'very high') or a dict of the raw parameters; passed
    straight through to `initialize_end_slice`. Default "none". Do not use
    the 'low' preset: it has an upstream UnboundLocalError bug in the
    vendored code.
    """
    import torch
    from phantom_sim.third_party.fibre_pack import fibre_packer as fp

    rng = torch.Generator().manual_seed(int(seed))
    packer = fp.from_fvf(domain_radius_um, fvf * 100.0, r_mean_um, r_sigma_um, rng=rng)
    packer.initialize_start_slice()
    packer.initialize_end_slice(misalignment)
    packer.interpolate_configuration(n_slices)
    packer.optimize_configuration(iters=iters)

    configuration = np.asarray(packer.configuration.detach().cpu().numpy(),
                               dtype=np.float64)     # (Z, 2, N)
    radii = np.asarray(packer.radii.detach().cpu().numpy(), dtype=np.float64)
    return configuration, radii


def make_straight_fibre_phantom(nz, ny, nx, voxel_um, domain_radius_um, fvf, r_mean_um,
                                mu_fibre, mu_matrix, iters, seed, supersample=4):
    """Pack a 2D fibre cross-section and extrude it straight along dim 0.

    Returns (vol float32 (nz, ny, nx), radii). Straight (unidirectional) fibres: pack a
    2-slice aligned configuration (cheap), voxelise a single transverse slice (anti-aliased),
    then repeat over nz so every slice is identical. Keeps deep volumes cheap to build.
    """
    cfg, radii = pack_bundle(domain_radius_um=domain_radius_um, fvf=fvf, r_mean_um=r_mean_um,
                             r_sigma_um=0.0, n_slices=2, misalignment="none",
                             iters=iters, seed=seed)
    one = voxelize_config(cfg[:1], radii, (ny, nx), voxel_um=voxel_um, axis=0,
                          mu_fibre=mu_fibre, mu_matrix=mu_matrix, supersample=supersample)
    vol = np.repeat(one, nz, axis=0)
    return np.ascontiguousarray(vol, dtype=np.float32), radii
