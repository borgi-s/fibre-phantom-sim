"""CLI: point-spread plenoptic study. Longitudinal (z) resolution and MTF.

Places a small high-contrast sphere at the centre of an otherwise empty volume, images it in a
single plenoptic capture, and reconstructs. The reconstruction of a point-like object is the
point-spread function directly, so the intensity profiles along z (the fibre / optical axis) and
across it give the FWHM and the modulation transfer function (MTF) along each direction. The
plenoptic geometry undersamples the z direction, so the z point-spread is far wider than the
transverse one; that anisotropy is what limits detection of defects that vary quickly along z.
The reconstruction uses the same total variation prior along z as the fibre recons, so
the measured z resolution is the method's actual resolution; run with --tv-weight 0 to separate
the geometry-limited part from the prior-limited part.

Module-level code is stdlib + numpy only, so this module imports on a box without torch/astra.
`main` defers every torch/astra/CUDA import to its body.

Usage (cluster, CUDA interpreter):
    python -m phantom_sim.run_plenoptic_psf --smoke --out /tmp/psf_smoke
    python -m phantom_sim.run_plenoptic_psf --out phantom_sim_results/exp5_psf
"""

import argparse
import json
from pathlib import Path

import numpy as np


def make_sphere_phantom(nz, ny, nx, radius_vox, mu_fibre=1.0, mu_matrix=0.0):
    """A solid sphere of `mu_fibre` (radius `radius_vox` voxels) at the volume centre, in a
    `mu_matrix` background. Torch-free; the point-spread probe for the resolution measurement."""
    zc, yc, xc = nz // 2, ny // 2, nx // 2
    zz, yy, xx = np.ogrid[:nz, :ny, :nx]
    d2 = (zz - zc) ** 2 + (yy - yc) ** 2 + (xx - xc) ** 2
    return np.where(d2 <= radius_vox**2, mu_fibre, mu_matrix).astype(np.float32)


def axis_profile(vol, axis):
    """1D profile along `axis` through the centre of the other two axes."""
    vol = np.asarray(vol)
    sl = [s // 2 for s in vol.shape]
    sl[axis] = slice(None)
    return np.asarray(vol[tuple(sl)], dtype=np.float64)


def fwhm(profile, spacing=1.0):
    """Full width at half maximum of a peaked profile, in `spacing` units.

    Half maximum is taken relative to the profile's own baseline (its minimum). The crossings
    are linearly interpolated. Returns 0.0 for a flat profile or one that never crosses back.
    """
    p = np.asarray(profile, dtype=np.float64)
    base = p.min()
    peak = p.max()
    if peak <= base:
        return 0.0
    half = base + 0.5 * (peak - base)
    i_peak = int(np.argmax(p))

    left = None
    for i in range(i_peak, 0, -1):
        if p[i] >= half >= p[i - 1]:
            denom = p[i] - p[i - 1]
            frac = (half - p[i - 1]) / denom if denom != 0 else 0.0
            left = (i - 1) + frac
            break
    right = None
    for i in range(i_peak, len(p) - 1):
        if p[i] >= half >= p[i + 1]:
            denom = p[i] - p[i + 1]
            frac = (p[i] - half) / denom if denom != 0 else 0.0
            right = i + frac
            break
    if left is None or right is None:
        return 0.0
    return float((right - left) * spacing)


def mtf(lsf, spacing=1.0):
    """Modulation transfer function of a line-spread profile: normalised |rFFT|, and its
    spatial-frequency axis (cycles per `spacing` unit). MTF at zero frequency is 1."""
    line = np.asarray(lsf, dtype=np.float64)
    line = line - line.min()
    mag = np.abs(np.fft.rfft(line))
    if mag[0] != 0:
        mag = mag / mag[0]
    freqs = np.fft.rfftfreq(len(line), d=spacing)
    return freqs, mag


def psf_entry(axis_name, fwhm_um, profile, freqs, mtf_vals):
    """One direction's PSF metrics entry. Pure, no torch."""
    return {
        "axis": axis_name,
        "fwhm_um": float(fwhm_um),
        "profile": [float(x) for x in np.asarray(profile).ravel()],
        "freqs_inv_um": [float(x) for x in np.asarray(freqs).ravel()],
        "mtf": [float(x) for x in np.asarray(mtf_vals).ravel()],
    }


def assemble_psf_metrics(entries, meta):
    """Pure schema assembly for psf_metrics.json. No torch/astra."""
    return {"meta": meta, "psf": entries}


def build_parser():
    p = argparse.ArgumentParser(
        description="Point-spread plenoptic study: longitudinal (z) resolution and MTF."
    )
    p.add_argument("--out", required=True)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny volume/grid/iters for a fast cluster smoke test.",
    )
    p.add_argument("--nz", type=int, default=256)
    p.add_argument("--ny", type=int, default=256)
    p.add_argument("--nx", type=int, default=256)
    p.add_argument(
        "--radius-vox", type=float, default=2.0, help="Sphere radius in voxels."
    )
    p.add_argument("--voxel-um", type=float, default=2.0)
    p.add_argument("--mu-fibre", type=float, default=1.0)
    p.add_argument("--mu-matrix", type=float, default=0.0)
    p.add_argument("--sod-mm", type=float, default=129.24)
    p.add_argument("--sdd-mm", type=float, default=1171.57)
    p.add_argument("--detector-pixel-um", type=float, default=55.0)
    p.add_argument("--source-range-um", type=float, default=22000.0)
    p.add_argument("--det-range-um", type=float, default=177400.0)
    p.add_argument("--n-grid", type=int, default=21)
    p.add_argument("--super-sampling", type=int, default=2)
    p.add_argument("--pad-transverse-vox", type=int, default=32)
    p.add_argument("--pad-depth-vox", type=int, default=32)
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument(
        "--tv-weight",
        type=float,
        default=0.5,
        help="TV weight along z (the fibre axis). Set 0 to remove the prior's "
        "contribution to the z blur.",
    )
    p.add_argument("--log-every", type=int, default=10)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.smoke:
        nz, ny, nx = 32, 32, 32
        n_grid, n_iter, log_every = 3, 6, 1
        pad_t, pad_d = 4, 4
        radius = 2.0
    else:
        nz, ny, nx = args.nz, args.ny, args.nx
        n_grid, n_iter, log_every = args.n_grid, args.n_iter, args.log_every
        pad_t, pad_d = args.pad_transverse_vox, args.pad_depth_vox
        radius = args.radius_vox

    voxel_um = args.voxel_um
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from phantom_sim.geometry import (
        build_plenoptic_geometry,
        make_plenoptic_projector,
        plenoptic_grid,
        XrayOperator,
        delete_projector,
    )
    from phantom_sim.reconstruct import reconstruct
    from phantom_sim.metrics import corr_with_gt

    obj = make_sphere_phantom(nz, ny, nx, radius, args.mu_fibre, args.mu_matrix)
    vol = np.pad(obj, ((pad_d, pad_d), (pad_t, pad_t), (pad_t, pad_t)))
    padded_shape = vol.shape
    obj_sl = (
        slice(pad_d, pad_d + nz),
        slice(pad_t, pad_t + ny),
        slice(pad_t, pad_t + nx),
    )

    detector_z = (args.sdd_mm * 1000.0) / voxel_um
    pixel_size = args.detector_pixel_um / voxel_um
    src, det = plenoptic_grid(
        args.source_range_um / voxel_um, args.det_range_um / voxel_um, n=n_grid
    )
    num_imgs = n_grid * n_grid
    vol_z0 = (args.sod_mm * 1000.0) / voxel_um
    sinogram_size = [padded_shape[1], padded_shape[2], num_imgs]

    geom = build_plenoptic_geometry(
        src, det, vol_z0, detector_z, pixel_size, pixel_size
    )
    pid, _ = make_plenoptic_projector(
        padded_shape,
        geom,
        sinogram_size=sinogram_size,
        super_sampling=args.super_sampling,
    )
    A = XrayOperator(pid)
    volt = torch.tensor(vol, device="cuda")
    proj = A(volt).detach()
    del volt
    torch.cuda.empty_cache()

    def corr_metric(recon_padded_np):
        return corr_with_gt(np.ascontiguousarray(recon_padded_np[obj_sl]), obj)

    recon, history = reconstruct(
        A,
        proj,
        mask=None,
        n_iter=n_iter,
        lr=args.lr,
        tv_weights=(args.tv_weight, 0.0, 0.0),
        x_init=None,
        log_every=log_every,
        gt=vol,
        target_psnr=None,
        lower_clamp=0.0,
        plateau_scheduler=True,
        metric_fn=corr_metric,
        early_stop=False,
    )
    recon_obj = np.ascontiguousarray(recon[obj_sl])
    delete_projector(A.projector_id)
    del proj, A
    torch.cuda.empty_cache()

    np.save(out_dir / "psf_recon.npy", recon_obj.astype(np.float32))

    axes = [("z", 0), ("y", 1), ("x", 2)]
    entries = []
    for name, ax in axes:
        prof = axis_profile(recon_obj, ax)
        w = fwhm(prof, spacing=voxel_um)
        freqs, m = mtf(prof, spacing=voxel_um)
        entries.append(psf_entry(name, w, prof, freqs, m))
        print(f"  {name}-axis FWHM = {w:.2f} um")

    fwhm_by = {e["axis"]: e["fwhm_um"] for e in entries}
    transverse = 0.5 * (fwhm_by["x"] + fwhm_by["y"])
    ratio = (fwhm_by["z"] / transverse) if transverse else 0.0
    meta = {
        "smoke": bool(args.smoke),
        "voxel_um": voxel_um,
        "nz": nz,
        "ny": ny,
        "nx": nx,
        "radius_vox": radius,
        "mu_fibre": args.mu_fibre,
        "mu_matrix": args.mu_matrix,
        "sod_mm": args.sod_mm,
        "sdd_mm": args.sdd_mm,
        "n_grid": n_grid,
        "n_iter": n_iter,
        "lr": args.lr,
        "tv_weight": args.tv_weight,
        "pad_transverse_vox": pad_t,
        "pad_depth_vox": pad_d,
        "fwhm_z_um": fwhm_by["z"],
        "fwhm_transverse_um": transverse,
        "z_to_transverse_ratio": ratio,
        "final_corr_with_gt": float(corr_with_gt(recon_obj, obj)),
    }
    metrics = assemble_psf_metrics(entries, meta)
    with open(out_dir / "psf_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"Saved {out_dir / 'psf_metrics.json'}  z-FWHM {fwhm_by['z']:.1f} um vs transverse "
        f"{transverse:.1f} um (ratio {ratio:.1f})"
    )


if __name__ == "__main__":
    main()
