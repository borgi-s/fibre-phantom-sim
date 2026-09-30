"""CLI: tilt-sweep plenoptic study. Angular bandwidth of the plenoptic reconstruction.

Packs one straight fibre cross-section, then for a sweep of tilt angles shears the whole bundle
so every fibre makes the same angle theta with the optical axis (dim 0), images it in a single
plenoptic capture, and reconstructs. As theta grows past the largest angle the plenoptic array
samples, the reconstruction can no longer place the tilted fibres and its correlation with the
ground truth falls off. The angle at which it collapses, compared to the array's maximum ray
half-angle, is the angular bandwidth of the reconstruction.

The whole bundle is sheared about the central slice, so the ends displace by
tan(theta) * (nz/2) * voxel_um; the transverse frame must be wide enough that the tilted fibres
stay in view, otherwise the fall-off would measure the field of view rather than the angle. The
driver warns when a requested tilt approaches the frame edge.

Module-level code is stdlib + numpy only, so this module imports on a box without torch/astra.
`main` defers every torch/astra/CUDA import to its body.

Usage (cluster, CUDA interpreter):
    python -m phantom_sim.run_plenoptic_tilt --smoke --out /tmp/tilt_smoke
    python -m phantom_sim.run_plenoptic_tilt --out phantom_sim_results/exp4_tilt
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def tilt_config(base_centers, depth, theta_rad, voxel_um):
    """Shear a straight cross-section into a bundle whose fibres all tilt by theta_rad.

    base_centers: (2, N) fibre centres in um, index 0 = x, index 1 = y (one cross-section).
    depth: number of slices along the fibre axis (dim 0), each voxel_um apart.
    theta_rad: tilt angle from the optical axis (dim 0); the x-coordinate shears linearly with slice,
        sheared about the central slice so the ends displace symmetrically.
    Returns (depth, 2, N) config in um. theta_rad = 0 repeats the cross-section unchanged.
    """
    base = np.asarray(base_centers, dtype=np.float64)  # (2, N)
    z = np.arange(depth, dtype=np.float64)
    z_mid = (depth - 1) / 2.0
    shift = math.tan(theta_rad) * (z - z_mid) * voxel_um  # (depth,) in um
    cfg = np.repeat(base[None, :, :], depth, axis=0)  # (depth, 2, N)
    cfg[:, 0, :] = cfg[:, 0, :] + shift[:, None]  # shear x
    return cfg


def max_plenoptic_half_angle_deg(source_range_um, det_range_um, sdd_um):
    """Largest ray obliquity from the optical axis in the plenoptic array, in degrees.

    The most oblique ray runs from a source at one extreme of the source array to the opposed
    detector at the other extreme, a transverse span of (source_range + det_range) over the
    source-to-detector distance sdd.
    """
    return math.degrees(math.atan((source_range_um + det_range_um) / sdd_um))


def max_tilt_shift_vox(theta_deg, depth, voxel_um):
    """End displacement (in voxels) of the sheared bundle at tilt theta_deg."""
    return math.tan(math.radians(theta_deg)) * (depth / 2.0)


def tilt_entry(theta_deg, corr_val, psnr_val, ssim_val, resolution):
    """One sweep-point metrics entry. Pure, no torch."""
    return {
        "theta_deg": float(theta_deg),
        "corr_with_gt": float(corr_val),
        "psnr": float(psnr_val),
        "ssim": float(ssim_val),
        "resolution": resolution,
    }


def assemble_tilt_metrics(entries, meta):
    """Pure schema assembly for tilt_metrics.json. No torch/astra."""
    return {"meta": meta, "sweep": entries}


def default_thetas():
    return [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]


def build_parser():
    p = argparse.ArgumentParser(
        description="Tilt-sweep plenoptic study: angular bandwidth of the reconstruction."
    )
    p.add_argument("--out", required=True)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny bundle/grid/iters for a fast cluster smoke test.",
    )
    p.add_argument(
        "--thetas",
        type=float,
        nargs="+",
        default=None,
        help="Tilt angles in degrees (default: 0..20). 0 must be present as the "
        "straight-bundle reference.",
    )
    p.add_argument(
        "--nz", type=int, default=256, help="Depth (slices along the fibre axis)."
    )
    p.add_argument("--ny", type=int, default=384)
    p.add_argument("--nx", type=int, default=384)
    p.add_argument("--domain-radius-um", type=float, default=120.0)
    p.add_argument("--fvf", type=float, default=0.4)
    p.add_argument("--r-mean-um", type=float, default=6.0)
    p.add_argument("--voxel-um", type=float, default=2.0)
    p.add_argument("--mu-fibre", type=float, default=1.0)
    p.add_argument("--mu-matrix", type=float, default=0.0)
    p.add_argument(
        "--pack-iters", type=int, default=200, help="Packing optimisation iters."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sod-mm", type=float, default=129.24)
    p.add_argument("--sdd-mm", type=float, default=1171.57)
    p.add_argument("--detector-pixel-um", type=float, default=55.0)
    p.add_argument("--source-range-um", type=float, default=22000.0)
    p.add_argument("--det-range-um", type=float, default=177400.0)
    p.add_argument("--n-grid", type=int, default=21)
    p.add_argument("--super-sampling", type=int, default=2)
    p.add_argument("--pad-transverse-vox", type=int, default=64)
    p.add_argument("--pad-depth-vox", type=int, default=32)
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--tv-weight", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=10)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.smoke:
        nz, ny, nx = 16, 48, 48
        n_grid, n_iter, log_every = 3, 6, 1
        pad_t, pad_d = 6, 2
        pack_iters = 10
        domain_radius = 40.0
        thetas = [0.0, 10.0, 20.0]
    else:
        nz, ny, nx = args.nz, args.ny, args.nx
        n_grid, n_iter, log_every = args.n_grid, args.n_iter, args.log_every
        pad_t, pad_d = args.pad_transverse_vox, args.pad_depth_vox
        pack_iters = args.pack_iters
        domain_radius = args.domain_radius_um
        thetas = args.thetas if args.thetas is not None else default_thetas()

    if 0.0 not in thetas:
        thetas = [0.0] + list(thetas)
    voxel_um = args.voxel_um
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from phantom_sim.phantom import pack_bundle, voxelize_config
    import torch
    from phantom_sim.geometry import (
        build_plenoptic_geometry,
        make_plenoptic_projector,
        plenoptic_grid,
        XrayOperator,
        delete_projector,
    )
    from phantom_sim.reconstruct import reconstruct
    from phantom_sim.metrics import (
        psnr,
        ssim,
        register,
        resolution_along_across,
        corr_with_gt,
    )

    # 1. Pack one straight cross-section; its centres are the base the sweep shears.
    print(f"Packing straight cross-section (fvf={args.fvf}, r={args.r_mean_um} um) ...")
    cfg, radii = pack_bundle(
        domain_radius_um=domain_radius,
        fvf=args.fvf,
        r_mean_um=args.r_mean_um,
        r_sigma_um=0.0,
        n_slices=2,
        misalignment="none",
        iters=pack_iters,
        seed=args.seed,
    )
    base_centers = cfg[0]  # (2, N)
    print(f"  {radii.shape[0]} fibres")

    # 2. Geometry (single plenoptic capture; only the object changes between tilts).
    detector_z = (args.sdd_mm * 1000.0) / voxel_um
    pixel_size = args.detector_pixel_um / voxel_um
    src, det = plenoptic_grid(
        args.source_range_um / voxel_um, args.det_range_um / voxel_um, n=n_grid
    )
    num_imgs = n_grid * n_grid
    vol_z0 = (args.sod_mm * 1000.0) / voxel_um
    padded_shape = (nz + 2 * pad_d, ny + 2 * pad_t, nx + 2 * pad_t)
    obj_sl = (
        slice(pad_d, pad_d + nz),
        slice(pad_t, pad_t + ny),
        slice(pad_t, pad_t + nx),
    )
    sinogram_size = [padded_shape[1], padded_shape[2], num_imgs]
    max_half_angle = max_plenoptic_half_angle_deg(
        args.source_range_um, args.det_range_um, args.sdd_mm * 1000.0
    )
    print(f"  plenoptic max ray half-angle = {max_half_angle:.2f} deg")

    # Warn if any tilt shears the bundle near the transverse frame edge.
    frame_margin_vox = (nx / 2.0) - (domain_radius / voxel_um)
    for td in thetas:
        if max_tilt_shift_vox(td, nz, voxel_um) > frame_margin_vox:
            print(
                f"  WARNING: tilt {td} deg shears fibres ~{max_tilt_shift_vox(td, nz, voxel_um):.0f} "
                f"vox, near the {frame_margin_vox:.0f} vox frame margin; widen --nx/--ny or "
                f"reduce --nz/--domain-radius-um to keep tilted fibres in view."
            )

    def solve(theta_deg):
        obj = voxelize_config(
            tilt_config(base_centers, nz, math.radians(theta_deg), voxel_um),
            radii,
            (ny, nx),
            voxel_um=voxel_um,
            axis=0,
            mu_fibre=args.mu_fibre,
            mu_matrix=args.mu_matrix,
            supersample=4,
        )
        vol = np.pad(obj, ((pad_d, pad_d), (pad_t, pad_t), (pad_t, pad_t)))
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
        c_val = corr_with_gt(recon_obj, obj)
        p_val = psnr(recon_obj, obj)
        s_val = ssim(recon_obj, obj)
        res = resolution_along_across(register(recon_obj, obj), fibre_axis=0)
        delete_projector(A.projector_id)
        del proj, A
        torch.cuda.empty_cache()
        return recon_obj, c_val, p_val, s_val, res

    entries = []
    for td in sorted(thetas):
        recon_obj, c_val, p_val, s_val, res = solve(td)
        e = tilt_entry(td, c_val, p_val, s_val, res)
        entries.append(e)
        np.save(out_dir / f"tilt_recon_{td:04.1f}.npy", recon_obj.astype(np.float32))
        print(
            f"  tilt {td:5.1f} deg: corr={c_val:.4f} psnr={p_val:.2f} ssim={s_val:.3f}"
        )

    meta = {
        "smoke": bool(args.smoke),
        "thetas_deg": [float(t) for t in sorted(thetas)],
        "max_half_angle_deg": float(max_half_angle),
        "nz": nz,
        "ny": ny,
        "nx": nx,
        "voxel_um": voxel_um,
        "domain_radius_um": domain_radius,
        "fvf": args.fvf,
        "r_mean_um": args.r_mean_um,
        "mu_fibre": args.mu_fibre,
        "mu_matrix": args.mu_matrix,
        "n_fibres": int(radii.shape[0]),
        "seed": args.seed,
        "sod_mm": args.sod_mm,
        "sdd_mm": args.sdd_mm,
        "detector_pixel_um": args.detector_pixel_um,
        "source_range_um": args.source_range_um,
        "det_range_um": args.det_range_um,
        "n_grid": n_grid,
        "n_iter": n_iter,
        "lr": args.lr,
        "tv_weight": args.tv_weight,
        "pad_transverse_vox": pad_t,
        "pad_depth_vox": pad_d,
    }
    metrics = assemble_tilt_metrics(entries, meta)
    with open(out_dir / "tilt_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"Saved {out_dir / 'tilt_metrics.json'}  max half-angle {max_half_angle:.2f} deg"
    )
    print(
        "  corr vs tilt: "
        + ", ".join(f"{e['theta_deg']:.0f}:{e['corr_with_gt']:.3f}" for e in entries)
    )


if __name__ == "__main__":
    main()
