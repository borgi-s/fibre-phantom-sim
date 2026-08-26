"""Faithful reproduction of the PlenoXFiber plenoptic demo (known-good baseline).

Every geometry and solver value here is taken verbatim from
https://github.com/lsbesley/PlenoXFiber/blob/main/plenoptic-ASTRA_DemoScript-GH.ipynb
(see [[reference_plenoxfiber_demo]] in memory). The point is to reproduce a CLEAN,
working plenoptic reconstruction of a unidirectional fibre composite in this repo's own
geometry/reconstruct modules, BEFORE layering on the study extensions (CT fibre
orientation, plenoptic multi z-position and warm start). If this shows recovered fibres,
the stack is sound and any remaining trouble in the study is in the extensions, not the
core.

Differences from the study drivers, on purpose (this is the reference recipe):
  - 10 x 10 = 100 views (n_scans_xz=10), 256 x 256 detector (NOT 21 x 21 / 300 x 300).
  - CLEAN line integrals: proj = A(vol), no Beer-Lambert exp / Poisson (the demo has no
    noise; noise is a study extension, meaningful only once absorbances are physical).
  - Deep 650 x 650 x 800 volume: the 800-voxel beam depth accumulates strong along-fibre
    contrast (matrix ~0.28, fibre ~0.50 line integral), which is most of why the demo
    reconstructs cleanly where a shallow volume does not.
  - Solver: lr 1e-1, squared-L2 TV along the fibre axis only (tvz=0.5, tvy=tvx=0),
    non-negativity box clamp [0, 10], 100 iterations.

Straight (unidirectional) fibres are packed as one 2D cross-section and extruded along the
beam axis (array dim 0), so the phantom generation stays cheap even at 800 slices.

This-box vs cluster: `make_straight_fibre_phantom` is torch-CPU (fibre-pack) and unit
tested here; the ASTRA cuda3d forward projection and the recon run on the cluster. Every
astra/CUDA import is deferred into `main`.

Usage (cluster):
    python -m phantom_sim.repro_plenoxfiber --out phantom_sim_results/repro
    python -m phantom_sim.repro_plenoxfiber --smoke --out /tmp/repro_smoke

Outputs under --out:
    repro_gt.npy / repro_recon.npy   ground-truth phantom and reconstruction (float32)
    repro_meta.json                  all variables, plus recovered-contrast diagnostics
"""
import argparse
import json
from pathlib import Path

import numpy as np

from phantom_sim.phantom import make_straight_fibre_phantom


def write_meta(path, meta):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _registered_corr(recon, gt):
    """Pearson correlation of recon with gt (scale/offset invariant since corr ignores them)."""
    a = np.asarray(recon, dtype=np.float64).ravel()
    b = np.asarray(gt, dtype=np.float64).ravel()
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def build_parser():
    p = argparse.ArgumentParser(
        description="Faithful PlenoXFiber plenoptic reproduction (clean baseline).")
    p.add_argument("--out", required=True, help="Output directory (created if missing).")
    p.add_argument("--smoke", action="store_true",
                   help="Small volume, few views, few iters, for a fast cluster sanity run.")

    # reference geometry variables (notebook cell 15), all in recon-pixel (voxel) units
    p.add_argument("--recon-pixel-size", type=float, default=2.0, help="Voxel size, um.")
    p.add_argument("--source-range-um", type=float, default=22.0e3,
                   help="Plenoptic source grid half-range, um (reference 22 mm).")
    p.add_argument("--det-range-um", type=float, default=167.2e3,
                   help="Plenoptic detector grid half-range, um (reference 167.2 mm; this "
                        "satisfies det_range/source_range = SDD/SOD - 1).")
    p.add_argument("--n-scans-xz", type=int, default=10,
                   help="Source/detector grid side length (n x n views); reference 10.")
    p.add_argument("--sod-um", type=float, default=13.240e4,
                   help="Sample-to-source distance, um (reference 132.4 mm).")
    p.add_argument("--sdd-um", type=float, default=113.75e4,
                   help="Source-to-detector distance, um (reference 1137.5 mm).")
    p.add_argument("--detector-pixel-um", type=float, default=55.0, help="Detector pixel, um.")
    p.add_argument("--detector-size", type=int, default=256, help="Detector side, pixels.")
    p.add_argument("--super-sampling", type=int, default=2, help="ASTRA supersampling.")

    # volume (reference CT_vol_size = [650, 650, 800] as [nx, ny, nz]; array is (nz, ny, nx))
    p.add_argument("--nx", type=int, default=650, help="Transverse width, voxels.")
    p.add_argument("--ny", type=int, default=650, help="Transverse height, voxels.")
    p.add_argument("--nz", type=int, default=800, help="Beam-axis depth, voxels (fibre length).")

    # phantom (reference-scale per-voxel absorbance; ~3px fibres)
    p.add_argument("--domain-radius-um", type=float, default=400.0,
                   help="Fibre-bundle cross-section radius, um.")
    p.add_argument("--fvf", type=float, default=0.3, help="Fibre volume fraction.")
    p.add_argument("--r-mean-um", type=float, default=6.0,
                   help="Mean fibre radius, um (glass, ~12um diameter = 6um radius). At "
                        "2um voxel this is a 6-voxel disc, anti-aliased to a clean circle.")
    p.add_argument("--mu-fibre", type=float, default=6.2e-4, help="Fibre per-voxel absorbance.")
    p.add_argument("--mu-matrix", type=float, default=3.5e-4, help="Matrix per-voxel absorbance.")
    p.add_argument("--pack-iters", type=int, default=200, help="2D packing iterations.")
    p.add_argument("--seed", type=int, default=0, help="Packing seed.")

    # solver (reference cell 34/36)
    p.add_argument("--n-iter", type=int, default=100, help="Iterations (reference 100).")
    p.add_argument("--lr", type=float, default=1e-1, help="Adam learning rate (reference 0.1).")
    p.add_argument("--tv-weight", type=float, default=0.5,
                   help="Squared-L2 TV along the fibre axis (dim 0) only (reference tvz 0.5).")
    p.add_argument("--clamp-max", type=float, default=10.0, help="Upper box clamp (reference 10).")
    p.add_argument("--log-every", type=int, default=10, help="Logging interval, iterations.")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.smoke:
        nx = ny = 96
        nz = 64
        n_scans_xz = 4
        detector_size = 96
        domain_radius_um = 40.0
        pack_iters = 20
        n_iter = 20
    else:
        nx, ny, nz = args.nx, args.ny, args.nz
        n_scans_xz = args.n_scans_xz
        detector_size = args.detector_size
        domain_radius_um = args.domain_radius_um
        pack_iters = args.pack_iters
        n_iter = args.n_iter

    voxel_um = args.recon_pixel_size
    source_range_pix = args.source_range_um / voxel_um
    det_range_pix = args.det_range_um / voxel_um
    vol_z0_pix = args.sod_um / voxel_um
    detector_z_pix = args.sdd_um / voxel_um
    pixel_size = args.detector_pixel_um / voxel_um
    mag_factor = (vol_z0_pix + detector_z_pix) / vol_z0_pix  # reference print
    det_range_mult = det_range_pix / source_range_pix        # ~ SDD/SOD - 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating straight-fibre phantom (%d, %d, %d) ..." % (nz, ny, nx))
    vol, radii = make_straight_fibre_phantom(
        nz=nz, ny=ny, nx=nx, voxel_um=voxel_um, domain_radius_um=domain_radius_um,
        fvf=args.fvf, r_mean_um=args.r_mean_um, mu_fibre=args.mu_fibre,
        mu_matrix=args.mu_matrix, iters=pack_iters, seed=args.seed)
    print("  phantom ready: %d fibres, mu in [%.2e, %.2e]" % (radii.shape[0], vol.min(), vol.max()))
    print("  det_range multiplier %.3f (SDD/SOD-1 = %.3f), magnification %.3f"
          % (det_range_mult, detector_z_pix / vol_z0_pix - 1.0, mag_factor))

    # Deferred: astra + CUDA torch.
    import torch
    from phantom_sim.geometry import (
        build_plenoptic_geometry, make_plenoptic_projector, plenoptic_grid,
        XrayOperator, make_projection_fn,
    )
    from phantom_sim.reconstruct import reconstruct

    src, det = plenoptic_grid(source_range_pix, det_range_pix, n=n_scans_xz)
    geom = build_plenoptic_geometry(src, det, vol_z0_pix, detector_z_pix, pixel_size, pixel_size)
    num_views = n_scans_xz * n_scans_xz
    sinogram_size = [detector_size, detector_size, num_views]

    pid, _ = make_plenoptic_projector(vol.shape, geom, sinogram_size=sinogram_size,
                                      super_sampling=args.super_sampling)
    A = XrayOperator(pid)
    Projection = make_projection_fn()

    volt = torch.tensor(vol, device="cuda")
    # CLEAN line integrals, exactly like the reference (no exp / Poisson / -log).
    projections = Projection.apply(A, volt).detach()
    print("  projections: shape %s, sum %.3e" % (tuple(projections.shape), float(projections.sum())))

    recon, history = reconstruct(
        A, projections, mask=None, n_iter=n_iter, lr=args.lr,
        tv_weights=(args.tv_weight, 0.0, 0.0), x_init=None, log_every=args.log_every,
        gt=None, target_psnr=None, lower_clamp=0.0, upper_clamp=args.clamp_max,
    )

    corr = _registered_corr(recon, vol)
    print("  final data_loss %.3e, recon corr-with-GT %.4f, recon mu in [%.2e, %.2e]"
          % (history["data_loss"][-1], corr, recon.min(), recon.max()))

    gt_path = out_dir / "repro_gt.npy"
    recon_path = out_dir / "repro_recon.npy"
    np.save(gt_path, vol)
    np.save(recon_path, recon.astype(np.float32))
    meta = {
        "reference": "lsbesley/PlenoXFiber plenoptic-ASTRA_DemoScript-GH.ipynb",
        "smoke": bool(args.smoke),
        "shape": [nz, ny, nx],
        "voxel_um": voxel_um,
        "n_views": num_views,
        "detector_size": detector_size,
        "geometry": {
            "source_range_pix": source_range_pix,
            "det_range_pix": det_range_pix,
            "det_range_mult": det_range_mult,
            "vol_z0_pix": vol_z0_pix,
            "detector_z_pix": detector_z_pix,
            "pixel_size_pix": pixel_size,
            "magnification": mag_factor,
            "super_sampling": args.super_sampling,
        },
        "phantom": {
            "domain_radius_um": domain_radius_um,
            "fvf": args.fvf,
            "r_mean_um": args.r_mean_um,
            "mu_fibre": args.mu_fibre,
            "mu_matrix": args.mu_matrix,
            "n_fibres": int(radii.shape[0]),
            "pack_iters": pack_iters,
            "seed": args.seed,
        },
        "solver": {
            "forward": "clean_line_integrals_no_noise",
            "n_iter": n_iter,
            "lr": args.lr,
            "tv_weight_z": args.tv_weight,
            "clamp": [0.0, args.clamp_max],
        },
        "diagnostics": {
            "recon_corr_with_gt": corr,
            "recon_min": float(recon.min()),
            "recon_max": float(recon.max()),
            "final_data_loss": history["data_loss"][-1],
        },
        "history": history,
    }
    write_meta(out_dir / "repro_meta.json", meta)
    print("Saved %s, %s, and repro_meta.json" % (gt_path.name, recon_path.name))
    print("RECOVERED-FIBRE CHECK: corr-with-GT = %.4f  (>~0.5 means fibres are recovered)" % corr)


if __name__ == "__main__":
    main()
